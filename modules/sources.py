"""
Where the per-scene tasselled cap (TCW/TCB/TCG) values come from.

DEASource (default)
    Searches Digital Earth Australia's public STAC catalogue for Landsat
    5/7/8/9 Collection 3 ARD scenes (ga_ls*_ard_3) over the waterbody and
    reads only the few pixels it needs from each cloud-optimised GeoTIFF -
    no full scenes are downloaded. Two passes keep this quick:
      1. read the oa_fmask band for every scene and work out how much of the
         footprint is clear;
      2. read the six NBART bands only for scenes that pass the
         "minimum clear" setting.
    Pixels are kept where fmask is valid, snow or water (1, 4, 5) - the same
    categories DEA's load_ard() keeps, which is how the original
    "Melb region" rasters were cloud-masked.
    Scenes from the same solar day (adjacent tiles of one overpass) are
    merged into one observation.

LocalTCISource
    Reads the existing "TCI Melb region <timestamp>.tif" rasters
    (D:/DEA landsat rasters - TCW, TCB, TCG). Useful offline, or to compare
    against results produced by the original R workflow.

Tasselled cap coefficients are the DEA ones used in
"Script for creating TCI rasters.R", applied to reflectance x 10000.
"""
from __future__ import annotations

import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Callable, Optional

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import from_bounds

from .footprint import GRID_CRS, Footprint

NBART_BANDS = ["nbart_blue", "nbart_green", "nbart_red", "nbart_nir", "nbart_swir_1", "nbart_swir_2"]
FMASK_BAND = "oa_fmask"

TC_COEFFS = {
    "wetness": [0.0315, 0.2021, 0.3102, 0.1594, -0.6806, -0.6109],
    "brightness": [0.2043, 0.4158, 0.5524, 0.5741, 0.3124, -0.2303],
    "greenness": [-0.1603, -0.2819, -0.4934, 0.7940, -0.0002, -0.1446],
}

LS7_SLC_FAIL = datetime(2003, 5, 31, tzinfo=timezone.utc)

GDAL_ENV = dict(
    GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
    CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.tiff",
    GDAL_HTTP_MULTIRANGE="YES",
    GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES",
    GDAL_HTTP_MAX_RETRY="5",
    GDAL_HTTP_RETRY_DELAY="2",
    GDAL_HTTP_TIMEOUT="60",
    GDAL_HTTP_CONNECTTIMEOUT="30",
    AWS_NO_SIGN_REQUEST="YES",
    AWS_REGION="ap-southeast-2",
    VSI_CACHE="TRUE",
)

ProgressCB = Optional[Callable[[int, int, str], None]]


@dataclass
class Scene:
    key: str  # unique id for caching: solar date + platform
    datetime_utc: datetime
    solar_date: date
    platform: str
    parts: list = field(default_factory=list)  # pystac Items, or file paths
    source: str = "DEA"

    @property
    def utc_date(self) -> str:
        return self.datetime_utc.strftime("%Y-%m-%d")

    @property
    def utc_time(self) -> str:
        return self.datetime_utc.strftime("%H:%M:%S")


def tasselled_cap(bands: np.ndarray) -> dict[str, np.ndarray]:
    """bands: (6, h, w) reflectance x 10000, NaN where invalid. Returns
    float32 arrays (the original TCI rasters were float32)."""
    out = {}
    for name, c in TC_COEFFS.items():
        out[name] = np.tensordot(np.asarray(c), bands, axes=1).astype(np.float32)
    return out


def _solar_date(dt_utc: datetime, lon: float) -> date:
    return (dt_utc + timedelta(hours=lon / 15.0)).date()


# --------------------------------------------------------------------------
# DEA (online)
# --------------------------------------------------------------------------

def _gdal_path(href: str) -> list[str]:
    """Candidate GDAL paths for a DEA asset href (https first, then S3)."""
    if href.startswith("s3://"):
        bucket_key = href[5:]
        bucket, _, key = bucket_key.partition("/")
        cands = []
        if bucket == "dea-public-data":
            cands.append("https://data.dea.ga.gov.au/" + key)
        cands.append("/vsis3/" + bucket_key)
        return cands
    return [href]


def _read_window(path: str, fp: Footprint, resampling=Resampling.nearest, dtype="float64") -> np.ndarray:
    """Reads `path` onto the footprint grid. Returns float array, NaN = nodata."""
    with rasterio.open(path) as src:
        nodata = src.nodata
        if src.crs and src.crs.to_epsg() == 3577 and np.isclose(src.transform.a, 30) and \
                np.isclose((fp.transform.c - src.transform.c) % 30, 0) and np.isclose((fp.transform.f - src.transform.f) % 30, 0):
            win = from_bounds(*fp.bounds, transform=src.transform).round_offsets().round_lengths()
            arr = src.read(1, window=win, boundless=True, fill_value=nodata if nodata is not None else 0,
                           out_shape=(fp.height, fp.width))
        else:
            with WarpedVRT(src, crs=GRID_CRS, transform=fp.transform, width=fp.width, height=fp.height,
                           resampling=resampling, src_nodata=nodata, nodata=nodata) as vrt:
                arr = vrt.read(1)
    arr = arr.astype(dtype)
    if nodata is not None:
        arr[arr == nodata] = np.nan
    return arr


class DEASource:
    name = "DEA"

    def __init__(self, stac_urls: list[str], collections: list[str], fmask_good=(1, 4, 5), max_workers: int = 16):
        self.stac_urls = stac_urls
        self.collections = collections
        self.fmask_good = list(fmask_good)
        self.max_workers = max_workers

    # ---- search --------------------------------------------------------
    def search(self, fp: Footprint, poly_wgs84, start: date, end: date,
               include_ls7_slc_off: bool = True, max_scene_cloud: float = 100.0) -> list[Scene]:
        import pystac_client
        import shapely
        from shapely.geometry import shape

        bbox = list(poly_wgs84.buffer(0.001).bounds)
        last_err = None
        items = None
        for url in self.stac_urls:
            for attempt in range(3):
                try:
                    client = pystac_client.Client.open(url)
                    search = client.search(collections=self.collections, bbox=bbox,
                                           datetime=f"{start.isoformat()}/{end.isoformat()}", limit=500)
                    items = list(search.items())
                    break
                except Exception as e:  # noqa: BLE001
                    last_err = e
                    time.sleep(2 * (attempt + 1))
            if items is not None:
                break
        if items is None:
            raise RuntimeError(f"Could not search the DEA catalogue: {last_err}")

        lon = poly_wgs84.centroid.x
        groups: dict[tuple, list] = {}
        for it in items:
            try:
                if it.geometry is not None and not shape(it.geometry).intersects(poly_wgs84):
                    continue
            except Exception:  # noqa: BLE001
                pass
            dt = it.datetime or datetime.fromisoformat(str(it.properties.get("datetime")).replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            platform = str(it.properties.get("platform") or it.properties.get("eo:platform") or it.collection_id)
            if not include_ls7_slc_off and ("7" in platform or "ls7" in str(it.collection_id)) and dt >= LS7_SLC_FAIL:
                continue
            cc = it.properties.get("eo:cloud_cover")
            if cc is not None and cc > max_scene_cloud:
                continue
            # skip non-final (near-real-time) versions when a final exists - keep all, dedupe later
            groups.setdefault((_solar_date(dt, lon), platform), []).append((dt, it))

        scenes = []
        for (sd, platform), members in groups.items():
            members.sort(key=lambda m: (0 if "final" in str(m[1].properties.get("dea:dataset_maturity", "final")) else 1, m[0]))
            dt0 = min(m[0] for m in members)
            scenes.append(Scene(key=f"{sd.isoformat()}_{platform}", datetime_utc=dt0, solar_date=sd,
                                platform=platform, parts=[m[1] for m in members], source="DEA"))
        scenes.sort(key=lambda s: s.datetime_utc)
        return scenes

    # ---- reading -------------------------------------------------------
    def _read_band(self, item, band: str, fp: Footprint) -> np.ndarray:
        asset = item.assets.get(band)
        if asset is None:
            # some catalogues use the short alias (e.g. "fmask", "blue")
            alias = band.replace("nbart_", "").replace("oa_", "")
            asset = item.assets.get(alias)
        if asset is None:
            raise KeyError(f"Scene {item.id} has no '{band}' asset")
        last = None
        for path in _gdal_path(asset.href):
            for attempt in range(3):
                try:
                    with rasterio.Env(**GDAL_ENV):
                        return _read_window(path, fp)
                except Exception as e:  # noqa: BLE001
                    last = e
                    time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"Failed to read {band} for {item.id}: {last}")

    def _merged(self, scene: Scene, band: str, fp: Footprint) -> np.ndarray:
        out = None
        for item in scene.parts:
            arr = self._read_band(item, band, fp)
            if band == FMASK_BAND:
                arr[arr == 0] = np.nan
            if out is None:
                out = arr
            else:
                fill = np.isnan(out) & ~np.isnan(arr)
                out[fill] = arr[fill]
            if not np.isnan(out[fp.rows, fp.cols]).any():
                break
        return out

    def clear_mask(self, scene: Scene, fp: Footprint) -> np.ndarray:
        """Bool per footprint pixel: clear (fmask valid/snow/water)."""
        fm = self._merged(scene, FMASK_BAND, fp)
        vals = fm[fp.rows, fp.cols]
        return np.isin(vals, self.fmask_good)

    def read_tci(self, scene: Scene, fp: Footprint, clear: np.ndarray) -> dict[str, np.ndarray]:
        bands = np.stack([self._merged(scene, b, fp)[fp.rows, fp.cols] for b in NBART_BANDS])
        bands[:, ~clear] = np.nan
        # DEA NBART nodata is -999; any band missing -> pixel invalid
        bad = np.isnan(bands).any(axis=0)
        bands[:, bad] = np.nan
        return tasselled_cap(bands)


# --------------------------------------------------------------------------
# Local TCI rasters
# --------------------------------------------------------------------------

_TS = re.compile(r"(\d{4}-\d{2}-\d{2})T(\d{2})_(\d{2})_(\d{2})")


class LocalTCISource:
    """Folder of 3-band (TCW, TCB, TCG) rasters named '... <YYYY-MM-DD>T<HH_MM_SS...>.tif'."""

    name = "Local TCI rasters"

    def __init__(self, folder: str, max_workers: int = 4):
        self.folder = folder
        self.max_workers = max_workers

    def search(self, fp: Footprint, poly_wgs84, start: date, end: date, **_) -> list[Scene]:
        if not os.path.isdir(self.folder):
            raise RuntimeError(f"Local TCI folder not found: {self.folder}")
        lon = poly_wgs84.centroid.x
        scenes = []
        for f in sorted(os.listdir(self.folder)):
            if not f.lower().endswith(".tif"):
                continue
            m = _TS.search(f)
            if not m:
                continue
            dt = datetime.fromisoformat(f"{m.group(1)}T{m.group(2)}:{m.group(3)}:{m.group(4)}").replace(tzinfo=timezone.utc)
            if not (start <= dt.date() <= end):
                continue
            scenes.append(Scene(key=f[:-4], datetime_utc=dt, solar_date=_solar_date(dt, lon), platform="landsat",
                                parts=[os.path.join(self.folder, f)], source="local"))
        return scenes

    def _read_all(self, scene: Scene, fp: Footprint) -> np.ndarray:
        path = scene.parts[0]
        with rasterio.open(path) as src:
            win = from_bounds(*fp.bounds, transform=src.transform).round_offsets().round_lengths()
            arr = src.read([1, 2, 3], window=win, boundless=True, out_shape=(3, fp.height, fp.width), masked=True)
        arr = np.ma.filled(arr.astype(np.float64), np.nan)
        arr[np.abs(arr) > 1e30] = np.nan
        return arr[:, fp.rows, fp.cols]

    def clear_mask(self, scene: Scene, fp: Footprint) -> np.ndarray:
        return np.isfinite(self._read_all(scene, fp)).all(axis=0)

    def read_tci(self, scene: Scene, fp: Footprint, clear: np.ndarray) -> dict[str, np.ndarray]:
        vals = self._read_all(scene, fp).astype(np.float32)
        vals[:, ~clear] = np.nan
        return {"wetness": vals[0], "brightness": vals[1], "greenness": vals[2]}


def run_parallel(fn, scenes: list[Scene], max_workers: int, progress: ProgressCB = None, label: str = "",
                 should_stop: Optional[Callable[[], bool]] = None):
    """Runs fn(scene) over scenes in a thread pool. Returns {key: result or Exception}."""
    results = {}
    total = len(scenes)
    done = 0
    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as ex:
        futs = {ex.submit(fn, s): s for s in scenes}
        for fut in as_completed(futs):
            s = futs[fut]
            try:
                results[s.key] = fut.result()
            except Exception as e:  # noqa: BLE001
                results[s.key] = e
            done += 1
            if progress:
                progress(done, total, f"{label} {done}/{total}")
            if should_stop and should_stop():
                for f in futs:
                    f.cancel()
                break
    return results
