"""
Reads the long-term "background" predictors (mean, 10th and 90th percentile
tasselled cap values) for a waterbody's pixels.

Two sources, tried in this order:
  1. the background pixel table shipped with the app
     (data/background_pixels.parquet) - the values of every 30 m pixel that
     touches any waterbody in the layer, pre-extracted from the
     whole-landscape raster by scripts/build_background_table.py. This is
     what lets the app run without the D: drive.
  2. the whole-landscape raster itself
     (D:/whole landscape rasters/background reprojection.tif), used for any
     pixel the table doesn't have.

Band names follow the order the original R script assigned them:
mean_TCW, mean_TCG, mean_TCB, Q10W, Q10G, Q10B, Q90W, Q90G, Q90B
(configurable in config/settings.json -> background_band_names).

If the raster is already on the 30 m Albers grid the values are read
directly; otherwise it is warped onto the waterbody's grid with bilinear
resampling (the same resampling the original script used).
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import from_bounds

from .footprint import GRID_CRS, Footprint

REQUIRED = ["mean_TCW", "mean_TCB", "mean_TCG", "Q10W", "Q90W", "Q10B", "Q90B", "Q10G", "Q90G"]


class BackgroundError(Exception):
    pass


def _aligned(src, fp: Footprint) -> bool:
    if src.crs is None or src.crs.to_epsg() != 3577:
        return False
    t = src.transform
    if not (np.isclose(t.a, 30) and np.isclose(t.e, -30) and t.b == 0 and t.d == 0):
        return False
    return np.isclose((fp.transform.c - t.c) % 30, 0) and np.isclose((fp.transform.f - t.f) % 30, 0)


def pixel_keys(x_centre: np.ndarray, y_centre: np.ndarray) -> np.ndarray:
    """int64 key of a 30 m Albers pixel from its centre coordinates."""
    ix = np.round((np.asarray(x_centre) - 15.0) / 30.0).astype(np.int64)
    iy = np.round((np.asarray(y_centre) - 15.0) / 30.0).astype(np.int64)
    return (ix << 32) | (iy & 0xFFFFFFFF)


_TABLE_CACHE: dict = {}


def _load_table(path: str):
    mtime = os.path.getmtime(path)
    hit = _TABLE_CACHE.get(path)
    if hit and hit[0] == mtime:
        return hit[1]
    df = pd.read_parquet(path)
    df = df.sort_values("key").reset_index(drop=True)
    tbl = (df["key"].to_numpy(np.int64), {c: df[c].to_numpy(np.float64) for c in REQUIRED})
    _TABLE_CACHE.clear()
    _TABLE_CACHE[path] = (mtime, tbl)
    return tbl


def read_background(raster_path: str, band_names: list[str], fp: Footprint,
                    table_path: str | None = None) -> pd.DataFrame:
    """Background predictors for the footprint's pixels (table first, raster
    for anything the table lacks)."""
    out = None
    if table_path and os.path.exists(table_path):
        keys, cols = _load_table(table_path)
        want = pixel_keys(fp.pixels["x_albers"].to_numpy(), fp.pixels["y_albers"].to_numpy())
        pos = np.searchsorted(keys, want)
        pos_c = np.clip(pos, 0, len(keys) - 1)
        hit = keys[pos_c] == want
        out = pd.DataFrame({c: np.where(hit, v[pos_c], np.nan) for c, v in cols.items()}, index=fp.pixels.index)
        if hit.all():
            return out
    if raster_path and os.path.exists(raster_path):
        ras = read_background_raster(raster_path, band_names, fp)
        if out is None:
            return ras
        return out.fillna(ras[REQUIRED])
    if out is not None:
        return out  # some pixels missing - they become no_data
    raise BackgroundError(
        "No background data available: the background pixel table "
        f"('{table_path}') was not found and the background raster ('{raster_path}') isn't reachable. "
        "Connect the drive with the raster, or build the table with scripts/build_background_table.py.")


def bilinear_sample(arr: np.ndarray, transform, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Bilinear interpolation of a 2-D array (NaN = no data) at points xs, ys
    given the array's affine transform (no rotation). NaN if any of the four
    surrounding cells is missing or the point is off the array."""
    fc = (np.asarray(xs) - transform.c) / transform.a - 0.5
    fr = (np.asarray(ys) - transform.f) / transform.e - 0.5
    c0 = np.floor(fc).astype(np.int64)
    r0 = np.floor(fr).astype(np.int64)
    dc, dr = fc - c0, fr - r0
    h, w = arr.shape
    ok = (c0 >= 0) & (r0 >= 0) & (c0 + 1 < w) & (r0 + 1 < h)
    out = np.full(len(fc), np.nan)
    if ok.any():
        c, r, u, v = c0[ok], r0[ok], dc[ok], dr[ok]
        out[ok] = (arr[r, c] * (1 - u) * (1 - v) + arr[r, c + 1] * u * (1 - v)
                   + arr[r + 1, c] * (1 - u) * v + arr[r + 1, c + 1] * u * v)
    return out


def _same_crs_no_rotation(src) -> bool:
    t = src.transform
    return src.crs is not None and src.crs.to_epsg() == 3577 and t.b == 0 and t.d == 0


def read_background_raster(path: str, band_names: list[str], fp: Footprint) -> pd.DataFrame:
    missing = [b for b in REQUIRED if b not in band_names]
    if missing:
        raise BackgroundError(f"background_band_names is missing {missing}")
    try:
        src = rasterio.open(path)
    except Exception as e:  # noqa: BLE001
        raise BackgroundError(f"Could not open background raster '{path}': {e}") from e
    with src:
        if src.count < len(band_names):
            raise BackgroundError(f"Background raster has {src.count} bands but {len(band_names)} names were configured")
        if _aligned(src, fp):
            win = from_bounds(*fp.bounds, transform=src.transform).round_offsets().round_lengths()
            arr = src.read(list(range(1, len(band_names) + 1)), window=win, boundless=True, masked=True,
                           out_shape=(len(band_names), fp.height, fp.width))
        elif _same_crs_no_rotation(src):
            # Same projection, different pixel size/origin (e.g. the 29.86 m
            # "background reprojection.tif"): bilinear interpolation at the
            # 30 m pixel centres, done here so the result is exact and matches
            # scripts/build_background_table.py.
            x0, y0, x1, y1 = fp.bounds
            win = from_bounds(x0 - 60, y0 - 60, x1 + 60, y1 + 60, transform=src.transform).round_offsets().round_lengths()
            raw = src.read(list(range(1, len(band_names) + 1)), window=win, boundless=True, masked=True)
            raw = np.ma.filled(raw.astype(np.float64), np.nan)
            raw[np.abs(raw) > 1e30] = np.nan
            wt = src.window_transform(win)
            xs, ys = fp.pixels["x_albers"].to_numpy(), fp.pixels["y_albers"].to_numpy()
            return pd.DataFrame({name: bilinear_sample(raw[i], wt, xs, ys) for i, name in enumerate(band_names)},
                                index=fp.pixels.index)
        else:
            with WarpedVRT(src, crs=GRID_CRS, transform=fp.transform, width=fp.width, height=fp.height,
                           resampling=Resampling.bilinear) as vrt:
                arr = vrt.read(list(range(1, len(band_names) + 1)), masked=True)
    data = np.ma.filled(arr.astype(np.float64), np.nan)
    data[np.abs(data) > 1e30] = np.nan  # R's float NA (-3.4e38) if not declared as nodata
    out = {name: data[i][fp.rows, fp.cols] for i, name in enumerate(band_names)}
    return pd.DataFrame(out, index=fp.pixels.index)


def background_sanity(bg: pd.DataFrame) -> list[str]:
    """Plain-language warnings if the band naming looks wrong."""
    warn = []
    if bg[REQUIRED].isna().all(axis=1).all():
        warn.append("The background raster has no data over this waterbody (it may lie outside the raster's extent), "
                    "so no predictions can be made.")
        return warn
    for idx in ["W", "B", "G"]:
        lo, hi = bg[f"Q10{idx}"], bg[f"Q90{idx}"]
        ok = lo.notna() & hi.notna()
        if ok.any() and (lo[ok] > hi[ok]).mean() > 0.5:
            warn.append(f"Q10{idx} is above Q90{idx} for most pixels - check background_band_names in config/settings.json.")
    ok = bg["mean_TCB"].notna() & bg["mean_TCG"].notna()
    if ok.any() and (bg.loc[ok, "mean_TCB"] < bg.loc[ok, "mean_TCG"]).mean() > 0.5:
        warn.append("The band named mean_TCB is lower than mean_TCG for most pixels here. In the models' training data "
                    "mean_TCB was higher than mean_TCG at every site, so the brightness/greenness band names may be "
                    "swapped - check background_band_names (run scripts/check_background_bands.py).")
    if bg["mean_TCW"].notna().any() and (bg["mean_TCW"] > 0).mean() > 0.9:
        warn.append("mean_TCW is positive almost everywhere, which is unusual for tasselled cap wetness - "
                    "check background_band_names in config/settings.json.")
    return warn
