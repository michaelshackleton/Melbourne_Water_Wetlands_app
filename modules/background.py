"""
Reads the long-term "background" predictors (mean, 10th and 90th percentile
tasselled cap values) for a waterbody's pixels from the whole-landscape
raster (D:/whole landscape rasters/background reprojection.tif by default).

Band names follow the order the original R script assigned them:
mean_TCW, mean_TCG, mean_TCB, Q10W, Q10G, Q10B, Q90W, Q90G, Q90B
(configurable in config/settings.json -> background_band_names).

If the raster is already on the 30 m Albers grid the values are read
directly; otherwise it is warped onto the waterbody's grid with bilinear
resampling (the same resampling the original script used).
"""
from __future__ import annotations

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


def read_background(path: str, band_names: list[str], fp: Footprint) -> pd.DataFrame:
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
