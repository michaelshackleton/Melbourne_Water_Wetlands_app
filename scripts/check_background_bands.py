"""
One-off check of the background raster's band order.

Samples ~200,000 pixels across the whole raster and prints each band's
summary next to the name the app gives it (config/settings.json ->
background_band_names), alongside the same statistics from the models'
training data. If e.g. the band called mean_TCB looks like greenness
(hundreds) rather than brightness (thousands), the names are in the wrong
order.

Usage (from the app folder, environment active):
    python scripts/check_background_bands.py ["path/to/background reprojection.tif"]
"""
import json
import os
import sys

import numpy as np
import rasterio

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cfg = json.load(open(os.path.join(APP, "config", "settings.json")))
user = os.path.join(APP, "config", "user_settings.json")
if os.path.exists(user):
    cfg.update(json.load(open(user)))
path = sys.argv[1] if len(sys.argv) > 1 else cfg["background_raster"]
names = cfg["background_band_names"]

# medians at the training sites (Final version training data, no coords, 04122025)
TRAINING_MEDIANS = {"mean_TCW": -1055, "mean_TCB": 2622, "mean_TCG": 763, "Q10W": None, "Q90W": None,
                    "Q10B": 1916, "Q90B": 3305, "Q10G": 212, "Q90G": 1526}

with rasterio.open(path) as src:
    print(f"{path}\n  CRS: {src.crs}\n  pixel size: {src.res}\n  size: {src.width} x {src.height}, {src.count} bands, "
          f"dtype {src.dtypes[0]}, nodata {src.nodata}\n")
    rng = np.random.default_rng(0)
    out_shape = (src.count, min(src.height, 600), min(src.width, 600))
    arr = src.read(out_shape=out_shape, masked=True).astype(float).filled(np.nan)
arr[np.abs(arr) > 1e30] = np.nan
print(f"{'band':>4}  {'app name':<10} {'median':>9} {'p10':>9} {'p90':>9}   training-site median")
for i in range(arr.shape[0]):
    b = arr[i][np.isfinite(arr[i])]
    name = names[i] if i < len(names) else "?"
    tm = TRAINING_MEDIANS.get(name)
    print(f"{i + 1:>4}  {name:<10} {np.median(b):>9.0f} {np.percentile(b, 10):>9.0f} {np.percentile(b, 90):>9.0f}   "
          f"{'' if tm is None else tm}")
print("\nBrightness (B) bands should be in the thousands, greenness (G) in the hundreds, wetness (W) negative.")
