"""
Builds data/background_pixels.parquet: the nine long-term background
predictors (mean_TCW, mean_TCB, mean_TCG, Q10/Q90 W/B/G) for every 30 m
Albers pixel that touches any waterbody in the app's waterbodies layer.

With this table in the app folder the app no longer needs the 1.6 GB
whole-landscape raster (or the D: drive) to make predictions. Run it once,
on a machine that can see the raster, then commit the table with the app.

Values are taken exactly as the app takes them from the raster: bilinear
interpolation at each 30 m Albers pixel centre (or the pixel itself if the
raster is already on that grid), with bands named by background_band_names
from config/settings.json. Memory: one band of the raster at a time (~350 MB
for the Melbourne-region raster).

Usage (from the app folder, environment active):
    python scripts/build_background_table.py ["D:/whole landscape rasters/background reprojection.tif"]
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import rasterio  # noqa: E402
from affine import Affine  # noqa: E402

from modules.background import REQUIRED, _same_crs_no_rotation, bilinear_sample, pixel_keys  # noqa: E402
from modules.waterbodies import WaterbodyLayer  # noqa: E402


def main():
    cfg = json.load(open(os.path.join(APP, "config", "settings.json")))
    user = os.path.join(APP, "config", "user_settings.json")
    if os.path.exists(user):
        cfg.update(json.load(open(user)))
    raster = sys.argv[1] if len(sys.argv) > 1 else cfg["background_raster"]
    names = cfg["background_band_names"]
    wb_path = cfg["waterbodies_path"]
    wb_path = wb_path if os.path.isabs(wb_path) else os.path.join(APP, wb_path)
    out_path = cfg.get("background_table", "data/background_pixels.parquet")
    out_path = out_path if os.path.isabs(out_path) else os.path.join(APP, out_path)

    t0 = time.time()
    print(f"Waterbodies: {wb_path}")
    layer = WaterbodyLayer(wb_path, cfg["waterbody_id_field"])
    # Every 30 m pixel touched by any polygon (all_touched=True gives a slight
    # superset of the pixels the app uses - harmless).
    from rasterio.features import rasterize
    b = layer.gdf_albers.total_bounds
    gx0, gy1 = math.floor(b[0] / 30) * 30 - 30, math.ceil(b[3] / 30) * 30 + 30
    gw = int(math.ceil((b[2] - gx0) / 30)) + 2
    gh = int(math.ceil((gy1 - b[1]) / 30)) + 2
    mask = rasterize(((g, 1) for g in layer.gdf_albers.geometry.values), out_shape=(gh, gw),
                     transform=Affine(30, 0, gx0, 0, -30, gy1), all_touched=True, dtype="uint8")
    rr, cc = np.nonzero(mask)
    keys = np.unique(pixel_keys(gx0 + (cc + 0.5) * 30, gy1 - (rr + 0.5) * 30))
    ix = keys >> 32
    iy = (keys & 0xFFFFFFFF).astype(np.int64)
    iy = np.where(iy >= 2**31, iy - 2**32, iy)  # undo the unsigned wrap for negative northings
    print(f"{len(keys):,} unique pixels to extract ({time.time() - t0:.0f}s)")
    print(f"Background raster: {raster}")
    xs = (ix * 30 + 15).astype(np.float64)  # pixel centres
    ys = (iy * 30 + 15).astype(np.float64)
    vals = np.full((len(names), len(keys)), np.nan, dtype=np.float64)
    with rasterio.open(raster) as src:
        if src.count < len(names):
            sys.exit(f"Raster has {src.count} bands but {len(names)} names are configured.")
        if not _same_crs_no_rotation(src):
            sys.exit("The background raster isn't in EPSG:3577 - reproject it to Australian Albers first.")
        aligned = math.isclose(src.transform.a, 30) and math.isclose(src.transform.e, -30) and \
            math.isclose(src.transform.c % 30, 0, abs_tol=1e-6) and math.isclose(src.transform.f % 30, 0, abs_tol=1e-6)
        for b in range(len(names)):
            band = src.read(b + 1, masked=True)
            band = np.ma.filled(band.astype(np.float64), np.nan)
            band[np.abs(band) > 1e30] = np.nan
            if aligned:   # already on the 30 m grid: take the pixel itself
                c = np.floor((xs - src.transform.c) / 30).astype(np.int64)
                r = np.floor((src.transform.f - ys) / 30).astype(np.int64)
                ok = (c >= 0) & (r >= 0) & (c < src.width) & (r < src.height)
                vals[b, ok] = band[r[ok], c[ok]]
            else:
                vals[b] = bilinear_sample(band, src.transform, xs, ys)
            print(f"  band {b + 1}/{len(names)} ({names[b]})", flush=True)
            del band
    vals[np.abs(vals) > 1e30] = np.nan

    df = pd.DataFrame({"key": keys})
    for i, name in enumerate(names):
        df[name] = vals[i].astype(np.float32)
    have = df[REQUIRED].notna().any(axis=1)
    print(f"{have.sum():,} of {len(df):,} pixels have background values "
          f"({(~have).sum():,} fall outside the raster)")
    df = df[have][["key"] + REQUIRED]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_parquet(out_path, index=False, compression="zstd", compression_level=19)
    meta = dict(source_raster=raster, band_names=names, n_pixels=int(len(df)),
                built=time.strftime("%Y-%m-%d %H:%M"), waterbodies=os.path.basename(wb_path))
    with open(out_path.replace(".parquet", "_info.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote {out_path} ({os.path.getsize(out_path) / 1e6:.1f} MB) in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
