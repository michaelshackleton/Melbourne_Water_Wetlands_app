"""
Builds the background (long-term) predictor table for the Waterbody Cover
Predictor app from the three whole-landscape composite rasters in this folder:

    mean TCI composite image of Melbourne region.tif        -> mean_TCW, mean_TCB, mean_TCG
    10th percentile composite image of Melbourne region.tif -> Q10W, Q10B, Q10G
    90th percentile composite image of Melbourne region.tif -> Q90W, Q90B, Q90G

For every 30 m Landsat pixel that touches a waterbody in the app's waterbodies
layer it writes:

  * <app>/data/background_pixels.parquet  - the table the app reads
  * background_pixels_for_waterbodies.csv - the same values as a CSV, with
    pixel-centre coordinates (Albers EPSG:3577 and lng/lat), next to this script

WHICH BANDS ARE WETNESS / BRIGHTNESS / GREENNESS
The composites have 9 bands. The script assumes bands 1-6 are the six
reflectance bands (blue, green, red, NIR, SWIR1, SWIR2) and finds the three
tasselled cap bands among the others by computing TCW/TCB/TCG from bands 1-6
with the DEA coefficients (the ones in "Script for creating TCI rasters.R")
and matching:
  * mean composite: the mean of a tasselled cap index equals the index of the
    mean reflectances, so the right band matches almost exactly;
  * 10th / 90th percentile composites: matched by correlation (a percentile of
    the index isn't exactly the index of the percentile reflectances, but they
    are very highly correlated).
Everything it finds is printed. If it gets it wrong, re-run with
--bands W,B,G band numbers, e.g. --bands 7,8,9 - re-runs are quick because
the pixel values are cached after the first run.

Source rasters (built Oct 2026): G:\\SABE\\CFE\\Projects\\Melbourne Uni\\Completed or closed projects\\
17_01455 Spatial prioritisation of waterway management for biodiversity outcomes\\Data\\wetland inundation
model\\GIS\\whole landscape rasters. Each has 9 bands: blue, green, red, NIR, SWIR1, SWIR2, TCW, TCG, TCB.
The resulting table matched the original R extraction ("background data for wetlands.csv") exactly.

RUNNING (PowerShell, in the rasters folder - or anywhere with --rasters "<folder>"):

    & "C:\\Users\\Meshackleton\\Documents\\rs-utils-main\\waterbodies app\\.venv\\Scripts\\python.exe" make_background_table.py

Options:
    --app PATH       the app folder (default C:\\Users\\Meshackleton\\Documents\\rs-utils-main\\waterbodies app)
    --bands W,B,G    band numbers of wetness, brightness, greenness (same in all three rasters)
    --no-csv         only write the parquet table for the app
    --refresh        ignore the cache and re-read the rasters

Time: each raster is read once, in strips (~15 min each from the network
drive); the extracted values are cached in background_extract_cache.npz here.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from rasterio.features import rasterize
from rasterio.windows import Window

HERE = os.path.dirname(os.path.abspath(__file__))  # folder holding the three composite rasters (see --rasters)
DEFAULT_APP = r"C:\Users\Meshackleton\Documents\rs-utils-main\waterbodies app"
CACHE = os.path.join(HERE, "background_extract_cache.npz")
RASTERS = {
    "mean": "mean TCI composite image of Melbourne region.tif",
    "Q10": "10th percentile composite image of Melbourne region.tif",
    "Q90": "90th percentile composite image of Melbourne region.tif",
}
TC = {  # DEA coefficients, applied to blue, green, red, NIR, SWIR1, SWIR2
    "W": [0.0315, 0.2021, 0.3102, 0.1594, -0.6806, -0.6109],
    "B": [0.2043, 0.4158, 0.5524, 0.5741, 0.3124, -0.2303],
    "G": [-0.1603, -0.2819, -0.4934, 0.7940, -0.0002, -0.1446],
}


def colname(stat: str, idx: str) -> str:
    return f"mean_TC{idx}" if stat == "mean" else f"{stat}{idx}"


def pixel_list(app: str):
    sys.path.insert(0, app)
    from modules.background import REQUIRED, pixel_keys
    from modules.waterbodies import read_layer

    cfg = json.load(open(os.path.join(app, "config", "settings.json")))
    wb_path = cfg["waterbodies_path"]
    wb_path = wb_path if os.path.isabs(wb_path) else os.path.join(app, wb_path)
    print(f"Waterbodies layer: {wb_path}")
    gdf = read_layer(wb_path).to_crs(3577)
    b = gdf.total_bounds
    gx0, gy1 = math.floor(b[0] / 30) * 30 - 30, math.ceil(b[3] / 30) * 30 + 30
    gw, gh = int(math.ceil((b[2] - gx0) / 30)) + 2, int(math.ceil((gy1 - b[1]) / 30)) + 2
    mask = rasterize(((g, 1) for g in gdf.geometry.values if g is not None), out_shape=(gh, gw),
                     transform=Affine(30, 0, gx0, 0, -30, gy1), all_touched=True, dtype="uint8")
    rr, cc = np.nonzero(mask)
    xs, ys = gx0 + (cc + 0.5) * 30.0, gy1 - (rr + 0.5) * 30.0
    print(f"  {len(gdf):,} waterbodies, {len(xs):,} pixels touch a waterbody")
    return xs, ys, pixel_keys(xs, ys), REQUIRED, cfg, wb_path


def extract(path: str, xs: np.ndarray, ys: np.ndarray, t0: float):
    """All bands at the pixel centres, reading the raster once in strips.
    Returns (values[bands, n], medians[bands], info)."""
    with rasterio.open(path) as src:
        t = src.transform
        info = f"CRS {src.crs}, pixel {src.res[0]:.3f} m, {src.width} x {src.height}, {src.count} bands, {src.dtypes[0]}"
        print(f"  {info}")
        aligned = (math.isclose(t.a, 30) and math.isclose(t.e, -30)
                   and math.isclose(t.c % 30, 0, abs_tol=1e-6) and math.isclose(t.f % 30, 0, abs_tol=1e-6))
        if not aligned:
            sys.exit("  This raster isn't on the 30 m Landsat grid - tell Claude, the script needs adjusting.")
        c = np.floor((xs - t.c) / 30).astype(np.int64)
        r = np.floor((t.f - ys) / 30).astype(np.int64)
        ok = (c >= 0) & (r >= 0) & (c < src.width) & (r < src.height)
        vals = np.full((src.count, len(xs)), np.nan, dtype=np.float32)
        idx = np.where(ok)[0]
        idx = idx[np.argsort(r[idx])]
        r_sorted = r[idx]
        sample = []
        strip = 256
        for r0 in range(0, src.height, strip):
            h = min(strip, src.height - r0)
            a = src.read(window=Window(0, r0, src.width, h), masked=True).astype(np.float32).filled(np.nan)
            a[np.abs(a) > 1e30] = np.nan
            if r0 % (strip * 8) == 0:
                sample.append(a[:, ::16, ::16].reshape(src.count, -1))
            lo, hi = np.searchsorted(r_sorted, [r0, r0 + h])
            if hi > lo:
                sel = idx[lo:hi]
                vals[:, sel] = a[:, r[sel] - r0, c[sel]]
            if (r0 // strip) % 6 == 0:
                print(f"    rows {r0 + h:,}/{src.height:,} ({time.time() - t0:.0f}s)", flush=True)
        med = np.nanmedian(np.concatenate(sample, axis=1), axis=1)
    return vals, med, info


def tc_from_reflectance(v: np.ndarray) -> dict[str, np.ndarray]:
    return {k: np.tensordot(np.asarray(c), v[:6].astype(np.float64), axes=1) for k, c in TC.items()}


def find_tc_bands(stat: str, v: np.ndarray, med: np.ndarray) -> dict[str, int]:
    """Band index (0-based) holding W, B, G, found by comparing with the
    indices computed from bands 1-6."""
    derived = tc_from_reflectance(v)
    cand = list(range(6, v.shape[0])) if v.shape[0] >= 9 else list(range(v.shape[0]))
    ok = np.isfinite(v).all(axis=0)
    print(f"  matching against tasselled cap computed from bands 1-6 ({ok.sum():,} pixels):")
    scores = {}
    for k in "WBG":
        d = derived[k][ok]
        for b in cand:
            x = v[b, ok].astype(np.float64)
            mad = float(np.median(np.abs(x - d)))
            cor = float(np.corrcoef(x, d)[0, 1]) if np.std(x) > 0 else 0.0
            scores[(k, b)] = (mad, cor)
        best = sorted(cand, key=lambda b: -scores[(k, b)][1])
        print("    TC" + k + ": " + ", ".join(f"band {b + 1} r={scores[(k, b)][1]:.4f} median|diff|={scores[(k, b)][0]:.1f}"
                                         for b in best))
    # one-to-one assignment maximising total correlation
    from itertools import permutations
    best_perm = max(permutations(cand, 3), key=lambda p: sum(scores[(k, b)][1] for k, b in zip("WBG", p)))
    return dict(zip("WBG", best_perm))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--app", default=DEFAULT_APP)
    ap.add_argument("--bands", help="band numbers of W,B,G, e.g. 7,8,9")
    ap.add_argument("--no-csv", action="store_true")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--rasters", help="folder holding the three composite rasters (default: this script's folder)")
    args = ap.parse_args()
    if args.rasters:
        global HERE, CACHE
        HERE = args.rasters
        CACHE = os.path.join(HERE, "background_extract_cache.npz")

    t0 = time.time()
    xs, ys, keys, REQUIRED, cfg, wb_path = pixel_list(args.app)

    cached = None
    if os.path.exists(CACHE) and not args.refresh:
        z = np.load(CACHE, allow_pickle=False)
        if len(z["keys"]) == len(keys) and np.array_equal(z["keys"], keys):
            cached = z
            print(f"Using cached pixel values from {CACHE} (delete it or use --refresh to re-read the rasters)")

    extracted, medians = {}, {}
    for stat, fname in RASTERS.items():
        path = os.path.join(HERE, fname)
        print(f"\n{fname}")
        if cached is not None:
            extracted[stat], medians[stat] = cached[f"{stat}_vals"], cached[f"{stat}_med"]
        else:
            if not os.path.exists(path):
                sys.exit(f"Missing raster: {path}")
            extracted[stat], medians[stat], _ = extract(path, xs, ys, t0)
        print("  band medians (whole raster sample): " +
              ", ".join(f"{i + 1}: {m:.0f}" for i, m in enumerate(medians[stat])))
    if cached is None:
        np.savez(CACHE, keys=keys, **{f"{s}_vals": v for s, v in extracted.items()},
                 **{f"{s}_med": m for s, m in medians.items()})
        print(f"\nCached the extracted values in {CACHE}")

    if args.bands:
        nums = [int(b) - 1 for b in args.bands.split(",")]
        mapping = {s: dict(zip("WBG", nums)) for s in RASTERS}
        print(f"\nUsing the bands given: W={nums[0] + 1}, B={nums[1] + 1}, G={nums[2] + 1}")
    else:
        mapping = {}
        for stat in RASTERS:
            print(f"\n{stat} composite:")
            mapping[stat] = find_tc_bands(stat, extracted[stat], medians[stat])
            m = mapping[stat]
            print(f"  -> W = band {m['W'] + 1}, B = band {m['B'] + 1}, G = band {m['G'] + 1}")
        if len({tuple(sorted(m.items())) for m in mapping.values()}) > 1:
            print("\nNOTE: the three rasters came out with different band layouts - check the matches above.")

    df = pd.DataFrame({"key": keys, "x_albers": xs, "y_albers": ys})
    for stat in RASTERS:
        for k in "WBG":
            df[colname(stat, k)] = extracted[stat][mapping[stat][k]].astype(np.float32)
    have = df[REQUIRED].notna().all(axis=1)
    print(f"\n{have.sum():,} of {len(df):,} pixels have all nine values")
    d = df[have]

    print("\nChecks (should all be close to 100%):")
    for idx in "WBG":
        m = colname("mean", idx)
        share = ((d[f"Q10{idx}"] <= d[m]) & (d[m] <= d[f"Q90{idx}"])).mean()
        print(f"  Q10{idx} <= {m} <= Q90{idx}: {share:.1%}")
    print(f"  mean_TCB > mean_TCG: {(d['mean_TCB'] > d['mean_TCG']).mean():.1%} (100% at the model's training sites)")
    print(f"  mean_TCW < 0: {(d['mean_TCW'] < 0).mean():.1%}  (wetness is mostly negative)")
    print("\nMedians over waterbody pixels (model training-site medians in brackets):")
    train = {"mean_TCW": -1055, "mean_TCB": 2622, "mean_TCG": 763, "Q10B": 1916, "Q90B": 3305, "Q10G": 212, "Q90G": 1526}
    for c in REQUIRED:
        print(f"  {c:9s} {d[c].median():8.0f}   ({train.get(c, '')})")

    out = cfg.get("background_table", "data/background_pixels.parquet")
    out = out if os.path.isabs(out) else os.path.join(args.app, out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    d[["key"] + REQUIRED].to_parquet(out, index=False, compression="zstd", compression_level=19)
    with open(out.replace(".parquet", "_info.json"), "w") as f:
        json.dump(dict(source=[os.path.join(HERE, f) for f in RASTERS.values()],
                       bands_used={s: {k: v + 1 for k, v in m.items()} for s, m in mapping.items()},
                       n_pixels=int(len(d)), built=time.strftime("%Y-%m-%d %H:%M"), waterbodies=wb_path), f, indent=2)
    print(f"\nWrote {out} ({os.path.getsize(out) / 1e6:.1f} MB) - the app will use this automatically")

    if not args.no_csv:
        from pyproj import Transformer
        lng, lat = Transformer.from_crs(3577, 4326, always_xy=True).transform(d["x_albers"].to_numpy(),
                                                                              d["y_albers"].to_numpy())
        csv = os.path.join(HERE, "background_pixels_for_waterbodies.csv")
        d.drop(columns="key").assign(lng=lng, lat=lat).to_csv(csv, index=False, float_format="%.6f")
        print(f"Wrote {csv}")
    print(f"Done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
