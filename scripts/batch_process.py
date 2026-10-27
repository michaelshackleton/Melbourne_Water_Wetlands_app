"""
Batch version of the app - a streamlined replacement for
"script june 2026 no lat longs.R" + "Per waterbody concat results script.R".

Processes a list of waterbodies one after another with the same code the
app uses, saving to the same results folder (so anything done here shows up
in the app and vice versa), then optionally writes per-scene cover-area tables to a folder.

Examples (from the app folder, environment active):

    # three waterbodies, full archive from DEA
    python scripts/batch_process.py --ids 70730 70731 70732

    # every Unique_ID in a CSV / shapefile column, 2000-2024, local TCI rasters
    python scripts/batch_process.py --ids-file "D:/eDNA project/eDNA_waterbodies_1000m_buffer_selection.shp" \
        --start 2000-01-01 --end 2024-12-31 --source local

    # write per-scene cover-area tables for everything processed so far
    python scripts/batch_process.py --ids-file ids.csv --skip-processing --out "D:/cover results"

Re-running is safe: scenes already processed for a waterbody are skipped.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import date

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)

import pandas as pd  # noqa: E402

from modules import batch as batchmod  # noqa: E402
from modules import pipeline  # noqa: E402
from modules.classify import CoverLookup, load_thresholds  # noqa: E402
from modules.rf_engine import engine_name, load_forests  # noqa: E402
from modules.sources import DEASource, LocalTCISource  # noqa: E402
from modules.waterbodies import WaterbodyLayer  # noqa: E402


def read_ids(args, id_field) -> list[str]:
    ids = [str(i) for i in (args.ids or [])]
    if args.ids_file:
        f = args.ids_file
        if f.lower().endswith((".shp", ".gpkg", ".geojson")):
            import geopandas as gpd
            df = gpd.read_file(f)
        elif f.lower().endswith(".csv"):
            df = pd.read_csv(f)
        else:
            df = pd.DataFrame({id_field: [l.strip() for l in open(f) if l.strip()]})
        col = id_field if id_field in df.columns else df.columns[0]
        ids += df[col].astype(str).str.replace(r"\.0$", "", regex=True).tolist()
    return list(dict.fromkeys(ids))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ids", nargs="*", help="Unique_IDs to process")
    ap.add_argument("--ids-file", help="CSV / text / shapefile containing Unique_IDs")
    ap.add_argument("--start", default="1987-01-01")
    ap.add_argument("--end", default=date.today().isoformat())
    ap.add_argument("--source", choices=["dea", "local"], default="dea")
    ap.add_argument("--min-clear", type=float, default=0.5)
    ap.add_argument("--no-slc-off", action="store_true", help="exclude Landsat 7 scenes after May 2003")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--out", metavar="FOLDER", help="write per-scene cover-area tables (one per waterbody plus "
                                                     "all_waterbodies_cover_area_by_scene.csv) to FOLDER - same "
                                                     "output as the app's Batch processing tab")
    ap.add_argument("--pixel-tables", action="store_true", help="with --out, also write pixel-by-scene parquet files")
    ap.add_argument("--skip-processing", action="store_true", help="only write tables for waterbodies already processed")
    args = ap.parse_args()

    cfg = json.load(open(os.path.join(APP, "config", "settings.json")))
    user = os.path.join(APP, "config", "user_settings.json")
    if os.path.exists(user):
        cfg.update(json.load(open(user)))
    results_base = cfg["results_dir"] if os.path.isabs(cfg["results_dir"]) else os.path.join(APP, cfg["results_dir"])
    workers = args.workers or cfg.get("max_workers", 16)

    print("Loading waterbodies layer...")
    layer = WaterbodyLayer(cfg["waterbodies_path"], cfg["waterbody_id_field"], cfg.get("waterbody_label_fields"))
    ids = read_ids(args, cfg["waterbody_id_field"])
    if not ids:
        ap.error("give --ids and/or --ids-file")
    forests, predictors = load_forests(os.path.join(APP, "models", "wetland_cover_models.npz"))
    thresholds = load_thresholds(os.path.join(APP, "config", "thresholds.csv"))
    lookup = CoverLookup(os.path.join(APP, "config", "cover_combinations.csv"))
    source = (DEASource(cfg["stac_urls"], cfg["collections"], cfg.get("fmask_good_values", [1, 4, 5]), workers)
              if args.source == "dea" else LocalTCISource(cfg["local_tci_folder"], min(workers, 4)))
    print(f"{len(ids)} waterbodies, source: {source.name}, prediction engine: {engine_name()}")

    if not args.skip_processing:
        t_all = time.time()
        for n, wb in enumerate(ids, 1):
            sel = layer.get(wb)
            if sel is None:
                print(f"[{n}/{len(ids)}] {wb}: not in the waterbodies layer - skipped")
                continue
            last = [0.0]

            def prog(f, msg, wb=wb, n=n):
                if f - last[0] >= 0.1 or f >= 1:
                    last[0] = f
                    print(f"  [{n}/{len(ids)}] {wb}: {msg}", flush=True)
            try:
                r = pipeline.process_waterbody(
                    wb, sel["albers"], sel["wgs"], source, date.fromisoformat(args.start), date.fromisoformat(args.end),
                    results_base=results_base, background_path=cfg["background_raster"],
                    background_bands=cfg["background_band_names"], forests=forests, predictors=predictors,
                    thresholds=thresholds, lookup=lookup, min_clear=args.min_clear,
                    include_ls7_slc_off=not args.no_slc_off, pad_m=cfg.get("pad_m", 60), max_workers=workers,
                    progress=prog, model_file="wetland_cover_models.npz")
                print(f"[{n}/{len(ids)}] {wb}: {r['processed']} new scenes ({r['scenes_found']} found) in "
                      f"{r['seconds'] / 60:.1f} min. {' '.join(r['warnings'])}")
            except Exception as e:  # noqa: BLE001
                print(f"[{n}/{len(ids)}] {wb}: FAILED - {e}")
        print(f"All done in {(time.time() - t_all) / 60:.1f} min")

    if args.out:
        for wb in ids:
            sel = layer.get(wb)
            if sel is None:
                continue
            batchmod.write_waterbody(pipeline.results_dir(results_base, source.name, wb), wb, sel["albers"],
                                     args.out, args.pixel_tables, start=date.fromisoformat(args.start),
                                     end=date.fromisoformat(args.end), min_clear=args.min_clear)
        out = batchmod.combine(args.out, ids)
        print(f"Wrote {out}" if out else "Nothing to write.")

if __name__ == "__main__":
    main()
