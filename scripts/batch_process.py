"""
Batch version of the app - a streamlined replacement for
"script june 2026 no lat longs.R" + "Per waterbody concat results script.R".

Processes a list of waterbodies one after another with the same code the
app uses, saving to the same results folder (so anything done here shows up
in the app and vice versa), then optionally writes per-scene cover-area tables to a folder.

Examples (from the app folder, environment active):

    # a list of waterbodies, grouped by 10 km cells (default), tables written to a folder
    python scripts/batch_process.py --ids-file ids.csv --out "D:/cover results"

    # three waterbodies, full archive from DEA, one at a time
    python scripts/batch_process.py --ids 70730 70731 70732 --group-km 0

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
    ap.add_argument("--group-km", type=float, default=10.0,
                    help="process waterbodies in the same N km grid cell together (one download per scene for the "
                         "group; identical results, much faster). 0 = one at a time. Needs --out. Default 10")
    ap.add_argument("--rewrite", action="store_true",
                    help="with grouping: also re-process waterbodies whose table is already in --out")
    ap.add_argument("--skip-processing", action="store_true", help="only write tables for waterbodies already processed")
    args = ap.parse_args()

    cfg = json.load(open(os.path.join(APP, "config", "settings.json")))
    user = os.path.join(APP, "config", "user_settings.json")
    if os.path.exists(user):
        cfg.update(json.load(open(user)))
    results_base = cfg["results_dir"] if os.path.isabs(cfg["results_dir"]) else os.path.join(APP, cfg["results_dir"])
    workers = args.workers or cfg.get("max_workers", 16)

    print("Loading waterbodies layer...")
    rel = lambda p: p if os.path.isabs(p) else os.path.join(APP, p)  # noqa: E731
    layer = WaterbodyLayer(rel(cfg["waterbodies_path"]), cfg["waterbody_id_field"], cfg.get("waterbody_label_fields"))
    ids = read_ids(args, cfg["waterbody_id_field"])
    if not ids:
        ap.error("give --ids and/or --ids-file")
    forests, predictors = load_forests(os.path.join(APP, "models", "wetland_cover_models.npz"))
    thresholds = load_thresholds(os.path.join(APP, "config", "thresholds.csv"))
    lookup = CoverLookup(os.path.join(APP, "config", "cover_combinations.csv"))
    source = (DEASource(cfg["stac_urls"], cfg["collections"], cfg.get("fmask_good_values", [1, 4, 5]), workers)
              if args.source == "dea" else LocalTCISource(cfg["local_tci_folder"], min(workers, 4)))
    print(f"{len(ids)} waterbodies, source: {source.name}, prediction engine: {engine_name()}")

    if args.group_km > 0 and args.out and not args.skip_processing:
        from modules.group import make_groups
        os.makedirs(args.out, exist_ok=True)
        log_path = os.path.join(args.out, "batch_log.csv")
        log = pd.read_csv(log_path, dtype={"Unique_ID": str}) if os.path.exists(log_path) else pd.DataFrame()
        known = set(layer.ids())
        todo = [i for i in ids if i in known and (args.rewrite or not os.path.exists(
            os.path.join(args.out, "per_waterbody", f"{i}_cover_area_by_scene.csv")))]
        missing = [i for i in ids if i not in known]
        if missing:
            print(f"{len(missing)} IDs not in the waterbodies layer (skipped): {', '.join(missing[:10])}")
        groups = make_groups(layer.gdf_albers, layer.id_field, todo, args.group_km)
        print(f"{len(todo):,} waterbodies to process in {len(groups):,} groups of up to {args.group_km:g} km "
              f"({len(ids) - len(todo) - len(missing):,} already written)")
        t_all = time.time()
        gmins = []
        state = {"last": -1.0}

        def g_start(gi, ng, gids, done_n, total):
            left = (ng - gi + 1) * sum(gmins) / len(gmins) if gmins else None
            eta = "" if left is None else (f", about {left / 60:.1f} h left" if left >= 90 else f", about {left:.0f} min left")
            print(f"\nGroup {gi}/{ng}: {len(gids)} waterbodies ({done_n:,}/{total:,} done, "
                  f"{(time.time() - t_all) / 60:.1f} min elapsed{eta})", flush=True)
            state["last"] = -1.0

        def g_prog(f, msg):
            if f - state["last"] >= 0.1 or f >= 1:
                state["last"] = f
                print(f"  {msg}", flush=True)

        def g_done(gi, ng, recs, wall, done_n, total):
            gmins.append(wall / 60)
            st = {}
            for r in recs:
                st[str(r.get("status")).split(" -")[0]] = st.get(str(r.get("status")).split(" -")[0], 0) + 1
            print(f"Group {gi}/{ng} finished in {wall / 60:.1f} min: " + ", ".join(f"{v} {k}" for k, v in st.items()),
                  flush=True)

        proc_kwargs = dict(results_base=results_base, background_path=cfg["background_raster"],
                           background_table=rel(cfg.get("background_table", "data/background_pixels.parquet")),
                           background_bands=cfg["background_band_names"], forests=forests, predictors=predictors,
                           thresholds=thresholds, lookup=lookup, min_clear=args.min_clear,
                           include_ls7_slc_off=not args.no_slc_off, pad_m=cfg.get("pad_m", 60), max_workers=workers,
                           model_file="wetland_cover_models.npz")
        log = batchmod.run_grouped(layer, groups, source, date.fromisoformat(args.start), date.fromisoformat(args.end),
                                   args.out, args.pixel_tables, proc_kwargs, log, log_path, ids,
                                   on_group_start=g_start, on_progress=g_prog, on_group_done=g_done)
        bad = log[log["Unique_ID"].isin(todo) & log["status"].astype(str).str.startswith(("incomplete", "failed"))]
        print(f"\nAll done in {(time.time() - t_all) / 60:.1f} min. Combined table: "
              f"{batchmod.combine(args.out, ids)}")
        if len(bad):
            print(f"{len(bad)} waterbodies incomplete or failed - run the same command again to retry them.")
        return

    log_path = os.path.join(args.out, "batch_log.csv") if args.out else None
    if args.out:
        os.makedirs(args.out, exist_ok=True)
    log = pd.read_csv(log_path, dtype={"Unique_ID": str}) if log_path and os.path.exists(log_path) else pd.DataFrame()
    t_all = time.time()
    done_minutes = []
    for n, wb in enumerate(ids, 1):
        sel = layer.get(wb)
        if sel is None:
            print(f"[{n}/{len(ids)}] {wb}: not in the waterbodies layer - skipped")
            continue
        t_wb = time.time()
        rec = dict(Unique_ID=wb, started=pd.Timestamp.now().isoformat(timespec="seconds"))
        eta = f", about {(len(ids) - n + 1) * sum(done_minutes) / len(done_minutes):.0f} min left" if done_minutes else ""
        print(f"[{n}/{len(ids)}] {wb}: starting ({(time.time() - t_all) / 60:.1f} min elapsed{eta})", flush=True)
        last = [0.0]

        def prog(f, msg, wb=wb, n=n):
            if f - last[0] >= 0.1 or f >= 1:
                last[0] = f
                print(f"  [{n}/{len(ids)}] {wb}: {msg}", flush=True)
        try:
            res = dict(processed=0, timing={})
            if not args.skip_processing:
                res = pipeline.process_waterbody(
                    wb, sel["albers"], sel["wgs"], source, date.fromisoformat(args.start), date.fromisoformat(args.end),
                    results_base=results_base, background_path=cfg["background_raster"],
                    background_table=rel(cfg.get("background_table", "data/background_pixels.parquet")),
                    background_bands=cfg["background_band_names"], forests=forests, predictors=predictors,
                    thresholds=thresholds, lookup=lookup, min_clear=args.min_clear,
                    include_ls7_slc_off=not args.no_slc_off, pad_m=cfg.get("pad_m", 60), max_workers=workers,
                    progress=prog, model_file="wetland_cover_models.npz")
            t_write = time.time()
            n_rows = None
            if args.out:
                tbl = batchmod.write_waterbody(pipeline.results_dir(results_base, source.name, wb), wb, sel["albers"],
                                               args.out, args.pixel_tables, start=date.fromisoformat(args.start),
                                               end=date.fromisoformat(args.end), min_clear=args.min_clear)
                n_rows = len(tbl)
            wall = time.time() - t_wb
            rec.update(status="done" if n_rows != 0 else "no scenes passed the filters", scenes_in_table=n_rows,
                       **batchmod.timing_fields(res, wall, time.time() - t_write),
                       warnings=" ".join(res.get("warnings", [])))
            done_minutes.append(wall / 60)
            t = res.get("timing", {})
            print(f"[{n}/{len(ids)}] {wb}: {res.get('processed', 0)} new scenes ({res.get('scenes_found', 0)} found) "
                  f"in {wall / 60:.1f} min - search {t.get('search_s', 0) / 60:.1f}, cloud check "
                  f"{t.get('cloud_check_s', 0) / 60:.1f}, download {t.get('download_s', 0) / 60:.1f}, "
                  f"models {t.get('model_s', 0) / 60:.1f} min. {rec['warnings']}", flush=True)
        except Exception as e:  # noqa: BLE001
            rec.update(status="failed", minutes_total=round((time.time() - t_wb) / 60, 2), warnings=str(e))
            print(f"[{n}/{len(ids)}] {wb}: FAILED - {e}")
        rec["finished"] = pd.Timestamp.now().isoformat(timespec="seconds")
        if log_path:
            log = batchmod.save_log(log_path, log, rec)
            batchmod.combine(args.out, ids)
    print(f"All done in {(time.time() - t_all) / 60:.1f} min")
    if args.out:
        out = batchmod.combine(args.out, ids)
        print(f"Wrote {out} and {log_path}" if out else "No tables written.")

if __name__ == "__main__":
    main()
