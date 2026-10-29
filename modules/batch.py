"""Batch processing: reading a list of Unique_IDs and running/writing each waterbody."""
from __future__ import annotations

import io
import os
import tempfile
import zipfile

import pandas as pd

from . import pipeline
from .footprint import PIXEL_AREA_M2


def ids_from_upload(name: str, data: bytes, id_field: str = "Unique_ID") -> tuple[list[str], list[str], str]:
    """Returns (ids, column names found, column used). Accepts CSV, TXT (one ID
    per line), a zipped shapefile, GeoPackage or GeoJSON."""
    low = name.lower()
    if low.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(data), dtype=str)
    elif low.endswith((".txt", ".tsv")):
        txt = data.decode("utf-8-sig", errors="ignore")
        first = txt.splitlines()[0] if txt.strip() else ""
        if "\t" in first:
            df = pd.read_csv(io.StringIO(txt), sep="\t", dtype=str)
        else:
            vals = [v.strip() for line in txt.splitlines() for v in line.replace(",", " ").split() if v.strip()]
            df = pd.DataFrame({id_field: vals})
    elif low.endswith((".zip", ".gpkg", ".geojson", ".json")):
        import geopandas as gpd
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, os.path.basename(name))
            with open(p, "wb") as f:
                f.write(data)
            if low.endswith(".zip"):
                with zipfile.ZipFile(p) as z:
                    z.extractall(tmp)
                shp = [os.path.join(r, f) for r, _, fs in os.walk(tmp) for f in fs if f.lower().endswith(".shp")]
                if not shp:
                    raise ValueError("The zip file doesn't contain a .shp file.")
                p = shp[0]
            df = pd.DataFrame(gpd.read_file(p).drop(columns="geometry"))
    else:
        raise ValueError("Use a .csv, .txt, zipped shapefile (.zip), .gpkg or .geojson file.")
    cols = list(df.columns)
    col = id_field if id_field in cols else cols[0]
    ids = df[col].dropna().astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    ids = [i for i in dict.fromkeys(ids) if i]
    return ids, cols, col


def write_waterbody(out_dir_results: str, wb_id: str, polygon_albers, out_folder: str,
                    write_pixel_table: bool = False, start=None, end=None, min_clear: float = 0.0) -> pd.DataFrame:
    """Builds the per-scene area table for a processed waterbody and writes it
    to <out_folder>/per_waterbody/<id>_cover_area_by_scene.csv.

    Only scenes within start..end (UTC date) and at least `min_clear` clear are
    written, so scenes cached by earlier runs with other settings don't leak in."""
    per = os.path.join(out_folder, "per_waterbody")
    os.makedirs(per, exist_ok=True)
    empty_path = os.path.join(per, f"{wb_id}_cover_area_by_scene.csv")

    def _empty():
        # header-only file: marks the waterbody as finished (no usable scenes) so resumed batches skip it
        pd.DataFrame(columns=["Unique_ID", "date"]).to_csv(empty_path, index=False)
        return pd.DataFrame()

    res = pipeline.load_results(out_dir_results)
    if res is None or res[0].empty:
        return _empty()
    obs, pixels, _ = res
    keep = obs["clear_fraction"].fillna(0) >= min_clear - 1e-9
    if start is not None:
        keep &= obs["date"] >= pd.Timestamp(start)
    if end is not None:
        keep &= obs["date"] <= pd.Timestamp(end)
    obs = obs[keep]
    if obs.empty:
        return _empty()
    tbl = pipeline.scene_area_table(
        obs, wb_id, polygon_area_ha=polygon_albers.area / 10_000, n_pixels=len(pixels),
        footprint_area_ha=float(pixels["coverage_fraction"].sum()) * PIXEL_AREA_M2 / 10_000)
    per = os.path.join(out_folder, "per_waterbody")
    os.makedirs(per, exist_ok=True)
    tbl.to_csv(os.path.join(per, f"{wb_id}_cover_area_by_scene.csv"), index=False)
    if write_pixel_table:
        obs.to_parquet(os.path.join(per, f"{wb_id}_pixel_by_scene.parquet"), index=False)
    return tbl


def combine(out_folder: str, ids: list[str]) -> str | None:
    per = os.path.join(out_folder, "per_waterbody")
    parts = []
    for i in ids:
        p = os.path.join(per, f"{i}_cover_area_by_scene.csv")
        if os.path.exists(p):
            part = pd.read_csv(p, dtype={"Unique_ID": str})
            if len(part):
                parts.append(part)
    if not parts:
        return None
    out = os.path.join(out_folder, "all_waterbodies_cover_area_by_scene.csv")
    pd.concat(parts, ignore_index=True).to_csv(out, index=False)
    return out


LOG_COLUMNS = ["Unique_ID", "status", "group", "group_size", "group_minutes", "started", "finished", "minutes_total", "min_search", "min_cloud_check",
               "min_download", "min_model", "min_write_tables", "n_pixels", "scenes_found", "cloud_checked",
               "new_scenes", "scenes_in_table", "sec_per_new_scene", "warnings"]


def timing_fields(res: dict, wall_s: float, write_s: float) -> dict:
    """Per-waterbody timing for batch_log.csv (minutes)."""
    t = res.get("timing", {})
    m = lambda k: round(t.get(k, 0.0) / 60, 2)  # noqa: E731
    new = res.get("processed", 0)
    return dict(minutes_total=round(wall_s / 60, 2), min_search=m("search_s"), min_cloud_check=m("cloud_check_s"),
                min_download=m("download_s"), min_model=m("model_s"), min_write_tables=round(write_s / 60, 2),
                n_pixels=res.get("n_pixels"), scenes_found=res.get("scenes_found"),
                cloud_checked=res.get("cloud_checked", 0), new_scenes=new,
                sec_per_new_scene=round(wall_s / new, 2) if new else None)


def save_log(log_path: str, log: pd.DataFrame, rec: dict) -> pd.DataFrame:
    log = pd.concat([log, pd.DataFrame([rec])], ignore_index=True).drop_duplicates("Unique_ID", keep="last")
    cols = [c for c in LOG_COLUMNS if c in log.columns] + [c for c in log.columns if c not in LOG_COLUMNS]
    log = log[cols]
    log.to_csv(log_path, index=False)
    return log


def run_grouped(layer, groups: list[list[str]], source, start, end, out_folder: str, pixel_tables: bool,
                proc_kwargs: dict, log: pd.DataFrame, log_path: str, all_ids: list[str],
                on_group_start=None, on_progress=None, on_group_done=None) -> pd.DataFrame:
    """Runs a batch group by group (see modules/group.py). Tables are written
    only for members whose full date range is complete."""
    import time
    from .group import process_group

    total = sum(len(g) for g in groups)
    finished = 0
    for gi, ids in enumerate(groups, 1):
        if on_group_start:
            on_group_start(gi, len(groups), ids, finished, total)
        started = pd.Timestamp.now().isoformat(timespec="seconds")
        t0 = time.time()
        members = []
        for i in ids:
            info = layer.get(i)
            if info is not None:
                members.append((i, info["albers"], info["wgs"]))
        try:
            res = process_group(members, source, start, end, progress=on_progress, **proc_kwargs)
            err = None
        except Exception as e:  # noqa: BLE001
            res, err = dict(members={}, timing={}), str(e)
        wall = time.time() - t0
        t = res.get("timing", {})
        rows = []
        for i in ids:
            st = res["members"].get(i)
            rec = dict(Unique_ID=i, group=gi, group_size=len(ids), group_minutes=round(wall / 60, 2),
                       started=started, minutes_total=round(wall / 60 / max(1, len(ids)), 2),
                       min_search=round(t.get("search_s", 0) / 60, 2),
                       min_cloud_check=round(t.get("cloud_check_s", 0) / 60, 2),
                       min_download=round(t.get("download_s", 0) / 60, 2),
                       min_model=round(t.get("model_s", 0) / 60, 2))
            if err or st is None:
                rec.update(status="failed", warnings=err or "not processed")
            elif not st["complete"]:
                rec.update(status=f"incomplete - {st['failed']} scene(s) could not be read; re-run to retry",
                           n_pixels=st["n_pixels"], scenes_found=st["scenes_found"], new_scenes=st["processed"],
                           warnings=" ".join(st.get("warnings", [])))
            else:
                tw = time.time()
                tbl = write_waterbody(st["out_dir"], i, layer.get(i)["albers"], out_folder, pixel_tables,
                                      start=start, end=end, min_clear=proc_kwargs.get("min_clear", 0.0))
                rec.update(status="done" if len(tbl) else "no scenes passed the filters",
                           n_pixels=st["n_pixels"], scenes_found=st["scenes_found"], new_scenes=st["processed"],
                           scenes_in_table=len(tbl), min_write_tables=round((time.time() - tw) / 60, 2),
                           warnings=" ".join(st.get("warnings", [])))
            rec["finished"] = pd.Timestamp.now().isoformat(timespec="seconds")
            log = save_log(log_path, log, rec)
            rows.append(rec)
        combine(out_folder, all_ids)
        finished += len(ids)
        if on_group_done:
            on_group_done(gi, len(groups), rows, wall, finished, total)
    return log
