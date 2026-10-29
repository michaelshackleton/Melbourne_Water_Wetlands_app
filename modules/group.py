"""
Group processing for batches: nearby waterbodies share one read per scene.

Downloading dominates the run time, and neighbouring waterbodies sit in the
same satellite image tiles, so processing them one at a time fetches the same
tiles over and over. Here waterbodies are grouped by a regular grid (default
10 km cells, by polygon centroid). For each group:

  * DEA is searched once for the group's area;
  * each scene's cloud mask and bands are read once, over one window covering
    every member's footprint;
  * the pixels are then split back out to each member, which is classified
    exactly as in single-waterbody mode (same footprint, background values,
    models and thresholds) and saved to the same results/<source>/<Unique_ID>/
    folder, with the same scene keys - so single and group runs share a cache.

Completeness: a group is worked through for every scene in the date range
before anything is reported as finished. Failed reads are retried once at the
end of the group; any member still missing scenes is reported as incomplete
(its batch table is not written), and the next run fills in only what is
missing.
"""
from __future__ import annotations

import json
import os
import time
from datetime import date, datetime
from typing import Callable, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from affine import Affine

from . import background as bgmod
from .footprint import RES, Footprint, build_footprint
from .pipeline import (CHUNK_SCENES, _load_manifest, _save_manifest, _scene_row, predict_chunk, results_dir)
from .sources import Scene, run_parallel

FINAL_STATUSES = {"processed", "skipped_cloudy", "no_clear_pixels"}


def make_groups(gdf_albers: gpd.GeoDataFrame, id_field: str, ids: list[str], cell_km: float = 10.0) -> list[list[str]]:
    """Group IDs by the grid cell their polygon centroid falls in. Groups are
    ordered west-to-east, north-to-south so neighbouring groups run in turn."""
    sel = gdf_albers[gdf_albers[id_field].isin(ids)]
    c = sel.geometry.centroid
    cell = cell_km * 1000.0
    key = pd.DataFrame({"id": sel[id_field].values, "cx": np.floor(c.x.values / cell).astype(int),
                        "cy": np.floor(c.y.values / cell).astype(int)})
    order = {i: n for n, i in enumerate(ids)}
    groups = []
    for (_, _), g in key.sort_values(["cy", "cx"], ascending=[False, True]).groupby(["cy", "cx"], sort=False):
        groups.append(sorted(g["id"].tolist(), key=lambda i: order.get(i, 0)))
    return groups


class _Member:
    def __init__(self, wb_id, albers, wgs, pad_m):
        self.id = str(wb_id)
        self.albers = albers
        self.wgs = wgs
        self.fp = build_footprint(wb_id, albers, pad_m=pad_m)


def _group_footprint(members: list[_Member]) -> tuple[Footprint, list[slice]]:
    """One grid covering every member; its 'pixels' are all members' pixels
    (concatenated, mapped onto the group grid)."""
    x0 = min(m.fp.bounds[0] for m in members)
    y0 = min(m.fp.bounds[1] for m in members)
    x1 = max(m.fp.bounds[2] for m in members)
    y1 = max(m.fp.bounds[3] for m in members)
    width, height = int(round((x1 - x0) / RES)), int(round((y1 - y0) / RES))
    rows, cols, xs, ys, cov, slices = [], [], [], [], [], []
    n = 0
    for m in members:
        mx0, _, _, my1 = m.fp.bounds
        dr, dc = int(round((y1 - my1) / RES)), int(round((mx0 - x0) / RES))
        p = m.fp.pixels
        rows.append(p["row"].to_numpy() + dr)
        cols.append(p["col"].to_numpy() + dc)
        xs.append(p["x_albers"].to_numpy())
        ys.append(p["y_albers"].to_numpy())
        cov.append(p["coverage_fraction"].to_numpy())
        slices.append(slice(n, n + len(p)))
        n += len(p)
    pixels = pd.DataFrame({"pixel_id": np.arange(n), "row": np.concatenate(rows), "col": np.concatenate(cols),
                           "x_albers": np.concatenate(xs), "y_albers": np.concatenate(ys),
                           "coverage_fraction": np.concatenate(cov)})
    poly = shapely.union_all([m.albers for m in members])
    return Footprint("group", poly, Affine(RES, 0, x0, 0, -RES, y1), width, height, pixels), slices


def _touches(scene: Scene, poly_wgs) -> bool:
    from shapely.geometry import shape
    for it in scene.parts:
        geom = getattr(it, "geometry", None)
        if geom is None:
            return True
        try:
            if shape(geom).intersects(poly_wgs):
                return True
        except Exception:  # noqa: BLE001
            return True
    return False


def _slice_tci(tci: dict, sl: slice) -> dict:
    return {k: v[sl] for k, v in tci.items()}


def process_group(
    members_in: list[tuple],  # (wb_id, polygon_3577, polygon_wgs84)
    source,
    start: date,
    end: date,
    *,
    results_base: str,
    background_path: str,
    background_bands: list[str],
    background_table: str | None,
    forests, predictors, thresholds, lookup,
    min_clear: float = 0.5,
    include_ls7_slc_off: bool = True,
    pad_m: float = 60.0,
    max_workers: int = 16,
    progress: Optional[Callable[[float, str], None]] = None,
    model_file: str = "",
    retries: int = 1,
) -> dict:
    """Processes every member of a group over the full date range. Returns
    {"members": {id: stats}, "timing": {...}, "scenes_found": n}."""
    t0 = time.time()
    timing = dict(search_s=0.0, cloud_check_s=0.0, download_s=0.0, model_s=0.0)

    def report(frac, msg):
        if progress:
            progress(max(0.0, min(1.0, frac)), msg)

    # ---- set up members (footprints, background, files) ---------------------
    report(0.01, "Setting up pixel grids and background values...")
    members, stats, skipped = [], {}, {}
    for wb_id, albers, wgs in members_in:
        m = _Member(wb_id, albers, wgs, pad_m)
        m.out_dir = results_dir(results_base, source.name, m.id)
        os.makedirs(os.path.join(m.out_dir, "obs"), exist_ok=True)
        if len(m.fp.pixels) == 0:
            skipped[m.id] = "polygon touches no pixels"
            continue
        bg = bgmod.read_background(background_path, background_bands, m.fp, table_path=background_table)
        m.warnings = bgmod.background_sanity(bg)
        m.bg = bg[[c for c in predictors if c in bg.columns]]
        pixels = pd.concat([m.fp.pixels, m.bg], axis=1)
        pixels.insert(0, "Unique_ID", m.id)
        pixels.to_parquet(os.path.join(m.out_dir, "pixels.parquet"), index=False)
        gpd.GeoDataFrame({"Unique_ID": [m.id]}, geometry=[wgs], crs="EPSG:4326").to_file(
            os.path.join(m.out_dir, "footprint.geojson"), driver="GeoJSON")
        with open(os.path.join(m.out_dir, "run_info.json"), "w") as f:
            json.dump(dict(Unique_ID=m.id, source=source.name, model_file=model_file, thresholds=thresholds,
                           background_raster=background_path, background_table=background_table,
                           background_band_names=background_bands, pad_m=pad_m,
                           last_run=datetime.now().isoformat(timespec="seconds"),
                           last_date_range=[start.isoformat(), end.isoformat()], min_clear=min_clear,
                           include_ls7_slc_off=include_ls7_slc_off, processed_in_group=True), f, indent=2)
        if m.bg.isna().all(axis=1).all():
            skipped[m.id] = "no background data for this waterbody"
            continue
        m.cov = m.fp.pixels["coverage_fraction"].to_numpy()
        members.append(m)
    for i, why in skipped.items():
        stats[i] = dict(n_pixels=0, scenes_found=0, processed=0, failed=0, complete=False, warnings=[why])
    if not members:
        return dict(members=stats, timing=dict(total_s=time.time() - t0, **timing), scenes_found=0)

    gfp, slices = _group_footprint(members)
    union_wgs = shapely.union_all([m.wgs for m in members])

    report(0.03, f"Searching {source.name} for scenes over {len(members)} waterbodies...")
    _t = time.time()
    scenes = source.search(gfp, union_wgs, start, end, include_ls7_slc_off=include_ls7_slc_off)
    timing["search_s"] = time.time() - _t
    by_key = {s.key: s for s in scenes}
    for m in members:
        m.scenes = [s.key for s in scenes if _touches(s, m.wgs)]
        m.processed_new = 0

    for attempt in range(retries + 1):
        # ---- what each member still needs (from its own scene list) ------------
        need_mask, cand_for = set(), {}
        for m in members:
            m.man = _load_manifest(m.out_dir)
            last = m.man.drop_duplicates("scene_key", keep="last").set_index("scene_key")
            m.known = {k: v for k, v in last["clear_fraction"].items() if np.isfinite(v)} if len(last) else {}
            m.done = set(last.index[last["status"] == "processed"]) if len(last) else set()
            m.todo = [k for k in m.scenes if k not in m.done]
            for k in m.todo:
                if k not in m.known:
                    need_mask.add(k)
        need_mask = [by_key[k] for k in sorted(need_mask, key=lambda k: by_key[k].datetime_utc)]
        if not any(m.todo for m in members):
            break
        label = "" if attempt == 0 else " (retrying failed scenes)"

        # ---- pass 1: one cloud-mask read per scene for the whole group --------
        masks = {}
        if need_mask:
            _t = time.time()
            res = run_parallel(lambda s: source.clear_mask(s, gfp), need_mask, max_workers,
                               progress=lambda d, n, _m: report(0.05 + 0.3 * d / n,
                                                                f"Checking cloud cover{label}: scene {d}/{n}"))
            timing["cloud_check_s"] += time.time() - _t
            for m, sl in zip(members, slices):
                rows = []
                for s in need_mask:
                    if s.key not in m.todo:
                        continue
                    r = res.get(s.key)
                    if isinstance(r, Exception) or r is None:
                        rows.append(_scene_row(s, np.nan, "failed", str(r) if r is not None else "not run"))
                        continue
                    cs = m.cov.sum()
                    m.known[s.key] = float((m.cov * r[sl]).sum() / cs) if cs > 0 else 0.0
                if rows:
                    m.man = pd.concat([m.man, pd.DataFrame(rows)], ignore_index=True)
            for s in need_mask:
                r = res.get(s.key)
                if r is not None and not isinstance(r, Exception):
                    masks[s.key] = r

        # ---- decide per member: process / skip ---------------------------------
        for m in members:
            rows = []
            m.cand = set()
            for k in m.todo:
                cf = m.known.get(k)
                if cf is None:
                    continue
                if cf <= 0:
                    rows.append(_scene_row(by_key[k], cf, "no_clear_pixels"))
                elif cf < min_clear:
                    rows.append(_scene_row(by_key[k], cf, "skipped_cloudy"))
                else:
                    m.cand.add(k)
                    cand_for.setdefault(k, []).append(m)
            if rows:
                m.man = pd.concat([m.man, pd.DataFrame(rows)], ignore_index=True)
            _save_manifest(m.out_dir, m.man)

        # ---- pass 2: one band read per scene, split out to members -------------
        cand = sorted(cand_for, key=lambda k: by_key[k].datetime_utc)
        for ci in range(0, len(cand), CHUNK_SCENES):
            chunk = [by_key[k] for k in cand[ci:ci + CHUNK_SCENES]]

            def read_one(s: Scene):
                clear = masks.get(s.key)
                if clear is None:
                    clear = source.clear_mask(s, gfp)
                return source.read_tci(s, gfp, clear)

            base = 0.35 + 0.63 * ci / max(1, len(cand))
            span = 0.63 * len(chunk) / max(1, len(cand))
            _t = time.time()
            res = run_parallel(read_one, chunk, max_workers,
                               progress=lambda d, n, _m: report(base + span * 0.85 * d / n,
                                                                f"Reading scenes{label} {ci + d}/{len(cand)}"))
            timing["download_s"] += time.time() - _t
            report(base + span * 0.9, f"Running models for {len(members)} waterbodies...")
            _t = time.time()
            for m, sl in zip(members, slices):
                ok, rows = [], []
                for s in chunk:
                    if s.key not in m.cand:
                        continue
                    r = res.get(s.key)
                    if isinstance(r, Exception) or r is None:
                        rows.append(_scene_row(s, m.known.get(s.key), "failed",
                                               str(r) if r is not None else "not run"))
                        continue
                    ok.append((s, _slice_tci(r, sl)))
                    rows.append(_scene_row(s, m.known.get(s.key), "processed"))
                if ok:
                    df = predict_chunk(ok, m.fp, m.bg, forests, predictors, thresholds, lookup, m.id)
                    df.to_parquet(os.path.join(m.out_dir, "obs", f"part-{int(time.time() * 1000)}.parquet"),
                                  index=False)
                    m.processed_new += len(ok)
                if rows:
                    m.man = pd.concat([m.man, pd.DataFrame(rows)], ignore_index=True)
                    _save_manifest(m.out_dir, m.man)
            timing["model_s"] += time.time() - _t
            for s in chunk:
                masks.pop(s.key, None)

    # ---- completeness per member ----------------------------------------------
    for m in members:
        man = _load_manifest(m.out_dir).drop_duplicates("scene_key", keep="last").set_index("scene_key")
        status = man["status"].to_dict() if len(man) else {}
        missing = [k for k in m.scenes if status.get(k) not in FINAL_STATUSES
                   or (status.get(k) == "skipped_cloudy" and m.known.get(k, 0) >= min_clear)]
        stats[m.id] = dict(n_pixels=len(m.fp.pixels), scenes_found=len(m.scenes), processed=m.processed_new,
                           failed=len(missing), complete=not missing, warnings=m.warnings, out_dir=m.out_dir)
    report(1.0, "Group done")
    timing["total_s"] = time.time() - t0
    return dict(members=stats, timing=timing, scenes_found=len(scenes), n_members=len(members),
                group_pixels=len(gfp.pixels), window=(gfp.width, gfp.height))
