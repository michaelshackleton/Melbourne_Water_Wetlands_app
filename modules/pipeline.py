"""
End-to-end processing for one waterbody - replaces the per-waterbody loop in
"script june 2026 no lat longs.R" plus the compile step in
"Per waterbody concat results script.R".

For each waterbody:
  1. build the 30 m pixel grid and coverage fractions (footprint.py);
  2. read the 9 background predictors once (background.py) - the original
     script re-cropped and re-projected them for every scene;
  3. find scenes (sources.py) and work out how much of the footprint each
     one sees clearly (fmask pass);
  4. for scenes that are clear enough, read the six NBART bands only over
     the footprint, compute TCW/TCB/TCG, run the 7 models (rf_engine.py) and
     classify (classify.py);
  5. save everything under results/<source>/<Unique_ID>/ as it goes, so a
     re-run (e.g. with a longer date range) only fetches new scenes and an
     interrupted run keeps what it already finished.

Saved files
  pixels.parquet       one row per pixel: coverage fraction, coordinates,
                       background predictors
  obs/part-*.parquet   one row per pixel per processed scene
  scenes.csv           every scene considered, with clear fraction and status
  footprint.geojson    the waterbody polygon
  run_info.json        model/threshold/source details
"""
from __future__ import annotations

import json
import os
import time
from datetime import date, datetime, timezone
from typing import Callable, Optional

import geopandas as gpd
import numpy as np
import pandas as pd

from . import background as bgmod
from .classify import MIX_ORDER, NO_DATA, CoverLookup
from .footprint import PIXEL_AREA_M2, Footprint, build_footprint
from .rf_engine import Forest, predict_presence
from .sources import Scene, run_parallel

ProgressCB = Optional[Callable[[float, str], None]]

SCENE_COLUMNS = ["scene_key", "datetime_utc", "date", "time", "solar_date", "platform", "source",
                 "n_parts", "clear_fraction", "status", "error", "processed_at"]

CHUNK_SCENES = 60


def results_dir(base: str, source_name: str, wb_id) -> str:
    safe_source = "dea" if source_name == "DEA" else "local_tci"
    return os.path.join(base, safe_source, str(wb_id))


def _load_manifest(out_dir: str) -> pd.DataFrame:
    p = os.path.join(out_dir, "scenes.csv")
    if os.path.exists(p):
        df = pd.read_csv(p, dtype={"scene_key": str})
        for c in SCENE_COLUMNS:
            if c not in df.columns:
                df[c] = np.nan
        return df[SCENE_COLUMNS]
    return pd.DataFrame(columns=SCENE_COLUMNS)


def _save_manifest(out_dir: str, man: pd.DataFrame):
    man = man.sort_values("datetime_utc").drop_duplicates("scene_key", keep="last")
    tmp = os.path.join(out_dir, "scenes.csv.tmp")
    man.to_csv(tmp, index=False)
    os.replace(tmp, os.path.join(out_dir, "scenes.csv"))


def _scene_row(s: Scene, clear_fraction, status, error="") -> dict:
    return dict(scene_key=s.key, datetime_utc=s.datetime_utc.isoformat(), date=s.utc_date, time=s.utc_time,
                solar_date=s.solar_date.isoformat(), platform=s.platform, source=s.source, n_parts=len(s.parts),
                clear_fraction=clear_fraction, status=status, error=error,
                processed_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))


def predict_chunk(scene_tci: list[tuple[Scene, dict]], fp: Footprint, bg: pd.DataFrame,
                  forests: dict[str, Forest], predictors: list[str], thresholds: dict[str, float],
                  lookup: CoverLookup, wb_id) -> pd.DataFrame:
    n_px = len(fp.pixels)
    n_sc = len(scene_tci)
    cols = {}
    for name in ["wetness", "brightness", "greenness"]:
        cols[name] = np.concatenate([t[name] for _, t in scene_tci]).astype(np.float32)
    for name in bg.columns:
        cols[name] = np.tile(bg[name].to_numpy(), n_sc)
    X = np.column_stack([cols[p].astype(np.float64) for p in predictors])
    valid = np.isfinite(X).all(axis=1)

    out = {
        "Unique_ID": np.repeat(str(wb_id), n_px * n_sc),
        "scene_key": np.repeat([s.key for s, _ in scene_tci], n_px),
        "pixel_id": np.tile(fp.pixels["pixel_id"].to_numpy(), n_sc),
        "wetness": cols["wetness"], "brightness": cols["brightness"], "greenness": cols["greenness"],
    }
    binaries = {}
    for m in MIX_ORDER:
        p = predict_presence(forests[m], X)
        out[m] = p.astype(np.float32)
        b = np.where(valid, (p >= thresholds[m]).astype(np.float32), np.nan)
        out[f"{m}_binary"] = b
        binaries[m] = b
    desc, mix = lookup.describe(binaries, valid)
    out["cover_mix"] = mix
    out["cover_description"] = desc
    return pd.DataFrame(out)


def process_waterbody(
    wb_id,
    polygon_3577,
    polygon_wgs84,
    source,
    start: date,
    end: date,
    *,
    results_base: str,
    background_path: str,
    background_bands: list[str],
    background_table: str | None = None,
    forests: dict[str, Forest],
    predictors: list[str],
    thresholds: dict[str, float],
    lookup: CoverLookup,
    min_clear: float = 0.5,
    include_ls7_slc_off: bool = True,
    pad_m: float = 60.0,
    max_workers: int = 16,
    progress: ProgressCB = None,
    should_stop: Optional[Callable[[], bool]] = None,
    model_file: str = "",
) -> dict:
    """Processes (or extends) one waterbody. Returns a dict of run statistics."""
    t0 = time.time()

    def report(frac, msg):
        if progress:
            progress(max(0.0, min(1.0, frac)), msg)

    out_dir = results_dir(results_base, source.name, wb_id)
    os.makedirs(os.path.join(out_dir, "obs"), exist_ok=True)

    report(0.01, "Building pixel grid...")
    fp = build_footprint(wb_id, polygon_3577, pad_m=pad_m)
    if len(fp.pixels) == 0:
        raise RuntimeError("The waterbody polygon does not touch any pixels.")

    report(0.02, "Reading background (long-term) predictors...")
    bg = bgmod.read_background(background_path, background_bands, fp, table_path=background_table)
    warnings = bgmod.background_sanity(bg)
    bg = bg[[c for c in predictors if c in bg.columns]]

    pixels = pd.concat([fp.pixels, bg], axis=1)
    pixels.insert(0, "Unique_ID", str(wb_id))
    pixels.to_parquet(os.path.join(out_dir, "pixels.parquet"), index=False)
    gpd.GeoDataFrame({"Unique_ID": [str(wb_id)]}, geometry=[polygon_wgs84], crs="EPSG:4326").to_file(
        os.path.join(out_dir, "footprint.geojson"), driver="GeoJSON")
    with open(os.path.join(out_dir, "run_info.json"), "w") as f:
        json.dump(dict(Unique_ID=str(wb_id), source=source.name, model_file=model_file, thresholds=thresholds,
                       background_raster=background_path, background_table=background_table,
                       background_band_names=background_bands,
                       pad_m=pad_m, last_run=datetime.now().isoformat(timespec="seconds"),
                       last_date_range=[start.isoformat(), end.isoformat()], min_clear=min_clear,
                       include_ls7_slc_off=include_ls7_slc_off), f, indent=2)

    if bg.isna().all(axis=1).all():
        return dict(out_dir=out_dir, n_pixels=len(fp.pixels), scenes_found=0, processed=0, warnings=warnings,
                    seconds=time.time() - t0, stopped=False)

    report(0.04, f"Searching {source.name} for scenes...")
    scenes = source.search(fp, polygon_wgs84, start, end, include_ls7_slc_off=include_ls7_slc_off)
    man = _load_manifest(out_dir)
    done = set(man.loc[man["status"] == "processed", "scene_key"].astype(str))
    known_clear = dict(zip(man["scene_key"].astype(str), man["clear_fraction"]))
    todo = [s for s in scenes if s.key not in done]
    cov = fp.pixels["coverage_fraction"].to_numpy()
    cov_sum = cov.sum()

    # ---- pass 1: clear fraction for scenes we haven't evaluated -------------
    need_mask = [s for s in todo if not np.isfinite(known_clear.get(s.key, np.nan))]
    masks: dict[str, np.ndarray] = {}
    new_rows = []
    if need_mask:
        res = run_parallel(lambda s: source.clear_mask(s, fp), need_mask, max_workers,
                           progress=lambda d, n, m: report(0.05 + 0.35 * d / n, f"Checking cloud cover: scene {d}/{n}"),
                           should_stop=should_stop)
        for s in need_mask:
            r = res.get(s.key)
            if isinstance(r, Exception) or r is None:
                new_rows.append(_scene_row(s, np.nan, "failed", str(r) if r is not None else "not run"))
                continue
            masks[s.key] = r
            known_clear[s.key] = float((cov * r).sum() / cov_sum) if cov_sum > 0 else 0.0

    candidates = []
    for s in todo:
        cf = known_clear.get(s.key, np.nan)
        if not np.isfinite(cf):
            continue
        if cf <= 0:
            new_rows.append(_scene_row(s, cf, "no_clear_pixels"))
        elif cf < min_clear:
            new_rows.append(_scene_row(s, cf, "skipped_cloudy"))
        else:
            candidates.append(s)
    if new_rows:
        man = pd.concat([man, pd.DataFrame(new_rows)], ignore_index=True)
        _save_manifest(out_dir, man)

    stopped = bool(should_stop and should_stop())
    processed = 0
    # ---- pass 2: read bands, predict, save in chunks -------------------------
    for ci in range(0, len(candidates), CHUNK_SCENES):
        if should_stop and should_stop():
            stopped = True
            break
        chunk = candidates[ci:ci + CHUNK_SCENES]

        def read_one(s: Scene):
            clear = masks.get(s.key)
            if clear is None:
                clear = source.clear_mask(s, fp)
            return clear, source.read_tci(s, fp, clear)

        base = 0.40 + 0.58 * ci / max(1, len(candidates))
        span = 0.58 * len(chunk) / max(1, len(candidates))
        res = run_parallel(read_one, chunk, max_workers,
                           progress=lambda d, n, m: report(base + span * 0.85 * d / n,
                                                           f"Reading scenes {ci + d}/{len(candidates)}"),
                           should_stop=should_stop)
        ok, rows = [], []
        for s in chunk:
            r = res.get(s.key)
            if isinstance(r, Exception) or r is None:
                rows.append(_scene_row(s, known_clear.get(s.key), "failed", str(r) if r is not None else "not run"))
                continue
            ok.append((s, r[1]))
            rows.append(_scene_row(s, known_clear.get(s.key), "processed"))
        if ok:
            report(base + span * 0.9, f"Running models on {len(ok)} scenes x {len(fp.pixels)} pixels...")
            df = predict_chunk(ok, fp, bg, forests, predictors, thresholds, lookup, wb_id)
            df.to_parquet(os.path.join(out_dir, "obs", f"part-{int(time.time() * 1000)}.parquet"), index=False)
            processed += len(ok)
        man = pd.concat([man, pd.DataFrame(rows)], ignore_index=True)
        _save_manifest(out_dir, man)
        masks = {k: v for k, v in masks.items() if k not in {s.key for s in chunk}}

    report(1.0, "Done")
    return dict(out_dir=out_dir, n_pixels=len(fp.pixels), scenes_found=len(scenes), already_done=len(done),
                processed=processed, candidates=len(candidates), warnings=warnings,
                seconds=time.time() - t0, stopped=stopped)


# --------------------------------------------------------------------------
# Loading results
# --------------------------------------------------------------------------

def load_results(out_dir: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
    """Returns (observations, pixels, scenes) or None if nothing saved yet.
    Observations get the scene date/time and per-scene no-data stats the
    concat script added (number_of_pixels, n_blank, proportion_no_data)."""
    pix_p = os.path.join(out_dir, "pixels.parquet")
    obs_dir = os.path.join(out_dir, "obs")
    if not os.path.exists(pix_p) or not os.path.isdir(obs_dir):
        return None
    parts = [os.path.join(obs_dir, f) for f in os.listdir(obs_dir) if f.endswith(".parquet")]
    pixels = pd.read_parquet(pix_p)
    scenes = _load_manifest(out_dir)
    if not parts:
        return pd.DataFrame(), pixels, scenes
    obs = pd.concat([pd.read_parquet(p) for p in sorted(parts)], ignore_index=True)
    obs = obs.drop_duplicates(["scene_key", "pixel_id"], keep="last")
    sc = scenes[scenes["status"] == "processed"][["scene_key", "date", "time", "datetime_utc", "solar_date", "platform",
                                                   "clear_fraction"]].drop_duplicates("scene_key", keep="last")
    obs = obs.merge(sc, on="scene_key", how="inner")
    obs["date"] = pd.to_datetime(obs["date"])
    obs = obs.merge(pixels[["pixel_id", "coverage_fraction", "lng", "lat", "x_albers", "y_albers"]], on="pixel_id", how="left")
    blank = obs["cover_description"] == NO_DATA
    g = obs.assign(_b=blank).groupby("scene_key")["_b"]
    obs["number_of_pixels"] = g.transform("size")
    obs["n_blank"] = g.transform("sum").astype(int)
    obs["proportion_no_data"] = obs["n_blank"] / obs["number_of_pixels"]
    return obs, pixels, scenes


def scene_cover_areas(obs: pd.DataFrame) -> pd.DataFrame:
    """Area (ha) and share of the footprint of each cover description per scene."""
    if obs.empty:
        return pd.DataFrame()
    a = obs.assign(area_ha=obs["coverage_fraction"] * PIXEL_AREA_M2 / 10_000)
    out = (a.groupby(["scene_key", "date", "time", "platform", "proportion_no_data", "cover_description"], as_index=False)
           .agg(area_ha=("area_ha", "sum"), n_pixels=("pixel_id", "size")))
    tot = out.groupby("scene_key")["area_ha"].transform("sum")
    out["share_of_footprint"] = out["area_ha"] / tot
    clear_tot = out[out["cover_description"] != NO_DATA].groupby("scene_key")["area_ha"].sum()
    out["share_of_clear_area"] = np.where(out["cover_description"] == NO_DATA, np.nan,
                                          out["area_ha"] / out["scene_key"].map(clear_tot))
    return out.sort_values(["date", "cover_description"])


def pixel_frequencies(obs: pd.DataFrame) -> pd.DataFrame:
    """Per pixel: number of clear observations, share of them in each cover
    description, share in which each single model said 'present', and the
    most frequent cover description."""
    valid = obs[obs["cover_description"] != NO_DATA]
    if valid.empty:
        return pd.DataFrame()
    n = valid.groupby("pixel_id").size().rename("n_clear_obs")
    desc = (valid.groupby(["pixel_id", "cover_description"]).size().unstack(fill_value=0))
    desc_freq = desc.div(n, axis=0)
    desc_freq.columns = [f"freq: {c}" for c in desc_freq.columns]
    model_freq = valid.groupby("pixel_id")[[f"{m}_binary" for m in MIX_ORDER]].mean()
    model_freq.columns = [f"presence: {c.replace('_binary', '')}" for c in model_freq.columns]
    dominant = desc.idxmax(axis=1).rename("dominant_cover")
    return pd.concat([n, dominant, desc_freq, model_freq], axis=1).reset_index()


def _slug(desc: str) -> str:
    import re
    return re.sub(r"[^a-z0-9]+", "_", desc.lower()).strip("_")


def scene_area_table(obs: pd.DataFrame, wb_id, polygon_area_ha: float, n_pixels: int,
                     footprint_area_ha: float) -> pd.DataFrame:
    """One row per scene: area (ha) of every cover description plus the
    footprint totals needed to turn areas into proportions.

    Areas are coverage-weighted (each pixel counts coverage_fraction x 0.09 ha),
    so the cover-type columns plus no_data_area_ha add up to footprint_area_ha
    (= polygon area, up to rounding at the pixel edges).
    """
    from .classify import COVER_ORDER

    cover_cols = [f"{_slug(c)}_ha" for c in COVER_ORDER if c != NO_DATA]
    if obs.empty:
        return pd.DataFrame()
    a = obs.assign(_area=obs["coverage_fraction"] * PIXEL_AREA_M2 / 10_000,
                   _clear=obs["cover_description"] != NO_DATA)
    wide = a.pivot_table(index="scene_key", columns="cover_description", values="_area", aggfunc="sum", fill_value=0.0)
    wide.columns = [f"{_slug(c)}_ha" if c != NO_DATA else "no_data_area_ha" for c in wide.columns]
    for c in cover_cols + ["no_data_area_ha"]:
        if c not in wide.columns:
            wide[c] = 0.0
    info = a.groupby("scene_key").agg(
        date=("date", "first"), time=("time", "first"), solar_date=("solar_date", "first"),
        platform=("platform", "first"), fmask_clear_fraction=("clear_fraction", "first"),
        n_clear_pixels=("_clear", "sum"), proportion_no_data=("proportion_no_data", "first"))
    out = info.join(wide)
    out["clear_area_ha"] = out[cover_cols].sum(axis=1)
    out.insert(0, "Unique_ID", str(wb_id))
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    out["n_pixels"] = n_pixels
    out["polygon_area_ha"] = polygon_area_ha
    out["pixel_area_ha"] = n_pixels * PIXEL_AREA_M2 / 10_000
    out["footprint_area_ha"] = footprint_area_ha
    out["clear_share_of_footprint"] = out["clear_area_ha"] / footprint_area_ha
    lead = ["Unique_ID", "date", "time", "solar_date", "platform", "n_pixels", "n_clear_pixels",
            "proportion_no_data", "fmask_clear_fraction", "polygon_area_ha", "pixel_area_ha", "footprint_area_ha",
            "clear_area_ha", "no_data_area_ha", "clear_share_of_footprint"]
    return out.reset_index()[lead + cover_cols + ["scene_key"]].sort_values("date").reset_index(drop=True)
