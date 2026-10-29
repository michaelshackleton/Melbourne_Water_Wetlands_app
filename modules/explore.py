"""
Loading already-processed results for the "Explore saved results" tab -
nothing is downloaded. Three kinds of input are understood:

  * waterbody result folders (results/<source>/<Unique_ID>/ with pixels.parquet,
    obs/ and scenes.csv) - point at the results folder, the dea/ folder, one
    waterbody's folder, or its pixels.parquet;
  * batch "pixel-by-scene" parquet files (<Unique_ID>_pixel_by_scene.parquet);
  * batch cover-area tables (<Unique_ID>_cover_area_by_scene.csv or
    all_waterbodies_cover_area_by_scene.csv) - time series only, no pixel maps.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

from . import pipeline
from .classify import COVER_ORDER, MIX_ORDER, NO_DATA
from .footprint import PIXEL_AREA_M2

PIXEL_SUFFIX = "_pixel_by_scene.parquet"
TABLE_SUFFIX = "_cover_area_by_scene.csv"
COMBINED_TABLE = "all_waterbodies" + TABLE_SUFFIX


def _is_wb_dir(d: str) -> bool:
    return os.path.isfile(os.path.join(d, "pixels.parquet")) and os.path.isdir(os.path.join(d, "obs"))


def find_pixel_sources(path: str) -> dict[str, tuple[str, str]]:
    """{Unique_ID: ("dir", folder) or ("pq", file)} for everything with pixel data under `path`."""
    path = (path or "").strip().strip('"')
    out: dict[str, tuple[str, str]] = {}
    if not path or not os.path.exists(path):
        return out
    if os.path.isfile(path):
        name = os.path.basename(path)
        if name == "pixels.parquet" and _is_wb_dir(os.path.dirname(path)):
            d = os.path.dirname(path)
            out[os.path.basename(d)] = ("dir", d)
        elif name.endswith(PIXEL_SUFFIX):
            out[name[: -len(PIXEL_SUFFIX)]] = ("pq", path)
        return out
    if _is_wb_dir(path):
        return {os.path.basename(os.path.normpath(path)): ("dir", path)}
    # results root (dea / local_tci inside), a source folder, or a batch output folder
    roots = [path] + [os.path.join(path, s) for s in ("dea", "local_tci", "per_waterbody")
                      if os.path.isdir(os.path.join(path, s))]
    for root in roots:
        for d in sorted(os.listdir(root)):
            full = os.path.join(root, d)
            if os.path.isdir(full) and _is_wb_dir(full):
                out.setdefault(d, ("dir", full))
        for f in sorted(glob.glob(os.path.join(root, "*" + PIXEL_SUFFIX))):
            out.setdefault(os.path.basename(f)[: -len(PIXEL_SUFFIX)], ("pq", f))
    return out


def find_table_sources(path: str) -> dict[str, str]:
    """{Unique_ID or '*': csv file} for batch cover-area tables under `path`."""
    path = (path or "").strip().strip('"')
    out: dict[str, str] = {}
    if not path or not os.path.exists(path):
        return out
    if os.path.isfile(path):
        if path.lower().endswith(".csv"):
            name = os.path.basename(path)
            out["*" if name == COMBINED_TABLE or not name.endswith(TABLE_SUFFIX) else name[: -len(TABLE_SUFFIX)]] = path
        return out
    per = os.path.join(path, "per_waterbody")
    for root in [path, per] if os.path.isdir(per) else [path]:
        for f in sorted(glob.glob(os.path.join(root, "*" + TABLE_SUFFIX))):
            name = os.path.basename(f)
            if name != COMBINED_TABLE:
                out.setdefault(name[: -len(TABLE_SUFFIX)], f)
    if not out and os.path.isfile(os.path.join(path, COMBINED_TABLE)):
        out["*"] = os.path.join(path, COMBINED_TABLE)
    return out


def load_pixel_source(kind: str, path: str) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """(obs, pixels) for one waterbody, in the same form as the single-waterbody tab."""
    if kind == "dir":
        res = pipeline.load_results(path)
        if res is None or res[0].empty:
            return None
        return res[0], res[1]
    obs = pd.read_parquet(path)
    if obs.empty:
        return None
    obs["date"] = pd.to_datetime(obs["date"])
    if "Unique_ID" in obs:
        obs["Unique_ID"] = obs["Unique_ID"].astype(str)
    if "proportion_no_data" not in obs:
        blank = obs["cover_description"] == NO_DATA
        g = obs.assign(_b=blank).groupby("scene_key")["_b"]
        obs["number_of_pixels"] = g.transform("size")
        obs["n_blank"] = g.transform("sum").astype(int)
        obs["proportion_no_data"] = obs["n_blank"] / obs["number_of_pixels"]
    pix_cols = [c for c in ["pixel_id", "coverage_fraction", "lng", "lat", "x_albers", "y_albers"] if c in obs]
    pixels = obs[pix_cols].drop_duplicates("pixel_id").sort_values("pixel_id").reset_index(drop=True)
    return obs, pixels


def table_to_areas(df: pd.DataFrame) -> pd.DataFrame:
    """Batch per-scene wide table -> long per-scene cover areas (the form
    pipeline.scene_cover_areas() returns)."""
    if df.empty:
        return pd.DataFrame()
    slug = {pipeline._slug(c) + "_ha": c for c in COVER_ORDER if c != NO_DATA}
    slug["no_data_area_ha"] = NO_DATA
    cols = [c for c in slug if c in df.columns]
    d = df.copy()
    d["Unique_ID"] = d["Unique_ID"].astype(str)
    if "scene_key" not in d:
        d["scene_key"] = d["date"].astype(str) + "_" + d.get("platform", "").astype(str)
    d["date"] = pd.to_datetime(d["date"])
    d["acquired"] = d["date"] + pd.to_timedelta(d["time"].astype(str))
    keep = ["Unique_ID", "scene_key", "date", "time", "acquired", "platform", "proportion_no_data"]
    long = d[[c for c in keep if c in d] + cols].melt(id_vars=[c for c in keep if c in d], value_vars=cols,
                                                      var_name="col", value_name="area_ha")
    long["cover_description"] = long["col"].map(slug)
    long = long.drop(columns="col")
    clear = long[long["cover_description"] != NO_DATA].groupby(["Unique_ID", "scene_key"])["area_ha"].transform("sum")
    long["share_of_clear_area"] = np.nan
    m = long["cover_description"] != NO_DATA
    long.loc[m, "share_of_clear_area"] = long.loc[m, "area_ha"] / clear.replace(0, np.nan)
    return long


def obs_to_areas(obs: pd.DataFrame, wb_id: str) -> pd.DataFrame:
    a = pipeline.scene_cover_areas(obs)
    if a.empty:
        return a
    a.insert(0, "Unique_ID", str(wb_id))
    return a


def period_table(areas: pd.DataFrame, agg: str) -> pd.DataFrame:
    """Per waterbody and time step: area (ha) of each cover type and of the
    clear area. 'Each scene' keeps scenes; monthly/annual are the mean of the
    scenes in that period (as in the single-waterbody tab)."""
    d = areas.copy()
    wide = d.pivot_table(index=["Unique_ID", "scene_key", "acquired"], columns="cover_description",
                         values="area_ha", aggfunc="sum", fill_value=0.0)
    covers = [c for c in wide.columns if c != NO_DATA]
    wide["_clear"] = wide[covers].sum(axis=1)
    for c in covers:
        wide[f"_share::{c}"] = np.where(wide["_clear"] > 0, wide[c] / wide["_clear"].where(wide["_clear"] > 0), np.nan)
    wide = wide.reset_index()
    if agg != "Each scene":
        freq = "MS" if agg == "Monthly mean" else "YS"
        wide = (wide.drop(columns="scene_key").groupby(["Unique_ID", pd.Grouper(key="acquired", freq=freq)])
                .mean(numeric_only=True).dropna(how="all").reset_index())
        wide["scene_key"] = wide["acquired"].astype(str)
    rows = []
    for c in covers + ([NO_DATA] if NO_DATA in wide.columns else []):
        part = wide[["Unique_ID", "scene_key", "acquired", "_clear"]].copy()
        part["cover"] = c
        part["area_ha"] = wide[c]
        part["share"] = wide[f"_share::{c}"] if c != NO_DATA else np.nan
        rows.append(part)
    return pd.concat(rows, ignore_index=True).rename(columns={"_clear": "clear_ha"})


def combine_waterbodies(pt: pd.DataFrame) -> pd.DataFrame:
    """Sum of areas across waterbodies per time step; share = total cover area /
    total clear area (so bigger waterbodies weigh more)."""
    g = pt.groupby(["acquired", "cover"], as_index=False).agg(area_ha=("area_ha", "sum"))
    clear = pt.drop_duplicates(["Unique_ID", "acquired"]).groupby("acquired")["clear_ha"].sum()
    g["clear_ha"] = g["acquired"].map(clear)
    g["share"] = np.where(g["cover"] == NO_DATA, np.nan, g["area_ha"] / g["clear_ha"].replace(0, np.nan))
    g["Unique_ID"] = "All selected"
    g["scene_key"] = g["acquired"].astype(str)
    return g


def yearly_pixel_frequencies(obs: pd.DataFrame) -> pd.DataFrame:
    """Per year and pixel: clear observations, most frequent cover type and the
    share of clear observations in each cover type / with each model present."""
    o = obs[obs["cover_description"] != NO_DATA].copy()
    if o.empty:
        return pd.DataFrame()
    o["year"] = pd.to_datetime(o["date"]).dt.year
    n = o.groupby(["year", "pixel_id"]).size().rename("n_clear_obs")
    desc = o.groupby(["year", "pixel_id", "cover_description"]).size().unstack(fill_value=0)
    freq = desc.div(n, axis=0)
    freq.columns = [f"freq: {c}" for c in freq.columns]
    pres = o.groupby(["year", "pixel_id"])[[f"{m}_binary" for m in MIX_ORDER]].mean()
    pres.columns = [f"presence: {c.replace('_binary', '')}" for c in pres.columns]
    dom = desc.idxmax(axis=1).rename("dominant_cover")
    return pd.concat([n, dom, freq, pres], axis=1).reset_index()


def pixel_area_ha(pixels: pd.DataFrame) -> float:
    return float(pixels["coverage_fraction"].sum()) * PIXEL_AREA_M2 / 10_000
