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
    res = pipeline.load_results(out_dir_results)
    if res is None or res[0].empty:
        return pd.DataFrame()
    obs, pixels, _ = res
    keep = obs["clear_fraction"].fillna(0) >= min_clear - 1e-9
    if start is not None:
        keep &= obs["date"] >= pd.Timestamp(start)
    if end is not None:
        keep &= obs["date"] <= pd.Timestamp(end)
    obs = obs[keep]
    if obs.empty:
        return pd.DataFrame()
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
            parts.append(pd.read_csv(p, dtype={"Unique_ID": str}))
    if not parts:
        return None
    out = os.path.join(out_folder, "all_waterbodies_cover_area_by_scene.csv")
    pd.concat(parts, ignore_index=True).to_csv(out, index=False)
    return out
