"""
Waterbody Cover Predictor
=========================

Pick a waterbody, and the app fetches the Landsat archive for its footprint
from Digital Earth Australia, runs the seven binomial cover models on every
pixel of every clear scene, and shows:

* cover-type area over time;
* how often each pixel was each cover type (average presence across time);
* downloadable tables (per scene, per pixel, and the full pixel-by-scene table).

Run with run_app.bat, or:  streamlit run app.py
"""
from __future__ import annotations

import io
import json
import os
import time
from datetime import date

import altair as alt
import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import streamlit as st
from branca.colormap import LinearColormap
from streamlit_folium import st_folium

from modules import batch as batchmod
from modules import charts, explore
from modules import pipeline
from modules.classify import COVER_COLOURS, COVER_ORDER, MIX_ORDER, MODEL_DISPLAY, NO_DATA, NO_PREDICTION, CoverLookup, load_thresholds
from modules.footprint import build_footprint, pixel_polygons_wgs84
from modules.rf_engine import engine_name, load_forests
from modules.sources import DEASource, LocalTCISource
from modules.waterbodies import WaterbodyLayer

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SETTINGS = os.path.join(APP_DIR, "config", "settings.json")
USER_SETTINGS = os.path.join(APP_DIR, "config", "user_settings.json")
MODEL_NPZ = os.path.join(APP_DIR, "models", "wetland_cover_models.npz")
THRESHOLDS = os.path.join(APP_DIR, "config", "thresholds.csv")
COMBOS = os.path.join(APP_DIR, "config", "cover_combinations.csv")
MELBOURNE = (-37.95, 145.0)


st.set_page_config(page_title="Waterbody Cover Predictor", layout="wide")

# Streamlit >= 1.46 replaced use_container_width=True with width="stretch"
_v = tuple(int(x) for x in st.__version__.split(".")[:2])
FULL = {"width": "stretch"} if _v >= (1, 46) else {"use_container_width": True}
CONTENT = {"width": "content"} if _v >= (1, 46) else {"use_container_width": False}
alt.data_transformers.disable_max_rows()


# --------------------------------------------------------------------------
# Settings & cached resources
# --------------------------------------------------------------------------

def load_settings() -> dict:
    with open(DEFAULT_SETTINGS) as f:
        s = json.load(f)
    if os.path.exists(USER_SETTINGS):
        try:
            with open(USER_SETTINGS) as f:
                s.update(json.load(f))
        except Exception:  # noqa: BLE001
            pass
    return s


def save_user_settings(values: dict):
    with open(USER_SETTINGS, "w") as f:
        json.dump(values, f, indent=2)


def abs_path(p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(APP_DIR, p)


@st.cache_resource(show_spinner="Loading waterbodies layer...")
def get_layer(path, id_field, label_fields) -> WaterbodyLayer:
    return WaterbodyLayer(path, id_field, list(label_fields))


@st.cache_resource(show_spinner="Loading models...")
def get_models():
    forests, predictors = load_forests(MODEL_NPZ)
    return forests, predictors, load_thresholds(THRESHOLDS), CoverLookup(COMBOS)


@st.cache_data(show_spinner="Loading results...")
def get_results(out_dir: str, stamp: float):
    res = pipeline.load_results(out_dir)
    if res is None:
        return None
    obs, pixels, scenes = res
    return obs, pixels, scenes


def results_stamp(out_dir: str) -> float:
    p = os.path.join(out_dir, "scenes.csv")
    obs_dir = os.path.join(out_dir, "obs")
    t = os.path.getmtime(p) if os.path.exists(p) else 0
    if os.path.isdir(obs_dir):
        t = max([t] + [os.path.getmtime(os.path.join(obs_dir, f)) for f in os.listdir(obs_dir)])
    return t


def pick_path(kind: str, initial: str | None) -> str | None:
    """Native file/folder dialog (works because Streamlit runs locally)."""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        return None
    root = tk.Tk()
    root.withdraw()
    root.wm_attributes("-topmost", 1)
    init = initial if initial and os.path.isdir(initial) else (os.path.dirname(initial) if initial else APP_DIR)
    if kind == "folder":
        out = filedialog.askdirectory(initialdir=init)
    elif kind == "shp":
        out = filedialog.askopenfilename(initialdir=init, filetypes=[("Waterbodies layer", "*.parquet *.shp *.gpkg"), ("All", "*.*")])
    elif kind == "anyfile":
        out = filedialog.askopenfilename(initialdir=init, filetypes=[("Results files", "*.parquet *.csv"), ("All", "*.*")])
    elif kind == "file":
        out = filedialog.askopenfilename(initialdir=init, filetypes=[("Parquet", "*.parquet"), ("All", "*.*")])
    else:
        out = filedialog.askopenfilename(initialdir=init, filetypes=[("GeoTIFF", "*.tif *.tiff"), ("All", "*.*")])
    root.destroy()
    return out or None


def _browse(state_key: str, kind: str):
    chosen = pick_path(kind, st.session_state.get(state_key))
    if chosen:
        st.session_state[state_key] = chosen


settings = load_settings()
for k, v in {
    "selected_id": None, "map_version": 0, "last_click": None, "run_message": None,
    "set_waterbodies_path": settings["waterbodies_path"],
    "set_background_raster": settings["background_raster"],
    "set_background_table": settings.get("background_table", "data/background_pixels.parquet"),
    "set_local_tci_folder": settings["local_tci_folder"],
    "set_results_dir": settings["results_dir"],
    "set_band_names": ", ".join(settings["background_band_names"]),
    "batch_out_dir": "",
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------

st.sidebar.title("Waterbody Cover Predictor")
st.sidebar.caption("Landsat (DEA) → tasselled cap → 7 random-forest cover models → cover type per 30 m pixel.")

source_choice = st.sidebar.radio("Satellite data", ["DEA online (Landsat 5/7/8/9)", "Local TCI rasters (offline)"],
                                 help="DEA online reads only the footprint's pixels from Digital Earth Australia's public "
                                      "archive. The local option re-uses the existing 'TCI Melb region' rasters.")
c1, c2 = st.sidebar.columns(2)
start_date = c1.date_input("From", value=date(1987, 1, 1), min_value=date(1986, 1, 1), max_value=date.today())
end_date = c2.date_input("To", value=date.today(), min_value=date(1986, 1, 1), max_value=date.today())
min_clear = st.sidebar.slider(
    "Skip scenes with less than this much of the footprint clear", 0.0, 1.0, 0.5, 0.05,
    help="Share of the footprint (area-weighted) not covered by cloud, cloud shadow or missing data. "
         "Skipping cloudy scenes saves download time; you can lower it later and re-run - "
         "only the newly-eligible scenes are fetched.")
include_slc_off = st.sidebar.checkbox("Include Landsat 7 after May 2003 (striped 'SLC-off' scenes)", value=True,
                                      help="Striped pixels are treated as missing data, like cloud.")

with st.sidebar.expander("Paths & advanced settings"):
    st.caption("Saved to config/user_settings.json when you press Save.")
    for key, label, kind in [("set_waterbodies_path", "Waterbodies layer", "shp"),
                             ("set_background_table", "Background pixel table (ships with the app)", "file"),
                             ("set_background_raster", "Background raster (optional - used for pixels not in the table)", "tif"),
                             ("set_local_tci_folder", "Local TCI rasters folder", "folder"),
                             ("set_results_dir", "Results folder", "folder")]:
        a, b = st.columns([4, 1])
        a.text_input(label, key=key)
        b.button("…", key=f"browse_{key}", on_click=_browse, args=(key, kind), help="Browse")
    st.text_input("Background band names (in band order)", key="set_band_names",
                  help="Names given to the background raster's 9 bands, in order. The default is what the "
                       "original R script assigned.")
    max_workers = st.slider("Parallel reads", 1, 32, int(settings.get("max_workers", 16)),
                            help="How many scenes are read from DEA at once.")
    pad_m = st.number_input("Padding around waterbody (m)", 0, 300, int(settings.get("pad_m", 60)), 30)
    if st.button("Save settings"):
        save_user_settings(dict(
            waterbodies_path=st.session_state.set_waterbodies_path,
            background_raster=st.session_state.set_background_raster,
            background_table=st.session_state.set_background_table,
            local_tci_folder=st.session_state.set_local_tci_folder,
            results_dir=st.session_state.set_results_dir,
            background_band_names=[s.strip() for s in st.session_state.set_band_names.split(",") if s.strip()],
            max_workers=max_workers, pad_m=pad_m,
        ))
        st.success("Saved.")
    st.caption(f"Prediction engine: {engine_name()}")

band_names = [s.strip() for s in st.session_state.set_band_names.split(",") if s.strip()]
if not os.path.exists(abs_path(st.session_state.set_background_table)) and \
        not os.path.exists(st.session_state.set_background_raster):
    st.sidebar.warning("No background data found: neither the background pixel table "
                       f"({st.session_state.set_background_table}) nor the background raster is available, so "
                       "predictions can't run yet. Build the table once with scripts/build_background_table.py "
                       "on a computer with the D: drive (see README).")
results_base = abs_path(st.session_state.set_results_dir)

if source_choice.startswith("DEA"):
    source = DEASource(settings["stac_urls"], settings["collections"], settings.get("fmask_good_values", [1, 4, 5]),
                       max_workers=max_workers)
else:
    source = LocalTCISource(st.session_state.set_local_tci_folder, max_workers=min(max_workers, 4))

# --------------------------------------------------------------------------
# Load layer
# --------------------------------------------------------------------------

try:
    layer = get_layer(abs_path(st.session_state.set_waterbodies_path), settings["waterbody_id_field"],
                      tuple(settings.get("waterbody_label_fields", [])))
except Exception as e:  # noqa: BLE001
    st.error(f"Could not load the waterbodies layer: {e}\n\nSet the path under 'Paths & advanced settings' in the sidebar.")
    st.stop()

forests, predictors, thresholds, lookup = get_models()

# --------------------------------------------------------------------------
# Batch processing tab
# --------------------------------------------------------------------------

def render_batch():
    st.subheader("Batch processing")
    st.caption("Upload a list of Unique_IDs. Each waterbody is processed in turn with the settings in the sidebar "
               "(data source, dates, minimum clear share, Landsat 7) and a per-scene table of cover-type areas is "
               "written to the folder you choose. No charts are drawn.")
    up = st.file_uploader("List of Unique_IDs", type=["csv", "txt", "zip", "gpkg", "geojson"],
                          help="CSV with a Unique_ID column (otherwise the first column is used), a text file with "
                               "one ID per line, or a zipped shapefile / GeoPackage / GeoJSON with a Unique_ID field.")
    pasted = st.text_area("…or paste IDs (separated by spaces, commas or new lines)", height=80)
    ids, note = [], ""
    if up is not None:
        try:
            ids, cols, col = batchmod.ids_from_upload(up.name, up.getvalue(), layer.id_field)
            if len(cols) > 1 and col != layer.id_field:
                col = st.selectbox("Column holding the IDs", cols, index=cols.index(col))
                ids, _, _ = batchmod.ids_from_upload(up.name, up.getvalue(), col)
            note = f"column '{col}'"
        except Exception as e:  # noqa: BLE001
            st.error(f"Couldn't read {up.name}: {e}")
    if pasted.strip():
        ids += [v for v in pasted.replace(",", " ").split() if v]
        ids = list(dict.fromkeys(ids))
    if not ids:
        st.info("Add a list of IDs to start.")
        return
    known = set(layer.ids())
    found = [i for i in ids if i in known]
    missing = [i for i in ids if i not in known]
    st.write(f"**{len(ids):,} IDs** {('from ' + note) if note else ''}: {len(found):,} found in the waterbodies layer"
             + (f", {len(missing):,} not found ({', '.join(missing[:10])}{' …' if len(missing) > 10 else ''})" if missing else ""))

    a, b = st.columns([5, 1])
    a.text_input("Output folder", key="batch_out_dir")
    b.button("…", key="browse_batch_out", on_click=_browse, args=("batch_out_dir", "folder"), help="Browse")
    c1, c2 = st.columns(2)
    skip_written = c1.checkbox("Skip waterbodies already written to this folder", value=True,
                               help="Lets you stop a batch and carry on later.")
    pixel_tables = c2.checkbox("Also write the full pixel-by-scene table for each waterbody (parquet)", value=False)
    g1, g2 = st.columns(2)
    grouped = g1.checkbox("Process nearby waterbodies together (much faster)", value=True,
                          help="Waterbodies in the same grid cell share one download per scene. Results are "
                               "identical to processing them one at a time. Each group is completed for the whole "
                               "date range before its tables are written.")
    cell_km = g2.number_input("Group size (grid cell, km)", 2.0, 50.0, 10.0, 1.0, disabled=not grouped,
                              help="Larger cells mean fewer, bigger groups. 10 km suits Landsat's image tiles.")
    todo_ids = [i for i in found if not (skip_written and os.path.exists(os.path.join(
        st.session_state.batch_out_dir.strip() or ".", "per_waterbody", f"{i}_cover_area_by_scene.csv")))]
    if grouped and todo_ids:
        from modules.group import make_groups
        groups = make_groups(layer.gdf_albers, layer.id_field, todo_ids, cell_km)
        st.info(f"{len(todo_ids):,} waterbodies to process in **{len(groups):,} groups** "
                f"(largest group: {max(len(g) for g in groups)}). As a rough guide, each group takes about as long "
                f"as one waterbody on its own (~5-10 min for 20 years), so roughly "
                + (f"{len(groups) * 5 / 60:.0f}-{len(groups) * 10 / 60:.0f} hours." if len(groups) * 10 >= 120
                   else f"{len(groups) * 5:.0f}-{len(groups) * 10:.0f} minutes."))
    elif todo_ids:
        st.info(f"{len(todo_ids):,} waterbodies to process one at a time (~5-10 min each for 20 years).")
    st.caption(f"Source: {source.name} · {start_date} to {end_date} · skip scenes under {min_clear:.0%} clear · "
               f"Landsat 7 SLC-off {'included' if include_slc_off else 'excluded'}. Change these in the sidebar.")
    st.warning("While a batch is running, don't change any other setting in the app - that restarts the page and "
               "stops the batch. Use the Stop button (top right) to stop it; finished waterbodies are kept.",
               icon="⚠️")
    if not st.button("Start batch", type="primary", disabled=not found):
        return
    out_folder = st.session_state.batch_out_dir.strip()
    if not out_folder:
        st.error("Choose an output folder first.")
        return
    os.makedirs(out_folder, exist_ok=True)
    log_path = os.path.join(out_folder, "batch_log.csv")
    log = pd.read_csv(log_path, dtype={"Unique_ID": str}) if os.path.exists(log_path) else pd.DataFrame()
    overall = st.progress(0.0, text="Starting batch...")
    current = st.progress(0.0, text="")
    status_box = st.empty()
    if grouped:
        if not todo_ids:
            st.success("Everything in the list has already been written to this folder.")
            return
        batch_t0 = time.time()
        shown = []
        group_minutes = []

        def g_start(gi, ng, ids, done_n, total):
            eta = ""
            if group_minutes:
                left = (ng - gi + 1) * float(np.mean(group_minutes))
                eta = f" · about {left / 60:.1f} h left" if left >= 90 else f" · about {left:.0f} min left"
            overall.progress(done_n / total, text=f"Group {gi} of {ng} ({len(ids)} waterbodies) · "
                                                  f"{done_n:,} of {total:,} waterbodies done · "
                                                  f"{(time.time() - batch_t0) / 60:.1f} min elapsed{eta}")

        def g_done(gi, ng, recs, wall, done_n, total):
            group_minutes.append(wall / 60)
            shown.extend(recs)
            df = pd.DataFrame(shown)
            status_box.dataframe(df[[c for c in batchmod.LOG_COLUMNS if c in df.columns]], **FULL, height=300)

        proc_kwargs = dict(results_base=results_base, background_path=st.session_state.set_background_raster,
                           background_table=abs_path(st.session_state.set_background_table),
                           background_bands=band_names, forests=forests, predictors=predictors,
                           thresholds=thresholds, lookup=lookup, min_clear=min_clear,
                           include_ls7_slc_off=include_slc_off, pad_m=pad_m, max_workers=max_workers,
                           model_file=os.path.basename(MODEL_NPZ))
        log = batchmod.run_grouped(layer, groups, source, start_date, end_date, out_folder, pixel_tables,
                                   proc_kwargs, log, log_path, found, on_group_start=g_start,
                                   on_progress=lambda f, msg: current.progress(f, text=msg), on_group_done=g_done)
        overall.progress(1.0, text=f"Batch finished: {len(todo_ids):,} waterbodies in {len(groups)} groups, "
                                   f"{(time.time() - batch_t0) / 60:.1f} min")
        n_inc = sum(1 for r in shown if str(r.get("status", "")).startswith(("incomplete", "failed")))
        combined = batchmod.combine(out_folder, found)
        if n_inc:
            st.warning(f"{n_inc} waterbodies are incomplete or failed (see batch_log.csv) - run the batch again "
                       "to retry just the missing scenes.")
        st.success(f"Finished. Combined table: {combined}" if combined else "Finished, but no table was written.")
        return
    rows = []
    batch_t0 = time.time()
    done_minutes = []  # wall time of waterbodies actually processed (not skipped), for the estimate
    for n, wb in enumerate(found, 1):
        elapsed = (time.time() - batch_t0) / 60
        eta = ""
        if done_minutes:
            left = (len(found) - n + 1) * float(np.mean(done_minutes))
            eta = f" · about {left:.0f} min left" if left >= 1 else " · under a minute left"
        overall.progress((n - 1) / len(found),
                         text=f"Waterbody {n} of {len(found)}: {wb} · {elapsed:.1f} min elapsed{eta}")
        per_file = os.path.join(out_folder, "per_waterbody", f"{wb}_cover_area_by_scene.csv")
        t_wb = time.time()
        rec = dict(Unique_ID=wb, started=pd.Timestamp.now().isoformat(timespec="seconds"))
        if skip_written and os.path.exists(per_file):
            rec.update(status="skipped (already written)")
        else:
            info = layer.get(wb)
            try:
                res = pipeline.process_waterbody(
                    wb, info["albers"], info["wgs"], source, start_date, end_date,
                    results_base=results_base, background_path=st.session_state.set_background_raster,
                    background_table=abs_path(st.session_state.set_background_table),
                    background_bands=band_names, forests=forests, predictors=predictors, thresholds=thresholds,
                    lookup=lookup, min_clear=min_clear, include_ls7_slc_off=include_slc_off, pad_m=pad_m,
                    max_workers=max_workers, progress=lambda f, msg: current.progress(f, text=f"{wb}: {msg}"),
                    model_file=os.path.basename(MODEL_NPZ))
                t_write = time.time()
                tbl = batchmod.write_waterbody(res["out_dir"], wb, info["albers"], out_folder, pixel_tables,
                                               start=start_date, end=end_date, min_clear=min_clear)
                wall = time.time() - t_wb
                rec.update(status="done" if len(tbl) else "no scenes passed the filters", scenes_in_table=len(tbl),
                           **batchmod.timing_fields(res, wall, time.time() - t_write),
                           warnings=" ".join(res.get("warnings", [])))
                done_minutes.append(wall / 60)
            except Exception as e:  # noqa: BLE001
                rec.update(status="failed", minutes_total=round((time.time() - t_wb) / 60, 2), warnings=str(e))
        rec["finished"] = pd.Timestamp.now().isoformat(timespec="seconds")
        rows.append(rec)
        log = batchmod.save_log(log_path, log, rec)
        batchmod.combine(out_folder, found)
        show = pd.DataFrame(rows)
        status_box.dataframe(show[[c for c in batchmod.LOG_COLUMNS if c in show.columns]], **FULL, height=260)
    overall.progress(1.0, text=f"Batch finished: {len(found)} waterbodies in "
                               f"{(time.time() - batch_t0) / 60:.1f} min")
    combined = batchmod.combine(out_folder, found)
    st.success(f"Finished. Combined table: {combined}" if combined else "Finished, but no waterbody produced a table.")


# --------------------------------------------------------------------------
# Explore saved results tab
# --------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _load_pixel_source(kind: str, path: str, stamp: float):
    return explore.load_pixel_source(kind, path)


@st.cache_data(show_spinner=False)
def _load_table(path: str, stamp: float):
    return pd.read_csv(path, dtype={"Unique_ID": str})


def _stamp(kind: str, path: str) -> float:
    return results_stamp(path) if kind == "dir" else (os.path.getmtime(path) if os.path.exists(path) else 0)


@st.cache_data(show_spinner=False, max_entries=8)
def _available_geojson(ids: tuple, simplify_m: float) -> tuple[str | None, str, list, int]:
    """(outlines GeoJSON or None when there are too many, points GeoJSON, bounds, count)."""
    g = layer.gdf_wgs[layer.gdf_wgs[layer.id_field].isin(ids)][[layer.id_field, "_label", "geometry"]].copy()
    bounds = list(g.total_bounds)
    pts = g.copy()
    pts["geometry"] = layer.gdf_albers.loc[g.index].geometry.representative_point().to_crs(4326).values
    outlines = None
    if len(g) <= 2500:
        if simplify_m > 0:
            g["geometry"] = layer.gdf_albers.loc[g.index].geometry.simplify(simplify_m).to_crs(4326).values
        outlines = g.to_json()
    return outlines, pts.to_json(), bounds, len(g)


def _explore_map(ids_avail: list, selected: list, labels: dict, path: str):
    outlines, points, b, n = _available_geojson(tuple(ids_avail), 3.0)
    span = max(b[2] - b[0], (b[3] - b[1]) * 1.3, 1e-4)
    zoom = int(np.clip(np.floor(np.log2(360 * 1.6 / span)), 8, 17))
    m = folium.Map(location=((b[1] + b[3]) / 2, (b[0] + b[2]) / 2), zoom_start=zoom, tiles=None, control_scale=True)
    folium.TileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                     attr="Esri, Maxar, Earthstar Geographics").add_to(m)
    tip = folium.GeoJsonTooltip(fields=["_label"], aliases=[""])
    if outlines:
        folium.GeoJson(outlines, name="Outlines",
                       style_function=lambda f: {"color": "#7FDBFF", "weight": 1.5, "fillOpacity": 0.15},
                       tooltip=tip).add_to(m)
    folium.GeoJson(points, name="Waterbodies",
                   marker=folium.CircleMarker(radius=4, color="#0B5394", fill=True, fill_color="#7FDBFF",
                                              fill_opacity=0.9, weight=1),
                   tooltip=folium.GeoJsonTooltip(fields=["_label"], aliases=[""])).add_to(m)
    fg = folium.FeatureGroup(name="Selected")
    sel_g = layer.gdf_wgs[layer.gdf_wgs[layer.id_field].isin(selected)]
    if len(sel_g):
        # highlight only (interactive=False): clicks pass through to the waterbody underneath, so clicking a
        # selected waterbody again removes it
        folium.GeoJson(sel_g[[layer.id_field, "_label", "geometry"]].to_json(),
                       style_function=lambda f: {"color": "#FFD700", "weight": 3, "fillOpacity": 0.35,
                                                 "fillColor": "#FFD700"}, interactive=False).add_to(fg)
        sp = sel_g[[layer.id_field, "_label"]].copy()
        sp["geometry"] = layer.gdf_albers.loc[sel_g.index].geometry.representative_point().to_crs(4326).values
        folium.GeoJson(gpd.GeoDataFrame(sp, crs=4326).to_json(),
                       marker=folium.CircleMarker(radius=6, color="#7F6000", fill=True, fill_color="#FFD700",
                                                  fill_opacity=1, weight=1.5, interactive=False),
                       interactive=False).add_to(fg)
    out = st_folium(m, key=f"ex_map_{abs(hash((path, n))) % 10**8}", height=430, use_container_width=True,
                    feature_group_to_add=fg, returned_objects=["last_object_clicked", "last_clicked"])
    st.caption(f"{n:,} waterbodies (blue dots{'' if outlines else ' - too many to draw outlines'}); "
               "selected ones in yellow. Zoom in to see outlines of small waterbodies. To undo a click, remove it "
               "with × in the list below, or click somewhere else on the map before clicking it again.")
    click = (out or {}).get("last_object_clicked") or (out or {}).get("last_clicked")
    if click and click != st.session_state.get("ex_last_click"):
        st.session_state.ex_last_click = click
        hit = _nearest_available(click["lng"], click["lat"], ids_avail)
        if hit:
            st.session_state.ex_pending_toggle = hit
            st.rerun()


def _nearest_available(lng: float, lat: float, ids_avail: list, max_m: float = 300.0) -> str | None:
    from shapely.geometry import Point
    p = Point(lng, lat)
    g = layer.gdf_wgs
    cand_idx = list(layer._sindex.query(p.buffer(max_m / 90_000)))
    if not cand_idx:
        return None
    cand = g.iloc[cand_idx]
    cand = cand[cand[layer.id_field].isin(set(ids_avail))]
    if cand.empty:
        return None
    inside = cand[cand.geometry.contains(p)]
    if len(inside):
        return inside.iloc[int(np.argmin(layer.gdf_albers.loc[inside.index].area.to_numpy()))][layer.id_field]
    d = cand.geometry.distance(p).to_numpy()
    return cand.iloc[int(np.argmin(d))][layer.id_field]


def render_explore():
    st.subheader("Explore saved results")
    st.caption("Charts from results already on disk - nothing is downloaded. Point to the app's results folder "
               "(or one waterbody's folder / its pixels.parquet, or batch pixel-by-scene parquet files) for full "
               "pixel data, or to batch output tables for the cover-over-time charts only.")
    kind = st.radio("Load", ["Saved pixel results (results folder, pixels.parquet, pixel_by_scene.parquet)",
                             "Batch output tables (cover_area_by_scene CSV)"], key="ex_kind")
    pixel_mode = kind.startswith("Saved")
    default = os.path.join(results_base, "dea") if pixel_mode else st.session_state.get("batch_out_dir", "")
    if st.session_state.get("ex_mode_prev") != pixel_mode:
        st.session_state.ex_path = default
        st.session_state.ex_mode_prev = pixel_mode
    a, b, c = st.columns([5, 1, 1])
    a.text_input("Folder or file", key="ex_path")
    b.button("Folder…", key="ex_browse_dir", on_click=_browse, args=("ex_path", "folder"))
    c.button("File…", key="ex_browse_file", on_click=_browse, args=("ex_path", "anyfile"))
    path = st.session_state.ex_path

    if pixel_mode:
        sources = explore.find_pixel_sources(path)
    else:
        sources = explore.find_table_sources(path)
    if not sources:
        st.info("Nothing found there yet. " + ("Choose the results folder (e.g. …\\waterbodies app\\results\\dea), "
                "a waterbody folder, a pixels.parquet, or a folder of *_pixel_by_scene.parquet files."
                if pixel_mode else "Choose a batch output folder, its per_waterbody folder, or a *_cover_area_by_scene.csv."))
        return

    table = None
    if not pixel_mode and "*" in sources:  # combined table: IDs come from inside it
        table = _load_table(sources["*"], os.path.getmtime(sources["*"]))
        ids_avail = sorted(table["Unique_ID"].astype(str).unique(), key=lambda x: (len(x), x))
    else:
        ids_avail = sorted(sources, key=lambda x: (len(x), x))

    names = (layer.gdf_wgs.set_index(layer.id_field)["WB_Name"].dropna().to_dict()
             if "WB_Name" in layer.gdf_wgs.columns else {})
    labels = {i: (f"{i} | {names[i]}" if names.get(i) else i) for i in ids_avail}
    st.write(f"**{len(ids_avail):,} waterbodies** found.")

    # selection state: keep only IDs available here; apply a pending map click before the widget is drawn
    avail_set = set(ids_avail)
    if "ex_ids" not in st.session_state or not set(st.session_state.ex_ids) <= avail_set:
        st.session_state.ex_ids = [i for i in st.session_state.get("ex_ids", []) if i in avail_set] or ids_avail[:1]
    pending = st.session_state.pop("ex_pending_toggle", None)
    if pending in avail_set:
        cur = list(st.session_state.ex_ids)
        st.session_state.ex_ids = [i for i in cur if i != pending] if pending in cur else cur + [pending]

    with st.expander("Map of available waterbodies - click one to add or remove it", expanded=True):
        _explore_map(ids_avail, list(st.session_state.ex_ids), labels, path)
    sel = st.multiselect("Waterbodies to show", ids_avail, format_func=lambda i: labels.get(i, i), key="ex_ids")
    if not sel:
        return

    # ---- load ----------------------------------------------------------------
    areas_list, obs_by_id, pix_by_id = [], {}, {}
    with st.spinner(f"Loading {len(sel)} waterbodies..."):
        for i in sel:
            if pixel_mode:
                k, p = sources[i]
                res = _load_pixel_source(k, p, _stamp(k, p))
                if res is None:
                    continue
                obs_by_id[i], pix_by_id[i] = res
                areas_list.append(explore.obs_to_areas(res[0], i))
            else:
                t = table[table["Unique_ID"] == i] if table is not None else _load_table(sources[i], os.path.getmtime(sources[i]))
                if len(t):
                    areas_list.append(explore.table_to_areas(t))
    areas_all = pd.concat([x for x in areas_list if len(x)], ignore_index=True) if areas_list else pd.DataFrame()
    if areas_all.empty:
        st.warning("No processed scenes in the selected waterbodies.")
        return

    # ---- filters -------------------------------------------------------------
    f1, f2, f3 = st.columns(3)
    max_nd = f1.slider("Max share of pixels with no data in a scene", 0.0, 1.0, 0.1, 0.05, key="ex_nd")
    dmin, dmax = areas_all["acquired"].min().date(), areas_all["acquired"].max().date()
    dr = f2.slider("Dates", min_value=dmin, max_value=dmax, value=(dmin, dmax), format="MMM YYYY",
                   key="ex_dates") if dmin < dmax else (dmin, dmax)
    months = f3.multiselect("Months (blank = all)", list(range(1, 13)),
                            format_func=lambda m: date(2000, m, 1).strftime("%b"), key="ex_months")
    keep = (areas_all["proportion_no_data"] <= max_nd + 1e-9) & \
           (areas_all["acquired"].dt.date >= dr[0]) & (areas_all["acquired"].dt.date <= dr[1])
    if months:
        keep &= areas_all["acquired"].dt.month.isin(months)
    areas = areas_all[keep]
    if areas.empty:
        st.warning("No scenes left after filtering - loosen the filters.")
        return
    st.caption(f"{areas[['Unique_ID', 'scene_key']].drop_duplicates().shape[0]:,} waterbody-scenes shown.")

    # ---- cover over time -----------------------------------------------------
    st.markdown("#### Cover over time")
    c1, c2, c3 = st.columns(3)
    measure = c1.radio("Show", ["Share of clear area", "Area (ha)"], horizontal=True, key="ex_measure")
    agg = c2.radio("Time step", ["Each scene", "Monthly mean", "Annual mean"], index=0 if len(sel) == 1 else 2,
                   horizontal=True, key="ex_agg")
    how = "Separately"
    if len(sel) > 1:
        how = c3.radio("Waterbodies", ["Separately", "Combined"], horizontal=True, key="ex_how",
                       help="Combined adds the areas of all selected waterbodies in each time step "
                            "(shares are area-weighted). Not available for 'Each scene'.")
    value = "area_ha" if measure == "Area (ha)" else "share"
    pt = explore.period_table(areas, agg)
    if how == "Combined":
        if agg == "Each scene":
            st.info("Combining needs a time step - showing monthly means.")
            agg = "Monthly mean"
            pt = explore.period_table(areas, agg)
        pt = explore.combine_waterbodies(pt)
        st.altair_chart(charts.cover_time_chart(pt, value, agg), **FULL)
    elif len(sel) > 1:
        if len(sel) > 20:
            st.info("Showing the first 20 selected waterbodies - choose fewer, or use Combined.")
            pt = pt[pt["Unique_ID"].isin(sel[:20])]
        st.altair_chart(charts.cover_time_chart(pt, value, agg, facet=True, labels=labels), **CONTENT)
    else:
        st.altair_chart(charts.cover_time_chart(pt, value, agg), **FULL)
    st.download_button("Download the charted values (CSV)",
                       pt.rename(columns={"acquired": "acquired_or_period"}).to_csv(index=False).encode(),
                       file_name="cover_over_time.csv", mime="text/csv", key="ex_dl_pt")

    # ---- pixel maps by year --------------------------------------------------
    st.markdown("#### Pixel maps by year")
    if not pixel_mode:
        st.info("Pixel maps need the saved pixel data - choose 'Saved pixel results' above.")
        return
    wb = st.selectbox("Waterbody", [i for i in sel if i in obs_by_id], format_func=lambda i: labels.get(i, i),
                      key="ex_map_wb")
    obs = obs_by_id[wb]
    o = obs[(obs["proportion_no_data"] <= max_nd + 1e-9) & (obs["date"].dt.date >= dr[0]) &
            (obs["date"].dt.date <= dr[1])]
    if months:
        o = o[o["date"].dt.month.isin(months)]
    fy = explore.yearly_pixel_frequencies(o)
    if fy.empty:
        st.info("No clear observations for this waterbody with the current filters.")
        return
    desc_cols = sorted([c for c in fy.columns if c.startswith("freq: ")],
                       key=lambda c: COVER_ORDER.index(c[6:]) if c[6:] in COVER_ORDER else 99)
    choices = ["Most frequent cover type"] + [f"How often: {c[6:]}" for c in desc_cols] + \
              [f"Model presence: {MODEL_DISPLAY[m]}" for m in MIX_ORDER]
    m1, m2 = st.columns([3, 2])
    pick = m1.selectbox("Map", choices, key="ex_map_layer")
    years = sorted(fy["year"].unique())
    yr = m2.select_slider("Years", options=years, value=(years[0], years[-1]), key="ex_years") if len(years) > 1 \
        else (years[0], years[0])
    fy = fy[(fy["year"] >= yr[0]) & (fy["year"] <= yr[1])]
    if pick == "Most frequent cover type":
        col, cover_name = "dominant_cover", None
    elif pick.startswith("How often"):
        cover_name = pick[len("How often: "):]
        col = "freq: " + cover_name
    else:
        cover_name = None
        col = "presence: " + [k for k, v in MODEL_DISPLAY.items() if v == pick[len("Model presence: "):]][0]
    n_years = fy["year"].nunique()
    st.caption(f"{n_years} years · each square is a 30 m pixel · value = share of that year's clear scenes "
               "(after the filters above).")
    l1, l2 = st.columns(2)
    per_row = l1.slider("Maps per row", 1, 20, min(n_years, 8), key="ex_per_row")
    total_w = l2.slider("Width of the map panel (pixels on screen)", 600, 2600, 1200, 50, key="ex_total_w",
                        help="Set this to roughly the width of your browser window so the maps fill the page.")
    panel_px = max(60, int((total_w - 20 * per_row) / per_row))
    st.altair_chart(charts.pixel_year_chart(fy, pix_by_id[wb], col, cover_name, columns=per_row,
                                            panel_px=panel_px), **CONTENT)
    st.download_button("Download per-pixel, per-year values (CSV)",
                       fy.merge(pix_by_id[wb], on="pixel_id", how="left").to_csv(index=False).encode(),
                       file_name=f"{wb}_pixels_by_year.csv", mime="text/csv", key="ex_dl_fy")


tab_single, tab_batch, tab_explore = st.tabs(["Single waterbody", "Batch processing", "Explore saved results"])
with tab_batch:
    render_batch()
with tab_explore:
    render_explore()

with tab_single:
    # --------------------------------------------------------------------------
    # 1. Choose waterbody
    # --------------------------------------------------------------------------

    st.header("1. Choose a waterbody")
    map_col, info_col = st.columns([3, 2])

    with info_col:
        q = st.text_input("Search by Unique_ID, name or WB_ID", placeholder="e.g. 70730 or Edithvale")
        if q:
            hits = layer.search(q)
            if hits.empty:
                st.info("No matches.")
            else:
                options = hits[layer.id_field].tolist()
                labels = dict(zip(hits[layer.id_field], hits["_label"]))
                pick = st.selectbox(f"{len(hits)} match(es)", options, format_func=lambda i: labels.get(i, i))
                if st.button("Select this waterbody", type="primary"):
                    st.session_state.selected_id = pick
                    st.session_state.map_version += 1
                    st.rerun()
        st.caption("…or zoom in on the map and click a waterbody.")

    sel = layer.get(st.session_state.selected_id) if st.session_state.selected_id else None

    with map_col:
        if sel is not None:
            b = sel["wgs"].bounds
            center = ((b[1] + b[3]) / 2, (b[0] + b[2]) / 2)
            span = max(b[2] - b[0], (b[3] - b[1]) * 1.3, 1e-4)
            zoom = int(np.clip(np.floor(np.log2(360 * 1.2 / span)), 8, 18))
        else:
            center, zoom = MELBOURNE, 10
        m = folium.Map(location=center, zoom_start=zoom, tiles=None, control_scale=True)
        folium.TileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                         attr="Esri, Maxar, Earthstar Geographics", name="Satellite").add_to(m)
        fg = folium.FeatureGroup(name="Waterbodies")
        bounds = st.session_state.get("map_bounds")
        try:
            bounds_ok = bounds is not None and all(
                isinstance(bounds[c][k], (int, float)) for c in ("_southWest", "_northEast") for k in ("lat", "lng"))
        except (KeyError, TypeError):
            bounds_ok = False
        if bounds_ok:
            vis = layer.in_bounds(bounds["_southWest"]["lat"], bounds["_southWest"]["lng"],
                                  bounds["_northEast"]["lat"], bounds["_northEast"]["lng"])
            if len(vis):
                folium.GeoJson(vis[[layer.id_field, "_label", "geometry"]].to_json(),
                               style_function=lambda f: {"color": "#7FDBFF", "weight": 1.2, "fillOpacity": 0.08},
                               tooltip=folium.GeoJsonTooltip(fields=["_label"], aliases=[""])).add_to(fg)
        if sel is not None:
            folium.GeoJson(sel["wgs"].__geo_interface__,
                           style_function=lambda f: {"color": "#FFD700", "weight": 3, "fillOpacity": 0.15}).add_to(fg)
        out = st_folium(m, key=f"pickmap_{st.session_state.map_version}", height=520, use_container_width=True,
                        feature_group_to_add=fg, returned_objects=["last_clicked", "last_object_clicked", "bounds", "zoom"])
        if out:
            nb = out.get("bounds")
            if nb and (nb.get("_southWest") or {}).get("lat") is not None and nb != st.session_state.get("map_bounds"):
                st.session_state.map_bounds = nb
                st.rerun()
            click = out.get("last_object_clicked") or out.get("last_clicked")
            if click and click != st.session_state.last_click:
                st.session_state.last_click = click
                found = layer.at_point(click["lng"], click["lat"])
                if found and found != st.session_state.selected_id:
                    st.session_state.selected_id = found
                    st.rerun()
        if not bounds or (out and (out.get("zoom") or 0) < 13):
            st.caption("Waterbody outlines appear once you zoom in (up to 1,500 at a time).")

    if sel is None:
        st.info("Select a waterbody to continue.")
        st.stop()

    row = sel["row"]
    wb_id = str(row[layer.id_field])
    fp = build_footprint(wb_id, sel["albers"], pad_m=pad_m)
    out_dir = pipeline.results_dir(results_base, source.name, wb_id)

    with info_col:
        st.subheader(f"Waterbody {wb_id}")
        details = {f: row[f] for f in layer.label_fields if isinstance(row.get(f), str) and row.get(f)}
        for f in ["WB_Class", "ConsStatus", "Assmt2020"]:
            if f in row.index and isinstance(row[f], str) and row[f]:
                details[f] = row[f]
        st.markdown("  \n".join(f"**{k}:** {v}" for k, v in details.items()) or "—")
        ha = row["area_m2"] / 10_000
        st.markdown(f"**Area:** {ha:,.2f} ha &nbsp;·&nbsp; **Pixels touched:** {len(fp.pixels):,} &nbsp;·&nbsp; "
                    f"**Pixel-equivalents:** {fp.effective_pixels:,.1f}",
                    help="Pixels touched = 30 m pixels overlapping the polygon. Pixel-equivalents = their summed "
                         "coverage fractions (polygon area / 900 m²).")
        if fp.effective_pixels < 1:
            st.warning("This waterbody is smaller than one 30 m Landsat pixel, so every pixel it touches is mostly "
                       "something else - treat the results with caution.")
        elif fp.effective_pixels < 4:
            st.warning("Very small waterbody (under 4 pixels) - edge pixels dominate the results.")

        man = pipeline._load_manifest(out_dir) if os.path.exists(os.path.join(out_dir, "scenes.csv")) else None
        if man is not None and len(man):
            done = man[man["status"] == "processed"]
            if len(done):
                st.success(f"Saved results: {len(done):,} scenes processed ({done['date'].min()} to {done['date'].max()}). "
                           "Running again only fetches scenes not yet processed.")
        run = st.button("Run / update predictions", type="primary", **FULL)
        st.caption(f"Source: {source.name}. Results are saved to {out_dir}")

    if run:
        if end_date < start_date:
            st.error("'To' date is before 'From' date.")
        else:
            bar = st.progress(0.0, text="Starting...")
            try:
                res = pipeline.process_waterbody(
                    wb_id, sel["albers"], sel["wgs"], source, start_date, end_date,
                    results_base=results_base, background_path=st.session_state.set_background_raster,
                    background_table=abs_path(st.session_state.set_background_table),
                    background_bands=band_names, forests=forests, predictors=predictors, thresholds=thresholds,
                    lookup=lookup, min_clear=min_clear, include_ls7_slc_off=include_slc_off, pad_m=pad_m,
                    max_workers=max_workers, progress=lambda f, msg: bar.progress(f, text=msg),
                    model_file=os.path.basename(MODEL_NPZ))
                t = res.get("timing", {})
                msg = (f"Found {res['scenes_found']:,} scenes; {res.get('already_done', 0):,} were already processed; "
                       f"processed {res['processed']:,} new scenes in {res['seconds'] / 60:.1f} min "
                       f"(search {t.get('search_s', 0) / 60:.1f}, cloud check {t.get('cloud_check_s', 0) / 60:.1f}, "
                       f"download {t.get('download_s', 0) / 60:.1f}, models {t.get('model_s', 0) / 60:.1f} min).")
                st.session_state.run_message = ("success", msg, res.get("warnings", []))
            except Exception as e:  # noqa: BLE001
                st.session_state.run_message = ("error", f"Run failed: {e}", [])
            st.rerun()

    if st.session_state.run_message:
        kind, msg, warns = st.session_state.run_message
        (st.success if kind == "success" else st.error)(msg)
        for w in warns:
            st.warning(w)

    # --------------------------------------------------------------------------
    # 2. Results
    # --------------------------------------------------------------------------

    loaded = get_results(out_dir, results_stamp(out_dir)) if os.path.isdir(out_dir) else None
    if loaded is None or loaded[0].empty:
        st.info("No results yet for this waterbody and data source - press **Run / update predictions**.")
        st.stop()
    obs_all, pixels, scenes = loaded

    st.header("2. Results")
    f1, f2, f3 = st.columns([2, 2, 2])
    max_nodata = f1.slider("Max share of pixels with no data in a scene", 0.0, 1.0, 0.1, 0.05,
                           help="Scenes where more of the footprint than this was cloud/shadow/missing are left out of "
                                "the charts, maps and summary tables (the WTP app used 0).")
    dmin, dmax = obs_all["date"].min().date(), obs_all["date"].max().date()
    drange = f2.slider("Dates", min_value=dmin, max_value=dmax, value=(dmin, dmax), format="MMM YYYY") if dmin < dmax else (dmin, dmax)
    months = f3.multiselect("Months (blank = all)", list(range(1, 13)),
                            format_func=lambda m: date(2000, m, 1).strftime("%b"))

    obs = obs_all[(obs_all["proportion_no_data"] <= max_nodata + 1e-9) &
                  (obs_all["date"].dt.date >= drange[0]) & (obs_all["date"].dt.date <= drange[1])]
    if months:
        obs = obs[obs["date"].dt.month.isin(months)]
    n_sc = obs["scene_key"].nunique()
    st.caption(f"Showing {n_sc:,} of {obs_all['scene_key'].nunique():,} processed scenes.")
    if obs.empty:
        st.warning("No scenes left after filtering - loosen the filters above.")
        st.stop()

    tab_time, tab_space, tab_tables = st.tabs(["Cover over time", "Average cover across the footprint", "Tables & downloads"])

    areas = pipeline.scene_cover_areas(obs)

    with tab_time:
        a1, a2 = st.columns(2)
        measure = a1.radio("Show", ["Area (ha)", "Share of clear area"], horizontal=True)
        agg = a2.radio("Time step", ["Each scene", "Monthly mean", "Annual mean"], horizontal=True)
        value = "area_ha" if measure == "Area (ha)" else "share"
        ytitle = "Area (ha)" if value == "area_ha" else "Share of clear area"
        pt = explore.period_table(areas.assign(Unique_ID=wb_id), agg)
        st.altair_chart(charts.cover_time_chart(pt, value, agg), **FULL)

        st.markdown("**Single model over time** - area where one model says the cover is present")
        mpick = st.selectbox("Model", MIX_ORDER, index=MIX_ORDER.index("openwater_50"), format_func=lambda k: MODEL_DISPLAY[k])
        v = obs[obs["cover_description"] != NO_DATA]
        per = (v.assign(a=v[f"{mpick}_binary"] * v["coverage_fraction"] * 0.09, c=v["coverage_fraction"] * 0.09)
               .assign(acquired=lambda x: pd.to_datetime(x["date"]) + pd.to_timedelta(x["time"].astype(str)))
               .groupby(["scene_key", "acquired"], as_index=False).agg(area_ha=("a", "sum"), clear_ha=("c", "sum"))
               .sort_values("acquired"))
        per["share"] = per["area_ha"] / per["clear_ha"]
        line = alt.Chart(per).mark_line(point=True, color=COVER_COLOURS["Water"] if "water" in mpick else "#2D6B2D").encode(
            x=alt.X("acquired:T", title=None),
            y=alt.Y("area_ha:Q" if value == "area_ha" else "share:Q", title=ytitle,
                    axis=alt.Axis(format="%" if value != "area_ha" else ",.2f")),
            tooltip=[alt.Tooltip("acquired:T", title="Acquired (UTC)", format="%Y-%m-%d %H:%M"),
                     alt.Tooltip("area_ha:Q", format=",.3f"), alt.Tooltip("share:Q", format=".1%")],
        ).properties(height=260)
        st.altair_chart(line.interactive(bind_y=False), **FULL)

        st.markdown("**Average over the scenes shown**")
        mean_tbl = (areas.groupby("cover_description")
                    .agg(mean_area_ha=("area_ha", lambda s: s.sum() / n_sc),
                         mean_share_of_clear_area=("share_of_clear_area", lambda s: s.sum() / n_sc),
                         scenes_present=("scene_key", "nunique"))
                    .reindex([c for c in COVER_ORDER if c in areas["cover_description"].unique()]))
        st.dataframe(mean_tbl.style.format({"mean_area_ha": "{:,.3f}", "mean_share_of_clear_area": "{:.1%}"}),
                     **FULL)

    with tab_space:
        freqs = pipeline.pixel_frequencies(obs)
        if freqs.empty:
            st.info("No clear observations in the filtered scenes.")
        else:
            pix = pixels.merge(freqs, on="pixel_id", how="left")
            desc_cols = [c for c in freqs.columns if c.startswith("freq: ")]
            model_cols = [f"presence: {m}" for m in MIX_ORDER]
            choices = ["Most frequent cover type"] + \
                      [f"How often: {c[6:]}" for c in sorted(desc_cols, key=lambda c: COVER_ORDER.index(c[6:]) if c[6:] in COVER_ORDER else 99)] + \
                      [f"Model presence: {MODEL_DISPLAY[c[10:]]}" for c in model_cols]
            layer_pick = st.selectbox("Map layer", choices, index=0)
            polys = pixel_polygons_wgs84(pix)
            feats = []
            if layer_pick == "Most frequent cover type":
                for i, r in pix.iterrows():
                    dom = r["dominant_cover"] if isinstance(r["dominant_cover"], str) else None
                    feats.append(dict(type="Feature", geometry=dict(type="Polygon", coordinates=[polys[i]]),
                                      properties=dict(c=COVER_COLOURS.get(dom, "#000000") if dom else "#000000",
                                                      o=0.85 if dom else 0.0, t=f"{dom or 'no clear obs'} "
                                                      f"(cover {r['coverage_fraction']:.0%}, {int(np.nan_to_num(r['n_clear_obs']))} obs)")))
                cmap = None
            else:
                col = ("freq: " + layer_pick[len("How often: "):]) if layer_pick.startswith("How often") else \
                      "presence: " + [k for k, v in MODEL_DISPLAY.items() if v == layer_pick[len("Model presence: "):]][0]
                cover_name = layer_pick[len("How often: "):] if layer_pick.startswith("How often") else None
                hi = COVER_COLOURS.get(cover_name, "#1A4A8A") if cover_name else "#1A4A8A"
                cmap = LinearColormap(["#FFFFFF", hi], vmin=0, vmax=1, caption="Share of clear observations")
                for i, r in pix.iterrows():
                    val = r.get(col)
                    ok = val is not None and np.isfinite(val)
                    feats.append(dict(type="Feature", geometry=dict(type="Polygon", coordinates=[polys[i]]),
                                      properties=dict(c=cmap(val) if ok else "#000000", o=0.85 if ok else 0.0,
                                                      t=(f"{val:.0%} of {int(r['n_clear_obs'])} obs" if ok else "no clear obs") +
                                                      f" (cover {r['coverage_fraction']:.0%})")))
            b = sel["wgs"].bounds
            span = max(b[2] - b[0], (b[3] - b[1]) * 1.3, 1e-4)
            zoom_px = int(np.clip(np.floor(np.log2(360 * 3.2 / span)), 8, 19))  # ~800 px wide map
            pm = folium.Map(location=((b[1] + b[3]) / 2, (b[0] + b[2]) / 2), zoom_start=zoom_px, tiles=None,
                            control_scale=True, max_zoom=21)
            folium.TileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                             attr="Esri, Maxar, Earthstar Geographics").add_to(pm)
            folium.GeoJson(dict(type="FeatureCollection", features=feats),
                           style_function=lambda f: {"fillColor": f["properties"]["c"], "fillOpacity": f["properties"]["o"],
                                                     "color": "#333333", "weight": 0.3},
                           tooltip=folium.GeoJsonTooltip(fields=["t"], aliases=[""])).add_to(pm)
            folium.GeoJson(sel["wgs"].__geo_interface__,
                           style_function=lambda f: {"color": "#FFD700", "weight": 2.5, "fillOpacity": 0}).add_to(pm)
            if cmap is not None:
                cmap.add_to(pm)
            st_folium(pm, key=f"pixmap_{wb_id}_{layer_pick}", height=560, use_container_width=True, returned_objects=[])
            if layer_pick == "Most frequent cover type":
                present = [c for c in COVER_ORDER if c in set(pix["dominant_cover"].dropna())]
                st.markdown(" ".join(f"<span style='display:inline-block;width:12px;height:12px;background:{COVER_COLOURS[c]};"
                                     f"border:1px solid #555;margin:0 4px 0 12px'></span>{c}" for c in present),
                            unsafe_allow_html=True)
            st.caption("Each square is one 30 m Landsat pixel; the yellow line is the waterbody boundary. Pixel shares are "
                       "of the scenes kept by the filters above in which that pixel was clear.")

    with tab_tables:
        st.markdown("**Per scene - area of each cover type** (filtered scenes)")
        st.dataframe(areas, **FULL, height=260)
        st.download_button("Download per-scene cover areas (CSV)", areas.to_csv(index=False).encode(),
                           file_name=f"{wb_id}_cover_area_by_scene.csv", mime="text/csv")
        st.markdown("**Per pixel - how often each cover type / model was present** (filtered scenes)")
        pix_out = pixels.merge(pipeline.pixel_frequencies(obs), on="pixel_id", how="left")
        st.dataframe(pix_out, **FULL, height=260)
        st.download_button("Download per-pixel summary (CSV)", pix_out.to_csv(index=False).encode(),
                           file_name=f"{wb_id}_pixel_summary.csv", mime="text/csv")

        st.markdown("**Full table - every pixel in every scene** (same columns as the R workflow's per-waterbody results)")
        full_cols = ["Unique_ID", "pixel_id", "coverage_fraction", "lng", "lat", "x_albers", "y_albers", "date", "time",
                     "solar_date", "platform", "scene_key", "wetness", "brightness", "greenness"] + \
                    [c for m in MIX_ORDER for c in (m, f"{m}_binary")] + \
                    ["cover_mix", "cover_description", "number_of_pixels", "n_blank", "proportion_no_data"]
        which = st.radio("Rows", ["Filtered scenes", "All processed scenes"], horizontal=True)
        full = (obs if which == "Filtered scenes" else obs_all)[full_cols]
        st.caption(f"{len(full):,} rows.")
        fmt = st.radio("Format", ["CSV", "Parquet (smaller, opens in R with arrow::read_parquet)"], horizontal=True)
        if st.button("Prepare download"):
            if fmt == "CSV":
                st.session_state.full_dl = (full.to_csv(index=False).encode(), f"{wb_id}_pixel_by_scene.csv", "text/csv")
            else:
                buf = io.BytesIO()
                full.to_parquet(buf, index=False)
                st.session_state.full_dl = (buf.getvalue(), f"{wb_id}_pixel_by_scene.parquet", "application/octet-stream")
        if st.session_state.get("full_dl") and st.session_state.full_dl[1].startswith(wb_id):
            data, fname, mime = st.session_state.full_dl
            st.download_button(f"Download {fname} ({len(data) / 1e6:,.1f} MB)", data, file_name=fname, mime=mime)

        st.markdown("**Scenes considered**")
        st.dataframe(scenes.sort_values("datetime_utc"), **FULL, height=260)
        st.download_button("Download scene list (CSV)", scenes.to_csv(index=False).encode(),
                           file_name=f"{wb_id}_scenes.csv", mime="text/csv")
