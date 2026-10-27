# Waterbody Cover Predictor

A Streamlit app that does, for one waterbody at a time, what
`script june 2026 no lat longs.R` and `Per waterbody concat results script.R`
do in batch:

1. you pick a waterbody (search by Unique_ID / name, or click it on the map);
2. the app reads the Landsat 5/7/8/9 archive for just that footprint from
   Digital Earth Australia (DEA) over your date range;
3. it computes tasselled cap wetness/brightness/greenness, adds the long-term
   mean/10th/90th percentile values from the whole-landscape raster, and runs
   the seven binomial random-forest models
   (`bare_40, dryVeg_20, dryVegBare_50, greenVeg_40, openwater_50, veg_30, water_60`)
   on every 30 m pixel of every clear scene;
4. it classifies each pixel into the cover descriptions in
   `all possible combinations.csv` and shows:
   * **Cover over time** – stacked area (ha or share of clear area) per scene,
     or monthly/annual means, plus a single-model line (e.g. open-water area);
   * **Average cover across the footprint** – per-pixel maps of the most
     frequent cover type, how often each cover type occurred, or how often each
     model said "present";
   * **Tables & downloads** – per-scene cover areas, per-pixel summaries, the
     full pixel-by-scene table (same columns as the R workflow) and the scene list.

No R installation is needed: the models were exported from the trained R
objects and the Python predictions match R's `predict(..., type = "prob")`
exactly (see *Model check*).

## Setup (one-off)

```
cd "waterbodies app"
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

As with the other apps in rs-utils-main, `.venv` is not portable – if the
folder is moved, delete `.venv` and repeat the three commands.

## Running

Double-click `run_app.bat`, or from an activated environment:
`streamlit run app.py`.

### Data that ships with the app

| File | What it is |
| --- | --- |
| `data/waterbodies/MWregion_Waterbodies_v1.3_part1/2.parquet` | `MWregion_Waterbodies_v1.3.shp`, all attributes, geometry unchanged, as GeoParquet split in two (~20 MB each) |
| `data/background_pixels.parquet` | the nine background predictors for every 30 m pixel touching a waterbody (850,717 pixels, ~33 MB) |

With both files in place the app needs nothing from the D: drive (the local
TCI rasters option still does).

**How the background table was built.** It holds the values of the three
whole-landscape composites (mean, 10th and 90th percentile; 9 bands each:
six reflectance bands then TCW, TCG, TCB) at every 30 m pixel touching a
waterbody, read directly (they are on the Landsat grid, so no resampling).
It was made with `scripts/make_background_table.py` from the composites in
`G:\SABE\CFE\Projects\Melbourne Uni\Completed or closed projects\17_01455 …\GIS\whole landscape rasters`,
which identifies the tasselled cap bands by recomputing them from the
reflectance bands. Its values match the original R extraction
(`background data for wetlands.csv`) exactly. To rebuild (e.g. after editing
the waterbodies layer):

```
.venv\Scripts\activate
python scripts/make_background_table.py --rasters "G:\…\GIS\whole landscape rasters"
```

(`scripts/build_background_table.py` does the same from the 29.86 m
`background reprojection.tif`, with bilinear interpolation, if ever needed.)

### Paths

Defaults are in `config/settings.json` (relative paths are inside the app
folder); change them in the sidebar under **Paths & advanced settings** and
press **Save** (stored in `config/user_settings.json`, which git ignores).

| Setting | Default |
| --- | --- |
| Waterbodies layer | `data/waterbodies` (a folder of GeoParquet parts; a .shp/.gpkg/.parquet file also works) |
| Background pixel table | `data/background_pixels.parquet` |
| Background raster (optional fallback) | `D:/whole landscape rasters/background reprojection.tif` |
| Background band names (band order) | `mean_TCW, mean_TCG, mean_TCB, Q10W, Q10G, Q10B, Q90W, Q90G, Q90B` (checked with `scripts/check_background_bands.py`) |
| Local TCI rasters folder (optional) | `D:/DEA landsat rasters - TCW, TCB, TCG` |
| Results folder | `results` (inside the app folder, ignored by git) |

## Using it

* **Satellite data** – *DEA online* (default) reads only the footprint's
  pixels from DEA's public cloud-optimised GeoTIFFs; no full scenes are
  downloaded and no login is needed. *Local TCI rasters* re-uses the existing
  `TCI Melb region ….tif` files (offline; useful for comparing with the R results).
* **Skip scenes with less than X of the footprint clear** – the app first reads
  only the cloud mask (fmask) for every scene, then fetches the six reflectance
  bands only for scenes clear enough. Lowering it later and re-running only
  fetches the newly-eligible scenes.
* **Run / update predictions** – results are saved under
  `results/dea/<Unique_ID>/` (or `results/local_tci/…`) as they are produced.
  Re-running (e.g. with a later end date) only processes scenes not done
  before; an interrupted run keeps what it finished.
* **Results filters** – max share of no-data pixels per scene (the WTP app's
  slider; default 10%), date range and months. They apply to all charts,
  maps and summary tables.

### Batch processing

The **Batch processing** tab (next to *Single waterbody*) takes a list of
Unique_IDs - a CSV with a `Unique_ID` column (or pick the column), a text file
with one ID per line, a zipped shapefile / GeoPackage / GeoJSON, or IDs pasted
in - and processes them one after another with the sidebar settings. Instead
of charts it writes to the folder you choose:

* `per_waterbody/<Unique_ID>_cover_area_by_scene.csv` - one row per scene;
* `all_waterbodies_cover_area_by_scene.csv` - all of the above stacked
  (rewritten after every waterbody, so it is usable mid-batch);
* `batch_log.csv` - status, scene counts, time and any warnings per waterbody;
* optionally `per_waterbody/<Unique_ID>_pixel_by_scene.parquet` (full table).

"Skip waterbodies already written to this folder" lets you stop a batch and
carry on later; scenes already fetched for a waterbody (by the app or a
previous batch) are never fetched again. Only scenes inside the sidebar's date
range and clear threshold are written. While a batch runs, leave the other
settings alone - changing anything restarts the page and stops the batch.

Columns of the per-scene table:

| Column | Meaning |
| --- | --- |
| `Unique_ID, date, time` | waterbody; acquisition date/time in UTC (as in the R outputs) |
| `solar_date, platform` | local date; landsat-5/7/8/9 |
| `n_pixels, n_clear_pixels` | 30 m pixels touching the polygon; of those, pixels with a prediction in this scene |
| `proportion_no_data` | share of pixels with no data (cloud, shadow, SLC-off stripe, missing) - same as the R column |
| `fmask_clear_fraction` | area-weighted clear share from the cloud mask (what "skip scenes under X clear" uses) |
| `polygon_area_ha` | area of the waterbody polygon |
| `pixel_area_ha` | full area of all pixels touching the polygon (`n_pixels` x 0.09 ha) |
| `footprint_area_ha` | coverage-weighted pixel area (sum of coverage fractions x 0.09 ha) - equals the polygon area |
| `clear_area_ha` | coverage-weighted area with a prediction in this scene |
| `no_data_area_ha` | coverage-weighted area with no data in this scene |
| `clear_share_of_footprint` | `clear_area_ha / footprint_area_ha` |
| `water_ha … bare_ha, no_prediction_ha` | coverage-weighted area of each cover description |

The cover-type columns plus `no_data_area_ha` add up to `footprint_area_ha`, so
divide by `clear_area_ha` for proportions of the visible area, or by
`footprint_area_ha` / `polygon_area_ha` for proportions of the whole waterbody.

### Command-line batch

`scripts/batch_process.py` does the same from a terminal (handy for very long
lists or overnight runs) and writes the identical tables:

```
python scripts/batch_process.py --ids-file "D:/eDNA project/eDNA_waterbodies_1000m_buffer_selection.shp" --out "D:/eDNA cover results"
python scripts/batch_process.py --ids 70730 70731 --start 2000-01-01 --end 2025-12-31 --out "D:/cover results"
```

## How it differs from the R workflow

Deliberate changes (to be faster or more correct):

| R workflow | App |
| --- | --- |
| Reads each full Melbourne-region scene from disk for every waterbody, re-crops and re-projects the background raster for every scene | Reads only the footprint window, background read once per waterbody |
| Re-projects every scene to lat/long with **bilinear** resampling before extracting | Works on the native 30 m Albers (EPSG:3577) pixels – the TCI values are the satellite values, not interpolated (lng/lat of pixel centres are still reported) |
| Tiles from the same overpass saved as separate files | Same-day tiles are merged into one observation |
| Archive frozen at whatever was downloaded | Any date range up to the latest DEA scene, Landsat 9 included |
| Thresholds matched by row position (`thesholds[1,2]` …) | Matched by model name |
| Cover description from cleaned/sorted strings + `all possible combinations.csv` | Same table, looked up by the set of present labels (same answers) |

Columns kept for compatibility: `Unique_ID, coverage_fraction, lng, lat,
<model>, <model>_binary, cover_mix, cover_description, date, time,
number_of_pixels, n_blank, proportion_no_data`. `date`/`time` are UTC, like
the R filenames; `solar_date` (local date) is added because Landsat passes
Melbourne at ~23:30 UTC, i.e. the R `date` is the day *before* the local date.

## Things to check / issues found in the R scripts

1. **Background band order** – checked: `background reprojection.tif` is
   W, G, B, matching the names the R script assigns. Its pixels are 29.86 m,
   so values are bilinearly interpolated onto the 30 m Landsat grid (as the R
   script's `projectRaster` also did).
2. **The "90% coverage" filter doesn't do what its comment says.** It drops
   pixels with `cover_mix == ";;;;;;"` (all models absent = "No prediction"),
   not cloud/no-data pixels (`"no_data;no_data;…"`), so it filters scenes by
   share of "No prediction", not by satellite coverage. The app records the
   clear fraction per scene and filters on that instead.
3. **The concat script's join is a right join.**
   `combined_data[all_combs, on = "cmix"]` keeps every row of `all_combs`, so
   each waterbody file gets extra all-NA rows for combinations that never
   occurred (and would drop any combination missing from the table).
4. **Cloud masking.** The raw rasters were masked keeping fmask valid, snow and
   water (load_ard's default); the app does the same.
5. **Small waterbodies.** The median waterbody in the layer is ~650 m², smaller
   than one Landsat pixel. The app shows pixel counts and warns below 4
   pixel-equivalents.

## Model check

`models/wetland_cover_models.npz` was built from
`Final binomial model no coordinates unbalanced models 05 Dec 2025.rda` with
`scripts/export_trees.R` → `scripts/build_model_npz.py`. On the 805 training
rows plus 3,000 jittered rows, and on a real waterbody run through the full
pipeline, the Python probabilities equal R's `predict(model, x, type="prob")`
to the last digit for all seven models. To use retrained models, re-run those
two scripts.

## Speed

* Prediction: ~7,500 pixel-scenes per second on a 2-core machine with numba
  (several times faster on a typical laptop).
* DEA reads dominate: roughly one cloud-mask read per scene plus six band reads
  per clear scene. A full 1987–present archive (~2,000–3,000 scenes over most
  Melbourne waterbodies) typically takes several minutes to ~15 minutes
  depending on the connection; **Parallel reads** (sidebar) controls how many
  happen at once.

## Folder layout

* `app.py` – the Streamlit UI
* `modules/waterbodies.py` – loading/searching the layer, map click lookup
* `modules/footprint.py` – 30 m pixel grid and exact coverage fractions
* `modules/background.py` – background predictors from the pixel table, or the raster as a fallback
* `modules/sources.py` – DEA STAC search + windowed COG reads, local TCI reader, tasselled cap
* `modules/rf_engine.py` – random forest prediction (numba, numpy fallback)
* `modules/classify.py` – thresholds, cover mix → cover description, colours
* `modules/pipeline.py` – per-waterbody processing, caching, summaries
* `modules/batch.py` – ID list reading and batch output tables
* `config/` – settings, thresholds (`maxSensSpec thresholds for models 15122025.csv`),
  cover combinations (`all possible combinations.csv`)
* `models/` – the exported forests
* `data/` – the waterbodies layer and background pixel table
* `scripts/` – batch processing, background table builders, background band check, model export/rebuild
