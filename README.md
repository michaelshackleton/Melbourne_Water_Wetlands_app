# Waterbody Cover Predictor

A desktop app (Python / Streamlit) that predicts **land-cover types within
waterbody footprints across the greater Melbourne region**, using the Landsat
satellite archive. For any waterbody in the region it reads every Landsat 5, 7, 8 and 9 scene from
Digital Earth Australia (DEA) over a chosen date range, runs seven random-forest
cover models on each 30 m pixel, and shows how open water, wet ground, green
and dry vegetation and bare ground change through time and across the
waterbody.

It can be used interactively for one waterbody at a time, as a batch over
thousands of waterbodies, or to explore results already processed.

---

## What it produces

For every clear Landsat scene and every 30 m pixel touching the waterbody:

* the tasselled cap **wetness, brightness and greenness** of the pixel;
* the probability from each of the **seven cover models** and its
  presence/absence after thresholding;
* a combined **cover description**, e.g. *Water*, *Green vegetation and water*,
  *Bare and dry vegetation*, *No prediction* (no model above threshold) or
  *no_data* (cloud, shadow or missing data).

From these the app reports:

* **cover over time** – area (ha) or share of the visible area in each cover
  type, per scene or as monthly / annual means;
* **cover across the footprint** – per-pixel maps of the most frequent cover
  type, how often each cover type occurred, or how often each model said
  "present", for the whole record or year by year;
* **tables** – per-scene cover areas, per-pixel summaries, the full
  pixel-by-scene table and the list of scenes used.

## Requirements

* Windows, macOS or Linux with **Python 3.10 or newer**
* An internet connection (satellite data is read from DEA's public archive –
  no account or login needed)
* A few GB of free disk space if you process many waterbodies

## Installation (one-off)

From a terminal in the app folder:

```
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

The `.venv` folder is tied to the location it was created in; if the app folder
is moved, delete `.venv` and repeat these steps.

## Starting the app

Double-click `run_app.bat` (Windows), or from the activated environment:

```
streamlit run app.py
```

The app opens in your web browser but runs entirely on your own computer.

---

## Using the app

### Sidebar settings (apply to all tabs that process data)

| Setting | What it does |
| --- | --- |
| **Satellite data** | *DEA online* (default) reads Landsat directly from Digital Earth Australia. *Local TCI rasters* reads a folder of pre-computed tasselled cap rasters instead (offline). |
| **From / To** | Date range of scenes to process. |
| **Skip scenes with less than this much of the footprint clear** | Scenes where less than this share of the waterbody is free of cloud, shadow and missing data are not downloaded or modelled (default 50%). |
| **Include Landsat 7 after May 2003** | Landsat 7's scan-line corrector failed in 2003, leaving striped gaps; the gaps are treated like cloud. |
| **Paths & advanced settings** | File locations, number of parallel downloads, padding around the waterbody. Press **Save** to keep changes. |

### Tab 1 – Single waterbody

1. Find a waterbody by typing its `Unique_ID`, name or `WB_ID`, or zoom in on
   the map and click it. Its type, area and number of Landsat pixels are shown;
   very small waterbodies (under one 30 m pixel) are flagged, because each pixel
   then mostly covers the surrounding land.
2. Press **Run / update predictions**. Progress is shown as the app checks each
   scene's cloud cover and reads the clear ones. Results are saved as they are
   produced; running again later (e.g. with a later end date) only fetches
   scenes not already processed.
3. Explore the results:
   * **Filters** – maximum share of no-data pixels per scene (default 10%),
     date range and months;
   * **Cover over time** – stacked chart of cover types (hectares or share of
     the clear area), plus the area where a single chosen model says "present";
   * **Average cover across the footprint** – pixel maps on satellite imagery;
   * **Tables & downloads** – CSV/Parquet downloads of everything above.

### Tab 2 – Batch processing

Processes a list of waterbodies and writes tables to a folder instead of
drawing charts.

* **Input:** a CSV with a `Unique_ID` column (or choose the column), a text
  file with one ID per line, a zipped shapefile / GeoPackage / GeoJSON with a
  `Unique_ID` field, or IDs pasted into the box.
* **Grouping (on by default):** waterbodies whose centres fall in the same
  10 km grid cell are processed together – each scene is downloaded once for
  the whole group and the pixels are then split back out to each waterbody.
  Results are identical to processing them one at a time, but much faster,
  because downloading is almost all of the run time. Before starting, the tab
  shows how many groups your list forms and a rough time estimate.
* **Completeness:** each group is processed for every scene in the date range
  before its tables are written. Scenes that fail to download are retried; a
  waterbody still missing scenes is marked *incomplete* and gets no table, and
  the next run fetches only what is missing.
* **Stopping and resuming:** progress is saved scene by scene. With *Skip
  waterbodies already written to this folder* ticked (default), starting the
  same batch again carries on where it stopped. While a batch is running,
  don't change other settings in the app – that restarts the page and stops
  the batch.

Files written to the output folder:

| File | Contents |
| --- | --- |
| `per_waterbody/<Unique_ID>_cover_area_by_scene.csv` | one row per scene for that waterbody (columns below) |
| `all_waterbodies_cover_area_by_scene.csv` | all the per-waterbody tables stacked; updated as the batch runs |
| `batch_log.csv` | one row per waterbody: status, group, start/finish times, time spent searching / cloud-checking / downloading / modelling / writing, pixel and scene counts, warnings |
| `per_waterbody/<Unique_ID>_pixel_by_scene.parquet` | optional: the full pixel-by-scene table |

Columns of the per-scene table:

| Column | Meaning |
| --- | --- |
| `Unique_ID` | waterbody ID |
| `date`, `time` | acquisition date and time in **UTC** (Landsat passes Melbourne around 23:30 UTC, i.e. the previous calendar day) |
| `solar_date` | local date of the acquisition |
| `platform` | landsat-5, -7, -8 or -9 |
| `n_pixels` | 30 m pixels touching the waterbody polygon |
| `n_clear_pixels` | of those, pixels with a prediction in this scene |
| `proportion_no_data` | share of pixels with no data (cloud, shadow, stripe, missing) |
| `fmask_clear_fraction` | area-weighted clear share of the footprint from the cloud mask |
| `polygon_area_ha` | area of the waterbody polygon |
| `pixel_area_ha` | full area of all pixels touching the polygon (`n_pixels` × 0.09 ha) |
| `footprint_area_ha` | pixel area weighted by the share of each pixel inside the polygon (equals the polygon area) |
| `clear_area_ha` | footprint area with a prediction in this scene |
| `no_data_area_ha` | footprint area with no data in this scene |
| `clear_share_of_footprint` | `clear_area_ha / footprint_area_ha` |
| `water_ha` … `bare_ha`, `no_prediction_ha` | area of each cover description (footprint-weighted) |

The cover-type columns plus `no_data_area_ha` add up to `footprint_area_ha`.
Divide by `clear_area_ha` for proportions of the visible area, or by
`polygon_area_ha` for proportions of the whole waterbody.

### Tab 3 – Explore saved results

Charts results already on disk – nothing is downloaded, so it works offline.

* **Saved pixel results** – point it at the results folder (default), one
  waterbody's folder, its `pixels.parquet`, or a folder of batch
  `*_pixel_by_scene.parquet` files. Gives cover-over-time charts **and**
  per-year pixel maps.
* **Batch output tables** – point it at a batch output folder, its
  `per_waterbody` folder or any `*_cover_area_by_scene.csv`. Gives
  cover-over-time charts only.

Available waterbodies are shown on a map (click to add or remove) and in a
list. With several selected, the cover-over-time chart shows them separately or
combined (areas summed per month/year). **Pixel maps by year** show one map per
year of every pixel – the most frequent cover type, how often a chosen cover
type occurred, or how often a model said "present" – with adjustable maps per
row and panel width. Charted values can be downloaded as CSV.

### Command-line batch processing

`scripts/batch_process.py` does the same as the batch tab from a terminal,
which is more robust for long or overnight runs. From the app folder:

```
.venv\Scripts\python.exe scripts\batch_process.py --ids-file "C:\path\to\ids.csv" --out "C:\path\to\output folder" --start 2006-01-01 --end 2026-10-31
```

| Option | Meaning | Default |
| --- | --- | --- |
| `--ids-file` / `--ids` | file of Unique_IDs, or IDs listed directly | – |
| `--out` | output folder for the tables and log | – |
| `--start`, `--end` | date range (YYYY-MM-DD) | 1987-01-01 to today |
| `--min-clear` | skip scenes with less than this share of the footprint clear | 0.5 |
| `--no-slc-off` | leave out striped Landsat 7 scenes after May 2003 | included |
| `--workers` | parallel downloads | 16 |
| `--group-km` | grid cell size for grouping (`0` = one waterbody at a time) | 10 |
| `--pixel-tables` | also write the full pixel-by-scene table per waterbody | off |
| `--skip-processing` | only (re)write tables from results already saved – no downloading | off |
| `--source local` | use local tasselled cap rasters instead of DEA | DEA |

Press Ctrl+C to stop; run the same command again to resume. Keep the computer
from sleeping during long runs.

### Using the results in R

`scripts/read_results.R` loads a waterbody's saved results and maps cover per
pixel and scene (needs the `arrow`, `dplyr`, `ggplot2`, `sf` and `terra`
packages):

```r
source("scripts/read_results.R")
d <- load_waterbody("path/to/waterbodies app/results", "12664")
list_scenes(d)                              # scenes and their no-data share
plot_scene(d, "2019-02-14")                 # map one scene
plot_scenes(d, max_no_data = 0.1, n = 12)   # small multiples through time
r <- cover_rasters(d, max_no_data = 0.1)    # raster stack, one layer per scene
terra::writeRaster(r, "12664_cover.tif")
```

---

## Where results are stored

Everything the app calculates is saved permanently in the **results folder**
(`results/` inside the app folder unless changed under *Paths & advanced
settings*), one subfolder per waterbody:

```
results/dea/<Unique_ID>/
    pixels.parquet       one row per 30 m pixel: coordinates, share inside the polygon, background predictors
    obs/part-*.parquet   one row per pixel per processed scene: band indices, model probabilities, cover type
    scenes.csv           every scene considered, with its clear share and status
    footprint.geojson    the waterbody outline
    run_info.json        settings used
```

This folder is shared by all three tabs and the command-line script, is never
cleared by the app, and is what makes re-runs and resumed batches skip work
already done. Batch output tables can always be regenerated from it
(`--skip-processing`). Keep it on a local drive (not a cloud-synced folder)
while processing, and back it up afterwards. It is excluded from git.

---

## Method

1. **Pixel grid.** Each waterbody polygon is overlaid on Landsat's native
   30 m grid (GDA94 / Australian Albers, EPSG:3577). Every pixel touching the
   polygon is used, with the share of the pixel inside the polygon recorded and
   used to weight areas.
2. **Scenes.** Landsat 5/7/8/9 Collection 3 Analysis Ready Data (NBART surface
   reflectance, `ga_ls*_ard_3`) are found through DEA's STAC catalogue. Only the
   pixels over the waterbody are read from DEA's cloud-optimised GeoTIFFs.
   Scenes from the same overpass (adjacent tiles on the same solar day) are
   merged into one observation.
3. **Cloud masking.** The DEA `oa_fmask` layer is read first; pixels classed as
   valid, snow or water are kept and cloud, cloud shadow and missing pixels are
   treated as no data. Scenes below the clear-share setting are skipped before
   the reflectance bands are downloaded.
4. **Tasselled cap indices.** Wetness, brightness and greenness are computed
   from the six reflectance bands (×10,000) with the coefficients:

   | | blue | green | red | NIR | SWIR1 | SWIR2 |
   | --- | --- | --- | --- | --- | --- | --- |
   | wetness | 0.0315 | 0.2021 | 0.3102 | 0.1594 | −0.6806 | −0.6109 |
   | brightness | 0.2043 | 0.4158 | 0.5524 | 0.5741 | 0.3124 | −0.2303 |
   | greenness | −0.1603 | −0.2819 | −0.4934 | 0.7940 | −0.0002 | −0.1446 |

5. **Background predictors.** Each pixel's long-term mean, 10th and 90th
   percentile of wetness, brightness and greenness (from whole-landscape
   Landsat composites) are added – nine predictors describing the pixel's usual
   condition.
6. **Cover models.** Seven binomial random forests (1,000 trees each, trained
   in R with `caret`/`randomForest` on the 12 predictors above) give the
   probability that a cover type is present at the level in the model's name:
   `openwater_50`, `water_60` (wet), `veg_30`, `greenVeg_40`, `dryVeg_20`,
   `dryVegBare_50` and `bare_40`. Each probability is converted to
   presence/absence with the threshold that maximises sensitivity +
   specificity (`config/thresholds.csv`).
7. **Cover description.** The combination of models saying "present" is
   mapped to a cover description using `config/cover_combinations.csv`
   (e.g. *wet + water + greenVeg → Green vegetation and water*). No model
   present → *No prediction*; no data → *no_data*.

The Python implementation of the random forests reproduces R's
`predict(model, newdata, type = "prob")` exactly (identical probabilities on
the training data and on independent test rows).

---

## Data included with the app

| File | Contents |
| --- | --- |
| `data/waterbodies/*.parquet` | Melbourne Water region waterbodies layer (v1.3), all attributes, as GeoParquet (split in two files) |
| `data/background_pixels.parquet` | the nine background predictors for every 30 m pixel touching a waterbody in the layer (~850,000 pixels) |
| `models/wetland_cover_models.npz` | the seven random forests, exported from the trained R models |
| `config/thresholds.csv` | probability thresholds for the seven models |
| `config/cover_combinations.csv` | lookup from model combinations to cover descriptions |
| `test_waterbodies.csv` | five example Unique_IDs for trying the batch tab |

### Rebuilding the included data

* **Background table** – built from the three whole-landscape composite
  rasters (mean, 10th and 90th percentile; nine bands each: six reflectance
  bands followed by the three tasselled cap indices) with
  `scripts/make_background_table.py --rasters "<folder with the composites>"`.
  The script identifies the tasselled cap bands by recomputing them from the
  reflectance bands and prints checks. Rebuild it if the waterbodies layer
  changes. (`scripts/build_background_table.py` does the same from a single
  nine-band background raster.)
* **Models** – if the models are retrained, export them with
  `scripts/export_trees.R` and convert with `scripts/build_model_npz.py`
  (instructions in each script), and update `config/thresholds.csv`. Results
  already saved keep the old predictions, so clear or rename the results
  folder afterwards.

---

## Configuration

Defaults are in `config/settings.json`; changes saved from the app go to
`config/user_settings.json` (personal, not tracked by git).

| Setting | Default |
| --- | --- |
| Waterbodies layer | `data/waterbodies` (a .shp/.gpkg/.parquet file also works) |
| Background pixel table | `data/background_pixels.parquet` |
| Background raster (optional, used for pixels missing from the table) | none required |
| Local tasselled cap rasters folder (optional) | – |
| Results folder | `results` |
| DEA STAC catalogues | `explorer.sandbox.dea.ga.gov.au/stac`, `explorer.dea.ga.gov.au/stac` |
| Padding around each waterbody | 60 m |
| Parallel downloads | 16 |

---

## Performance

* Downloading dominates. A single waterbody needs roughly one cloud-mask read
  per scene plus six band reads per clear scene – typically 5–10 minutes for
  20 years, depending on the connection. **Parallel reads** (sidebar) or
  `--workers` controls how many happen at once.
* In batches, grouping means a group of nearby waterbodies takes about as long
  as one waterbody on its own. As a guide, 5,000 waterbodies spread across the
  region form ~150 groups.
* Model prediction is fast (thousands of pixel-scenes per second with `numba`).

## Limitations and notes

* **Small waterbodies.** Many waterbodies are smaller than one 30 m Landsat
  pixel, so their pixels are dominated by surrounding land. The app reports
  pixel counts and warns below four pixel-equivalents.
* **Geographic extent.** Predictions need the background predictors, which
  cover the Melbourne Water region; waterbodies outside them get no
  predictions.
* **Dates are UTC.** `date`/`time` are UTC acquisition times; use `solar_date`
  for the local date.
* **Two scenes on one date** can occur (different satellites or overlapping
  paths); they are kept as separate rows with different times.
* **Landsat 7 after May 2003** has striped gaps, treated as no data.

## Folder layout

| Path | Contents |
| --- | --- |
| `app.py` | the Streamlit app |
| `modules/waterbodies.py` | loading and searching the waterbodies layer |
| `modules/footprint.py` | 30 m pixel grid and coverage fractions |
| `modules/background.py` | background predictors |
| `modules/sources.py` | DEA search and reading, local rasters, tasselled cap |
| `modules/rf_engine.py` | random-forest prediction |
| `modules/classify.py` | thresholds, cover descriptions, colours |
| `modules/pipeline.py` | per-waterbody processing and saved results |
| `modules/group.py` | grouped processing of nearby waterbodies |
| `modules/batch.py` | batch input lists, output tables and log |
| `modules/explore.py`, `modules/charts.py` | loading saved results; charts |
| `scripts/` | command-line batch, background table builders, band check, model export/rebuild, R helper |
| `config/`, `models/`, `data/` | settings, models and included data |

## Data sources and acknowledgements

* Landsat surface reflectance and cloud masks: **Digital Earth Australia**,
  Geoscience Australia – Landsat Collection 3 Analysis Ready Data, provided
  under a Creative Commons Attribution 4.0 licence.
* Waterbodies layer: Melbourne Water region waterbodies v1.3.
* Satellite basemap in the app: Esri World Imagery.
