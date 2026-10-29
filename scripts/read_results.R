# Reading the app's saved results in R and mapping cover type per pixel.
#
# Everything the app works out is saved per waterbody under the app folder:
#
#   results/dea/<Unique_ID>/          (results/local_tci/... for the offline source)
#     pixels.parquet      one row per 30 m pixel: pixel_id, x_albers, y_albers (pixel
#                         centre, EPSG:3577), lng, lat, coverage_fraction, background values
#     obs/part-*.parquet  one row per pixel per processed scene: scene_key, pixel_id,
#                         wetness/brightness/greenness, the 7 model probabilities and
#                         _binary columns, cover_mix, cover_description
#     scenes.csv          every scene considered: scene_key, date, time (UTC), solar_date,
#                         platform, clear_fraction, status
#     footprint.geojson   the waterbody polygon
#
# (If you ticked "Also write the full pixel-by-scene table" in a batch, the batch
# folder's per_waterbody/<Unique_ID>_pixel_by_scene.parquet already has all of this
# joined together - read it with arrow::read_parquet() and skip load_waterbody().)
#
# Packages: install.packages(c("arrow", "dplyr", "ggplot2", "sf", "terra"))
#
# Usage:
#   source("scripts/read_results.R")
#   d <- load_waterbody("C:/Users/Meshackleton/Documents/rs-utils-main/waterbodies app/results", "40318")
#   list_scenes(d)                                   # scenes with their clear share
#   plot_scene(d, "2019-02-14")                      # one scene (by date, or a scene_key)
#   plot_scenes(d, max_no_data = 0.1, n = 12)        # small multiples of the clearest scenes
#   r <- cover_rasters(d, max_no_data = 0.1)         # SpatRaster, one layer per scene
#   terra::plot(r[[1:4]])
#   terra::writeRaster(r, "40318_cover.tif", overwrite = TRUE)   # cover codes 1..n, with their names
#   ow <- cover_rasters(d, value = "openwater_50")   # open-water probability per scene instead

library(ggplot2)
# dplyr functions are called as dplyr::f() so other packages (raster, terra, MASS...)
# that also define select()/filter() can't get in the way.

.read_pq <- function(path) as.data.frame(arrow::read_parquet(path))

cover_colours <- c(
  "Bare" = "#D4B483", "Bare and dry vegetation" = "#B8956A", "Bare and green vegetation" = "#8B9D5E",
  "Bare and vegetation" = "#A8995A", "Bare and water" = "#8BA8B5", "Bare, dry vegetation and water" = "#7A8F96",
  "Bare, green vegetation and water" = "#6B9E8A", "Bare, vegetation and water" = "#7D9E82",
  "Dry vegetation" = "#C8A45A", "Dry vegetation and water" = "#7B9EA8", "Green vegetation" = "#2D6B2D",
  "Green vegetation and water" = "#3A8C6E", "Vegetation" = "#7AAF5A", "Vegetation and water" = "#5A9E7A",
  "Water" = "#1A4A8A", "No prediction" = "#D9534F", "no_data" = "#E6E6E6"
)

# Read one waterbody's results and join pixels + observations + scene info.
load_waterbody <- function(results_dir, unique_id, source = c("dea", "local_tci")) {
  source <- match.arg(source)
  dir <- file.path(results_dir, source, unique_id)
  if (!dir.exists(dir)) stop("No results folder: ", dir)
  pixels <- .read_pq(file.path(dir, "pixels.parquet"))
  parts <- list.files(file.path(dir, "obs"), pattern = "\\.parquet$", full.names = TRUE)
  if (!length(parts)) stop("No processed scenes yet for ", unique_id)
  obs <- dplyr::bind_rows(lapply(sort(parts), .read_pq)) |>
    dplyr::distinct(scene_key, pixel_id, .keep_all = TRUE)
  scenes <- read.csv(file.path(dir, "scenes.csv"), colClasses = c(scene_key = "character")) |>
    dplyr::filter(status == "processed") |>
    dplyr::distinct(scene_key, .keep_all = TRUE) |>
    dplyr::select(scene_key, date, time, solar_date, platform, clear_fraction)
  out <- obs |>
    dplyr::inner_join(scenes, by = "scene_key") |>
    dplyr::left_join(dplyr::select(pixels, pixel_id, x_albers, y_albers, lng, lat, coverage_fraction), by = "pixel_id") |>
    dplyr::mutate(date = as.Date(date),
           acquired = as.POSIXct(paste(date, time), tz = "UTC")) |>
    dplyr::group_by(scene_key) |>
    dplyr::mutate(proportion_no_data = mean(cover_description == "no_data")) |>
    dplyr::ungroup()
  attr(out, "footprint") <- file.path(dir, "footprint.geojson")
  attr(out, "unique_id") <- unique_id
  out
}

# One row per scene: when, which satellite, and how much of the footprint had data.
list_scenes <- function(d) {
  d |>
    dplyr::group_by(scene_key, acquired, date, platform) |>
    dplyr::summarise(proportion_no_data = dplyr::first(proportion_no_data), n_pixels = dplyr::n(), .groups = "drop") |>
    dplyr::arrange(acquired)
}

.pick_scene <- function(d, scene) {
  if (scene %in% d$scene_key) return(scene)
  keys <- unique(d$scene_key[d$date == as.Date(scene)])
  if (!length(keys)) stop("No processed scene with key or date '", scene, "'")
  if (length(keys) > 1) message("Two scenes on ", scene, " - showing ", keys[1], " (pass the scene_key for the other)")
  keys[1]
}

.outline <- function(d) {
  f <- attr(d, "footprint")
  if (is.null(f) || !file.exists(f)) return(NULL)
  sf::st_transform(sf::st_read(f, quiet = TRUE), 3577)
}

# Map cover type for one scene: each square is a 30 m Landsat pixel, the line is
# the waterbody boundary.
plot_scene <- function(d, scene, outline = TRUE) {
  key <- .pick_scene(d, scene)
  s <- dplyr::filter(d, scene_key == key)
  g <- ggplot() +
    geom_tile(data = s, aes(x_albers, y_albers, fill = cover_description), width = 30, height = 30,
              colour = "grey40", linewidth = 0.1) +
    scale_fill_manual(values = cover_colours, name = "Cover type", drop = TRUE)
  ol <- if (outline) .outline(d) else NULL
  if (!is.null(ol)) g <- g + geom_sf(data = ol, fill = NA, colour = "gold", linewidth = 0.8)
  g + coord_sf(crs = 3577, datum = sf::st_crs(3577)) +
    labs(title = paste("Waterbody", attr(d, "unique_id")),
         subtitle = paste(format(s$acquired[1], "%Y-%m-%d %H:%M UTC"), s$platform[1]), x = NULL, y = NULL) +
    theme_bw() + theme(axis.text = element_blank(), axis.ticks = element_blank())
}

# Small multiples: up to n scenes with at most max_no_data of the footprint missing,
# spread evenly through time (or pass dates = c("2010-01-05", ...)).
plot_scenes <- function(d, max_no_data = 0.1, n = 12, dates = NULL, outline = TRUE) {
  sc <- list_scenes(d) |> dplyr::filter(proportion_no_data <= max_no_data)
  if (!is.null(dates)) sc <- dplyr::filter(sc, date %in% as.Date(dates))
  if (!nrow(sc)) stop("No scenes pass the filter")
  if (nrow(sc) > n) sc <- sc[unique(round(seq(1, nrow(sc), length.out = n))), ]
  s <- d |> dplyr::filter(scene_key %in% sc$scene_key) |>
    dplyr::mutate(panel = format(acquired, "%Y-%m-%d %H:%M"))
  g <- ggplot() +
    geom_tile(data = s, aes(x_albers, y_albers, fill = cover_description), width = 30, height = 30) +
    scale_fill_manual(values = cover_colours, name = "Cover type", drop = TRUE) +
    facet_wrap(~panel)
  ol <- if (outline) .outline(d) else NULL
  if (!is.null(ol)) g <- g + geom_sf(data = ol, fill = NA, colour = "gold", linewidth = 0.5)
  g + coord_sf(crs = 3577, datum = sf::st_crs(3577)) + labs(x = NULL, y = NULL,
                                                            title = paste("Waterbody", attr(d, "unique_id"))) +
    theme_bw() + theme(axis.text = element_blank(), axis.ticks = element_blank())
}

# Cover type as a categorical raster stack (EPSG:3577, 30 m), one layer per scene,
# named by acquisition time - for GIS or further analysis in terra.
# value = "cover_description" (categorical) or any numeric column, e.g. "openwater_50"
# (probability) or "openwater_50_binary".
cover_rasters <- function(d, max_no_data = 1, value = "cover_description") {
  sc <- list_scenes(d) |> dplyr::filter(proportion_no_data <= max_no_data)
  s <- dplyr::filter(d, scene_key %in% sc$scene_key)
  categorical <- is.character(s[[value]])
  lv <- if (categorical) intersect(names(cover_colours), unique(s[[value]])) else NULL
  layers <- lapply(sc$scene_key, function(k) {
    x <- s[s$scene_key == k, c("x_albers", "y_albers", value)]
    if (categorical) x[[value]] <- match(x[[value]], lv)
    terra::rast(x, type = "xyz", crs = "EPSG:3577")
  })
  # put every layer on the same grid (pixels are all on the 30 m Landsat grid)
  ext <- Reduce(terra::union, lapply(layers, terra::ext))
  tmpl <- terra::rast(ext, resolution = 30, crs = "EPSG:3577")
  r <- terra::rast(lapply(layers, function(l) terra::extend(l, tmpl)))
  if (categorical) {
    cats <- data.frame(id = seq_along(lv), cover = lv)
    levels(r) <- rep(list(cats), terra::nlyr(r))
    attr(r, "cover_colours") <- cover_colours[lv]  # for terra::plot(r[[1]], col = attr(r, "cover_colours"))
  }
  names(r) <- format(sc$acquired, "%Y-%m-%d_%H%M")
  r
}
