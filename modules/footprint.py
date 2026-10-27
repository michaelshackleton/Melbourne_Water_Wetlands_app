"""
The pixel grid for one waterbody.

Everything is done on DEA's native Landsat grid: GDA94 / Australian Albers
(EPSG:3577), 30 m pixels with edges on multiples of 30 m - the same grid the
original "Melb region" rasters were on. Each pixel that touches the
waterbody polygon gets a coverage fraction (share of the 30 x 30 m pixel
inside the polygon), equivalent to exactextractr's coverage_fraction.

Unlike the original script, the rasters are NOT re-projected to lat/long
before extraction: values are read straight off the native pixels, so no
bilinear resampling smooths the TCI values. Pixel-centre lng/lat are still
reported for compatibility with the existing plotting code.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import shapely
from affine import Affine
from pyproj import Transformer

GRID_CRS = "EPSG:3577"
RES = 30.0
PIXEL_AREA_M2 = RES * RES


@dataclass
class Footprint:
    unique_id: str
    polygon: shapely.Geometry  # in EPSG:3577
    transform: Affine
    width: int
    height: int
    pixels: pd.DataFrame  # one row per pixel touching the polygon

    @property
    def rows(self) -> np.ndarray:
        return self.pixels["row"].to_numpy()

    @property
    def cols(self) -> np.ndarray:
        return self.pixels["col"].to_numpy()

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        x0, y1 = self.transform * (0, 0)
        x1, y0 = self.transform * (self.width, self.height)
        return x0, y0, x1, y1

    @property
    def area_ha(self) -> float:
        return self.polygon.area / 10_000

    @property
    def effective_pixels(self) -> float:
        return float(self.pixels["coverage_fraction"].sum())


def build_footprint(unique_id, polygon_3577, pad_m: float = 60.0) -> Footprint:
    poly = shapely.make_valid(polygon_3577)
    minx, miny, maxx, maxy = poly.bounds
    minx = np.floor((minx - pad_m) / RES) * RES
    miny = np.floor((miny - pad_m) / RES) * RES
    maxx = np.ceil((maxx + pad_m) / RES) * RES
    maxy = np.ceil((maxy + pad_m) / RES) * RES
    width = int(round((maxx - minx) / RES))
    height = int(round((maxy - miny) / RES))
    transform = Affine(RES, 0, minx, 0, -RES, maxy)

    rr, cc = np.mgrid[0:height, 0:width]
    rr, cc = rr.ravel(), cc.ravel()
    x0 = minx + cc * RES
    y1 = maxy - rr * RES
    boxes = shapely.box(x0, y1 - RES, x0 + RES, y1)
    shapely.prepare(poly)
    touches = shapely.intersects(poly, boxes)
    inside = np.zeros_like(touches)
    inside[touches] = shapely.contains(poly, boxes[touches])
    frac = np.zeros(len(boxes))
    frac[inside] = 1.0
    edge = touches & ~inside
    if edge.any():
        frac[edge] = shapely.area(shapely.intersection(boxes[edge], poly)) / PIXEL_AREA_M2
    keep = frac > 0
    rr, cc, frac = rr[keep], cc[keep], np.clip(frac[keep], 0, 1)
    xc = minx + (cc + 0.5) * RES
    yc = maxy - (rr + 0.5) * RES
    lng, lat = Transformer.from_crs(GRID_CRS, "EPSG:4326", always_xy=True).transform(xc, yc)
    pixels = pd.DataFrame({
        "pixel_id": np.arange(len(rr)),
        "row": rr, "col": cc,
        "x_albers": xc, "y_albers": yc,
        "lng": lng, "lat": lat,
        "coverage_fraction": frac,
    })
    return Footprint(str(unique_id), poly, transform, width, height, pixels)


def pixel_polygons_wgs84(pixels: pd.DataFrame) -> list:
    """Pixel squares (from pixel-centre x_albers/y_albers) as WGS84 rings."""
    tr = Transformer.from_crs(GRID_CRS, "EPSG:4326", always_xy=True)
    x0 = pixels["x_albers"].to_numpy() - RES / 2
    y0 = pixels["y_albers"].to_numpy() - RES / 2
    xs = np.stack([x0, x0 + RES, x0 + RES, x0, x0], axis=1)
    ys = np.stack([y0, y0, y0 + RES, y0 + RES, y0], axis=1)
    lx, ly = tr.transform(xs, ys)
    return [[list(p) for p in zip(a, b)] for a, b in zip(lx, ly)]
