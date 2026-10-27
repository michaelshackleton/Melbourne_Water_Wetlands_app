"""Loading and searching the waterbodies layer (MWregion_Waterbodies_v1.3)."""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import Point


class WaterbodyLayer:
    def __init__(self, path: str, id_field: str = "Unique_ID", label_fields: list[str] | None = None):
        gdf = gpd.read_file(path)
        if id_field not in gdf.columns:
            raise ValueError(f"'{id_field}' is not a column in {path}. Columns: {list(gdf.columns)}")
        gdf = gdf[gdf.geometry.notna()].copy()
        gdf[id_field] = gdf[id_field].astype(str)
        self.id_field = id_field
        self.label_fields = [f for f in (label_fields or []) if f in gdf.columns]
        self.gdf_albers = gdf.to_crs(3577)
        self.gdf_wgs = gdf.to_crs(4326)
        self.gdf_wgs["area_m2"] = self.gdf_albers.geometry.area.round(1)
        self._sindex = self.gdf_wgs.sindex
        lab = self.gdf_wgs[id_field].copy()
        for f in self.label_fields:
            v = self.gdf_wgs[f].fillna("").astype(str)
            lab = lab + np.where(v != "", " | " + v, "")
        self.gdf_wgs["_label"] = lab
        self._search = self.gdf_wgs["_label"].str.lower()

    def __len__(self):
        return len(self.gdf_wgs)

    def ids(self) -> list[str]:
        return self.gdf_wgs[self.id_field].tolist()

    def search(self, text: str, limit: int = 200) -> pd.DataFrame:
        text = (text or "").strip().lower()
        df = self.gdf_wgs
        if not text:
            return df.iloc[0:0]
        exact = df[df[self.id_field].str.lower() == text]
        hits = df[self._search.str.contains(text, regex=False)]
        out = pd.concat([exact, hits]).drop_duplicates(self.id_field)
        return out.head(limit)

    def get(self, wb_id: str):
        sel = self.gdf_wgs[self.gdf_wgs[self.id_field] == str(wb_id)]
        if sel.empty:
            return None
        i = sel.index[0]
        return dict(row=sel.iloc[0], wgs=sel.geometry.iloc[0], albers=self.gdf_albers.geometry.loc[i])

    def at_point(self, lng: float, lat: float, max_dist_m: float = 150.0) -> str | None:
        p = Point(lng, lat)
        idx = list(self._sindex.query(p, predicate="intersects"))
        if idx:
            cand = self.gdf_wgs.iloc[idx]
            # smallest polygon containing the click (nested polygons are common)
            best = cand.iloc[int(np.argmin(self.gdf_albers.geometry.iloc[idx].area.to_numpy()))]
            return best[self.id_field]
        deg = max_dist_m / 90_000
        idx = list(self._sindex.query(p.buffer(deg)))
        if not idx:
            return None
        cand = self.gdf_wgs.iloc[idx]
        d = shapely.distance(cand.geometry.to_numpy(), p)
        return cand.iloc[int(np.argmin(d))][self.id_field]

    def in_bounds(self, south, west, north, east, limit: int = 1500) -> gpd.GeoDataFrame:
        idx = list(self._sindex.query(shapely.box(west, south, east, north)))
        if len(idx) > limit:
            return self.gdf_wgs.iloc[0:0]
        return self.gdf_wgs.iloc[idx]
