"""Altair charts shared by the single-waterbody and "Explore saved results" tabs."""
from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd

from .classify import COVER_COLOURS, COVER_ORDER, NO_DATA


def _cover_scale(covers) -> tuple[list[str], list[str]]:
    present = [c for c in COVER_ORDER if c in set(covers)] + [c for c in covers if c not in COVER_ORDER]
    present = list(dict.fromkeys(present))
    return present, [COVER_COLOURS.get(c, "#999999") for c in present]


def cover_time_chart(pt: pd.DataFrame, value: str, agg: str, facet: bool = False,
                     labels: dict | None = None, height: int = 420):
    """Stacked area of cover types over time. `pt` comes from
    explore.period_table() / combine_waterbodies(): columns Unique_ID, acquired,
    cover, area_ha, share. value = "area_ha" or "share"."""
    d = pt.copy()
    if value == "share":
        d = d[d["cover"] != NO_DATA]
    d["value"] = d[value].fillna(0.0)
    present, colours = _cover_scale(d["cover"].unique())
    d["order"] = d["cover"].map({c: i for i, c in enumerate(present)})
    if labels:
        d["waterbody"] = d["Unique_ID"].map(lambda i: labels.get(i, i))
    else:
        d["waterbody"] = d["Unique_ID"]
    each = agg == "Each scene"
    ytitle = "Area (ha)" if value == "area_ha" else "Share of clear area"
    base = alt.Chart(d).mark_area(interpolate="step-after" if agg == "Annual mean" else "linear").encode(
        x=alt.X("acquired:T", title=None),
        y=alt.Y("value:Q", stack="zero", title=ytitle, axis=alt.Axis(format="%" if value != "area_ha" else ",.2f")),
        color=alt.Color("cover:N", scale=alt.Scale(domain=present, range=colours), title="Cover type",
                        legend=alt.Legend(orient="right", labelLimit=320)),
        order=alt.Order("order:Q"),
        tooltip=["waterbody:N",
                 alt.Tooltip("acquired:T", title="Acquired (UTC)" if each else "Period",
                             format="%Y-%m-%d %H:%M" if each else ("%Y-%m" if agg == "Monthly mean" else "%Y")),
                 "cover:N", alt.Tooltip("value:Q", format=",.3f" if value == "area_ha" else ".1%")],
    )
    if facet and d["waterbody"].nunique() > 1:
        # one chart per waterbody (not a facet): Vega-Lite fills dates missing from a panel - i.e. the other
        # waterbodies' scenes - with zeros when stacking, which shows up as white gaps
        panels = [base.transform_filter(alt.datum.waterbody == wb).properties(height=170, width=820, title=wb)
                  for wb in list(dict.fromkeys(d["waterbody"]))]
        return alt.vconcat(*panels).resolve_scale(color="shared",
                                                  y="independent" if value == "area_ha" else "shared")
    return base.properties(height=height).interactive(bind_y=False)


def pixel_year_chart(freq_year: pd.DataFrame, pixels: pd.DataFrame, column: str, cover_name: str | None = None,
                     columns: int = 5, panel_px: int = 170):
    """Small-multiple maps, one per year, of every 30 m pixel. column =
    "dominant_cover" (categorical) or a "freq: ..." / "presence: ..." column."""
    # every footprint pixel in every year, so pixels with no clear scene that year show as light grey
    years = sorted(freq_year["year"].unique())
    grid = pd.MultiIndex.from_product([years, pixels["pixel_id"].to_numpy()], names=["year", "pixel_id"]).to_frame(index=False)
    d = grid.merge(freq_year, on=["year", "pixel_id"], how="left").merge(
        pixels[["pixel_id", "x_albers", "y_albers", "coverage_fraction"]], on="pixel_id", how="left")
    d["has"] = d["n_clear_obs"].notna()
    d["n_clear_obs"] = d["n_clear_obs"].fillna(0)
    d["dominant_cover"] = d["dominant_cover"].fillna("no clear scene")
    d["x0"], d["x1"] = d["x_albers"] - 15, d["x_albers"] + 15
    d["y0"], d["y1"] = d["y_albers"] - 15, d["y_albers"] + 15
    xmin, xmax = pixels["x_albers"].min() - 15, pixels["x_albers"].max() + 15
    ymin, ymax = pixels["y_albers"].min() - 15, pixels["y_albers"].max() + 15
    span = max(xmax - xmin, ymax - ymin)
    w = int(panel_px * (xmax - xmin) / span)
    h = int(panel_px * (ymax - ymin) / span)
    x = alt.X("x0:Q", scale=alt.Scale(domain=[xmin, xmax], nice=False, zero=False), axis=None)
    y = alt.Y("y0:Q", scale=alt.Scale(domain=[ymin, ymax], nice=False, zero=False), axis=None)
    tips = [alt.Tooltip("year:O"), alt.Tooltip("pixel_id:Q"), alt.Tooltip("n_clear_obs:Q", title="clear scenes"),
            alt.Tooltip("dominant_cover:N", title="most frequent")]
    if column == "dominant_cover":
        present, colours = _cover_scale([c for c in d["dominant_cover"].unique() if c != "no clear scene"])
        if not d["has"].all():
            present, colours = present + ["no clear scene"], colours + ["#EEEEEE"]
        color = alt.Color("dominant_cover:N", scale=alt.Scale(domain=present, range=colours),
                          title="Most frequent cover", legend=alt.Legend(orient="bottom", columns=3, labelLimit=320))
    else:
        d["value"] = d[column] if column in d else np.nan
        hi = COVER_COLOURS.get(cover_name, "#1A4A8A") if cover_name else "#1A4A8A"
        color = alt.condition(alt.datum.has,
                              alt.Color("value:Q", scale=alt.Scale(domain=[0, 1], range=["#FFFFFF", hi]),
                                        title="Share of clear scenes", legend=alt.Legend(format="%", orient="bottom")),
                              alt.value("#EEEEEE"))
        tips.append(alt.Tooltip("value:Q", title="share", format=".0%"))
    d["has"] = d["has"].astype(bool)
    return alt.Chart(d).mark_rect(stroke="#BBBBBB", strokeWidth=0.2).encode(
        x=x, x2="x1:Q", y=y, y2="y1:Q", color=color, tooltip=tips,
    ).properties(width=w, height=h).facet(
        facet=alt.Facet("year:O", title=None, header=alt.Header(labelFontSize=12, labelFontWeight="bold")),
        columns=columns, spacing=12,
    ).configure_view(stroke="#999999")
