"""
Turns the seven model probabilities into presence/absence flags and the
combined cover description, reproducing the logic of
"script june 2026 no lat longs.R" + "Per waterbody concat results script.R":

* each model's probability is compared (>=) with its maxSensSpec threshold
  (config/thresholds.csv);
* the "present" labels (bare, dryVegBare, dryVeg, greenVeg, veg, wet, water)
  form the cover mix;
* the mix is looked up in config/cover_combinations.csv
  ("all possible combinations.csv") to get the cover description;
* a pixel with no valid satellite/background data is "no_data", and a pixel
  where every model says absent is "No prediction".

Rather than building ";"-joined strings and sorting them as the R scripts
did, the lookup is keyed on the *set* of present labels, which gives the same
answer without the string cleaning step.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

# model name -> label used in the cover mix (same as the R case_when()s)
MODEL_LABELS = {
    "bare_40": "bare",
    "dryVegBare_50": "dryVegBare",
    "dryVeg_20": "dryVeg",
    "greenVeg_40": "greenVeg",
    "veg_30": "veg",
    "water_60": "wet",
    "openwater_50": "water",
}
# order the R script pasted the labels together in
MIX_ORDER = ["bare_40", "dryVegBare_50", "dryVeg_20", "greenVeg_40", "veg_30", "water_60", "openwater_50"]

NO_DATA = "no_data"
NO_PREDICTION = "No prediction"

# Colours from the WTP shiny app / plotting script
COVER_COLOURS = {
    "Bare": "#D4B483",
    "Bare and dry vegetation": "#B8956A",
    "Bare and green vegetation": "#8B9D5E",
    "Bare and vegetation": "#A8995A",
    "Bare and water": "#8BA8B5",
    "Bare, dry vegetation and water": "#7A8F96",
    "Bare, green vegetation and water": "#6B9E8A",
    "Bare, vegetation and water": "#7D9E82",
    "Dry vegetation": "#C8A45A",
    "Dry vegetation and water": "#7B9EA8",
    "Green vegetation": "#2D6B2D",
    "Green vegetation and water": "#3A8C6E",
    "Vegetation": "#7AAF5A",
    "Vegetation and water": "#5A9E7A",
    "Water": "#1A4A8A",
    NO_PREDICTION: "#D9534F",
    NO_DATA: "#E6E6E6",
}

# display / stacking order of cover descriptions (water first)
COVER_ORDER = [
    "Water", "Green vegetation and water", "Vegetation and water", "Dry vegetation and water", "Bare and water",
    "Bare, green vegetation and water", "Bare, vegetation and water", "Bare, dry vegetation and water",
    "Green vegetation", "Vegetation", "Bare and green vegetation", "Bare and vegetation", "Dry vegetation",
    "Bare and dry vegetation", "Bare", NO_PREDICTION, NO_DATA,
]


MODEL_DISPLAY = {
    "openwater_50": "Open water (>=50%)",
    "water_60": "Wet / water (>=60%)",
    "veg_30": "Vegetation (>=30%)",
    "greenVeg_40": "Green vegetation (>=40%)",
    "dryVeg_20": "Dry vegetation (>=20%)",
    "dryVegBare_50": "Dry vegetation + bare (>=50%)",
    "bare_40": "Bare (>=40%)",
}


def load_thresholds(path: str) -> dict[str, float]:
    df = pd.read_csv(path)
    return dict(zip(df["model"].astype(str), df["threshold"].astype(float)))


class CoverLookup:
    """Maps a 7-bit presence pattern to a cover description."""

    def __init__(self, combinations_csv: str):
        df = pd.read_csv(combinations_csv, keep_default_na=False)
        by_set: dict[frozenset, str] = {}
        for cmix, desc in zip(df["cmix"], df["cover_description"]):
            parts = frozenset(p.strip() for p in str(cmix).split(",") if p.strip())
            if NO_DATA in parts:
                continue
            by_set[parts] = desc
        self.by_set = by_set
        # pre-compute all 128 bit patterns (bit i = MIX_ORDER[i] present)
        table = []
        for code in range(128):
            labels = frozenset(MODEL_LABELS[m] for i, m in enumerate(MIX_ORDER) if code >> i & 1)
            if not labels:
                table.append(NO_PREDICTION)
            else:
                table.append(by_set.get(labels, "Unmatched combination"))
        self.table = np.array(table, dtype=object)

    def describe(self, binaries: dict[str, np.ndarray], valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns (cover_description, cover_mix) arrays."""
        code = np.zeros(len(valid), dtype=np.int16)
        for i, m in enumerate(MIX_ORDER):
            code |= (np.nan_to_num(binaries[m]).astype(np.int16) << i)
        desc = self.table[code]
        desc = np.where(valid, desc, NO_DATA)
        mix_lookup = np.array([
            ", ".join(MODEL_LABELS[m] for i, m in enumerate(MIX_ORDER) if c >> i & 1) for c in range(128)
        ], dtype=object)
        mix = np.where(valid, mix_lookup[code], NO_DATA)
        return desc, mix


def default_paths(app_dir: str) -> tuple[str, str]:
    return (os.path.join(app_dir, "config", "thresholds.csv"),
            os.path.join(app_dir, "config", "cover_combinations.csv"))
