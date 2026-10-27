"""
Converts the per-model tree tables written by scripts/export_trees.R into the
single compact file the app reads (models/wetland_cover_models.npz).

Usage (from the app folder, with the app's environment active):
    python scripts/build_model_npz.py <folder with the exported CSVs> [output .npz]

Only needed if the R models are retrained - the app ships with the .npz
already built from "Final binomial model no coordinates unbalanced models
05 Dec 2025.rda".
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

PREDICTORS = ["wetness", "brightness", "greenness", "mean_TCW", "mean_TCB", "mean_TCG",
              "Q10W", "Q90W", "Q10B", "Q90B", "Q10G", "Q90G"]
POSITIVE_CLASS = "Presence"


def convert_model(csv_path: str, predictors: list[str]) -> dict[str, np.ndarray]:
    df = pd.read_csv(csv_path)
    df = df.sort_values(["tree", "node"]).reset_index(drop=True)
    n_per_tree = df.groupby("tree", sort=True).size().to_numpy()
    offsets = np.concatenate([[0], np.cumsum(n_per_tree)[:-1]]).astype(np.int64)
    tree_offset = np.repeat(offsets, n_per_tree)

    status = df["status"].to_numpy()
    is_leaf = status == -1
    feat_index = {p: i for i, p in enumerate(predictors)}
    sv = df["split var"].to_numpy(dtype=object)
    feature = np.full(len(df), -1, dtype=np.int16)
    for i in np.where(~is_leaf)[0]:
        name = str(sv[i])
        if name not in feat_index:
            raise ValueError(f"{csv_path}: split variable {name!r} is not one of the expected predictors {predictors}")
        feature[i] = feat_index[name]

    # getTree() daughters are 1-based row numbers within each tree (0 at leaves)
    left = np.where(is_leaf, -1, df["left daughter"].to_numpy() - 1 + tree_offset).astype(np.int32)
    right = np.where(is_leaf, -1, df["right daughter"].to_numpy() - 1 + tree_offset).astype(np.int32)
    threshold = df["split point"].to_numpy(dtype=np.float64)
    pred = df["prediction"].astype(str).to_numpy()
    leaf_value = np.where(is_leaf & (pred == POSITIVE_CLASS), 1, 0).astype(np.uint8)
    return dict(feature=feature, threshold=threshold, left=left, right=right,
                leaf_value=leaf_value, roots=offsets.astype(np.int32))


def main():
    src = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(__file__), "..", "models", "wetland_cover_models.npz")
    csvs = sorted(f for f in os.listdir(src) if f.endswith(".csv") and not f.startswith("r_validation"))
    arrays: dict[str, np.ndarray] = {}
    names = []
    for f in csvs:
        name = f[:-4]
        names.append(name)
        for k, v in convert_model(os.path.join(src, f), PREDICTORS).items():
            arrays[f"{name}__{k}"] = v
        print(f"{name}: {len(arrays[name + '__roots'])} trees, {len(arrays[name + '__feature'])} nodes")
    arrays["meta"] = np.array(json.dumps({"models": names, "predictors": PREDICTORS, "positive_class": POSITIVE_CLASS}))
    np.savez_compressed(out, **arrays)
    print("Wrote", os.path.abspath(out))


if __name__ == "__main__":
    main()
