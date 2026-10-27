"""
Random forest prediction for the seven binomial cover models.

The models were trained in R (caret + randomForest, 1000 trees each, 12
numeric predictors, no pre-processing). Their trees were exported with R's
own getTree() (scripts/export_trees.R) and packed into
models/wetland_cover_models.npz (scripts/build_model_npz.py), so the app
needs neither R nor the 58 MB .rda file.

Prediction follows randomForest's rules exactly: a numeric split sends a
case left when value <= split point, and the "prob" output is the fraction
of trees whose terminal node is "Presence". This was checked against R's
predict(model, newdata, type = "prob") on the training data plus 3,000
jittered rows - see README "Model check".

Rows with any missing predictor are not predicted (returned as NaN), which
mirrors caret's default na.omit behaviour in the original R script, where
those pixels became "no_data".
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

try:  # numba makes prediction ~50x faster; fall back to numpy if missing
    from numba import njit, prange

    _HAVE_NUMBA = True
except Exception:  # noqa: BLE001
    _HAVE_NUMBA = False


@dataclass
class Forest:
    name: str
    feature: np.ndarray  # int16, -1 at leaves
    threshold: np.ndarray  # float64
    left: np.ndarray  # int32 global node index, -1 at leaves
    right: np.ndarray  # int32
    leaf_value: np.ndarray  # uint8, 1 = Presence leaf
    roots: np.ndarray  # int32, root node index of each tree

    @property
    def n_trees(self) -> int:
        return len(self.roots)


def load_forests(npz_path: str) -> tuple[dict[str, Forest], list[str]]:
    z = np.load(npz_path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    forests = {}
    for name in meta["models"]:
        forests[name] = Forest(
            name=name,
            feature=z[f"{name}__feature"].astype(np.int16),
            threshold=z[f"{name}__threshold"].astype(np.float64),
            left=z[f"{name}__left"].astype(np.int32),
            right=z[f"{name}__right"].astype(np.int32),
            leaf_value=z[f"{name}__leaf_value"].astype(np.uint8),
            roots=z[f"{name}__roots"].astype(np.int32),
        )
    return forests, list(meta["predictors"])


if _HAVE_NUMBA:

    @njit(parallel=True, cache=True)
    def _predict_numba(X, feature, threshold, left, right, leaf_value, roots):  # pragma: no cover - compiled
        n = X.shape[0]
        out = np.zeros(n, dtype=np.float64)
        n_trees = roots.shape[0]
        for i in prange(n):
            votes = 0
            for t in range(n_trees):
                node = roots[t]
                while feature[node] >= 0:
                    if X[i, feature[node]] <= threshold[node]:
                        node = left[node]
                    else:
                        node = right[node]
                votes += leaf_value[node]
            out[i] = votes / n_trees
        return out


def _predict_numpy(X, f: Forest) -> np.ndarray:
    """Level-by-level traversal, vectorised over rows. Slower than numba but
    has no extra dependency."""
    n = X.shape[0]
    votes = np.zeros(n, dtype=np.int32)
    rows = np.arange(n)
    for root in f.roots:
        node = np.full(n, root, dtype=np.int64)
        active = f.feature[node] >= 0
        while active.any():
            idx = np.where(active)[0]
            nd = node[idx]
            go_left = X[rows[idx], f.feature[nd]] <= f.threshold[nd]
            node[idx] = np.where(go_left, f.left[nd], f.right[nd])
            active[idx] = f.feature[node[idx]] >= 0
        votes += f.leaf_value[node]
    return votes / f.n_trees


def predict_presence(forest: Forest, X: np.ndarray) -> np.ndarray:
    """Probability of 'Presence' (fraction of tree votes) per row of X
    (columns in the model's predictor order). Rows with NaN get NaN."""
    X = np.ascontiguousarray(X, dtype=np.float64)
    ok = np.isfinite(X).all(axis=1)
    out = np.full(X.shape[0], np.nan)
    if ok.any():
        Xo = X[ok]
        if _HAVE_NUMBA:
            out[ok] = _predict_numba(Xo, forest.feature, forest.threshold, forest.left,
                                     forest.right, forest.leaf_value, forest.roots)
        else:
            out[ok] = _predict_numpy(Xo, forest)
    return out


def engine_name() -> str:
    return "numba (compiled)" if _HAVE_NUMBA else "numpy (slower - install numba for speed)"
