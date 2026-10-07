"""Exact path-dependent Tree SHAP in pure NumPy (no numba/llvmlite dependency).

This implements Algorithm 2 of Lundberg et al., "Consistent Individualized Feature Attribution
for Tree Ensembles" (2018) -- the same exact algorithm that ``shap.TreeExplainer`` uses for tree
models. It is provided as a fallback for environments where the ``shap`` package cannot be
imported (e.g. Windows Smart App Control blocking the unsigned ``llvmlite.dll`` that numba, and
hence shap, load at import time). The two produce identical Shapley values; this version simply
has no native dependency.

The implementation is validated by the Shapley *efficiency* property: for every sample, the base
value plus the sum of per-feature SHAP values equals the model's prediction.
"""

from __future__ import annotations

import numpy as np


def _tree_shap_single(tree, x: np.ndarray, n_features: int) -> np.ndarray:
    """Exact SHAP values for one sample against one sklearn decision tree."""
    children_left = tree.children_left
    children_right = tree.children_right
    features = tree.feature
    thresholds = tree.threshold
    values = tree.value[:, 0, 0]
    node_sample = tree.weighted_n_node_samples

    phi = np.zeros(n_features)

    # Each path element carries: feature index, zero-fraction, one-fraction, proportion weight.
    def extend(path, pz, po, pi):
        path = [list(p) for p in path]
        length = len(path)
        path.append([pi, pz, po, 1.0 if length == 0 else 0.0])
        for i in range(length - 1, -1, -1):
            path[i + 1][3] += po * path[i][3] * (i + 1) / (length + 1)
            path[i][3] = pz * path[i][3] * (length - i) / (length + 1)
        return path

    def unwind(path, i):
        path = [list(p) for p in path]
        length = len(path) - 1
        n = path[length][3]
        for j in range(length - 1, -1, -1):
            if path[i][2] != 0:  # one-fraction
                tmp = path[j][3]
                path[j][3] = n * (length + 1) / ((j + 1) * path[i][2])
                n = tmp - path[j][3] * path[i][1] * (length - j) / (length + 1)
            else:
                path[j][3] = (path[j][3] * (length + 1)) / (path[i][1] * (length - j))
        for j in range(i, length):
            path[j][1] = path[j + 1][1]
            path[j][2] = path[j + 1][2]
        return path[:-1]

    def recurse(node, path, pz, po, pi):
        path = extend(path, pz, po, pi)
        if children_left[node] == children_right[node]:  # leaf
            for i in range(1, len(path)):
                w = sum(row[3] for row in unwind(path, i))
                phi[path[i][0]] += w * (path[i][2] - path[i][1]) * values[node]
        else:
            feat = features[node]
            if x[feat] <= thresholds[node]:
                hot, cold = children_left[node], children_right[node]
            else:
                hot, cold = children_right[node], children_left[node]
            iz = io = 1.0
            # if this feature already appears on the path, merge (unwind then re-extend).
            k = next((j for j in range(1, len(path)) if path[j][0] == feat), None)
            if k is not None:
                iz, io = path[k][1], path[k][2]
                path = unwind(path, k)
            w_hot = node_sample[hot] / node_sample[node]
            w_cold = node_sample[cold] / node_sample[node]
            recurse(hot, path, iz * w_hot, io, feat)
            recurse(cold, path, iz * w_cold, 0.0, feat)

    recurse(0, [], 1.0, 1.0, -1)
    return phi


def tree_shap_values(model, X: np.ndarray) -> tuple[np.ndarray, float]:
    """Exact Tree SHAP values for a fitted sklearn tree/forest regressor.

    Returns ``(shap_values, base_value)`` with ``shap_values`` of shape ``(n_samples, n_features)``.
    Satisfies ``base_value + shap_values.sum(axis=1) ≈ model.predict(X)``.
    """
    X = np.asarray(X, dtype=float)
    n_samples, n_features = X.shape

    # Duck-typed: accept a forest (``.estimators_`` list), a single tree estimator
    # (``.tree_``), or a raw sklearn ``Tree`` object. Avoids importing sklearn.ensemble,
    # which may be blocked by OS application-control policies on some machines.
    if hasattr(model, "estimators_"):
        estimators = [est.tree_ for est in model.estimators_]
    elif hasattr(model, "tree_"):
        estimators = [model.tree_]
    elif hasattr(model, "children_left"):
        estimators = [model]
    else:
        raise TypeError(f"Unsupported model type for exact Tree SHAP: {type(model)!r}")

    shap_values = np.zeros((n_samples, n_features))
    base = 0.0
    for tree in estimators:
        base += tree.value[0, 0, 0]
        for s in range(n_samples):
            shap_values[s] += _tree_shap_single(tree, X[s], n_features)
    n_trees = len(estimators)
    return shap_values / n_trees, base / n_trees
