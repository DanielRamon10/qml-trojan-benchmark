"""Tests for the SHAP explanation path (exact Tree SHAP fallback)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from qmltrojan._tree_shap import tree_shap_values
from qmltrojan.explain import BootstrapForest, explain_target, shap_backend


def test_tree_shap_efficiency_property():
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, (200, 5))
    y = 2 * X[:, 0] + (X[:, 1] > 0.5) + 0.1 * rng.normal(size=200)
    forest = BootstrapForest(n_estimators=30, max_depth=5, random_state=0).fit(X, y)
    shap_values, base = tree_shap_values(forest, X[:40])
    recon = base + shap_values.sum(axis=1)
    assert np.allclose(recon, forest.predict(X[:40]), atol=1e-8)


def test_explain_ranks_the_true_driver_first():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({
        "gate": rng.choice(list("hxyzt"), 300),
        "mode": rng.choice(["random", "idle", "controlled"], 300),
        "qubit": rng.integers(0, 4, 300),
        "op_position": rng.integers(0, 10, 300),
        "depth_before": rng.integers(5, 20, 300),
        "depth_delta": 1, "size_before": rng.integers(20, 60, 300), "size_delta": 1,
        "n_qubits": rng.choice([2, 4], 300),
    })
    # Target depends almost entirely on whether the gate is x/y.
    df["tvd_mean"] = df["gate"].isin(["x", "y"]).astype(float) * 0.4 + rng.uniform(0, 0.05, 300)
    importance, _, _ = explain_target(df, "tvd_mean", shap_sample=120)
    top_features = set(importance.head(2)["feature"])
    assert any(f.startswith("gate_") for f in top_features)


def test_shap_backend_reports_a_name():
    assert isinstance(shap_backend(), str) and shap_backend()
