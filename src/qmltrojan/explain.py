"""SHAP explanations of what drives trojan impact.

We fit a random forest that predicts a target (mean TVD, or classification-accuracy drop) from the
*structural* attributes of each insertion -- inserted gate, insertion mode, target qubit, position,
depth/size deltas, circuit family and size. Exact Tree SHAP then attributes each prediction to the
features, and we report the mean absolute SHAP value per feature (global importance). This answers
"which properties of an inserted gate matter most?".

Portability note
----------------
If the ``shap`` package imports successfully it is used directly (``shap.TreeExplainer``). On some
locked-down machines ``shap`` cannot be imported because it depends on numba/llvmlite, whose
unsigned native library is blocked by Windows Smart App Control; the same policy also blocks
``sklearn.ensemble``. In that case we fall back to (a) a small bootstrap random forest built from
``sklearn.tree.DecisionTreeRegressor`` and (b) the exact path-dependent Tree SHAP implementation in
:mod:`qmltrojan._tree_shap`. Both paths compute the *same* exact Shapley values.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

from qmltrojan._tree_shap import tree_shap_values

try:  # Prefer the real libraries when the environment allows them.
    import shap  # noqa: F401
    from sklearn.ensemble import RandomForestRegressor

    _HAVE_SHAP = True
except Exception:  # pragma: no cover - environment dependent
    _HAVE_SHAP = False

#: Structural attributes used as explanatory features (categoricals are one-hot encoded).
NUMERIC_FEATURES: tuple[str, ...] = (
    "qubit", "op_position", "depth_before", "depth_delta", "size_before", "size_delta",
    "n_qubits",
)
CATEGORICAL_FEATURES: tuple[str, ...] = ("gate", "mode", "placement", "encoding", "family")


class BootstrapForest:
    """Minimal random-forest regressor built from independent decision trees.

    Used only when ``sklearn.ensemble`` is unavailable. Each tree is trained on a bootstrap
    sample with ``max_features`` split subsampling, matching the standard random-forest recipe.
    Exposes ``estimators_`` so exact Tree SHAP can traverse the individual trees.
    """

    def __init__(self, n_estimators: int = 120, max_depth: int | None = 6,
                 max_features: str | float = "sqrt", random_state: int = 0):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.random_state = random_state
        self.estimators_: list[DecisionTreeRegressor] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> BootstrapForest:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n = len(X)
        rng = np.random.default_rng(self.random_state)
        self.estimators_ = []
        for _ in range(self.n_estimators):
            idx = rng.integers(0, n, size=n)  # bootstrap sample
            tree = DecisionTreeRegressor(max_depth=self.max_depth,
                                         max_features=self.max_features,
                                         random_state=int(rng.integers(0, 2**31)))
            tree.fit(X[idx], y[idx])
            self.estimators_.append(tree)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = np.column_stack([t.predict(np.asarray(X, dtype=float))
                                 for t in self.estimators_])
        return preds.mean(axis=1)


def build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """One-hot encode categoricals and select the structural feature columns."""
    cols_num = [c for c in NUMERIC_FEATURES if c in df.columns]
    cols_cat = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    X = df[cols_num].copy()
    for c in cols_cat:
        dummies = pd.get_dummies(df[c].astype(str), prefix=c)
        X = pd.concat([X, dummies], axis=1)
    X = X.fillna(0.0).astype(float)
    return X, list(X.columns)


def _fit_forest(X: np.ndarray, y: np.ndarray, seed: int):
    if _HAVE_SHAP:
        model = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=seed, n_jobs=-1)
    else:
        model = BootstrapForest(n_estimators=120, max_depth=6, max_features="sqrt",
                                random_state=seed)
    model.fit(X, y)
    return model


def _shap_values(model, X: np.ndarray) -> np.ndarray:
    if _HAVE_SHAP:
        import shap
        return np.asarray(shap.TreeExplainer(model).shap_values(X))
    values, _base = tree_shap_values(model, X)
    return values


def explain_target(df: pd.DataFrame, target: str, seed: int = 0, shap_sample: int = 200,
                   ) -> tuple[pd.DataFrame, object, np.ndarray]:
    """Fit a forest on structural features and compute SHAP importances for ``target``.

    The forest is trained on all rows; SHAP values are computed on up to ``shap_sample`` rows
    (seeded) because the pure-Python exact Tree SHAP fallback is the cost bottleneck and global
    mean-|SHAP| importance is stable under row subsampling. Returns a DataFrame of
    (feature, mean_abs_shap, rank), the fitted model, and the raw SHAP values array.
    """
    if target not in df.columns:
        raise ValueError(f"target {target!r} not in dataframe columns")
    X, names = build_feature_matrix(df)
    y = df[target].to_numpy(dtype=float)
    mask = ~np.isnan(y)
    X_arr = X.loc[mask].to_numpy(dtype=float)
    y = y[mask]

    model = _fit_forest(X_arr, y, seed)
    if len(X_arr) > shap_sample:
        rng = np.random.default_rng(seed)
        sel = rng.choice(len(X_arr), size=shap_sample, replace=False)
        X_shap = X_arr[sel]
    else:
        X_shap = X_arr
    shap_values = _shap_values(model, X_shap)

    mean_abs = np.abs(shap_values).mean(axis=0)
    importance = (
        pd.DataFrame({"feature": names, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    importance["rank"] = importance.index + 1
    return importance, model, shap_values


def shap_backend() -> str:
    """Name of the active SHAP backend (for logging/documentation)."""
    return "shap.TreeExplainer" if _HAVE_SHAP else "qmltrojan._tree_shap (exact, no numba)"
