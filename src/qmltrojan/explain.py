"""SHAP explanations of what drives trojan impact.

We fit a ``RandomForestRegressor`` that predicts a target (mean TVD, or classification-accuracy
drop) from the *structural* attributes of each insertion -- inserted gate, insertion mode,
target qubit, position, depth/size deltas, circuit family and size. A ``TreeExplainer`` then
attributes the prediction to each feature, and we report the mean absolute SHAP value per
feature (global importance). This answers "which properties of an inserted gate matter most?".
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor

#: Structural attributes used as explanatory features (categoricals are one-hot encoded).
NUMERIC_FEATURES: tuple[str, ...] = (
    "qubit", "op_position", "depth_before", "depth_delta", "size_before", "size_delta",
    "n_qubits",
)
CATEGORICAL_FEATURES: tuple[str, ...] = ("gate", "mode", "placement", "encoding", "family")


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


def explain_target(df: pd.DataFrame, target: str, seed: int = 0,
                   ) -> tuple[pd.DataFrame, RandomForestRegressor, object]:
    """Fit a RandomForest on structural features and compute SHAP importances for ``target``.

    Returns a DataFrame of (feature, mean_abs_shap, importance_rank), the fitted model, and the
    raw SHAP values array (for optional plotting).
    """
    if target not in df.columns:
        raise ValueError(f"target {target!r} not in dataframe columns")
    X, names = build_feature_matrix(df)
    y = df[target].to_numpy(dtype=float)
    mask = ~np.isnan(y)
    X, y = X.loc[mask], y[mask]

    model = RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=-1)
    model.fit(X, y)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    mean_abs = np.abs(shap_values).mean(axis=0)
    importance = (
        pd.DataFrame({"feature": names, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    importance["rank"] = importance.index + 1
    return importance, model, shap_values
