"""Datasets, dimensionality reduction and angle-encoding scaling.

Every dataset is reduced to a *binary* problem with ``n_qubits`` real features so that
one feature is encoded per qubit. The preprocessing pipeline is

    StandardScaler -> PCA(n_qubits) -> MinMaxScaler(feature_range) -> clip

and it is fitted on the training split only (no test-set leakage). Labels are in {0, 1}.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.datasets import load_breast_cancer, load_digits, load_iris
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

DATASETS: tuple[str, ...] = ("iris", "digits01", "breast_cancer")
DEFAULT_FEATURE_RANGE: tuple[float, float] = (0.0, float(np.pi))


@dataclass
class DatasetSplit:
    """A preprocessed binary classification split ready for angle encoding."""

    name: str
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_range: tuple[float, float]
    seed: int
    meta: dict = field(default_factory=dict)

    @property
    def n_features(self) -> int:
        return int(self.X_train.shape[1])


def _load_raw(name: str) -> tuple[np.ndarray, np.ndarray]:
    """Return the raw binary (X, y) problem for ``name``."""
    if name == "iris":
        # Classic binary Iris: setosa (0) vs versicolor (1).
        X, y = load_iris(return_X_y=True)
        mask = y < 2
        return X[mask], y[mask]
    if name == "digits01":
        X, y = load_digits(return_X_y=True)
        mask = (y == 0) | (y == 1)
        return X[mask], y[mask]
    if name == "breast_cancer":
        return load_breast_cancer(return_X_y=True)
    raise ValueError(f"Unknown dataset {name!r}; expected one of {DATASETS}")


def load_dataset(
    name: str,
    n_qubits: int,
    seed: int,
    test_size: float = 0.3,
    max_train: int | None = None,
    feature_range: tuple[float, float] = DEFAULT_FEATURE_RANGE,
) -> DatasetSplit:
    """Load, split and preprocess a dataset for an ``n_qubits`` angle-encoded VQC.

    Args:
        name: one of :data:`DATASETS`.
        n_qubits: number of PCA components (= qubits, one feature per qubit).
        seed: controls the stratified split and the optional training subsample.
        test_size: fraction of samples held out for testing.
        max_train: optional cap on the number of training samples (stratified subsample),
            used to keep simulation cost at laptop scale.
        feature_range: target interval of the encoding angles.

    Returns:
        A :class:`DatasetSplit` with features in ``feature_range`` and labels in {0, 1}.
    """
    X, y = _load_raw(name)
    y = y.astype(int)
    if n_qubits > X.shape[1]:
        raise ValueError(f"{name} has only {X.shape[1]} features; cannot use {n_qubits} qubits")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    if max_train is not None and len(X_tr) > max_train:
        X_tr, _, y_tr, _ = train_test_split(
            X_tr, y_tr, train_size=max_train, stratify=y_tr, random_state=seed
        )

    lo, hi = feature_range
    prep = Pipeline(
        [
            ("std", StandardScaler()),
            ("pca", PCA(n_components=n_qubits, random_state=seed)),
            ("minmax", MinMaxScaler(feature_range=(lo, hi))),
        ]
    )
    X_tr = prep.fit_transform(X_tr)
    # Test points may fall slightly outside the training range; clip them into the
    # encoding interval so that every angle is valid for the feature map.
    X_te = np.clip(prep.transform(X_te), lo, hi)

    pca: PCA = prep.named_steps["pca"]
    return DatasetSplit(
        name=name,
        X_train=X_tr,
        X_test=X_te,
        y_train=y_tr,
        y_test=y_te,
        feature_range=(float(lo), float(hi)),
        seed=seed,
        meta={"explained_variance": float(pca.explained_variance_ratio_.sum())},
    )
