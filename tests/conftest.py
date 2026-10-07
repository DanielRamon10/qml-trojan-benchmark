"""Shared fixtures: a small trained-ish VQC and its compiled golden circuit."""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pytest

from qmltrojan.data import load_dataset
from qmltrojan.vqc import VQCSpec, build_ansatz, compile_golden


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


@pytest.fixture(scope="session")
def spec() -> VQCSpec:
    return VQCSpec(dataset="breast_cancer", encoding="z", ansatz_reps=1, n_qubits=4, seed=0)


@pytest.fixture(scope="session")
def weights(spec: VQCSpec) -> np.ndarray:
    """Random (untrained) weights -- enough to exercise structure without a slow fit."""
    n = build_ansatz(spec.n_qubits, spec.ansatz_reps).num_parameters
    return np.random.default_rng(spec.seed).uniform(-np.pi, np.pi, n)


@pytest.fixture(scope="session")
def golden(spec: VQCSpec, weights: np.ndarray):
    return compile_golden(spec, weights)


@pytest.fixture(scope="session")
def data(spec: VQCSpec):
    return load_dataset(spec.dataset, spec.n_qubits, spec.seed, max_train=40)
