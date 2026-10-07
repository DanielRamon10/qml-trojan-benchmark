"""Trojan insertion and triggered-backdoor tests."""

from __future__ import annotations

import numpy as np
import pytest

from qmltrojan.sensitivity import distribution_metrics, predict_sampled
from qmltrojan.trojan import (
    INSERTABLE_GATES,
    INSERTION_MODES,
    BackdoorSpec,
    TrojanSpec,
    apply_trojan,
    build_backdoor,
    make_trigger_inputs,
)


@pytest.mark.parametrize("mode", INSERTION_MODES)
def test_insertion_adds_exactly_one_gate(golden, mode):
    infected, record = apply_trojan(golden, TrojanSpec(gate="h", mode=mode, seed=1))
    assert infected.size() == golden.size() + 1
    assert record["size_delta"] == 1
    assert record["gate"] == "h"


@pytest.mark.parametrize("gate", list(INSERTABLE_GATES))
def test_metadata_is_complete(golden, gate):
    _, record = apply_trojan(golden, TrojanSpec(gate=gate, mode="random", seed=2))
    for key in ("gate", "mode", "qubit", "op_position", "placement", "depth_before",
                "depth_after", "depth_delta", "size_before", "size_after", "size_delta",
                "n_qubits", "n_clbits", "family", "model_id"):
        assert key in record


def test_nondiagonal_trojan_changes_distribution(golden, data):
    # An X at the encoding/ansatz interface must shift the output distribution above the
    # sampling baseline (TVD > 0 and clearly above golden-vs-golden noise).
    infected, _ = apply_trojan(golden, TrojanSpec(gate="x", mode="controlled", seed=0))
    met = distribution_metrics(golden, infected, data.X_test[:4], shots=512, reps=5, seed=0)
    assert met["tvd_mean"] > met["tvd_baseline_mean"]
    assert met["tvd_mean"] > 0.0


def test_backdoor_fires_on_trigger_and_spares_clean(spec, weights):
    bd, gold, record = build_backdoor(spec, weights, BackdoorSpec(k_trigger=3))
    n_data = spec.n_qubits
    rng = np.random.default_rng(0)
    X_clean = rng.uniform(0, np.pi / 2, size=(40, n_data))
    X_trig = make_trigger_inputs(n_data, 3, np.pi / 2, 40, base=X_clean, seed=0)

    pred_g_clean = predict_sampled(gold, X_clean, n_data, shots=2048, seed=0)
    pred_b_clean = predict_sampled(bd, X_clean, n_data, shots=2048, seed=0)
    pred_g_trig = predict_sampled(gold, X_trig, n_data, shots=2048, seed=0)
    pred_b_trig = predict_sampled(bd, X_trig, n_data, shots=2048, seed=0)

    asr = np.mean(pred_g_trig != pred_b_trig)
    clean_agreement = np.mean(pred_g_clean == pred_b_clean)
    assert asr > 0.8  # strong attack at the exact trigger
    assert clean_agreement > asr - 0.3  # trigger is more selective than it is leaky


def test_backdoor_has_one_ancilla(spec, weights):
    bd, gold, _ = build_backdoor(spec, weights, BackdoorSpec(k_trigger=2))
    assert bd.num_qubits == gold.num_qubits + 1
