"""Trojan insertion and triggered-backdoor tests."""

from __future__ import annotations

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit.library import z_feature_map
from qiskit.quantum_info import Statevector

from qmltrojan.sensitivity import distribution_metrics, predict_sampled
from qmltrojan.trojan import (
    INSERTABLE_GATES,
    INSERTION_MODES,
    BackdoorSpec,
    TrojanSpec,
    _real_ops_per_wire,
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
    for key in ("gate", "mode", "qubit", "op_position", "placement", "at_wire_end", "depth_before",
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


def test_idle_ignores_barrier_and_measure_layers():
    # Every gate layer keeps both qubits busy; only the final barrier/measure layers look idle.
    # Those must not count as idle slots, so the pass falls back to a random position.
    qc = QuantumCircuit(2)
    qc.h([0, 1])
    qc.cx(0, 1)
    qc.measure_all()
    _, record = apply_trojan(qc, TrojanSpec(gate="z", mode="idle", seed=0))
    assert record["placement"] == "random-fallback"


def test_idle_samples_distinct_insertion_points():
    # q1 sits idle for two consecutive layers before the cx: both layers name the same slot
    # (q1, before its 2nd op). Barrier/measure layers and duplicate layers must not add
    # end-of-circuit candidates, so that slot is the only possible pick.
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.h(1)
    qc.h(0)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()
    for seed in range(20):
        _, record = apply_trojan(qc, TrojanSpec(gate="z", mode="idle", seed=seed))
        assert (record["qubit"], record["op_position"], record["placement"]) == (1, 1, "idle")


def test_random_position_indexes_real_operations(golden):
    n_real = _real_ops_per_wire(golden)
    for seed in range(50):
        _, record = apply_trojan(golden, TrojanSpec(gate="x", mode="random", seed=seed))
        assert 0 <= record["op_position"] <= n_real[record["qubit"]]


def test_backdoor_detector_is_a_soft_threshold():
    # For the z encoding (reps=2) undone at pi/2, a qubit returns to |0> with prob. sin^2(x).
    undo = z_feature_map(1, reps=2).assign_parameters([np.pi / 2]).decompose().inverse()
    for x in np.linspace(0, np.pi / 2, 9):
        enc = z_feature_map(1, reps=2).assign_parameters([x]).decompose()
        p0 = abs(Statevector(enc.compose(undo)).data[0]) ** 2
        assert np.isclose(p0, np.sin(x) ** 2, atol=1e-9)


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
