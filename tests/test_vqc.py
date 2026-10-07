"""VQC construction and output-format tests."""

from __future__ import annotations

import numpy as np

from qmltrojan.vqc import (
    ENCODINGS,
    build_feature_map,
    build_template,
    expectation_values,
    parity_observable,
    predict,
)


def test_golden_has_measurements_and_right_width(golden, spec):
    assert golden.num_qubits == spec.n_qubits
    assert golden.num_clbits == spec.n_qubits
    assert golden.count_ops().get("measure", 0) == spec.n_qubits


def test_golden_inputs_match_qubits(golden, spec):
    # After binding the trained weights, the only free parameters are the data inputs.
    assert golden.num_parameters == spec.n_qubits


def test_feature_map_input_count():
    for enc in ENCODINGS:
        fm = build_feature_map(enc, 3, reps=2)
        assert fm.num_parameters == 3  # one input per qubit regardless of reps


def test_template_composes_feature_map_and_ansatz(spec):
    fmap, ansatz, circuit = build_template(spec)
    assert circuit.num_parameters == fmap.num_parameters + ansatz.num_parameters


def test_parity_observable_shape():
    obs = parity_observable(2, 3)
    assert obs.num_qubits == 3
    assert str(obs.paulis[0]) == "IZZ"


def test_predictions_are_binary_and_deterministic(golden, data):
    p1 = predict(golden, data.X_test)
    p2 = predict(golden, data.X_test)
    assert set(np.unique(p1)).issubset({0, 1})
    assert np.array_equal(p1, p2)  # exact estimator is deterministic


def test_expectation_values_in_range(golden, data):
    ev = expectation_values(golden, data.X_test)
    assert ev.shape[0] == len(data.X_test)
    assert np.all(ev >= -1.0 - 1e-9) and np.all(ev <= 1.0 + 1e-9)
