"""OpenQASM 3 round-trip and structural-statistics tests."""

from __future__ import annotations

import numpy as np

from qmltrojan.io_qasm import (
    align_inputs,
    circuit_stats,
    from_qasm3,
    input_index,
    save_qasm,
    sha256_text,
    to_qasm3,
)
from qmltrojan.vqc import accuracy


def test_qasm_roundtrip_preserves_behaviour(golden, data):
    text = to_qasm3(golden)
    restored = from_qasm3(text)
    assert accuracy(restored, data.X_test, data.y_test) == accuracy(golden, data.X_test,
                                                                    data.y_test)


def test_qasm_file_roundtrip(tmp_path, golden):
    path = tmp_path / "c.qasm"
    text = save_qasm(golden, path)
    assert path.read_text(encoding="utf-8") == text
    restored = from_qasm3(path.read_text(encoding="utf-8"))
    assert restored.num_qubits == golden.num_qubits


def test_input_index_parses_both_spellings():
    # The two spellings Qiskit uses for an input before/after a QASM round-trip.
    from qiskit.circuit import Parameter

    assert input_index(Parameter("x[2]")) == 2
    assert input_index(Parameter("_x_5_")) == 5


def test_align_inputs_reorders_by_feature_index(golden):
    X = np.arange(golden.num_qubits, dtype=float).reshape(1, -1)
    aligned = align_inputs(golden, X)
    order = [input_index(p) for p in golden.parameters]
    assert np.array_equal(aligned[0], X[0, order])


def test_circuit_stats_keys(golden):
    stats = circuit_stats(golden)
    assert {"depth", "size", "n_qubits", "n_clbits", "count_ops"} <= set(stats)
    assert stats["n_qubits"] == golden.num_qubits


def test_sha256_is_stable(golden):
    text = to_qasm3(golden)
    assert sha256_text(text) == sha256_text(text)
    assert len(sha256_text(text)) == 64
