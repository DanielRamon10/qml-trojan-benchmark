"""OpenQASM 3 export/parse, input-parameter handling and structural statistics.

Why OpenQASM 3? A *trained* VQC is still a parametric program: the trained weights are
fixed numbers, but the data features are run-time inputs. OpenQASM 2 cannot represent
unbound parameters (Qiskit raises ``QASM2ExportError``), whereas OpenQASM 3 declares them
as ``input float[64] _x_0_;``. Each exported file is therefore one complete, deployable
classifier, which is exactly the artifact a compromised compiler pass would emit.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit, qasm3
from qiskit.circuit import Parameter

#: Gate basis of the (clean) compilation flow. It is the target of every compiled circuit,
#: so golden and infected files share the same instruction vocabulary.
BASIS_GATES: list[str] = ["h", "x", "y", "z", "s", "sdg", "t", "tdg", "rx", "ry", "rz", "p", "cx"]

_INPUT_RE = re.compile(r"x\D*(\d+)")


def to_qasm3(circuit: QuantumCircuit) -> str:
    """Serialize a circuit to OpenQASM 3 text."""
    return qasm3.dumps(circuit)


def from_qasm3(text: str) -> QuantumCircuit:
    """Parse OpenQASM 3 text (requires the ``qiskit-qasm3-import`` package)."""
    return qasm3.loads(text)


def save_qasm(circuit: QuantumCircuit, path: str | Path) -> str:
    """Write ``circuit`` as OpenQASM 3 to ``path`` and return the text."""
    text = to_qasm3(circuit)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return text


def load_qasm(path: str | Path) -> QuantumCircuit:
    """Read an OpenQASM 3 file."""
    return from_qasm3(Path(path).read_text(encoding="utf-8"))


def sha256_text(text: str) -> str:
    """Hex SHA-256 digest of a text (used in the file inventory)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def input_index(param: Parameter) -> int:
    """Feature index of an input parameter, for both ``x[3]`` and ``_x_3_`` spellings."""
    match = _INPUT_RE.search(param.name)
    if match is None:
        raise ValueError(f"{param.name!r} is not an input parameter")
    return int(match.group(1))


def input_parameters(circuit: QuantumCircuit) -> list[Parameter]:
    """Input (feature) parameters of a circuit, in the order Qiskit binds them."""
    return list(circuit.parameters)


def align_inputs(circuit: QuantumCircuit, X: np.ndarray) -> np.ndarray:
    """Reorder the columns of ``X`` to match ``circuit.parameters``.

    Qiskit binds parameter arrays in the (sorted) order of ``circuit.parameters``; after a
    QASM round-trip the names change (``x[0]`` -> ``_x_0_``), so we always align by the
    feature index embedded in the name rather than assuming an order.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    order = [input_index(p) for p in circuit.parameters]
    if sorted(order) != list(range(X.shape[1])):
        raise ValueError(f"circuit inputs {order} do not match {X.shape[1]} features")
    return X[:, order]


def wire_operations(circuit: QuantumCircuit) -> dict[int, list[int]]:
    """Map each qubit index to the list of instruction indices acting on it (in order)."""
    wires: dict[int, list[int]] = {q: [] for q in range(circuit.num_qubits)}
    for idx, inst in enumerate(circuit.data):
        if inst.operation.name == "barrier":
            continue
        for qubit in inst.qubits:
            wires[circuit.find_bit(qubit).index].append(idx)
    return wires


def circuit_stats(circuit: QuantumCircuit) -> dict:
    """Structural statistics in the vocabulary of the trojan-dataset paper.

    ``depth`` and ``size`` follow Qiskit's defaults (directives such as barriers are
    excluded; measurements are included).
    """
    return {
        "depth": int(circuit.depth()),
        "size": int(circuit.size()),
        "n_qubits": int(circuit.num_qubits),
        "n_clbits": int(circuit.num_clbits),
        "count_ops": json.dumps(dict(sorted(circuit.count_ops().items()))),
    }
