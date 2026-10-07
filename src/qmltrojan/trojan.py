"""Malicious transpiler passes: single-gate insertion and a triggered backdoor.

Adversary model (software supply chain)
---------------------------------------
A once-trusted optimization dependency (a Qiskit transpiler pass) is compromised and, during
compilation, injects operations into an already-trained VQC. The attacker controls neither
the algorithm's source, the cloud provider, nor the hardware. The defender holds both the
*golden* (clean) and *infected* compiled artifacts and their measurements. We model the
attack as a real :class:`~qiskit.transpiler.TransformationPass` so that the infected circuit
is produced by the compilation flow itself, exactly as it would be in the field.

Two attack families are implemented:

* :class:`SingleGateInsertionPass` -- inserts **one** gate from {H, T, X, Y, Z} at a
  position chosen by one of three modes (``random``, ``idle`` layer of the DAG, or
  ``controlled``). This mirrors the sensitivity-analysis dataset of the quantum-trojan paper.
* :func:`insert_backdoor` -- a QAML-specific triggered backdoor that flips the classifier's
  output only for inputs matching a trigger pattern, while leaving other inputs (nearly)
  untouched.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit import Gate
from qiskit.circuit.library import HGate, TGate, XGate, YGate, ZGate, z_feature_map
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.transpiler import PassManager, TransformationPass

from qmltrojan.io_qasm import circuit_stats, wire_operations
from qmltrojan.vqc import VQCSpec, build_ansatz, build_feature_map

#: Single-qubit gates the attacker may insert, keyed by their QASM name.
INSERTABLE_GATES: dict[str, type[Gate]] = {
    "h": HGate,
    "t": TGate,
    "x": XGate,
    "y": YGate,
    "z": ZGate,
}

#: Insertion modes (how the attacker chooses *where* to place the gate).
INSERTION_MODES: tuple[str, ...] = ("random", "idle", "controlled")


@dataclass
class TrojanSpec:
    """Specification of one single-gate insertion."""

    gate: str
    mode: str
    seed: int
    #: For ``controlled`` mode: fraction (0..1) of the wire at which to insert. ``None``
    #: means "the encoding/ansatz interface" (just after the feature-map operations).
    position: float | None = None
    #: For ``controlled`` mode: target qubit. ``None`` means qubit 0.
    qubit: int | None = None

    def __post_init__(self) -> None:
        if self.gate not in INSERTABLE_GATES:
            raise ValueError(f"gate {self.gate!r} not in {sorted(INSERTABLE_GATES)}")
        if self.mode not in INSERTION_MODES:
            raise ValueError(f"mode {self.mode!r} not in {INSERTION_MODES}")


class SingleGateInsertionPass(TransformationPass):
    """A compromised optimization pass that injects exactly one gate.

    The pass records, in :attr:`record`, the concrete (gate, mode, qubit, position) it chose,
    so that every infected artifact carries structured provenance metadata.
    """

    def __init__(self, spec: TrojanSpec, fm_ops_per_qubit: dict[int, int] | None = None):
        super().__init__()
        self.spec = spec
        self.fm_ops_per_qubit = fm_ops_per_qubit or {}
        self.record: dict | None = None

    # -- placement helpers -------------------------------------------------------------
    def _insert_in_circuit(self, circuit: QuantumCircuit, qubit: int,
                           op_position: int) -> QuantumCircuit:
        """Return a copy of ``circuit`` with the trojan gate inserted on ``qubit``.

        ``op_position`` is an index into the list of (non-barrier, non-measure) operations
        on that wire; the gate is placed *before* that operation (or last, if out of range).
        """
        gate = INSERTABLE_GATES[self.spec.gate]()
        out = circuit.copy_empty_like()
        wire_counter = 0
        inserted = False
        target = circuit.qubits[qubit]
        # Index, on the target wire, of the operation we insert before.
        for inst in circuit.data:
            touches_target = target in inst.qubits
            real = inst.operation.name not in ("barrier", "measure")
            if touches_target and real and wire_counter == op_position and not inserted:
                out.append(gate, [target])
                inserted = True
            out.append(inst)
            if touches_target and real:
                wire_counter += 1
        if not inserted:  # position beyond the last op: append before measurements
            out = _append_before_measure(circuit, gate, target)
        return out

    def run(self, dag):  # noqa: D102 (TransformationPass API)
        circuit = dag_to_circuit(dag)
        rng = np.random.default_rng(self.spec.seed)
        n_qubits = circuit.num_qubits
        wires = wire_operations(circuit)

        if self.spec.mode == "random":
            qubit = int(rng.integers(n_qubits))
            n_ops = len(wires[qubit])
            op_position = int(rng.integers(n_ops + 1)) if n_ops else 0
            place = "random"
        elif self.spec.mode == "idle":
            qubit, op_position, place = self._idle_slot(circuit, rng)
        else:  # controlled
            qubit = self.spec.qubit if self.spec.qubit is not None else 0
            if self.spec.position is None:
                op_position = int(self.fm_ops_per_qubit.get(qubit, 0))
                place = "interface"
            else:
                op_position = int(round(self.spec.position * len(wires[qubit])))
                place = "fractional"

        new_circuit = self._insert_in_circuit(circuit, qubit, op_position)
        self.record = {
            "gate": self.spec.gate,
            "mode": self.spec.mode,
            "qubit": int(qubit),
            "op_position": int(op_position),
            "placement": place,
        }
        new = circuit_to_dag(new_circuit)
        return new

    def _idle_slot(self, circuit: QuantumCircuit, rng) -> tuple[int, int, str]:
        """Find a (qubit, op_position) where the DAG layer leaves the qubit idle.

        We scan DAG layers; in each layer the qubits not touched by any op are "idle" at
        that depth. We pick one uniformly. If none exists (fully dense circuit) we fall
        back to a random wire position.
        """
        dag = circuit_to_dag(circuit)
        candidates: list[tuple[int, int]] = []
        # ops seen per wire up to the current layer = op_position to insert at.
        seen = {q: 0 for q in range(circuit.num_qubits)}
        for layer in dag.layers():
            active = set()
            for node in layer["graph"].op_nodes():
                if node.op.name in ("barrier", "measure"):
                    continue
                for q in node.qargs:
                    active.add(circuit.find_bit(q).index)
            for q in range(circuit.num_qubits):
                if q not in active:
                    candidates.append((q, seen[q]))
            for q in active:
                seen[q] += 1
        if not candidates:
            q = int(rng.integers(circuit.num_qubits))
            return q, int(rng.integers(len(wire_operations(circuit)[q]) + 1)), "random-fallback"
        q, pos = candidates[int(rng.integers(len(candidates)))]
        return q, pos, "idle"


def _append_before_measure(circuit: QuantumCircuit, gate: Gate,
                           target) -> QuantumCircuit:
    """Insert ``gate`` on ``target`` just before the final measurements/barrier."""
    out = circuit.copy_empty_like()
    tail = list(circuit.data)
    split = len(tail)
    for i in range(len(tail) - 1, -1, -1):
        if tail[i].operation.name in ("measure", "barrier"):
            split = i
        else:
            break
    for inst in tail[:split]:
        out.append(inst)
    out.append(gate, [target])
    for inst in tail[split:]:
        out.append(inst)
    return out


def apply_trojan(golden: QuantumCircuit, spec: TrojanSpec) -> tuple[QuantumCircuit, dict]:
    """Run a :class:`SingleGateInsertionPass` on a golden circuit.

    Returns the infected circuit and a metadata record combining the attack provenance with
    before/after structural statistics.
    """
    before = circuit_stats(golden)
    fm_ops = (golden.metadata or {}).get("fm_ops_per_qubit", {})
    fm_ops = {int(k): int(v) for k, v in fm_ops.items()}
    pass_ = SingleGateInsertionPass(spec, fm_ops_per_qubit=fm_ops)
    infected = PassManager([pass_]).run(golden)
    infected.metadata = dict(golden.metadata or {})
    after = circuit_stats(infected)

    record = dict(asdict(spec))
    record.update(pass_.record or {})
    record.update(
        depth_before=before["depth"],
        depth_after=after["depth"],
        depth_delta=after["depth"] - before["depth"],
        size_before=before["size"],
        size_after=after["size"],
        size_delta=after["size"] - before["size"],
        n_qubits=before["n_qubits"],
        n_clbits=before["n_clbits"],
        family=(golden.metadata or {}).get("family", "unknown"),
        model_id=(golden.metadata or {}).get("model_id", golden.name),
    )
    return infected, record


# --------------------------------------------------------------------------------------
# Triggered backdoor (QAML-specific)
# --------------------------------------------------------------------------------------
@dataclass
class BackdoorSpec:
    """Specification of a triggered backdoor (stealthy conditional trojan).

    The trigger is "the first ``k_trigger`` data qubits all encode a feature at the top of the
    encoding range ``angle_hi``". A compromised pass inserts, *between the feature map and the
    ansatz*, a constant detector: on each trigger qubit it applies the inverse single-qubit
    encoding evaluated at the fixed trigger angle, ``U_enc(angle_hi)^dagger``. An input encoded
    at exactly ``angle_hi`` is sent back to ``|0>``; a multi-controlled-X (controls-on-zero via
    surrounding ``X`` gates) then sets an ancilla, and the encoding is recomputed so the data
    qubits are undisturbed. After the ansatz, the ancilla is measured and a classically
    controlled ``X`` on qubit 0 flips the parity -- and hence the predicted class -- only when
    the trigger fired. This is the conditional-trojan construction of John et al. (2025).

    Because ``U_enc`` is a single-qubit (product) operation, the detector is *exact* for the
    ``z`` encoding. For the entangling ``zz`` encoding the per-qubit uncompute is only
    approximate, so the attack degrades -- a limitation we measure and report honestly.
    Selectivity grows with ``k_trigger``: requiring several qubits at ``angle_hi`` at once makes
    the trigger rarer, reducing leakage onto clean inputs.
    """

    k_trigger: int = 2
    angle_hi: float = float(np.pi / 2)
    seed: int = 0


def build_backdoor(spec: VQCSpec, weights: np.ndarray,
                   bspec: BackdoorSpec) -> tuple[QuantumCircuit, QuantumCircuit, dict]:
    """Build a triggered-backdoor circuit and its matching golden circuit from a VQC spec.

    Returns ``(backdoor, golden, record)``. Both circuits measure the ``n_data`` data qubits
    into a ``meas`` register; the backdoor additionally uses one ancilla and a one-bit ``trig``
    register for the mid-circuit trigger measurement. The backdoor contains a classically
    controlled gate, so it must be evaluated by *sampling* (see
    :func:`~qmltrojan.sensitivity.predict_sampled`), not by the exact estimator.
    """
    n_data = spec.n_qubits
    k = min(bspec.k_trigger, n_data)
    fmap = build_feature_map(spec.encoding, n_data, spec.feature_map_reps)
    ansatz = build_ansatz(n_data, spec.ansatz_reps)
    ansatz = ansatz.assign_parameters(dict(zip(ansatz.parameters, weights, strict=True)))

    # Constant detector gate: inverse single-qubit encoding at the fixed trigger angle.
    enc1 = z_feature_map(1, reps=spec.feature_map_reps).assign_parameters([bspec.angle_hi])
    enc1 = enc1.decompose()
    undo, redo = enc1.inverse(), enc1

    qr = QuantumRegister(n_data + 1, "q")
    meas = ClassicalRegister(n_data, "meas")
    trig = ClassicalRegister(1, "trig")
    ancilla = n_data
    bd = QuantumCircuit(qr, meas, trig, name=f"{spec.model_id}_backdoor_k{k}")
    bd.compose(fmap, range(n_data), inplace=True)
    bd.barrier()
    for q in range(k):
        bd.compose(undo, [q], inplace=True)
    bd.x(range(k))
    if k == 1:
        bd.cx(0, ancilla)
    else:
        bd.mcx(list(range(k)), ancilla)
    bd.x(range(k))
    for q in range(k):
        bd.compose(redo, [q], inplace=True)
    bd.barrier()
    bd.compose(ansatz, range(n_data), inplace=True)
    bd.measure(ancilla, trig[0])
    with bd.if_test((trig[0], 1)):
        bd.x(0)
    bd.measure(range(n_data), meas)
    bd.metadata = {"model_id": spec.model_id, "family": spec.family + "-backdoor",
                   "n_data_qubits": n_data, "backdoor_k": k, "angle_hi": bspec.angle_hi}

    golden = QuantumCircuit(n_data, n_data, name=spec.model_id)
    golden.compose(fmap, inplace=True)
    golden.compose(ansatz, inplace=True)
    golden.measure(range(n_data), range(n_data))
    golden.metadata = {"model_id": spec.model_id, "family": spec.family, "n_data_qubits": n_data}

    record = {
        "model_id": spec.model_id, "encoding": spec.encoding, "k_trigger": k,
        "angle_hi": bspec.angle_hi, "n_data_qubits": n_data,
        "depth_golden": golden.depth(), "depth_backdoor": bd.depth(),
        "size_golden": golden.size(), "size_backdoor": bd.size(),
    }
    return bd, golden, record


def make_trigger_inputs(n_data: int, k_trigger: int, angle_hi: float, n_samples: int,
                        base: np.ndarray | None = None, seed: int = 0) -> np.ndarray:
    """Build inputs that satisfy the trigger: first ``k`` features set to ``angle_hi``.

    If ``base`` is given, its rows are copied and only the trigger features are overwritten,
    producing realistic triggered versions of genuine test points.
    """
    rng = np.random.default_rng(seed)
    if base is None:
        X = rng.uniform(0, angle_hi, size=(n_samples, n_data))
    else:
        idx = rng.integers(0, len(base), size=n_samples)
        X = np.asarray(base, dtype=float)[idx].copy()
    X[:, :k_trigger] = angle_hi
    return X
