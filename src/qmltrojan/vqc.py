"""Golden variational quantum classifiers (VQCs): construction, training and compilation.

Model
-----
``|psi(x, theta)> = A(theta) U_phi(x) |0...0>`` where ``U_phi`` is an angle-encoding
feature map (``z_feature_map`` or ``zz_feature_map``) and ``A`` is a ``real_amplitudes``
ansatz with ``reps`` layers. The model output is the expectation of the global parity
observable ``Z^{(x)n}``, i.e. ``f(x) = P(even parity) - P(odd parity)`` in [-1, 1], and the
predicted class is ``1`` if ``f(x) > 0`` else ``0``. Because ``f`` is a function of the
measured bit-string distribution, trojan effects on the output distribution (TVD) and on
the predicted label are directly comparable.

Training uses qiskit-machine-learning (``EstimatorQNN`` + ``NeuralNetworkClassifier``)
with COBYLA on Aer's exact (statevector, ``default_precision=0``) estimator.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import real_amplitudes, z_feature_map, zz_feature_map
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer.primitives import EstimatorV2 as AerEstimator
from qiskit_machine_learning.algorithms import NeuralNetworkClassifier
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.optimizers import COBYLA

from qmltrojan.data import DatasetSplit
from qmltrojan.io_qasm import BASIS_GATES, align_inputs, wire_operations

ENCODINGS: tuple[str, ...] = ("z", "zz")


@dataclass(frozen=True)
class VQCSpec:
    """Hyper-parameters that identify one golden VQC."""

    dataset: str
    encoding: str
    ansatz_reps: int
    n_qubits: int
    seed: int
    feature_map_reps: int = 2

    @property
    def model_id(self) -> str:
        return f"{self.dataset}_{self.encoding}_r{self.ansatz_reps}_q{self.n_qubits}_s{self.seed}"

    @property
    def family(self) -> str:
        return f"vqc-{self.encoding}fm-realamp"


@dataclass
class TrainedVQC:
    """Result of training one golden VQC."""

    spec: VQCSpec
    weights: np.ndarray
    train_accuracy: float
    test_accuracy: float
    final_loss: float
    n_evaluations: int
    fit_seconds: float

    def to_record(self) -> dict:
        rec = asdict(self.spec)
        rec.update(
            model_id=self.spec.model_id,
            family=self.spec.family,
            train_accuracy=self.train_accuracy,
            test_accuracy=self.test_accuracy,
            final_loss=self.final_loss,
            n_evaluations=self.n_evaluations,
            fit_seconds=self.fit_seconds,
            weights=list(map(float, self.weights)),
        )
        return rec


def build_feature_map(encoding: str, n_qubits: int, reps: int = 2) -> QuantumCircuit:
    """Angle-encoding feature map with input parameters named ``x[i]``."""
    if encoding == "z":
        return z_feature_map(n_qubits, reps=reps, parameter_prefix="x")
    if encoding == "zz":
        return zz_feature_map(n_qubits, reps=reps, entanglement="linear", parameter_prefix="x")
    raise ValueError(f"Unknown encoding {encoding!r}; expected one of {ENCODINGS}")


def build_ansatz(n_qubits: int, reps: int) -> QuantumCircuit:
    """``real_amplitudes`` ansatz (RY layers + linear CX entanglers), params ``theta[j]``."""
    return real_amplitudes(n_qubits, reps=reps, entanglement="reverse_linear",
                           parameter_prefix="theta")


def build_template(spec: VQCSpec) -> tuple[QuantumCircuit, QuantumCircuit, QuantumCircuit]:
    """Return ``(feature_map, ansatz, feature_map ∘ ansatz)`` (no measurements)."""
    fmap = build_feature_map(spec.encoding, spec.n_qubits, spec.feature_map_reps)
    ansatz = build_ansatz(spec.n_qubits, spec.ansatz_reps)
    return fmap, ansatz, fmap.compose(ansatz)


def parity_observable(n_data: int, n_total: int | None = None) -> SparsePauliOp:
    """``Z`` on the first ``n_data`` qubits, identity on any extra (ancilla) qubits."""
    n_total = n_data if n_total is None else n_total
    return SparsePauliOp("I" * (n_total - n_data) + "Z" * n_data)


def make_estimator(seed: int = 0) -> AerEstimator:
    """Aer estimator returning exact expectation values (no shot noise)."""
    return AerEstimator(
        options={
            "default_precision": 0.0,
            "backend_options": {"method": "statevector", "seed_simulator": seed,
                                "max_parallel_threads": 1},
        }
    )


def train_vqc(spec: VQCSpec, data: DatasetSplit, maxiter: int = 150) -> TrainedVQC:
    """Train a golden VQC with COBYLA from a seeded random initial point."""
    if data.n_features != spec.n_qubits:
        raise ValueError("dataset features must equal the number of qubits")
    fmap, ansatz, circuit = build_template(spec)
    qnn = EstimatorQNN(
        circuit=circuit,
        observables=parity_observable(spec.n_qubits),
        input_params=list(fmap.parameters),
        weight_params=list(ansatz.parameters),
        estimator=make_estimator(spec.seed),
    )
    rng = np.random.default_rng(spec.seed)
    initial = rng.uniform(-np.pi, np.pi, ansatz.num_parameters)
    losses: list[float] = []
    clf = NeuralNetworkClassifier(
        qnn,
        optimizer=COBYLA(maxiter=maxiter),
        loss="squared_error",
        initial_point=initial,
        callback=lambda _w, loss: losses.append(float(loss)),
    )
    y_pm = 2 * data.y_train - 1  # {0,1} -> {-1,+1} to match the parity output range
    start = time.perf_counter()
    clf.fit(data.X_train, y_pm)
    elapsed = time.perf_counter() - start
    weights = np.asarray(clf.weights, dtype=float)

    golden = compile_golden(spec, weights)
    return TrainedVQC(
        spec=spec,
        weights=weights,
        train_accuracy=accuracy(golden, data.X_train, data.y_train),
        test_accuracy=accuracy(golden, data.X_test, data.y_test),
        final_loss=losses[-1] if losses else float("nan"),
        n_evaluations=len(losses),
        fit_seconds=elapsed,
    )


def compile_golden(spec: VQCSpec, weights: np.ndarray) -> QuantumCircuit:
    """Bind trained weights, add measurements and run the *clean* compilation flow.

    The clean flow unrolls to :data:`~qmltrojan.io_qasm.BASIS_GATES` at optimization
    level 0 (deterministic, no gate cancellation). Inputs remain symbolic. The metadata
    records, per qubit, how many leading operations belong to the feature map, so that
    insertion positions can be labelled as *encoding*, *interface* or *ansatz*.
    """
    fmap, ansatz, circuit = build_template(spec)
    bound = circuit.assign_parameters(dict(zip(ansatz.parameters, weights, strict=True)))
    bound.measure_all()
    compiled = transpile(bound, basis_gates=BASIS_GATES, optimization_level=0,
                         seed_transpiler=spec.seed)
    fmap_compiled = transpile(fmap, basis_gates=BASIS_GATES, optimization_level=0,
                              seed_transpiler=spec.seed)
    fm_ops = {q: len(ops) for q, ops in wire_operations(fmap_compiled).items()}
    compiled.name = spec.model_id
    compiled.metadata = {
        "model_id": spec.model_id,
        "family": spec.family,
        "n_data_qubits": spec.n_qubits,
        "fm_ops_per_qubit": fm_ops,
    }
    return compiled


def expectation_values(circuit: QuantumCircuit, X: np.ndarray, n_data: int | None = None,
                       seed: int = 0) -> np.ndarray:
    """Exact parity expectation ``<Z^n>`` on the data qubits for every row of ``X``."""
    unitary = circuit.remove_final_measurements(inplace=False)
    n_data = unitary.num_qubits if n_data is None else n_data
    obs = parity_observable(n_data, unitary.num_qubits)
    result = make_estimator(seed).run([(unitary, obs, align_inputs(unitary, X))]).result()
    return np.asarray(result[0].data.evs, dtype=float).reshape(-1)


def predict(circuit: QuantumCircuit, X: np.ndarray, n_data: int | None = None) -> np.ndarray:
    """Predicted labels in {0, 1} (class 1 iff the parity expectation is positive)."""
    return (expectation_values(circuit, X, n_data) > 0).astype(int)


def accuracy(circuit: QuantumCircuit, X: np.ndarray, y: np.ndarray,
             n_data: int | None = None) -> float:
    """Classification accuracy of a (compiled, possibly infected) VQC circuit."""
    return float(np.mean(predict(circuit, X, n_data) == np.asarray(y)))
