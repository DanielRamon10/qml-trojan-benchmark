"""Output-distribution sensitivity metrics and classification metrics.

For a golden circuit G and an infected circuit I evaluated on the same input x, we compare
their measured bit-string distributions ``p = P_G(.|x)`` and ``q = P_I(.|x)`` with three
metrics from the quantum-trojan sensitivity literature:

* **TVD** (total variation distance): ``0.5 * sum_i |p_i - q_i|`` in ``[0, 1]``.
* **BC** (Bhattacharyya coefficient): ``sum_i sqrt(p_i q_i)`` in ``[0, 1]`` (1 = identical).
* **BD** (Bhattacharyya distance): ``-ln(BC)`` in ``[0, inf)``.

Because measurements use a finite number of shots, two runs of the *same* circuit already differ
by sampling noise. We therefore also report a **golden-vs-golden baseline** (same circuit, two
independent shot sets): a trojan's TVD is only meaningful above this floor.

The classification metrics (clean accuracy, accuracy drop, and -- for the backdoor -- attack
success rate) use the exact estimator for single-gate trojans and sampling for the backdoor
(which contains a mid-circuit measurement and a classically controlled gate).
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

from qmltrojan.io_qasm import align_inputs

EPS = 1e-12


def _probs(counts: dict[str, int], n_outcomes: int, index: dict[str, int]) -> np.ndarray:
    vec = np.zeros(n_outcomes)
    total = sum(counts.values())
    for bits, c in counts.items():
        vec[index[bits]] = c
    return vec / max(total, 1)


def tvd(p: np.ndarray, q: np.ndarray) -> float:
    """Total variation distance between two probability vectors."""
    return float(0.5 * np.abs(p - q).sum())


def bhattacharyya_coefficient(p: np.ndarray, q: np.ndarray) -> float:
    """Bhattacharyya coefficient (overlap) between two probability vectors."""
    return float(np.sqrt(np.clip(p, 0, None) * np.clip(q, 0, None)).sum())


def bhattacharyya_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Bhattacharyya distance ``-ln(BC)`` (clipped to avoid ``log(0)``)."""
    return float(-np.log(max(bhattacharyya_coefficient(p, q), EPS)))


def _data_marginal_counts(counts: dict[str, int], n_data: int) -> dict[str, int]:
    """Reduce multi-register counts to the ``n_data``-bit data (``meas``) register.

    Aer joins register outcomes with spaces (most-significant register first). We keep the
    token whose length equals ``n_data`` and marginalize over any ancilla/trigger registers.
    """
    out: dict[str, int] = {}
    for bits, c in counts.items():
        tokens = bits.split()
        data = next(t for t in tokens if len(t) == n_data) if len(tokens) > 1 else bits
        out[data] = out.get(data, 0) + c
    return out


def simulate_counts(circuit: QuantumCircuit, X: np.ndarray, shots: int, seed: int,
                    n_data: int | None = None) -> list[dict[str, int]]:
    """Run ``circuit`` on each row of ``X`` and return data-register counts per input."""
    sim = AerSimulator(seed_simulator=seed)
    compiled = transpile(circuit, sim, optimization_level=0)
    n_data = circuit.num_clbits if n_data is None else n_data
    Xa = align_inputs(circuit, X)
    bound = [compiled.assign_parameters(x) for x in Xa]
    result = sim.run(bound, shots=shots, seed_simulator=seed).result()
    return [_data_marginal_counts(result.get_counts(i), n_data) for i in range(len(Xa))]


def distribution_metrics(golden: QuantumCircuit, infected: QuantumCircuit, X: np.ndarray,
                         shots: int = 512, reps: int = 15, seed: int = 0,
                         n_data: int | None = None) -> dict:
    """Mean/std TVD, BC, BD between golden and infected over ``reps`` shot sets.

    A golden-vs-golden baseline (two independent shot sets of the golden circuit) is computed
    with the same budget so the trojan signal can be read against the sampling floor.
    """
    n_data = golden.num_clbits if n_data is None else n_data
    n_out = 2 ** n_data
    index = {format(i, f"0{n_data}b"): i for i in range(n_out)}
    rng = np.random.default_rng(seed)

    tvds, bcs, bds, base_tvds = [], [], [], []
    for r in range(reps):
        s1, s2, s3 = (int(x) for x in rng.integers(0, 2**31, size=3))
        g1 = simulate_counts(golden, X, shots, s1, n_data)
        g2 = simulate_counts(golden, X, shots, s2, n_data)
        inf = simulate_counts(infected, X, shots, s3, n_data)
        for ci_g1, ci_g2, ci_inf in zip(g1, g2, inf, strict=True):
            p = _probs(ci_g1, n_out, index)
            q = _probs(ci_inf, n_out, index)
            pb = _probs(ci_g2, n_out, index)
            tvds.append(tvd(p, q))
            bcs.append(bhattacharyya_coefficient(p, q))
            bds.append(bhattacharyya_distance(p, q))
            base_tvds.append(tvd(p, pb))

    def ms(a: list[float]) -> tuple[float, float]:
        arr = np.asarray(a)
        return float(arr.mean()), float(arr.std())

    tvd_m, tvd_s = ms(tvds)
    bc_m, bc_s = ms(bcs)
    bd_m, bd_s = ms(bds)
    base_m, base_s = ms(base_tvds)
    return {
        "tvd_mean": tvd_m, "tvd_std": tvd_s,
        "bc_mean": bc_m, "bc_std": bc_s,
        "bd_mean": bd_m, "bd_std": bd_s,
        "tvd_baseline_mean": base_m, "tvd_baseline_std": base_s,
        "shots": shots, "reps": reps, "n_inputs": int(np.atleast_2d(X).shape[0]),
    }


def predict_sampled(circuit: QuantumCircuit, X: np.ndarray, n_data: int, shots: int = 2048,
                    seed: int = 0) -> np.ndarray:
    """Predicted labels from sampled parity (needed for circuits with conditional gates)."""
    counts = simulate_counts(circuit, X, shots, seed, n_data)
    preds = []
    for c in counts:
        exp = 0
        for bits, cnt in c.items():
            exp += (1 if bits.count("1") % 2 == 0 else -1) * cnt
        preds.append(1 if exp > 0 else 0)
    return np.asarray(preds, dtype=int)


def sampled_accuracy(circuit: QuantumCircuit, X: np.ndarray, y: np.ndarray, n_data: int,
                     shots: int = 2048, seed: int = 0) -> float:
    """Accuracy of a (possibly conditional) circuit via sampling."""
    return float(np.mean(predict_sampled(circuit, X, n_data, shots, seed) == np.asarray(y)))
