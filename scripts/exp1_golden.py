"""Experiment 1 -- train every golden VQC and record clean classification accuracy.

Outputs (under ``results/<mode>/``):
* ``golden/<model_id>.qasm``   -- each trained VQC as OpenQASM 3
* ``golden_models.json``       -- specs, trained weights and accuracies (for later stages)
* ``golden_accuracy.csv``      -- clean train/test accuracy per model
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import itertools

import pandas as pd
from _common import (
    Timer,
    get_logger,
    load_config,
    mode_from_args,
    parse_args,
    resolve_paths,
    save_json,
)

from qmltrojan.data import load_dataset
from qmltrojan.io_qasm import save_qasm
from qmltrojan.vqc import VQCSpec, compile_golden, train_vqc

LOG = get_logger("exp1")


def iter_specs(cfg: dict):
    for ds, enc, reps, nq, seed in itertools.product(
        cfg["datasets"], cfg["encodings"], cfg["ansatz_reps"], cfg["n_qubits"], cfg["seeds"]
    ):
        yield VQCSpec(dataset=ds, encoding=enc, ansatz_reps=reps, n_qubits=nq, seed=seed,
                      feature_map_reps=cfg["feature_map_reps"])


def main() -> None:
    args = parse_args(__doc__)
    mode = mode_from_args(args)
    cfg = load_config(mode)
    paths = resolve_paths(mode)
    specs = list(iter_specs(cfg))
    LOG.info("Training %d golden VQCs (mode=%s)", len(specs), mode)

    models, rows = {}, []
    with Timer(LOG, "exp1 total"):
        for i, spec in enumerate(specs, 1):
            data = load_dataset(spec.dataset, spec.n_qubits, spec.seed,
                                max_train=cfg["max_train"])
            trained = train_vqc(spec, data, maxiter=cfg["maxiter"])
            golden = compile_golden(spec, trained.weights)
            save_qasm(golden, paths.golden / f"{spec.model_id}.qasm")
            models[spec.model_id] = trained.to_record()
            rows.append({
                "model_id": spec.model_id, "dataset": spec.dataset, "encoding": spec.encoding,
                "ansatz_reps": spec.ansatz_reps, "n_qubits": spec.n_qubits, "seed": spec.seed,
                "train_accuracy": trained.train_accuracy, "test_accuracy": trained.test_accuracy,
                "final_loss": trained.final_loss, "fit_seconds": trained.fit_seconds,
            })
            if i % 10 == 0 or i == len(specs):
                LOG.info("  %d/%d  %s  test=%.3f", i, len(specs), spec.model_id,
                         trained.test_accuracy)

    save_json(models, paths.root / "golden_models.json")
    df = pd.DataFrame(rows)
    df.to_csv(paths.root / "golden_accuracy.csv", index=False)
    LOG.info("Mean clean test accuracy: %.3f (n=%d). Wrote golden_models.json + "
             "golden_accuracy.csv", df["test_accuracy"].mean(), len(df))


if __name__ == "__main__":
    main()
