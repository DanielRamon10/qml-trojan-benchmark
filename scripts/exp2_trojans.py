"""Experiment 2 -- generate infected variants and analyze sensitivity + accuracy impact.

For every golden VQC we insert ``variants_per_model`` single-gate trojans (one per gate x mode
combination). For each variant we measure, on a fixed set of test inputs, the output-distribution
shift (TVD/BC/BD vs a golden-vs-golden sampling baseline) and the exact classification accuracy
(and its drop from golden). A RandomForest + SHAP analysis then ranks which structural attributes
drive the mean TVD and the accuracy drop.

Outputs (under ``results/<mode>/``):
* ``infected/<...>.qasm``   -- each infected circuit
* ``metadata.csv``          -- per-variant provenance + structural stats
* ``sensitivity.csv``       -- per-variant TVD/BC/BD + clean/infected accuracy + drop
* ``inventory.csv``         -- every QASM file with size and SHA-256
* ``errors.csv``            -- any variant that failed (empty if all succeeded)
* ``shap_tvd.csv`` / ``shap_accuracy_drop.csv`` -- SHAP importances
* figures: ``tvd_by_gate.png``, ``tvd_by_mode.png``, ``tvd_by_nqubits.png``,
  ``accdrop_by_gate.png``, ``shap_tvd.png``, ``shap_accuracy_drop.png``
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import traceback

import numpy as np
import pandas as pd
from _common import (
    Timer,
    get_logger,
    load_config,
    load_json,
    mode_from_args,
    parse_args,
    resolve_paths,
)

from qmltrojan.data import load_dataset
from qmltrojan.explain import explain_target
from qmltrojan.io_qasm import load_qasm, save_qasm, sha256_text
from qmltrojan.plotting import bar_importance, violin_strip_by
from qmltrojan.sensitivity import distribution_metrics, golden_count_sets
from qmltrojan.trojan import TrojanSpec, apply_trojan
from qmltrojan.vqc import VQCSpec, accuracy, compile_golden

LOG = get_logger("exp2")


def variant_specs(cfg: dict, base_seed: int) -> list[TrojanSpec]:
    specs = []
    for g_i, gate in enumerate(cfg["gates"]):
        for m_i, mode in enumerate(cfg["modes"]):
            specs.append(TrojanSpec(gate=gate, mode=mode, seed=base_seed + g_i * 10 + m_i))
    return specs


def main() -> None:
    args = parse_args(__doc__)
    mode = mode_from_args(args)
    cfg = load_config(mode)
    paths = resolve_paths(mode)
    models = load_json(paths.root / "golden_models.json")
    LOG.info("Infecting %d golden models x %d variants", len(models), cfg["variants_per_model"])

    meta_rows, sens_rows, err_rows = [], [], []
    n_inputs = cfg["n_eval_inputs"]

    with Timer(LOG, "exp2 total"):
        for mi, (model_id, rec) in enumerate(models.items(), 1):
            spec = VQCSpec(dataset=rec["dataset"], encoding=rec["encoding"],
                           ansatz_reps=rec["ansatz_reps"], n_qubits=rec["n_qubits"],
                           seed=rec["seed"], feature_map_reps=rec["feature_map_reps"])
            weights = np.asarray(rec["weights"], dtype=float)
            golden = compile_golden(spec, weights)
            data = load_dataset(spec.dataset, spec.n_qubits, spec.seed, max_train=cfg["max_train"])
            X_eval = data.X_test[:n_inputs]
            n_data = spec.n_qubits

            acc_clean = accuracy(golden, data.X_test, data.y_test)
            # Pre-simulate the golden circuit once; reuse across all variants of this model.
            gsets = golden_count_sets(golden, X_eval, cfg["shots"], cfg["sim_reps"],
                                      seed=spec.seed, n_data=n_data)

            for ts in variant_specs(cfg, base_seed=spec.seed * 1000):
                try:
                    infected, record = apply_trojan(golden, ts)
                    fname = f"{model_id}__{ts.gate}_{ts.mode}_s{ts.seed}.qasm"
                    save_qasm(infected, paths.infected / fname)
                    # QASM round-trip guards that the serialized artifact is faithful.
                    acc_inf = accuracy(load_qasm(paths.infected / fname), data.X_test,
                                       data.y_test)

                    record.update(model_id=model_id, dataset=spec.dataset,
                                  encoding=spec.encoding, ansatz_reps=spec.ansatz_reps,
                                  qasm_file=fname)
                    meta_rows.append(record)

                    met = distribution_metrics(golden, infected, X_eval, shots=cfg["shots"],
                                               reps=cfg["sim_reps"], seed=spec.seed,
                                               n_data=n_data, golden_sets=gsets)
                    met.update(model_id=model_id, qasm_file=fname, gate=ts.gate, mode=ts.mode,
                               placement=record["placement"], qubit=record["qubit"],
                               op_position=record["op_position"], encoding=spec.encoding,
                               n_qubits=spec.n_qubits, ansatz_reps=spec.ansatz_reps,
                               family=record["family"], depth_before=record["depth_before"],
                               depth_delta=record["depth_delta"], size_before=record["size_before"],
                               size_delta=record["size_delta"], dataset=spec.dataset,
                               acc_clean=acc_clean, acc_infected=acc_inf,
                               accuracy_drop=acc_clean - acc_inf)
                    sens_rows.append(met)
                except Exception as exc:  # noqa: BLE001 - record and continue
                    err_rows.append({"model_id": model_id, "gate": ts.gate, "mode": ts.mode,
                                     "error": repr(exc), "trace": traceback.format_exc()})
                    LOG.warning("variant failed %s %s/%s: %s", model_id, ts.gate, ts.mode, exc)
            if mi % 10 == 0 or mi == len(models):
                LOG.info("  model %d/%d done (%d variants, %d errors)", mi, len(models),
                         len(sens_rows), len(err_rows))

    meta_df = pd.DataFrame(meta_rows)
    sens_df = pd.DataFrame(sens_rows)
    meta_df.to_csv(paths.root / "metadata.csv", index=False)
    sens_df.to_csv(paths.root / "sensitivity.csv", index=False)
    pd.DataFrame(err_rows).to_csv(paths.root / "errors.csv", index=False)
    _write_inventory(paths)
    LOG.info("TVD mean=%.3f (baseline %.3f). Mean accuracy drop=%.3f",
             sens_df["tvd_mean"].mean(), sens_df["tvd_baseline_mean"].mean(),
             sens_df["accuracy_drop"].mean())

    _figures(sens_df, paths)
    _shap(sens_df, paths)
    LOG.info("exp2 complete: %d variants, %d errors", len(sens_df), len(err_rows))


def _write_inventory(paths) -> None:
    rows = []
    for kind, folder in (("golden", paths.golden), ("infected", paths.infected),
                         ("backdoor", paths.backdoor)):
        for f in sorted(folder.glob("*.qasm")):
            text = f.read_text(encoding="utf-8")
            rows.append({"kind": kind, "file": f.name, "bytes": len(text.encode("utf-8")),
                         "sha256": sha256_text(text)})
    pd.DataFrame(rows).to_csv(paths.root / "inventory.csv", index=False)


def _figures(sens_df: pd.DataFrame, paths) -> None:
    base = float(sens_df["tvd_baseline_mean"].mean())
    violin_strip_by(sens_df, "tvd_mean", "gate", paths.figures / "tvd_by_gate.png",
                    "Output-distribution shift (TVD) by inserted gate", baseline=base)
    violin_strip_by(sens_df, "tvd_mean", "mode", paths.figures / "tvd_by_mode.png",
                    "Output-distribution shift (TVD) by insertion mode", baseline=base)
    sens_df = sens_df.assign(n_qubits_str=sens_df["n_qubits"].astype(str))
    violin_strip_by(sens_df, "tvd_mean", "n_qubits_str", paths.figures / "tvd_by_nqubits.png",
                    "Output-distribution shift (TVD) by number of qubits", baseline=base)
    violin_strip_by(sens_df, "accuracy_drop", "gate", paths.figures / "accdrop_by_gate.png",
                    "Classification accuracy drop by inserted gate")


def _shap(sens_df: pd.DataFrame, paths) -> None:
    for target, tag, title in (
        ("tvd_mean", "tvd", "SHAP importance for mean TVD"),
        ("accuracy_drop", "accuracy_drop", "SHAP importance for accuracy drop"),
    ):
        try:
            importance, _, _ = explain_target(sens_df, target)
            importance.to_csv(paths.root / f"shap_{tag}.csv", index=False)
            bar_importance(importance, paths.figures / f"shap_{tag}.png", title)
        except Exception as exc:  # noqa: BLE001
            LOG.warning("SHAP for %s failed: %s", target, exc)


if __name__ == "__main__":
    main()
