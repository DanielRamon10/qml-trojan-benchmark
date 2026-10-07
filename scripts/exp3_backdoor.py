"""Experiment 3 -- triggered backdoor: attack success rate vs clean accuracy.

For selected 4-qubit models we build a conditional backdoor for each trigger width ``k`` and
measure (by sampling): clean accuracy of the golden model, clean accuracy of the backdoored model
(should stay close to golden), and the attack success rate (ASR, fraction of triggered inputs whose
predicted label flips relative to golden). We report these honestly -- a strong ASR that also wrecks
clean accuracy is not a useful backdoor.

Outputs (under ``results/<mode>/``):
* ``backdoor/<model_id>_k<k>.qasm`` -- each backdoored circuit (OpenQASM 3)
* ``backdoor.csv``                  -- ASR, clean accuracies and accuracy drop per (model, k)
* figures: ``backdoor_asr_vs_clean.png``, ``backdoor_asr_by_k.png``
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

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
from qmltrojan.io_qasm import save_qasm
from qmltrojan.plotting import grouped_bars, scatter_asr_vs_clean
from qmltrojan.sensitivity import predict_sampled, sampled_accuracy
from qmltrojan.trojan import BackdoorSpec, build_backdoor, make_trigger_inputs
from qmltrojan.vqc import VQCSpec

LOG = get_logger("exp3")


def main() -> None:
    args = parse_args(__doc__)
    mode = mode_from_args(args)
    cfg = load_config(mode)
    paths = resolve_paths(mode)
    models = load_json(paths.root / "golden_models.json")

    targets = [(mid, r) for mid, r in models.items()
               if r["dataset"] in cfg["backdoor_datasets"] and r["n_qubits"] == 4
               and r["encoding"] == "z"]
    LOG.info("Backdoor on %d z-encoding 4-qubit models x k in %s", len(targets), cfg["backdoor_k"])

    rows = []
    shots = cfg["backdoor_shots"]
    n_eval = cfg["backdoor_eval_inputs"]
    angle_hi = cfg["backdoor_angle_hi"]

    with Timer(LOG, "exp3 total"):
        for mid, rec in targets:
            spec = VQCSpec(dataset=rec["dataset"], encoding=rec["encoding"],
                           ansatz_reps=rec["ansatz_reps"], n_qubits=rec["n_qubits"],
                           seed=rec["seed"], feature_map_reps=rec["feature_map_reps"])
            weights = np.asarray(rec["weights"], dtype=float)
            data = load_dataset(spec.dataset, spec.n_qubits, spec.seed, max_train=cfg["max_train"])
            n_data = spec.n_qubits
            X_test, y_test = data.X_test, data.y_test

            for k in cfg["backdoor_k"]:
                bspec = BackdoorSpec(k_trigger=k, angle_hi=angle_hi, seed=spec.seed)
                bd, golden, record = build_backdoor(spec, weights, bspec)
                save_qasm(bd, paths.backdoor / f"{mid}_k{k}.qasm")

                clean_golden = sampled_accuracy(golden, X_test, y_test, n_data, shots, spec.seed)
                clean_bd = sampled_accuracy(bd, X_test, y_test, n_data, shots, spec.seed)
                X_trig = make_trigger_inputs(n_data, k, angle_hi, n_eval, base=X_test,
                                             seed=spec.seed)
                pred_g = predict_sampled(golden, X_trig, n_data, shots, spec.seed)
                pred_b = predict_sampled(bd, X_trig, n_data, shots, spec.seed)
                asr = float(np.mean(pred_g != pred_b))

                rows.append({
                    "model_id": mid, "dataset": spec.dataset, "k_trigger": k,
                    "clean_acc_golden": clean_golden, "clean_acc_backdoor": clean_bd,
                    "clean_accuracy_drop": clean_golden - clean_bd, "asr": asr,
                    "depth_golden": record["depth_golden"], "depth_backdoor": record["depth_backdoor"],
                })
            LOG.info("  %s done", mid)

    df = pd.DataFrame(rows)
    df.to_csv(paths.root / "backdoor.csv", index=False)
    summary = df.groupby("k_trigger")[["asr", "clean_acc_backdoor", "clean_accuracy_drop"]].mean()
    LOG.info("Backdoor summary by k:\n%s", summary.round(3).to_string())

    if not df.empty:
        scatter_asr_vs_clean(df, paths.figures / "backdoor_asr_vs_clean.png",
                             "Triggered backdoor: ASR vs clean accuracy")
        df_k = df.assign(metric="ASR").rename(columns={"asr": "value"})
        grouped_bars(df_k, "k_trigger", "metric", "value",
                     paths.figures / "backdoor_asr_by_k.png",
                     "Attack success rate by trigger width", "ASR")
    LOG.info("exp3 complete: %d (model,k) rows", len(df))


if __name__ == "__main__":
    main()
