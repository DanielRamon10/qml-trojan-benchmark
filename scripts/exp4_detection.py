"""Experiment 4 -- text-based detection of infected VQCs (golden vs infected).

Treats each compiled circuit's OpenQASM as a document. Runs every
(representation x classifier x cross-validation) combination over ``detect_seeds`` seeds and
reports the full metric suite as ``mean +/- std``, plus confusion matrices. The positive class is
``infected``; GroupKFold keeps all variants of one base VQC in the same fold.

Outputs (under ``results/<mode>/``):
* ``detection.csv``                 -- per (seed, representation, classifier, cv) metrics
* ``detection_summary.csv``         -- mean +/- std across seeds (paper-style table)
* ``detection_confusions.npz``      -- summed confusion matrices
* figures: ``detection_recall.png``, ``detection_balacc.png``,
  ``confusions_group.png``, ``confusions_stratified.png``
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
    mode_from_args,
    parse_args,
    resolve_paths,
)

from qmltrojan.detect import format_mean_std, run_detection_grid
from qmltrojan.plotting import confusion_grid, grouped_bars

LOG = get_logger("exp4")

METRICS = ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "f1_macro", "roc_auc"]


def build_corpus(paths) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Load golden (label 0) and infected (label 1) QASM texts with base-model group ids."""
    texts, labels, groups = [], [], []
    model_ids: dict[str, int] = {}

    def gid(model_id: str) -> int:
        return model_ids.setdefault(model_id, len(model_ids))

    for f in sorted(paths.golden.glob("*.qasm")):
        texts.append(f.read_text(encoding="utf-8"))
        labels.append(0)
        groups.append(gid(f.stem))
    for f in sorted(paths.infected.glob("*.qasm")):
        texts.append(f.read_text(encoding="utf-8"))
        labels.append(1)
        groups.append(gid(f.stem.split("__")[0]))
    return texts, np.asarray(labels), np.asarray(groups)


def main() -> None:
    args = parse_args(__doc__)
    mode = mode_from_args(args)
    cfg = load_config(mode)
    paths = resolve_paths(mode)

    texts, labels, groups = build_corpus(paths)
    LOG.info("Corpus: %d circuits (%d golden, %d infected), %d base models",
             len(texts), int((labels == 0).sum()), int((labels == 1).sum()),
             len(np.unique(groups)))

    all_rows, confusions_by_seed = [], []
    with Timer(LOG, "exp4 total"):
        for seed in cfg["detect_seeds"]:
            df, confusions = run_detection_grid(texts, labels, groups,
                                                n_splits=cfg["cv_splits"], seed=seed)
            df["seed"] = seed
            all_rows.append(df)
            confusions_by_seed.append(confusions)

    detection = pd.concat(all_rows, ignore_index=True)
    detection.to_csv(paths.root / "detection.csv", index=False)

    # Mean +/- std across seeds for each (cv, representation, classifier) cell.
    summary = _summarize(detection)
    summary.to_csv(paths.root / "detection_summary.csv", index=False)
    LOG.info("Detection summary (recall | balanced_accuracy | roc_auc):\n%s",
             summary[["cv", "vectorizer", "classifier", "recall", "balanced_accuracy",
                      "roc_auc"]].to_string(index=False))

    # Sum confusion matrices across seeds.
    summed = {key: sum(cs[key] for cs in confusions_by_seed)
              for key in confusions_by_seed[0]}
    np.savez(paths.root / "detection_confusions.npz",
             **{k: v for k, v in summed.items()})

    _figures(detection, summed, paths)
    LOG.info("exp4 complete.")


def _summarize(detection: pd.DataFrame) -> pd.DataFrame:
    keys = ["cv", "vectorizer", "classifier"]
    agg = detection.groupby(keys).agg(
        {f"{m}_mean": "mean" for m in METRICS}
        | {f"{m}_std": "mean" for m in METRICS}
    ).reset_index()
    out = agg[keys].copy()
    for m in METRICS:
        out[m] = format_mean_std(agg, m)
    return out


def _figures(detection: pd.DataFrame, confusions: dict, paths) -> None:
    for metric, fname, title in (
        ("recall_mean", "detection_recall.png", "Detection recall (positive = infected)"),
        ("balanced_accuracy_mean", "detection_balacc.png", "Detection balanced accuracy"),
    ):
        tmp = detection.copy()
        tmp["rep_cv"] = tmp["vectorizer"] + "/" + tmp["cv"]
        grouped_bars(tmp, "classifier", "rep_cv", metric, paths.figures / fname, title,
                     metric.replace("_mean", ""))
    group = {k.replace("group_", ""): v for k, v in confusions.items() if k.startswith("group_")}
    strat = {k.replace("stratified_", ""): v for k, v in confusions.items()
             if k.startswith("stratified_")}
    confusion_grid(group, paths.figures / "confusions_group.png",
                   "Confusion matrices -- GroupKFold")
    confusion_grid(strat, paths.figures / "confusions_stratified.png",
                   "Confusion matrices -- StratifiedKFold")


if __name__ == "__main__":
    main()
