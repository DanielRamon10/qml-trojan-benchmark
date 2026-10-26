"""Experiment 4 -- text-based detection of infected VQCs (golden vs infected).

Treats each compiled circuit's OpenQASM as a document. Runs every
(representation x classifier x cross-validation) combination over ``detect_seeds`` seeds and
reports the full metric suite as ``mean +/- std``, plus confusion matrices. The positive class is
``infected``. Three CV schemes: ``group`` (GroupKFold by base model), ``arch`` (GroupKFold by
architecture = encoding x ansatz reps x qubits, i.e. leave-architecture-out) and ``stratified``.

Outputs (under ``results/<mode>/``):
* ``detection.csv``                 -- per (seed, representation, classifier, cv) metrics
* ``detection_summary.csv``         -- mean +/- std across seeds (paper-style table)
* ``detection_hard_summary.csv``    -- same, for the in-vocabulary (H-only) sub-problem
* ``detection_confusions.npz``      -- summed confusion matrices
* figures: ``detection_recall.png``, ``detection_balacc.png``,
  ``confusions_group.png``, ``confusions_arch.png``, ``confusions_stratified.png``
"""

from __future__ import annotations

import re
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

#: ``<dataset>_<encoding>_r<reps>_q<qubits>_s<seed>``; the dataset name may contain underscores.
_ARCH_RE = re.compile(r"_(z|zz)_r(\d+)_q(\d+)_s\d+$")


def architecture_of(model_id: str) -> str:
    """Architecture key (encoding x ansatz reps x qubits) of a model id, ignoring dataset/seed."""
    match = _ARCH_RE.search(model_id)
    if match is None:
        raise ValueError(f"cannot parse architecture from model id {model_id!r}")
    enc, reps, nq = match.groups()
    return f"{enc}_r{reps}_q{nq}"


def build_corpus(paths, infected_gates: tuple[str, ...] | None = None,
                 ) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    """Load golden (label 0) and infected (label 1) QASM texts with model and architecture ids.

    Returns ``(texts, labels, model_groups, arch_groups)``. ``infected_gates`` optionally
    restricts the infected class to variants whose inserted gate is in the set (filenames look
    like ``<model>__<gate>_<mode>_s<seed>.qasm``). Restricting to the ``h`` insertion gives the
    *in-vocabulary* (hard) detection problem, because golden circuits in this basis never contain
    bare ``x``/``y``/``z``/``t`` tokens, whereas ``h`` already occurs.
    """
    texts, labels, groups, archs = [], [], [], []
    model_ids: dict[str, int] = {}
    arch_ids: dict[str, int] = {}

    def add(text: str, label: int, model_id: str) -> None:
        texts.append(text)
        labels.append(label)
        groups.append(model_ids.setdefault(model_id, len(model_ids)))
        arch = architecture_of(model_id)
        archs.append(arch_ids.setdefault(arch, len(arch_ids)))

    for f in sorted(paths.golden.glob("*.qasm")):
        add(f.read_text(encoding="utf-8"), 0, f.stem)
    for f in sorted(paths.infected.glob("*.qasm")):
        gate = f.stem.split("__")[1].split("_")[0]
        if infected_gates is not None and gate not in infected_gates:
            continue
        add(f.read_text(encoding="utf-8"), 1, f.stem.split("__")[0])
    return texts, np.asarray(labels), np.asarray(groups), np.asarray(archs)


def main() -> None:
    args = parse_args(__doc__)
    mode = mode_from_args(args)
    cfg = load_config(mode)
    paths = resolve_paths(mode)

    texts, labels, groups, archs = build_corpus(paths)
    LOG.info("Corpus: %d circuits (%d golden, %d infected), %d base models, %d architectures",
             len(texts), int((labels == 0).sum()), int((labels == 1).sum()),
             len(np.unique(groups)), len(np.unique(archs)))

    all_rows, confusions_by_seed = [], []
    with Timer(LOG, "exp4 total"):
        for seed in cfg["detect_seeds"]:
            df, confusions = run_detection_grid(texts, labels, groups,
                                                n_splits=cfg["cv_splits"], seed=seed,
                                                arch_groups=archs)
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

    # Hard sub-problem: detect the IN-VOCABULARY insertion (H only). X/Y/Z/T are bare tokens the
    # clean basis never emits, so they are trivially detectable; H already occurs in golden.
    _run_hard_subset(cfg, paths)
    LOG.info("exp4 complete.")


def _run_hard_subset(cfg, paths) -> None:
    texts, labels, groups, archs = build_corpus(paths, infected_gates=("h",))
    n_pos, n_neg = int((labels == 1).sum()), int((labels == 0).sum())
    if n_pos == 0 or len(np.unique(groups)) < cfg["cv_splits"]:
        LOG.info("Hard (H-only) subset skipped (insufficient data).")
        return
    LOG.info("Hard subset (H insertions only): %d golden + %d infected", n_neg, n_pos)
    rows = []
    for seed in cfg["detect_seeds"]:
        df, _ = run_detection_grid(texts, labels, groups, n_splits=cfg["cv_splits"], seed=seed,
                                   arch_groups=archs)
        df["seed"] = seed
        rows.append(df)
    hard = pd.concat(rows, ignore_index=True)
    summary = _summarize(hard)
    summary.to_csv(paths.root / "detection_hard_summary.csv", index=False)
    LOG.info("Hard (H-only) detection summary (recall | balanced_accuracy | roc_auc):\n%s",
             summary[["cv", "vectorizer", "classifier", "recall", "balanced_accuracy",
                      "roc_auc"]].to_string(index=False))


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
    for cv, fname, title in (
        ("group", "confusions_group.png", "Confusion matrices -- GroupKFold (by base model)"),
        ("arch", "confusions_arch.png", "Confusion matrices -- leave-architecture-out"),
        ("stratified", "confusions_stratified.png", "Confusion matrices -- StratifiedKFold"),
    ):
        sub = {k.removeprefix(f"{cv}_"): v for k, v in confusions.items()
               if k.startswith(f"{cv}_")}
        if sub:
            confusion_grid(sub, paths.figures / fname, title)


if __name__ == "__main__":
    main()
