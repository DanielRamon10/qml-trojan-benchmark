"""Generate the README 'Results' section (Markdown) from a completed run's CSVs.

Reads ``results/<mode>/`` and prints a Markdown block with the headline numbers and figure links.
Used to fill the ``<!-- RESULTS:START -->`` / ``<!-- RESULTS:END -->`` markers in README.md.

Usage:
    python scripts/make_results_section.py --config full
"""

from __future__ import annotations

import sys

import pandas as pd
from _common import load_config, mode_from_args, parse_args, resolve_paths

FIG = "results/{mode}/figures"


def _fmt(x: float, n: int = 3) -> str:
    return f"{x:.{n}f}"


def main() -> None:
    args = parse_args(__doc__)
    mode = mode_from_args(args)
    _ = load_config(mode)
    paths = resolve_paths(mode)
    fig = FIG.format(mode=mode)
    out: list[str] = []

    # --- Experiment 1: clean accuracy ---
    acc = pd.read_csv(paths.root / "golden_accuracy.csv")
    out.append("### 1. Clean accuracy of the golden VQCs\n")
    out.append(f"Trained **{len(acc)}** golden VQCs; mean clean test accuracy "
               f"**{_fmt(acc['test_accuracy'].mean())} ± {_fmt(acc['test_accuracy'].std())}**.\n")
    piv = acc.groupby(["dataset", "encoding"])["test_accuracy"].mean().unstack().round(3)
    out.append(piv.to_markdown() + "\n")

    # --- Experiment 2: sensitivity + accuracy drop ---
    sens = pd.read_csv(paths.root / "sensitivity.csv")
    base = sens["tvd_baseline_mean"].mean()
    out.append("### 2. Sensitivity (TVD) and accuracy impact\n")
    out.append(f"Across **{len(sens)}** infected variants, mean TVD "
               f"**{_fmt(sens['tvd_mean'].mean())}** vs a golden-vs-golden sampling baseline of "
               f"**{_fmt(base)}**; mean accuracy drop **{_fmt(sens['accuracy_drop'].mean())}**; "
               f"mean prediction flip rate **{_fmt(sens['flip_rate'].mean())}**. TVD is "
               f"estimated on the first {int(sens['n_inputs'].iloc[0])} test inputs of each "
               "model; accuracy and flip rate use the whole test split.\n")
    by_gate = sens.groupby("gate").agg(
        tvd_mean=("tvd_mean", "mean"), bc_mean=("bc_mean", "mean"), bd_mean=("bd_mean", "mean"),
        accuracy_drop=("accuracy_drop", "mean"), flip_rate=("flip_rate", "mean")).round(3)
    out.append("**By inserted gate:**\n")
    out.append(by_gate.to_markdown() + "\n")
    by_mode = sens.groupby("mode").agg(
        tvd_mean=("tvd_mean", "mean"), accuracy_drop=("accuracy_drop", "mean"),
        flip_rate=("flip_rate", "mean"), at_wire_end=("at_wire_end", "mean")).round(3)
    out.append("**By insertion mode** (`at_wire_end` = fraction of insertions placed after the "
               "wire's last gate, i.e. right before measurement):\n")
    out.append(by_mode.to_markdown() + "\n")
    diag = sens[sens["gate"].isin(["z", "t"])].groupby("at_wire_end").agg(
        n=("tvd_mean", "size"), tvd_mean=("tvd_mean", "mean"),
        tvd_baseline=("tvd_baseline_mean", "mean"), flip_rate=("flip_rate", "mean")).round(3)
    out.append("**Diagonal gates (`Z`, `T`) by position:** right before measurement they commute "
               "with the Z-basis readout, so their TVD should sit at the sampling floor.\n")
    out.append(diag.to_markdown() + "\n")
    out.append(f"![TVD by gate]({fig}/tvd_by_gate.png)\n")
    out.append(f"![Accuracy drop by gate]({fig}/accdrop_by_gate.png)\n")
    out.append(f"![Flip rate by gate]({fig}/flip_by_gate.png)\n")

    # SHAP
    for tag, label in (("tvd", "mean TVD"), ("flip_rate", "prediction flip rate")):
        try:
            shap_df = pd.read_csv(paths.root / f"shap_{tag}.csv").head(6)
        except FileNotFoundError:
            continue
        out.append(f"**SHAP — top drivers of {label}:** "
                   + ", ".join(f"`{r.feature}` ({r.mean_abs_shap:.3f})"
                               for r in shap_df.itertuples()) + "\n")
        out.append(f"![SHAP {label}]({fig}/shap_{tag}.png)\n")

    # --- Experiment 3: backdoor ---
    bd = pd.read_csv(paths.root / "backdoor.csv")
    out.append("### 3. Triggered backdoor: ASR vs clean accuracy\n")
    by_k = bd.groupby("k_trigger").agg(
        asr=("asr", "mean"), clean_acc_backdoor=("clean_acc_backdoor", "mean"),
        clean_accuracy_drop=("clean_accuracy_drop", "mean")).round(3)
    out.append("Conditional backdoor on the 4-qubit `z`-encoding models "
               f"({bd['model_id'].nunique()} models):\n")
    out.append(by_k.to_markdown() + "\n")
    out.append("Higher trigger width `k` keeps attack success high while reducing clean-accuracy "
               "leakage. The detector is a soft threshold (per trigger qubit it fires with "
               "probability sin²(x)), so with `k=1` every clean input with x₀ > π/4 is flipped; "
               "requiring `k` qubits multiplies these probabilities and makes the trigger more "
               "selective.\n")
    out.append(f"![Backdoor ASR vs clean]({fig}/backdoor_asr_vs_clean.png)\n")

    # --- Experiment 4: detection ---
    det = pd.read_csv(paths.root / "detection_summary.csv")
    out.append("### 4. Text-based detection (golden vs infected)\n")
    out.append("Positive class = **infected**. Mean ± std across seeds. `group` = GroupKFold by "
               "base model; `arch` = GroupKFold by architecture (encoding × ansatz reps × "
               "qubits), i.e. the detector never sees the test circuit *structure* during "
               "training; `stratified` = StratifiedKFold.\n")
    cols = ["cv", "vectorizer", "classifier", "recall", "balanced_accuracy", "f1", "roc_auc"]
    out.append(det[cols].to_markdown(index=False) + "\n")
    out.append(f"![Detection recall]({fig}/detection_recall.png)\n")
    out.append(f"![Confusion matrices (leave-architecture-out)]({fig}/confusions_arch.png)\n")

    hard_path = paths.root / "detection_hard_summary.csv"
    if hard_path.exists():
        hard = pd.read_csv(hard_path)
        out.append("\n**Hard sub-problem — detecting the *in-vocabulary* `H` insertion only.** "
                   "Golden circuits in this basis emit only `ry, p, cx, h`, so inserted "
                   "`x/y/z/t` are out-of-vocabulary tokens and trivially flagged; the `H` "
                   "insertion is the harder case because `h` already occurs in golden. It is "
                   "still lexically easy: every golden circuit has exactly as many `h` as "
                   "encoding phases `p(2*_x_i_)` (one of each per qubit per feature-map rep), "
                   "and an inserted `h` breaks that count equality.\n")
        out.append(hard[cols].to_markdown(index=False) + "\n")

    # The Markdown contains non-Latin-1 symbols (π, x₀); Windows consoles default to cp1252.
    sys.stdout.reconfigure(encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
