"""Generate the README 'Results' section (Markdown) from a completed run's CSVs.

Reads ``results/<mode>/`` and prints a Markdown block with the headline numbers and figure links.
Used to fill the ``<!-- RESULTS:START -->`` / ``<!-- RESULTS:END -->`` markers in README.md.

Usage:
    python scripts/make_results_section.py --config full
"""

from __future__ import annotations

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
               f"**{_fmt(base)}**; mean accuracy drop **{_fmt(sens['accuracy_drop'].mean())}**.\n")
    by_gate = sens.groupby("gate").agg(
        tvd_mean=("tvd_mean", "mean"), bc_mean=("bc_mean", "mean"), bd_mean=("bd_mean", "mean"),
        accuracy_drop=("accuracy_drop", "mean")).round(3)
    out.append("**By inserted gate:**\n")
    out.append(by_gate.to_markdown() + "\n")
    by_mode = sens.groupby("mode").agg(
        tvd_mean=("tvd_mean", "mean"), accuracy_drop=("accuracy_drop", "mean")).round(3)
    out.append("**By insertion mode:**\n")
    out.append(by_mode.to_markdown() + "\n")
    out.append(f"![TVD by gate]({fig}/tvd_by_gate.png)\n")
    out.append(f"![Accuracy drop by gate]({fig}/accdrop_by_gate.png)\n")

    # SHAP
    try:
        shap_tvd = pd.read_csv(paths.root / "shap_tvd.csv").head(6)
        out.append("**SHAP — top drivers of mean TVD:** "
                   + ", ".join(f"`{r.feature}` ({r.mean_abs_shap:.3f})"
                               for r in shap_tvd.itertuples()) + "\n")
        out.append(f"![SHAP TVD]({fig}/shap_tvd.png)\n")
    except FileNotFoundError:
        pass

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
               "leakage (a more selective, stealthier trigger).\n")
    out.append(f"![Backdoor ASR vs clean]({fig}/backdoor_asr_vs_clean.png)\n")

    # --- Experiment 4: detection ---
    det = pd.read_csv(paths.root / "detection_summary.csv")
    out.append("### 4. Text-based detection (golden vs infected)\n")
    out.append("Positive class = **infected**. Mean ± std across seeds; "
               "GroupKFold keeps all variants of a base circuit in one fold.\n")
    cols = ["cv", "vectorizer", "classifier", "recall", "balanced_accuracy", "f1", "roc_auc"]
    out.append(det[cols].to_markdown(index=False) + "\n")
    out.append(f"![Detection recall]({fig}/detection_recall.png)\n")
    out.append(f"![Confusion matrices (GroupKFold)]({fig}/confusions_group.png)\n")

    hard_path = paths.root / "detection_hard_summary.csv"
    if hard_path.exists():
        hard = pd.read_csv(hard_path)
        out.append("\n**Hard sub-problem — detecting the *in-vocabulary* `H` insertion only.** "
                   "Golden circuits in this basis emit only `ry, p, cx, h`, so inserted "
                   "`x/y/z/t` are out-of-vocabulary tokens and trivially flagged; the `H` "
                   "insertion is the genuinely hard case because `h` already occurs in golden.\n")
        out.append(hard[cols].to_markdown(index=False) + "\n")

    print("\n".join(out))


if __name__ == "__main__":
    main()
