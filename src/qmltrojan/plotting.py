"""Shared matplotlib helpers and figure generators (PNG output).

Styling follows a validated, colorblind-safe categorical palette (OKLab CVD-checked). All
figures carry axis labels and, where marks sit on a light surface, direct labels or legends, so
identity never relies on color alone.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Validated categorical palette (light mode), assigned in fixed order.
PALETTE: list[str] = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
GATE_COLORS: dict[str, str] = {"h": "#2a78d6", "t": "#eb6834", "x": "#1baf7a",
                               "y": "#eda100", "z": "#e87ba4"}
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#52514e"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
    "grid.color": "#e7e7e2", "grid.linewidth": 0.8, "axes.axisbelow": True,
    "font.size": 11, "axes.titlesize": 13, "figure.dpi": 130,
})


def _save(fig: plt.Figure, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return path


def violin_strip_by(df: pd.DataFrame, value: str, by: str, path: str | Path,
                    title: str, baseline: float | None = None) -> Path:
    """Violin + jittered strip of ``value`` grouped by the categorical column ``by``."""
    groups = sorted(df[by].dropna().unique())
    data = [df.loc[df[by] == g, value].dropna().to_numpy() for g in groups]
    fig, ax = plt.subplots(figsize=(1.4 * len(groups) + 2.5, 4.2))
    parts = ax.violinplot(data, showmeans=False, showextrema=False)
    for i, body in enumerate(parts["bodies"]):
        body.set_facecolor(PALETTE[i % len(PALETTE)])
        body.set_alpha(0.35)
        body.set_edgecolor(MUTED)
    rng = np.random.default_rng(0)
    for i, arr in enumerate(data, start=1):
        if len(arr) == 0:
            continue
        x = i + rng.uniform(-0.09, 0.09, size=len(arr))
        ax.scatter(x, arr, s=16, color=PALETTE[(i - 1) % len(PALETTE)],
                   edgecolor="white", linewidth=0.4, zorder=3)
        ax.scatter([i], [np.mean(arr)], marker="D", s=46, color=INK, zorder=4)
    if baseline is not None:
        ax.axhline(baseline, color=MUTED, ls="--", lw=1.2)
        ax.text(0.5, baseline, f" baseline {baseline:.3f}", va="bottom", ha="left",
                color=MUTED, fontsize=9)
    ax.set_xticks(range(1, len(groups) + 1))
    ax.set_xticklabels(groups)
    ax.set_xlabel(by)
    ax.set_ylabel(value)
    ax.set_title(title)
    return _save(fig, path)


def bar_importance(importance: pd.DataFrame, path: str | Path, title: str,
                   top: int = 12) -> Path:
    """Horizontal bar of mean |SHAP| importances (highest at top)."""
    imp = importance.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.42 * len(imp) + 1.2))
    ax.barh(imp["feature"], imp["mean_abs_shap"], color=PALETTE[0], edgecolor="white")
    for y, v in enumerate(imp["mean_abs_shap"]):
        ax.text(v, y, f" {v:.3f}", va="center", color=INK, fontsize=9)
    ax.set_xlabel("mean |SHAP value|")
    ax.set_title(title)
    ax.grid(axis="y", visible=False)
    return _save(fig, path)


def grouped_bars(df: pd.DataFrame, index: str, columns: str, value: str,
                 path: str | Path, title: str, ylabel: str) -> Path:
    """Grouped bar chart of ``value`` pivoted by ``index`` (x) and ``columns`` (series)."""
    pivot = df.pivot_table(index=index, columns=columns, values=value, aggfunc="mean")
    cats = list(pivot.index)
    series = list(pivot.columns)
    x = np.arange(len(cats))
    width = 0.8 / max(len(series), 1)
    fig, ax = plt.subplots(figsize=(1.3 * len(cats) + 2.5, 4.2))
    for j, s in enumerate(series):
        ax.bar(x + j * width, pivot[s].to_numpy(), width=width * 0.92,
               label=str(s), color=PALETTE[j % len(PALETTE)], edgecolor="white")
    ax.set_xticks(x + width * (len(series) - 1) / 2)
    ax.set_xticklabels(cats, rotation=20, ha="right")
    ax.set_xlabel(index)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(title=columns, frameon=False)
    return _save(fig, path)


def confusion_grid(confusions: dict[str, np.ndarray], path: str | Path, title: str,
                   ncols: int = 4) -> Path:
    """Grid of confusion matrices (rows=true, cols=pred; labels clean/infected)."""
    keys = list(confusions)
    nrows = int(np.ceil(len(keys) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 3.0 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, key in zip(axes, keys, strict=False):
        cm = confusions[key]
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color=INK if cm[i, j] < cm.max() / 2 else "white", fontsize=10)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["clean", "infected"], fontsize=8)
        ax.set_yticks([0, 1]); ax.set_yticklabels(["clean", "infected"], fontsize=8)
        ax.set_xlabel("predicted", fontsize=8); ax.set_ylabel("true", fontsize=8)
        ax.set_title(key, fontsize=9)
    for ax in axes[len(keys):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return _save(fig, path)


def scatter_asr_vs_clean(df: pd.DataFrame, path: str | Path, title: str) -> Path:
    """Scatter of attack success rate vs clean accuracy, colored/annotated by k."""
    fig, ax = plt.subplots(figsize=(6, 4.4))
    ks = sorted(df["k_trigger"].unique())
    for i, k in enumerate(ks):
        sub = df[df["k_trigger"] == k]
        ax.scatter(sub["clean_acc_backdoor"], sub["asr"], s=70,
                   color=PALETTE[i % len(PALETTE)], edgecolor="white",
                   label=f"k={k}", zorder=3)
    ax.set_xlabel("clean accuracy (backdoor model)")
    ax.set_ylabel("attack success rate (ASR)")
    ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
    ax.set_title(title)
    ax.legend(title="trigger qubits", frameon=False)
    return _save(fig, path)
