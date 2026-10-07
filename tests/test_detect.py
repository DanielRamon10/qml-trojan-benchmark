"""Detection-pipeline tests: undersampling safety and metric/CV plumbing."""

from __future__ import annotations

import numpy as np

from qmltrojan.detect import _undersample_train, run_detection


def _toy_corpus(n_models: int = 6, variants: int = 8):
    """Synthetic QASM-like corpus where infected texts carry a tell-tale token."""
    texts, labels, groups = [], [], []
    rng = np.random.default_rng(0)
    for m in range(n_models):
        base = f"h q[0]; cx q[0],q[1]; rz({rng.random():.3f}) q[1];"
        texts.append(base)
        labels.append(0)
        groups.append(m)
        for _ in range(variants):
            texts.append(base + " x q[2];")  # inserted-gate signature
            labels.append(1)
            groups.append(m)
    return texts, np.array(labels), np.array(groups)


def test_undersample_balances_and_keeps_only_given_indices():
    y = np.array([0, 0, 1, 1, 1, 1, 1, 1])
    idx = np.arange(len(y))
    out = _undersample_train(idx, y, seed=0)
    assert set(out).issubset(set(idx))
    assert (y[out] == 0).sum() == (y[out] == 1).sum()  # balanced


def test_undersample_never_touches_test_indices():
    # Only training indices are passed in; the test fold must remain untouched.
    y = np.array([0, 1, 1, 1, 0, 1, 1, 1, 0, 1])
    train_idx = np.array([0, 1, 2, 3, 4])
    test_idx = np.array([5, 6, 7, 8, 9])
    out = _undersample_train(train_idx, y, seed=1)
    assert len(np.intersect1d(out, test_idx)) == 0


def test_run_detection_returns_metrics_and_confusion():
    texts, labels, groups = _toy_corpus()
    summary, cm = run_detection(texts, labels, groups, "count", "logreg", "group",
                                n_splits=3, seed=0)
    for key in ("recall_mean", "balanced_accuracy_mean", "f1_mean", "roc_auc_mean"):
        assert key in summary
    assert cm.shape == (2, 2)
    assert cm.sum() == len(texts)


def test_groupkfold_separates_models():
    texts, labels, groups = _toy_corpus()
    # With a clear signature the detector should recall most infected circuits.
    summary, _ = run_detection(texts, labels, groups, "count", "linsvc", "group",
                               n_splits=3, seed=0)
    assert summary["recall_mean"] > 0.5
