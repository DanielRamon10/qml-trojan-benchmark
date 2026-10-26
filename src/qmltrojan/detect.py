"""Text-based detection: golden vs infected, treating the QASM as a document.

This replicates the detection table of the quantum-trojan sensitivity paper. Each compiled
circuit is serialized to OpenQASM and vectorized as text (bag-of-words or TF-IDF, with unigrams
and bigrams). Four classifiers are compared under three cross-validation schemes:

* ``group`` -- **GroupKFold by base model**: all variants of one trained VQC stay in the same
  fold, so no golden/infected pair of the same model is split across train and test. Note that
  models sharing an *architecture* (encoding x ansatz reps x qubits) but differing in dataset or
  seed have structurally identical QASM (only the numeric angles differ), so this scheme still
  shows the detector the test architectures during training.
* ``arch`` -- **GroupKFold by architecture** (leave-architecture-out): every model of a given
  architecture is held out together, so the detector is tested on circuit *structures* it has
  never seen -- the realistic deployment question.
* ``stratified`` -- **StratifiedKFold** ignores families; it is the optimistic baseline.

The positive class is ``infected`` and it is the *majority* (many infected variants per golden).
To avoid a detector that trivially flags everything, we (a) random-undersample the majority class
*in the training fold only* (seeded per fold), keeping the real class ratio in the test fold, and
(b) also pass ``class_weight='balanced'`` to the linear models. We therefore report recall
*together with* balanced accuracy, F1 and ROC-AUC.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.svm import LinearSVC

#: QASM tokenizer: gate names, register tokens, numbers and brackets (keeps structure).
_TOKEN_PATTERN = r"[A-Za-z_][A-Za-z0-9_]*|[-+]?\d*\.\d+|\d+|[\[\]();,]"


def make_vectorizer(kind: str):
    """Return a BoW (``count``) or TF-IDF vectorizer over unigrams and bigrams."""
    common = {"token_pattern": _TOKEN_PATTERN, "ngram_range": (1, 2), "lowercase": False,
              "min_df": 2}
    if kind == "count":
        return CountVectorizer(**common)
    if kind == "tfidf":
        return TfidfVectorizer(**common)
    raise ValueError(f"Unknown vectorizer {kind!r}")


def make_classifier(name: str, class_weight: str | None):
    """Instantiate a detection classifier.

    ``class_weight`` is honored only by the linear models; the Naive Bayes variants ignore it.
    """
    if name == "logreg":
        return LogisticRegression(max_iter=2000, class_weight=class_weight)
    if name == "linsvc":
        return LinearSVC(class_weight=class_weight)
    if name == "multinb":
        return MultinomialNB()
    if name == "complnb":
        return ComplementNB()
    raise ValueError(f"Unknown classifier {name!r}")


CLASSIFIERS: tuple[str, ...] = ("logreg", "linsvc", "multinb", "complnb")
VECTORIZERS: tuple[str, ...] = ("count", "tfidf")
CV_SCHEMES: tuple[str, ...] = ("group", "arch", "stratified")


def _undersample_train(idx: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    """Randomly undersample the majority class within the training indices (seeded)."""
    rng = np.random.default_rng(seed)
    pos = idx[y[idx] == 1]
    neg = idx[y[idx] == 0]
    n = min(len(pos), len(neg))
    if n == 0:
        return idx
    keep_pos = rng.choice(pos, size=n, replace=False)
    keep_neg = rng.choice(neg, size=n, replace=False)
    out = np.concatenate([keep_pos, keep_neg])
    rng.shuffle(out)
    return out


def _scores(clf, X_test) -> np.ndarray | None:
    """Continuous scores for ROC-AUC (probabilities or SVM decision function)."""
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(X_test)[:, 1]
    if hasattr(clf, "decision_function"):
        return clf.decision_function(X_test)
    return None


def run_detection(texts: list[str], labels: np.ndarray, groups: np.ndarray,
                  vectorizer: str, classifier: str, cv_scheme: str,
                  n_splits: int = 5, seed: int = 0) -> tuple[dict, np.ndarray]:
    """Cross-validated detection for one (vectorizer, classifier, cv) combination.

    For the ``group`` and ``arch`` schemes, ``groups`` holds the grouping to keep together
    (base-model ids or architecture ids, respectively); both run a GroupKFold with at most
    as many folds as there are groups.

    Returns a metrics dict (mean/std across folds) and the summed confusion matrix.
    """
    labels = np.asarray(labels)
    groups = np.asarray(groups)
    use_weight = classifier in ("logreg", "linsvc")
    class_weight = "balanced" if use_weight else None

    if cv_scheme in ("group", "arch"):
        splitter = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))
        split_iter = splitter.split(texts, labels, groups)
    elif cv_scheme == "stratified":
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        split_iter = splitter.split(texts, labels)
    else:
        raise ValueError(f"Unknown cv scheme {cv_scheme!r}")

    per_fold: list[dict] = []
    cm_total = np.zeros((2, 2), dtype=int)
    texts_arr = np.asarray(texts, dtype=object)

    for fold, (train_idx, test_idx) in enumerate(split_iter):
        train_idx = _undersample_train(train_idx, labels, seed + fold)
        vec = make_vectorizer(vectorizer)
        Xtr = vec.fit_transform(texts_arr[train_idx].tolist())
        Xte = vec.transform(texts_arr[test_idx].tolist())
        ytr, yte = labels[train_idx], labels[test_idx]
        clf = make_classifier(classifier, class_weight)
        clf.fit(Xtr, ytr)
        pred = clf.predict(Xte)
        scores = _scores(clf, Xte)

        row = {
            "accuracy": accuracy_score(yte, pred),
            "balanced_accuracy": balanced_accuracy_score(yte, pred),
            "precision": precision_score(yte, pred, zero_division=0),
            "recall": recall_score(yte, pred, zero_division=0),
            "f1": f1_score(yte, pred, zero_division=0),
            "f1_macro": f1_score(yte, pred, average="macro", zero_division=0),
            "roc_auc": roc_auc_score(yte, scores) if scores is not None
            and len(np.unique(yte)) > 1 else np.nan,
        }
        per_fold.append(row)
        cm_total += confusion_matrix(yte, pred, labels=[0, 1])

    df = pd.DataFrame(per_fold)
    summary = {"vectorizer": vectorizer, "classifier": classifier, "cv": cv_scheme,
               "n_splits": len(per_fold)}
    for col in df.columns:
        summary[f"{col}_mean"] = float(df[col].mean())
        summary[f"{col}_std"] = float(df[col].std(ddof=0))
    return summary, cm_total


def run_detection_grid(texts: list[str], labels: np.ndarray, groups: np.ndarray,
                       n_splits: int = 5, seed: int = 0,
                       arch_groups: np.ndarray | None = None,
                       ) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Run every (vectorizer x classifier x cv) combination; return a table and confusions.

    ``groups`` are base-model ids (``group`` scheme); ``arch_groups`` are architecture ids for
    the leave-architecture-out ``arch`` scheme, which is skipped when not given or when there
    are fewer than two architectures.
    """
    rows: list[dict] = []
    confusions: dict[str, np.ndarray] = {}
    for cv in CV_SCHEMES:
        cv_groups = arch_groups if cv == "arch" else groups
        if cv == "arch" and (arch_groups is None or len(np.unique(arch_groups)) < 2):
            continue
        for vec in VECTORIZERS:
            for clf in CLASSIFIERS:
                summary, cm = run_detection(texts, labels, cv_groups, vec, clf, cv,
                                            n_splits=n_splits, seed=seed)
                rows.append(summary)
                confusions[f"{cv}_{vec}_{clf}"] = cm
    return pd.DataFrame(rows), confusions


def format_mean_std(df: pd.DataFrame, metric: str) -> pd.Series:
    """Return a ``mean ± std`` string column for one metric (paper-style table)."""
    return df[f"{metric}_mean"].map("{:.3f}".format) + " ± " + df[f"{metric}_std"].map(
        "{:.3f}".format)
