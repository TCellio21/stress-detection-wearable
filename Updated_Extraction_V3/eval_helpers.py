"""
Reusable LOSO evaluation helpers for V3 experiments.

Used by notebooks/04, 05, 06, 07. Stays here (not in a notebook) so the
methodology is version-controlled and identical across experiments. The
notebook *orchestrates* — this module computes.

Design points:
- Model-agnostic: takes a `model_factory` callable returning a fresh sklearn
  estimator each fold. Lets us swap RF/HGB/XGB/LR without re-implementing
  the LOSO loop.
- Returns both aggregate metrics and a per-subject recall vector — the
  per-subject question is the load-bearing one for our project (audit §6
  #4) and it should not be hidden behind aggregates.
- Handles NaN: HGB consumes NaN natively; for models that don't, callers
  pre-impute (or use this helper's `nan_fill='subject_median'` shortcut).
"""

from __future__ import annotations

from typing import Callable, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import LeaveOneGroupOut


def fill_subject_median(df: pd.DataFrame, feature_cols: list[str],
                        subject_col: str = "subject_id") -> pd.DataFrame:
    """Per-subject median imputation; columns still all-NaN within a subject become 0."""
    out = df.copy()
    for sid, idx in out.groupby(subject_col).groups.items():
        sub = out.loc[idx, feature_cols]
        med = sub.median(skipna=True)
        out.loc[idx, feature_cols] = sub.fillna(med)
    out[feature_cols] = out[feature_cols].fillna(0.0)
    return out


def loso_evaluate(
    df: pd.DataFrame,
    feature_cols: list[str],
    model_factory: Callable[[], "sklearn.base.ClassifierMixin"],
    label_col: str = "label",
    positive_class: str = "stress",
    subject_col: str = "subject_id",
    nan_fill: str | None = None,
) -> dict:
    """LOSO cross-validation with a model factory.

    Parameters
    ----------
    df : DataFrame with one row per window, including subject_col, label_col, and features
    feature_cols : list of column names to use as model input
    model_factory : zero-arg callable returning a fresh sklearn estimator. The factory
                    is called once per fold so each fold gets a freshly-fit model.
    label_col : column with class labels (string or int); converted to binary {1 if == positive_class else 0}
    positive_class : label value treated as the positive (stress) class
    subject_col : column whose values define the LOSO groups
    nan_fill : if "subject_median", per-subject median fill before training; else leave NaNs
               (HGB and similar models handle NaN natively; RF/LR/SVM do not)

    Returns
    -------
    dict with:
        fold_df : per-fold metrics DataFrame (rows = folds, cols = subject/F1/recall/...)
        per_subject_recall : dict subject_id -> recall (positive class)
        mean_f1, std_f1 : aggregate F1 stress
        mean_recall, std_recall : aggregate recall stress
        mean_precision, mean_accuracy
        min_subject_recall, n_subjects_recall_below_0_5, n_subjects_recall_zero
    """
    if nan_fill == "subject_median":
        df = fill_subject_median(df, feature_cols, subject_col=subject_col)

    X = df[feature_cols].values
    y = (df[label_col].values == positive_class).astype(int)
    groups = df[subject_col].values

    logo = LeaveOneGroupOut()
    rows = []
    per_subject = {}
    for train_idx, test_idx in logo.split(X, y, groups):
        test_subj = groups[test_idx[0]]
        clf = model_factory()
        clf.fit(X[train_idx], y[train_idx])
        pred = clf.predict(X[test_idx])
        yte = y[test_idx]
        rows.append({
            "subject": test_subj,
            "n_test": len(yte),
            "n_stress": int(yte.sum()),
            "accuracy": accuracy_score(yte, pred),
            "f1": f1_score(yte, pred, zero_division=0),
            "recall": recall_score(yte, pred, zero_division=0),
            "precision": precision_score(yte, pred, zero_division=0),
        })
        per_subject[test_subj] = rows[-1]["recall"]

    fold_df = pd.DataFrame(rows)
    return {
        "fold_df": fold_df,
        "per_subject_recall": per_subject,
        "mean_f1": float(fold_df["f1"].mean()),
        "std_f1": float(fold_df["f1"].std()),
        "mean_recall": float(fold_df["recall"].mean()),
        "std_recall": float(fold_df["recall"].std()),
        "mean_precision": float(fold_df["precision"].mean()),
        "mean_accuracy": float(fold_df["accuracy"].mean()),
        "min_subject_recall": float(fold_df["recall"].min()),
        "n_subjects_recall_below_0_5": int((fold_df["recall"] < 0.5).sum()),
        "n_subjects_recall_zero": int((fold_df["recall"] == 0).sum()),
    }


def hgb_factory(random_state: int = 42, **kwargs):
    """Default factory for the V3 fixed comparator: HistGradientBoosting.

    Scale-invariant (so z-score vs raw is a generalization-not-fitting question),
    handles NaN natively (so motion-gated HRV doesn't need imputation), and is
    V1's best-performing model. Use this as the comparator for normalization
    and feature-selection experiments.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier

    def factory():
        return HistGradientBoostingClassifier(
            max_iter=300, max_depth=4, learning_rate=0.05,
            class_weight="balanced", random_state=random_state, **kwargs,
        )
    return factory


def rf_factory(random_state: int = 42, **kwargs):
    """RandomForest factory matching Phase 2's settings, for direct comparison."""
    from sklearn.ensemble import RandomForestClassifier

    def factory():
        return RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_leaf=2,
            class_weight="balanced", random_state=random_state, n_jobs=-1, **kwargs,
        )
    return factory


def summarize_results(results_by_variant: dict[str, dict]) -> pd.DataFrame:
    """Turn a dict {variant_name: loso_evaluate(...)} into a tidy summary DataFrame."""
    rows = []
    for name, r in results_by_variant.items():
        rows.append({
            "variant": name,
            "mean_f1": r["mean_f1"],
            "std_f1": r["std_f1"],
            "mean_recall": r["mean_recall"],
            "std_recall": r["std_recall"],
            "mean_precision": r["mean_precision"],
            "mean_accuracy": r["mean_accuracy"],
            "min_subject_recall": r["min_subject_recall"],
            "n_subjects_recall_below_0_5": r["n_subjects_recall_below_0_5"],
            "n_subjects_recall_zero": r["n_subjects_recall_zero"],
        })
    return pd.DataFrame(rows).sort_values("mean_f1", ascending=False).reset_index(drop=True)


def per_subject_recall_matrix(results_by_variant: dict[str, dict]) -> pd.DataFrame:
    """Subjects × variants matrix of per-fold recalls."""
    return pd.DataFrame(
        {name: r["per_subject_recall"] for name, r in results_by_variant.items()}
    ).sort_index()
