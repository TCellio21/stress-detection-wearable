"""
Phase 2 — windowing/step sweep with LOSO Random Forest baseline.

Sweeps (W, step) ∈ {30,60,90,120}s × {15,30,60}s with the constraint that
step ≤ W (no gaps between windows). Picks the (W, step) maximizing F1 with
no per-subject recall < 0.5.

Per-config pipeline:
  preprocess each subject (cached across configs)
  → window/feature with V3 features
  → per-subject calibration normalization (z-score against label=1 windows)
  → fill NaN HRV (motion-gated) with subject median
  → LeaveOneSubjectOut RandomForest, class_weight='balanced'
  → metrics: F1/recall/accuracy, per-subject recall

Outputs (under reports/02_windowing/):
  - results.csv          (per-(W,step) aggregate metrics)
  - per_subject_recall.csv (recall heatmap data: subject × config)
  - heatmap.png          (F1 over W × step grid)
  - per_subject_heatmap.png (recall heatmap)
"""

from __future__ import annotations

import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import LeaveOneGroupOut

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_SCRIPT_DIR))

from config_loader import load_config  # noqa: E402
import dataset_builder as db  # noqa: E402
import features as feats_mod  # noqa: E402

REPORTS_DIR = _REPO_ROOT / "reports" / "02_windowing"

WINDOW_OPTIONS = [30, 60, 90, 120]
STEP_OPTIONS = [15, 30, 60]
RANDOM_SEED = 42


def fill_subject_median(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Per-subject median imputation for NaN feature columns (motion-gated HRV)."""
    out = df.copy()
    for sid, idx in out.groupby("subject_id").groups.items():
        sub = out.loc[idx, feature_cols]
        med = sub.median(skipna=True)
        out.loc[idx, feature_cols] = sub.fillna(med)
    # Any column still all-NaN within a subject becomes 0
    out[feature_cols] = out[feature_cols].fillna(0.0)
    return out


def loso_rf_eval(df: pd.DataFrame, feature_cols: list[str], seed: int = RANDOM_SEED) -> dict:
    X = df[feature_cols].values
    y = (df["label"].values == "stress").astype(int)
    groups = df["subject_id"].values

    logo = LeaveOneGroupOut()
    fold_metrics = []
    per_subject = {}

    for train_idx, test_idx in logo.split(X, y, groups):
        test_subj = groups[test_idx[0]]
        Xtr, Xte = X[train_idx], X[test_idx]
        ytr, yte = y[train_idx], y[test_idx]

        clf = RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_leaf=2,
            class_weight="balanced", random_state=seed, n_jobs=-1,
        )
        clf.fit(Xtr, ytr)
        pred = clf.predict(Xte)

        fold = {
            "subject": test_subj,
            "n_test": len(yte),
            "n_stress": int(yte.sum()),
            "accuracy": accuracy_score(yte, pred),
            "f1": f1_score(yte, pred, zero_division=0),
            "recall": recall_score(yte, pred, zero_division=0),
            "precision": precision_score(yte, pred, zero_division=0),
        }
        fold_metrics.append(fold)
        per_subject[test_subj] = fold["recall"]

    fold_df = pd.DataFrame(fold_metrics)
    return {
        "fold_df": fold_df,
        "per_subject_recall": per_subject,
        "mean_f1": float(fold_df["f1"].mean()),
        "std_f1": float(fold_df["f1"].std()),
        "mean_recall": float(fold_df["recall"].mean()),
        "std_recall": float(fold_df["recall"].std()),
        "mean_accuracy": float(fold_df["accuracy"].mean()),
        "mean_precision": float(fold_df["precision"].mean()),
        "n_subjects_recall_below_0_5": int((fold_df["recall"] < 0.5).sum()),
        "n_subjects_recall_zero": int((fold_df["recall"] == 0).sum()),
        "min_subject_recall": float(fold_df["recall"].min()),
    }


def run_one_config(window_sec: int, step_sec: int, cfg: dict, subjects: list[str],
                   wesad_path: str, cache: dict) -> dict:
    t0 = time.time()
    df = db.build_full_dataset(subjects, wesad_path, cfg, window_sec, step_sec,
                               label_rule="majority", cache=cache)
    if len(df) == 0:
        return {"window_sec": window_sec, "step_sec": step_sec, "n_windows": 0, "error": "empty"}

    feature_cols_z = [f"{c}_z" for c in feats_mod.ALL_FEATURES if f"{c}_z" in df.columns]
    df = fill_subject_median(df, feature_cols_z)

    metrics = loso_rf_eval(df, feature_cols_z)
    elapsed = time.time() - t0
    n_stress = int((df["label"] == "stress").sum())
    n_total = len(df)
    return {
        "window_sec": window_sec, "step_sec": step_sec,
        "n_windows": n_total, "n_stress_windows": n_stress,
        "stress_fraction": n_stress / n_total,
        "n_features": len(feature_cols_z),
        "mean_f1": metrics["mean_f1"], "std_f1": metrics["std_f1"],
        "mean_recall": metrics["mean_recall"], "std_recall": metrics["std_recall"],
        "mean_accuracy": metrics["mean_accuracy"],
        "mean_precision": metrics["mean_precision"],
        "min_subject_recall": metrics["min_subject_recall"],
        "n_subjects_recall_below_0_5": metrics["n_subjects_recall_below_0_5"],
        "n_subjects_recall_zero": metrics["n_subjects_recall_zero"],
        "per_subject_recall": metrics["per_subject_recall"],
        "elapsed_sec": round(elapsed, 1),
    }


def plot_heatmaps(results_df: pd.DataFrame, per_subj_df: pd.DataFrame) -> None:
    pivot_f1 = results_df.pivot(index="window_sec", columns="step_sec", values="mean_f1")
    pivot_rec = results_df.pivot(index="window_sec", columns="step_sec", values="mean_recall")
    pivot_min = results_df.pivot(index="window_sec", columns="step_sec", values="min_subject_recall")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, pv, title in zip(
        axes,
        [pivot_f1, pivot_rec, pivot_min],
        ["mean F1 (LOSO)", "mean recall (LOSO)", "min per-subject recall"],
    ):
        im = ax.imshow(pv.values, aspect="auto", cmap="viridis", vmin=0, vmax=1)
        ax.set_xticks(range(len(pv.columns))); ax.set_xticklabels(pv.columns)
        ax.set_yticks(range(len(pv.index))); ax.set_yticklabels(pv.index)
        ax.set_xlabel("step (s)"); ax.set_ylabel("window (s)")
        ax.set_title(title)
        for i in range(pv.shape[0]):
            for j in range(pv.shape[1]):
                v = pv.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                            color="white" if v < 0.5 else "black", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "heatmap.png", dpi=120)
    plt.close(fig)

    # Per-subject recall heatmap (rows = subject, cols = config "WxS")
    fig, ax = plt.subplots(figsize=(max(8, 0.6 * len(per_subj_df.columns)), 6))
    im = ax.imshow(per_subj_df.values, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(per_subj_df.columns))); ax.set_xticklabels(per_subj_df.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(per_subj_df.index))); ax.set_yticklabels(per_subj_df.index)
    ax.set_xlabel("config (WxS in seconds)"); ax.set_ylabel("subject")
    ax.set_title("Per-subject stress recall — LOSO RandomForest")
    for i in range(per_subj_df.shape[0]):
        for j in range(per_subj_df.shape[1]):
            v = per_subj_df.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if v < 0.5 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.025)
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "per_subject_heatmap.png", dpi=120)
    plt.close(fig)


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    subjects = cfg["subjects"]["include"]
    wesad_path = cfg["paths"]["wesad_path"]

    print("=" * 70)
    print("Phase 2 — windowing/step sweep")
    print("=" * 70)
    print(f"Subjects:       {len(subjects)} (S14 excluded)")
    print(f"Feature count:  {feats_mod.feature_count()} (each gets a _z companion)")
    print(f"Window options: {WINDOW_OPTIONS}")
    print(f"Step options:   {STEP_OPTIONS} (will skip configs with step > window)")
    print(f"Outputs:        {REPORTS_DIR.relative_to(_REPO_ROOT)}")
    print()

    print("Preprocessing all subjects (cached for the sweep)...")
    cache = {}
    for s in subjects:
        t0 = time.time()
        cache[s] = db.preprocess_subject(s, wesad_path, cfg)
        print(f"  {s}: {time.time() - t0:.1f}s")
    print()

    rows = []
    per_subject_rows = {s: {} for s in subjects}

    configs = [(w, st) for w in WINDOW_OPTIONS for st in STEP_OPTIONS if st <= w]
    print(f"Running {len(configs)} configurations:\n")
    for w, st in configs:
        print(f"--- W={w}s, step={st}s ---")
        res = run_one_config(w, st, cfg, subjects, wesad_path, cache)
        rows.append(res)
        cfg_label = f"W{w}s{st}"
        for sid, recall in res.get("per_subject_recall", {}).items():
            per_subject_rows.setdefault(sid, {})[cfg_label] = recall
        print(f"  windows={res.get('n_windows')}, stress_frac={res.get('stress_fraction', float('nan')):.3f}, "
              f"F1={res.get('mean_f1', float('nan')):.3f}±{res.get('std_f1', float('nan')):.3f}, "
              f"recall={res.get('mean_recall', float('nan')):.3f}±{res.get('std_recall', float('nan')):.3f}, "
              f"min_subj_recall={res.get('min_subject_recall', float('nan')):.3f}, "
              f"<0.5: {res.get('n_subjects_recall_below_0_5')}/14, "
              f"=0: {res.get('n_subjects_recall_zero')}/14, "
              f"{res.get('elapsed_sec')}s")
        print()

    summary = pd.DataFrame([{k: v for k, v in r.items() if k != "per_subject_recall"} for r in rows])
    summary.to_csv(REPORTS_DIR / "results.csv", index=False)

    per_subject_df = pd.DataFrame(per_subject_rows).T
    per_subject_df.to_csv(REPORTS_DIR / "per_subject_recall.csv")

    plot_heatmaps(summary, per_subject_df)

    print("=" * 70)
    print("Summary (sorted by F1):")
    pretty = summary.sort_values("mean_f1", ascending=False).copy()
    cols_to_show = ["window_sec", "step_sec", "n_windows", "stress_fraction",
                    "mean_f1", "std_f1", "mean_recall", "min_subject_recall",
                    "n_subjects_recall_below_0_5", "n_subjects_recall_zero", "elapsed_sec"]
    with pd.option_context("display.max_columns", None, "display.width", 160):
        print(pretty[cols_to_show].to_string(index=False))
    print()

    print("Recommendation:")
    feasible = summary[summary["n_subjects_recall_zero"] == 0].sort_values("mean_f1", ascending=False)
    if len(feasible):
        best = feasible.iloc[0]
        print(f"  Best config with no subject zero-recall: W={int(best['window_sec'])}s, step={int(best['step_sec'])}s")
        print(f"    F1={best['mean_f1']:.3f}±{best['std_f1']:.3f}, recall={best['mean_recall']:.3f}, "
              f"min_subj_recall={best['min_subject_recall']:.3f}")
    else:
        print("  No config achieved zero-failure across all subjects. Picking by F1:")
        best = summary.sort_values("mean_f1", ascending=False).iloc[0]
        print(f"    W={int(best['window_sec'])}s, step={int(best['step_sec'])}s, F1={best['mean_f1']:.3f}, "
              f"min_subj_recall={best['min_subject_recall']:.3f}")
    print()
    print("Done.")


if __name__ == "__main__":
    main()
