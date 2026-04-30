# -*- coding: utf-8 -*-
"""
Baseline Window Count Check
===========================

Checks per-subject window counts (baseline, stress, non-stress) from the
feature CSV. Used to find subjects with no baseline windows, which can cause
92 NaNs in normalized features and 0 recall in LOSO (see UPDATED_PIPELINE_ANALYSIS.md).

Usage:
    python "Updated Extraction/check_baseline_windows.py"

Output:
    Printed table + optional CSV in Updated Extraction/results/
"""

import pandas as pd
from pathlib import Path

# Paths
DATA_FILE = Path(__file__).parent / "all_subject_features_updated.csv"
RESULTS_DIR = Path(__file__).parent / "results"
OUTPUT_CSV = RESULTS_DIR / "baseline_window_counts.csv"

# raw_label: 1=baseline, 2=stress, 3=amusement, 4=meditation
LABEL_BASELINE = 1
LABEL_STRESS = 2
LABEL_AMUSEMENT = 3
LABEL_MEDITATION = 4


def main():
    if not DATA_FILE.exists():
        print(f"ERROR: Data file not found: {DATA_FILE}")
        return

    df = pd.read_csv(DATA_FILE)

    if "raw_label" not in df.columns:
        print("ERROR: Column 'raw_label' not found. Cannot compute baseline counts.")
        return

    # Build per-subject counts
    subjects = sorted(df["subject_id"].unique())
    rows = []

    for subj in subjects:
        subj_df = df[df["subject_id"] == subj]
        n_baseline = (subj_df["raw_label"] == LABEL_BASELINE).sum()
        n_stress = (subj_df["raw_label"] == LABEL_STRESS).sum()
        n_amusement = (subj_df["raw_label"] == LABEL_AMUSEMENT).sum()
        n_meditation = (subj_df["raw_label"] == LABEL_MEDITATION).sum()
        n_non_stress = n_baseline + n_amusement + n_meditation
        n_total = len(subj_df)

        # Stress windows after TSST prep drop: first 3 stress windows are excluded
        # In the CSV, stress windows are already post-drop, so n_stress is as-is
        no_baseline = n_baseline == 0

        rows.append({
            "subject_id": subj,
            "baseline": n_baseline,
            "stress": n_stress,
            "amusement": n_amusement,
            "meditation": n_meditation,
            "non_stress_total": n_non_stress,
            "total_windows": n_total,
            "no_baseline": no_baseline,
        })

    result_df = pd.DataFrame(rows)

    # Optional: count NaNs in normalized columns for subjects with no baseline
    meta = ["label", "subject_id", "window_idx", "raw_label", "hrv_valid", "hrv_n_peaks"]
    feature_cols = [c for c in df.columns if c not in meta]
    # Normalized columns typically end with _z_score or _percent_change
    normalized_cols = [c for c in feature_cols if c.endswith("_z_score") or c.endswith("_percent_change")]
    if normalized_cols:
        nan_counts = df.groupby("subject_id")[normalized_cols].apply(
            lambda x: x.isna().all(axis=1).sum()
        )
        result_df["windows_with_all_normalized_nan"] = result_df["subject_id"].map(nan_counts)

    # Print report
    print("=" * 70)
    print("BASELINE WINDOW COUNT PER SUBJECT")
    print("=" * 70)
    print(f"\nData file: {DATA_FILE.name}")
    print(f"Subjects: {len(subjects)}")
    print()
    print(f"{'Subject':>8} | {'Baseline':>8} | {'Stress':>6} | {'Amuse':>6} | {'Medit':>6} | {'NonStr':>6} | {'Total':>6} | No baseline?")
    print("-" * 75)

    for _, r in result_df.iterrows():
        flag = " ** NO BASELINE **" if r["no_baseline"] else ""
        print(f"{r['subject_id']:>8} | {r['baseline']:>8} | {r['stress']:>6} | {r['amusement']:>6} | {r['meditation']:>6} | {r['non_stress_total']:>6} | {r['total_windows']:>6} |{flag}")

    if "windows_with_all_normalized_nan" in result_df.columns:
        print()
        print("Windows with all normalized features NaN (per subject):")
        for _, r in result_df.iterrows():
            n_nan = r["windows_with_all_normalized_nan"]
            if n_nan > 0:
                print(f"  {r['subject_id']}: {int(n_nan)} windows")

    # Warnings
    no_baseline_subjects = result_df[result_df["no_baseline"]]["subject_id"].tolist()
    if no_baseline_subjects:
        print()
        print("=" * 70)
        print("WARNING: Subjects with ZERO baseline windows (raw_label=1)")
        print("=" * 70)
        print(f"  {no_baseline_subjects}")
        print("  These subjects get no normalized features -> 92 NaNs -> median imputation")
        print("  at test time -> model tends to predict non-stress -> 0 RECALL.")
        print("  See UPDATED_PIPELINE_ANALYSIS.md and fix normalization or exclude these subjects.")
    else:
        print()
        print("No subjects with zero baseline windows.")

    # Save
    RESULTS_DIR.mkdir(exist_ok=True)
    result_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved: {OUTPUT_CSV}")
    print("=" * 70)


if __name__ == "__main__":
    main()
