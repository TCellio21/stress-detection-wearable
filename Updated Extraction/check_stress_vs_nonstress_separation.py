# -*- coding: utf-8 -*-
"""
Stress vs Non-Stress Separation Check
=====================================

For each subject, compares mean feature values on STRESS windows vs NON-STRESS
windows. If for a subject stress and non-stress look the same (no separation),
the model cannot learn to predict stress for that subject; if they separate
well, the issue is cross-subject generalization.

Usage:
    python "Updated Extraction/check_stress_vs_nonstress_separation.py"

Output:
    Printed table + CSV in Updated Extraction/results/
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_FILE = Path(__file__).parent / "all_subject_features_updated.csv"
RESULTS_DIR = Path(__file__).parent / "results"
OUTPUT_CSV = RESULTS_DIR / "stress_vs_nonstress_separation.csv"

METADATA_COLS = ["label", "subject_id", "window_idx", "raw_label", "hrv_valid", "hrv_n_peaks"]

# Features that typically discriminate stress (EDA-heavy; model uses these)
KEY_FEATURES = [
    "scr_peak_count",
    "scr_amplitude_sum",
    "scl_mean",
    "scr_peak_count_percent_change",
    "scr_amplitude_sum_percent_change",
    "scl_mean_z_score",
]


def main():
    if not DATA_FILE.exists():
        print(f"ERROR: Data file not found: {DATA_FILE}")
        return

    df = pd.read_csv(DATA_FILE)
    feature_cols = [c for c in df.columns if c not in METADATA_COLS]
    # Use only key features that exist
    feats = [f for f in KEY_FEATURES if f in feature_cols]

    print("=" * 70)
    print("STRESS vs NON-STRESS SEPARATION (within-subject)")
    print("=" * 70)
    print("\nFor each subject: mean(stress) - mean(non-stress) and effect size.")
    print("Positive = stress windows have higher values (expected for EDA).")
    print("Near zero or negative = stress does not separate -> model cannot learn.")
    print()

    rows = []
    for subj in sorted(df["subject_id"].unique()):
        subj_df = df[df["subject_id"] == subj]
        s_stress = subj_df.loc[subj_df["label"] == "stress", feats]
        s_non = subj_df.loc[subj_df["label"] == "non-stress", feats]
        n_stress = len(s_stress)
        n_non = len(s_non)
        if n_stress == 0:
            continue
        mean_stress = s_stress.mean()
        mean_non = s_non.mean()
        # Pooled std for effect size (Cohen's d per feature, then average absolute)
        std_stress = s_stress.std().replace(0, np.nan)
        std_non = s_non.std().replace(0, np.nan)
        pooled = np.sqrt((std_stress ** 2 + std_non ** 2) / 2)
        diff = mean_stress - mean_non
        d = (diff / pooled).fillna(0)
        # One summary: mean absolute Cohen's d across key features
        mean_effect = d.abs().mean()

        row = {
            "subject_id": subj,
            "n_stress": n_stress,
            "n_non_stress": n_non,
            "mean_effect_size_abs": round(mean_effect, 3),
        }
        for f in feats:
            row[f"diff_{f}"] = round(mean_stress[f] - mean_non[f], 4)
            row[f"d_{f}"] = round(d[f], 3)
        rows.append(row)

        flag = ""
        if mean_effect < 0.3:
            flag = "  [LOW SEPARATION - stress ~ non-stress]"
        elif "scr_peak_count" in feats and (mean_stress["scr_peak_count"] - mean_non["scr_peak_count"]) < 0:
            flag = "  [WRONG DIRECTION - stress lower than non-stress]"

        print(f"--- {subj} (n_stress={n_stress}, n_non={n_non}) ---")
        print(f"  Mean |effect size| across key features: {mean_effect:.3f}{flag}")
        for f in feats[:4]:  # First 4 in console
            d_val = d[f]
            diff_val = mean_stress[f] - mean_non[f]
            print(f"    {f}: diff={diff_val:+.3f}, Cohen d={d_val:+.2f}")
        if len(feats) > 4:
            print(f"    ... and {len(feats) - 4} more (see CSV)")
        print()

    result_df = pd.DataFrame(rows)

    # Highlight low-separation subjects (often 0 recall)
    low_sep = result_df[result_df["mean_effect_size_abs"] < 0.3]["subject_id"].tolist()
    if low_sep:
        print("=" * 70)
        print("SUBJECTS WITH LOW STRESS vs NON-STRESS SEPARATION (effect size < 0.3)")
        print("=" * 70)
        print(f"  {low_sep}")
        print("  These subjects' stress windows do not look clearly different from non-stress")
        print("  in the key features -> model tends to predict non-stress -> low/zero recall.")
    else:
        print("All subjects show some separation. If recall is still 0 for some, issue may be")
        print("cross-subject: model learned other subjects' scales that don't transfer.")

    RESULTS_DIR.mkdir(exist_ok=True)
    result_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved: {OUTPUT_CSV}")
    print("=" * 70)


if __name__ == "__main__":
    main()
