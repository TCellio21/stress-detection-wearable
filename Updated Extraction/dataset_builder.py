"""
WESAD Feature Extraction Pipeline - Dataset Builder
====================================================

Main orchestrator for the unified feature extraction pipeline.
Merges best practices from Grant and Tanner's approaches.

FIXES APPLIED:
1. Time Continuity: Windows continuous signals first, then filters by label
   (Tanner pre-masked signals which broke cross-sensor alignment)
2. TSST Prep Exclusion: Drops first 3 windows (180s) from stress condition
3. HRV Mapping: Correctly calculates HR from MeanNN (60000/MeanNN)
4. S14 Exclusion: Removed from subject list (hyporesponsive)
5. cvxEDA Only: No fallbacks - hard-fail ensures consistent decomposition
6. Label Mapping: Non-stress={1,3,4}, Stress={2}

OUTPUT: 46 raw features + 92 normalized = 138 total features

Usage:
    py -3.12 "Updated Extraction/dataset_builder.py"

Author: Merged pipeline (Grant + Tanner approaches)
"""

import pickle
import random
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import warnings
import json
from datetime import datetime

warnings.filterwarnings('ignore')

# Import our modules
from features import (
    extract_eda_features_window,
    extract_hrv_features_window,
    extract_temp_features_window,
    extract_acc_features_window,
)
from normalization import normalize_subject_features
from config_loader import load_config


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def align_labels_to_signal(signal_length, signal_fs, labels, label_fs):
    """
    Align high-rate labels (700 Hz) to lower-rate sensor using timestamp interpolation.
    
    This is Grant's method - uses searchsorted to find closest label timestamp
    for each signal sample. Preserves time alignment across all sensors.
    
    Args:
        signal_length: Number of samples in the signal
        signal_fs: Sampling frequency of the signal (Hz)
        labels: Label array at 700 Hz
        label_fs: Label sampling frequency (700 Hz)
    
    Returns:
        Aligned labels matching signal length
    """
    signal_times = np.arange(signal_length) / signal_fs
    label_times = np.arange(len(labels)) / label_fs
    label_indices = np.searchsorted(label_times, signal_times)
    label_indices = np.clip(label_indices, 0, len(labels) - 1)
    return labels[label_indices]


def get_window_label(labels_window):
    """
    Get majority label for a window using mode.
    
    Args:
        labels_window: Label array for this window
    
    Returns:
        Most common label in the window
    """
    mode_result = stats.mode(labels_window, keepdims=True)
    return mode_result.mode[0]


# =============================================================================
# MAIN PROCESSING FUNCTION
# =============================================================================

def process_subject(subject_id, config):
    """
    Process single subject: load data, extract features, return DataFrame.

    Pipeline:
    1. Load pkl file and extract wrist signals
    2. Align labels to EDA timebase (4 Hz)
    3. Create windows from CONTINUOUS signals (preserves time alignment)
    4. Filter windows by label (drop label=0)
    5. Exclude TSST preparation phase (first 3 stress windows)
    6. Extract features for each valid window
    7. Apply two-pass normalization

    Args:
        subject_id: Subject ID (e.g., 'S2')
        config: Configuration dict from load_config()

    Returns:
        DataFrame with 138 features per window
    """
    paths = config["paths"]
    sr = config["sampling_rates"]
    win = config["windowing"]
    eda_cfg = config["eda"]
    tsst = config["tsst"]
    label_mapping = config["labels"]["mapping"]

    wesad_path = paths["wesad_path"]
    fs_eda = sr["eda"]
    fs_bvp = sr["bvp"]
    fs_temp = sr["temp"]
    fs_acc = sr["acc"]
    fs_label = sr["label"]
    window_size = win["window_size"]
    step_size = win["step_size"]
    prep_windows = tsst["prep_duration"] // window_size

    window_samples_eda = window_size * fs_eda
    window_samples_bvp = window_size * fs_bvp
    window_samples_temp = window_size * fs_temp
    window_samples_acc = window_size * fs_acc

    print(f"\nProcessing {subject_id}...")

    # Load data
    file_path = Path(wesad_path) / subject_id / f"{subject_id}.pkl"
    with open(file_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")

    # Extract wrist signals
    eda_raw = data["signal"]["wrist"]["EDA"].flatten()
    bvp_raw = data["signal"]["wrist"]["BVP"].flatten()
    temp_raw = data["signal"]["wrist"]["TEMP"].flatten()
    acc_raw = data["signal"]["wrist"]["ACC"]
    labels = data["label"]

    # Align labels to EDA timebase (4 Hz) - this is the reference
    labels_aligned = align_labels_to_signal(len(eda_raw), fs_eda, labels, fs_label)

    num_windows = len(eda_raw) // window_samples_eda

    print(f"  Signal lengths: EDA={len(eda_raw)}, BVP={len(bvp_raw)}, TEMP={len(temp_raw)}, ACC={len(acc_raw)}")
    print(f"  Total possible windows: {num_windows}")

    # Process each window
    features_list = []
    raw_labels_list = []
    stress_window_count = 0  # Track stress windows for TSST prep exclusion

    for w in range(num_windows):
        # Calculate indices for each modality
        # KEY: Use time-aligned indices, not pre-masked signals
        eda_start = w * window_samples_eda
        eda_end = eda_start + window_samples_eda

        bvp_start = w * window_samples_bvp
        bvp_end = bvp_start + window_samples_bvp

        temp_start = w * window_samples_temp
        temp_end = temp_start + window_samples_temp

        acc_start = w * window_samples_acc
        acc_end = acc_start + window_samples_acc

        # Check bounds
        if (eda_end > len(eda_raw) or bvp_end > len(bvp_raw) or
            temp_end > len(temp_raw) or acc_end > len(acc_raw)):
            break

        # Get window label (mode of aligned labels for this EDA window)
        window_label = get_window_label(labels_aligned[eda_start:eda_end])

        # Skip labels not in our mapping (0=transient, 5,6,7=ignore)
        if window_label not in label_mapping:
            continue

        # TSST prep exclusion: skip first 3 windows of stress condition
        if window_label == 2:  # stress
            stress_window_count += 1
            if stress_window_count <= prep_windows:
                continue  # Skip preparation phase

        # Extract window data (time-aligned across all sensors)
        eda_window = eda_raw[eda_start:eda_end]
        bvp_window = bvp_raw[bvp_start:bvp_end]
        temp_window = temp_raw[temp_start:temp_end]
        acc_x = acc_raw[acc_start:acc_end, 0]
        acc_y = acc_raw[acc_start:acc_end, 1]
        acc_z = acc_raw[acc_start:acc_end, 2]

        # Extract features from each modality
        try:
            eda_features = extract_eda_features_window(
                eda_window, fs_eda, eda_cfg["peak_threshold"],
                eda_cfg["outlier_min"], eda_cfg["outlier_max"]
            )
        except Exception as e:
            print(f"  WARNING: EDA extraction failed for window {w}: {e}")
            continue

        hrv_features = extract_hrv_features_window(bvp_window, fs_bvp)
        temp_features = extract_temp_features_window(temp_window, fs_temp)
        acc_features = extract_acc_features_window(acc_x, acc_y, acc_z, fs_acc)

        # Combine all features
        window_features = {}
        window_features.update(eda_features)
        window_features.update(hrv_features)
        window_features.update(temp_features)
        window_features.update(acc_features)

        # Add metadata
        window_features["subject_id"] = subject_id
        window_features["window_idx"] = w
        window_features["raw_label"] = window_label  # Keep original for normalization
        window_features["label"] = label_mapping[window_label]  # Binary label

        features_list.append(window_features)
        raw_labels_list.append(window_label)
    
    # Convert to DataFrame
    subject_df = pd.DataFrame(features_list)
    raw_labels = np.array(raw_labels_list)
    
    # Count windows by class
    nonstress_count = sum(1 for l in raw_labels if l in [1, 3, 4])
    stress_count = sum(1 for l in raw_labels if l == 2)
    print(f"  Extracted {len(subject_df)} windows ({nonstress_count} non-stress, {stress_count} stress)")
    
    # Apply two-pass normalization
    print(f"  Normalizing features (baseline stats from label=1 only)...")
    normalized_df = normalize_subject_features(subject_df, raw_labels)
    
    # Count features
    raw_cols = [c for c in normalized_df.columns if not c.endswith(('_percent_change', '_z_score'))
                and c not in ['subject_id', 'window_idx', 'raw_label', 'label']]
    print(f"  Features: {len(raw_cols)} raw → {len(normalized_df.columns) - 4} total")
    
    return normalized_df


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main entry point for the feature extraction pipeline."""
    config = load_config()
    config_source = config.get("_config_source", "defaults")

    # Set random seed for reproducibility
    seed = config["reproducibility"]["random_seed"]
    np.random.seed(seed)
    random.seed(seed)

    paths = config["paths"]
    subjects = config["subjects"]["include"]
    win = config["windowing"]
    tsst = config["tsst"]

    output_dir = Path(__file__).parent / paths["output_dir"]
    output_file = paths["output_file"]
    prep_duration = tsst["prep_duration"]
    prep_windows = prep_duration // win["window_size"]

    print("=" * 70)
    print("WESAD Feature Extraction Pipeline (Merged)")
    print("=" * 70)

    print(f"\nConfiguration (source: {config_source}):")
    print(f"  Dataset path: {paths['wesad_path']}")
    print(f"  Subjects: {len(subjects)} (S14 excluded)")
    print(f"  Window size: {win['window_size']}s (non-overlapping)")
    print(f"  Random seed: {seed}")
    print(f"  Raw features: 46 (17 EDA + 15 HRV + 6 Temp + 8 Acc)")
    print(f"  Normalized features: 92 (46 × 2: percent_change + z_score)")
    print(f"  Total features: 138")
    print(f"  TSST prep exclusion: {prep_duration}s ({prep_windows} windows)")

    print(f"\nFixes applied:")
    print(f"  [x] Time continuity: Window continuous signals, then filter")
    print(f"  [x] TSST prep exclusion: Drop first 3 stress windows")
    print(f"  [x] HRV mean_hr: Correctly calculated as 60000/MeanNN")
    print(f"  [x] S14 excluded: Hyporesponsive subject removed")
    print(f"  [x] cvxEDA only: No fallbacks (consistent decomposition)")
    print(f"  [x] Label mapping: non-stress={{1,3,4}}, stress={{2}}")

    print(f"\nClassification Task:")
    print(f"  Binary: NON-STRESS vs STRESS")
    print(f"  Non-Stress: Baseline + Amusement + Meditation")
    print(f"  Stress: TSST (excluding prep phase)")

    # Process all subjects
    all_subject_features = []

    for subject_id in subjects:
        try:
            subject_features = process_subject(subject_id, config)
            all_subject_features.append(subject_features)
        except Exception as e:
            print(f"  ERROR processing {subject_id}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Combine all subjects
    if not all_subject_features:
        print(f"\n{'=' * 70}")
        print("No subjects were processed (check WESAD path and config). Exiting.")
        return

    print(f"\n{'=' * 70}")
    print("Combining all subjects...")
    final_df = pd.concat(all_subject_features, ignore_index=True)

    # Reorder columns: features first, then metadata
    metadata_cols = ["label", "subject_id", "window_idx", "raw_label"]
    feature_cols = [col for col in final_df.columns if col not in metadata_cols]
    final_df = final_df[feature_cols + metadata_cols]

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save to CSV
    output_path = output_dir / output_file
    final_df.to_csv(output_path, index=False)

    # Also save a copy in the Updated Extraction folder for convenience
    local_copy = Path(__file__).parent / output_file
    final_df.to_csv(local_copy, index=False)

    # Save manifest
    manifest = {
        "created": datetime.now().isoformat(),
        "random_seed": seed,
        "config_source": config_source,
        "subjects": subjects,
        "total_windows": len(final_df),
        "total_features": len(feature_cols),
        "raw_features": len([c for c in feature_cols if not c.endswith(("_percent_change", "_z_score"))]),
        "normalized_features": len([c for c in feature_cols if c.endswith(("_percent_change", "_z_score"))]),
        "class_distribution": final_df["label"].value_counts().to_dict(),
        "windows_per_subject": final_df.groupby("subject_id").size().to_dict(),
        "config": {
            "window_size": win["window_size"],
            "step_size": win["step_size"],
            "tsst_prep_excluded": prep_duration,
            "cvxeda_only": True,
            "s14_excluded": True,
        },
    }

    manifest_path = output_dir / "run_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    
    # Print summary
    print(f"\n{'=' * 70}")
    print("EXTRACTION COMPLETE!")
    print(f"{'=' * 70}")
    print(f"\nDataset Summary:")
    print(f"  Total windows: {len(final_df)}")
    print(f"  Total features: {len(feature_cols)}")
    print(f"  Subjects: {final_df['subject_id'].nunique()}")
    print(f"\nClass distribution:")
    print(final_df["label"].value_counts())
    print(f"\nOutput files:")
    print(f"  {output_path}")
    print(f"  {local_copy}")
    print(f"  {output_dir / 'run_manifest.json'}")


if __name__ == "__main__":
    main()

