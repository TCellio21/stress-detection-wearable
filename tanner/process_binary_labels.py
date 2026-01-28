"""
Binary Label Data Processing Script - Maximum Data Strategy

This script processes the WESAD dataset to create binary stress classification:
- Uses ALL affective states for maximum data utilization
- Non-stress: Baseline (1) + Amusement (3) + Meditation (4)
- Stress: TSST (2)
- Extracts wrist-only features for deployment on wearable devices

Binary Mapping:
- Stress (label 2) -> 1
- Non-stress (labels 1, 3, 4) -> 0 (baseline, amusement, meditation)

Rationale:
- Maximum data diversity (~550-650 windows)
- Most representative of real-world non-stress states
- Better generalization across different calm/positive states
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
import yaml
from datetime import datetime

# Add src to path
sys.path.append('src')

from data_loader import WESADDataLoader
from preprocessing import SignalPreprocessor, E4_Converter
from feature_extraction import MultimodalFeatureExtractor
from config import load_config


def load_binary_mapping(config_path: Path) -> dict:
    """Load binary label mapping from config file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config['labels']['binary_mapping']


def process_subject(loader: WESADDataLoader, subject_id: str,
                   binary_mapping: dict, window_size: int = 60,
                   step_size: int = 60) -> tuple:
    """
    Process a single subject's data with binary labels.

    Args:
        loader: WESADDataLoader instance
        subject_id: Subject ID (e.g., 'S2')
        binary_mapping: Dictionary mapping original labels to binary
        window_size: Window size in seconds
        step_size: Step size in seconds

    Returns:
        Tuple of (features_df, labels_binary, original_labels, metadata)
    """
    print(f"\nProcessing {subject_id}...")

    # Load data
    pkl_data, metadata, quest_df = loader.load_subject_data(subject_id)

    # Extract wrist signals (for deployment on wrist device)
    wrist_signals = loader.extract_wrist_signals(pkl_data)
    labels = loader.get_labels(pkl_data)

    # Convert to SI units
    wrist_signals_si = {
        'BVP': E4_Converter.convert_bvp(wrist_signals['BVP']),
        'EDA': E4_Converter.convert_eda(wrist_signals['EDA']),
        'TEMP': E4_Converter.convert_temp(wrist_signals['TEMP']),
        'ACC': E4_Converter.convert_acc(wrist_signals['ACC'])
    }

    # Downsample labels to match EDA sampling rate FIRST (4 Hz)
    # Labels are at 700 Hz (chest device rate), wrist EDA is at 4 Hz
    fs_eda = 4  # Hz
    fs_labels = 700  # Hz
    downsample_factor = fs_labels // fs_eda  # 175

    # Downsample labels to match EDA length
    labels_downsampled = labels[::downsample_factor][:len(wrist_signals_si['EDA'])]

    # Filter to relevant labels (1-4: baseline, stress, amusement, meditation)
    # Create mask for relevant labels
    mask = np.isin(labels_downsampled, loader.RELEVANT_LABELS)

    # Filter each wrist signal
    filtered_signals = {}
    for key, signal in wrist_signals_si.items():
        if signal.ndim == 1:
            # For 1D signals (BVP, EDA, TEMP), use mask directly
            if key == 'EDA' or key == 'TEMP':
                # EDA and TEMP are at 4 Hz, same as downsampled labels
                filtered_signals[key] = signal[mask]
            elif key == 'BVP':
                # BVP is at 64 Hz, need to resample mask
                bvp_factor = 64 // 4  # 16
                bvp_mask = np.repeat(mask, bvp_factor)[:len(signal)]
                filtered_signals[key] = signal[bvp_mask]
        else:  # Multi-dimensional (ACC)
            # ACC is at 32 Hz
            acc_factor = 32 // 4  # 8
            acc_mask = np.repeat(mask, acc_factor)[:len(signal)]
            filtered_signals[key] = signal[acc_mask, :]

    # Filter labels
    filtered_labels = labels_downsampled[mask]

    # Segment signals into windows
    preprocessor = SignalPreprocessor()

    # Segment EDA (determines window positions)
    eda_segments, segment_labels_list = preprocessor.segment_signal(
        filtered_signals['EDA'], filtered_labels, window_size, step_size, fs_eda
    )

    # Convert segment_labels from list to numpy array
    segment_labels = np.array(segment_labels_list)

    print(f"  Created {len(eda_segments)} windows")
    print(f"  Original label distribution (filtered): {dict(zip(*np.unique(segment_labels, return_counts=True)))}")

    # Convert to binary labels
    segment_labels_binary = loader.convert_to_binary_stress_labels(
        segment_labels, mapping=binary_mapping
    )

    print(f"  Binary label distribution: {dict(zip(*np.unique(segment_labels_binary, return_counts=True)))}")

    # Extract features from each window
    features_list = []

    # Sampling frequencies for wrist signals
    fs_dict = {
        'BVP': 64,
        'EDA': 4,
        'ACC': 32,
        'TEMP': 4
    }

    for idx in range(len(segment_labels)):
        # Calculate indices for each signal based on their sampling rates
        start_eda = idx * step_size * fs_eda
        end_eda = start_eda + window_size * fs_eda

        start_bvp = idx * step_size * 64
        end_bvp = start_bvp + window_size * 64

        start_acc = idx * step_size * 32
        end_acc = start_acc + window_size * 32

        # Extract signals for this window
        window_signals = {
            'BVP': filtered_signals['BVP'][start_bvp:min(end_bvp, len(filtered_signals['BVP']))],
            'EDA': filtered_signals['EDA'][start_eda:min(end_eda, len(filtered_signals['EDA']))],
            'ACC': filtered_signals['ACC'][start_acc:min(end_acc, len(filtered_signals['ACC']))],
            'TEMP': filtered_signals['TEMP'][start_eda:min(end_eda, len(filtered_signals['TEMP']))]
        }

        # Extract features
        try:
            features = MultimodalFeatureExtractor.extract_features_from_window(
                window_signals, fs_dict, device='wrist'
            )

            # Add metadata
            features['subject_id'] = subject_id
            features['window_idx'] = idx
            features['label_binary'] = segment_labels_binary[idx]
            features['label_original'] = segment_labels[idx]

            features_list.append(features)

        except Exception as e:
            print(f"  Warning: Failed to extract features for window {idx}: {e}")
            continue

    # Convert to DataFrame
    features_df = pd.DataFrame(features_list)

    print(f"  Extracted features for {len(features_df)} windows")
    print(f"  Feature count: {len(features_df.columns) - 4} features (+ 4 metadata columns)")

    return features_df, segment_labels_binary, segment_labels, metadata


def main():
    """Main processing pipeline."""
    print("="*80)
    print("WESAD Binary Stress Classification - Data Processing")
    print("="*80)
    print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Setup paths
    project_root = Path(__file__).parent.parent
    config_path = project_root / 'config.yaml'
    dataset_path = project_root / 'data' / 'raw'
    output_dir = project_root / 'data' / 'processed'

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load configuration
    print("\n[1/4] Loading configuration...")
    binary_mapping = load_binary_mapping(config_path)
    print(f"  Binary mapping: {binary_mapping}")
    print(f"  Stress (1): label 2")
    print(f"  Non-stress (0): labels 1, 3, 4")

    # Initialize data loader
    print("\n[2/4] Initializing data loader...")
    loader = WESADDataLoader(dataset_path)
    print(f"  Found {len(loader.subjects)} subjects: {loader.subjects}")

    if len(loader.subjects) == 0:
        print("ERROR: No subjects found. Please check dataset path.")
        return

    # Process all subjects
    print("\n[3/4] Processing subjects...")
    all_features = []
    subject_summaries = []

    for subject_id in loader.subjects:
        try:
            features_df, binary_labels, original_labels, metadata = process_subject(
                loader, subject_id, binary_mapping
            )

            all_features.append(features_df)

            # Create subject summary
            summary = {
                'subject_id': subject_id,
                'total_windows': len(features_df),
                'stress_windows': int(np.sum(binary_labels == 1)),
                'non_stress_windows': int(np.sum(binary_labels == 0)),
                'stress_percentage': float(np.mean(binary_labels == 1) * 100),
                'age': metadata.get('Age', 'N/A'),
                'gender': metadata.get('Gender', 'N/A')
            }
            subject_summaries.append(summary)

        except Exception as e:
            print(f"  ERROR processing {subject_id}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Combine all features
    print("\n[4/4] Normalizing and saving data...")
    combined_features = pd.concat(all_features, ignore_index=True)

    # Apply subject-specific normalization (Grant's approach)
    print("\nApplying subject-specific normalization...")
    print("  Method: Z-score + Percent change (Grant's approach)")
    from feature_extraction import SubjectNormalizer

    # Use robust=False to match Grant's exact method (mean/std)
    # Set robust=True for improved outlier resistance
    combined_features = SubjectNormalizer.normalize_features(
        combined_features,
        use_robust=False  # False = Grant's method (mean/std), True = robust (median/IQR)
    )

    # Save combined dataset
    output_file = output_dir / 'wesad_binary_stress_wrist_features.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(combined_features, f)
    print(f"  Saved: {output_file}")

    # Also save as CSV (for easier inspection)
    csv_file = output_dir / 'wesad_binary_stress_wrist_features.csv'
    combined_features.to_csv(csv_file, index=False)
    print(f"  Saved: {csv_file}")

    # Save subject summaries
    summary_df = pd.DataFrame(subject_summaries)
    summary_file = output_dir / 'subject_summaries_binary.csv'
    summary_df.to_csv(summary_file, index=False)
    print(f"  Saved: {summary_file}")

    # Print overall statistics
    print("\n" + "="*80)
    print("PROCESSING COMPLETE - Summary Statistics")
    print("="*80)

    print(f"\nTotal subjects processed: {len(all_features)}")
    print(f"Total windows: {len(combined_features)}")
    print(f"Total features per window: {len(combined_features.columns) - 4}")

    print("\n--- Binary Label Distribution ---")
    print("Strategy: Maximum Data (Baseline + Amusement + Meditation vs Stress)")
    binary_counts = combined_features['label_binary'].value_counts()
    print(f"Non-stress (0): {binary_counts.get(0, 0)} windows ({binary_counts.get(0, 0)/len(combined_features)*100:.2f}%)")
    print(f"Stress (1): {binary_counts.get(1, 0)} windows ({binary_counts.get(1, 0)/len(combined_features)*100:.2f}%)")

    print("\n--- Original Label Distribution ---")
    original_counts = combined_features['label_original'].value_counts().sort_index()
    label_names = {1: 'Baseline', 2: 'Stress', 3: 'Amusement', 4: 'Meditation'}
    for label, count in original_counts.items():
        label_name = label_names.get(int(label), f'Label {int(label)}')
        print(f"{label_name} ({int(label)}): {count} windows ({count/len(combined_features)*100:.2f}%)")

    print("\n--- Per-Subject Summary ---")
    print(summary_df.to_string(index=False))

    print("\n--- Feature Categories ---")
    feature_cols = [col for col in combined_features.columns if col not in ['subject_id', 'window_idx', 'label_binary', 'label_original']]
    feature_categories = {
        'HRV': [col for col in feature_cols if col.startswith('hrv')],
        'EDA': [col for col in feature_cols if col.startswith('eda')],
        'Accelerometer': [col for col in feature_cols if col.startswith('acc')],
        'Temperature': [col for col in feature_cols if col.startswith('temp')]
    }

    for category, features in feature_categories.items():
        print(f"{category}: {len(features)} features")

    print("\n--- Output Files ---")
    print(f"1. {output_file.name} - Full dataset (pickle)")
    print(f"2. {csv_file.name} - Full dataset (CSV)")
    print(f"3. {summary_file.name} - Subject summaries")

    print("\n" + "="*80)
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    print("\nNext steps:")
    print("1. Run notebooks/03_verify_binary_data.ipynb to validate the processed data")
    print("2. Begin model training with binary stress classification")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nScript interrupted by user.")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
