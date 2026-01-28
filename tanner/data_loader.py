"""
WESAD Data Loader Module

This module provides utilities to load and parse WESAD dataset files including:
- .pkl files with synchronized sensor data and labels
- CSV files from Empatica E4 device
- TXT files from RespiBAN device
- Questionnaire data
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import warnings


class WESADDataLoader:
    """Main class for loading WESAD dataset files."""

    # Label mapping from WESAD documentation
    LABEL_MAPPING = {
        0: 'transient',
        1: 'baseline',
        2: 'stress',
        3: 'amusement',
        4: 'meditation',
        5: 'ignore',
        6: 'ignore',
        7: 'ignore'
    }

    # Relevant labels for stress detection
    RELEVANT_LABELS = [1, 2, 3, 4]  # baseline, stress, amusement, meditation

    def __init__(self, dataset_path: str):
        """
        Initialize the data loader.

        Args:
            dataset_path: Path to the WESAD dataset root directory
        """
        self.dataset_path = Path(dataset_path)
        self.subjects = self._get_available_subjects()

    def _get_available_subjects(self) -> List[str]:
        """Get list of available subject IDs."""
        subject_dirs = [d for d in self.dataset_path.iterdir()
                       if d.is_dir() and d.name.startswith('S')]
        subjects = sorted([d.name for d in subject_dirs])
        return subjects

    def load_subject_pkl(self, subject_id: str) -> Dict:
        """
        Load synchronized data from subject's .pkl file.

        Args:
            subject_id: Subject ID (e.g., 'S2', 'S13')

        Returns:
            Dictionary containing:
                - 'subject': Subject ID
                - 'signal': Dict with 'chest' and 'wrist' sensor data
                - 'label': Array of labels sampled at 700 Hz
        """
        pkl_file = self.dataset_path / subject_id / f"{subject_id}.pkl"

        if not pkl_file.exists():
            raise FileNotFoundError(f"PKL file not found: {pkl_file}")

        with open(pkl_file, 'rb') as f:
            data = pickle.load(f, encoding='latin1')

        return data

    def load_subject_data(self, subject_id: str) -> Tuple[Dict, Dict, pd.DataFrame]:
        """
        Load all data for a subject including pkl, questionnaire, and metadata.

        Args:
            subject_id: Subject ID (e.g., 'S2', 'S13')

        Returns:
            Tuple of (pkl_data, metadata, questionnaire_df)
        """
        # Load synchronized pkl data
        pkl_data = self.load_subject_pkl(subject_id)

        # Load metadata from readme
        metadata = self._load_subject_readme(subject_id)

        # Load questionnaire data
        quest_df = self._load_questionnaire(subject_id)

        return pkl_data, metadata, quest_df

    def _load_subject_readme(self, subject_id: str) -> Dict:
        """Parse subject readme file for metadata."""
        readme_file = self.dataset_path / subject_id / f"{subject_id}_readme.txt"

        if not readme_file.exists():
            warnings.warn(f"Readme file not found: {readme_file}")
            return {}

        metadata = {}
        with open(readme_file, 'r') as f:
            for line in f:
                line = line.strip()
                if ':' in line:
                    key, value = line.split(':', 1)
                    metadata[key.strip()] = value.strip()

        return metadata

    def _load_questionnaire(self, subject_id: str) -> pd.DataFrame:
        """Load and parse questionnaire CSV file."""
        quest_file = self.dataset_path / subject_id / f"{subject_id}_quest.csv"

        if not quest_file.exists():
            warnings.warn(f"Questionnaire file not found: {quest_file}")
            return pd.DataFrame()

        # Read the CSV - it has a complex structure with comments
        df = pd.read_csv(quest_file, sep=';', comment='#', header=None)

        return df

    def extract_chest_signals(self, pkl_data: Dict) -> Dict[str, np.ndarray]:
        """
        Extract individual chest sensor signals from pkl data.

        Args:
            pkl_data: Loaded pkl data dictionary

        Returns:
            Dictionary with keys: 'ACC', 'ECG', 'EDA', 'EMG', 'RESP', 'TEMP'
        """
        chest_data = pkl_data['signal']['chest']

        # Chest signals are already separated in dictionaries
        signals = {}

        # Handle accelerometer (3D)
        if 'ACC' in chest_data:
            signals['ACC'] = chest_data['ACC']

        # Handle other signals (squeeze to remove extra dimensions)
        if 'ECG' in chest_data:
            signals['ECG'] = chest_data['ECG'].squeeze() if chest_data['ECG'].ndim > 1 else chest_data['ECG']
        if 'EMG' in chest_data:
            signals['EMG'] = chest_data['EMG'].squeeze() if chest_data['EMG'].ndim > 1 else chest_data['EMG']
        if 'EDA' in chest_data:
            signals['EDA'] = chest_data['EDA'].squeeze() if chest_data['EDA'].ndim > 1 else chest_data['EDA']
        if 'Temp' in chest_data:  # Note: capital T in source
            signals['TEMP'] = chest_data['Temp'].squeeze() if chest_data['Temp'].ndim > 1 else chest_data['Temp']
        if 'Resp' in chest_data:  # Note: capital R in source
            signals['RESP'] = chest_data['Resp'].squeeze() if chest_data['Resp'].ndim > 1 else chest_data['Resp']

        return signals

    def extract_wrist_signals(self, pkl_data: Dict) -> Dict[str, np.ndarray]:
        """
        Extract individual wrist sensor signals from pkl data.

        Args:
            pkl_data: Loaded pkl data dictionary

        Returns:
            Dictionary with keys: 'ACC', 'BVP', 'EDA', 'TEMP'
        """
        wrist_data = pkl_data['signal']['wrist']

        # Wrist signals are already separated in dictionaries
        signals = {}

        # Handle accelerometer (3D)
        if 'ACC' in wrist_data:
            signals['ACC'] = wrist_data['ACC']

        # Handle other signals (squeeze to remove extra dimensions)
        if 'BVP' in wrist_data:
            signals['BVP'] = wrist_data['BVP'].squeeze() if wrist_data['BVP'].ndim > 1 else wrist_data['BVP']
        if 'EDA' in wrist_data:
            signals['EDA'] = wrist_data['EDA'].squeeze() if wrist_data['EDA'].ndim > 1 else wrist_data['EDA']
        if 'TEMP' in wrist_data:  # All caps in wrist data
            signals['TEMP'] = wrist_data['TEMP'].squeeze() if wrist_data['TEMP'].ndim > 1 else wrist_data['TEMP']

        return signals

    def get_labels(self, pkl_data: Dict) -> np.ndarray:
        """
        Extract labels from pkl data.

        Args:
            pkl_data: Loaded pkl data dictionary

        Returns:
            Array of labels (700 Hz sampling rate)
        """
        return pkl_data['label']

    def filter_relevant_data(self, signals: Dict[str, np.ndarray],
                            labels: np.ndarray) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """
        Filter data to keep only relevant labels (1-4: baseline, stress, amusement, meditation).

        Args:
            signals: Dictionary of sensor signals
            labels: Array of labels

        Returns:
            Tuple of (filtered_signals, filtered_labels)
        """
        # Create mask for relevant labels
        mask = np.isin(labels, self.RELEVANT_LABELS)

        # Filter labels
        filtered_labels = labels[mask]

        # Filter each signal
        filtered_signals = {}
        for key, signal in signals.items():
            if signal.ndim == 1:
                filtered_signals[key] = signal[mask]
            else:  # Multi-dimensional (e.g., ACC)
                filtered_signals[key] = signal[mask, :]

        return filtered_signals, filtered_labels

    def convert_to_binary_stress_labels(self, labels: np.ndarray,
                                       mapping: Optional[Dict[int, int]] = None) -> np.ndarray:
        """
        Convert multi-class labels to binary stress classification.

        Args:
            labels: Array of original labels (1-4: baseline, stress, amusement, meditation)
            mapping: Optional custom mapping dict. Defaults to:
                     {1: 0, 2: 1, 3: 0, 4: 0} (stress=1 vs all others=0)

        Returns:
            Binary labels where 0=non-stress, 1=stress

        Example:
            >>> labels = np.array([1, 2, 3, 4, 2, 1])
            >>> binary_labels = loader.convert_to_binary_stress_labels(labels)
            >>> print(binary_labels)
            [0 1 0 0 1 0]  # baseline, stress, amusement, meditation, stress, baseline
        """
        if mapping is None:
            # Default: stress (2) -> 1, all others (1,3,4) -> 0
            mapping = {
                1: 0,  # baseline -> non-stress
                2: 1,  # stress -> stress
                3: 0,  # amusement -> non-stress
                4: 0,  # meditation -> non-stress
            }

        # Initialize binary labels array
        binary_labels = np.zeros_like(labels, dtype=np.int32)

        # Apply mapping
        for original_label, binary_label in mapping.items():
            binary_labels[labels == original_label] = binary_label

        return binary_labels

    def get_data_summary(self, subject_id: str) -> Dict:
        """
        Get summary statistics for a subject's data.

        Args:
            subject_id: Subject ID

        Returns:
            Dictionary with data summary
        """
        pkl_data = self.load_subject_pkl(subject_id)
        labels = self.get_labels(pkl_data)

        # Count samples per label
        unique, counts = np.unique(labels, return_counts=True)
        label_counts = dict(zip(unique, counts))

        # Duration per label (at 700 Hz)
        duration_per_label = {int(label): count / 700.0 / 60.0  # minutes
                             for label, count in label_counts.items()}

        # Get signal shapes (signals are stored as dicts)
        chest_signals = pkl_data['signal']['chest']
        wrist_signals = pkl_data['signal']['wrist']

        # Get shape info from individual signals
        chest_shape_info = {}
        for key, signal in chest_signals.items():
            if hasattr(signal, 'shape'):
                chest_shape_info[key] = signal.shape

        wrist_shape_info = {}
        for key, signal in wrist_signals.items():
            if hasattr(signal, 'shape'):
                wrist_shape_info[key] = signal.shape

        summary = {
            'subject_id': subject_id,
            'total_samples': len(labels),
            'total_duration_min': len(labels) / 700.0 / 60.0,
            'label_counts': {int(k): int(v) for k, v in label_counts.items()},
            'duration_per_label_min': duration_per_label,
            'chest_signals': chest_shape_info,
            'wrist_signals': wrist_shape_info
        }

        return summary

    def load_all_subjects(self) -> Dict[str, Dict]:
        """
        Load data for all available subjects.

        Returns:
            Dictionary mapping subject_id to their data
        """
        all_data = {}

        for subject_id in self.subjects:
            print(f"Loading {subject_id}...")
            try:
                data = self.load_subject_pkl(subject_id)
                all_data[subject_id] = data
            except Exception as e:
                warnings.warn(f"Failed to load {subject_id}: {e}")

        return all_data

    @staticmethod
    def exclude_tsst_preparation(labels: np.ndarray, label_fs: float = 700.0,
                                 prep_duration: int = 180) -> np.ndarray:
        """
        Exclude TSST preparation phase from stress condition.

        Grant's method: Excludes first 3 minutes (180 seconds) of stress labels
        because stress response takes time to develop physiologically.

        Args:
            labels: Array of labels (1=baseline, 2=stress, 3=amusement, 4=meditation)
            label_fs: Label sampling frequency (Hz)
            prep_duration: Preparation duration to exclude (seconds, default=180)

        Returns:
            Modified labels array with stress preparation phase set to 0 (transient)

        Example:
            # Exclude first 3 minutes of stress
            labels_clean = WESADDataLoader.exclude_tsst_preparation(labels)
        """
        # Create copy to avoid modifying original
        labels_modified = labels.copy()

        # Find all stress periods (label = 2)
        stress_mask = labels == 2

        if not np.any(stress_mask):
            return labels_modified  # No stress periods found

        # Find stress period boundaries
        stress_diff = np.diff(np.concatenate(([0], stress_mask.astype(int), [0])))
        stress_starts = np.where(stress_diff == 1)[0]
        stress_ends = np.where(stress_diff == -1)[0]

        # For each stress period, exclude first prep_duration seconds
        prep_samples = int(prep_duration * label_fs)

        for start_idx in stress_starts:
            end_of_prep = min(start_idx + prep_samples, len(labels_modified))
            # Set preparation phase to transient (0)
            labels_modified[start_idx:end_of_prep] = 0

        return labels_modified

    @staticmethod
    def exclude_tsst_prep_from_windows(window_labels: np.ndarray,
                                       window_size: int = 60,
                                       prep_duration: int = 180) -> np.ndarray:
        """
        Exclude TSST preparation windows from already-windowed data.

        Grant's method applied to windowed data: Excludes first N windows
        of stress condition (where N = prep_duration / window_size).

        Args:
            window_labels: Array of window labels (one per window)
            window_size: Window size in seconds (default=60)
            prep_duration: Preparation duration to exclude (seconds, default=180)

        Returns:
            Mask array (True = keep window, False = exclude window)

        Example:
            # After extracting features with 60s windows
            keep_mask = exclude_tsst_prep_from_windows(window_labels)
            features_filtered = features[keep_mask]
            labels_filtered = window_labels[keep_mask]
        """
        prep_windows = prep_duration // window_size

        # Find stress periods
        stress_mask = window_labels == 2

        if not np.any(stress_mask):
            return np.ones(len(window_labels), dtype=bool)  # Keep all

        # Find first stress window
        stress_indices = np.where(stress_mask)[0]
        first_stress_idx = stress_indices[0]

        # Create keep mask
        keep_mask = np.ones(len(window_labels), dtype=bool)

        # Exclude first prep_windows of stress
        exclude_end = min(first_stress_idx + prep_windows, len(window_labels))
        keep_mask[first_stress_idx:exclude_end] = False

        return keep_mask


def load_e4_csv_files(subject_path: Path) -> Dict[str, pd.DataFrame]:
    """
    Load raw CSV files from Empatica E4 device.

    Args:
        subject_path: Path to subject directory

    Returns:
        Dictionary of DataFrames for each E4 sensor
    """
    e4_path = subject_path / f"{subject_path.name}_E4_Data"

    if not e4_path.exists():
        raise FileNotFoundError(f"E4 data directory not found: {e4_path}")

    e4_files = {
        'ACC': 'ACC.csv',
        'BVP': 'BVP.csv',
        'EDA': 'EDA.csv',
        'TEMP': 'TEMP.csv',
        'HR': 'HR.csv',
        'IBI': 'IBI.csv'
    }

    data = {}
    for sensor, filename in e4_files.items():
        filepath = e4_path / filename
        if filepath.exists():
            data[sensor] = pd.read_csv(filepath, header=None)

    return data


if __name__ == "__main__":
    # Example usage
    import sys

    dataset_path = Path(__file__).parent.parent
    loader = WESADDataLoader(dataset_path)

    print(f"Found {len(loader.subjects)} subjects: {loader.subjects}")

    # Load and summarize first subject
    if loader.subjects:
        subject_id = loader.subjects[0]
        summary = loader.get_data_summary(subject_id)
        print(f"\n{subject_id} Summary:")
        for key, value in summary.items():
            print(f"  {key}: {value}")
