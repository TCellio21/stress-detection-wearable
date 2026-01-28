"""
Signal Preprocessing Module

This module provides signal processing functions for physiological data:
- Filtering (bandpass, lowpass, highpass)
- Artifact detection and removal
- Normalization
- Resampling
- Segmentation into time windows
"""

import numpy as np
from scipy import signal
from scipy.interpolate import interp1d
from typing import Tuple, Optional, List
import warnings


class SignalPreprocessor:
    """Class for preprocessing physiological signals."""

    def __init__(self):
        """Initialize the preprocessor."""
        pass

    @staticmethod
    def butter_filter(data: np.ndarray, cutoff, fs: float, btype: str = 'low',
                     order: int = 4) -> np.ndarray:
        """
        Apply Butterworth filter to signal.

        Args:
            data: Input signal
            cutoff: Cutoff frequency or [low, high] for bandpass
            fs: Sampling frequency
            btype: Filter type ('low', 'high', 'band')
            order: Filter order

        Returns:
            Filtered signal
        """
        nyquist = 0.5 * fs
        if isinstance(cutoff, (list, tuple)):
            normal_cutoff = [c / nyquist for c in cutoff]
        else:
            normal_cutoff = cutoff / nyquist

        b, a = signal.butter(order, normal_cutoff, btype=btype, analog=False)
        filtered_data = signal.filtfilt(b, a, data)

        return filtered_data

    @staticmethod
    def filter_ecg(ecg: np.ndarray, fs: float = 700.0) -> np.ndarray:
        """
        Apply bandpass filter for ECG signal (0.5-50 Hz).

        Args:
            ecg: ECG signal
            fs: Sampling frequency

        Returns:
            Filtered ECG signal
        """
        return SignalPreprocessor.butter_filter(ecg, [0.5, 50], fs, btype='band')

    @staticmethod
    def filter_eda(eda: np.ndarray, fs: float = 700.0) -> np.ndarray:
        """
        Apply lowpass filter for EDA signal.
        
        The cutoff frequency is adaptive based on sampling rate:
        - For high-rate signals (>=100 Hz): 5 Hz cutoff
        - For low-rate signals (<100 Hz): 0.4 * Nyquist frequency
        
        This ensures the cutoff is always valid and appropriate for the signal.

        Args:
            eda: EDA signal
            fs: Sampling frequency

        Returns:
            Filtered EDA signal
        """
        # Adaptive cutoff frequency based on sampling rate
        nyquist = 0.5 * fs
        if fs >= 100:
            # High sampling rate (chest sensor): use 5 Hz
            cutoff = min(5.0, 0.9 * nyquist)
        else:
            # Low sampling rate (wrist sensor): use 40% of Nyquist
            # For 4 Hz sampling, this gives 0.8 Hz cutoff
            cutoff = 0.4 * nyquist
        
        return SignalPreprocessor.butter_filter(eda, cutoff, fs, btype='low')

    @staticmethod
    def filter_bvp(bvp: np.ndarray, fs: float = 64.0) -> np.ndarray:
        """
        Apply bandpass filter for BVP/PPG signal (0.5-8 Hz).

        Args:
            bvp: BVP signal
            fs: Sampling frequency

        Returns:
            Filtered BVP signal
        """
        return SignalPreprocessor.butter_filter(bvp, [0.5, 8], fs, btype='band')

    @staticmethod
    def detect_artifacts_eda(eda: np.ndarray, threshold: float = 5.0) -> np.ndarray:
        """
        Detect artifacts in EDA signal using threshold on derivative.

        Args:
            eda: EDA signal
            threshold: Threshold for artifact detection (μS/sample)

        Returns:
            Boolean mask (True = artifact)
        """
        # Calculate derivative
        eda_diff = np.diff(eda, prepend=eda[0])

        # Detect large jumps
        artifacts = np.abs(eda_diff) > threshold

        return artifacts

    @staticmethod
    def detect_artifacts_accel(acc: np.ndarray, threshold: float = 1.5) -> np.ndarray:
        """
        Detect motion artifacts using accelerometer magnitude.

        Args:
            acc: Acceleration data (N x 3)
            threshold: Threshold in g

        Returns:
            Boolean mask (True = artifact)
        """
        # Calculate acceleration magnitude
        acc_mag = np.sqrt(np.sum(acc**2, axis=1))

        # Detect large movements (deviation from gravity)
        artifacts = np.abs(acc_mag - 1.0) > threshold

        return artifacts

    @staticmethod
    def normalize_zscore(data: np.ndarray, axis: int = 0) -> np.ndarray:
        """
        Z-score normalization (zero mean, unit variance).

        Args:
            data: Input signal
            axis: Axis along which to normalize

        Returns:
            Normalized signal
        """
        mean = np.mean(data, axis=axis, keepdims=True)
        std = np.std(data, axis=axis, keepdims=True)

        # Avoid division by zero
        std = np.where(std == 0, 1, std)

        normalized = (data - mean) / std

        return normalized

    @staticmethod
    def normalize_minmax(data: np.ndarray, feature_range: Tuple[float, float] = (0, 1),
                        axis: int = 0) -> np.ndarray:
        """
        Min-max normalization to specified range.

        Args:
            data: Input signal
            feature_range: Target range (min, max)
            axis: Axis along which to normalize

        Returns:
            Normalized signal
        """
        min_val = np.min(data, axis=axis, keepdims=True)
        max_val = np.max(data, axis=axis, keepdims=True)

        # Avoid division by zero
        range_val = max_val - min_val
        range_val = np.where(range_val == 0, 1, range_val)

        # Scale to feature_range
        scaled = (data - min_val) / range_val
        scaled = scaled * (feature_range[1] - feature_range[0]) + feature_range[0]

        return scaled

    @staticmethod
    def resample_signal(data: np.ndarray, original_fs: float, target_fs: float) -> np.ndarray:
        """
        Resample signal to target frequency.

        Args:
            data: Input signal
            original_fs: Original sampling frequency
            target_fs: Target sampling frequency

        Returns:
            Resampled signal
        """
        if original_fs == target_fs:
            return data

        # Calculate number of samples in resampled signal
        duration = len(data) / original_fs
        n_samples_target = int(duration * target_fs)

        # Create time vectors
        t_original = np.arange(len(data)) / original_fs
        t_target = np.arange(n_samples_target) / target_fs

        # Interpolate
        if data.ndim == 1:
            interp_func = interp1d(t_original, data, kind='linear',
                                  fill_value='extrapolate')
            resampled = interp_func(t_target)
        else:
            # Multi-dimensional signal
            resampled = np.zeros((n_samples_target, data.shape[1]))
            for i in range(data.shape[1]):
                interp_func = interp1d(t_original, data[:, i], kind='linear',
                                      fill_value='extrapolate')
                resampled[:, i] = interp_func(t_target)

        return resampled

    @staticmethod
    def segment_signal(data: np.ndarray, labels: np.ndarray, window_size: int,
                      step_size: int, fs: float) -> Tuple[List[np.ndarray], List[int]]:
        """
        Segment signal into overlapping windows.

        Args:
            data: Input signal
            labels: Corresponding labels
            window_size: Window size in seconds
            step_size: Step size in seconds
            fs: Sampling frequency

        Returns:
            Tuple of (list of segments, list of segment labels)
        """
        window_samples = int(window_size * fs)
        step_samples = int(step_size * fs)

        segments = []
        segment_labels = []

        for start in range(0, len(data) - window_samples + 1, step_samples):
            end = start + window_samples

            # Extract segment
            if data.ndim == 1:
                segment = data[start:end]
            else:
                segment = data[start:end, :]

            # Get label for this window (majority vote)
            window_labels = labels[start:end]
            unique, counts = np.unique(window_labels, return_counts=True)
            segment_label = unique[np.argmax(counts)]

            segments.append(segment)
            segment_labels.append(segment_label)

        return segments, segment_labels

    @staticmethod
    def remove_baseline_drift(signal_data: np.ndarray, fs: float, cutoff: float = 0.05) -> np.ndarray:
        """
        Remove baseline drift using high-pass filter.

        Args:
            signal_data: Input signal
            fs: Sampling frequency
            cutoff: High-pass cutoff frequency

        Returns:
            Signal with baseline drift removed
        """
        return SignalPreprocessor.butter_filter(signal_data, cutoff, fs, btype='high')

    @staticmethod
    def align_labels_to_signal(signal_length: int, signal_fs: float,
                               labels: np.ndarray, label_fs: float) -> np.ndarray:
        """
        Align high-rate labels to lower-rate sensor using timestamp interpolation.

        This is Grant's method for proper time-based label alignment.
        Uses searchsorted to find the closest label timestamp for each signal sample.

        Args:
            signal_length: Number of samples in the signal
            signal_fs: Sampling frequency of the signal (Hz)
            labels: Label array at high sampling rate
            label_fs: Sampling frequency of labels (typically 700 Hz)

        Returns:
            Aligned labels matching signal length

        Example:
            # Align 700 Hz labels to 4 Hz EDA signal
            eda_length = 240  # 60 seconds at 4 Hz
            labels_aligned = align_labels_to_signal(240, 4, labels, 700)
            # Now labels_aligned[i] corresponds to eda[i]
        """
        # Create time arrays (timestamps in seconds)
        signal_times = np.arange(signal_length) / signal_fs
        label_times = np.arange(len(labels)) / label_fs

        # Find closest label timestamp for each signal timestamp
        label_indices = np.searchsorted(label_times, signal_times)

        # Clip to valid range (handle edge cases)
        label_indices = np.clip(label_indices, 0, len(labels) - 1)

        return labels[label_indices]


class RespiBAN_Converter:
    """Convert raw RespiBAN sensor values to SI units."""

    VCC = 3.0
    CHAN_BIT = 2**16

    @staticmethod
    def convert_ecg(raw: np.ndarray) -> np.ndarray:
        """Convert ECG to mV."""
        return ((raw / RespiBAN_Converter.CHAN_BIT - 0.5) *
                RespiBAN_Converter.VCC)

    @staticmethod
    def convert_eda(raw: np.ndarray) -> np.ndarray:
        """Convert EDA to μS."""
        return (((raw / RespiBAN_Converter.CHAN_BIT) *
                RespiBAN_Converter.VCC) / 0.12)

    @staticmethod
    def convert_emg(raw: np.ndarray) -> np.ndarray:
        """Convert EMG to mV."""
        return ((raw / RespiBAN_Converter.CHAN_BIT - 0.5) *
                RespiBAN_Converter.VCC)

    @staticmethod
    def convert_temp(raw: np.ndarray) -> np.ndarray:
        """Convert temperature to °C."""
        vcc = RespiBAN_Converter.VCC
        chan_bit = RespiBAN_Converter.CHAN_BIT

        vout = (raw * vcc) / (chan_bit - 1.0)
        rntc = ((10**4) * vout) / (vcc - vout)

        temp_kelvin = 1.0 / (1.12764514e-3 + 2.34282709e-4 * np.log(rntc) +
                             8.77303013e-8 * (np.log(rntc)**3))
        temp_celsius = temp_kelvin - 273.15

        return temp_celsius

    @staticmethod
    def convert_acc(raw: np.ndarray, cmin: float = 28000, cmax: float = 38000) -> np.ndarray:
        """Convert accelerometer to g."""
        return (raw - cmin) / (cmax - cmin) * 2 - 1

    @staticmethod
    def convert_resp(raw: np.ndarray) -> np.ndarray:
        """Convert respiration to %."""
        return (raw / RespiBAN_Converter.CHAN_BIT - 0.5) * 100


class E4_Converter:
    """Convert Empatica E4 sensor values (mostly already in SI units)."""

    @staticmethod
    def convert_acc(raw: np.ndarray) -> np.ndarray:
        """Convert accelerometer from 1/64g to g."""
        return raw / 64.0

    @staticmethod
    def convert_eda(raw: np.ndarray) -> np.ndarray:
        """EDA already in μS, no conversion needed."""
        return raw

    @staticmethod
    def convert_temp(raw: np.ndarray) -> np.ndarray:
        """Temperature already in °C, no conversion needed."""
        return raw

    @staticmethod
    def convert_bvp(raw: np.ndarray) -> np.ndarray:
        """BVP is raw photoplethysmograph, no conversion."""
        return raw


if __name__ == "__main__":
    # Example usage
    print("Signal Preprocessor initialized successfully")

    # Test filtering
    fs = 700
    t = np.linspace(0, 10, fs * 10)
    test_signal = np.sin(2 * np.pi * 1.0 * t) + 0.5 * np.sin(2 * np.pi * 30 * t)

    preprocessor = SignalPreprocessor()
    filtered = preprocessor.butter_filter(test_signal, [0.5, 10], fs, btype='band')

    print(f"Original signal shape: {test_signal.shape}")
    print(f"Filtered signal shape: {filtered.shape}")
    print("Preprocessing module working correctly")
