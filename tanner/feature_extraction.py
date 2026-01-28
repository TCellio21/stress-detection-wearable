"""
Feature Extraction Module

This module extracts features from physiological signals for machine learning:
- HRV features (time-domain, frequency-domain, non-linear)
- EDA features (tonic/phasic components, SCR metrics)
- Accelerometer features (activity, movement patterns)
- Temperature features
- Respiration features
"""

import numpy as np
from scipy import signal, stats
try:
    from scipy.integrate import trapz
except ImportError:
    # trapz was moved in newer scipy versions, use numpy instead
    from numpy import trapezoid as trapz
from typing import Dict, Tuple, Optional
import warnings

# Import NeuroKit2 for Grant's methods
try:
    import neurokit2 as nk
    NEUROKIT_AVAILABLE = True
except ImportError:
    NEUROKIT_AVAILABLE = False
    warnings.warn("NeuroKit2 not available. Install with: pip install neurokit2")


class HRVFeatures:
    """Extract Heart Rate Variability features from ECG or BVP."""

    @staticmethod
    def detect_peaks(signal_data: np.ndarray, fs: float, distance: Optional[int] = None) -> np.ndarray:
        """
        Detect R-peaks in ECG or peaks in BVP.

        Args:
            signal_data: ECG or BVP signal
            fs: Sampling frequency
            distance: Minimum distance between peaks (samples)

        Returns:
            Array of peak indices
        """
        if distance is None:
            # Assume HR between 40-180 bpm
            min_hr = 40
            max_hr = 180
            distance = int(fs * 60 / max_hr)

        peaks, _ = signal.find_peaks(signal_data, distance=distance, prominence=0.5)
        return peaks

    @staticmethod
    def calculate_rr_intervals(peaks: np.ndarray, fs: float) -> np.ndarray:
        """
        Calculate RR intervals from peaks.

        Args:
            peaks: Array of peak indices
            fs: Sampling frequency

        Returns:
            RR intervals in milliseconds
        """
        rr_intervals = np.diff(peaks) / fs * 1000  # Convert to ms
        return rr_intervals

    @staticmethod
    def extract_time_domain_features(rr_intervals: np.ndarray) -> Dict[str, float]:
        """
        Extract time-domain HRV features.

        Args:
            rr_intervals: RR intervals in ms

        Returns:
            Dictionary of features
        """
        if len(rr_intervals) < 5:
            return {f'hrv_time_{k}': np.nan for k in
                   ['mean_hr', 'mean_rr', 'sdnn', 'rmssd', 'pnn50', 'pnn20']}

        features = {}

        # Mean heart rate
        features['hrv_time_mean_hr'] = 60000 / np.mean(rr_intervals)  # bpm

        # Mean RR interval
        features['hrv_time_mean_rr'] = np.mean(rr_intervals)

        # SDNN - Standard deviation of NN intervals
        features['hrv_time_sdnn'] = np.std(rr_intervals, ddof=1)

        # RMSSD - Root mean square of successive differences
        successive_diffs = np.diff(rr_intervals)
        features['hrv_time_rmssd'] = np.sqrt(np.mean(successive_diffs**2))

        # pNN50 - Percentage of successive RR intervals that differ by more than 50 ms
        nn50 = np.sum(np.abs(successive_diffs) > 50)
        features['hrv_time_pnn50'] = (nn50 / len(successive_diffs)) * 100

        # pNN20
        nn20 = np.sum(np.abs(successive_diffs) > 20)
        features['hrv_time_pnn20'] = (nn20 / len(successive_diffs)) * 100

        return features

    @staticmethod
    def extract_frequency_domain_features(rr_intervals: np.ndarray, fs_rr: float = 4.0) -> Dict[str, float]:
        """
        Extract frequency-domain HRV features using Welch's method.

        Args:
            rr_intervals: RR intervals in ms
            fs_rr: Resampling frequency for RR intervals (Hz)

        Returns:
            Dictionary of features
        """
        if len(rr_intervals) < 10:
            return {f'hrv_freq_{k}': np.nan for k in
                   ['vlf', 'lf', 'hf', 'lf_hf_ratio', 'total_power']}

        # Interpolate RR intervals to create evenly sampled signal
        # Create time points starting from 0
        time_rr = np.insert(np.cumsum(rr_intervals), 0, 0) / 1000  # Convert to seconds, add 0 at start
        time_interp = np.arange(0, time_rr[-1], 1/fs_rr)

        # Prepare RR values for interpolation (repeat first value at start)
        rr_values = np.insert(rr_intervals, 0, rr_intervals[0])

        # Interpolate (now both arrays have matching lengths: N+1)
        rr_interp = np.interp(time_interp, time_rr, rr_values)

        # Compute PSD using Welch's method
        freqs, psd = signal.welch(rr_interp, fs=fs_rr, nperseg=min(256, len(rr_interp)))

        features = {}

        # Define frequency bands
        vlf_band = (0.003, 0.04)
        lf_band = (0.04, 0.15)
        hf_band = (0.15, 0.4)

        # Calculate power in each band
        vlf_power = trapz(psd[(freqs >= vlf_band[0]) & (freqs < vlf_band[1])],
                         freqs[(freqs >= vlf_band[0]) & (freqs < vlf_band[1])])
        lf_power = trapz(psd[(freqs >= lf_band[0]) & (freqs < lf_band[1])],
                        freqs[(freqs >= lf_band[0]) & (freqs < lf_band[1])])
        hf_power = trapz(psd[(freqs >= hf_band[0]) & (freqs < hf_band[1])],
                        freqs[(freqs >= hf_band[0]) & (freqs < hf_band[1])])

        features['hrv_freq_vlf'] = vlf_power
        features['hrv_freq_lf'] = lf_power
        features['hrv_freq_hf'] = hf_power
        features['hrv_freq_lf_hf_ratio'] = lf_power / hf_power if hf_power > 0 else np.nan
        features['hrv_freq_total_power'] = vlf_power + lf_power + hf_power

        return features

    @staticmethod
    def extract_nonlinear_features(rr_intervals: np.ndarray) -> Dict[str, float]:
        """
        Extract non-linear HRV features.

        Args:
            rr_intervals: RR intervals in ms

        Returns:
            Dictionary of features
        """
        if len(rr_intervals) < 10:
            return {f'hrv_nonlinear_{k}': np.nan for k in ['sd1', 'sd2', 'sd_ratio']}

        features = {}

        # Poincaré plot features
        rr1 = rr_intervals[:-1]
        rr2 = rr_intervals[1:]

        # SD1 - Standard deviation perpendicular to identity line
        features['hrv_nonlinear_sd1'] = np.std(rr1 - rr2, ddof=1) / np.sqrt(2)

        # SD2 - Standard deviation along identity line
        features['hrv_nonlinear_sd2'] = np.std(rr1 + rr2, ddof=1) / np.sqrt(2)

        # SD1/SD2 ratio
        features['hrv_nonlinear_sd_ratio'] = features['hrv_nonlinear_sd1'] / features['hrv_nonlinear_sd2'] \
            if features['hrv_nonlinear_sd2'] > 0 else np.nan

        return features

    @staticmethod
    def extract_all_hrv_features(signal_data: np.ndarray, fs: float,
                                 device: str = 'wrist', use_grant_method: bool = True) -> Dict[str, float]:
        """
        Extract all HRV features from ECG or BVP signal.

        Uses Grant's method by default: NeuroKit2 PPG/ECG processing pipeline
        with automatic artifact correction and validated peak detection.

        Args:
            signal_data: ECG or BVP signal
            fs: Sampling frequency
            device: 'wrist' (BVP) or 'chest' (ECG)
            use_grant_method: If True, use NeuroKit2 (Grant's approach)

        Returns:
            Dictionary of all HRV features
        """
        if use_grant_method and NEUROKIT_AVAILABLE:
            try:
                # Grant's method: Use NeuroKit2 for robust processing
                if device == 'wrist':
                    # Process BVP/PPG signal
                    _, info = nk.ppg_process(signal_data, sampling_rate=fs)
                    peaks = info['PPG_Peaks']
                else:
                    # Process ECG signal
                    _, info = nk.ecg_process(signal_data, sampling_rate=fs)
                    peaks = info['ECG_R_Peaks']

                # Need at least 2 peaks for HRV
                if len(peaks) < 2:
                    return {f'hrv_{k}': 0.0 for k in
                           ['time_mean_hr', 'time_sdnn', 'time_rmssd', 'freq_lf',
                            'freq_hf', 'freq_lf_hf_ratio', 'nonlinear_sd1']}

                # Extract HRV features using NeuroKit2
                hrv_time = nk.hrv_time(peaks, sampling_rate=fs, show=False)
                hrv_freq = nk.hrv_frequency(peaks, sampling_rate=fs, show=False)
                hrv_nonlinear = nk.hrv_nonlinear(peaks, sampling_rate=fs, show=False)

                # Map to consistent feature names
                features = {}

                # Time-domain features
                features['hrv_time_mean_hr'] = hrv_time['HRV_MeanNN'].values[0] if 'HRV_MeanNN' in hrv_time.columns else np.nan
                features['hrv_time_mean_rr'] = hrv_time['HRV_MeanNN'].values[0] if 'HRV_MeanNN' in hrv_time.columns else np.nan
                features['hrv_time_sdnn'] = hrv_time['HRV_SDNN'].values[0] if 'HRV_SDNN' in hrv_time.columns else np.nan
                features['hrv_time_rmssd'] = hrv_time['HRV_RMSSD'].values[0] if 'HRV_RMSSD' in hrv_time.columns else np.nan
                features['hrv_time_pnn50'] = hrv_time['HRV_pNN50'].values[0] if 'HRV_pNN50' in hrv_time.columns else np.nan
                features['hrv_time_pnn20'] = hrv_time['HRV_pNN20'].values[0] if 'HRV_pNN20' in hrv_time.columns else np.nan

                # Frequency-domain features
                features['hrv_freq_vlf'] = hrv_freq['HRV_VLF'].values[0] if 'HRV_VLF' in hrv_freq.columns else np.nan
                features['hrv_freq_lf'] = hrv_freq['HRV_LF'].values[0] if 'HRV_LF' in hrv_freq.columns else np.nan
                features['hrv_freq_hf'] = hrv_freq['HRV_HF'].values[0] if 'HRV_HF' in hrv_freq.columns else np.nan
                features['hrv_freq_lf_hf_ratio'] = hrv_freq['HRV_LFHF'].values[0] if 'HRV_LFHF' in hrv_freq.columns else np.nan
                features['hrv_freq_total_power'] = hrv_freq['HRV_TP'].values[0] if 'HRV_TP' in hrv_freq.columns else np.nan

                # Non-linear features
                features['hrv_nonlinear_sd1'] = hrv_nonlinear['HRV_SD1'].values[0] if 'HRV_SD1' in hrv_nonlinear.columns else np.nan
                features['hrv_nonlinear_sd2'] = hrv_nonlinear['HRV_SD2'].values[0] if 'HRV_SD2' in hrv_nonlinear.columns else np.nan
                features['hrv_nonlinear_sd_ratio'] = hrv_nonlinear['HRV_SD1SD2'].values[0] if 'HRV_SD1SD2' in hrv_nonlinear.columns else np.nan

                return features

            except Exception as e:
                warnings.warn(f"NeuroKit2 HRV extraction failed: {e}. Using manual method.")

        # Fallback: Original manual method
        try:
            # Detect peaks
            peaks = HRVFeatures.detect_peaks(signal_data, fs)

            # Calculate RR intervals
            rr_intervals = HRVFeatures.calculate_rr_intervals(peaks, fs)

            # Remove outliers (RR intervals outside 300-2000 ms)
            valid_mask = (rr_intervals >= 300) & (rr_intervals <= 2000)
            rr_intervals = rr_intervals[valid_mask]

            # Extract features
            features = {}
            features.update(HRVFeatures.extract_time_domain_features(rr_intervals))
            features.update(HRVFeatures.extract_frequency_domain_features(rr_intervals))
            features.update(HRVFeatures.extract_nonlinear_features(rr_intervals))

            return features

        except Exception as e:
            warnings.warn(f"HRV feature extraction failed: {e}")
            return {f'hrv_{k}': np.nan for k in
                   ['time_mean_hr', 'time_sdnn', 'freq_lf', 'freq_hf', 'freq_lf_hf_ratio']}


class EDAFeatures:
    """Extract Electrodermal Activity features."""

    @staticmethod
    def decompose_eda(eda: np.ndarray, fs: float, use_cvxeda: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decompose EDA into tonic (SCL) and phasic (SCR) components.

        Uses Grant's method: NeuroKit2 with cvxEDA algorithm (gold standard).
        Falls back to simple filtering if NeuroKit2 unavailable or cvxEDA fails.

        Args:
            eda: EDA signal
            fs: Sampling frequency
            use_cvxeda: If True, attempt cvxEDA method (Grant's approach)

        Returns:
            Tuple of (tonic, phasic) components
        """
        if use_cvxeda and NEUROKIT_AVAILABLE:
            try:
                # Grant's method: Use NeuroKit2 with cvxEDA
                # First apply lowpass filter (1.8 Hz recommended by NeuroKit2)
                from scipy.signal import butter, sosfiltfilt
                nyquist = fs / 2.0
                cutoff = min(1.8, nyquist * 0.9)
                sos = butter(4, cutoff, fs=fs, btype='low', output='sos')
                eda_filtered = sosfiltfilt(sos, eda)

                # Decompose using cvxEDA (requires cvxopt library)
                try:
                    eda_decomposed = nk.eda_phasic(eda_filtered, sampling_rate=fs, method='cvxeda')
                except Exception as cvx_error:
                    warnings.warn(f"cvxEDA failed ({cvx_error}), trying smoothmedian fallback")
                    # Fallback to smoothmedian if cvxopt not installed
                    eda_decomposed = nk.eda_phasic(eda_filtered, sampling_rate=fs, method='smoothmedian')

                tonic = eda_decomposed['EDA_Tonic'].values
                phasic = eda_decomposed['EDA_Phasic'].values

                return tonic, phasic

            except Exception as e:
                warnings.warn(f"NeuroKit2 decomposition failed: {e}. Using simple filtering.")

        # Fallback: Original simple filtering method
        from scipy.signal import butter, filtfilt

        # Use low-pass filter to extract tonic component
        cutoff = 0.05  # Hz
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(4, normal_cutoff, btype='low')

        tonic = filtfilt(b, a, eda)
        phasic = eda - tonic

        return tonic, phasic

    @staticmethod
    def detect_scr_peaks(phasic: np.ndarray, fs: float, threshold: float = 0.01,
                        use_neurokit: bool = True) -> Tuple[np.ndarray, Dict]:
        """
        Detect SCR (Skin Conductance Response) peaks.

        Uses Grant's method: NeuroKit2 for peak detection with outlier filtering.
        Falls back to scipy.signal.find_peaks if NeuroKit2 unavailable.

        Args:
            phasic: Phasic EDA component
            fs: Sampling frequency
            threshold: Minimum amplitude threshold (μS)
            use_neurokit: If True, use NeuroKit2 (Grant's approach)

        Returns:
            Tuple of (peak_indices, peak_info_dict)
            peak_info_dict contains: amplitudes, rise_times, recovery_times
        """
        if use_neurokit and NEUROKIT_AVAILABLE:
            try:
                # Grant's method: Use NeuroKit2 for peak detection
                _, peaks_info = nk.eda_peaks(phasic, sampling_rate=fs,
                                            amplitude_min=threshold)

                peak_indices = peaks_info['SCR_Peaks']
                peak_amplitudes = peaks_info['SCR_Amplitude']
                peak_rise_times = peaks_info['SCR_RiseTime']
                peak_recovery_times = peaks_info['SCR_Recovery']

                # Grant's outlier filtering: 0.01 to 1.0 μS
                OUTLIER_MIN = 0.01  # μS (physiological minimum)
                OUTLIER_MAX = 1.0   # μS (physiological maximum for wrist)

                valid_mask = (peak_amplitudes >= OUTLIER_MIN) & (peak_amplitudes <= OUTLIER_MAX)

                return peak_indices[valid_mask], {
                    'amplitudes': peak_amplitudes[valid_mask],
                    'rise_times': peak_rise_times[valid_mask],
                    'recovery_times': peak_recovery_times[valid_mask]
                }

            except Exception as e:
                warnings.warn(f"NeuroKit2 peak detection failed: {e}. Using scipy fallback.")

        # Fallback: Original scipy method
        min_distance = int(1.0 * fs)  # Minimum 1 second between peaks
        peaks, properties = signal.find_peaks(phasic, height=threshold, distance=min_distance)

        # Get amplitudes from properties
        amplitudes = phasic[peaks] if len(peaks) > 0 else np.array([])

        return peaks, {
            'amplitudes': amplitudes,
            'rise_times': np.array([]),  # Not available with scipy method
            'recovery_times': np.array([])
        }

    @staticmethod
    def extract_eda_features(eda: np.ndarray, fs: float, use_grant_method: bool = True) -> Dict[str, float]:
        """
        Extract EDA features.

        Uses Grant's methods by default:
        - cvxEDA for decomposition
        - NeuroKit2 for peak detection
        - Outlier filtering (0.01-1.0 μS)

        Args:
            eda: EDA signal in μS
            fs: Sampling frequency
            use_grant_method: If True, use Grant's NeuroKit2 methods

        Returns:
            Dictionary of EDA features
        """
        try:
            features = {}

            # Decompose into tonic and phasic (Grant's cvxEDA method)
            tonic, phasic = EDAFeatures.decompose_eda(eda, fs, use_cvxeda=use_grant_method)

            # Tonic features (SCL)
            features['eda_tonic_mean'] = np.mean(tonic)
            features['eda_tonic_std'] = np.std(tonic)
            features['eda_tonic_min'] = np.min(tonic)
            features['eda_tonic_max'] = np.max(tonic)
            features['eda_tonic_range'] = np.max(tonic) - np.min(tonic)
            features['eda_tonic_slope'] = np.polyfit(np.arange(len(tonic)), tonic, 1)[0]

            # Phasic features (SCR)
            features['eda_phasic_mean'] = np.mean(phasic)
            features['eda_phasic_std'] = np.std(phasic)
            features['eda_phasic_max'] = np.max(phasic)

            # Detect SCR peaks (Grant's method with outlier filtering)
            peaks, peak_info = EDAFeatures.detect_scr_peaks(phasic, fs, use_neurokit=use_grant_method)

            features['eda_scr_count'] = len(peaks)
            features['eda_scr_rate'] = len(peaks) / (len(eda) / fs / 60)  # per minute

            # Use filtered amplitudes from peak_info
            if len(peak_info['amplitudes']) > 0:
                features['eda_scr_mean_amplitude'] = np.mean(peak_info['amplitudes'])
                features['eda_scr_max_amplitude'] = np.max(peak_info['amplitudes'])

                # Add rise/recovery time features if available (Grant's additional features)
                if len(peak_info['rise_times']) > 0:
                    features['eda_scr_rise_time_mean'] = np.mean(peak_info['rise_times'])
                if len(peak_info['recovery_times']) > 0:
                    features['eda_scr_recovery_time_mean'] = np.mean(peak_info['recovery_times'])
            else:
                features['eda_scr_mean_amplitude'] = 0
                features['eda_scr_max_amplitude'] = 0
                features['eda_scr_rise_time_mean'] = 0
                features['eda_scr_recovery_time_mean'] = 0

            # Statistical features
            features['eda_raw_mean'] = np.mean(eda)
            features['eda_raw_std'] = np.std(eda)

            return features

        except Exception as e:
            warnings.warn(f"EDA feature extraction failed: {e}")
            return {f'eda_{k}': np.nan for k in
                   ['tonic_mean', 'phasic_std', 'scr_count', 'scr_rate']}


class AccelerometerFeatures:
    """Extract accelerometer-based features."""

    @staticmethod
    def calculate_magnitude(acc: np.ndarray) -> np.ndarray:
        """Calculate acceleration magnitude."""
        return np.sqrt(np.sum(acc**2, axis=1))

    @staticmethod
    def extract_acc_features(acc: np.ndarray, fs: float) -> Dict[str, float]:
        """
        Extract accelerometer features.

        Args:
            acc: Acceleration data (N x 3)
            fs: Sampling frequency

        Returns:
            Dictionary of features
        """
        try:
            features = {}

            # Calculate magnitude
            acc_mag = AccelerometerFeatures.calculate_magnitude(acc)

            # Time-domain features
            features['acc_mean_x'] = np.mean(acc[:, 0])
            features['acc_mean_y'] = np.mean(acc[:, 1])
            features['acc_mean_z'] = np.mean(acc[:, 2])
            features['acc_mean_mag'] = np.mean(acc_mag)

            features['acc_std_x'] = np.std(acc[:, 0])
            features['acc_std_y'] = np.std(acc[:, 1])
            features['acc_std_z'] = np.std(acc[:, 2])
            features['acc_std_mag'] = np.std(acc_mag)

            features['acc_range_mag'] = np.max(acc_mag) - np.min(acc_mag)

            # Activity intensity
            features['acc_activity_intensity'] = np.mean(np.abs(acc_mag - 1.0))  # Deviation from gravity

            # Jerk (rate of change of acceleration)
            jerk = np.diff(acc, axis=0)
            jerk_mag = AccelerometerFeatures.calculate_magnitude(jerk)
            features['acc_jerk_mean'] = np.mean(jerk_mag)
            features['acc_jerk_std'] = np.std(jerk_mag)

            # Frequency-domain features
            freqs, psd = signal.welch(acc_mag, fs=fs, nperseg=min(256, len(acc_mag)))
            features['acc_dominant_freq'] = freqs[np.argmax(psd)]
            features['acc_spectral_entropy'] = stats.entropy(psd + 1e-10)  # Add small constant to avoid log(0)

            # Movement variability
            features['acc_variability'] = np.sum(features['acc_std_mag'])

            return features

        except Exception as e:
            warnings.warn(f"Accelerometer feature extraction failed: {e}")
            return {f'acc_{k}': np.nan for k in
                   ['mean_mag', 'std_mag', 'activity_intensity', 'jerk_mean']}


class TemperatureFeatures:
    """Extract temperature features."""

    @staticmethod
    def extract_temp_features(temp: np.ndarray, fs: float) -> Dict[str, float]:
        """
        Extract temperature features.

        Args:
            temp: Temperature signal in °C
            fs: Sampling frequency

        Returns:
            Dictionary of features
        """
        try:
            features = {}

            features['temp_mean'] = np.mean(temp)
            features['temp_std'] = np.std(temp)
            features['temp_min'] = np.min(temp)
            features['temp_max'] = np.max(temp)
            features['temp_range'] = np.max(temp) - np.min(temp)

            # Rate of change
            features['temp_slope'] = np.polyfit(np.arange(len(temp)), temp, 1)[0]

            # Derivative statistics
            temp_diff = np.diff(temp)
            features['temp_rate_mean'] = np.mean(temp_diff)
            features['temp_rate_std'] = np.std(temp_diff)

            return features

        except Exception as e:
            warnings.warn(f"Temperature feature extraction failed: {e}")
            return {f'temp_{k}': np.nan for k in ['mean', 'std', 'slope', 'range']}


class RespirationFeatures:
    """Extract respiration features."""

    @staticmethod
    def detect_resp_peaks(resp: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
        """Detect inhalation and exhalation peaks."""
        # Detect inhalation peaks (maxima)
        inhale_peaks, _ = signal.find_peaks(resp, distance=int(fs*1.5))

        # Detect exhalation peaks (minima)
        exhale_peaks, _ = signal.find_peaks(-resp, distance=int(fs*1.5))

        return inhale_peaks, exhale_peaks

    @staticmethod
    def extract_resp_features(resp: np.ndarray, fs: float) -> Dict[str, float]:
        """
        Extract respiration features.

        Args:
            resp: Respiration signal
            fs: Sampling frequency

        Returns:
            Dictionary of features
        """
        try:
            features = {}

            # Detect peaks
            inhale_peaks, exhale_peaks = RespirationFeatures.detect_resp_peaks(resp, fs)

            # Respiratory rate
            if len(inhale_peaks) > 1:
                breath_intervals = np.diff(inhale_peaks) / fs  # seconds
                features['resp_rate'] = 60 / np.mean(breath_intervals)  # breaths per minute
                features['resp_rate_std'] = np.std(60 / breath_intervals)
            else:
                features['resp_rate'] = np.nan
                features['resp_rate_std'] = np.nan

            # Amplitude features
            features['resp_amplitude_mean'] = np.mean(np.abs(resp))
            features['resp_amplitude_std'] = np.std(resp)

            # Statistical features
            features['resp_mean'] = np.mean(resp)
            features['resp_std'] = np.std(resp)

            return features

        except Exception as e:
            warnings.warn(f"Respiration feature extraction failed: {e}")
            return {f'resp_{k}': np.nan for k in ['rate', 'amplitude_mean', 'std']}


class MultimodalFeatureExtractor:
    """Main feature extractor combining all modalities."""

    @staticmethod
    def extract_features_from_window(signals: Dict[str, np.ndarray],
                                     fs_dict: Dict[str, float],
                                     device: str = 'wrist',
                                     use_grant_methods: bool = True) -> Dict[str, float]:
        """
        Extract all features from a signal window.

        Uses Grant's methods by default for consistent processing:
        - NeuroKit2 for HRV extraction
        - cvxEDA for EDA decomposition
        - Outlier filtering for SCR peaks

        Args:
            signals: Dictionary of signals
            fs_dict: Dictionary of sampling frequencies for each signal
            device: 'wrist' or 'chest'
            use_grant_methods: If True, use Grant's NeuroKit2-based methods

        Returns:
            Dictionary of all extracted features
        """
        features = {}

        # Extract features based on available signals
        if 'BVP' in signals and device == 'wrist':
            features.update(HRVFeatures.extract_all_hrv_features(
                signals['BVP'], fs_dict.get('BVP', 64.0),
                device='wrist', use_grant_method=use_grant_methods))

        if 'ECG' in signals and device == 'chest':
            features.update(HRVFeatures.extract_all_hrv_features(
                signals['ECG'], fs_dict.get('ECG', 700.0),
                device='chest', use_grant_method=use_grant_methods))

        if 'EDA' in signals:
            features.update(EDAFeatures.extract_eda_features(
                signals['EDA'], fs_dict.get('EDA', 4.0),
                use_grant_method=use_grant_methods))

        if 'ACC' in signals:
            features.update(AccelerometerFeatures.extract_acc_features(
                signals['ACC'], fs_dict.get('ACC', 32.0)))

        if 'TEMP' in signals:
            features.update(TemperatureFeatures.extract_temp_features(
                signals['TEMP'], fs_dict.get('TEMP', 4.0)))

        if 'RESP' in signals and device == 'chest':
            features.update(RespirationFeatures.extract_resp_features(
                signals['RESP'], fs_dict.get('RESP', 700.0)))

        return features


class SubjectNormalizer:
    """
    Subject-specific normalization for domain adaptation.

    Two-pass approach:
    1. Calculate baseline statistics from TRUE baseline (label=1) ONLY
    2. Normalize ALL windows using those statistics

    Creates person-invariant features for wearable deployment.
    """

    @staticmethod
    def normalize_features(features_df, subject_col='subject_id',
                          label_col='label_original', use_robust=True):
        """
        Apply subject-specific normalization to all features.

        Args:
            features_df: DataFrame with raw features
            subject_col: Column name for subject IDs
            label_col: Column name for labels (1=baseline, 2=stress, etc.)
            use_robust: If True, use median/IQR (robust); else mean/std (Grant's method)

        Returns:
            DataFrame with raw + normalized features (46 raw → 138 total)
        """
        import pandas as pd
        import numpy as np

        # Identify metadata columns
        metadata_cols = [subject_col, label_col, 'label_binary', 'window_idx']

        # Identify raw feature columns
        feature_cols = [col for col in features_df.columns
                       if col not in metadata_cols
                       and not col.endswith(('_z_score', '_robust_z', '_pct_change'))]

        print(f"Normalizing {len(feature_cols)} features...")

        normalized_subjects = []

        for subject_id in features_df[subject_col].unique():
            subject_data = features_df[features_df[subject_col] == subject_id].copy()

            # PASS 1: Get baseline windows (label=1 ONLY)
            baseline = subject_data[subject_data[label_col] == 1]

            if len(baseline) < 2:
                warnings.warn(f"Subject {subject_id}: Insufficient baseline windows ({len(baseline)})")
                normalized_subjects.append(subject_data)
                continue

            # PASS 2: Calculate stats and normalize ALL windows
            for feature in feature_cols:
                if feature not in baseline.columns:
                    continue

                baseline_values = baseline[feature].dropna()

                if len(baseline_values) < 2:
                    continue

                # Calculate statistics
                if use_robust:
                    # Robust method (median/IQR)
                    center = baseline_values.median()
                    Q1 = baseline_values.quantile(0.25)
                    Q3 = baseline_values.quantile(0.75)
                    spread = Q3 - Q1
                    mean = baseline_values.mean()  # For percent change

                    if spread < 1e-10:  # IQR too small, fallback to std
                        spread = baseline_values.std()
                    if spread < 1e-10:  # Still too small
                        spread = 1.0
                else:
                    # Grant's method (mean/std)
                    center = baseline_values.mean()
                    spread = baseline_values.std()
                    mean = center

                    if spread < 1e-10:
                        spread = 1.0

                # Normalize: Z-score
                z_col = f'{feature}_robust_z' if use_robust else f'{feature}_z_score'
                subject_data[z_col] = subject_data[feature].apply(
                    lambda x: (x - center) / spread if spread > 0 else 0.0
                )

                # Normalize: Percent change
                pct_col = f'{feature}_pct_change'
                subject_data[pct_col] = subject_data[feature].apply(
                    lambda x: ((x - mean) / mean * 100) if mean != 0 else 0.0
                )

            normalized_subjects.append(subject_data)

        # Combine all subjects
        result = pd.concat(normalized_subjects, ignore_index=True)

        # Count features
        total_features = len([c for c in result.columns if c not in metadata_cols])
        print(f"✓ Normalization complete: {len(feature_cols)} raw → {total_features} total features")

        return result


if __name__ == "__main__":
    print("Feature extraction module initialized")
    print("\nAvailable feature extractors:")
    print("- HRVFeatures: Heart rate variability (time, frequency, non-linear)")
    print("- EDAFeatures: Electrodermal activity (tonic, phasic, SCR)")
    print("- AccelerometerFeatures: Movement and activity patterns")
    print("- TemperatureFeatures: Temperature statistics")
    print("- RespirationFeatures: Breathing rate and patterns")
    print("- MultimodalFeatureExtractor: Combined feature extraction")
