"""
Feature Extraction Module
=========================

Extracts 46 raw features from wrist-worn sensor data:
- EDA: 17 features (8 tonic/SCL + 7 phasic/SCR + 2 distribution)
- HRV: 15 features (9 time-domain + 6 frequency-domain)
- Temperature: 6 features
- Accelerometer: 8 features

Based on Grant's working implementation with these fixes:
1. HRV mean_hr correctly uses HRV_MeanNN converted to HR (60000/MeanNN), not raw MeanNN
2. NumPy compatibility: uses scipy.integrate.trapezoid instead of np.trapz
3. cvxEDA ONLY - no fallbacks (hard-fail if cvxopt not installed)

Author: Merged pipeline (Grant + Tanner approaches)
"""

import numpy as np
import neurokit2 as nk
from scipy import stats, signal

# Import trapezoid from scipy (compatible with all versions)
try:
    from scipy.integrate import trapezoid
except ImportError:
    # Fallback for older scipy versions
    from numpy import trapz as trapezoid


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def lowpass_filter(signal_data, fs, cutoff=5.0, order=4):
    """
    Apply lowpass Butterworth filter with Nyquist-safe cutoff.
    
    Args:
        signal_data: Input signal array
        fs: Sampling frequency (Hz)
        cutoff: Cutoff frequency (Hz)
        order: Filter order
    
    Returns:
        Filtered signal
    """
    nyquist = fs / 2.0
    if cutoff >= nyquist:
        cutoff = nyquist * 0.9
    sos = signal.butter(order, cutoff, fs=fs, btype='low', output='sos')
    return signal.sosfiltfilt(sos, signal_data)


# =============================================================================
# EDA FEATURE EXTRACTION (17 features)
# =============================================================================

def extract_eda_features_window(eda_window, fs, peak_threshold=0.01, 
                                 outlier_min=0.01, outlier_max=1.0):
    """
    Extract 17 EDA features from a single window.
    
    Features:
        Tonic (SCL) - 8 features:
            scl_mean, scl_std, scl_min, scl_max, scl_range, scl_median, scl_slope, scl_auc
        Phasic (SCR) - 7 features:
            scr_peak_count, scr_amplitude_mean, scr_amplitude_max, scr_amplitude_std,
            scr_amplitude_sum, scr_rise_time_mean, scr_recovery_time_mean
        Distribution - 2 features:
            eda_skewness, eda_kurtosis
    
    Args:
        eda_window: EDA signal array for this window
        fs: Sampling frequency (Hz)
        peak_threshold: Minimum SCR amplitude (μS)
        outlier_min: Minimum valid SCR amplitude (μS)
        outlier_max: Maximum valid SCR amplitude (μS)
    
    Returns:
        Dictionary of 17 EDA features
    
    Note:
        This function uses cvxEDA decomposition ONLY.
        If cvxopt is not installed, it will raise an error.
        We intentionally do NOT fall back to other methods because
        they produce materially different results.
    """
    # Filter EDA signal (1.8 Hz lowpass as recommended by NeuroKit2)
    eda_filtered = lowpass_filter(eda_window, fs, cutoff=1.8)
    
    # Decompose into tonic and phasic using cvxEDA ONLY
    # NO FALLBACK - this will fail if cvxopt is not installed
    try:
        eda_decomposed = nk.eda_phasic(eda_filtered, sampling_rate=fs, method='cvxeda')
    except Exception as e:
        raise RuntimeError(
            f"cvxEDA decomposition failed: {e}\n"
            "This pipeline intentionally disables fallbacks because alternative methods "
            "(smoothmedian, simple filtering) produce materially different results.\n"
            "Install cvxopt: pip install cvxopt"
        )
    
    eda_tonic = eda_decomposed['EDA_Tonic'].values
    eda_phasic = eda_decomposed['EDA_Phasic'].values
    
    # Detect SCR peaks
    try:
        _, peaks_info = nk.eda_peaks(eda_phasic, sampling_rate=fs, amplitude_min=peak_threshold)
        scr_peak_indices = peaks_info['SCR_Peaks']
        scr_amplitudes = peaks_info['SCR_Amplitude']
        scr_rise_times = peaks_info['SCR_RiseTime']
        scr_recovery_times = peaks_info['SCR_Recovery']
    except:
        # No peaks found
        scr_peak_indices = np.array([])
        scr_amplitudes = np.array([])
        scr_rise_times = np.array([])
        scr_recovery_times = np.array([])
    
    # Apply outlier rejection to SCR amplitudes
    if len(scr_amplitudes) > 0:
        valid_mask = (scr_amplitudes >= outlier_min) & (scr_amplitudes <= outlier_max)
        valid_amplitudes = scr_amplitudes[valid_mask]
        valid_rise_times = scr_rise_times[valid_mask] if len(scr_rise_times) == len(scr_amplitudes) else np.array([])
        valid_recovery_times = scr_recovery_times[valid_mask] if len(scr_recovery_times) == len(scr_amplitudes) else np.array([])
    else:
        valid_amplitudes = np.array([])
        valid_rise_times = np.array([])
        valid_recovery_times = np.array([])
    
    # ============== TONIC (SCL) FEATURES (8) ==============
    scl_mean = np.mean(eda_tonic)
    scl_std = np.std(eda_tonic)
    scl_min = np.min(eda_tonic)
    scl_max = np.max(eda_tonic)
    scl_range = scl_max - scl_min
    scl_median = np.median(eda_tonic)
    
    # SCL slope (linear trend)
    x = np.arange(len(eda_tonic))
    slope, _ = np.polyfit(x, eda_tonic, 1)
    scl_slope = slope
    
    # SCL area under curve (using scipy.integrate.trapezoid for compatibility)
    scl_auc = trapezoid(eda_tonic)
    
    # ============== PHASIC (SCR) FEATURES (7) ==============
    scr_peak_count = len(valid_amplitudes)
    scr_amplitude_mean = np.mean(valid_amplitudes) if scr_peak_count > 0 else 0.0
    scr_amplitude_max = np.max(valid_amplitudes) if scr_peak_count > 0 else 0.0
    scr_amplitude_std = np.std(valid_amplitudes) if scr_peak_count > 0 else 0.0
    scr_amplitude_sum = np.sum(valid_amplitudes) if scr_peak_count > 0 else 0.0
    scr_rise_time_mean = np.mean(valid_rise_times) if len(valid_rise_times) > 0 else 0.0
    scr_recovery_time_mean = np.mean(valid_recovery_times) if len(valid_recovery_times) > 0 else 0.0
    
    # ============== DISTRIBUTION FEATURES (2) ==============
    eda_skewness = stats.skew(eda_filtered)
    eda_kurtosis = stats.kurtosis(eda_filtered)
    
    return {
        # Tonic features (8)
        'scl_mean': scl_mean,
        'scl_std': scl_std,
        'scl_min': scl_min,
        'scl_max': scl_max,
        'scl_range': scl_range,
        'scl_median': scl_median,
        'scl_slope': scl_slope,
        'scl_auc': scl_auc,
        # Phasic features (7)
        'scr_peak_count': scr_peak_count,
        'scr_amplitude_mean': scr_amplitude_mean,
        'scr_amplitude_max': scr_amplitude_max,
        'scr_amplitude_std': scr_amplitude_std,
        'scr_amplitude_sum': scr_amplitude_sum,
        'scr_rise_time_mean': scr_rise_time_mean,
        'scr_recovery_time_mean': scr_recovery_time_mean,
        # Distribution features (2)
        'eda_skewness': eda_skewness,
        'eda_kurtosis': eda_kurtosis,
    }


# =============================================================================
# HRV FEATURE EXTRACTION (15 features)
# =============================================================================

def extract_hrv_features_window(bvp_window, fs):
    """
    Extract 15 HRV features from a single window of BVP data.
    
    Features:
        Time-domain (9):
            hrv_mean_hr, hrv_rmssd, hrv_sdnn, hrv_pnn50, hrv_mean_rr,
            hrv_median_rr, hrv_sdsd, hrv_min_rr, hrv_max_rr
        Frequency-domain (5):
            hrv_lf_power, hrv_hf_power, hrv_lf_hf_ratio, hrv_vlf_power, hrv_total_power
        Non-linear (1):
            hrv_sd1
    
    Args:
        bvp_window: BVP signal array for this window
        fs: Sampling frequency (Hz)
    
    Returns:
        Dictionary of 15 HRV features
    
    Note:
        FIXED: hrv_mean_hr now correctly represents heart rate in BPM.
        Grant's original code incorrectly used HRV_MeanNN (which is mean NN interval in ms).
        Correct calculation: HR = 60000 / MeanNN_ms
    """
    # Default features (returned if processing fails)
    default_features = {
        'hrv_mean_hr': 0.0,
        'hrv_rmssd': 0.0,
        'hrv_sdnn': 0.0,
        'hrv_pnn50': 0.0,
        'hrv_mean_rr': 0.0,
        'hrv_median_rr': 0.0,
        'hrv_sdsd': 0.0,
        'hrv_min_rr': 0.0,
        'hrv_max_rr': 0.0,
        'hrv_lf_power': 0.0,
        'hrv_hf_power': 0.0,
        'hrv_lf_hf_ratio': 0.0,
        'hrv_vlf_power': 0.0,
        'hrv_total_power': 0.0,
        'hrv_sd1': 0.0,
    }
    
    try:
        # Clean BVP signal and detect peaks using NeuroKit2
        ppg_signals, info = nk.ppg_process(bvp_window, sampling_rate=fs)
        peaks = info['PPG_Peaks']
        
        # Need at least 2 peaks to calculate HRV
        if len(peaks) < 2:
            return default_features
        
        # Calculate HRV metrics using NeuroKit2
        hrv_time = nk.hrv_time(peaks, sampling_rate=fs, show=False)
        hrv_freq = nk.hrv_frequency(peaks, sampling_rate=fs, show=False)
        hrv_nonlinear = nk.hrv_nonlinear(peaks, sampling_rate=fs, show=False)
        
        # Extract time-domain features (9)
        # FIXED: Calculate actual heart rate from MeanNN
        # MeanNN is in milliseconds, HR = 60000 / MeanNN
        mean_nn = hrv_time['HRV_MeanNN'].values[0] if 'HRV_MeanNN' in hrv_time.columns else 0.0
        hrv_mean_hr = 60000.0 / mean_nn if mean_nn > 0 else 0.0  # CORRECTED
        
        features = {
            'hrv_mean_hr': hrv_mean_hr,  # Now correctly in BPM
            'hrv_rmssd': hrv_time['HRV_RMSSD'].values[0] if 'HRV_RMSSD' in hrv_time.columns else 0.0,
            'hrv_sdnn': hrv_time['HRV_SDNN'].values[0] if 'HRV_SDNN' in hrv_time.columns else 0.0,
            'hrv_pnn50': hrv_time['HRV_pNN50'].values[0] if 'HRV_pNN50' in hrv_time.columns else 0.0,
            'hrv_mean_rr': mean_nn,  # Keep MeanNN as hrv_mean_rr (in ms)
            'hrv_median_rr': hrv_time['HRV_MedianNN'].values[0] if 'HRV_MedianNN' in hrv_time.columns else 0.0,
            'hrv_sdsd': hrv_time['HRV_SDSD'].values[0] if 'HRV_SDSD' in hrv_time.columns else 0.0,
            'hrv_min_rr': hrv_time['HRV_MinNN'].values[0] if 'HRV_MinNN' in hrv_time.columns else 0.0,
            'hrv_max_rr': hrv_time['HRV_MaxNN'].values[0] if 'HRV_MaxNN' in hrv_time.columns else 0.0,
        }
        
        # Extract frequency-domain features (5)
        features.update({
            'hrv_lf_power': hrv_freq['HRV_LF'].values[0] if 'HRV_LF' in hrv_freq.columns else 0.0,
            'hrv_hf_power': hrv_freq['HRV_HF'].values[0] if 'HRV_HF' in hrv_freq.columns else 0.0,
            'hrv_lf_hf_ratio': hrv_freq['HRV_LFHF'].values[0] if 'HRV_LFHF' in hrv_freq.columns else 0.0,
            'hrv_vlf_power': hrv_freq['HRV_VLF'].values[0] if 'HRV_VLF' in hrv_freq.columns else 0.0,
            'hrv_total_power': hrv_freq['HRV_TP'].values[0] if 'HRV_TP' in hrv_freq.columns else 0.0,
        })
        
        # Extract non-linear feature (1)
        features['hrv_sd1'] = hrv_nonlinear['HRV_SD1'].values[0] if 'HRV_SD1' in hrv_nonlinear.columns else 0.0
        
        return features
        
    except Exception as e:
        # If HRV extraction fails for this window, return defaults
        return default_features


# =============================================================================
# TEMPERATURE FEATURE EXTRACTION (6 features)
# =============================================================================

def extract_temp_features_window(temp_window, fs):
    """
    Extract 6 temperature features from a single window.
    
    Features:
        temp_mean, temp_std, temp_min, temp_max, temp_median, temp_slope
    
    Args:
        temp_window: Temperature signal array for this window
        fs: Sampling frequency (Hz)
    
    Returns:
        Dictionary of 6 temperature features
    """
    temp_mean = np.mean(temp_window)
    temp_std = np.std(temp_window)
    temp_min = np.min(temp_window)
    temp_max = np.max(temp_window)
    temp_median = np.median(temp_window)
    
    # Temperature slope (linear trend)
    x = np.arange(len(temp_window))
    slope, _ = np.polyfit(x, temp_window, 1)
    temp_slope = slope
    
    return {
        'temp_mean': temp_mean,
        'temp_std': temp_std,
        'temp_min': temp_min,
        'temp_max': temp_max,
        'temp_median': temp_median,
        'temp_slope': temp_slope,
    }


# =============================================================================
# ACCELEROMETER FEATURE EXTRACTION (8 features)
# =============================================================================

def extract_acc_features_window(acc_x, acc_y, acc_z, fs):
    """
    Extract 8 accelerometer features from a single window.
    
    Features:
        Per-axis std (3):
            acc_x_std, acc_y_std, acc_z_std
        Magnitude features (3):
            acc_magnitude_mean, acc_magnitude_std, acc_magnitude_max
        Activity metrics (2):
            acc_sma (Signal Magnitude Area), acc_energy
    
    Args:
        acc_x, acc_y, acc_z: Accelerometer data arrays for this window
        fs: Sampling frequency (Hz)
    
    Returns:
        Dictionary of 8 accelerometer features
    """
    # Per-axis std
    acc_x_std = np.std(acc_x)
    acc_y_std = np.std(acc_y)
    acc_z_std = np.std(acc_z)
    
    # Magnitude features
    magnitude = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
    acc_magnitude_mean = np.mean(magnitude)
    acc_magnitude_std = np.std(magnitude)
    acc_magnitude_max = np.max(magnitude)
    
    # Signal Magnitude Area (SMA) - average of absolute values
    acc_sma = (np.sum(np.abs(acc_x)) + np.sum(np.abs(acc_y)) + np.sum(np.abs(acc_z))) / len(acc_x)
    
    # Energy
    acc_energy = np.sum(magnitude**2) / len(magnitude)
    
    return {
        'acc_x_std': acc_x_std,
        'acc_y_std': acc_y_std,
        'acc_z_std': acc_z_std,
        'acc_magnitude_mean': acc_magnitude_mean,
        'acc_magnitude_std': acc_magnitude_std,
        'acc_magnitude_max': acc_magnitude_max,
        'acc_sma': acc_sma,
        'acc_energy': acc_energy,
    }

