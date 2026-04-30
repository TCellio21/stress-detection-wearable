"""
WESAD Windowed Dataset Extraction Script
========================================
Supports sliding windows, decoupled HRV extraction (3 min freq, 1 min time),
Causal EDA alignment (6.4s shift), and dual dataset generation (30s and 60s strides).
"""

import os
import pickle
import numpy as np
if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid
import pandas as pd
import neurokit2 as nk
from scipy import stats, signal
from pathlib import Path
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

# =====================================================================
# CONFIGURATION
# =====================================================================

# Load .env from repo root, then this script's folder (matches Updated Extraction pipeline)
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[1]  # script is at <repo>/dataset/WESAD Extraction/
load_dotenv(_REPO_ROOT / ".env")
load_dotenv(_SCRIPT_DIR / ".env")

WESAD_PATH = os.environ.get("WESAD_PATH")
if not WESAD_PATH:
    raise RuntimeError(
        "WESAD_PATH not set. Copy .env.example to .env at the repo root "
        "and set WESAD_PATH=/path/to/WESAD"
    )

OUTPUT_DIR = _SCRIPT_DIR

SUBJECTS = ['S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 
            'S13', 'S14', 'S15', 'S16', 'S17']

WINDOW_SIZE_SEC = 60
HRV_FREQ_WINDOW_SEC = 180
STRIDE_SEC = 30  # Step size. We will filter rows later to get the 60s stride dataset.

FS_EDA = 4  
FS_BVP = 64  
FS_TEMP = 4  
FS_ACC = 32  
FS_LABEL = 700  

PEAK_THRESHOLD = 0.01  
OUTLIER_MIN = 0.01  
OUTLIER_MAX = 5.0  

# =====================================================================
# FEATURE BUCKETS
# =====================================================================

BUCKET_1_RAW = [
    'acc_x_std', 'acc_y_std', 'acc_z_std', 'acc_magnitude_mean', 'acc_magnitude_std', 
    'acc_magnitude_max', 'acc_sma', 'acc_energy',
    'eda_skewness', 'eda_kurtosis',
    'hrv_lf_hf_ratio'
]

BUCKET_2_RAW_Z = [
    'hrv_mean_hr', 'hrv_rmssd', 'hrv_pnn50',
    'temp_mean', 'temp_min', 'temp_max', 'temp_median',
    'scr_peak_count'
]

BUCKET_3_Z = [
    'scl_mean', 'scl_std', 'scl_min', 'scl_max', 'scl_median', 'scl_range', 'scl_auc', 'scl_slope',
    'scr_amplitude_mean', 'scr_amplitude_max', 'scr_amplitude_std', 'scr_amplitude_sum', 
    'scr_rise_time_mean', 'scr_recovery_time_mean',
    'temp_std', 'temp_slope',
    'hrv_sdnn', 'hrv_mean_rr', 'hrv_median_rr', 'hrv_sdsd', 'hrv_min_rr', 'hrv_max_rr', 
    'hrv_lf_power', 'hrv_hf_power', 'hrv_vlf_power', 'hrv_total_power', 'hrv_sd1'
]

# =====================================================================
# HELPER FUNCTIONS
# =====================================================================

def _safe_z_score(value, baseline_mean, baseline_std):
    if pd.isna(value) or pd.isna(baseline_mean) or pd.isna(baseline_std) or baseline_std == 0:
        return 0.0
    return (value - baseline_mean) / baseline_std

# =====================================================================
# FEATURE EXTRACTION 
# =====================================================================

def filter_and_align_eda(eda_raw, fs=4):
    nyq = 0.5 * fs
    
    # 1. Artifact Smoothing
    b1, a1 = signal.butter(1, 1.0 / nyq, btype='low')
    zi1 = signal.lfilter_zi(b1, a1) * eda_raw[0]
    smoothed_eda, _ = signal.lfilter(b1, a1, eda_raw, zi=zi1)
    
    # 2. Tonic Extraction
    b2, a2 = signal.butter(2, 0.05 / nyq, btype='low')
    zi2 = signal.lfilter_zi(b2, a2) * smoothed_eda[0]
    eda_tonic, _ = signal.lfilter(b2, a2, smoothed_eda, zi=zi2)
    
    # 3. Phasic Extraction
    dt = 1.0 / fs
    derivative = np.gradient(smoothed_eda, dt)
    eda_phasic = np.maximum(0, derivative)
    
    # 4. ALIGNMENT (Shift backward by 6.4s to correct group delay)
    delay_samples = round(6.4 * fs)
    eda_tonic_aligned = np.pad(eda_tonic[delay_samples:], (0, delay_samples), mode='edge')
    eda_phasic_aligned = np.pad(eda_phasic[delay_samples:], (0, delay_samples), mode='constant', constant_values=0)
    smoothed_eda_aligned = np.pad(smoothed_eda[delay_samples:], (0, delay_samples), mode='edge')
    
    # Detect peaks on ALIGNED phasic signal
    _, peaks_info = nk.eda_peaks(eda_phasic_aligned, sampling_rate=fs, amplitude_min=PEAK_THRESHOLD)
    
    return smoothed_eda_aligned, eda_tonic_aligned, eda_phasic_aligned, peaks_info


def extract_eda_features_window(smoothed_eda, eda_tonic, eda_phasic, peaks_info, start_idx, end_idx):
    window_tonic = eda_tonic[start_idx:end_idx]
    window_raw = smoothed_eda[start_idx:end_idx]
    
    # Tonic (SCL)
    scl_mean = np.mean(window_tonic) if len(window_tonic) > 0 else 0
    scl_std = np.std(window_tonic) if len(window_tonic) > 0 else 0
    scl_min = np.min(window_tonic) if len(window_tonic) > 0 else 0
    scl_max = np.max(window_tonic) if len(window_tonic) > 0 else 0
    scl_range = scl_max - scl_min
    scl_median = np.median(window_tonic) if len(window_tonic) > 0 else 0
    scl_slope = np.polyfit(np.arange(len(window_tonic)), window_tonic, 1)[0] if len(window_tonic) > 1 else 0
    scl_auc = np.trapz(window_tonic) if len(window_tonic) > 0 else 0

    # Phasic (SCR)
    scr_peak_indices = peaks_info['SCR_Peaks']
    mask = (scr_peak_indices >= start_idx) & (scr_peak_indices < end_idx)
    
    valid_amplitudes = peaks_info['SCR_Amplitude'][mask]
    valid_rise = peaks_info['SCR_RiseTime'][mask]
    valid_recovery = peaks_info['SCR_Recovery'][mask]
    
    # Apply Outlier boundaries
    outlier_mask = (valid_amplitudes >= OUTLIER_MIN) & (valid_amplitudes <= OUTLIER_MAX)
    valid_amplitudes = valid_amplitudes[outlier_mask]
    valid_rise = valid_rise[outlier_mask]
    valid_recovery = valid_recovery[outlier_mask]

    scr_peak_count = len(valid_amplitudes)
    scr_amplitude_mean = np.mean(valid_amplitudes) if scr_peak_count > 0 else 0.0
    scr_amplitude_max = np.max(valid_amplitudes) if scr_peak_count > 0 else 0.0
    scr_amplitude_std = np.std(valid_amplitudes) if scr_peak_count > 0 else 0.0
    scr_amplitude_sum = np.sum(valid_amplitudes) if scr_peak_count > 0 else 0.0
    scr_rise_time_mean = np.mean(valid_rise) if len(valid_rise) > 0 else 0.0
    scr_recovery_time_mean = np.mean(valid_recovery) if len(valid_recovery) > 0 else 0.0

    return {
        'scl_mean': scl_mean, 'scl_std': scl_std, 'scl_min': scl_min, 'scl_max': scl_max,
        'scl_range': scl_range, 'scl_median': scl_median, 'scl_slope': scl_slope, 'scl_auc': scl_auc,
        'scr_peak_count': scr_peak_count, 'scr_amplitude_mean': scr_amplitude_mean,
        'scr_amplitude_max': scr_amplitude_max, 'scr_amplitude_std': scr_amplitude_std,
        'scr_amplitude_sum': scr_amplitude_sum, 'scr_rise_time_mean': scr_rise_time_mean,
        'scr_recovery_time_mean': scr_recovery_time_mean,
        'eda_skewness': stats.skew(window_raw) if len(window_raw) > 0 else 0, 
        'eda_kurtosis': stats.kurtosis(window_raw) if len(window_raw) > 0 else 0
    }

def extract_temp_features_window(temp_raw, start_idx, end_idx):
    window_temp = temp_raw[start_idx:end_idx]
    if len(window_temp) == 0:
        return {'temp_mean': 0, 'temp_std': 0, 'temp_min': 0, 'temp_max': 0, 'temp_median': 0, 'temp_slope': 0}
    return {
        'temp_mean': np.mean(window_temp), 'temp_std': np.std(window_temp),
        'temp_min': np.min(window_temp), 'temp_max': np.max(window_temp),
        'temp_median': np.median(window_temp),
        'temp_slope': np.polyfit(np.arange(len(window_temp)), window_temp, 1)[0] if len(window_temp) > 1 else 0
    }

def extract_acc_features_window(acc_raw, start_idx, end_idx):
    window = acc_raw[start_idx:end_idx]
    if len(window) == 0:
        return {
            'acc_x_std': 0, 'acc_y_std': 0, 'acc_z_std': 0, 'acc_magnitude_mean': 0,
            'acc_magnitude_std': 0, 'acc_magnitude_max': 0, 'acc_sma': 0, 'acc_energy': 0
        }
    x, y, z = window[:,0], window[:,1], window[:,2]
    mag = np.sqrt(x**2 + y**2 + z**2)
    return {
        'acc_x_std': np.std(x), 'acc_y_std': np.std(y), 'acc_z_std': np.std(z),
        'acc_magnitude_mean': np.mean(mag), 'acc_magnitude_std': np.std(mag), 'acc_magnitude_max': np.max(mag),
        'acc_sma': (np.sum(np.abs(x)) + np.sum(np.abs(y)) + np.sum(np.abs(z))) / len(x),
        'acc_energy': np.sum(mag**2) / len(mag)
    }

def extract_hrv_decoupled_window(bvp_raw, start_idx_60, start_idx_180, end_idx, fs=64):
    window_bvp_60 = bvp_raw[max(0, start_idx_60):end_idx]
    window_bvp_180 = bvp_raw[max(0, start_idx_180):end_idx]

    hrv = {
        'hrv_mean_hr': 0.0, 'hrv_rmssd': 0.0, 'hrv_sdnn': 0.0, 'hrv_pnn50': 0.0,
        'hrv_mean_rr': 0.0, 'hrv_median_rr': 0.0, 'hrv_sdsd': 0.0, 'hrv_min_rr': 0.0,
        'hrv_max_rr': 0.0, 'hrv_lf_power': 0.0, 'hrv_hf_power': 0.0,
        'hrv_lf_hf_ratio': 0.0, 'hrv_vlf_power': 0.0, 'hrv_total_power': 0.0,
        'hrv_sd1': 0.0
    }

    try:
        if len(window_bvp_60) > fs * 10:
            _, info_60 = nk.ppg_process(window_bvp_60, sampling_rate=fs)
            peaks_60 = info_60['PPG_Peaks']
            if len(peaks_60) >= 2:
                hrv_time = nk.hrv_time(peaks_60, sampling_rate=fs, show=False)
                hrv_nl = nk.hrv_nonlinear(peaks_60, sampling_rate=fs, show=False)
                hrv.update({
                    'hrv_mean_hr': hrv_time['HRV_MeanNN'].values[0] if 'HRV_MeanNN' in hrv_time else 0.0,
                    'hrv_rmssd': hrv_time['HRV_RMSSD'].values[0] if 'HRV_RMSSD' in hrv_time else 0.0,
                    'hrv_sdnn': hrv_time['HRV_SDNN'].values[0] if 'HRV_SDNN' in hrv_time else 0.0,
                    'hrv_pnn50': hrv_time['HRV_pNN50'].values[0] if 'HRV_pNN50' in hrv_time else 0.0,
                    'hrv_mean_rr': hrv_time['HRV_MeanNN'].values[0] if 'HRV_MeanNN' in hrv_time else 0.0,
                    'hrv_median_rr': hrv_time['HRV_MedianNN'].values[0] if 'HRV_MedianNN' in hrv_time else 0.0,
                    'hrv_sdsd': hrv_time['HRV_SDSD'].values[0] if 'HRV_SDSD' in hrv_time else 0.0,
                    'hrv_min_rr': hrv_time['HRV_MinNN'].values[0] if 'HRV_MinNN' in hrv_time else 0.0,
                    'hrv_max_rr': hrv_time['HRV_MaxNN'].values[0] if 'HRV_MaxNN' in hrv_time else 0.0,
                    'hrv_sd1': hrv_nl['HRV_SD1'].values[0] if 'HRV_SD1' in hrv_nl else 0.0,
                })
    except Exception: pass

    try:
        if len(window_bvp_180) > fs * 60:
            _, info_180 = nk.ppg_process(window_bvp_180, sampling_rate=fs)
            peaks_180 = info_180['PPG_Peaks']
            if len(peaks_180) >= 2:
                hrv_freq = nk.hrv_frequency(peaks_180, sampling_rate=fs, show=False)
                hrv.update({
                    'hrv_lf_power': hrv_freq['HRV_LF'].values[0] if 'HRV_LF' in hrv_freq else 0.0,
                    'hrv_hf_power': hrv_freq['HRV_HF'].values[0] if 'HRV_HF' in hrv_freq else 0.0,
                    'hrv_lf_hf_ratio': hrv_freq['HRV_LFHF'].values[0] if 'HRV_LFHF' in hrv_freq else 0.0,
                    'hrv_vlf_power': hrv_freq['HRV_VLF'].values[0] if 'HRV_VLF' in hrv_freq else 0.0,
                    'hrv_total_power': hrv_freq['HRV_TP'].values[0] if 'HRV_TP' in hrv_freq else 0.0,
                })
    except Exception: pass

    return hrv


# =====================================================================
# MAIN PROCESSING 
# =====================================================================

def process_subject(subject_id):
    print(f"\nProcessing {subject_id}...")
    file_path = Path(WESAD_PATH) / subject_id / f"{subject_id}.pkl"
    with open(file_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')

    eda_raw = data['signal']['wrist']['EDA'].flatten()
    bvp_raw = data['signal']['wrist']['BVP'].flatten()
    temp_raw = data['signal']['wrist']['TEMP'].flatten()
    acc_raw = data['signal']['wrist']['ACC']
    labels = data['label'].flatten()

    print("  Aligning & Filtering EDA...")
    smoothed_eda, eda_tonic, eda_phasic, peaks_info = filter_and_align_eda(eda_raw, FS_EDA)

    total_sec = len(labels) / FS_LABEL
    start_sec = max(WINDOW_SIZE_SEC, HRV_FREQ_WINDOW_SEC)
    
    windows_features = []

    print(f"  Sliding Window Extraction (Stride {STRIDE_SEC}s)...")
    for t_end in np.arange(start_sec, total_sec, STRIDE_SEC):
        t_start_60 = t_end - WINDOW_SIZE_SEC
        t_start_180 = t_end - HRV_FREQ_WINDOW_SEC

        # Get the label using Mode for the 60s window
        lbl_start, lbl_end = int(t_start_60 * FS_LABEL), int(t_end * FS_LABEL)
        win_labels = labels[lbl_start:lbl_end]
        if len(win_labels) == 0: continue
        
        mode_result = stats.mode(win_labels, keepdims=True)
        window_label = mode_result[0][0]
        
        if window_label == 0: 
            continue # Skip transient / unlabelled regions

        # Extract EDA features
        e_start, e_end = int(t_start_60 * FS_EDA), int(t_end * FS_EDA)
        eda_feats = extract_eda_features_window(smoothed_eda, eda_tonic, eda_phasic, peaks_info, e_start, e_end)

        # Extract TEMP features
        t_start, t_end_idx = int(t_start_60 * FS_TEMP), int(t_end * FS_TEMP)
        temp_feats = extract_temp_features_window(temp_raw, t_start, t_end_idx)

        # Extract ACC features
        a_start, a_end = int(t_start_60 * FS_ACC), int(t_end * FS_ACC)
        acc_feats = extract_acc_features_window(acc_raw, a_start, a_end)

        # Extract Decoupled HRV features
        h_start_60, h_end_idx = int(t_start_60 * FS_BVP), int(t_end * FS_BVP)
        h_start_180 = int(t_start_180 * FS_BVP)
        hrv_feats = extract_hrv_decoupled_window(bvp_raw, h_start_60, h_start_180, h_end_idx, FS_BVP)

        # Combine
        combined = {'subject_id': subject_id, 'raw_label': window_label, 'time_end_sec': t_end}
        combined.update(eda_feats)
        combined.update(temp_feats)
        combined.update(acc_feats)
        combined.update(hrv_feats)
        
        windows_features.append(combined)

    return pd.DataFrame(windows_features)

if __name__ == "__main__":
    print("="*70)
    print("WESAD Sliding Window Feature Extraction (Decoupled & Aligned)")
    print("="*70)
    
    all_subject_data = []

    for subject_id in SUBJECTS:
        try:
            df_subj = process_subject(subject_id)
            all_subject_data.append(df_subj)
        except Exception as e:
            print(f"  ERROR processing {subject_id}: {e}")

    final_df = pd.concat(all_subject_data, ignore_index=True)
    
    # -----------------------------------------------------------------
    # Global Z-Scoring per Subject
    # -----------------------------------------------------------------
    print(f"\n{'='*70}")
    print("Calculating Global Z-Scores and Partitions...")
    
    normalized_list = []
    
    for subject_id, subj_df in final_df.groupby('subject_id'):
        # Establish globals using all Baseline (label 1) data for this subject
        baseline_df = subj_df[subj_df['raw_label'] == 1]
        baseline_stats = {}
        
        if not baseline_df.empty:
            for col in BUCKET_2_RAW_Z + BUCKET_3_Z:
                col_values = baseline_df[col].dropna()
                if len(col_values) > 0:
                    baseline_stats[col] = {'mean': col_values.mean(), 'std': col_values.std(ddof=0)}

        subj_norm = pd.DataFrame()
        subj_norm['subject_id'] = subj_df['subject_id']
        subj_norm['raw_label'] = subj_df['raw_label']
        subj_norm['time_end_sec'] = subj_df['time_end_sec']
        
        # 3-Class Mapping
        subj_norm['label'] = subj_norm['raw_label'].map({1: 'non-stress', 2: 'stress', 3: 'non-stress', 4: 'non-stress'})

        # Bucket 1: Strictly Raw
        for col in BUCKET_1_RAW:
            subj_norm[col] = subj_df[col]
            
        # Bucket 2: Raw + Z-Scored
        for col in BUCKET_2_RAW_Z:
            subj_norm[f"raw_{col}"] = subj_df[col]
            stats_col = baseline_stats.get(col)
            if not stats_col:
                subj_norm[f"z_{col}"] = 0.0
            else:
                subj_norm[f"z_{col}"] = subj_df[col].apply(lambda x: _safe_z_score(x, stats_col['mean'], stats_col['std']))
                
        # Bucket 3: Strictly Z-Scored
        for col in BUCKET_3_Z:
            stats_col = baseline_stats.get(col)
            if not stats_col:
                subj_norm[f"z_{col}"] = 0.0
            else:
                subj_norm[f"z_{col}"] = subj_df[col].apply(lambda x: _safe_z_score(x, stats_col['mean'], stats_col['std']))

        normalized_list.append(subj_norm)

    final_output_df = pd.concat(normalized_list, ignore_index=True)
    
    # Drop NAs based on the 3-class mapping (label 0 will drop entirely)
    final_output_df = final_output_df.dropna(subset=['label'])

    # Enforce Order
    final_cols = BUCKET_1_RAW + \
                 [f"raw_{c}" for c in BUCKET_2_RAW_Z] + \
                 [f"z_{c}" for c in BUCKET_2_RAW_Z] + \
                 [f"z_{c}" for c in BUCKET_3_Z] + \
                 ['raw_label', 'label', 'time_end_sec', 'subject_id']
                 
    final_output_df = final_output_df[final_cols]

    # Save Datasets
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    out_30 = OUTPUT_DIR / "causal_features_stride30.csv"
    final_output_df.to_csv(out_30, index=False)
    
    # Generate 60s stride by filtering based on time_end_sec mod 60 == 0
    # Because start_sec is 180 (mod 60 = 0), steps of 30 give 180, 210, 240, 270, 300
    # Filtering mod 60 == 0 gives 180, 240, 300, matching a 60s contiguous jump exactly.
    df_60 = final_output_df[final_output_df['time_end_sec'] % 60 == 0]
    out_60 = OUTPUT_DIR / "causal_features_stride60.csv"
    df_60.to_csv(out_60, index=False)

    print(f"\nEXTRACTION & PARTITIONING COMPLETE!")
    print(f"Total features per window: {len(final_cols) - 4}")
    print(f"30s Stride Dataset Saved: {out_30} (Shape: {final_output_df.shape})")
    print(f"60s Stride Dataset Saved: {out_60} (Shape: {df_60.shape})")
    print("="*70)
