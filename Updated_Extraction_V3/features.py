"""
V3 feature extraction.

Operates on per-window slices of the preprocessed signals from preprocessing.py.
Extracts a 48-feature catalog covering all four wrist modalities. Phase 3 will
add unit tests and the full citation-backed catalog doc; this module is the
runnable feature extractor that Phase 2 needs to evaluate (window, step) tuples.

Feature counts vs. V1 (46) and V2 (46):
- EDA:  17 (unchanged from V1; the underlying decomposition is V3 HP-phasic)
- HRV:  13 (V1's 15 minus hrv_vlf_power and hrv_sd1 — see audit §2.2)
- TEMP: 6  (unchanged)
- ACC:  12 (V1's 8 + 4 V3 jerk/activity features)
- Total: 48

HRV motion gating: when the window's 95th-percentile jerk exceeds the
configured threshold, all HRV features are returned as NaN with hrv_valid=0.
Phase 6 chooses imputation strategy.
"""

from __future__ import annotations

from typing import Optional

import neurokit2 as nk
import numpy as np
from scipy import stats

try:
    from scipy.integrate import trapezoid as _trapezoid  # SciPy ≥ 1.6
except ImportError:  # pragma: no cover
    from numpy import trapz as _trapezoid

from preprocessing import EdaResult, BvpResult, TempResult, AccResult, window_motion_metric


# ============================================================================
# EDA — 17 features
# ============================================================================

def extract_eda_features(eda: EdaResult, start: int, end: int, fs: int,
                         scr_amp_min: float, scr_amp_max: float) -> dict:
    """Extract 17 EDA features from samples [start, end) of preprocessed EDA.

    Distribution features (skew, kurtosis) are computed on the smoothed signal
    (matching V1; the bipolar phasic would give meaningless higher moments).
    SCR-related features filter the subject-level peaks_info to peaks falling
    within [start, end).
    """
    tonic = eda.tonic[start:end]
    smoothed = eda.smoothed[start:end]

    if len(tonic) == 0:
        return _eda_defaults()

    # Tonic (8)
    scl_min = float(np.min(tonic))
    scl_max = float(np.max(tonic))
    scl_slope, _ = np.polyfit(np.arange(len(tonic)), tonic, 1)
    feats = {
        "scl_mean": float(np.mean(tonic)),
        "scl_std": float(np.std(tonic)),
        "scl_min": scl_min,
        "scl_max": scl_max,
        "scl_range": scl_max - scl_min,
        "scl_median": float(np.median(tonic)),
        "scl_slope": float(scl_slope),
        "scl_auc": float(_trapezoid(tonic)),
    }

    # Phasic / SCR (7) — filter subject-level peaks to this window
    peaks = np.asarray(eda.peaks_info.get("SCR_Peaks", []), dtype=int)
    amps = np.asarray(eda.peaks_info.get("SCR_Amplitude", []), dtype=float)
    rises = np.asarray(eda.peaks_info.get("SCR_RiseTime", []), dtype=float)
    recovs = np.asarray(eda.peaks_info.get("SCR_Recovery", []), dtype=float)

    in_window = (peaks >= start) & (peaks < end)
    amps_w = amps[in_window] if len(amps) == len(peaks) else np.array([])
    rises_w = rises[in_window] if len(rises) == len(peaks) else np.array([])
    recovs_w = recovs[in_window] if len(recovs) == len(peaks) else np.array([])

    if len(amps_w) > 0:
        bound = (amps_w >= scr_amp_min) & (amps_w <= scr_amp_max)
        amps_w = amps_w[bound]
        rises_w = rises_w[bound] if len(rises_w) == len(bound) else rises_w
        recovs_w = recovs_w[bound] if len(recovs_w) == len(bound) else recovs_w

    n_scr = int(len(amps_w))
    feats.update({
        "scr_peak_count": n_scr,
        "scr_amplitude_mean": float(np.mean(amps_w)) if n_scr else 0.0,
        "scr_amplitude_max": float(np.max(amps_w)) if n_scr else 0.0,
        "scr_amplitude_std": float(np.std(amps_w)) if n_scr else 0.0,
        "scr_amplitude_sum": float(np.sum(amps_w)) if n_scr else 0.0,
        "scr_rise_time_mean": float(np.nanmean(rises_w)) if len(rises_w) and np.any(np.isfinite(rises_w)) else 0.0,
        "scr_recovery_time_mean": float(np.nanmean(recovs_w)) if len(recovs_w) and np.any(np.isfinite(recovs_w)) else 0.0,
    })

    # Distribution (2) — on smoothed (matches V1 convention)
    feats["eda_skewness"] = float(stats.skew(smoothed))
    feats["eda_kurtosis"] = float(stats.kurtosis(smoothed))
    return feats


def _eda_defaults() -> dict:
    return {k: 0.0 for k in [
        "scl_mean", "scl_std", "scl_min", "scl_max", "scl_range", "scl_median",
        "scl_slope", "scl_auc",
        "scr_peak_count", "scr_amplitude_mean", "scr_amplitude_max",
        "scr_amplitude_std", "scr_amplitude_sum", "scr_rise_time_mean",
        "scr_recovery_time_mean",
        "eda_skewness", "eda_kurtosis",
    ]}


# ============================================================================
# HRV — 13 features (V1's 15 minus hrv_vlf_power and hrv_sd1)
# ============================================================================

def extract_hrv_features(
    bvp: BvpResult,
    time_start_bvp: int, time_end_bvp: int,
    freq_start_bvp: int, freq_end_bvp: int,
    fs_bvp: int,
    motion_corrupted: bool,
) -> dict:
    """Extract HRV features with decoupled time/freq windows (Task Force 1996).

    - **Time-domain** features (mean_hr, RMSSD, SDNN, pNN50, etc.) come from the
      peaks falling in [time_start_bvp, time_end_bvp). This is typically the
      60 s feature window — Task Force 1996 says short-term time-domain HRV is
      stable at 60 s.
    - **Frequency-domain** features (LF, HF, LF/HF, total power) come from the
      peaks in [freq_start_bvp, freq_end_bvp). This is typically a wider 180 s
      lookback ending at the same `time_end_bvp` — Task Force 1996 requires
      ≥ 2 minutes for stable LF (0.04–0.15 Hz, lower-edge period 25 s).

    Returns NaN HRV (`hrv_valid=0`) if the window is motion-corrupted, has < 2
    peaks in the time window, or NK2 time-domain math fails. Frequency features
    are NaN'd individually (without invalidating time-domain) when the freq
    window is too short or has too few peaks for stable spectral estimation.
    """
    defaults = _hrv_defaults()
    if motion_corrupted:
        return defaults

    # ----- Time-domain on the time window -----
    time_peaks = bvp.peaks[(bvp.peaks >= time_start_bvp) & (bvp.peaks < time_end_bvp)]
    n_time_peaks = int(len(time_peaks))
    if n_time_peaks < 2:
        defaults["hrv_n_peaks"] = n_time_peaks
        return defaults

    time_peaks_local = time_peaks - time_start_bvp

    def _g(df, key):
        try:
            v = df[key].values[0] if key in df.columns else np.nan
            return float(v) if np.isfinite(v) else np.nan
        except Exception:
            return np.nan

    try:
        hrv_t = nk.hrv_time(time_peaks_local, sampling_rate=fs_bvp, show=False)
    except Exception:
        return defaults

    mean_nn = _g(hrv_t, "HRV_MeanNN")
    feats = {
        "hrv_mean_hr": 60000.0 / mean_nn if (mean_nn and mean_nn > 0) else np.nan,
        "hrv_rmssd": _g(hrv_t, "HRV_RMSSD"),
        "hrv_sdnn": _g(hrv_t, "HRV_SDNN"),
        "hrv_pnn50": _g(hrv_t, "HRV_pNN50"),
        "hrv_mean_rr": mean_nn,
        "hrv_median_rr": _g(hrv_t, "HRV_MedianNN"),
        "hrv_sdsd": _g(hrv_t, "HRV_SDSD"),
        "hrv_min_rr": _g(hrv_t, "HRV_MinNN"),
        "hrv_max_rr": _g(hrv_t, "HRV_MaxNN"),
        # Freq features default to NaN; filled below iff the freq window is long enough.
        "hrv_lf_power": np.nan,
        "hrv_hf_power": np.nan,
        "hrv_lf_hf_ratio": np.nan,
        "hrv_total_power": np.nan,
        "hrv_valid": 1,
        "hrv_n_peaks": n_time_peaks,
    }

    # ----- Frequency-domain on the (typically wider) freq window -----
    # Task Force 1996: skip freq features when the available window is too short
    # for the LF band's lower edge (0.04 Hz, period 25 s). 120 s is the minimum
    # the literature considers acceptable for short-term LF/HF estimates.
    freq_window_sec = (freq_end_bvp - freq_start_bvp) / fs_bvp
    if freq_window_sec >= 120:
        freq_peaks = bvp.peaks[(bvp.peaks >= freq_start_bvp) & (bvp.peaks < freq_end_bvp)]
        # ~30 beats minimum for a usable PSD on the IBI series (~1 LF cycle).
        if len(freq_peaks) >= 30:
            freq_peaks_local = freq_peaks - freq_start_bvp
            try:
                hrv_f = nk.hrv_frequency(freq_peaks_local, sampling_rate=fs_bvp, show=False)
                feats["hrv_lf_power"]    = _g(hrv_f, "HRV_LF")
                feats["hrv_hf_power"]    = _g(hrv_f, "HRV_HF")
                feats["hrv_lf_hf_ratio"] = _g(hrv_f, "HRV_LFHF")
                feats["hrv_total_power"] = _g(hrv_f, "HRV_TP")
            except Exception:
                pass

    return feats


def _hrv_defaults() -> dict:
    return {
        "hrv_mean_hr": np.nan, "hrv_rmssd": np.nan, "hrv_sdnn": np.nan,
        "hrv_pnn50": np.nan, "hrv_mean_rr": np.nan, "hrv_median_rr": np.nan,
        "hrv_sdsd": np.nan, "hrv_min_rr": np.nan, "hrv_max_rr": np.nan,
        "hrv_lf_power": np.nan, "hrv_hf_power": np.nan,
        "hrv_lf_hf_ratio": np.nan, "hrv_total_power": np.nan,
        "hrv_valid": 0, "hrv_n_peaks": 0,
    }


# ============================================================================
# TEMP — 6 features
# ============================================================================

def extract_temp_features(temp: TempResult, start: int, end: int, fs: int) -> dict:
    win = temp.smoothed[start:end]
    if len(win) == 0:
        return {k: 0.0 for k in [
            "temp_mean", "temp_std", "temp_min", "temp_max", "temp_median", "temp_slope"
        ]}
    slope, _ = np.polyfit(np.arange(len(win)), win, 1)
    return {
        "temp_mean": float(np.mean(win)),
        "temp_std": float(np.std(win)),
        "temp_min": float(np.min(win)),
        "temp_max": float(np.max(win)),
        "temp_median": float(np.median(win)),
        "temp_slope": float(slope),
    }


# ============================================================================
# ACC — 12 features (V1's 8 + 4 V3 jerk/activity)
# ============================================================================

def extract_acc_features(acc: AccResult, start: int, end: int, fs: int) -> dict:
    """ACC features in g (post unit-conversion). 4 new V3 features capture
    the fidget-band signal PROJECT_ADVICE.md identifies as critical for the
    classroom application.
    """
    if end <= start:
        return _acc_defaults()

    x = acc.x_g[start:end]
    y = acc.y_g[start:end]
    z = acc.z_g[start:end]
    mag = acc.magnitude[start:end]
    mag_act = acc.magnitude_activity[start:end]
    jerk = acc.jerk_magnitude[start:end]
    n = len(x)
    if n == 0:
        return _acc_defaults()

    return {
        # V1 features (8) — now in g
        "acc_x_std": float(np.std(x)),
        "acc_y_std": float(np.std(y)),
        "acc_z_std": float(np.std(z)),
        "acc_magnitude_mean": float(np.mean(mag)),
        "acc_magnitude_std": float(np.std(mag)),
        "acc_magnitude_max": float(np.max(mag)),
        "acc_sma": float(np.sum(np.abs(x) + np.abs(y) + np.abs(z)) / n),
        "acc_energy": float(np.sum(mag * mag) / n),
        # V3 additions (4)
        "acc_activity_mean": float(np.mean(mag_act)),
        "acc_jerk_mag_mean": float(np.mean(jerk)),
        "acc_jerk_mag_std": float(np.std(jerk)),
        "acc_jerk_mag_p95": float(np.percentile(jerk, 95)),
    }


def _acc_defaults() -> dict:
    return {k: 0.0 for k in [
        "acc_x_std", "acc_y_std", "acc_z_std",
        "acc_magnitude_mean", "acc_magnitude_std", "acc_magnitude_max",
        "acc_sma", "acc_energy",
        "acc_activity_mean",
        "acc_jerk_mag_mean", "acc_jerk_mag_std", "acc_jerk_mag_p95",
    ]}


# ============================================================================
# Helpers used by dataset_builder
# ============================================================================

# Feature column order (deterministic for downstream LOSO/SHAP)
EDA_FEATURES = list(_eda_defaults().keys())
HRV_FEATURES = [k for k in _hrv_defaults().keys() if k not in {"hrv_valid", "hrv_n_peaks"}]
TEMP_FEATURES = ["temp_mean", "temp_std", "temp_min", "temp_max", "temp_median", "temp_slope"]
ACC_FEATURES = list(_acc_defaults().keys())
ALL_FEATURES = EDA_FEATURES + HRV_FEATURES + TEMP_FEATURES + ACC_FEATURES


def feature_count() -> int:
    return len(ALL_FEATURES)
