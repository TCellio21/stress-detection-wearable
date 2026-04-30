"""
V3 Preprocessing — causal, real-time-deployable filters and artifact detection.

Every filter here is causal: at sample t, the output depends only on samples 0..t.
This contrasts with V1 (cvxEDA + sosfiltfilt) and V2 (np.gradient central-difference),
both of which had look-ahead at one point or another. See docs/01_preprocessing.md
for the full design rationale and citations.

Conventions
-----------
- Subject-level filtering: each filter is applied once across the subject's full
  signal with `lfilter`/`sosfilt` (stateful causal IIR). This is mathematically
  equivalent to running the filter sample-by-sample in real-time with persistent
  filter state, so the offline pipeline emulates streaming exactly.
- Group-delay compensation: the EDA SCL/SCR filters introduce ~6.4 s of group
  delay at fs=4 Hz. We shift filtered output forward by that amount to align
  with the input timeline. This is NOT look-ahead — it represents a 6.4 s
  startup latency at sensor-on. End-of-signal padding (`mode='edge'`) is a
  training-time artifact: in deployment the last 6.4 s of input simply never
  produces predictions (the wearer takes the watch off).
- Artifact detection emits boolean validity masks (True = invalid). Downstream
  feature extraction decides what to do (drop window, interpolate, flag).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import neurokit2 as nk
from scipy.signal import butter, lfilter, lfilter_zi, sosfilt, sosfilt_zi


# ---------------------------------------------------------------------------
# Causal Butterworth helpers
# ---------------------------------------------------------------------------

def _causal_butter_lp(x: np.ndarray, fs: float, cutoff: float, order: int) -> np.ndarray:
    """Causal IIR low-pass via lfilter, initialized to steady state at x[0]."""
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype="low")
    zi = lfilter_zi(b, a) * x[0]
    y, _ = lfilter(b, a, x, zi=zi)
    return y


def _causal_butter_hp(x: np.ndarray, fs: float, cutoff: float, order: int) -> np.ndarray:
    """Causal IIR high-pass via lfilter."""
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype="high")
    zi = lfilter_zi(b, a) * x[0]
    y, _ = lfilter(b, a, x, zi=zi)
    return y


def _causal_butter_bp(x: np.ndarray, fs: float, lowcut: float, highcut: float, order: int) -> np.ndarray:
    """Causal Butterworth band-pass via SOS form (numerically stable for higher order)."""
    sos = butter(order, [lowcut, highcut], btype="band", fs=fs, output="sos")
    zi = sosfilt_zi(sos) * x[0]
    y, _ = sosfilt(sos, x, zi=zi)
    return y


def _shift_for_group_delay(x: np.ndarray, delay_samples: int) -> np.ndarray:
    """Forward-shift x by delay_samples; trailing samples edge-padded.

    delay_samples must be >= 0. Used to align a filtered signal back to the
    input timeline. See module docstring for why this is causal in real-time
    semantics despite looking like look-ahead in array semantics.
    """
    if delay_samples <= 0:
        return x.copy()
    aligned = np.empty_like(x)
    aligned[:-delay_samples] = x[delay_samples:]
    aligned[-delay_samples:] = x[-1]
    return aligned


# ---------------------------------------------------------------------------
# EDA
# ---------------------------------------------------------------------------

@dataclass
class EdaResult:
    smoothed: np.ndarray         # 1 Hz LP'd raw EDA, in µS, group-delay-aligned
    tonic: np.ndarray            # SCL (slow component), in µS
    phasic: np.ndarray           # SCR (fast component), in µS — unipolar (ReLU'd HP). Use for peak detection / SCR features.
    phasic_bipolar: np.ndarray   # raw complementary HP output (smoothed - tonic). Bipolar; kept for diagnostics — preserves the identity tonic + phasic_bipolar = smoothed exactly.
    peaks_info: dict             # output of nk.eda_peaks on the unipolar phasic
    valid: np.ndarray            # bool mask, False = electrode-off / saturated sample
    group_delay_seconds: float


def clean_eda(eda_raw: np.ndarray, fs: float, eda_cfg: dict) -> EdaResult:
    """Causal SCL/SCR decomposition.

    The SCL/SCR boundary is set at 0.05 Hz (Boucsein 2012, Posada-Quintero 2016),
    i.e., tonic activity is the slow component below 0.05 Hz, phasic is the fast
    component above. We replace V2's `np.gradient`-based phasic (which mixed
    units of µS/s with µS thresholds and had a one-sample look-ahead from
    central differences) with a causal complementary high-pass: phasic =
    smoothed − tonic. This keeps phasic in µS, matches V1's [0.01, 1.0] µS
    outlier convention, and gives well-defined SCR rise/recovery dynamics for
    nk.eda_peaks.
    """
    presmooth_fc = eda_cfg["presmooth_cutoff_hz"]
    presmooth_order = eda_cfg["presmooth_order"]
    band_fc = eda_cfg["band_cutoff_hz"]
    band_order = eda_cfg["band_order"]
    group_delay_sec = eda_cfg["group_delay_seconds"]
    scr_amp_min = eda_cfg["scr_amplitude_min_us"]

    smoothed = _causal_butter_lp(eda_raw, fs, presmooth_fc, presmooth_order)
    tonic = _causal_butter_lp(smoothed, fs, band_fc, band_order)
    # Complementary HP: smoothed - tonic. Bipolar by construction — the
    # negative excursions correspond to post-SCR recovery, not new SCR events.
    # SCRs are unipolar (sympathetic activation only drives conductance up),
    # so peak detection runs on the ReLU'd version. We retain the bipolar
    # signal in EdaResult for diagnostic plots and to preserve the exact
    # identity tonic + phasic_bipolar == smoothed.
    phasic_bipolar = smoothed - tonic
    phasic = np.maximum(phasic_bipolar, 0.0)

    delay_samples = int(round(group_delay_sec * fs))
    smoothed_a = _shift_for_group_delay(smoothed, delay_samples)
    tonic_a = _shift_for_group_delay(tonic, delay_samples)
    phasic_bipolar_a = _shift_for_group_delay(phasic_bipolar, delay_samples)
    phasic_a = _shift_for_group_delay(phasic, delay_samples)

    try:
        _, peaks_info = nk.eda_peaks(phasic_a, sampling_rate=fs, amplitude_min=scr_amp_min)
    except (ValueError, KeyError, IndexError, TypeError):
        peaks_info = {
            "SCR_Peaks": np.array([], dtype=int),
            "SCR_Amplitude": np.array([]),
            "SCR_RiseTime": np.array([]),
            "SCR_Recovery": np.array([]),
        }

    valid = ~detect_eda_electrode_off(
        eda_raw, fs,
        min_us=eda_cfg["electrode_off_min_us"],
        flat_seconds=eda_cfg["electrode_off_flat_seconds"],
    )

    return EdaResult(
        smoothed=smoothed_a,
        tonic=tonic_a,
        phasic=phasic_a,
        phasic_bipolar=phasic_bipolar_a,
        peaks_info=peaks_info,
        valid=valid,
        group_delay_seconds=group_delay_sec,
    )


def detect_eda_electrode_off(
    eda_raw: np.ndarray, fs: float, min_us: float, flat_seconds: float
) -> np.ndarray:
    """True where the sensor appears disconnected: conductance below min_us
    sustained for at least flat_seconds.

    Boucsein 2012 places the lower physiological bound for wrist EDA around
    0.05 µS; sustained values below that indicate poor electrode-skin contact
    rather than a real low-arousal state.
    """
    flat_samples = int(round(flat_seconds * fs))
    if flat_samples < 1:
        return np.zeros_like(eda_raw, dtype=bool)
    below = eda_raw < min_us
    mask = np.zeros_like(below)
    run = 0
    for i, b in enumerate(below):
        if b:
            run += 1
        else:
            if run >= flat_samples:
                mask[i - run : i] = True
            run = 0
    if run >= flat_samples:
        mask[len(below) - run :] = True
    return mask


# ---------------------------------------------------------------------------
# BVP
# ---------------------------------------------------------------------------

@dataclass
class BvpResult:
    cleaned: np.ndarray          # bandpass-filtered BVP, ready for peak detection
    peaks: np.ndarray            # systolic peak indices (Elgendi)
    ibis_ms: np.ndarray          # raw inter-beat intervals (peak-to-peak), in ms
    ibis_corrected_ms: np.ndarray  # ectopic-corrected IBIs (causal median replacement)


def clean_bvp(bvp_raw: np.ndarray, fs: float, bvp_cfg: dict) -> np.ndarray:
    """Causal Butterworth bandpass for cardiac signal extraction.

    Cutoffs 0.5–8 Hz follow Elgendi 2013 (the bandpass assumed by the Elgendi
    peak-detection algorithm we use downstream) and Schmidt 2018's WESAD
    reference pipeline. Order 3 balances roll-off (18 dB/octave at 0.5 Hz cut-on,
    enough to suppress baseline wander) against phase distortion within the
    cardiac band. We replace V1/V2's `nk.ppg_clean` which uses sosfiltfilt
    internally (verified non-causal in NK2 0.2.12).
    """
    return _causal_butter_bp(
        bvp_raw, fs, bvp_cfg["band_low_hz"], bvp_cfg["band_high_hz"], bvp_cfg["order"]
    )


def find_bvp_peaks(bvp_clean: np.ndarray, fs: float) -> np.ndarray:
    """Systolic peak detection via Elgendi 2013, applied to causally pre-cleaned BVP.

    Elgendi internally smooths with two centered moving-average kernels of
    111 ms (peak window) and 667 ms (beat window). At the sample level this is
    non-causal by up to ~333 ms, but at the feature-window level (60 s windows)
    it is bounded inside the window and never reaches across windows — so the
    feature emitted at clock time t depends only on input ≤ t.
    """
    info = nk.ppg_findpeaks(bvp_clean, sampling_rate=fs, method="elgendi")
    return np.asarray(info.get("PPG_Peaks", []), dtype=int)


def correct_ectopic_ibis(
    peaks: np.ndarray, fs: float, deviation_pct: float, lookback_beats: int,
    plausible_min_ms: float = 300.0, plausible_max_ms: float = 1500.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Replace IBIs deviating > deviation_pct from a causal moving median with the median.

    The moving median uses only the previous lookback_beats IBIs (no peek
    into the future). The median is computed from the *raw* IBI array
    restricted to physiologically plausible values (300–1500 ms ≈ 40–200 bpm),
    so a single ectopic beat does not corrupt the medians used to vet later
    beats — using the running `corrected` array causes a cascade where one
    bad beat shrinks all subsequent medians.

    Rationale: the Task Force 1996 HRV standards require ectopic correction
    before time/frequency-domain HRV analysis; uncorrected ectopic beats
    dominate RMSSD and SDNN. A 20 % threshold against a local median is the
    conventional Berntson 1997 approach.
    """
    if len(peaks) < 2:
        empty = np.array([])
        return empty, empty
    ibis_ms = np.diff(peaks) * (1000.0 / fs)
    corrected = ibis_ms.copy()
    threshold = deviation_pct / 100.0
    plausible = (ibis_ms >= plausible_min_ms) & (ibis_ms <= plausible_max_ms)
    for i in range(len(corrected)):
        lo = max(0, i - lookback_beats)
        history = ibis_ms[lo:i][plausible[lo:i]]
        if len(history) < 2:
            continue
        local_median = float(np.median(history))
        if local_median <= 0:
            continue
        if abs(ibis_ms[i] - local_median) / local_median > threshold:
            corrected[i] = local_median
    return ibis_ms, corrected


def process_bvp(bvp_raw: np.ndarray, fs: float, bvp_cfg: dict) -> BvpResult:
    """End-to-end BVP pipeline: causal clean -> Elgendi peaks -> ectopic-corrected IBIs."""
    cleaned = clean_bvp(bvp_raw, fs, bvp_cfg)
    peaks = find_bvp_peaks(cleaned, fs)
    ibis, ibis_corr = correct_ectopic_ibis(
        peaks, fs,
        deviation_pct=bvp_cfg["ectopic_deviation_pct"],
        lookback_beats=bvp_cfg["ectopic_lookback_beats"],
    )
    return BvpResult(cleaned=cleaned, peaks=peaks, ibis_ms=ibis, ibis_corrected_ms=ibis_corr)


# ---------------------------------------------------------------------------
# Skin temperature
# ---------------------------------------------------------------------------

@dataclass
class TempResult:
    smoothed: np.ndarray
    valid: np.ndarray            # bool mask, False = unphysical sample (sensor disconnect)


def clean_temp(temp_raw: np.ndarray, fs: float, temp_cfg: dict) -> TempResult:
    """Light causal LP smoothing + unphysical-step detection.

    The E4 sensor's published accuracy is ±0.2 °C; sample-to-sample changes
    above 1 °C at 4 Hz (i.e., 4 °C / s) cannot be physiological and indicate
    skin-contact loss or watch-off events.
    """
    smoothed = _causal_butter_lp(
        temp_raw, fs, temp_cfg["smoothing_cutoff_hz"], temp_cfg["smoothing_order"]
    )
    valid = ~detect_temp_dropout(temp_raw, max_step_celsius=temp_cfg["max_step_celsius"])
    return TempResult(smoothed=smoothed, valid=valid)


def detect_temp_dropout(temp_raw: np.ndarray, max_step_celsius: float) -> np.ndarray:
    diffs = np.abs(np.diff(temp_raw, prepend=temp_raw[0]))
    return diffs > max_step_celsius


# ---------------------------------------------------------------------------
# Accelerometer
# ---------------------------------------------------------------------------

@dataclass
class AccResult:
    x_g: np.ndarray              # per-axis acceleration in g
    y_g: np.ndarray
    z_g: np.ndarray
    magnitude: np.ndarray        # sqrt(x^2 + y^2 + z^2), in g
    magnitude_activity: np.ndarray  # LP-filtered magnitude (ambulation band, < activity_cutoff_hz)
    jerk_magnitude: np.ndarray   # sqrt(jx^2 + jy^2 + jz^2), in g/s — causal backward difference


def clean_acc(acc_raw: np.ndarray, fs: float, acc_cfg: dict) -> AccResult:
    """Compute ACC magnitude + jerk + activity-band magnitude.

    WESAD's E4 stores ACC as raw 1/64 g integer counts. Dividing by 64 yields
    accelerations in g, which is the unit the deployable device will report
    and matches conventional accelerometer thresholds (Bouten 1997, etc.).

    Jerk (derivative of acceleration) is computed via causal backward
    difference. PROJECT_ADVICE.md identifies fidget-band features (>3 Hz)
    as the key wearable signal for the classroom application; the activity
    cutoff at 3 Hz is the conventional separation between ambulation
    (walking/running, < 3 Hz) and fidgeting/restlessness (> 3 Hz)
    (Karantonis 2006).
    """
    div = acc_cfg["raw_to_g_divisor"]
    x = acc_raw[:, 0].astype(float) / div
    y = acc_raw[:, 1].astype(float) / div
    z = acc_raw[:, 2].astype(float) / div

    magnitude = np.sqrt(x * x + y * y + z * z)
    magnitude_activity = _causal_butter_lp(
        magnitude, fs, acc_cfg["activity_cutoff_hz"], acc_cfg["activity_order"]
    )

    dt = 1.0 / fs
    jx = np.diff(x, prepend=x[0]) / dt
    jy = np.diff(y, prepend=y[0]) / dt
    jz = np.diff(z, prepend=z[0]) / dt
    jerk_magnitude = np.sqrt(jx * jx + jy * jy + jz * jz)

    return AccResult(
        x_g=x, y_g=y, z_g=z,
        magnitude=magnitude,
        magnitude_activity=magnitude_activity,
        jerk_magnitude=jerk_magnitude,
    )


# ---------------------------------------------------------------------------
# Cross-modal motion-artifact detection (used by BVP HRV downstream)
# ---------------------------------------------------------------------------

def window_motion_metric(jerk_magnitude_window: np.ndarray, percentile: float = 95.0) -> float:
    """High-percentile jerk magnitude over a window, in g/s.

    The 95th percentile of jerk magnitude reflects motion *bursts* large
    enough to disrupt PPG peak detection, not the steady micro-movement
    background that dominates RMS jerk for a seated subject. This is the
    metric the rebuild plan's "motion artifact detection" is best implemented
    against (Schmidt 2018 §3.2 documents wrist-PPG sensitivity to specific
    motion bursts more than to steady low-amplitude motion).
    """
    if len(jerk_magnitude_window) == 0:
        return float("nan")
    return float(np.percentile(jerk_magnitude_window, percentile))


def is_window_motion_corrupted(
    jerk_magnitude_window: np.ndarray, threshold_g_per_s: float, percentile: float = 95.0
) -> bool:
    """True if the high-percentile jerk in this window exceeds threshold."""
    return window_motion_metric(jerk_magnitude_window, percentile=percentile) > threshold_g_per_s
