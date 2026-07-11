# Samsung Galaxy Watch 8 — Preprocessing Plan

Adaptation of the WESAD V3 pipeline to Samsung Watch data.
Reference implementation: `Updated_Extraction_V3/preprocessing.py` and `Updated_Extraction_V3/features.py`.

**Retraining intent:** The model will be retrained on Samsung Watch data. The existing WESAD-trained model is not applied directly.

---

## Sensor Rates

| Stream | Rate |
|---|---|
| EDA | ~1 Hz |
| PPG | ~25 Hz (per-sample timestamps available) |
| HR / IBI | ~1 Hz (IBI list in `v2` column of HR rows) |
| SKIN_TEMP | ~0.1 Hz |
| ACCEL | ~25 Hz (per-sample timestamps available) |

---

## General Decisions

- **Warm-up discard:** Drop the first 3–5 minutes of each session. EDA electrodes showed monotonic drift from 0.118 to 0.160 µS over 5 minutes with no plateau in initial testing — discard until stabilized.
- **Normalization:** Subject-specific z-score using the baseline period only. Same approach as V3 (`preprocessing.py` applies no global normalization; normalization is done at the dataset-builder level using baseline windows).
- **Windowing:** 60 s non-overlapping, same as V3.
- **Labels:** Must be manually annotated per session. No automated WESAD-style label files exist for Samsung sessions.

---

## EDA (1 Hz)

V3 reference: `clean_eda()` in `preprocessing.py` lines 94–154; `extract_eda_features()` in `features.py` lines 41–102.

### Preprocessing changes from V3

| V3 step | Samsung adaptation |
|---|---|
| Presmooth: order-1 causal LP at 1.0 Hz (`presmooth_cutoff_hz`) | **Skip.** Nyquist at 1 Hz sampling is 0.5 Hz — a 1.0 Hz cutoff is above Nyquist, making this filter meaningless. |
| Tonic (SCL): causal Butterworth LP at 0.05 Hz | **Keep, with order reduced to 1.** Normalized cutoff = 0.05 / 0.5 = 0.1. Low order required for numerical stability at 1 Hz sampling. |
| Phasic (SCR): complementary HP (smoothed − tonic), ReLU ≥ 0 | **Keep unchanged.** |
| Peak detection: `nk.eda_peaks()` | **Replace.** `nk.eda_peaks()` expects higher sampling rates. Use `scipy.signal.find_peaks()` with minimum spacing of 3 samples (3 s) and an amplitude threshold suited to the Samsung scale. |

### Features kept vs. dropped

**Keep (11):**
`scl_mean`, `scl_std`, `scl_min`, `scl_max`, `scl_range`, `scl_median`, `scl_slope`, `scl_auc`,
`scr_peak_count`, `scr_amplitude_mean`, `scr_amplitude_max`, `scr_amplitude_sum`

**Drop (2):**
`scr_rise_time_mean`, `scr_recovery_time_mean` — require sub-second resolution; unreliable at 1 Hz.

**Status of distribution features (`eda_skewness`, `eda_kurtosis`):** Not mentioned in the agreed plan — defer decision.

### Signal quality note

Samsung Watch EDA is clean: typical sample-to-sample change ~0.001 µS, max ~0.005 µS. No heavy denoising needed. Absolute values are much lower than WESAD E4 (0.1–0.16 µS vs. 1–20 µS) due to different electrode geometry. Subject-specific z-score normalization handles the scale difference.

---

## HRV (from HR / IBI rows, ~1 Hz)

V3 reference: `correct_ectopic_ibis()` in `preprocessing.py` lines 225–260; `extract_hrv_features()` in `features.py` lines 120–207.

### Source change

V3 derives IBIs from PPG peak detection via `nk.ppg_process()` on raw BVP at 64 Hz. That path is not used here.

**Samsung source:** Parse the `ibi_list` column from HR rows (values in ms). This is the pre-computed IBI stream from the watch SDK.

### Ectopic correction

Apply the same causal median filter as V3 (`correct_ectopic_ibis()`): 20% deviation threshold, 4-beat lookback, plausible range 300–1500 ms.

### Features

**Time-domain (computable per 60 s window):**
`hrv_rmssd`, `hrv_sdnn`, `hrv_pnn50`, `hrv_mean_hr`, `hrv_mean_rr`, `hrv_median_rr`, `hrv_min_rr`, `hrv_max_rr`, `hrv_sdsd`

**Frequency-domain (`hrv_lf_power`, `hrv_hf_power`, `hrv_lf_hf_ratio`, `hrv_total_power`):**
Include but flag as lower confidence. At ~1 Hz IBI sampling the spectral estimates will be noisy. V3 already requires ≥ 120 s window and ≥ 30 peaks before computing frequency features (`features.py` lines 192–205) — apply the same guard here.

### PPG cross-check

PPG is available at 25 Hz with per-sample timestamps. Peak detection on PPG can be attempted as a cross-check against the HR/IBI stream. The IBI list from HR rows remains the primary HRV source.

---

## Temperature (0.1 Hz)

V3 reference: `clean_temp()` in `preprocessing.py` lines 285–296; `extract_temp_features()` in `features.py` lines 225–239.

### Changes from V3

| V3 step | Samsung adaptation |
|---|---|
| Causal LP smoothing (0.05 Hz cutoff configured in `temp_cfg`) | **Skip.** Nyquist at 0.1 Hz is 0.05 Hz — no meaningful passband exists below the cutoff. |
| Unphysical step detection (`detect_temp_dropout`, `max_step_celsius`) | **Skip.** Designed for 4 Hz (E4); not meaningful at 0.1 Hz. |

### Features

Use raw stats only (same names as V3): `temp_mean`, `temp_min`, `temp_max`, `temp_median`, `temp_slope`.
Drop `temp_std` or retain at low confidence — expect only ~6 samples per 60 s window, making std coarse.

---

## Accelerometer (25 Hz)

V3 reference: `clean_acc()` in `preprocessing.py` lines 318–353; `extract_acc_features()` in `features.py` lines 246–279.

### Changes from V3

Per-sample timestamps are now available; use them directly for alignment. No other structural changes needed.

V3 converts E4 raw integer counts to g using `raw_to_g_divisor = 64`. Samsung Watch ACCEL may already report in g or m/s² — confirm units before applying the divisor.

### Features

All 12 V3 features are applicable:

**V1 carry-over (8):** `acc_x_std`, `acc_y_std`, `acc_z_std`, `acc_magnitude_mean`, `acc_magnitude_std`, `acc_magnitude_max`, `acc_sma`, `acc_energy`

**V3 additions (4):** `acc_activity_mean`, `acc_jerk_mag_mean`, `acc_jerk_mag_std`, `acc_jerk_mag_p95`

Jerk computed via causal backward difference (`np.diff`, `prepend=x[0]`) divided by `dt = 1/fs`, same as V3 lines 343–346.

**Note on activity band:** The 3 Hz cutoff for `magnitude_activity` (ambulation band separation) at 25 Hz sampling is borderline (~8 taps per 60 s window). Retain but note lower reliability relative to the E4 at 32 Hz.

---

## What Still Needs Validation

- Whether EDA signal plateaus after longer warm-up (session > 10 min pending).
- ACCEL availability in future sessions — not present in the session analyzed so far.
- Appropriate SCR amplitude threshold at Samsung Watch scale (0.1–0.16 µS range vs. WESAD E4 1–20 µS).
- Minimum session length needed for reliable frequency-domain HRV estimates.
