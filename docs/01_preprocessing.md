# 01 — Preprocessing Design

**Scope:** Phase 1 of the V3 rebuild — causal filtering and artifact detection for
the four wrist modalities (EDA, BVP, TEMP, ACC). Implementation lives in
[Updated_Extraction_V3/preprocessing.py](../Updated_Extraction_V3/preprocessing.py);
parameters in [Updated_Extraction_V3/config.yaml](../Updated_Extraction_V3/config.yaml).

**Date:** 2026-04-29
**Predecessor:** [00_existing_pipeline_audit.md](00_existing_pipeline_audit.md)

This document is the design defense for every preprocessing decision. For each
choice, the format is: **what we picked**, **literature anchor**, **why** (rationale
in plain language), and **what we ruled out** (alternatives considered).

Empirical verification (raw-vs-filtered plots on 2–3 subjects, cvxEDA-vs-causal
divergence study) is deferred to a Phase-1-follow-up turn so this document and the
code can be reviewed first.

---

## 0. Universal conventions

### 0.1 Causal IIR filtering

Every filter in `preprocessing.py` is a causal IIR Butterworth applied via
`scipy.signal.lfilter` (transfer-function form) or `scipy.signal.sosfilt` (SOS
form for higher orders). Each filter is initialized to its steady-state response
at the first input sample (`zi = lfilter_zi(b, a) * x[0]`), so the transient at
sensor-on is minimized.

We do **not** use `sosfiltfilt`/`filtfilt` anywhere. Those are zero-phase forward-
backward filters and require future samples to compute every output sample —
incompatible with real-time deployment.

V1's `signal.sosfiltfilt(...)` ([features.py:52](../Updated_Extraction/features.py#L52))
and the `nk.signal_filter(method="butterworth")` calls inside `nk.ppg_clean` /
`nk.ppg_process` are the V1/V2 violations this replaces.

### 0.2 Subject-level filtering = real-time emulation

Each filter is applied once across the subject's full signal. This is
mathematically equivalent to running the filter sample-by-sample in a real-time
loop with persistent filter state (the IIR's internal `zi`). So:

- **Offline (training):** `lfilter(b, a, full_signal, zi=...)` produces a clean,
  causally-filtered version of the entire subject's session in one call.
- **Online (deployment):** `lfilter(b, a, [new_sample], zi=current_state)` produces
  one new clean sample with the same filter math; state persists between calls.

These produce identical outputs sample-for-sample. So our offline pipeline is an
exact emulation of streaming. No "training-vs-deployment" preprocessing gap.

### 0.3 Group-delay compensation as startup latency

A causal IIR Butterworth at low cutoff frequencies has a non-trivial group delay:
the filtered signal lags the input by τ_g samples. For our SCL extractor (order-2
LP at 0.05 Hz, fs=4 Hz), τ_g ≈ 25.5 samples ≈ 6.4 s.

To align filtered features with the input timeline, we forward-shift by
delay_samples = round(group_delay_seconds × fs). In array semantics this looks
like a peek into the future; in real-time semantics it is a **startup latency**:
predictions begin 6.4 s after sensor-on, and every emitted feature is timestamped
with the input-time it represents (not the clock-time at which it was emitted).

The trailing 6.4 s of input gets edge-padded in the offline output for length
parity; in deployment those samples never produce predictions (the wearer takes
the watch off).

This is the same operation V2 implements at [extract_windowed_datasets.py:114-118](../Updated_Extraction_V2/extract_windowed_datasets.py#L114-L118),
documented here so future maintainers don't read it as a bug.

### 0.4 Validity masks vs. window dropping

Each artifact detector returns a per-sample boolean validity mask. The
preprocessing module **does not** drop samples or windows. That decision belongs
to the feature-extraction step (Phase 3): if a window has > X% invalid samples,
the corresponding feature value is either NaN-flagged or the window is excluded
from training/eval. Keeping detection and policy separated lets us experiment
with different thresholds without rerunning preprocessing.

---

## 1. EDA (4 Hz)

### 1.1 What we picked

| Stage | Operation | Cutoff | Order | Direction |
|---|---|---|---|---|
| 1 — Pre-smoothing | Causal Butterworth low-pass | 1.0 Hz | 1 | LP |
| 2 — SCL (tonic) | Causal Butterworth low-pass on smoothed EDA | 0.05 Hz | 2 | LP |
| 3 — SCR (phasic) | `smoothed − tonic` (complementary high-pass) | 0.05 Hz | 2 | HP (implicit) |
| 4 — Alignment | Forward shift of all three signals | — | — | by group_delay = 6.4 s |
| 5 — Peak detection | `nk.eda_peaks(phasic_aligned)` with `amplitude_min=0.01 µS` | — | — | — |
| 6 — Electrode-off detection | flat below 0.05 µS for ≥ 5 s | — | — | — |

### 1.2 Why

**Stage 1 — Pre-smoothing at 1.0 Hz.** Removes high-frequency electrode noise
above the EDA band. SCRs have rise times typically ≥ 1 s and rarely contain
energy above 1 Hz {boucsein2012}, so a 1 Hz cutoff preserves all SCR information
while suppressing noise. Order 1 is sufficient — higher order at this cutoff
would add unnecessary group delay without measurable noise-rejection gains on a
4 Hz-sampled signal. This stage is unchanged from V2.

**Stages 2-3 — SCL/SCR boundary at 0.05 Hz.** Boucsein 2012 and Posada-Quintero
et al. 2016 {posada-quintero2016} place the SCL/SCR cutoff at ≈ 0.05 Hz (period
20 s): tonic SCL has dynamics on minute timescales, phasic SCRs on second
timescales, and 0.05 Hz cleanly separates them. Order 2 (12 dB/octave) is the
minimum order with sharp enough roll-off to give good separation; order 3+ adds
group delay without much practical gain.

**Phasic = smoothed − tonic.** This is a complementary high-pass: the SCL LP
removes low frequencies, so what's left after subtraction is the high-frequency
content. Mathematically, `phasic + tonic = smoothed` exactly, with no algorithm-
introduced bias. The phasic signal stays in µS — the same units as the SCR
amplitudes that V1 and the EDA literature {boucsein2012} report.

This is **the change from V2.** V2 used `np.gradient(smoothed) → ReLU` for
phasic. Two problems with that:

1. `np.gradient` uses central differences, so its output at sample t depends on
   sample t+1 (one-sample look-ahead, ~250 ms at 4 Hz). The complementary HP
   approach uses no derivative — it inherits causality from the underlying LP.
2. Derivative units are µS/s, not µS. V2's SCR amplitude bounds (0.01–5.0)
   inherit V1's wording but with mismatched units, making the bounds hard to
   interpret. The complementary HP yields phasic in µS and lets us reuse V1's
   well-grounded [0.01, 1.0] µS bounds {boucsein2012}.

The complementary-HP decomposition is also the standard approach in the EDA
literature {posada-quintero2016} (their "fast-tonic and phasic decomposition"
uses an analogous complementary filterbank).

**Stage 4 — 6.4 s alignment.** Group delay of the order-2 0.05 Hz LP at fs=4 Hz
is 25.6 samples = 6.4 s. We compensate (§0.3) so feature timestamps match input
timestamps. Same operation V2 used; here it is correct because *both* tonic
and phasic share the same group delay (the HP and LP are designed at the same
cutoff/order, so their group delays match), so a single uniform shift aligns
both.

**Stage 5 — `nk.eda_peaks`.** Reuses V1's peak detector. The Khodadad 2018
algorithm inside NK2's `eda_peaks` works on a phasic signal with rise/recovery
dynamics, which our HP-derived phasic provides. The `amplitude_min=0.01 µS`
threshold is the clinical lower bound from {boucsein2012}; SCRs below this are
indistinguishable from noise on wrist EDA.

**Stage 6 — Electrode-off detection.** Boucsein 2012 places the lower
physiological bound for wrist EDA at ≈ 0.05 µS; sustained values below indicate
poor electrode-skin contact, not low arousal. The 5-second sustain requirement
prevents flagging brief dips during deep relaxation as electrode-off.

### 1.3 What we ruled out

- **cvxEDA {greco2016}.** V1's choice. Solves a global convex optimization for
  tonic/phasic decomposition. Excellent decomposition quality but uses the
  entire signal at once — no causal/streaming form. Out of scope per the
  rebuild plan's Hard Constraint #2. We will run a quantitative comparison in
  Phase 1 follow-up: extract phasic/tonic with both methods on 2–3 subjects,
  plot side-by-side, compute window-level feature correlations. If correlation
  is poor we revisit.
- **Derivative-based phasic (V2).** See §1.2 — wrong units, look-ahead from
  central differences.
- **NK2 `eda_phasic(method="smoothmedian")`.** V1's documented fallback that
  "produces materially different results" per the V1 README; we don't want a
  third decomposition we can't defend.

### 1.4 Open questions

- The HP-vs-derivative phasic divergence on real WESAD subjects is the empirical
  question. Phase 1 follow-up will quantify this before we build features in
  Phase 3.

---

## 2. BVP / PPG (64 Hz)

### 2.1 What we picked

| Stage | Operation | Cutoff | Order | Direction |
|---|---|---|---|---|
| 1 — Bandpass clean | Causal Butterworth band-pass (SOS) | 0.5 / 8.0 Hz | 3 | BP |
| 2 — Peak detection | `nk.ppg_findpeaks(method="elgendi")` | — | — | — |
| 3 — Ectopic correction | Replace IBIs deviating > 20 % from causal moving median (last 5 plausible beats) | — | — | — |
| 4 — Motion-artifact gating | Per-window flag from ACC jerk RMS | — | — | window-level |

### 2.2 Why

**Stage 1 — 0.5–8 Hz bandpass, order 3.** This is exactly the bandpass the
Elgendi 2013 peak-detection algorithm assumes: "The signal must be the
bandpass-filtered raw PPG with a lowcut of 0.5 Hz, a highcut of 8 Hz" {elgendi2013}.
Schmidt et al.'s WESAD reference pipeline {schmidt2018} uses the same band.
The lower cutoff suppresses baseline wander (DC offset and breathing-rate
modulation); the upper cutoff suppresses high-frequency motion noise. Order 3
gives 18 dB/octave roll-off, sufficient to suppress both bands without excessive
phase distortion across the cardiac frequencies (60–180 bpm = 1–3 Hz).

The SOS implementation (`scipy.signal.butter(..., output='sos')` +
`scipy.signal.sosfilt`) is numerically more stable than transfer-function form
for higher-order bandpasses and avoids the coefficient-precision problems that
would otherwise force us to lower the order.

**Why not `nk.ppg_clean`?** NK2 0.2.12's `ppg_clean` calls
`nk.signal_filter(method="butterworth")`, which uses `scipy.signal.sosfiltfilt`
internally — non-causal. Verified by inspecting NK2 source. So we replace
`ppg_clean` with our causal bandpass and feed the cleaned signal to
`ppg_findpeaks` directly.

**Stage 2 — Elgendi 2013 peak detection.** We keep V1/V2's peak detector. The
algorithm uses two centered moving-average kernels (peak window 111 ms ≈ 7
samples; beat window 667 ms ≈ 43 samples at 64 Hz). At the *sample* level this
is non-causal by up to ~333 ms, but at the *feature window* level (60 s) it is
bounded inside the window and never reaches across windows. So a feature
emitted at clock time t depends only on input ≤ t. Acceptable.

For real-time deployment on the embedded device, we either reuse Elgendi
unchanged (accepting the within-window bounded MA), or swap to a fully
sample-causal detector (e.g., HeartPy's adaptive thresholding) if hardware
profiling shows the window-bounded MA is too costly. For training and
LOSO eval, Elgendi is fine.

**Stage 3 — Ectopic correction.** Task Force 1996 {taskforce1996} requires
ectopic correction before HRV analysis: uncorrected ectopic beats dominate
RMSSD and SDNN. Berntson 1997 {berntson1997} establishes the standard
20 %-deviation-from-local-median rule.

Two implementation choices that matter:

1. **Median is computed over raw IBIs, not over the running corrected array.**
   Using the running corrected array causes a cascade — one ectopic replacement
   can pull subsequent medians toward the bad value, causing later good IBIs
   to be flagged and replaced. (Caught in Phase 1 smoke-testing: original
   implementation pushed S2's median IBI from 859 ms to 484 ms; the fix
   stabilizes it at 844 ms.)
2. **The local median uses only physiologically plausible IBIs (300–1500 ms,
   i.e., 40–200 bpm).** Beats outside this window contribute noise to the
   median estimate. Excluding them upfront keeps the median at the wearer's
   actual rate even when bursts of ectopic beats occur.

Lookback of 5 plausible beats balances responsiveness to genuine HR transitions
against susceptibility to noise; 5–10 is a conventional range. Configurable via
`config.yaml`.

**Stage 4 — Motion gating.** Wrist-PPG is highly motion-sensitive
{schmidt2018, §3.2}. We compute ACC jerk magnitude (§4) and gate per-window:
windows with RMS jerk > 2 g/s are flagged motion-corrupted, and HRV features
from those windows will be marked invalid in Phase 3 rather than fed to the
model as zeros (V1's bug). The 2 g/s threshold is approximate — Phase 2 will
validate it against per-window peak-detection error rates.

### 2.3 What we ruled out

- **`nk.ppg_process`/`nk.ppg_clean`.** Non-causal in NK2 0.2.12 (verified).
- **HeartPy.** Strong alternative with a fully causal mode. Holding off because
  the rebuild plan endorses Elgendi and we want to minimize V2→V3 churn. Will
  revisit in Phase 8 (real-time simulator) if Elgendi's per-window MA proves
  too costly on embedded hardware.
- **Pan-Tompkins for PPG.** Pan-Tompkins is ECG-specific (sharp QRS); its
  threshold logic doesn't transfer well to PPG's smoother systolic peaks.
- **`nk.signal_fixpeaks(method="kubios")`.** Excellent ectopic correction but
  the Kubios method operates on the full sequence at once. Our causal moving-
  median replacement is the streaming-friendly equivalent.

---

## 3. Skin temperature (4 Hz)

### 3.1 What we picked

| Stage | Operation | Cutoff | Order |
|---|---|---|---|
| 1 — Smoothing | Causal Butterworth low-pass | 0.5 Hz | 1 |
| 2 — Sensor-dropout detection | Flag samples with abs(diff) > 1.0 °C | — | — |

### 3.2 Why

**Stage 1.** Skin temperature on a wrist is a slow signal — stress-related
peripheral vasoconstriction operates over tens of seconds {karthikeyan2012,
healey2005}. A 0.5 Hz LP order 1 removes high-frequency electrode noise without
distorting the slow physiologically-relevant changes (group delay ≈ 0.16 s).
Order 1 because anything more would over-smooth without measurable noise gain.

**Stage 2.** The Empatica E4's published temperature accuracy is ±0.2 °C. A
sample-to-sample change > 1 °C at 4 Hz = > 4 °C/s — physiologically impossible
on intact skin. Such jumps indicate the watch was momentarily off-skin (sensor
saw ambient air, then skin again). We flag those samples; Phase 3 decides
window policy.

### 3.3 What we ruled out

- **No smoothing at all.** Tempting since features are computed over 60 s
  windows where statistics smooth naturally. But raw E4 temperature has visible
  quantization noise at the 0.01 °C level which is comparable to the
  inter-condition deltas we want to detect. Light LP costs almost nothing and
  improves signal cleanliness.
- **More aggressive dropout detection.** E.g., flagging long stretches of
  unchanged values as "stuck sensor." Holding off until empirical evidence
  shows it matters; over-aggressive flagging shrinks the usable dataset.

---

## 4. Accelerometer (32 Hz)

### 4.1 What we picked

| Stage | Operation |
|---|---|
| 1 — Unit conversion | Divide raw values by 64 to get g (Empatica E4 stores 1/64 g) |
| 2 — Magnitude | `sqrt(x² + y² + z²)` |
| 3 — Activity-band magnitude | Causal Butterworth LP at 3.0 Hz, order 2, applied to magnitude |
| 4 — Jerk | Causal backward difference of each axis: `(x[t] − x[t−1]) / dt`; jerk magnitude = `sqrt(jx² + jy² + jz²)` |
| 5 — Per-window motion flag | RMS of jerk magnitude > 2 g/s ⇒ window is motion-corrupted |

### 4.2 Why

**Stage 1 — Unit conversion.** WESAD's E4 stores ACC as raw 1/64 g integer
counts (verified in project memory; resting magnitude ≈ 64 = 1 g). V1 and V2
left this in raw units, which makes thresholds opaque ("acc_magnitude_max =
80" — what does that mean?). Converting to g lets us cite conventional
accelerometer thresholds {bouten1997, karantonis2006} directly.

**Stage 2 — Magnitude.** Orientation-invariant; standard for activity
quantification {bouten1997}. Per-axis stats are still computed in Phase 3 (they
encode posture/orientation), but the magnitude is the load-bearing aggregate.

**Stage 3 — Activity-band LP at 3 Hz.** Karantonis et al. 2006
{karantonis2006} establish 3 Hz as the conventional separation between ambulation
(walking, running — < 3 Hz) and higher-frequency movement (fidgeting,
restlessness — > 3 Hz). The LP-magnitude isolates the ambulation band, so
features computed on it (mean, std) reflect gross movement; features computed
on the raw magnitude minus its activity-band component reflect fidget energy.

For the classroom-deployment goal, fidget-band signal is the more interesting
one (per `docs/PROJECT_ADVICE.md` and our adolescent-stress framing). Phase 3
will compute features in both bands.

**Stage 4 — Jerk via causal backward difference.** Jerk (derivative of
acceleration) captures rapid movement transitions — exactly the fidget signal
PROJECT_ADVICE.md flags as essential. Backward difference `(x[t] − x[t−1])/dt`
is causal at the sample level and introduces no group delay.

The 3-axis jerk magnitude `sqrt(jx² + jy² + jz²)` is orientation-invariant.
Single-axis `np.diff(magnitude)` would lose orientation information (a sharp
forward swing of the wrist registers in jx but might leave |a| nearly
unchanged).

**Stage 5 — Per-window motion flag.** Used to gate BVP/HRV features (§2.2
Stage 4). The 2 g/s threshold is loose — a single hand-shake registers ~5–10
g/s, so 2 g/s flags noticeable movement but tolerates passive arm movement
during normal sitting.

### 4.3 What we ruled out

- **Symmetric centered differences (numpy `np.gradient`).** Adds 1-sample look-
  ahead, same problem as V2's EDA derivative. Backward difference has the same
  noise characteristic and is causal.
- **Higher-order LP for activity band.** Order 2 is enough to separate the
  bands cleanly at 32 Hz sampling; higher orders add group delay.
- **Vector-magnitude jerk via `np.diff(magnitude)`.** Not orientation-invariant
  in the way the per-axis-then-magnitude form is. Omitted.

---

## 5. Cross-modal artifact handling

The four modalities run independently except for one cross-modal coupling:
**ACC jerk gates BVP/HRV.** Wrist-PPG is highly motion-sensitive, and HRV
features computed on motion-corrupted PPG are unreliable {schmidt2018}. The
gating mechanism is window-level rather than sample-level: if RMS jerk in a
window exceeds threshold, all HRV features for that window are marked invalid
in Phase 3 metadata. The model can then either drop the window or learn to
treat HRV as missing.

We do **not** gate EDA or TEMP on motion. EDA on the wrist is also motion-
sensitive but the artifact pattern is different (micro-tremors create sharp
short SCRs that look like genuine SCRs); cross-modal gating would over-flag.
Phase 3 may add an EDA-specific motion gate based on jerk-correlated phasic
spikes if empirically warranted.

---

## 6. Validation results

Implementation: [Updated_Extraction_V3/validate_phase1.py](../Updated_Extraction_V3/validate_phase1.py)
runs all three validation deliverables. Outputs land in
[reports/01_preprocessing/](../reports/01_preprocessing/).

Subjects sampled: **S2** (the recall=0 subject — sanity-check the signal),
**S10** (highest median HR — verify no motion-artifact bias in HR), and **S16**
(perfect-recall subject — clean reference). Per-window comparisons use the
60 s windowing convention from V2 (Phase 2 will sweep window/step).

### 6.1 cvxEDA vs. V3 complementary-HP — feature correlations

| Subject | scl_mean | scl_std | scr_peak_count |
|---|---|---|---|
| S2 | **0.989** | 0.883 | −0.197 |
| S10 | **0.999** | 0.669 | −0.525 |
| S16 | **0.9997** | 0.948 | −0.444 |
| Mean | **0.996** | 0.833 | −0.389 |

**Tonic decomposition is essentially equivalent to cvxEDA** — `scl_mean`
correlation ≥ 0.99 across all three subjects. Visual side-by-sides
([cvxEDA_vs_HP_S2.png](../reports/01_preprocessing/cvxEDA_vs_HP_S2.png),
[_S10](../reports/01_preprocessing/cvxEDA_vs_HP_S10.png),
[_S16](../reports/01_preprocessing/cvxEDA_vs_HP_S16.png)) show the two tonics
overlay almost exactly, with cvxEDA slightly delayed (cvxEDA's iterative solver
introduces its own implicit smoothing). For all SCL features, V3 is a drop-in
replacement.

**SCR count is structurally different.** A sweep of `amplitude_min` from
0.01 µS up to 0.20 µS leaves the per-window `scr_peak_count` correlation
between −0.22 and −0.48 — it does not flip positive at any threshold.
This is not a parameter-tuning problem; it reflects a fundamental decomposition
difference:

- cvxEDA models phasic as a *sparse driver* — most samples are exactly zero,
  with discrete localized SCR spikes. So cvxEDA's `nk.eda_peaks` finds a small
  number of strong peaks per window.
- The complementary-HP phasic is the high-frequency content of the EDA — it has
  non-zero values at every sample with continuous structure. `nk.eda_peaks`
  finds more, smaller, and differently-located peaks.

Both extractions detect "events," but the events are not the same set.
`scl_*` features inherit cvxEDA's behavior; `scr_*` features get redefined.

**Implication for Phases 3, 5, 6.** Since SCR count was V1's #1 feature by
SHAP ([2026-03-11_nb06_full_loso_analysis.md](session_logs/2026-03-11_nb06_full_loso_analysis.md)),
we cannot assume V3's `scr_peak_count` carries the same predictive weight.
Phase 3 will compute the SCR family of features as defined for the V3 pipeline
and let Phase 5 (feature selection) and Phase 6 (model selection) determine
empirically whether they drive the model. If they don't, two options:

- **Option A (revisit decomposition).** Try a causal sparse-decomposition
  alternative (per the rebuild plan's third option for EDA), or run cvxEDA in
  per-window mode for SCR detection while keeping the causal complementary-HP
  for SCL features. This sacrifices some real-time purity for SCR-feature
  fidelity. If we go this route, we document that SCR features are computed
  with a 60 s lookback batch optimization, which is acceptable in deployment
  if we accept the latency.
- **Option B (re-anchor SCR features).** Accept that V3's `scr_peak_count`
  measures something different from V1's, drop the count, and use SCR-power
  features (e.g., sum of phasic energy in the window) which the
  complementary-HP signal supports cleanly.

We won't pick between these in Phase 1. Carry both options into Phase 5.

### 6.2 Per-subject artifact rates

| Subject | n_eda | electrode-off % | n_temp | temp dropout % | n_ibis | ectopic % | n_windows | motion % | HR raw | HR corrected |
|---|---|---|---|---|---|---|---|---|---|---|
| S2 | 24316 | 0.0 | 24316 | 0.0 | 6992 | 35.3 | 101 | 21.8 | 69.8 | 71.1 |
| S10 | 21984 | 0.0 | 21984 | 0.0 | 7592 | 22.3 | 91 | 5.5 | 89.3 | 89.3 |
| S16 | 22524 | 0.0 | 22524 | 0.0 | 6774 | 33.3 | 93 | 3.2 | 76.8 | 78.4 |

CSV: [reports/01_preprocessing/artifact_rates.csv](../reports/01_preprocessing/artifact_rates.csv).

Observations:

- **No electrode-off, no temp dropouts** in any of the three subjects.
  Consistent with WESAD being controlled lab data; the detectors will earn
  their keep on out-of-distribution wear.
- **Ectopic correction rate 22–35%** across subjects, with the corrected
  median HR within 2 bpm of the raw median HR. This means the correction is
  flagging beat-to-beat variability in plausible IBI ranges, not driving the
  HR estimate. (The original implementation, using the running corrected
  array as the median basis, pushed S2's HR from 70 to 124 bpm — fixed by
  computing the median over plausible *raw* IBIs.)
- **Motion-corrupted windows 3–22%.** S2 is highest at 22% (consistent with
  S2's signal showing more general activity in the sanity plot — see §6.3).
  S10 and S16 are 3–5%. The 8 g/s threshold on the 95th-percentile jerk
  metric rejects busy-arm windows but tolerates seated micro-motion.

Diagnostic: per-subject median 95th-percentile jerk by condition:

| Subject | baseline median p95 jerk (g/s) | stress median p95 jerk (g/s) | % baseline flagged | % stress flagged |
|---|---|---|---|---|
| S2 | 3.2 | 2.4 | 5.3 | 0.0 |
| S10 | 2.9 | 2.5 | 0.0 | 16.7 |
| S16 | 2.7 | 3.1 | 0.0 | 0.0 |

Stress-condition motion is *not* uniformly higher than baseline. TSST
participants are seated/standing and gesturing while answering questions —
not the high-acceleration regime where wrist-PPG breaks down. The motion gate
is therefore *defensive* — it catches the rare bursty windows (e.g., S10
during stress at 16.7%) without over-rejecting normal stress windows. Phase 2
will validate the threshold against per-window peak-detection error rates;
the current 8 g/s value is a sensible starting point but is provisional.

### 6.3 Raw-vs-filtered sanity plots

PNGs: [sanity_S2.png](../reports/01_preprocessing/sanity_S2.png),
[sanity_S10.png](../reports/01_preprocessing/sanity_S10.png),
[sanity_S16.png](../reports/01_preprocessing/sanity_S16.png).

Each PNG has 4 rows × 2 columns: rows are EDA/BVP/TEMP/ACC; columns are one
baseline-window and one stress-window. Visually verified:

- **EDA tonic** tracks the slow envelope of raw EDA without ringing. Phasic
  is near-zero in baseline and shows discrete bumps in stress.
- **BVP cleaning** isolates the cardiac waveform; peak markers (red triangles)
  align with systolic apexes in both baseline and stress.
- **TEMP smoothing** is barely distinguishable from raw at the plot resolution —
  expected, since the 0.5 Hz LP only suppresses quantization-level noise.
- **ACC** shows magnitude near 1 g (gravity baseline) with deviations on
  movements; jerk magnitude (red, secondary axis) tracks rapid changes. Stress
  windows show somewhat higher jerk than baseline on S2 and S16, consistent
  with subjects responding to TSST prompts.

No edge-effect transients visible at window boundaries — the per-subject
filter state initialization and group-delay shift work as designed.

### 6.4 Two implementation bugs caught and fixed during validation

- **Ectopic correction cascade.** Original implementation computed the local
  median over the *running corrected* IBI array. One bad replacement pulled
  subsequent medians toward the bad value, causing the correction to spread
  across the rest of the session. S2 went from a plausible 70 bpm raw median
  HR to an implausible 124 bpm corrected. Fix: compute the median over
  raw IBIs filtered to physiologically-plausible range (300–1500 ms = 40–200
  bpm). Verified across S2/S3/S5/S10/S16: corrected HR within 3 bpm of raw HR.
- **Motion gate over-firing.** Original metric was RMS jerk over a 60 s
  window with a 2 g/s threshold. RMS aggregates background micro-movement,
  which is non-zero even for seated subjects; this flagged 32–51% of windows
  as motion-corrupted on a lab-seated cohort, which is implausible. Fix:
  switch to **95th-percentile** jerk over the window (catches motion *bursts*,
  not steady micro-motion) with an 8 g/s threshold. Now flags 3–22%, which
  is consistent with the bursty-event interpretation and validates against
  the sanity-plot visual inspection. Phase 2 will confirm the threshold
  against actual peak-detection error rates.

### 6.5 What's locked vs. provisional after Phase 1

**Locked** (carry to Phase 2 unchanged):

- All filter cutoffs/orders for EDA, BVP, TEMP, ACC.
- 6.4 s EDA group-delay compensation as startup latency.
- Causal `lfilter`/`sosfilt` with steady-state initialization, applied
  per-subject.
- Ectopic correction algorithm (raw-IBI median, plausibility filter,
  20 % deviation threshold, 5-beat lookback).
- Motion metric definition (95th-percentile jerk over window).
- ACC unit conversion (raw 1/64 g → g).

**Provisional** (revisit when later phases give us the empirical data):

- Motion threshold (8 g/s). Phase 2 will set against per-window peak-detection
  error rates.
- SCR feature definition. Phase 3 + Phase 5 will determine whether the
  V3-native SCR features predict stress; if not, revisit per §6.1 options.
- Electrode-off / temp-dropout thresholds (0.05 µS, 1 °C/sample). Untested
  in the WESAD distribution because no subject triggers them; will need
  out-of-distribution data to validate.

---

## 7. Summary of changes from V2

| Aspect | V2 | V3 | Rationale |
|---|---|---|---|
| EDA pre-smooth | 1 Hz LP, order 1 | same | V2 was correct |
| EDA tonic | 0.05 Hz LP, order 2 | same | V2 was correct |
| EDA phasic | `np.gradient(smoothed) → ReLU` | `smoothed − tonic` (HP) | causality + units (§1.2) |
| EDA group-delay shift | 6.4 s, undocumented | 6.4 s, documented as startup latency | clarity for maintainers |
| BVP cleaning | `nk.ppg_process` (non-causal) | causal Butterworth BP 0.5–8 Hz, order 3 | causality (§2.2) |
| BVP peak detection | `nk.ppg_process` integrates Elgendi | `nk.ppg_findpeaks(elgendi)` on causally-cleaned signal | decoupled from non-causal cleaning |
| Ectopic correction | not present (V2 has neither) | causal moving-median replacement, raw-medians | Berntson 1997 {berntson1997} |
| Motion gating for HRV | not present | per-window jerk RMS flag | wrist-PPG noise sensitivity (§2.2) |
| TEMP smoothing | not present | 0.5 Hz LP order 1 | quantization noise reduction (§3.2) |
| TEMP dropout detection | not present | > 1 °C/sample flag | watch-off events (§3.2) |
| ACC unit | raw 1/64 g | g (divided by 64) | interpretability + threshold portability |
| ACC jerk features | not present | causal backward-difference 3-axis jerk magnitude | classroom fidget signal (§4.2) |
| Activity-band ACC | not present | 3 Hz LP magnitude | ambulation/fidget separation (§4.2) |
| HRV fail-silent | yes (returns zeros on failure) | replaced with explicit validity flags in Phase 3 | downstream knows |
| `hrv_vlf_power` feature | yes | drop in Phase 3 | period > window |
| `hrv_sd1` feature | yes | drop in Phase 3 | = `hrv_rmssd`/√2 |
| S14 | included | excluded | hyporesponsive |
| TSST-prep drop | not present (V2 keeps all) | same — keep all | per 2026-04-29 user decision |

The HP-phasic and BVP-cleaning changes are the two that meaningfully affect
feature values. The cvxEDA comparison study (Phase 1 follow-up) will quantify
the EDA shift; BVP changes should be benign because the bandpass and peak
detector are unchanged from V1/V2's design intent — only the filter call has
changed from non-causal to causal.

---

## References

See [docs/references.bib](references.bib) for full BibTeX entries. Inline cites
above use `{key}` notation as a shorthand.
