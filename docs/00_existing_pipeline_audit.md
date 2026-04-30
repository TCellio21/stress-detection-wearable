# 00 — Existing Pipeline Audit

**Author:** ML/signal-processing audit, conducted before Phase 1 of the theory-grounded rebuild.
**Date:** 2026-04-29
**Scope:** Inventory every component of the existing WESAD wrist-only pipeline, identify what is reusable for the deployable real-time pipeline, and identify what must be rebuilt to satisfy the causal/real-time/LOSO constraints in the rebuild plan.
**Decision rule:** A component is reusable only if (a) its formula is well-grounded, (b) it can be re-expressed causally without changing what is being measured, and (c) it has not silently leaked information across the LOSO boundary.

This document does **not** propose any code change. It catalogues what exists, why it exists, and what is wrong with it. Phase 1 begins after sign-off on the open questions at the bottom.

---

## 1. What pipelines currently exist

There are two coexisting feature-extraction pipelines plus stale scratch versions. Choosing which one is the "starting point" for the rebuild is itself a decision that needs sign-off.

| Pipeline | Path | Status | Core EDA method | Filter type | Normalization | Output features |
|---|---|---|---|---|---|---|
| **V1 — "Merged Grant + Tanner"** | [Updated_Extraction/](../Updated_Extraction/) | Canonical training pipeline. Notebooks 04/05/06 train on its output. | `nk.eda_phasic(method="cvxeda")` (global convex optimization) | Zero-phase (`sosfiltfilt`) | Per-subject z-score + percent-change vs. label=1 calibration baseline | 46 raw + 92 normalized = 138 |
| **V2 — "Causal & Aligned"** | [Updated_Extraction_V2/extract_windowed_datasets.py](../Updated_Extraction_V2/extract_windowed_datasets.py) | Newer attempt at a causal EDA path. Output CSV exists at [dataset/WESAD_causal_features_stride30.csv](../dataset/WESAD_causal_features_stride30.csv) but no model has been trained on it yet. | Butterworth low-pass at 0.05 Hz for tonic; first-derivative + ReLU for phasic | Causal (`lfilter`); features carry a 6.4 s startup latency from group-delay compensation (intentional, not look-ahead) | Per-subject z-score vs. label=1 calibration baseline (3 buckets: raw-only, raw+z, z-only) | 46 columns total (mix of raw and z) |
| Stale duplicate | [extract_windowed_datasets.py](../extract_windowed_datasets.py) (repo root) and [dataset/WESAD Extraction/extract_windowed_datasets.py](../dataset/WESAD%20Extraction/extract_windowed_datasets.py) | Older copies of V2; one has Grant's hardcoded path. Should be removed once we settle on the canonical layout. | — | — | — | — |
| Spaced-folder copy | `Updated Extraction/` (with space) | Deleted in working tree (see `git status`). Equivalent contents now live under `Updated_Extraction/` (underscore). | — | — | — | — |

The session logs (notebooks 04/05/06) report performance against **V1**. V2 has not been benchmarked end-to-end against V1.

**Deployability summary.** V1 is non-causal end-to-end (cvxEDA, `sosfiltfilt`, ppg_process default filtering) — needs preprocessing rebuild. V2 is causal in its filtering but has one true look-ahead (`np.gradient` central-differences for the phasic path) that needs fixing; its 6.4 s "alignment" shift is **not** a look-ahead — it's group-delay compensation that incurs a 6.4 s startup latency at wear-on, which is acceptable for stress detection (§4.2).

**On the deployment normalization model.** Both pipelines compute per-subject baseline statistics from label=1 windows, then z-score all subsequent windows against that fixed baseline. This matches the intended deployment paradigm: a wear-time **calibration period** (~15–20 min at sensor-on while the user is at rest) establishes the per-individual baseline, and every later window is scored as a deviation from it. This is *not* a sliding/trailing baseline — by design — and it does not violate causality because the calibration window is, by definition, in the past at prediction time. (Hard Constraint #2 in the rebuild plan literally specifies a trailing window of length B; the user has clarified the intended deployment is calibration-then-fixed-baseline, which is the simpler model and consistent with `PROJECT_ADVICE.md`'s "Week 1: Baseline establishment → Week 2+: Active use." Phase 4 is adjusted accordingly in §6.)

---

## 2. The 46 features — inventory

The 46 raw features are the same set in both V1 and V2. (V1 then doubles them with `_percent_change` and `_z_score` suffixes for 138 total; V2 keeps them in three buckets with selective z-scoring for ~46.)

### 2.1 EDA — 17 features (8 SCL + 7 SCR + 2 distribution)

| Feature | Type | Source code | Theoretical basis | Stress direction |
|---|---|---|---|---|
| `scl_mean`, `scl_median` | Tonic central tendency | [features.py:133-138](../Updated_Extraction/features.py#L133-L138) | Boucsein 2012 (EDA handbook): SCL reflects sympathetic tone. | ↑ |
| `scl_std`, `scl_range`, `scl_min`, `scl_max` | Tonic dispersion | [features.py:134-137](../Updated_Extraction/features.py#L134-L137) | Tonic variability is weak literature, but used in Schmidt 2018 (WESAD) feature set. | ambiguous |
| `scl_slope` | Linear trend in SCL across window | [features.py:142](../Updated_Extraction/features.py#L142) | Healey & Picard 2005 used SCL slope as a stress indicator. | ↑ during stress onset |
| `scl_auc` | Area under tonic curve | [features.py:146](../Updated_Extraction/features.py#L146) | Window-length-dependent; collinear with `scl_mean × window_size`. **Likely redundant.** | ↑ |
| `scr_peak_count` | # of detected SCRs | [features.py:149](../Updated_Extraction/features.py#L149) | Boucsein 2012: SCR rate is the most validated stress phasic indicator. **Top SHAP feature in V1 (rank #1).** | ↑ |
| `scr_amplitude_mean`, `scr_amplitude_max`, `scr_amplitude_std`, `scr_amplitude_sum` | SCR amplitude statistics | [features.py:150-153](../Updated_Extraction/features.py#L150-L153) | Boucsein 2012; Schmidt 2018. Amplitude reflects sympathetic burst magnitude. | ↑ |
| `scr_rise_time_mean` | Mean SCR rise time | [features.py:154](../Updated_Extraction/features.py#L154) | Standard EDA descriptor. | varies |
| `scr_recovery_time_mean` | Mean SCR recovery (50% decay) | [features.py:155](../Updated_Extraction/features.py#L155) | Boucsein 2012: T50/T63 recovery. | varies |
| `eda_skewness`, `eda_kurtosis` | Distribution shape on filtered EDA | [features.py:158-159](../Updated_Extraction/features.py#L158-L159) | Used in WESAD-style feature sets (Schmidt 2018 Table 2). Weak theoretical justification — keep as candidates, not core. | varies |

**SCL/SCR concept reminder.** Tonic SCL = slow background level (sec–min); phasic SCR = fast bursts in response to stimuli. Both rise under stress but on different timescales. `scl_slope` and `scr_peak_count` are the two features with the strongest literature support; `scl_auc` and the kurtosis/skewness pair are the weakest.

### 2.2 HRV — 15 features (9 time-domain + 5 frequency-domain + 1 nonlinear)

| Feature | Type | Source code | Theoretical basis | Stress direction |
|---|---|---|---|---|
| `hrv_mean_hr` | Mean HR (BPM) | [features.py:255](../Updated_Extraction/features.py#L255) | Basic SNS arousal indicator. **V1 fix:** HR is now correctly computed as `60000/MeanNN`; Grant's original code emitted `MeanNN` (ms) under this name. | ↑ |
| `hrv_mean_rr`, `hrv_median_rr`, `hrv_min_rr`, `hrv_max_rr` | Inter-beat interval stats | [features.py:262-266](../Updated_Extraction/features.py#L262-L266) | Inverse of HR; collinear with `hrv_mean_hr`. **Redundancy candidate** — keep only one. | ↓ |
| `hrv_rmssd` | RMS of successive RR differences | [features.py:259](../Updated_Extraction/features.py#L259) | Task Force 1996: vagal/parasympathetic tone. Most validated short-window HRV feature. | ↓ |
| `hrv_sdnn` | SD of NN intervals | [features.py:260](../Updated_Extraction/features.py#L260) | Task Force 1996: overall HRV. | ↓ |
| `hrv_sdsd` | SD of successive RR differences | [features.py:264](../Updated_Extraction/features.py#L264) | Task Force 1996. Highly correlated with RMSSD on short windows. | ↓ |
| `hrv_pnn50` | % of NN pairs differing > 50 ms | [features.py:261](../Updated_Extraction/features.py#L261) | Task Force 1996; vagal indicator, but coarse on 60 s windows. | ↓ |
| `hrv_lf_power`, `hrv_hf_power`, `hrv_vlf_power`, `hrv_total_power`, `hrv_lf_hf_ratio` | Welch PSD on resampled IBIs (LF=0.04–0.15, HF=0.15–0.4) | [features.py:271-275](../Updated_Extraction/features.py#L271-L275) | Task Force 1996. **VLF on a 60 s window is undefined** (period > window). LF (0.04–0.15 Hz, period 6.7–25 s) needs ≥ 60 s and ideally ≥ 120 s; HF is OK at 60 s. **V2 partly addresses this** by widening the freq window to 180 s. | LF/HF ↑ |
| `hrv_sd1` | Poincaré short-term variability | [features.py:279](../Updated_Extraction/features.py#L279) | Brennan 2001. SD1 is mathematically equivalent to RMSSD/√2. **Provably collinear with RMSSD.** | ↓ |

**Two structural issues with HRV as currently implemented:**
1. `hrv_vlf_power` on a 60 s window is computing noise — VLF is defined as 0.0033–0.04 Hz, which has period 25–300 s.
2. `hrv_sd1` and `hrv_rmssd` are mathematically the same feature up to a constant; including both costs nothing and adds nothing. Same for `hrv_min_rr`/`hrv_max_rr` vs. `hrv_mean_hr`.

### 2.3 Skin Temperature — 6 features

`temp_mean`, `temp_std`, `temp_min`, `temp_max`, `temp_median`, `temp_slope` — all simple summary statistics over the 60 s window ([features.py:296-329](../Updated_Extraction/features.py#L296-L329)).

**Theoretical basis.** Stress causes peripheral vasoconstriction → distal skin temperature drops over tens of seconds (Karthikeyan 2012, McFarland 1985). The most defensible feature is `temp_slope` (rate of change). `temp_mean` is room-temperature-confounded; `temp_std`/`min`/`max`/`median` are mostly redundant with `temp_mean` on slow signals.

### 2.4 Accelerometer — 8 features

| Feature | Source code | Theoretical basis |
|---|---|---|
| `acc_x_std`, `acc_y_std`, `acc_z_std` | [features.py:355-357](../Updated_Extraction/features.py#L355-L357) | Per-axis movement variability. Orientation-dependent. |
| `acc_magnitude_mean`, `acc_magnitude_std`, `acc_magnitude_max` | [features.py:361-363](../Updated_Extraction/features.py#L361-L363) | Orientation-invariant movement intensity. |
| `acc_sma` | [features.py:366](../Updated_Extraction/features.py#L366) | Signal Magnitude Area — Bouten 1997 standard activity metric. |
| `acc_energy` | [features.py:369](../Updated_Extraction/features.py#L369) | Movement energy. Strongly correlated with `acc_magnitude_mean²`. |

**Missing per the rebuild plan (Phase 3):** there are **no jerk features** in the current pipeline despite `PROJECT_ADVICE.md` and the Phase 3 spec calling for `mean jerk magnitude`, `std jerk`, and frequency-band energy (0.5–3 Hz ambulation, > 3 Hz fidgeting). For a wrist worn by an adolescent in a classroom, fidget detection is the key signal, not gross movement, so jerk-band features are likely the most valuable ACC additions.

### 2.5 Feature-count reconciliation

- 17 EDA + 15 HRV + 6 TEMP + 8 ACC = **46 raw features.** ✓ Confirmed.
- V1 outputs 46 raw + 46 `_percent_change` + 46 `_z_score` = **138 columns** (plus metadata). The 138 figure in session logs matches.
- The 95.6% accuracy claim in the rebuild prompt does **not** come from the 46/138-feature pipeline. It comes from a separate experiment in [experiments/rule_based_vs_ML/](../experiments/rule_based_vs_ML/) using **XGBoost on 11 hand-picked features** (`rule_based_vs_ml_comparison.md`). The actual best LOSO numbers on the 138-feature pipeline are **F1 = 0.814, recall = 0.821, accuracy ≈ 0.85** (HistGradientBoosting; [2026-03-11_nb05_nb06_results_analysis.md](session_logs/2026-03-11_nb05_nb06_results_analysis.md)). **The rebuild target should be the F1=0.814/recall=0.821 number, not 95.6%.** This is worth flagging because if "must beat 95.6%" leaks into design choices, we will over-engineer against an artifact.

---

## 3. Subject inclusion — discrepancy

| Pipeline | Subjects included | Excluded | Notes |
|---|---|---|---|
| V1 ([config.yaml:13-29](../Updated_Extraction/config.yaml#L13-L29)) | S2, S3, S4, S5, S6, S7, S8, S9, S10, S11, S13, S15, S16, S17 (n=14) | **S14 excluded** (hyporesponsive) | Documented decision; matches MEMORY.md and prior analyses. |
| V2 ([extract_windowed_datasets.py:40](../Updated_Extraction_V2/extract_windowed_datasets.py#L40)) | S2 … S14 … S17 (n=15) | None | **S14 is included.** Inconsistent with V1. |

S1 and S12 are absent from WESAD upstream. S9 has no E4 data per MEMORY.md but appears in both subject lists — needs verification at load time.

**Decision needed:** is S14 in or out of the new pipeline? §6 asks the user explicitly.

---

## 4. Real-time/causality audit — file:line of every violation

The rebuild plan mandates: no filter, decomposition, or normalization may use future samples relative to the prediction time. Every violation below blocks deployment as-is.

### 4.1 V1 (Updated_Extraction)

| Step | File:Line | Operation | Why it violates causality | Severity |
|---|---|---|---|---|
| EDA pre-filter | [features.py:52](../Updated_Extraction/features.py#L52) | `signal.sosfiltfilt(...)` (zero-phase forward-backward Butterworth, 1.8 Hz LP) | `sosfiltfilt` runs the filter both directions; every output sample depends on every input sample, including future. | **Blocker** — violates Phase 1 spec verbatim. |
| EDA decomposition | [features.py:95](../Updated_Extraction/features.py#L95) | `nk.eda_phasic(method="cvxeda")` | cvxEDA solves a convex optimization over the entire window simultaneously — the solution at sample t depends on samples t+1…end. Cannot be made causal without changing the algorithm. | **Blocker** — explicitly named in plan as the canonical example of what is OUT for the deployable pipeline. |
| BVP/HRV | [features.py:237](../Updated_Extraction/features.py#L237) | `nk.ppg_process(...)` | NeuroKit2's `ppg_process` calls `signal_filter` internally with `method="butterworth"`, which by default uses zero-phase filtering. **Need to verify** with the installed NK2 version, but historically non-causal. | High — verify in Phase 1. |
| Normalization | [normalization.py:67](../Updated_Extraction/normalization.py#L67) | Baseline mean/std computed from all label=1 windows in the subject's session | Matches the intended deployment paradigm (wear-time calibration, then fixed baseline). Causally OK: calibration data is always in the past at prediction time. **One caveat:** for LOSO evaluation we must verify that the calibration baseline used for the held-out subject draws only from that subject's own label=1 windows (it does — [normalization.py:66](../Updated_Extraction/normalization.py#L66) filters per-subject). | **Reusable** — formalize the calibration-window definition in Phase 4. |

### 4.2 V2 (Updated_Extraction_V2)

V2 *attempted* to fix V1's causality issues but introduced new look-ahead in the process.

| Step | File:Line | Operation | Why it violates causality | Severity |
|---|---|---|---|---|
| EDA artifact LP | [extract_windowed_datasets.py:101-102](../Updated_Extraction_V2/extract_windowed_datasets.py#L101-L102) | `signal.lfilter(b, a, eda_raw, zi=...)` (1 Hz LP, order 1) | Causal ✓ | Reusable. |
| EDA tonic LP | [extract_windowed_datasets.py:105-107](../Updated_Extraction_V2/extract_windowed_datasets.py#L105-L107) | `signal.lfilter` (0.05 Hz LP, order 2) | Causal ✓. **Choice of cutoff defensible** — Boucsein 2012 places the SCL/SCR boundary near 0.05 Hz. | Reusable with audit of cutoff. |
| EDA phasic | [extract_windowed_datasets.py:110-112](../Updated_Extraction_V2/extract_windowed_datasets.py#L110-L112) | `np.gradient(smoothed_eda, dt)` then `np.maximum(0, derivative)` | **`np.gradient` uses central differences:** `(x[i+1] − x[i−1]) / (2·dt)` for interior points. **This requires the next sample**, so the phasic estimate at t depends on t+1. | **Look-ahead by one sample (0.25 s at 4 Hz).** Easy fix: replace with backward difference `(x[i] − x[i−1])/dt` or a causal high-pass. |
| EDA "alignment" / group-delay compensation | [extract_windowed_datasets.py:114-118](../Updated_Extraction_V2/extract_windowed_datasets.py#L114-L118) | `eda_tonic[delay_samples:]` then `np.pad(..., (0, delay_samples))` — group-delay compensation | The causal LP filter introduces ~6.4 s of group delay (output peaks lag input peaks). To re-align filtered features with the input timeline (so a feature timestamped t reflects physiology *at* t, not at t-6.4 s), the code emits filtered output 6.4 s after each corresponding input sample. **This is not look-ahead** — it's a startup latency: at clock time t, the latest aligned sample available represents input-time t-6.4 s. In deployment, predictions begin 6.4 s after sensor-on. The end-of-signal padding (`mode="edge"`) is a training-time artifact (no future samples available); in real-time deployment, those last 6.4 s simply never produce predictions (the wearer takes the watch off). | **Reusable, document carefully.** Future maintainers will read this as a non-causal shift unless we comment the deployment semantic. |
| BVP/HRV | [extract_windowed_datasets.py:200, 235](../Updated_Extraction_V2/extract_windowed_datasets.py#L200) | Same `nk.ppg_process` as V1. | Same concern as V1.4. | High — verify. |
| Normalization | [extract_windowed_datasets.py:343-352](../Updated_Extraction_V2/extract_windowed_datasets.py#L343-L352) | Baseline z-score from all `raw_label==1` windows of the subject | Calibration paradigm (same as V1.5) — fine. | **Reusable.** |

### 4.3 Other things flagged

- **EDA SCR amplitude outlier bounds.** V1 uses `outlier_max=1.0 µS` ([config.yaml:45](../Updated_Extraction/config.yaml#L45)); V2 uses `5.0 µS` ([extract_windowed_datasets.py:55](../Updated_Extraction_V2/extract_windowed_datasets.py#L55)) **applied to a derivative-based phasic signal whose units are µS/s**. The threshold and the signal aren't in the same units in V2. Either the bound is unintentionally wider than V1's clinical 1 µS, or the units are inconsistent. Needs verification.
- **EDA SCR detection.** V2 runs `nk.eda_peaks` on a `np.maximum(0, derivative)` signal, not on a phasic component decomposed by cvxEDA. `nk.eda_peaks` was designed to operate on a properly-decomposed phasic signal; the resulting peaks-info amplitudes may not be directly comparable to V1's. This is an empirical question for Phase 1.
- **HRV fail-silent.** [features.py:215-289](../Updated_Extraction/features.py#L215-L289) returns a dict of zeros when peak detection finds < 2 peaks or any exception fires. The session log [2026-03-11_nb06_full_loso_analysis.md](session_logs/2026-03-11_nb06_full_loso_analysis.md) documents this is suspected to drive the high `hrv_median_rr` variance (CV = 12.79). Phase 3 must surface valid/invalid per window explicitly.
- **`np.trapz` shim.** Both pipelines monkey-patch `np.trapz = np.trapezoid` ([extract_windowed_datasets.py:11-12](../Updated_Extraction_V2/extract_windowed_datasets.py#L11-L12)) for NumPy ≥ 2.0 compatibility. Cosmetic, but documents that the env has been migrated past NumPy 2.0.
- **Bare `except`.** [features.py:114](../Updated_Extraction/features.py#L114) catches `(ValueError, KeyError, IndexError, TypeError)` to no-op SCR detection failures. Better than a bare except, acceptable, but the rebuild should turn these into explicit "no SCR detected" markers, not zeroed amplitudes.
- **Hardcoded path in stale duplicate.** [extract_windowed_datasets.py:23 (root)](../extract_windowed_datasets.py#L23) hardcodes Grant's machine path. Not used by anything live; should be deleted in Phase 1 cleanup.

---

## 5. Reusable vs. rebuild — component matrix

| Component | Verdict | Notes |
|---|---|---|
| Subject loop / window iteration | **Reusable** (logic) | The `for window in subject` pattern is fine. The "virtual microcontroller" loop from Hard Constraint #3 simplifies under the calibration paradigm: there's a one-time calibration phase that establishes baseline stats, then a steady-state loop of "advance cursor → extract features → z-score against fixed calibration stats → classify → log." No trailing-window recomputation per cursor step. |
| Label alignment ([dataset_builder.py:55-75](../Updated_Extraction/dataset_builder.py#L55-L75)) | **Reusable** | `searchsorted`-based alignment of 700 Hz labels to 4 Hz EDA timebase is correct and causal. Keep as-is. |
| Subject inclusion list | **Decision needed** (S14) | See §3. |
| TSST prep exclusion (first 3 stress windows) | **Reusable** but **revisit** | V1 drops 180 s of stress per subject. With small per-subject stress counts (7–9 windows), this is significant. Defensible per Schmidt 2018 (TSST instructions are a confound), but Phase 2 should re-test whether it actually helps. |
| EDA pre-filter | **Rebuild** | V1 zero-phase, V2 has the right idea (causal LP) but with broken alignment. Phase 1 picks one causal LP and lives with the group delay. |
| EDA decomposition (tonic/phasic) | **Rebuild** | cvxEDA out. V2's "causal LP for SCL + ReLU(derivative) for SCR" is a defensible starting point per the March 3 logbook proposal. Phase 1 must compare against cvxEDA on the same windows and quantify divergence (per the plan). |
| EDA peak/amplitude detection | **Rebuild** | Outlier bounds need to be re-derived for the causal phasic estimator (units, thresholds). |
| HRV peak detection (BVP) | **Verify, then likely rebuild** | NK2's `ppg_process` is the de facto standard but its causality on the installed version needs verification. If non-causal, swap to a causal alternative (e.g., HeartPy with its causal mode, or an explicit causal Pan-Tompkins-style detector tuned for PPG). |
| HRV time-domain math | **Reusable** | `RMSSD`, `SDNN`, `pNN50`, `MeanNN` are formula-only; once peaks are causal, the math is fine. |
| HRV frequency-domain | **Rebuild scope** | Drop `hrv_vlf_power` (period > window). Adopt V2's decoupled 180 s freq window. |
| HRV nonlinear (`SD1`) | **Drop** as redundant | Mathematically equivalent to `RMSSD/√2`. Keeping both does nothing. |
| TEMP features | **Reusable** | Pure summary stats; causal trivially. Possibly add a causal LP if we add a derivative. |
| ACC features (current 8) | **Reusable** | All causal trivially. |
| ACC features (jerk, frequency-band energy) | **New build** | Specified in Phase 3, not present today. |
| Normalization (per-subject z) | **Reusable** | Calibration paradigm confirmed as the design intent: collect ~15–20 min of label=1 windows at wear-on, compute mean/std, z-score everything afterwards. Math and code already match this. Phase 4 still runs the four-formula comparison (no-norm / z / percent-change / robust-z), but the *baseline window length* sweep is replaced with a *calibration length* sweep using only what's available pre-stress in WESAD. |
| Percent-change normalization | **Reusable but flag** | The session log notes near-zero baselines drive `_percent_change` features negative for some subjects. Phase 4's experiment over four normalization schemes must include the robust z-score variant to handle this. |
| Output schema (CSV + manifest) | **Reusable** | Run manifest with config snapshot is good practice. Keep. |
| Config loading (YAML + .env) | **Reusable** | [config_loader.py](../Updated_Extraction/config_loader.py) is well-structured. Reuse for the new pipeline. |
| LOSO splitting | **Reusable but extend** | Existing notebooks 04/05/06 use single-loop LOSO. Phase 6/7 require **nested** LOSO (outer LOSO for evaluation, inner k-fold over remaining subjects for hyperparameter tuning). Tuning currently happens on the outer test fold in some places — that's a hard fail per the rebuild plan. |
| Random seeds | **Verify** | `reproducibility.random_seed: 42` is in V1's config. Need to confirm sklearn/xgboost/optuna seeds get propagated in the new pipeline. |
| `experiments/rule_based_vs_ML/` | **Drop from main path** | The 95.6% claim originates here. Useful as a baseline-comparison demo for the defense doc but not the production pipeline. |
| Notebooks 04/05/06 | **Reference, not reuse** | Useful for replicating prior numbers. The rebuild produces a new training pipeline; notebooks 04/05/06 stay as historical baseline documentation. |
| Stale duplicate scripts | **Delete** | [extract_windowed_datasets.py (root)](../extract_windowed_datasets.py), [dataset/WESAD Extraction/extract_windowed_datasets.py](../dataset/WESAD%20Extraction/extract_windowed_datasets.py). |

---

## 6. Open questions — answer before Phase 1

These are decisions where the existing pipeline has an answer but the rebuild needs sign-off.

### Resolved by user clarification (2026-04-29)

- **Normalization paradigm.** Wear-time calibration period establishes a fixed per-subject baseline; subsequent predictions z-score against it. Not a sliding/trailing window. Phase 4's window-length sweep becomes "calibration-length sweep within what WESAD provides (≈20 min pre-stress)."
- **6.4 s EDA group-delay compensation.** Not look-ahead; intentional startup latency. V2's implementation stays.
- **Structural template:** **V2** (`Updated_Extraction_V2/`). V1 is reference for benchmarking only.
- **S14:** Excluded (hyporesponsive). V2's current inclusion of S14 must be removed in V3.
- **Performance target:** F1=0.814 / recall=0.821 is **aspirational**, not a hard gate. "Best results we can get" — push for higher recall but don't compromise the design to chase the number.
- **Subject 2:** Out of scope for the rebuild. Acknowledge in defense doc; don't optimize for it. (Future work.)
- **TSST-prep windows:** **Do not drop.** Keep all 180 s of stress-onset windows. V2's behavior already; V1's prep-drop logic is not ported to V3.
- **Stale files:** Root `extract_windowed_datasets.py` deleted. `dataset/WESAD Extraction/extract_windowed_datasets.py` is teammate's work — leave it.
- **Layout:** New pipeline at `Updated_Extraction_V3/` at repo root. V1 and V2 stay as historical references.

### Phase 1 implications of "V2 as template"

V2 is causal in its filtering structure but has these specific issues that Phase 1 must fix or audit:

1. **`np.gradient` central-difference** ([extract_windowed_datasets.py:111](../Updated_Extraction_V2/extract_windowed_datasets.py#L111)) — uses `(x[i+1]-x[i-1])/2dt` for interior points; one-sample look-ahead. Replace with backward difference `(x[i]-x[i-1])/dt` or a causal high-pass.
2. **SCR outlier bounds vs. units** ([extract_windowed_datasets.py:55, 149](../Updated_Extraction_V2/extract_windowed_datasets.py#L55)) — `OUTLIER_MAX=5.0` is applied to amplitudes returned by `nk.eda_peaks` on a derivative-based phasic signal. Need to verify what units `nk.eda_peaks` returns in this case and whether the threshold is right; V1 used 1.0 µS on a cvxEDA-derived phasic, which is not directly comparable.
3. **`nk.ppg_process` causality** ([extract_windowed_datasets.py:200, 235](../Updated_Extraction_V2/extract_windowed_datasets.py#L200)) — must verify the installed NK2's filter is causal; if it uses `filtfilt` internally, swap to a causal alternative.
4. **S14 exclusion** ([extract_windowed_datasets.py:40](../Updated_Extraction_V2/extract_windowed_datasets.py#L40)) — drop S14 from the SUBJECTS list.
5. **Document the 6.4 s startup latency** in code comments so it isn't read as a bug.
6. **`hrv_vlf_power`** is computing noise on a 60 s window (VLF period 25–300 s). Drop in V3.
7. **`hrv_sd1`** is mathematically `hrv_rmssd/√2`. Drop in V3.
8. **Add Phase 3 jerk features** to ACC: mean jerk magnitude, std jerk, frequency-band energy (0.5–3 Hz / >3 Hz). Currently absent.
9. **HRV fail-silent** behavior (zeroed features when peak detection fails) inherited from V1's `extract_hrv_features_window` should be replaced with explicit validity flags so downstream knows.

### Still open (lower priority — can be settled inside Phase 1/2)

- **Window labeling rule.** Both pipelines use majority vote. With 60 s windows on a clean TSST onset this rarely matters; one diagnostic plot at transition windows during Phase 2 will close it.

---

## 7. Summary — what to expect in Phase 1

Once the questions in §6 are answered, Phase 1 ("Preprocessing — filtering and artifact removal") will:
- Pick V1 as the structural template, port V2's causal `lfilter` LP for EDA (without the 6.4 s shift), and replace cvxEDA with the causal SCL/SCR pair (LP for tonic, causal HP or backward-difference for phasic).
- Replace `sosfiltfilt` everywhere with causal `sosfilt`.
- Verify `nk.ppg_process` causality and swap if needed.
- Quantify the divergence between cvxEDA and the causal alternative on the same windows (per the plan's "run a comparison" instruction).
- Document each decision in `docs/01_preprocessing.md` with citation, empirical evidence, rationale, and a sanity-plot notebook on 2–3 subjects.

**No code changes to existing files until §6 is signed off.**

---

## References (to be expanded into `docs/references.bib` in Phase 1)

- Boucsein, W. (2012). *Electrodermal Activity* (2nd ed.). Springer.
- Bouten, C. V. C., et al. (1997). A triaxial accelerometer and portable data processing unit for the assessment of daily physical activity. *IEEE TBME*.
- Brennan, M., Palaniswami, M., & Kamen, P. (2001). Do existing measures of Poincaré plot geometry reflect nonlinear features of heart rate variability? *IEEE TBME*.
- Healey, J., & Picard, R. W. (2005). Detecting stress during real-world driving tasks using physiological sensors. *IEEE Trans. ITS*.
- Karthikeyan, P., et al. (2012). Detection of human stress using short-term EDA, BVP, and skin temperature. *IEEM*.
- McFarland, R. A. (1985). Relationship of skin temperature changes to the emotions accompanying music. *Biofeedback and Self-Regulation*.
- Schmidt, P., et al. (2018). Introducing WESAD, a multimodal dataset for wearable stress and affect detection. *ICMI*.
- Task Force of ESC and NASPE. (1996). Heart rate variability: Standards of measurement, physiological interpretation, and clinical use. *Circulation*.
