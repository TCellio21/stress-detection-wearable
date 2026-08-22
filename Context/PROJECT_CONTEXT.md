# AI-Driven Self-Regulation System — LLM Context Brief
Last updated: 2026-06-08

Standalone briefing for LLM assistants with **no access to the project repository**. Paste this at the start of a conversation to restore full context.

---

## Product & Mission

We are building an **AI-driven self-regulation system** for K-12 classrooms using the **Samsung Galaxy Watch 8**. The watch continuously monitors physiological signals and runs a machine learning model to estimate rising arousal/stress. When early stress is detected, it delivers a **discreet, personalized coping prompt** — for example, a walking character on the display suggesting the student step out for a short break. The system reinforces coping strategies the student already knows and is comfortable with, empowering independent self-regulation **before** distraction, classroom disruption, or teacher intervention.

**Target market:** K-12 schools; educators, occupational therapists, and support staff working with students who struggle to manage dysregulation.

**Early adopter segment:** Students with ADHD, anxiety, and autism spectrum disorder — populations with acute need and growing prevalence in school districts.

**Market validation:** Survey of 100+ teachers, therapists, and school professionals across Chicagoland. Respondents were asked only about daily problems they face — the device was not described — yielding unbiased accounts of classroom dysregulation challenges. The team also works closely with a school district partner throughout development.

**Team:** Milwaukee School of Engineering (MSOE) Senior Design project.

**Hardware progression:** EmotiBit (research-grade) → HealthyPi Move (prototype wearable) → Samsung Galaxy Watch 8 (target consumer deployment device).

---

## Physiological Signals

| Signal | Purpose | Sampling (WESAD wrist) |
|--------|---------|------------------------|
| **EDA** (electrodermal activity) | Phasic SCR peaks + tonic SCL — primary stress indicator | 4 Hz |
| **PPG/BVP** | Heart rate variability (time + frequency features) | 64 Hz |
| **Skin temperature** | Thermoregulatory stress response | 4 Hz |
| **Accelerometer** | Movement artifact cleaning; planned hyperactivity proxy | 32 Hz |

Respiration is **not** used — it exists on WESAD's chest sensor only; all models use wrist data for deployability.

---

## ML Foundation: WESAD Dataset

**WESAD** is a public stress dataset with synchronized wrist (Empatica E4) and chest (RespiBAN) sensors. We train on **wrist signals only**.

**Subjects used:** 14 of 15 available (IDs S2–S17, excluding S14 due to data quality / atypical stress response). WESAD subjects are **adults (~28 ± 3 years)** — a known gap vs. our adolescent classroom target.

**Stress protocol in WESAD:** Baseline rest → Trier Social Stress Test (TSST) → amusement/recovery blocks. Labels are time-aligned at 700 Hz in each subject's pickle file.

**Binary classification:**
- **Stress (1):** WESAD label 2 (TSST)
- **Non-stress (0):** labels 1, 3, 4 (baseline, amusement, meditation)

**Critical preprocessing choices:**
- First 3 stress windows (180 s of TSST prep) dropped per subject — not true stress yet
- 60-second windows, 60-second step (non-overlapping)
- Majority-vote label per window
- Causal filters only (no look-ahead) for real-time deployability
- Raw features at inference — z-score normalization was tested and rejected (Phase 4)
- HRV frequency features use a 180 s lookback; time-domain HRV uses the 60 s window
- **Always evaluate with leave-one-subject-out (LOSO) CV** — never random splits across subjects

---

## Current ML Pipeline (Version 3 — Authoritative)

Earlier pipeline versions (V1: 138-feature Random Forest, V2: causal rewrite attempt) are **deprecated**. All current work uses **V3**, rebuilt in documented phases 0–7 (complete).

### Pipeline phases completed
0. Audit of prior pipelines
1. Causal preprocessing + artifact detection
2. Windowing sweep → locked at W=60 s, step=60 s
3. Feature extraction → 48 raw features per window
4. Normalization comparison → locked: **no normalization**
5. Feature selection → locked: **16 features** (combined stability + SHAP rank)
6. Model selection → HistGradientBoostingClassifier (HGB) primary; LightGBM as fast fallback
7. Nested-LOSO hyperparameter tuning → HGB+SVM-RBF probability ensemble recommended

### Locked 16-feature set
scr_recovery_time_mean, hrv_median_rr, scr_peak_count, hrv_pnn50, scl_max, hrv_sdsd, scl_range, scr_amplitude_sum, temp_max, hrv_min_rr, acc_jerk_mag_mean, temp_slope, scr_rise_time_mean, acc_magnitude_std, scr_amplitude_max, acc_y_std

EDA/HRV features dominate; temperature and accelerometer (jerk, magnitude) provide supporting signal.

### Model performance (LOSO, 14 subjects)

| Model | F1 | Other notes |
|-------|-----|-------------|
| **HGB + SVM-RBF ensemble** (recommended) | **0.935** | Best overall; recovers weak subject S3 to 0.727 recall |
| Default HGB | 0.931 | Recall 0.911, precision 0.972, accuracy 0.973 |
| Tuned HGB (nested LOSO, unbiased) | 0.917 ± 0.085 | Best worst-case recall — subject S9 reaches 0.636 vs 0.545 |
| LightGBM | 0.921 | ~1.4 ms inference (vs ~10 ms HGB); deployment fallback |

**Problem subjects:** S9 has structurally low recall (~0.545) across all models. S3 has corrupted baseline data; ensemble partially recovers performance.

**Dataset size:** 691 windows across 14 subjects at the locked window config.

### Inference
Models expect a 16-element feature vector per 60 s window (raw values, no scaling). Threshold 0.5 for binary stress prediction. Three exported variants exist: ensemble (ship), tuned HGB, default HGB.

---

## Hardware Validation Status

### WESAD Empatica E4
Primary training data. Wrist-only signals. All reported ML metrics above are on this data.

### EmotiBit (lab prototype)
Controlled stress data collected; signal analysis in progress. **No published end-to-end result** applying the WESAD-trained model to EmotiBit recordings with accuracy/F1 metrics.

**Ice test (cold pressor):** 1.8°C water. Protocol: 1 min calibration (discard) → 5 min baseline → 2 min hand in ice water (stress) → 5 min recovery. Goal: validate physiological stress response and justify **4 Hz minimum EDA sampling** (compare vs 1 Hz downsample).

**Stroop test (cognitive stress):** Seated, minimal movement. Protocol: 1 min calibration (discard) → 5 min baseline → 2 min Stroop task (stress) → 5 min recovery. Recordings collected from multiple team members (Grant, Adam, Tanner).

### HealthyPi Move (prototype wearable)
BLE streaming client built — records PPG, EDA, and skin temperature to CSV. Onboard SD recording also supported. **Accelerometer not available over BLE** in current firmware. **EDA reliability on this device is flagged as a concern** — an open question whether EDA features should be included in HealthyPi/Galaxy Watch deployment.

### Samsung Galaxy Watch 8 (target device)
Intended classroom deployment platform. **No Galaxy Watch data collection or on-device pipeline documented yet.** Next major hardware milestone.

---

## Other Completed Work

- Personalization concept: Week-1 baseline calibration, per-user z-score normalization, adaptive thresholds
- Rule-based vs ML comparison experiment (11-feature XGBoost reached ~95.6% accuracy on a separate feature set — not the shipping model)
- Causal-filtering XGBoost experiment: ~94% accuracy with or without EDA — EDA inclusion for deployment still undecided
- Phased technical documentation of all V3 design decisions (preprocessing through hyperparameter tuning)

---

## Not Yet Completed

- Real-time inference simulator on streaming data
- Dashboard / teacher-facing UI (scaffold only)
- Hyperactivity detection model (discussed; accelerometer clustering planned; not trained)
- Applying V3 model to EmotiBit, HealthyPi, or Galaxy Watch data with reported metrics
- Adolescent validation dataset (WESAD adults ≠ classroom teens)
- IRB-controlled stress study on target population using Galaxy Watch 8
- Commercial pilot agreement with school district partner
- On-watch coping prompt delivery (product feature)

---

## Known Gaps & Open Questions

1. **Domain gap:** WESAD adults in a lab vs. adolescents in classrooms — model will need fine-tuning / transfer learning on teen data.
2. **EDA on deployment hardware:** HealthyPi EDA may be unreliable; Galaxy Watch EDA quality TBD.
3. **Ground truth in the field:** Lab protocols have clear phase labels; classroom stress lacks clean ground truth — may need self-report, teacher observation, or proxy labels.
4. **Personalization:** How much baseline calibration is needed per student before reliable detection?
5. **Hyperactivity:** No labeled hyperactivity data yet; plan to derive proxy labels from accelerometer movement patterns.

---

## Future Plans (Near-Term Roadmap)

1. Collect Galaxy Watch 8 data via **IRB-approved controlled stress tests** on adolescent participants to refine the model for the actual use-case population.
2. Execute a **commercial pilot** with an initial school district partner after IRB validation.
3. Build real-time on-device inference pipeline on the watch.
4. Implement discreet coping prompts tied to each student's known regulation strategies.
5. Explore hyperactivity detection as a complementary signal for the ADHD segment.

---

## How to Assist This Team Effectively

- Treat **V3 results (F1 = 0.935 ensemble, 16 features, LOSO)** as the authoritative ML baseline. Older notes referencing Random Forest F1 ≈ 0.89 or 138-feature pipelines are outdated.
- Insist on **subject-wise splits** for any new evaluation — data leakage across subjects invalidates results.
- Remember the product is **proactive self-regulation**, not just stress classification — UX, discretion, and student autonomy matter as much as model accuracy.
- Flag when suggestions assume adolescent or classroom data that does not yet exist.
- WESAD is the training foundation; Galaxy Watch + IRB study is the path to deployment.

---

## One-Paragraph Summary

MSOE Senior Design team building a Samsung Galaxy Watch 8 system that detects rising stress in K-12 students and delivers discreet coping prompts before classroom disruption. ML models trained on the public WESAD wrist-sensor dataset achieve **F1 ≈ 0.94** (LOSO, 16-feature HGB+SVM ensemble) using EDA, PPG/HRV, temperature, and accelerometer features. Validated conceptually through EmotiBit lab tests (ice + Stroop) and a HealthyPi BLE prototype; market validated via 100+ educator survey and school district partnership. **Next:** IRB study collecting Galaxy Watch data on adolescents, model refinement for classroom deployment, then commercial school pilot.
