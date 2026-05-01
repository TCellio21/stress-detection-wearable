# 06 — Model selection

**Phase 6 of the V3 rebuild.** Compare six model classes on the locked 16 features from Phase 5.

**Date:** 2026-04-30
**Notebook:** [notebooks/06_model_selection.ipynb](../notebooks/06_model_selection.ipynb)
**Status:** ✅ chosen **HistGradientBoosting at threshold 0.5**. F1 = 0.931, recall = 0.911, min subj recall = 0.545. **LightGBM is the backup** for embedded deployment (~7× faster inference, similar performance).

---

## 1. Headline result

Six models compared on the locked 16-feature set, LOSO over 14 subjects, evaluated on F1 / recall / precision / accuracy / ROC-AUC plus inference time and pickle size.

| Rank | Model | F1 | Recall | Precision | Accuracy | ROC-AUC | min subj | inference (ms) | size (KB) |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | **HGB** ← chosen | **0.931** | 0.911 | 0.972 | 0.973 | 0.997 | 0.545 | 10.0 | 430 |
| 2 | LightGBM | 0.921 | 0.904 | 0.961 | 0.967 | 0.997 | 0.545 | **1.4** | 411 |
| 3 | XGBoost | 0.916 | 0.904 | 0.953 | 0.964 | 0.995 | 0.545 | 0.6 | 312 |
| 4 | RandomForest | 0.876 | 0.834 | 0.961 | 0.953 | 0.992 | 0.500 | 57.9 | 1305 |
| 5 | SVM-RBF | 0.846 | 0.859 | 0.860 | 0.931 | 0.978 | 0.545 | **0.3** | 34 |
| 6 | LogReg | 0.767 | 0.853 | 0.731 | 0.882 | 0.963 | 0.455 | 0.4 | **2** |

CSV: [reports/06_model_selection/deployment_summary.csv](../reports/06_model_selection/deployment_summary.csv).

**Three observations from the headline:**

1. **The three gradient-boosted tree methods cluster at the top.** HGB, LightGBM, XGBoost all land within 0.015 F1 of each other and have nearly identical ROC-AUC (0.995–0.997). They're trading rank-order positions on noise — any of them is a defensible primary model.
2. **Linear and kernel methods (LogReg, SVM-RBF) sit ~0.08–0.16 F1 below the boosters.** They're not bad — both meet recall ≥ 0.85 — but the precision drop tells us the 16 raw features encode non-linear physiological relationships that linear classifiers don't capture as well.
3. **min subj recall is identical (0.545) across all models except LogReg and RF.** S9 fails the same way regardless of model class (more on this in §3).

---

## 2. Per-subject recall heatmap

CSV: [reports/06_model_selection/per_subject_recall.csv](../reports/06_model_selection/per_subject_recall.csv).
Plot: [reports/06_model_selection/per_subject_heatmap.png](../reports/06_model_selection/per_subject_heatmap.png).

| Subject | HGB | LightGBM | XGBoost | RandomForest | SVM-RBF | LogReg |
|---|---:|---:|---:|---:|---:|---:|
| S2 | 0.900 | 0.900 | 0.900 | 0.800 | 0.900 | 0.900 |
| **S3** | 0.636 | 0.636 | 0.636 | 0.545 | **0.727** | **0.455** ⚠ |
| S4–S6 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| S7 | 1.000 | 1.000 | 1.000 | 0.900 | 0.800 | 1.000 |
| S8 | 1.000 | 1.000 | 1.000 | 0.909 | 1.000 | 1.000 |
| **S9** | **0.545** | **0.545** | **0.545** | **0.545** | **0.545** | **0.545** |
| S10 | 0.750 | 0.750 | 0.750 | 0.500 | 0.667 | 0.750 |
| S11 | 1.000 | 1.000 | 1.000 | 1.000 | 0.818 | 0.909 |
| S13 | 1.000 | 1.000 | 1.000 | 1.000 | 0.909 | 1.000 |
| S15 | 1.000 | 1.000 | 1.000 | 1.000 | 0.909 | 0.727 |
| **S16** | **1.000** | 0.909 | 0.909 | 0.636 | 1.000 | 0.909 |
| S17 | 0.917 | 0.917 | 0.917 | 0.833 | 0.750 | 0.750 |

Three findings worth highlighting:

### S9's 0.545 recall is structural across **every** model class

Every single one of the 6 models gets exactly 0.545 recall on S9 — they all miss the same 5 of S9's 11 stress windows. This is **not** a model-class problem. The §3 OOF probability histogram (see §5 below) shows those missed S9 windows have predicted probabilities < 0.001 (confidently mis-classified), so threshold tuning can't recover them either. **S9's recall floor is structural** — fixing it would require either (a) different/additional features (not in the locked 16), (b) per-subject calibration based on the 5 missed windows' physiological signatures, or (c) excluding S9 from training the way V1 excluded S14.

This contrasts with V1's session log, where the V1 S2 problem *was* model-class-specific (trees failed, linear models recovered S2). In V3 with raw features, S2 is recovered uniformly across all models at 0.900 recall — confirming the V3 pipeline doesn't have V1's particular pathology. S9 is a different problem, structurally tied to that subject's atypical stress signature.

### S3's recall varies with model class

S3's recall ranges from **0.455 (LogReg) to 0.727 (SVM-RBF)** — a 0.27 spread across model classes. S3's underlying dataset issue (corrupted-baseline per V2-author's diagnosis — see [memory/project_s3_corrupted_baseline.md](../../../memory/project_s3_corrupted_baseline.md)) means no model gets S3 perfect, but linear and kernel models handle it differently from trees:

- LogReg's 0.455 is below the 0.5 acceptance threshold — the **only sub-0.5 result anywhere in the table**. LogReg's reliance on monotonic feature relationships makes S3's inverted-baseline signature uniquely confusing.
- SVM-RBF's 0.727 is the **best S3 recall of any model**. Its non-linear kernel apparently finds a separating boundary that handles S3's inverted relationship better than tree-based threshold splits.
- Tree-based methods (HGB / LightGBM / XGBoost) all sit at 0.636 — same per-fold pattern, same misses.

This is a real signal but not enough to flip the choice. SVM-RBF's overall F1 (0.846) is too far below HGB (0.931) to justify the trade. Filed as a known limitation; potentially revisitable in Phase 7 if hyperparameter tuning on SVM closes the gap.

### S10 and S16 reveal RF's brittleness

RandomForest's S10 (0.500) and S16 (0.636) are the worst readings on those subjects in the entire table. Despite RF having competitive overall F1 (0.876), it produces the **highest per-subject variance** of any model. The 1.3 MB pickle size and 58 ms inference time confirm RF is over-parameterized for this 691-window dataset — the model is memorizing training-fold idiosyncrasies that don't transfer. Eliminated from contention.

---

## 3. Inference time and model size — deployability

Single-window prediction time, averaged over 1000 calls per model on CPU:

| Model | inference (ms) | size (KB) | passes < 100 ms target? |
|---|---:|---:|:---:|
| SVM-RBF | 0.283 | 33.5 | ✓ |
| LogReg | 0.417 | 1.9 | ✓ |
| **XGBoost** | **0.641** | **311.7** | ✓ |
| **LightGBM** | **1.382** | **411.0** | ✓ |
| **HGB** | **9.997** | **430.3** | ✓ |
| RandomForest | 57.914 | 1305.0 | ✓ |

CSVs: [inference_time.csv](../reports/06_model_selection/inference_time.csv), [model_size.csv](../reports/06_model_selection/model_size.csv).

**All six models pass the 100 ms PROJECT_ADVICE.md target by a wide margin.** That isn't surprising — single-window inference on a 16-feature input is cheap regardless of model class. The interesting comparison is among the top-three by F1:

- **HGB**: 10 ms, 430 KB. Default-fastest of the sklearn boosters.
- **LightGBM**: 1.4 ms, 411 KB. **7× faster than HGB** with very similar F1.
- **XGBoost**: 0.6 ms, 312 KB. **15× faster than HGB**, smaller pickle, but F1 drops 0.015.

For a wrist-worn embedded device (HealthyPi Move target), the 10 ms / 1.4 ms / 0.6 ms differences are immaterial — all are sub-perceptual on a 60 s prediction cadence. **Battery-life difference is also negligible** at this latency scale. Pickle size matters slightly more (smaller = less flash usage), but 300–500 KB is trivial on any modern MCU.

So there's no decisive deployability winner among the top three; all three are deployable as drop-in replacements for one another. The choice comes down to F1 + per-subject robustness.

---

## 4. Threshold sweep on the chosen model (HGB)

CSV: [threshold_sweep_HGB.csv](../reports/06_model_selection/threshold_sweep_HGB.csv).
Plot: [threshold_sweep_HGB.png](../reports/06_model_selection/threshold_sweep_HGB.png).

The default decision threshold of 0.5 was kept against the F1-optimal alternative of 0.45. Detail:

- **Default 0.5**: F1 = 0.931 (per-fold mean), recall = 0.911, precision = 0.972
- **F1-optimal 0.45**: F1 = 0.940 (OOF aggregate), recall = 0.916, precision = 0.966

The 0.009 F1 improvement from 0.45 over 0.5 is real but is computed on out-of-fold concatenated predictions (single F1 across 691 windows), whereas the chosen model's headline F1 is computed per-fold and averaged. The two are slightly different metrics; on the per-fold-mean metric, 0.45 and 0.5 are statistically indistinguishable.

We default to **threshold = 0.5** for the senior-design lock. Reasons:
1. The improvement at 0.45 is small (≤ 0.01 F1) and computed on the same data we'd be using to choose the threshold — borderline overfitting to the test fold.
2. The default 0.5 is more defensible to the committee ("we used the standard cutoff") than "we tuned the threshold by sweeping on the held-out predictions, then picked the F1-max."
3. Phase 7 may hyperparameter-tune the model and produce a different F1-optimal threshold; locking 0.45 now would be premature.

The recall ≥ 0.85 rebuild plan target is met at every threshold tested in [0.05, 0.95] — see §5.

---

## 5. Why threshold tuning has limited leverage — bimodal verification

A subtle empirical finding worth documenting in detail because it matters for Phase 8 deployment.

The original threshold sweep showed F1 essentially flat at ~0.93 across thresholds [0.05, 0.95]. Cause: **HGB outputs an aggressively bimodal probability distribution.** Two diagnostic plots verify this:

### 5.1 Distribution of OOF predicted probabilities

Plot: [oof_proba_histogram_HGB.png](../reports/06_model_selection/oof_proba_histogram_HGB.png).

Counts of OOF predicted probabilities across all 691 windows:

| Range | Count | % | Interpretation |
|---|---:|---:|---|
| P < 0.001 | 476 | 68.9 % | confidently non-stress |
| P > 0.999 | 117 | 16.9 % | confidently stress |
| 0.001 ≤ P ≤ 0.999 | 98 | 14.2 % | threshold-sensitive |

86 % of windows have probabilities pinned to the floor or ceiling. Only **14 %** are in the middle region where the threshold actually changes the prediction. That explains the flat F1 sweep: the threshold can move within a sparsely-populated middle without changing classifications for most windows.

### 5.2 Extended threshold sweep verifying the textbook collapse

Plot: [threshold_sweep_extended_HGB.png](../reports/06_model_selection/threshold_sweep_extended_HGB.png).

Extending the sweep to include threshold = 0:

| threshold | F1 | recall | precision |
|---:|---:|---:|---:|
| 0.0000 | 0.364 | 1.000 | 0.223 |
| 0.0001 | 0.691 | 1.000 | 0.527 |
| 0.0010 | 0.829 | 0.994 | 0.712 |
| 0.0100 | 0.876 | 0.968 | 0.801 |
| 0.0250 | 0.899 | 0.955 | 0.850 |
| 0.0500 | 0.901 | 0.942 | 0.863 |
| 0.1000 | 0.920 | 0.935 | 0.906 |
| 0.1500 | 0.923 | 0.929 | 0.917 |

At threshold = 0, the model classifies all 691 windows as stress, giving recall = 1.0, precision = 0.223 (= the stress prevalence), F1 = 0.364. Textbook behavior. The model's logic is working correctly; the original sweep just didn't reach this regime because the lower bound was 0.05.

The dramatic transition happens between thresholds 0 and 0.05 — that's the band where the lower-mode probabilities (P ≈ 0.001–0.05) fall on the wrong side of the cutoff. Above 0.05 we're in the flat plateau.

### 5.3 What this tells us about deployment

**On 86% of windows, the model's prediction is essentially deterministic** — the threshold can move from 0.05 to 0.95 without changing the classification. Only ~14% of decisions are threshold-sensitive, and those probably represent legitimately ambiguous physiological states (mild arousal, transitions between conditions, motion-marginal windows).

This is a strong defense-doc claim:
> "Our model produces high-confidence predictions on >85% of windows, where classification is robust to threshold choice. The remaining 14% represent physiologically ambiguous moments where the device's decision rule should reflect a clinical preference for sensitivity vs. specificity — and even there, the recall ≥ 0.85 target is met by a comfortable safety margin."

This also explains why threshold tuning *cannot* recover S9 or other persistently-failing subjects: their misclassified windows aren't in the threshold-sensitive middle band, they're at probabilities < 0.001. **Their recall floor is structural, not threshold-related.**

---

## 6. Decision rationale: why HGB

Two-axis criteria from the rebuild plan: (a) accuracy / robustness, (b) deployability.

**On accuracy/robustness:**
- HGB tops F1 at 0.931 by 0.010 over LightGBM and 0.015 over XGBoost. The gap is real but small relative to fold-to-fold std.
- All three boosters tie on min subject recall (0.545, both fail S9).
- HGB has the highest single per-subject recall on S16 (1.000 vs LightGBM/XGBoost 0.909).

**On deployability:**
- HGB at 10 ms is well within the 100 ms target.
- HGB at 430 KB pickle size is trivial for any wearable MCU.
- LightGBM and XGBoost are 7× and 15× faster respectively — but the absolute differences are sub-perceptual at our 60 s prediction cadence.

**The choice:** HGB primary, LightGBM as the explicit deployment fallback.

This pick is consistent with V1's notebook 05 model-selection result (HGB also won there on V1 features) and the Phase 4 / Phase 5 V3 chain (HGB has been the comparator throughout). Locking HGB now means Phases 7 and 8 don't shift gears.

**LightGBM is named as the "embedded backup"** in case Phase 8 (real-time simulator) reveals a deployment constraint that HGB doesn't satisfy — most likely scenario: stricter inference latency, or pickle-format compatibility issues with the embedded device's runtime. LightGBM's native `Booster.save_model` produces a portable JSON / text format that's easier to ship to a non-Python runtime than sklearn's pickle. Worth keeping that option open until Phase 8.

---

## 7. What's locked vs. provisional after Phase 6

**Locked:**
- **Primary model: `HistGradientBoostingClassifier(max_iter=300, max_depth=4, learning_rate=0.05, class_weight='balanced', random_state=42)`** as defined in `Updated_Extraction_V3/eval_helpers.py:hgb_factory`.
- **Decision threshold: 0.5** (default).
- **Fallback model: LightGBM** (same `class_weight='balanced'`, `n_estimators=300, num_leaves=31, max_depth=4, learning_rate=0.05`).
- The chosen fitted model is saved to [reports/06_model_selection/chosen_model.pkl](../reports/06_model_selection/chosen_model.pkl) for Phase 7 to load.

**Provisional / open:**
- HGB hyperparameters (`max_iter`, `max_depth`, `learning_rate`, `min_samples_leaf`, `l2_regularization`) are V1's settings, not tuned for V3. Phase 7 will Optuna-tune them and probably gain another 0.01–0.02 F1.
- The S9 floor of 0.545 is a known limitation. Future-work options: (a) add subject-specific features in a Phase 4-extension, (b) per-subject calibration via wear-time recalibration window, (c) ensemble of HGB + SVM-RBF (since SVM did better on S3 and might do better on S9 too — worth a quick test in Phase 7).
- Threshold optimization is conservative now (default 0.5). Phase 8's real-time simulator should re-confirm the threshold on streaming data — if probability distributions shift in streaming vs. offline, the optimal threshold may differ.

---

## 8. Implications for Phase 7 (hyperparameter tuning)

- **Model class: HGB.** Optuna search space:
  - `max_iter`: 100 – 1000 (currently 300)
  - `max_depth`: 3 – 20 (currently 4)
  - `learning_rate`: 0.005 – 0.5 (currently 0.05)
  - `min_samples_leaf`: 1 – 50 (currently 20, sklearn default)
  - `l2_regularization`: 0 – 10 (currently 0)
  - `max_leaf_nodes`: 10 – 200 (currently 31, sklearn default)
- **Nested LOSO**: outer LOSO over the 14 subjects, inner k-fold over the 13 training subjects (NOT within-subject — the rebuild plan's hard rule). That gives clean unbiased performance estimates.
- **Trials**: 100+ as the rebuild plan specifies.
- **Save**: Optuna study object (so the search trajectory is interrogatable), hyperparameters JSON, fitted final model.

Phase 7 should also run a quick **HGB + SVM-RBF ensemble** as a side experiment — SVM was best on S3 (0.727 vs HGB's 0.636); if averaging predictions from the two models recovers S3 without losing F1 elsewhere, that's a deployment-relevant finding. Low cost (~30 minutes), high optionality.

---

## References

Standard library: scikit-learn 1.6.1 ([HistGradientBoosting](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html), [SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)), XGBoost ([Chen & Guestrin 2016](https://doi.org/10.1145/2939672.2939785)), LightGBM ([Ke et al. 2017](https://papers.nips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html)).

V1 reference: [docs/session_logs/2026-03-11_nb05_nb06_results_analysis.md](session_logs/2026-03-11_nb05_nb06_results_analysis.md) — V1's notebook 05 also picked HGB on V1's 138-feature dataset.

Project-internal: [docs/05_feature_selection.md](05_feature_selection.md) for the locked 16-feature set; [memory/project_s3_corrupted_baseline.md](../../../memory/project_s3_corrupted_baseline.md) for the S3 limitation.
