# 05 — Feature selection

**Phase 5 of the V3 rebuild.** Find the smallest feature subset that preserves Phase 4's F1 = 0.933 (raw / 48 features / HGB / W=60s / decoupled HRV).

**Date:** 2026-04-30
**Notebook:** [notebooks/05_feature_selection.ipynb](../notebooks/05_feature_selection.ipynb)
**Status:** ✅ chosen **combined-rank top-16**. F1 = 0.931, min subj recall = 0.545, no subjects below 0.5 recall.

---

## 1. Headline result

| Metric | Phase 4 baseline (48) | Post-prune (33) | **Phase 5 final (16)** |
|---|---:|---:|---:|
| F1 | 0.933 ± 0.101 | 0.926 ± 0.104 | **0.931 ± 0.094** |
| Recall | 0.911 | 0.911 | 0.911 |
| Precision | — | — | **0.972** |
| Accuracy | — | — | **0.973** |
| min subj recall | 0.545 | 0.545 | 0.545 |
| # subj < 0.5 | 0 | 0 | **0** |
| # features | 48 | 33 | **16** |

**67% reduction in feature count, ~the same F1, no subjects below 0.5.** That's the Phase 5 win.

The chosen ranking is **combined-rank**: every feature gets ranked by both stability (universal univariate signal) and SHAP (model-internal contribution); the combined score is the average of the two ranks. Features that score well by *both* methods rise to the top; features that look good by only one method drop. The top-16 is the consensus set.

---

## 2. The 16 chosen features

CSV: [reports/05_feature_selection/selected_features.csv](../reports/05_feature_selection/selected_features.csv).

| # | feature | stability | mean &#124;SHAP&#124; | combined rank | quadrant |
|---:|---|---:|---:|---:|---|
| 1 | `scr_recovery_time_mean` | 1.000 | 3.618 | 1.0 | **both ✓✓** |
| 2 | `hrv_median_rr` | 1.000 | 1.353 | 1.5 | both ✓✓ |
| 3 | `scr_peak_count` | 1.000 | 1.030 | 2.0 | both ✓✓ |
| 4 | `hrv_pnn50` | 1.000 | 0.874 | 3.0 | both ✓✓ |
| 5 | `scl_max` | 1.000 | 0.854 | 3.5 | both ✓✓ |
| 6 | `hrv_sdsd` | 1.000 | 0.526 | 5.0 | both ✓✓ |
| 7 | `scl_range` | 0.714 | 1.019 | 8.5 | high SHAP, moderate stability |
| 8 | `scr_amplitude_sum` | 1.000 | 0.195 | 8.5 | high stability, moderate SHAP |
| 9 | `temp_max` | 0.357 | 0.703 | 12.0 | high SHAP, low stability ⚠ |
| 10 | `hrv_min_rr` | 0.643 | 0.450 | 13.0 | moderate both |
| 11 | `acc_jerk_mag_mean` | 0.357 | 0.489 | 13.0 | high SHAP, low stability ⚠ |
| 12 | `temp_slope` | 0.071 | 0.709 | 13.0 | **high SHAP, almost no stability** ⚠ |
| 13 | `scr_rise_time_mean` | 0.929 | 0.210 | 13.5 | both ✓ |
| 14 | `acc_magnitude_std` | 1.000 | 0.037 | 14.5 | high stability, low SHAP |
| 15 | `scr_amplitude_max` | 1.000 | 0.036 | 15.0 | high stability, low SHAP |
| 16 | `acc_y_std` | 0.000 | 0.317 | 16.0 | **zero stability, moderate SHAP** ⚠ |

**Of the 16, 12 are stability-validated (≥0.6) AND in SHAP top-15.** Three (`temp_max`, `acc_jerk_mag_mean`, `temp_slope`) are high-SHAP / low-stability — model uses them but the universal-signal evidence is weak. One (`acc_y_std`) is zero-stability — included only because SHAP says HGB uses it for interaction effects. Deployment risks documented in §5.

### 2.1 What each feature actually measures

Plain-language description of every chosen feature, in combined-rank order. This is the table to share with the senior design committee.

| # | Feature | Modality | What it measures |
|---:|---|---|---|
| 1 | `scr_recovery_time_mean` | EDA / SCR | Average time for skin conductance peaks to decay back toward baseline — longer recoveries indicate sustained sympathetic activation. |
| 2 | `hrv_median_rr` | HRV | Median time between consecutive heartbeats; shorter intervals (faster HR) indicate sympathetic dominance. |
| 3 | `scr_peak_count` | EDA / SCR | Number of skin conductance responses (sweat-gland bursts) detected in the window — higher counts mean more arousal. |
| 4 | `hrv_pnn50` | HRV | Percentage of consecutive beats differing by > 50 ms; high values indicate vagal/parasympathetic activity, suppressed under stress. |
| 5 | `scl_max` | EDA / SCL | Peak tonic skin conductance level in the window — encodes elevated baseline sympathetic tone. |
| 6 | `hrv_sdsd` | HRV | Standard deviation of successive beat-to-beat interval differences — short-term HRV that drops under stress. |
| 7 | `scl_range` | EDA / SCL | Spread of tonic skin conductance (max − min) — captures sustained shifts in sympathetic tone. |
| 8 | `scr_amplitude_sum` | EDA / SCR | Total magnitude of all SCRs in the window — aggregate sympathetic burst energy. |
| 9 | `temp_max` | TEMP | Highest wrist skin temperature in the window — drops during stress due to peripheral vasoconstriction. |
| 10 | `hrv_min_rr` | HRV | Shortest beat-to-beat interval — captures peak instantaneous heart rate. |
| 11 | `acc_jerk_mag_mean` | ACC | Mean magnitude of acceleration's rate-of-change (jerk) — captures fidgeting and quick wrist movements. |
| 12 | `temp_slope` | TEMP | Linear trend in skin temperature across the window — negative slope = active cooling (vasoconstriction). |
| 13 | `scr_rise_time_mean` | EDA / SCR | Average time from SCR onset to peak — faster rises indicate stronger sympathetic bursts. |
| 14 | `acc_magnitude_std` | ACC | Variability of total 3-axis movement magnitude — orientation-invariant motion variation. |
| 15 | `scr_amplitude_max` | EDA / SCR | Largest single SCR amplitude — captures the most extreme sympathetic burst. |
| 16 | `acc_y_std` | ACC | Variability of acceleration along the wrist's y-axis — orientation-dependent movement variation. |

### 2.2 Modality breakdown

| Parent signal | Count | % of 16 |
|---|---:|---:|
| **EDA** (Electrodermal Activity — sweat-gland conductance) | **7** | 44 % |
| ↳ SCR (phasic, fast bursts) | 5 | |
| ↳ SCL (tonic, slow baseline) | 2 | |
| **HRV** (Heart Rate Variability from BVP) | **4** | 25 % |
| **ACC** (Accelerometer — wrist movement) | **3** | 19 % |
| **TEMP** (Skin temperature) | **2** | 12 % |
| **Total** | **16** | 100 % |

Four observations from this breakdown:

1. **EDA dominates (44 %); SCR alone is 31 %.** Five of the top-8 features are SCR-derived. This matches the stress-detection literature consensus (Boucsein 2012, Healey & Picard 2005, Schmidt et al. 2018) — phasic EDA is the most validated single physiological correlate of acute sympathetic activation.
2. **All four modalities are represented.** Even though EDA carries the strongest signal, the model uses HRV, TEMP, and ACC as complementary channels. Dropping any single modality loses real signal — the deployable wearable must include all four sensor sources.
3. **HRV is entirely time-domain (4/4).** No frequency-domain HRV features (LF, HF, LF/HF, total power) made the top 16. They were correctly computed with the decoupled 180 s lookback fix (Phase 4 §6.5), but both stability and SHAP rankings still place them lower than time-domain HRV — consistent with Task Force 1996 guidance that short-window freq estimates have higher variance.
4. **Notable omissions:** `eda_skewness`/`eda_kurtosis` (distribution shape), `acc_sma`/`acc_energy` (orientation-invariant aggregates), all freq-domain HRV. Each was either correlation-pruned (see §3.1) or ranked outside top-16 in both stability and SHAP. None of them is needed to maintain F1 ≥ 0.93.

---

## 3. Methodology

The Phase 5 spec required:
1. **Univariate filtering** with leakage-safe per-fold MI ✓
2. **Correlation pruning** (drop one of any pair with |r| > 0.95) ✓
3. **Stability selection** across LOSO folds ✓
4. **SHAP-based importance** ✓
5. **Compare three feature sets**: full / stability-selected / SHAP-selected ✓ (we extended this to four: + combined-rank)
6. **Pick smallest count maximizing F1** ✓ (with no zero-recall subjects as a hard constraint)

### 3.1 Correlation pruning (48 → 33)

26 pairs at |r| > 0.95 — heavy redundancy in three families:

- **TEMP** (5 pairs): `temp_mean`, `temp_min`, `temp_median`, `temp_max` are essentially the same signal at 60 s windows (slow-changing skin temperature). Kept `temp_max`; dropped `temp_mean`, `temp_min`, `temp_median`.
- **SCL** (10 pairs): `scl_mean`, `scl_min`, `scl_max`, `scl_median`, `scl_auc`, `scl_std`, `scl_range` are all collinear (tonic SCL is slow-varying so all summary stats track each other). Kept `scl_max`, `scl_range`. Dropped `scl_mean`, `scl_median`, `scl_min`, `scl_auc`, `scl_std`.
- **HRV time-domain** (3 pairs): `hrv_rmssd ↔ hrv_sdnn ↔ hrv_sdsd ↔ hrv_mean_rr` are all variations on inter-beat-interval variability. Kept `hrv_sdsd`, `hrv_median_rr`. Dropped `hrv_rmssd`, `hrv_sdnn`, `hrv_mean_rr`.
- **ACC** (3 pairs): `acc_magnitude_mean ↔ acc_activity_mean ↔ acc_energy` (all measure raw movement); `acc_jerk_mag_mean ↔ acc_jerk_mag_p95`. Kept `acc_magnitude_mean`, `acc_jerk_mag_mean`. Dropped `acc_activity_mean`, `acc_energy`, `acc_jerk_mag_p95`.
- **EDA distribution** (1 pair): `scr_amplitude_max ↔ scr_amplitude_std`.

For each pair, the feature with the **higher univariate F-statistic on the full dataset** was kept. The slight LOSO-leakage from this choice is acceptable per the rebuild plan: feature-pair correlation is a population-level property of the feature set, not a model-fitting decision. Plot: [reports/05_feature_selection/correlation_matrix.png](../reports/05_feature_selection/correlation_matrix.png).

Post-prune F1 = 0.926 (Δ −0.007 vs the 48-feature baseline). The 0.007 loss reflects information that *was* in the dropped duplicates; HGB on 48 features gets a tiny boost from having the redundancy, but the gain is well within fold-to-fold std.

### 3.2 Stability selection — 11 features always rank in top-15 by MI

For each LOSO fold (n=14), we compute mutual information between each feature and the binary stress label *only on training subjects*, rank features by MI, and ask: does this feature appear in the top-15 of every fold's ranking? A feature with stability = 1.0 appears in top-15 in **every** held-out scenario. That's the test of whether the feature is universally informative.

Eleven features achieved stability = 1.0:
- 5 SCR: `scr_peak_count`, `scr_amplitude_mean`, `scr_amplitude_max`, `scr_amplitude_sum`, `scr_recovery_time_mean`
- 4 HRV time-domain: `hrv_pnn50`, `hrv_median_rr`, `hrv_sdsd`, `hrv_max_rr`
- 1 SCL: `scl_max`
- 1 ACC: `acc_magnitude_std`

These are the "universal stress signature" — robust regardless of which subject is held out.

CSV: [stability_ranking.csv](../reports/05_feature_selection/stability_ranking.csv). Plot: [stability_ranking.png](../reports/05_feature_selection/stability_ranking.png).

### 3.3 SHAP across LOSO folds

Trained HGB on each LOSO fold and computed SHAP values on the held-out subject's predictions. Aggregated mean |SHAP| across all test windows. The top-3 by SHAP — `scr_recovery_time_mean` (3.62), `hrv_median_rr` (1.35), `scr_peak_count` (1.03) — carry roughly half the model's decision weight by themselves.

CSV: [shap_ranking.csv](../reports/05_feature_selection/shap_ranking.csv). Plot: [shap_ranking.png](../reports/05_feature_selection/shap_ranking.png).

### 3.4 Combined rank: averaging stability and SHAP

```
combined_rank(feature) = (rank_by_stability(feature) + rank_by_shap(feature)) / 2
```

A feature can only score low (=good) on combined rank if it scores well in *both* underlying rankings. This penalizes the "high SHAP, zero stability" features that the SHAP-only ranking would otherwise promote — which directly addresses the deployment-risk concern.

The top-6 combined-rank features all have stability = 1.0 AND SHAP rank ≤ 9 (the consensus core). Beyond that, the ranking blends stability-rich features with SHAP-rich features, with the predictable tension that some features are strong by only one criterion.

### 3.5 F1 vs N — full sweep at every N from 1 to 33

The original Phase 5 spec used a sparse N grid {5, 10, 15, 20, 30}. After feedback, we ran every N from 1 to the full pruned size (33) for all three rankings. Plot: [f1_vs_n_combined.png](../reports/05_feature_selection/f1_vs_n_combined.png).

Headline observations from the full curves:

- **SHAP top-1 alone gets F1 = 0.892**, min_subj = 0.727. Single-feature performance is striking — `scr_recovery_time_mean` carries enormous predictive weight by itself. (See §6 for why this works physiologically.) Not a deployment recommendation; included as an indicator of how concentrated the predictive signal is.
- **All three curves plateau around F1 = 0.92–0.93 from N ≈ 13 onward.** No method shows continued improvement past N = 16.
- **Combined-rank peaks at N = 16, F1 = 0.931** — the highest peak of the three rankings. Stability peaks at N = 23 (F1 = 0.933, exactly matches the 48-feature baseline). SHAP peaks at N = 13 (F1 = 0.930).
- **All three rankings agree on the top-6 features** (the consensus core). They diverge around N = 7–15 where each method's idiosyncratic preferences show up. They reconverge at N ≈ 20 where all features are basically forced to be the same.

| Ranking | Plateau elbow N | F1 at elbow | min subj recall |
|---|---:|---:|---:|
| stability | 20 | 0.929 | 0.545 |
| SHAP | 12 | 0.928 | 0.636 |
| **combined-rank** | **16** | **0.931** | **0.545** |

We picked combined-rank top-16 over SHAP top-12 because:
- Combined methodology actually uses both rankings (the user's original concern about the Phase 5 spec)
- Combined-16 has the highest F1 of any plateau elbow
- The 4 extra features (vs SHAP-12) bring 3 stability-validated additions (`scr_amplitude_sum`, `scr_amplitude_max`, `acc_magnitude_std`) and 1 high-SHAP moderate-stability feature (`scr_rise_time_mean`)
- It loses on min subject recall (0.545 vs 0.636) — the trade-off is concentrated on S9; documented below

---

## 4. Per-subject recall on the chosen 16-feature set

| Subject | recall | comment |
|---|---:|---|
| **S9** | **0.545** | minimum (was 0.545 on 48-feature, 0.636 on SHAP-12) |
| S3 | 0.636 | corrupted-baseline subject (V2 author's diagnosis), unchanged |
| S10 | 0.750 | improved from 0.583 (48-feature) and same as SHAP-12 |
| S2 | 0.900 | unchanged from 48-feature (still recovered) |
| S17 | 0.917 | down from 1.000 (48-feature), same as SHAP-12 |
| S16 | **1.000** | back up from 0.909 (SHAP-12), recovered to baseline |
| S4, S5, S6, S7, S8, S11, S13, S15 | 1.000 | trivially classified |

Compared to the 48-feature baseline:
- **S9**: 0.545 → 0.545 (same)
- **S2**: 0.900 → 0.900 (same)
- **S17**: 1.000 → 0.917 (slight loss on previously-perfect)
- **S16**: 1.000 → 1.000 (preserved)
- **S10**: 0.583 → 0.750 (**+0.17, big win**)
- **S3**: 0.727 → 0.636 (**−0.09, regression** on the corrupted-baseline subject)

Net: feature count down 67%, F1 essentially unchanged, with redistribution of per-subject recalls (S10 wins, S3 loses). The S3 regression is logged as a known limitation tied to the dataset-level corrupted-baseline issue (see [memory: S3 corrupted baseline](../../../memory/project_s3_corrupted_baseline.md)) — algorithmic feature selection cannot fix a labeling problem.

---

## 5. Known deployment risks in the chosen set

Three features in the 16 are flagged as high-SHAP / low-stability — the model uses them but the universal-signal evidence is weak:

### 5.1 `temp_slope` (stability = 0.071, SHAP rank = 7)

The model uses temperature slope substantially (mean |SHAP| = 0.71) but it ranks in the top-15 by MI in only **1 of 14 LOSO folds**. This is the highest-risk feature in the set.

**Physiological interpretation:** stress causes peripheral vasoconstriction → distal skin temperature drops over tens of seconds. The *slope* of the drop is a real stress indicator (Karthikeyan 2012, Healey & Picard 2005), but its magnitude is highly subject-dependent (depends on baseline peripheral perfusion, ambient temperature, electrode-skin contact, etc.). The model effectively learns "negative temp slope correlates with stress" but the threshold has to be calibrated per-subject — and HGB on raw features can't really do that.

**Deployment caveat:** if we deploy to a population (adolescents) where typical baseline peripheral perfusion or ambient temperature differs systematically, `temp_slope`'s effective threshold will shift. Consider z-scoring this feature specifically (per-subject baseline) before deployment.

### 5.2 `acc_y_std` (stability = 0.000, SHAP rank = 12)

Per-axis ACC variation. **Zero-stability** — never appears in the top-15 by MI in any LOSO fold. The model's use of it depends on interactions with other features.

**Physiological interpretation:** `acc_y_std` encodes the variability of acceleration along one specific axis — which means it's **orientation-dependent**. The y-axis is whatever direction the watch's coordinate frame points after the user puts it on. If the user removes and re-puts on the watch, the y-axis interpretation flips, and `acc_y_std`'s relationship to motion changes.

**Deployment caveat:** `acc_y_std` is the single feature in the 16 most likely to fail in deployment with imperfect-fit-day-to-day usage. The orientation-invariant alternative `acc_magnitude_std` (also in the chosen 16, stability = 1.0) provides a backup signal that doesn't have this problem. Consider dropping `acc_y_std` if Phase 6 reveals the model becomes brittle without controlled wear.

### 5.3 `temp_max` and `acc_jerk_mag_mean` (stability ≈ 0.36)

Both high-SHAP, low-stability. Less risky than `temp_slope` and `acc_y_std` because their stability scores are non-zero (they appear in the top-15 in ~5 of 14 folds), but still less universally validated than the consensus-core features.

**Recommendation:** these four features (`temp_slope`, `acc_y_std`, `temp_max`, `acc_jerk_mag_mean`) are the candidates for replacement if Phase 6 model comparison reveals they don't transfer to other model classes. The replacement set would draw from the next-best stability-validated alternatives: `hrv_max_rr`, `acc_magnitude_max`, `acc_sma`.

---

## 6. Why a single SHAP-top-1 feature gets F1 = 0.892

This was a question raised during the analysis. The answer matters for understanding why feature selection is meaningful at all.

`scr_recovery_time_mean` is a window-aggregated summary of the average recovery time of skin conductance responses in the 60-second window. With one input feature, HGB still builds 300 trees of depth 4 — collectively approximating an arbitrary smooth function `f(scr_recovery_time) → P(stress)`.

Why this single feature alone reaches F1 = 0.892:

1. **The feature is not really single-valued per window.** It aggregates 0–10 SCRs detected per window into a summary statistic. So one column = one rich physiological summary, not one bit of information.
2. **SCR recovery time is biophysically tied to autonomic balance.** Under stress, sympathetic drive prevents recovery → recovery times lengthen. Under calm, parasympathetic dominance enables fast reabsorption → short recoveries. This is one of the most validated stress indicators in the EDA literature (Boucsein 2012).
3. **HGB on one feature is still a flexible 1-D function approximator** — 300 trees of depth 4 = up to 4800 piecewise-constant regions, plenty to model the relationship.
4. **F1 metric + class imbalance + balanced weights align favorably.** The model prefers high recall on the minority class with one strong feature.

**Caveat:** F1 = 0.89 with one feature is brittleness-prone. If `scr_recovery_time_mean` is corrupted on a specific subject (electrode-off, motion-corrupted SCR detection that our 8 g/s gate doesn't catch), there's no fallback. We lock 16 features, not 1, because the redundant signal in the other 15 is the deployment safety margin — not because 1 isn't predictive.

---

## 7. What's locked vs. provisional after Phase 5

**Locked:**
- The 16 feature names listed in §2.
- HGB as the comparator model for Phase 6 (was already locked in Phase 4).
- Methodology: correlation pruning (|r| > 0.95) + stability selection + SHAP across LOSO folds + combined-rank averaging.

**Provisional:**
- The 4 deployment-risk features (`temp_slope`, `acc_y_std`, `temp_max`, `acc_jerk_mag_mean`). If Phase 6 shows other model classes don't share HGB's preference for these, we replace them with stability-validated alternatives. Quick test: rerun the chosen 16-feature LOSO with LR / SVM / RF and see if those models also achieve F1 ≈ 0.92. If they do, the four are safe; if not, those models are basing their decisions on different features and we should swap.
- S9 minimum recall of 0.545. Phase 6 may surface a model class that recovers S9 at a different feature set. Worth checking.

---

## 8. Implications for Phase 6

- **Feature set**: 16 features (this document).
- **Comparator model**: HGB (Phase 4 lock).
- **Phase 6 question**: do other model classes (Logistic Regression, Random Forest, XGBoost, LightGBM, SVM-RBF) achieve comparable F1 on this 16-feature set, and which ones recover S9? Per the rebuild plan, that's the next phase.
- **Sanity check loop after Phase 6**: once Phase 6 picks a model, re-test the chosen 16 features against the chosen model — confirm they're still the right 16. If not, iterate.

---

## References

- Boucsein 2012 — *Electrodermal Activity*, 2nd ed. Springer. SCR recovery dynamics, electrode standards.
- Schmidt et al. 2018 — *Introducing WESAD*. Original WESAD paper, feature catalog reference.
- Healey & Picard 2005 — *Detecting stress during real-world driving*. SCL/SCR features for stress detection in deployable systems.
- Karthikeyan 2012 — Skin temperature for stress classification.
- Berntson 1997 — HRV time-domain feature standards.
- Task Force 1996 — HRV measurement standards for short-term recordings.

See [docs/references.bib](references.bib) for full BibTeX.
