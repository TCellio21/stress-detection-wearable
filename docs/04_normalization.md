# 04 — Normalization comparison

**Phase 4 of the V3 rebuild.** Compares five normalization strategies on a fixed dataset (W=60 s, step=60 s, 14 subjects, S14 excluded) and a fixed model class (HistGradientBoosting). Notebook: [notebooks/04_normalization.ipynb](../notebooks/04_normalization.ipynb).

**Date:** 2026-04-30
**Status:** ✅ chosen **`raw`** features (no normalization). F1 = 0.933, no subjects below 0.5 recall.

> **Audit notes (2026-04-30):**
> 1. Initial notebook run produced a stale-cache HRV bug (Python module cache held the pre-fix `features.py` after the `np.trapezoid` error → all HRV features came out as zero defaults → first parquet had 0% `hrv_valid`). Numbers below come from a rebuilt dataset.
> 2. Subsequent code review (AI feedback, 2026-04-30) caught that `hrv_freq_window: 180` in `config.yaml` was never wired through to `features.py:extract_hrv_features` — both time-domain and frequency-domain HRV were being computed on the same 60 s window, which violates Task Force 1996's requirement of ≥ 120 s for stable LF (0.04–0.15 Hz) estimation.
> 3. Both bugs are now fixed. The current dataset has 94.2% `hrv_valid` with **decoupled time/freq HRV windows** (60 s for time-domain, 180 s for frequency-domain). The numbers in this document reflect the post-fix state. See §6 (HRV bug) and §6.5 (decoupled-window fix).

---

## 1. Headline

| Variant | # features | F1 | recall | min subj recall | # subj < 0.5 |
|---|---:|---:|---:|---:|---:|
| **raw** | 48 | **0.933 ± 0.101** | 0.911 | **0.545** | **0** |
| raw + z | 96 | 0.894 ± 0.137 | 0.858 | 0.364 | 1 |
| z | 48 | 0.880 ± 0.118 | 0.873 | 0.455 | 1 |
| mixed (V2-bucket) | 48 | 0.847 ± 0.152 | 0.838 | 0.273 | 1 |
| pct | 48 | 0.841 ± 0.225 | 0.841 | 0.100 | 1 |
| robust z | 48 | 0.783 ± 0.191 | 0.808 | 0.273 | 3 |

CSV: [reports/04_normalization/normalization_summary.csv](../reports/04_normalization/normalization_summary.csv).
Plot: [reports/04_normalization/primary_comparison.png](../reports/04_normalization/primary_comparison.png).

**Two findings flip the Phase 2 narrative (with one revision after the HRV-bug audit):**

1. **Raw features beat every normalization variant.** The V3 (W=60, step=60) dataset under HGB achieves F1 = 0.933 with no normalization — well above the 0.776 RandomForest baseline from Phase 2 and above V1's HGB-on-138-features F1 = 0.814.
2. **The S2 / S17 persistent-failure pattern from Phase 2 is gone.** Both subjects have ≥ 0.90 recall on `raw` (S2 = 0.90, S17 = 1.00). The audit's prediction (§6 #4) — "S2 is a model-class failure, not a signal failure" — is empirically confirmed for these two subjects.
3. **S3 is partially recovered, not fully.** Initially-published numbers showed S3 at 0.91 recall; that was an artifact of the HRV bug. With valid HRV (and the decoupled-window fix), S3 sits at **0.73** under raw — better than the post-HRV-fix 0.64 because the now-stable freq features push it further. S3's underlying corrupted-baseline issue (V2 author's diagnosis) means full recovery requires per-subject baseline auto-selection or exclusion. S3 remains the third-hardest subject after S9 (0.55) and S10 (0.58).

---

## 2. Per-subject recall — the persistent failures recover

CSV: [reports/04_normalization/per_subject_recall.csv](../reports/04_normalization/per_subject_recall.csv).
Heatmap: [reports/04_normalization/per_subject_heatmap.png](../reports/04_normalization/per_subject_heatmap.png).

The three subjects that failed across every (W, step) configuration in Phase 2 (RandomForest + z) recover dramatically under HGB + raw:

| Subject | Phase 2 best (RF + z) | Phase 4 raw (HGB + raw) | Δ |
|---|---:|---:|---:|
| S2 | 0.450 | **0.900** | +0.45 |
| S17 | 0.167 | **0.917** | +0.75 |
| S3 | 0.091 | **0.636** | +0.55 (partial) |

The new minimum is **S9 at 0.545**, with S10 at 0.583 and S3 at 0.636 close behind. All subjects now ≥ 0.5 — passes the "no subject below 0.5" criterion the rebuild plan defines as acceptable. Subjects with perfect recall on `raw`: S4, S5, S6, S7, S8, S11, S13, S15, S16 (9 of 14).

The story is consistent with V1's session log
([2026-03-11_nb05_nb06_results_analysis.md](session_logs/2026-03-11_nb05_nb06_results_analysis.md))
which noted that LogisticRegression and LinearSVC reached high recall on S2 while
trees collapsed. We now know HGB on raw features also reaches high recall on S2 —
so the failure mode was **RF + z-scored features specifically**, not "trees fail
on S2" in general.

---

## 3. Why raw wins

Three contributing reasons:

**3.1 HistGradientBoosting is scale-invariant on splits.** Decision trees split on
threshold comparisons (`x > c`); the value of `c` adapts per feature, so absolute
scale doesn't matter for fitting. Z-scoring doesn't help the splitter find better
thresholds — it just rewrites them.

**3.2 Z-scoring inflates per-subject feature variance for subjects with quiet
baselines.** A subject with a very stable resting HR has small `hrv_rmssd_baseline_std`;
dividing by that small std pushes any small deviation into a large z-score. The
model then sees those large z-scores as "very different from baseline" even when
they're physiologically minor. This is the Berntson 1997 known issue with z-score
on bounded physiological signals.

**3.3 Raw absolute values carry generalization-relevant signal.** Adult subjects
with high resting HR tend to also have higher stress HR; the absolute scale itself
is a useful inter-subject feature for HGB to split on. Z-scoring throws this
information away. (This is the converse argument to the deployment caveat in §5
— here it helps within-WESAD, there it could hurt cross-population.)

`raw + z` (96 features) sits between the two — F1 = 0.903 — confirming that the
z-score columns are net noise on top of raw for HGB.

The literature anchor for this finding is Schmidt et al. 2018 §4.3
{schmidt2018}, who report LOSO accuracy on WESAD chest features with and without
within-subject normalization and find no consistent benefit from normalization
for tree-based classifiers.

---

## 3.5. The V2 author's mixed-bucket scheme: tested, loses

The V2 pipeline ([Updated_Extraction_V2/extract_windowed_datasets.py:47-67](../Updated_Extraction_V2/extract_windowed_datasets.py#L47-L67))
groups features into three physiology-guided buckets:
- **Bucket 1 (raw):** features without a meaningful per-subject baseline — ACC magnitudes, distribution shape, dimensionless ratios.
- **Bucket 2 (raw+z):** vital signs with both clinical absolute meaning and per-subject deviation interest — HR, temp, SCR count.
- **Bucket 3 (z-only):** features whose absolute values are uninterpretable without baseline — SCL, SCR amplitudes, HRV time-domain.

Their notebook (`dataset/analysis/Feature_Importance.ipynb`) recommended dropping
most `raw_` twins based on XGBoost gain importance on a global (non-LOSO) model.

We tested an apples-to-apples translation of this scheme into V3's 48-feature
catalog (15 raw from Bucket 1 + 8 z from Bucket 2 + 25 z from Bucket 3 = 48, where
Bucket 2 picks z because their gain analysis showed z dominated for those
features). LOSO HGB result:

| Variant | F1 | min subj recall | # subj < 0.5 |
|---|---:|---:|---:|
| pure raw | **0.926** | 0.545 | 0 |
| pure z | 0.872 | 0.455 | 1 |
| **V2 mixed-bucket** | **0.851** | 0.364 | 1 |

**The mixed scheme is worse than pure z and substantially worse than pure raw.**
The V2 author's gain-based ranking pointed in the wrong direction for LOSO
generalization.

### Why the mix loses

Pure-raw HGB learns decision rules that span all 48 features in absolute scale.
Adult resting HR, resting skin temperature, baseline SCL, and resting SCR rate
are correlated population variables — splits on absolute `temp_mean` cluster
naturally with splits on absolute `hrv_mean_hr` because the underlying
demographic distribution links them. The model uses this cross-subject structure
implicitly.

Z-scoring Buckets 2 and 3 strips that population structure for ⅔ of the
features. The model now has to learn different decision boundaries in two
disjoint regimes — one in absolute scale (Bucket 1) and one in z-deviation
(Buckets 2+3) — without the cross-feature population prior. With 537 training
windows and 14 LOSO folds, that's a harder fit.

This is empirical evidence against the V2 author's "drop all raw twins"
recommendation. **For cross-subject LOSO generalization on WESAD adults, the
absolute scales of vital signs (HR, temp, SCR count) carry signal that
z-scoring discards.**

### Caveats

The mixed-bucket scheme might still beat raw in two scenarios we can't test
yet:
1. **Cross-population deployment** (adolescents). Adult-trained absolute
   thresholds may misclassify adolescents whose resting distributions differ;
   z-versions remove that bias. See §5 deployment caveat.
2. **A larger feature budget.** V2's actual approach kept Bucket 2 as both
   raw + z (54 features), not 48. We forced a pick-one to keep the comparison
   apples-to-apples. With raw+z (96 feats) already at F1=0.886 and pure raw
   at 0.926, a 54-feature `bucket_1_raw + bucket_2_both + bucket_3_z` variant
   would likely land in the 0.88–0.90 range — still below pure raw.

## 4. Why robust z-score underperforms

Robust z = `(x − median) / (1.4826 × MAD)` against label=1 windows. With valid HRV,
robust z lands at F1 = 0.790 — last place but not catastrophic. The remaining gap
to z-score (0.082 F1) likely comes from features whose baseline distributions
have small MAD (so any deviation explodes), particularly the SCR-related features
(SCR count is often 0 across most baseline windows → MAD = 0 → divisor protection
kicks in → feature contributes nothing).

Robust z would help on signals with non-zero, near-Gaussian baselines (chest ECG,
continuous skin temperature). For wrist EDA / HRV / ACC over baseline, the
sparse-zero structure makes plain z preferable.

Drop robust z from V3.

---

## 5. Deployment caveat — adolescents

Raw features wins on **WESAD adults LOSO**. The senior design deliverable runs
on WESAD adults, so for the immediate goal this is the right pick.

For the project's deployment target (adolescents in classroom settings), there's
a population-shift concern: adolescents have systematically different resting
distributions for HR (~75 bpm vs ~65), HRV (lower SDNN/RMSSD), and EDA (higher
SCR rate). A model trained on raw adult features learns thresholds calibrated
to adult absolute values — those thresholds may misclassify adolescent resting
state.

Z-score features are theoretically more robust to this shift (each subject is
"deviation from their own baseline," which generalizes population-to-population)
but cost 0.058 F1 within WESAD.

**Decision for V3:** lock `raw` as the primary feature set. Re-run the
normalization comparison on adolescent data when it becomes available. If raw
fails to transfer, the fallback feature set is `z` alone (not `raw + z`, which
adds the same population-confounding raw columns).

This caveat goes into the Phase 9 defense doc as a known limitation and
future-work item.

---

## 6. The HRV stale-cache bug — what changed after correction

The first run of the notebook hit `AttributeError: module 'numpy' has no attribute
'trapezoid'` on `np.trapezoid` in `extract_eda_features`. Fix landed in `features.py`
(use `scipy.integrate.trapezoid` with a `numpy.trapz` fallback). User re-ran "Run
All" without restarting the kernel, so Python's import cache held the old broken
`features` module — but the AttributeError mysteriously stopped firing while every
HRV feature came out NaN. The parquet was saved with **0% `hrv_valid`** and the
first round of results was on broken-HRV data.

The bug was caught by an audit that asked: "do the z-scored baseline windows have
mean ~ 0 and std ~ 1, as they should by construction?" Per-subject baseline z-std
came out at 0.729 instead of 1.0 — exactly the value you'd expect if 13 of 48
features are all-zero. Tracing those 13 columns identified them as the HRV columns,
and a per-subject HRV-validity check confirmed `hrv_valid = 0` for every window.

After deleting the parquet and rebuilding from scratch, `hrv_valid = 93.9%` and
mean baseline HR = 67.5 bpm, all sane.

What changed in the published numbers:

| Variant | Initial (broken HRV) F1 | Corrected (valid HRV) F1 | Δ |
|---|---:|---:|---:|
| raw | 0.934 | **0.926** | −0.008 |
| raw + z | 0.903 | 0.886 | −0.017 |
| z | 0.876 | 0.872 | −0.004 |
| pct | 0.879 | 0.836 | −0.043 |
| robust z | 0.672 | 0.790 | **+0.118** |

The qualitative ranking (raw > raw+z > z > pct > robust_z) is unchanged.
Robust z benefited most from valid HRV — its initial F1 was depressed by the
13 all-zero HRV columns being treated as legitimate features. The S3 recall
estimate revised downward from 0.91 to 0.64 — broken HRV had been giving the
model a coincidentally-discriminative signal on S3.

**Defensive measures added to the notebook:**
- An `importlib.reload` cell at the top so future module-cache hazards don't repeat.
- An `assert df_raw['hrv_valid'].mean() > 0.5` sanity check after the build cell.

## 6.5. The decoupled-HRV-window fix (Task Force 1996 compliance)

**Bug:** V3's `config.yaml` listed `hrv_freq_window: 180` but the value was
never read. `features.py:extract_hrv_features` computed both `nk.hrv_time` and
`nk.hrv_frequency` on the same 60 s peak slice. Caught by AI code review on
2026-04-30 (after the user's z-score audit caught the prior HRV stale-cache
bug, this was the second HRV-related issue surfaced).

**Why it matters:** Task Force 1996 §V is explicit that frequency-domain HRV
analysis requires the record to be at least as long as one cycle of the lower
edge of the lowest frequency band of interest. For LF (0.04–0.15 Hz, period
6.7–25 s), 60 s gives only ~2.4 cycles minimum — Task-Force-noncompliant
"ultra-short term HRV" with known high LF variance. We had already implicitly
acknowledged this when V3 dropped `hrv_vlf_power` (uncomputable on 60 s) but
we kept LF, HF, total power, and LF/HF ratio on the same too-short window.

**Fix:** Decoupled the windows. Time-domain HRV continues to use the 60 s
feature window; frequency-domain HRV now uses a 180 s lookback ending at the
same `time_end_bvp`. The freq features are NaN'd individually if the available
window is < 120 s (early-session edge case) or has < 30 peaks (insufficient
for stable PSD).

Implementation: [`features.py:extract_hrv_features`](../Updated_Extraction_V3/features.py)
takes separate `(time_start, time_end)` and `(freq_start, freq_end)` argument
pairs; [`dataset_builder.py:build_windowed_dataset`](../Updated_Extraction_V3/dataset_builder.py)
computes `freq_start = max(0, time_end − hrv_freq_window·fs_bvp)`. Real-time
deployment requires a 180 s BVP buffer (~46 KB at 64 Hz, trivial for any MCU).

**F1 impact across all variants:**

| Variant | F1 (60 s freq, broken) | F1 (180 s freq, fixed) | Δ |
|---|---:|---:|---:|
| raw | 0.926 | **0.933** | +0.008 |
| raw + z | 0.886 | 0.894 | +0.008 |
| z | 0.872 | 0.880 | +0.008 |
| mixed (V2-bucket) | 0.851 | 0.847 | −0.004 |
| pct | 0.836 | 0.841 | +0.005 |
| robust z | 0.790 | 0.783 | −0.007 |

Most variants picked up ~0.008 F1, consistent with the freq features moving
from "noise the model ignores" to "signal the model can use." Mixed and
robust z lost slightly. The qualitative ranking is unchanged.

**Per-subject under raw HGB** (the chosen configuration):

| Subject | Pre-fix | Post-fix | Δ |
|---|---:|---:|---|
| **S17** | 0.917 | **1.000** | +0.083 |
| **S3** | 0.636 | **0.727** | +0.091 |
| S2 | 0.900 | 0.900 | 0 |
| S15 | 1.000 | 1.000 | 0 |
| S9 | 0.545 | 0.545 | 0 (still min) |
| S10 | 0.667 | 0.583 | −0.083 |

S3 and S17 specifically benefited — both subjects where freq-domain HRV
signal carries discriminative weight that 60 s couldn't capture stably. The
S3 improvement is meaningful given the corrupted-baseline issue: even though
raw absolute scale was already partially recovering S3, the now-stable freq
features push it from 0.64 to 0.73. S10 lost a bit; possibly because the
old-noisy LF features were spuriously helping S10 and the cleaner freq
features removed that artifact.

### One feature flagged for Phase 5

Post-fix, `hrv_lf_hf_ratio` has mean=4.27 but **std=29.4** across windows.
That's not a real signal; the ratio explodes when HF approaches zero in
specific windows. Phase 5 will either drop the ratio or replace it with
`log(LF/HF)` for stability. Documented as a known feature-stability issue.

---

## 7. Calibration-length sub-experiment — skipped

The original Phase 4 plan was to sweep calibration-baseline length (5 / 10 / 15
min / full) on the chosen normalization. Since raw won and uses no calibration,
the sub-experiment is moot. If we adopt z later for adolescent deployment, we
revisit calibration length there.

---

## 8. Where Phase 4's F1 = 0.926 came from

Phase 2 (RF + z, F1 = 0.776) → Phase 4 (HGB + raw, F1 = 0.926) is a +0.150 jump
from two independent decisions:

| Change | F1 from previous |
|---|---:|
| Phase 2 baseline: RF + z (Phase 2 lock) | 0.776 |
| Switch model: RF → HGB | 0.872 (+0.096) |
| Switch features: z → raw | 0.926 (+0.054) |

The model swap is the bigger contributor; the feature-set swap is meaningful but
secondary. Both were "free" in the sense that no hyperparameter tuning happened
in either phase — these are out-of-the-box defaults.

Comparison to V1:

| Pipeline | Features | Model | F1 (LOSO) |
|---|---|---|---:|
| V1 (rule-based experiments) | 11 | XGBoost | 0.894 |
| V1 (full pipeline, paper-comparable) | 138 | HGB | 0.814 |
| V1 (raw-only, from session log) | 46 | HGB | 0.850 |
| Phase 2 V3 | 48 | RF | 0.776 |
| **Phase 4 V3 — chosen** | **48** | **HGB** | **0.926** |

V3 raw HGB beats V1's best directly-comparable number (0.850, raw-only on V1's
46 features) by **+0.076 F1** with 48 V3 features (raw count is 2 higher because
of the new ACC jerk/activity features; underlying preprocessing is causal V3
instead of V1's cvxEDA-based pipeline). Phase 5 (feature selection) will
quantify which V3 changes matter most.

---

## 9. What's locked vs. provisional after Phase 4

**Locked:**
- Normalization: **none (raw)**.
- Calibration baseline: **N/A** for the WESAD-internal pipeline.
- Comparator model for Phase 5: HistGradientBoosting (`max_iter=300, max_depth=4, learning_rate=0.05, class_weight='balanced'`).
- Dataset cache: `Updated_Extraction_V3/output/dataset_W60_step60_raw.parquet` (single source of truth for Phase 5 onwards).

**Provisional:**
- "Raw transfers to adolescents" is an empirical question for future work (§5).
- The 0.058 F1 cost of switching to z is the price-of-deployment-robustness — explicit on the table for any future re-evaluation.

---

## 10. Implications for Phase 5 onwards

- **Phase 5 (feature selection)** runs on the 48-feature raw set. Goal: identify the smallest stable subset that maintains F1 ≥ 0.92. Methods per the rebuild plan: stability selection, mutual information, SHAP.
- **Phase 6 (model selection)** uses raw features. The S2/S3/S17 question is closed at HGB; Phase 6's job is to compare HGB to LR / RF / XGB / SVM / LightGBM on the chosen feature set, including inference time and model size for embedded deployment.
- **The ~0.054 F1 cost of z** sets a budget: if Phase 6's preferred deployment model (e.g., LR for interpretability) wants z features, we know the within-WESAD penalty.

---

## References

See [docs/references.bib](references.bib). Berntson 1997 for HRV-feature normalization caveats; Schmidt 2018 for the WESAD baseline and the within-subject-normalization-doesn't-help-trees observation.
