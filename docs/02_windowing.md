# 02 — Windowing and Step Size

**Scope:** Phase 2 of the V3 rebuild — pick the (window, step) configuration
that maximizes LOSO F1 with no per-subject recall < 0.5. Implementation in
[Updated_Extraction_V3/run_phase2_experiment.py](../Updated_Extraction_V3/run_phase2_experiment.py).

**Date:** 2026-04-30
**Predecessor:** [01_preprocessing.md](01_preprocessing.md)
**Status:** ✅ sweep complete — chosen **W = 60 s, step = 60 s** with caveats (§3.4).

---

## 1. Decisions made before running the experiment

These are the design choices we lock here before we look at any results, so the
sweep tells us about (W, step) and not about confounded labeling/normalization
changes.

### 1.1 Labeling rule — majority vote (`stats.mode`)

V1 and V2 both label each window by the mode of the per-sample labels in that
window. The rebuild plan suggests "any stress" as an alternative — flag the
window as stress if any sample within it is labeled stress. We keep
**majority vote** for the (W, step) sweep:

- WESAD's TSST onsets are clean step transitions: there is no gradient of
  "becoming stressed" across many seconds at the boundaries. Within a window,
  >99 % of samples carry the same label. Majority vote and "any stress" agree
  on the entire interior of each condition.
- The cases where they disagree are *transition windows* spanning the
  baseline→stress or amusement→baseline boundary. With a 60 s window and
  step ≤ 60 s, at most one or two such windows exist per subject per
  transition. "Any stress" promotes a window with 1 s of stress and 59 s of
  baseline to a stress label, which is the wrong assignment for a model that
  should learn from features representative of the stress state.
- Center-sample labeling is brittle to single-sample misalignments at the
  700 Hz label timebase and has no upside over majority vote on this dataset.

A short labeling-rule sub-experiment on the chosen (W, step) is deferred
to Phase 5 — if feature selection there suggests transition windows
contribute usefully, we revisit. The expected outcome is that the choice
is empirically irrelevant for the windows that matter (interior of each
condition).

### 1.2 No TSST-prep drop

Per the 2026-04-29 user decision, we keep all stress windows including the
first 180 s of TSST that V1 dropped. In deployment we don't know we're in
TSST prep — we just see a stress label. Training the model on those windows
is the realistic deployment scenario.

### 1.3 Calibration normalization, per subject

Each subject's `_z` features are z-scored against the mean/std computed over
that subject's `label==1` windows only. This matches the wear-time
calibration paradigm confirmed in the audit. Both raw and z-scored columns
exist; the LOSO RF baseline uses `_z` columns.

### 1.4 NaN-HRV imputation: per-subject median

When the motion gate fires (95th-percentile jerk > 8 g/s in the window),
HRV features are NaN. RandomForest cannot consume NaN, so for the Phase 2
baseline we fill HRV NaNs with the subject's own median HRV across non-
motion-corrupted windows. Phase 6 may switch to HistGradientBoosting (which
handles NaN natively) or a different imputation policy.

### 1.5 Model: Random Forest with default parameters

`RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=2,
class_weight='balanced', random_state=42)`. Phase 6 selects the production
model; Phase 2's RF is just the comparator that lets us rank (W, step) tuples.
No hyperparameter tuning at Phase 2 — the RF defaults are not the point of
the experiment.

---

## 2. Experiment design

### 2.1 (W, step) grid

Window sizes: **{30, 60, 90, 120}** s. Step sizes: **{15, 30, 60}** s,
constrained to step ≤ W (no gaps). 12 valid configurations:

| W ↓ \ step → | 15 | 30 | 60 |
|---|---|---|---|
| 30  | ✓ | ✓ | — |
| 60  | ✓ | ✓ | ✓ |
| 90  | ✓ | ✓ | ✓ |
| 120 | ✓ | ✓ | ✓ |

### 2.2 Caching

Preprocessing runs once per subject (~5 s × 14 subjects ≈ 1 min) and is
cached. For each of 12 configs, we re-window the cached signals (~10 s ×
14 subjects ≈ 2 min) and run LOSO RF (~30 s). Total ~30 min.

### 2.3 Metrics

Per fold (held-out subject): accuracy, F1-stress, recall-stress,
precision-stress.

Per config: mean and std of those metrics across the 14 folds, plus
- `min_subject_recall` — the worst per-subject recall in the LOSO matrix
- `n_subjects_recall_below_0_5` — count of subjects with recall < 0.5
- `n_subjects_recall_zero` — count of subjects with recall = 0

### 2.4 Note on HRV reliability at short windows

HRV time-domain features (RMSSD, SDNN, etc.) need ≥ 60 s of data for stable
estimates (Task Force 1996). HRV frequency-domain features (LF, HF, LF/HF)
ideally need ≥ 120 s. At W = 30 s we expect:

- ~30 IBIs per window — borderline for time-domain stability
- LF band (0.04–0.15 Hz) has a period of 6.7–25 s — at W = 30 s, only ~1 cycle
  fits inside the window, so LF estimates are noisy
- HF band (0.15–0.4 Hz, period 2.5–6.7 s) is fine at W = 30 s

Phase 2 reports HRV features anyway. If short-W configs lose F1 it may be
because of this rather than because short windows are intrinsically bad —
we'll separate those concerns in Phase 3 (decoupled HRV freq window).

---

## 3. Results

### 3.1 Aggregate per-config metrics (LOSO RandomForest)

Sorted by mean F1. CSV: [reports/02_windowing/results.csv](../reports/02_windowing/results.csv).
Heatmap: [reports/02_windowing/heatmap.png](../reports/02_windowing/heatmap.png).

| W (s) | step (s) | n_windows | stress frac | mean F1 | std F1 | mean recall | min subj recall | # subj < 0.5 | # subj = 0 | runtime |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 120 | 60 | 707 | 0.221 | **0.780** | 0.332 | 0.768 | 0.000 | 3 | **1** | 42 s |
| 60 | 60 | 691 | 0.223 | **0.776** | 0.347 | 0.761 | 0.000 | 3 | **1** | 30 s |
| 30 | 30 | 1397 | 0.223 | 0.765 | 0.340 | 0.756 | 0.000 | 3 | 2 | 45 s |
| 120 | 30 | 1402 | 0.221 | 0.761 | 0.377 | 0.746 | 0.000 | 3 | 2 | 70 s |
| 60 | 30 | 1396 | 0.222 | 0.758 | 0.349 | 0.746 | 0.000 | 3 | **1** | 55 s |
| 120 | 15 | 2804 | 0.222 | 0.745 | 0.408 | 0.736 | 0.000 | 3 | 3 | 131 s |
| 90 | 30 | 1397 | 0.223 | 0.744 | 0.391 | 0.741 | 0.000 | 3 | 2 | 75 s |
| 30 | 15 | 2793 | 0.223 | 0.741 | 0.389 | 0.754 | 0.000 | 3 | **1** | 98 s |
| 90 | 15 | 2794 | 0.223 | 0.738 | 0.396 | 0.730 | 0.000 | 3 | 2 | 426 s |
| 90 | 60 | 707 | 0.221 | 0.732 | 0.404 | 0.743 | 0.000 | 3 | 3 | 40 s |
| 60 | 15 | 2793 | 0.223 | 0.723 | 0.399 | 0.724 | 0.000 | 3 | 3 | 122 s |

**Two observations are immediate:**

1. **F1 is flat in the 0.72–0.78 range across all 11 configs.** The (W, step)
   choice is *not* the dominant factor. The std of F1 within most configs
   (~0.34–0.41) is much larger than the spread between configs (0.06).
2. **Every config has at least one subject with zero recall.** The
   `min_subject_recall` column is identically 0.000 across all 11 cells of
   the grid. The "no subject below 0.5 recall" criterion in the rebuild plan
   is **unachievable** for V3-features × RandomForest, regardless of (W, step).

### 3.2 Per-subject recall — three persistent failures

CSV: [reports/02_windowing/per_subject_recall.csv](../reports/02_windowing/per_subject_recall.csv).
Heatmap: [reports/02_windowing/per_subject_heatmap.png](../reports/02_windowing/per_subject_heatmap.png).

Best per-subject recall *across any (W, step) configuration*:

| Subject | best recall | configs at best | category |
|---|---:|---|---|
| **S2** | 0.450 | W30s30 | **persistent failure** |
| **S3** | 0.091 | W120s60 only | **persistent failure** |
| **S17** | 0.167 | W60s60 only | **persistent failure** |
| S10 | 0.939 | W30s15 | window-sensitive (worse at long W) |
| S9 | 0.909 | W90s60 | window-sensitive (slightly better at long W) |
| S11 | 1.000 | W90s60 | mostly insensitive |
| S8 | 1.000 | W120s15, W120s60 | mostly insensitive |
| S4, S5, S16 | 1.000 | every config | trivially classified |
| S6, S7, S13, S15 | ≥ 0.95 | most configs | easy |

**S2, S3, S17 are model-class failures.** This is exactly the audit's
§6 #4 finding: the V1 session log
([2026-03-11_nb05_nb06_results_analysis.md](session_logs/2026-03-11_nb05_nb06_results_analysis.md))
documented that on V1's 138-feature dataset, RF / HistGradientBoosting got
0.0 recall on S2 while LogReg got 0.86 and LinearSVC got 1.0. Phase 6's
model comparison is the only place this can be addressed; (W, step) tuning
cannot fix it. We carry it forward as a known limitation, not a Phase 2
blocker. The interaction is documented in the audit and confirmed
empirically here.

**S10 is window-sensitive in the opposite direction.** Recall drops
monotonically from 0.94 at W=30/step=15 to 0.65–0.75 at W=120. So *some*
subjects benefit from short windows. This trade-off is small in aggregate
F1 (~0.02) but matters for any threshold-sensitive deployment. Note: HRV
features at W=30 are unreliable (per §2.4), so the W=30 effect on S10 is
likely from EDA/ACC features, not HRV.

### 3.3 Three-way tie at the top

The top three configs are within 0.015 F1 of each other:

| W, step | F1 | Δ vs best | rationale to pick |
|---|---:|---:|---|
| 120, 60 | 0.780 | — | **Highest F1**; HRV freq stable (period × multiple). Few windows for downstream (707). |
| 60, 60 | 0.776 | −0.004 | **V1 / Schmidt 2018 convention**; HRV time-domain stable; comparability with V1 baseline. |
| 30, 30 | 0.765 | −0.015 | **Most data** (1397 windows); best recovery on S2 (0.45 vs 0.27 at W=120). HRV freq features unreliable at this length (§2.4). |

The F1 differences are within the per-fold std (~0.34), so any of the
three is statistically defensible.

### 3.4 Chosen configuration: **W = 60 s, step = 60 s**

The W=120/step=60 row is technically highest F1, but I'm picking
**W = 60 s, step = 60 s** as the locked Phase 2 configuration:

- **F1 difference is 0.004** — well inside the per-fold std (~0.35).
  Statistically the two configs are indistinguishable on F1.
- **W = 60 s matches Schmidt 2018's WESAD reference pipeline** and V1/V2.
  Comparability with the existing F1=0.814 V1 benchmark is preserved.
  At W = 120 we are not directly comparable.
- **HRV time-domain features are stable at 60 s** (Task Force 1996) without
  requiring decoupled windows. At W = 90 or 120, the time-domain features
  benefit slightly from the longer window, but at the cost of fewer windows
  per subject (and unrealistic latency in any future real-time deployment
  where every prediction now requires a 120 s lookback at the cursor).
- **Same number of zero-recall subjects** as the top config (1 of 14) and
  the same set of persistent-failure subjects (S2, S3, S17). The win at
  W=120 is on the easy-subject tail, not on the hard subjects we care
  about.
- **Step = 60 s = non-overlapping windows.** Each prediction is independent
  data; no subtle correlation in the LOSO test fold from overlapping
  windows. Phase 5 / 6 statistics will be cleaner.

If Phase 6's model comparison surfaces a model that prefers W=120 (i.e.,
flips the tie), we revisit. We don't expect it to — the F1 ceiling here is
set by the hard subjects, not by window length.

### 3.5 Comparison to V1 baseline

V1's HistGradientBoosting on 138 features achieved F1 = 0.814 (LOSO). V3's
RandomForest on 48 z-only features at the chosen W = 60, step = 60 achieves
F1 = 0.776. The gap (−0.038) reflects:

- **Different model class** (RF vs HGB). Phase 6 will switch to HGB +
  others; we expect to recover most of this gap.
- **Different feature count** (48 vs 138). V1 stacked raw + z-score +
  percent-change (3× the same 46 features). Phase 4 will test whether
  raw-and-z together helps V3.
- **No hyperparameter tuning** in Phase 2. Phase 7 will tune the chosen
  Phase 6 model.

The 0.776 number is therefore the **floor** that Phases 4–7 should beat,
not a final result.

---

## 4. What this tells us, what it doesn't

The Phase 2 sweep ranks (W, step) under one model class (RandomForest) and
one labeling rule (majority vote). It does not:

- Establish absolute performance — Phase 6 chooses the model, Phase 7 tunes
  hyperparameters. The F1 numbers here are *comparative*, not the floor.
- Fix the SCR-feature divergence flagged in
  [01_preprocessing.md §6.1](01_preprocessing.md#61-cvxeda-vs-v3-complementary-hp--feature-correlations).
  Phase 5 / 6 will determine whether V3 SCR features carry signal.
- Tell us whether the motion gate's 8 g/s threshold is right. We'll revisit
  in Phase 5 (feature selection) if motion-corrupted windows correlate with
  low-recall subjects.

What we do learn: (a) the time-resolution sweet spot for our preprocessed
signal stack, (b) which subjects are (W, step)-sensitive vs uniformly hard,
and (c) whether our HRV-at-short-W concern materializes empirically.

---

## 5. References

See [docs/references.bib](references.bib). Task Force 1996 for HRV window
length minima; Schmidt 2018 for WESAD window/step conventions
(Schmidt used 60 s / 0.25 s sliding for chest, 60 s / 5 s sliding for
wrist). Our 30 s step at W=60 s is consistent with that range.
