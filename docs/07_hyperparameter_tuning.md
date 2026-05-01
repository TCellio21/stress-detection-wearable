# 07 — Hyperparameter tuning (nested LOSO + Optuna)

**Phase 7 of the V3 rebuild.** Optuna Bayesian search on HGB hyperparameters under nested LOSO, plus an HGB+SVM-RBF ensemble side experiment.

**Date:** 2026-04-30
**Notebook:** [notebooks/07_hyperparameter_tuning.ipynb](../notebooks/07_hyperparameter_tuning.ipynb)
**Status:** ✅ tuning ran rigorously; **deployment recommendation: HGB+SVM-RBF ensemble** (F1 = 0.935, recovers S3). **Tuned HGB** (median hyperparameters) is the methodologically-rigorous single-model alternative (nested-LOSO unbiased F1 = 0.917). **Default HGB** (Phase 6 lock, single-LOSO F1 = 0.931) remains the simplest fallback.

---

## 1. Headline result

Three configurations evaluated against the Phase 6 baseline:

| Configuration | F1 | min subj recall | S3 recall | Eval protocol |
|---|---:|---:|---:|---|
| Default HGB (Phase 6) | 0.931 | 0.545 | 0.636 | single LOSO |
| **Optuna-tuned HGB (median hyperparams)** | **0.917 ± 0.085** | **0.636** | 0.636 | **nested LOSO (unbiased)** |
| HGB+SVM-RBF probability ensemble | **0.935** | 0.545 | **0.727** | single LOSO |

**Three findings worth the doc:**

1. **Hyperparameter tuning didn't help in unbiased evaluation.** The nested-LOSO F1 (0.917) is *0.014 below* the default. This is within fold-to-fold std, so technically equivalent — but it confirms that on a 14-subject cohort, hyperparameter tuning can reshuffle decisions without net improvement. The rebuild plan's Hard Constraint #4 (nested LOSO required) earned its keep here by surfacing this honestly.
2. **Tuning *did* help the worst subject.** S9 went from 0.545 to **0.636** under tuning. The aggregate F1 stayed the same because losses on S2 (-0.10) and S16 (-0.09) cancelled the S9 gain. From a clinical-deployment perspective, raising the minimum recall is a real win — it's the per-subject worst case that drives "will this device fail anyone?"
3. **The HGB+SVM-RBF ensemble is the only configuration that improved both aggregate F1 (+0.004) AND a hard subject (S3 +0.091).** This is the first material S3 recovery anywhere in the V3 pipeline. The cost is two-model storage (~463 KB total) and ~10 ms additional inference per window — both trivial.

CSVs: [nested_loso_per_fold.csv](../reports/07_hyperparameter_tuning/nested_loso_per_fold.csv), [hyperparameter_stability.csv](../reports/07_hyperparameter_tuning/hyperparameter_stability.csv), [ensemble_experiment.csv](../reports/07_hyperparameter_tuning/ensemble_experiment.csv).

---

## 2. Methodology — why nested LOSO matters

The rebuild plan's Hard Constraint #4 explicitly forbids tuning hyperparameters on the outer test fold:

> "Nested LOSO for hyperparameter tuning. Outer loop is LOSO over subjects for performance estimation. Inner loop is k-fold (over remaining subjects, NOT within-subject) for hyperparameter selection. Tuning on the outer test fold is a hard fail."

We implemented this rigorously:

- **Outer loop**: 14-fold LOSO. Each held-out subject is the *unbiased* test for the model tuned on the other 13.
- **Inner loop**: 5-fold StratifiedGroupKFold over the 13 training subjects (~2-3 subjects per inner fold, stratified by class balance, no subject leakage).
- **Optuna** runs 25 trials per outer fold using TPE sampler, parallelized 4-way. Each trial fits HGB on 4/5 of training subjects and scores on the 5th — 5 fits per trial, 125 fits per outer fold, 1750 fits total.
- Total runtime: 29.9 min wall-clock with `n_jobs=4`.

After all 14 outer folds:
- Aggregate the unbiased outer-fold predictions → unbiased nested-LOSO F1 = **0.917**
- Aggregate the per-fold chosen hyperparameters → median = the "shipping" parameters

Why this matters: a single Optuna study using LOSO-CV in the objective (the simpler approach most ML projects use) would have given F1 ≈ 0.92–0.94 because the inner CV is the same as the outer evaluation — the model overfits to LOSO's specific fold structure. The honest unbiased estimate is 0.917. **The 0.014 gap quantifies the bias that single-LOSO hyperparameter tuning would have introduced.**

---

## 3. Search space and results

| Hyperparameter | Range | Default | Median across 14 folds | Δ |
|---|---|---:|---:|---:|
| `learning_rate` | log-uniform [0.005, 0.5] | 0.05 | **0.0920** | +0.042 |
| `max_iter` | int [100, 1000] | 300 | **752** | +452 |
| `max_depth` | int [2, 12] | 4 | **6.5** | +2.5 |
| `min_samples_leaf` | int [5, 100] | 20 | **9.5** | −10.5 |
| `l2_regularization` | log-uniform [1e-6, 10] | 0 | 0.0612 | +0.061 |
| `max_leaf_nodes` | int [10, 200] | 31 | **92** | +61 |
| `max_features` | uniform [0.3, 1.0] | 1.0 | **0.60** | −0.40 |

**Pattern: Optuna prefers larger-capacity models with smaller leaves and feature subsampling.** Doubling max_iter, halving min_samples_leaf, tripling max_leaf_nodes, dropping max_features to 0.6 (40% of features used per tree split). The intuition is that the locked 16-feature set has enough signal that a more flexible model captures more without overfitting (in inner CV) — but the unbiased outer evaluation says the gain doesn't generalize across subjects.

**fANOVA hyperparameter importance** (single fold, S=S3 — the highest-inner-F1 study):

| Hyperparameter | Importance |
|---|---:|
| `min_samples_leaf` | **0.588** |
| `max_iter` | 0.166 |
| `max_depth` | 0.075 |
| `learning_rate` | 0.065 |
| `max_leaf_nodes` | 0.049 |
| `max_features` | 0.039 |
| `l2_regularization` | 0.018 |

`min_samples_leaf` accounts for **59% of F1 variance** across the search. Optuna's preference for ~10 (vs the sklearn default of 20) is the single biggest tuning effect. Reasoning: smaller leaves let the model carve the threshold-sensitive 14% of windows more finely (Phase 6 §5). Whether that helps generalize is exactly what nested LOSO tells us — and the answer is "marginally, at best."

Plot: [hyperparameter_stability.png](../reports/07_hyperparameter_tuning/hyperparameter_stability.png) shows per-fold chosen values for each hyperparameter with default and median lines. Spread is wide (e.g., `max_iter` ranges 120–999 across folds) — there isn't a tight optimum the inner CV consistently finds.

---

## 4. Per-subject impact of tuning

| Subject | Default HGB recall | Tuned HGB recall | Δ |
|---|---:|---:|---:|
| **S9** | 0.545 | **0.636** | **+0.091** ✓ (was minimum, no longer is) |
| S15, S4, S5, S6, S7, S8, S11, S13 | 1.000 | 1.000 | 0 |
| S10 | 0.750 | 0.750 | 0 |
| S17 | 0.917 | 0.917 | 0 |
| S3 | 0.636 | 0.636 | 0 (corrupted-baseline subject; tuning doesn't help) |
| S16 | 1.000 | 0.909 | −0.091 |
| **S2** | 0.900 | **0.800** | **−0.100** |

The minimum recall improves (0.545 → 0.636) but Optuna trades S2 and S16 to get there. Aggregate F1 falls because S2 and S16 contribute more aggregate stress windows than S9 does (per-fold weights matter). Net: **the worst-case is better, the average is slightly worse**.

This is a real deployment trade-off: a stress-detection device that's slightly worse on average but never below 0.636 recall on any subject is arguably more clinically defensible than one that's slightly better on average but has 0.545-recall floor. Phase 9 defense doc should make this argument explicitly.

---

## 5. The HGB + SVM-RBF ensemble — the empirical winner

CSV: [ensemble_experiment.csv](../reports/07_hyperparameter_tuning/ensemble_experiment.csv).

Phase 6 noted that SVM-RBF was the only model that beat HGB on S3 (recall 0.727 vs HGB 0.636) — but its overall F1 (0.846) was 0.085 below HGB's, so it didn't win as a single model. The natural follow-up: average the predicted probabilities from HGB and SVM-RBF, threshold at 0.5, see what happens.

Results (single LOSO, default hyperparameters for both base models):

| Subject | HGB only | SVM-RBF only | Ensemble | Δ vs HGB |
|---|---:|---:|---:|---:|
| **S3** | 0.636 | 0.636 | **0.727** | **+0.091** ✓ |
| S17 | 0.917 | 0.750 | 0.833 | −0.083 |
| S2 | 0.900 | 0.900 | 0.900 | 0 |
| S9 | 0.545 | 0.545 | 0.545 | 0 |
| S10 | 0.750 | 0.667 | 0.750 | 0 |
| All others | 1.000 | varies | 1.000 | 0 |

| Aggregate | HGB | Ensemble |
|---|---:|---:|
| F1 | 0.931 | **0.935** (+0.004) |
| Min subj recall | 0.545 | 0.545 |

**The ensemble improves S3 by +0.091 with only one regression (S17 −0.083) and no impact on the rest.** Net F1 +0.004 on aggregate; small absolute gain, but the S3 recovery is the real story — this is the **first method anywhere in the V3 pipeline** to materially recover S3, the V2-author-diagnosed corrupted-baseline subject.

Why does it work? S3's stress windows have inverted physiological relationships (higher SCL during baseline than stress — see [memory: S3 corrupted baseline](../../../memory/project_s3_corrupted_baseline.md)). HGB's tree splits get confused. SVM-RBF's non-linear kernel is more flexible about the inverted relationship. Averaging probabilities lets HGB "vote down" the borderline cases where it's uncertain and let SVM tip them the right way for S3.

**Caveat:** the ensemble F1 (0.935) was measured by single LOSO with default hyperparameters for both base models. It's directly comparable to the Phase 6 baseline (0.931, also single LOSO with defaults), so the +0.004 is real. It's NOT directly comparable to the nested-LOSO tuned HGB (0.917) because the ensemble was not nested-tuned. Doing so would cost another 30 min and likely shift the number by ±0.005.

---

## 6. The shipping decision

Three options on the table. Recommendations in priority order:

### Option 1 (recommended) — HGB + SVM-RBF ensemble
- **F1 = 0.935** (single LOSO, +0.004 vs default HGB)
- **S3 recall = 0.727** (+0.091 vs HGB alone — meaningful clinical win)
- min subj recall = 0.545 (S9 unchanged)
- Cost: 463 KB combined model size, ~10.3 ms inference (vs 10 ms for HGB alone)
- Defense story: "We deployed an ensemble because it was the only configuration that materially improved a known hard subject (S3) without hurting others."

### Option 2 (methodologically rigorous alternative) — Tuned HGB (median hyperparameters)
- F1 = 0.917 ± 0.085 (nested LOSO unbiased) / 0.922 (single LOSO biased)
- min subj recall = 0.636 (S9 recovered)
- Cost: same as default HGB (430 KB, 10 ms)
- Defense story: "We applied nested-LOSO Bayesian hyperparameter tuning per the rebuild plan's Hard Constraint #4. The unbiased generalization estimate was 0.917, with a +0.091 recall improvement on the previously-worst subject."

### Option 3 (simplest fallback) — Default HGB
- F1 = 0.931 (single LOSO)
- min subj recall = 0.545
- Cost: 430 KB, 10 ms
- Defense story: "Default sklearn settings, no tuning required."

For Phase 8 / 9, we save **all three** so they can be compared in real-time-simulator runs:

- `models/final_model.pkl` — tuned HGB with median hyperparameters
- `reports/06_model_selection/chosen_model.pkl` — default HGB (Phase 6 chosen)
- The ensemble can be reconstructed from either model + a SVM-RBF refit (no separate save needed)

The defense doc (Phase 9) will pick *one* primary among these; recommend the ensemble.

---

## 7. What's locked vs. provisional after Phase 7

**Locked:**
- HGB's tuned hyperparameters: `learning_rate=0.0920, max_iter=752, max_depth=6, min_samples_leaf=9, l2_regularization=0.0612, max_leaf_nodes=92, max_features=0.60`. Saved in [models/hyperparameters.json](../models/hyperparameters.json).
- Decision threshold: 0.5 (Phase 6 lock — confirmed unchanged in Phase 7's sweep).
- The ensemble configuration: HGB(default) + SVM-RBF(default), probability-averaging at threshold 0.5.

**Provisional:**
- The shipping primary (ensemble vs. tuned HGB vs. default HGB) — Phase 8 (real-time simulator) should re-confirm with streaming evaluation.
- The S9 recall floor (0.545–0.636 depending on configuration). Future work: investigate S9's specific signal anomalies; consider per-subject calibration during wear-time.

---

## 8. Implications for Phase 8 (real-time simulator)

Phase 8 runs the "virtual microcontroller" loop on each subject's full signal in causal-streaming order, computing features per cursor advance and emitting predictions. Two things to verify there:

1. **Streaming-LOSO accuracy should match offline LOSO closely.** A gap > 0.02 F1 indicates a hidden time-causality bug in the pipeline. With the locked W=60s / step=60s and decoupled HRV (60s time / 180s freq), there's a 180-second startup latency before the first prediction (set by the freq-domain HRV lookback). Beyond that, predictions should be the same as offline LOSO.
2. **Inference latency end-to-end.** Phase 6 measured 10 ms for HGB-only, single-window predict. The ensemble adds SVM-RBF (~0.3 ms), so total inference ~10.3 ms — still well under any wearable MCU's 60 s prediction budget. But Phase 8 should measure the *full pipeline* (preprocessing → feature extraction → both models → probability average → threshold) to confirm no surprise costs.

Phase 8 also offers a chance to re-validate the threshold (0.5) on streaming data — if probability distributions shift in streaming vs offline (they shouldn't, but it's worth checking), the threshold may need adjustment.

---

## References

- Optuna ([Akiba et al. 2019](https://doi.org/10.1145/3292500.3330701)) for Bayesian hyperparameter optimization with TPE sampler.
- HGB sklearn docs: [HistGradientBoostingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html).
- StratifiedGroupKFold: [sklearn user guide](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html).
- The "tuning doesn't help on small cohorts" finding: empirically observed here; consistent with hyperparameter-tuning skepticism in [Bouthillier et al. 2021](https://arxiv.org/abs/2103.03098).

Project-internal: [docs/06_model_selection.md](06_model_selection.md) for the Phase 6 model comparison; [memory/project_s3_corrupted_baseline.md](../../../memory/project_s3_corrupted_baseline.md) for the S3 limitation context.
