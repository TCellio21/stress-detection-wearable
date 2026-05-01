# Trained models — V3 stress detection

Three exported models for inference on new wearable data. All three were trained on all 14 WESAD adult subjects (S2–S17 minus S14) using the **locked 16-feature set from Phase 5**.

| File | What it is | F1 (LOSO) | Notes |
|---|---|---:|---|
| **`ensemble_hgb_svm.pkl`** | HGB + SVM-RBF probability-averaging ensemble (Phase 7) | **0.935** | **Recommended ship.** Best F1 in V3. Recovers S3 (corrupted-baseline subject) to 0.727. |
| `tuned_hgb.pkl` | HGB with median tuned hyperparameters (Phase 7 nested-LOSO Optuna) | 0.917 (unbiased) | Methodologically rigorous single-model alternative. min subj recall 0.636. |
| `default_hgb.pkl` | HGB with default hyperparameters (Phase 6 lock) | 0.931 | Simplest fallback. min subj recall 0.545. |

The 16-feature list is in `feature_list.json` and embedded in each model. Tuned hyperparameters are in `hyperparameters.json`.

## Quick start

```python
import sys
sys.path.insert(0, 'models')           # or wherever this directory lives
from inference import load_model

model = load_model('ensemble')          # or 'default' or 'tuned'
print(model)
# StressModel(name='ensemble_hgb_svm', n_features=16, n_estimators=2, threshold=0.5)

# Predict on already-extracted features (DataFrame with named columns)
predictions = model.predict(features_df)        # 1D array of {0, 1}
probabilities = model.predict_proba(features_df) # 1D array of P(stress)
```

The `feature_list` attribute on each model tells you the expected feature names:

```python
model.feature_list
# ['scr_recovery_time_mean', 'hrv_median_rr', 'scr_peak_count', ...]
```

If you pass a numpy array instead of a DataFrame, columns must be in `feature_list` order. Pass DataFrames with named columns to avoid silent ordering bugs.

## Full pipeline: from raw E4 signals to predictions

If you have raw signals (not features yet), use the V3 preprocessing modules to extract the 16 features first:

```python
import sys
sys.path.insert(0, 'Updated_Extraction_V3')
sys.path.insert(0, 'models')

from config_loader import load_config
import dataset_builder as db
from inference import load_model

cfg = load_config()
model = load_model('ensemble')

# Process a new subject's PKL (or adapt build_windowed_dataset for your raw-signal format)
signals = db.preprocess_subject('S99', wesad_path='/path/to/data', cfg=cfg)
windowed_df = db.build_windowed_dataset(signals, cfg, window_sec=60, step_sec=60,
                                        label_rule='majority')

# windowed_df has all 48 raw features; the model only uses the 16 it needs
predictions = model.predict(windowed_df)
probabilities = model.predict_proba(windowed_df)

print(f'Stress windows: {predictions.sum()} / {len(predictions)}')
print(f'Mean P(stress): {probabilities.mean():.3f}')
```

## Choosing which model

**Recommended: `ensemble`** — the best F1 in V3 and the only configuration that materially helps S3 (the corrupted-baseline subject from V2-author's diagnosis). Adds ~33 KB and ~0.3 ms inference per window over `default_hgb`. Worth it.

**Use `tuned_hgb` if** you want a methodologically rigorous single-model deployment with the best worst-case recall (S9 = 0.636 vs 0.545 for default). It comes from the rebuild plan's nested-LOSO Optuna search, so the F1 = 0.917 is an unbiased generalization estimate — directly defensible.

**Use `default_hgb` if** you want simplicity. Highest single-LOSO F1 of any single model (0.931), no tuning to defend.

See [`docs/06_model_selection.md`](../docs/06_model_selection.md) and [`docs/07_hyperparameter_tuning.md`](../docs/07_hyperparameter_tuning.md) for the full comparison and defense.

## Inputs the model expects

- 16 features per window (see `feature_list.json` for the names and order)
- Features must be the V3 raw values — no z-scoring; Phase 4 ruled it out
- 60-second windows, 60-second step (Phase 2 lock)
- HRV freq-domain features computed on a 180-second BVP lookback (Phase 4 fix)
- ACC features in g (divide raw E4 1/64g counts by 64)
- Threshold: 0.5 (Phase 6 lock; tuning didn't change this materially)

## Files in this directory

```
models/
├── README.md                   # this file
├── inference.py                # StressModel class + load_model
├── build_models.py             # rebuilds all three .pkl files (re-runnable)
├── default_hgb.pkl             # Phase 6 lock
├── tuned_hgb.pkl               # Phase 7 tuned (median across 14 outer folds)
├── ensemble_hgb_svm.pkl        # Phase 7 ensemble — recommended
├── feature_list.json           # the 16 feature names (also embedded in each model)
├── hyperparameters.json        # tuned hyperparameters (Phase 7)
└── final_model.pkl             # Phase 7 notebook's original save (legacy; superseded by tuned_hgb.pkl)
```

## Rebuilding from scratch

If the upstream dataset or hyperparameters change:

```bash
python models/build_models.py
```

This re-fits all three models on the current `Updated_Extraction_V3/output/dataset_W60_step60_raw.parquet` and saves fresh `.pkl` files. Takes ~3 seconds.

## API reference

```python
from inference import load_model, list_models, StressModel

# Load by name (with aliases)
model = load_model('ensemble')          # or 'ensemble_hgb_svm'
model = load_model('tuned')             # or 'tuned_hgb'
model = load_model('default')           # or 'default_hgb'

# Inspect what's available
list_models()
# {'default_hgb': {...}, 'tuned_hgb': {...}, 'ensemble_hgb_svm': {...}}

# Inspect a loaded model
model.feature_list                      # list of 16 feature names
model.threshold                         # 0.5
model.metadata                          # {'phase': 7, 'description': ..., 'loso_f1': ..., ...}
model.estimators                        # [hgb, svm] for ensemble; [hgb] for single

# Predict
model.predict_proba(X)                  # shape (n_samples,) of P(stress) in [0, 1]
model.predict(X)                        # shape (n_samples,) of {0, 1}
```

## Compatibility note

All three models are pickled with `protocol=pickle.DEFAULT_PROTOCOL`. They were created with **scikit-learn 1.6.1**. Loading on a different sklearn major version may produce warnings or errors — if you upgrade sklearn, re-run `python models/build_models.py` to regenerate the `.pkl` files.
