# Updated_Extraction_V3 — Causal, Real-Time-Deployable WESAD Pipeline

V3 is the rebuild target. It descends from V2 (`Updated_Extraction_V2/`) but fixes
V2's look-ahead bugs and unit inconsistencies, ports V1's clean config/orchestration
structure, and adds Phase-3 jerk features that V1/V2 lacked.

V1 (`Updated_Extraction/`) and V2 stay in the repo as historical references. Do not
extend them.

## Status

| Phase | Status | Output |
|---|---|---|
| 0 — Audit | ✅ | [docs/00_existing_pipeline_audit.md](../docs/00_existing_pipeline_audit.md) |
| 1 — Preprocessing | ✅ | [preprocessing.py](preprocessing.py), [docs/01_preprocessing.md](../docs/01_preprocessing.md), [reports/01_preprocessing/](../reports/01_preprocessing/) |
| 2 — Windowing/step sweep | ✅ | [features.py](features.py), [dataset_builder.py](dataset_builder.py), [run_phase2_experiment.py](run_phase2_experiment.py), [docs/02_windowing.md](../docs/02_windowing.md), [reports/02_windowing/](../reports/02_windowing/). Locked: **W=60s, step=60s**. RF baseline F1=0.776. |
| 3 — Feature extraction | (features.py exists; doc + tests pending) | |
| 4 — Normalization comparison | ✅ | [notebooks/04_normalization.ipynb](../notebooks/04_normalization.ipynb), [docs/04_normalization.md](../docs/04_normalization.md), [reports/04_normalization/](../reports/04_normalization/). Locked: **raw features (no normalization), decoupled HRV time/freq windows**. HGB F1=0.933, no subjects below 0.5 recall. S2 (0.90), S17 (1.00) recovered; S3 partial (0.73). |
| 5 — Feature selection | ✅ | [notebooks/05_feature_selection.ipynb](../notebooks/05_feature_selection.ipynb), [docs/05_feature_selection.md](../docs/05_feature_selection.md), [reports/05_feature_selection/](../reports/05_feature_selection/). Locked: **combined-rank top-16 features** (avg of stability rank and SHAP rank). HGB F1=0.931, no subjects below 0.5 recall. 67% feature reduction from 48 → 16. |
| 6 — Model selection | ✅ | [notebooks/06_model_selection.ipynb](../notebooks/06_model_selection.ipynb), [docs/06_model_selection.md](../docs/06_model_selection.md), [reports/06_model_selection/](../reports/06_model_selection/). Locked: **HGB at threshold 0.5** (F1=0.931, recall=0.911). LightGBM as deployment fallback (1.4ms inference vs HGB's 10ms, F1=0.921). |
| 7 — Hyperparameter tuning | ✅ | [notebooks/07_hyperparameter_tuning.ipynb](../notebooks/07_hyperparameter_tuning.ipynb), [docs/07_hyperparameter_tuning.md](../docs/07_hyperparameter_tuning.md), [reports/07_hyperparameter_tuning/](../reports/07_hyperparameter_tuning/), [models/](../models/). Nested-LOSO Optuna unbiased F1=0.917 (Δ -0.014 vs default; +0.091 on S9). **Recommended ship: HGB+SVM-RBF ensemble** (F1=0.935, **S3 recovered to 0.727**). |
| 8 — Real-time simulator | pending | |
| 9 — Defense doc | pending | |

See [docs/00_existing_pipeline_audit.md §6](../docs/00_existing_pipeline_audit.md#6--open-questions--answer-before-phase-1)
for the design decisions that anchor this pipeline.

## Layout

```
Updated_Extraction_V3/
├── config.yaml            # all tunables, no per-machine paths
├── config_loader.py       # YAML + .env loading (WESAD_PATH override)
├── preprocessing.py       # Phase 1 — causal filters + artifact detection
├── README.md              # this file
└── output/                # written by dataset_builder once Phase 3 lands
```

## How to run preprocessing on a single subject

```python
import pickle
from config_loader import load_config
import preprocessing as pp

cfg = load_config()
with open(f"{cfg['paths']['wesad_path']}/S2/S2.pkl", "rb") as f:
    data = pickle.load(f, encoding="latin1")

eda = pp.clean_eda(data["signal"]["wrist"]["EDA"].flatten(),
                   cfg["sampling_rates"]["eda"], cfg["preprocessing"]["eda"])
bvp = pp.process_bvp(data["signal"]["wrist"]["BVP"].flatten(),
                     cfg["sampling_rates"]["bvp"], cfg["preprocessing"]["bvp"])
temp = pp.clean_temp(data["signal"]["wrist"]["TEMP"].flatten(),
                     cfg["sampling_rates"]["temp"], cfg["preprocessing"]["temp"])
acc = pp.clean_acc(data["signal"]["wrist"]["ACC"],
                   cfg["sampling_rates"]["acc"], cfg["preprocessing"]["acc"])
```

## Configuration

`config.yaml` is committed and contains no machine paths. Set `WESAD_PATH` in the
repo-root `.env` (gitignored) for your local dataset location, exactly like V1 and V2.

## What's different from V2

See [docs/01_preprocessing.md](../docs/01_preprocessing.md) for the full diff with
citations. Headline changes:

1. **Phasic EDA.** V2 used `np.gradient` (central differences → 1-sample look-ahead)
   then `np.maximum(0, …)`. V3 uses a complementary causal high-pass at 0.05 Hz, so
   phasic + tonic = smoothed exactly, units stay in µS, and the SCR rise/recovery
   features stay well-defined for `nk.eda_peaks` downstream.
2. **BVP cleaning.** V2 (and V1) used `nk.ppg_process`, which calls `signal_filter`
   with `sosfiltfilt` — non-causal in NK2 0.2.12. V3 pre-cleans with a causal
   Butterworth bandpass `[0.5, 8] Hz, order 3` and feeds the result to
   `nk.ppg_findpeaks(method="elgendi")` (peak detection only — no filtering).
3. **Ectopic-IBI correction.** Berntson 1997-style replacement of IBIs deviating
   >20% from a causal moving median over the prior 5 plausible beats. Uses raw IBIs
   for the median (not the running corrected array — that cascades).
4. **ACC units.** V1/V2 left ACC in raw 1/64 g counts. V3 converts to g.
5. **Jerk features.** V3 computes 3-axis jerk magnitude via causal backward
   difference; V1/V2 had no jerk features. PROJECT_ADVICE.md identifies fidget-band
   movement as the key signal for the classroom application.
6. **S14.** Removed from the subject list (V2 included it; V1 and project memory
   exclude it as hyporesponsive).
7. **6.4 s startup latency.** Carried over from V2, now documented as intentional
   group-delay compensation, not look-ahead.

## What's deliberately the same as V2

- Causal `lfilter`-based IIR filtering (state-initialized at first sample)
- 1 Hz LP pre-smoothing → 0.05 Hz LP for SCL — V2's filter cascade
- Decoupled HRV freq-domain window (180 s) per Task Force 1996 stability requirement
- Per-subject calibration baseline (label=1 windows) for normalization (Phase 4)
- 60 s feature window, 30 s sliding step (Phase 2 will sweep)
- Majority-vote window labels via `stats.mode`
