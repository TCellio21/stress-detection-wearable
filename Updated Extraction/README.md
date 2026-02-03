# Updated Extraction

This folder contains a **modular, well-commented** WESAD wrist feature-extraction pipeline that is intended to be a cleaned-up, corrected version of Grant's extraction work.

## Key design decisions (intentional)

- **Time continuity preserved**: we do **not** "mask then stitch" signals before windowing. Windows are created on the original continuous time axis and then filtered by window label.
- **TSST preparation excluded**: for each subject, we drop the first **3 stress windows** (60s windows → 180s) after stress begins.
- **Exclude S14**: subject `S14` is excluded (data quality / atypical responder).
- **cvxEDA-only (no fallback)**: EDA decomposition uses `neurokit2.eda_phasic(..., method="cvxeda")` and will **hard-fail** if cvxEDA is unavailable or errors.
  - NOTE: A fallback method (e.g., smoothmedian) can materially change features and model results; during testing this produced altered results, so we disable fallbacks by design.
- **Binary labels**:
  - **stress** = original label 2 (TSST)
  - **non-stress** = original labels {1 (baseline), 3 (amusement), 4 (meditation)}
- **Baseline-only subject normalization**: normalization statistics are computed using **true baseline windows only** (original label 1), then applied to **all windows**.

## Output

`dataset_builder.py` writes:
- A CSV dataset (raw + normalized features)
- A small JSON run manifest (counts, feature list, config)

## How to run

From repo root:

```bash
python "Updated Extraction/dataset_builder.py"
```

You must have WESAD `.pkl` files available at the configured `WESAD_PATH` (edit the constant at top of `dataset_builder.py`).


