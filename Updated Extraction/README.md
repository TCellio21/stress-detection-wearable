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

## Configuration

The pipeline is **configurable** and **reproducible**:

- **Commit `config.yaml`:** Yes. Keep `config.yaml` in version control; it has no machine-specific paths (`paths.wesad_path` is left unset in the file).
- **Per-machine path via .env:** Copy `.env.example` to `.env` in the **repo root** (or in `Updated Extraction/`) and set `WESAD_PATH` to your local WESAD directory. `.env` is gitignored so each developer can have their own path without editing shared config.
- **Config file:** `config.yaml` in this folder overrides other defaults (subjects, window size, etc.). If missing or invalid, built-in defaults are used.
- **Path priority:** `WESAD_PATH` from `.env` or shell → `paths.wesad_path` in `config.yaml` → default `dataset/raw` relative to repo root.
- **Random seed:** Set `reproducibility.random_seed` in `config.yaml` (default `42`) for reproducibility; recorded in the run manifest.

## How to run

From repo root:

```bash
# Optional: one-time setup so the pipeline finds your WESAD data
cp .env.example .env
# Edit .env and set WESAD_PATH to your WESAD folder (e.g. C:\...\WESAD or /path/to/WESAD)

python "Updated Extraction/dataset_builder.py"
```

You must have WESAD `.pkl` files available. Set the path in one of these ways (first wins):

1. **Recommended:** Put `WESAD_PATH=...` in a `.env` file at the repo root (or in `Updated Extraction/`).
2. Set the `WESAD_PATH` environment variable in your shell.
3. Set `paths.wesad_path` in `Updated Extraction/config.yaml` (if you prefer not to use .env).
4. If unset, the default is `dataset/raw` relative to the repo (use this layout if you want no config).


