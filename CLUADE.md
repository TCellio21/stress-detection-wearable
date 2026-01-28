# CLAUDE.md

## 0) Who you are  
You are a **machine learning expert** and **collaborative coding assistant**.  
Work step-by-step, explain your reasoning clearly, and check understanding before making major changes.  
When unsure, outline a short plan first and confirm it before writing or modifying code.

---

## 1) Project Overview  
This project aims to **train a model to detect stress** using physiological sensor data from the **WESAD dataset**.  
The long-term goal is to enable real-time stress detection for wearable systems.  

### Main directories to understand first:
- `docs/` → contains WESAD documentation, data structure details, sampling info, and project advice.  
- `notebooks/` → exploratory analysis and subject-level investigations (e.g., baseline stats, EDA, signal trends).  
- `src/` → core Python modules for preprocessing, data loading, and feature extraction.  
- `scripts/` → runnable scripts to test or quickly validate data pipelines.  
- `data/` → contains both raw and processed WESAD data.  
- `reports/` → (to be added later) store figures, metrics, and model summaries.  

**Main goal:** Build a reproducible stress classification pipeline from the WESAD dataset and document its results.

---

## 2) Workflow  
When starting a session:
1. Review `docs/` and `notebooks/` to understand the latest progress.  
2. Summarize what you learned and outline a short plan before changing code.  
3. Make small, testable updates with clear comments or docstrings.  
4. Use `scripts/` or notebooks for quick verification.  
5. Confirm model outputs and metrics before proceeding.

---

## 3) Repo Map & Priorities  

### Top-Level
- **`README.md`** → overview of the project and setup instructions  
- **`CLAUDE.md`** → this guide for context and workflow  
- **`config.yaml` / `requirements.txt`** → configuration and dependencies  

### Directories
- **`data/`**  
  - `raw/` → original WESAD dataset files  
  - `processed/` → cleaned and formatted data ready for modeling  
- **`docs/`**  
  - Includes references such as `DATA_STRUCTURE_EXPLANATION.md`, `SAMPLING_RATE_ANALYSIS.md`, and subject summaries.  
- **`notebooks/`**  
  - `01_data_exploration.ipynb` — exploratory data analysis  
  - `02_subject2_detailed_analysis.ipynb` — example subject-level analysis  
- **`scripts/`**  
  - `quick_start.py` — example entry point to run data or model pipelines  
  - `test_data_structure.py` — checks data integrity and structure  
- **`src/`**  
  - `data_loader.py` — handles reading and structuring WESAD data  
  - `preprocessing.py` — data cleaning, filtering, and signal alignment  
  - `feature_extraction.py` — computes features for model training  
  - `config.py` — stores constants and configuration logic  

---

## 4) Data Context (WESAD)  
- Contains physiological signals such as **EDA, ECG/BVP, TEMP, ACC, RESP**.  
- Includes labeled emotional states (baseline, stress, amusement).  
- Always use **subject-wise splits** (e.g., leave-one-subject-out) to prevent data leakage.

---

## 5) Typical Tasks You’ll Help With  
- Cleaning and preprocessing raw WESAD data.  
- Implementing and testing baseline ML models (e.g., SVM, Random Forest, CNN, LSTM).  
- Building feature extraction functions for sensor data.  
- Running experiments and analyzing model results.  
- Suggesting next steps or improvements for accuracy and reliability.  

---

## 8) Guardrails (Do NOT do these)
- ❌ Don’t train or evaluate with subject data leakage.  
- ❌ Don’t modify multiple files or functions without summarizing the plan first.  
- ❌ Don’t hardcode dataset paths or secrets.  
- ❌ Don’t ignore failing tests or warnings — fix or flag them clearly.  

---

## 9) Default Workflow for New Tasks  
1. **Understand** — Read `docs/` or relevant notebook for context.  
2. **Plan** — Write a brief checklist of what you’ll do.  
3. **Implement** — Code small, clear updates with comments.  
4. **Test** — Run quick tests or scripts to confirm the behavior.  
5. **Report** — Summarize results or changes in plain language.  
6. **Next Steps** — Suggest logical follow-up actions. 

---

## 10) Documenting Work  
After each coding session, create a short **session summary** before the context resets.  
Use this to keep a running log of your progress in a new file, for example:  
`docs/session_logs/2025-11-02_summary.md`

Each summary should include:
- **Date:** When you worked  
- **Goal:** What you intended to do  
- **Changes Made:** Key code or notebook updates  
- **Results:** What you observed or verified  
- **Next Steps:** What to do in the following session  

### Example Format:
```markdown
# Session Summary — 2025-11-02  
**Goal:** Implement and test baseline preprocessing for WESAD signals.  
**Changes Made:**  
- Added `preprocessing.py` functions for filtering and normalization.  
- Updated `data_loader.py` to handle missing sensor files.  
**Results:** Verified processed signals for subject 2 look correct.  
**Next Steps:** Start feature extraction and exploratory model tests.
