# Workspace Notes - Project Context & Next Actions

## Purpose
This file contains workspace-specific context, current state, and next actions. 
Use `/prime` command to bootstrap a fresh instance (reads lines 260-316).
Use `/pass` command to update Next Actions section (lines 260-275).

---

## Project Structure Overview

### Active Codebase
- **`Updated Extraction/`** - Current production pipeline
  - Merged Grant + Tanner approaches
  - 138 features (46 raw + 92 normalized)
  - LOSO CV implementation
  - Hyperparameter optimization complete

### Documentation
- **`grant/context_reboot/`** - Technical documentation
  - `TECHNICAL_STATE.md` - Dataset & processing details
  - `SESSION_LOG.md` - Historical session notes
  - `PIPELINE_OVERVIEW.md` - Pipeline architecture
  - `PROJECT_CONTEXT.md` - Session tracking
  - `EDA_SIGNAL_PRIMER.md` - EDA fundamentals
  - `WESAD_PROTOCOL_AND_LABELS.md` - Protocol details
  - `ML_IMPLEMENTATION_CONTEXT.md` - ML implementation details
  - `WESAD_FEATURE_RESEARCH.md` - Feature research
  - `ML_EXPERIMENT_DOCUMENTATION_GUIDE.md` - Experiment docs
  - `PROMPT_OPTIMIZATION_README.md` - Prompt optimization

### Superseded/Archive
- **`grant/Feature Extraction/`** - Old version (superseded by `Updated Extraction/`)
- **`grant/Hyperparameter_Tuning/`** - Empty/minimal files (can delete)
- **`tanner/`** - Personal working directory (may be merged)

---

## Current Implementation Details

### Feature Extraction Pipeline
- **Entry Point:** `Updated Extraction/dataset_builder.py`
- **Output:** `all_subject_features_updated.csv` (1.1MB)
- **Features:** 46 raw + 92 normalized = 138 total
- **Normalization:** Baseline-only (label=1) statistics applied to all windows

### Model Training
- **Entry Point:** `Updated Extraction/train_model.py`
- **Hyperparameter Optimization:** `Updated Extraction/optimize_hyperparameters.py`
- **Best Config:** Saved in `Updated Extraction/best_configs.json`
- **Results:** `Updated Extraction/results/` directory

### Key Files
- `Updated Extraction/features.py` - Feature extraction functions
- `Updated Extraction/normalization.py` - Normalization logic
- `Updated Extraction/diagnose_model.py` - Model diagnostics
- `Updated Extraction/analyze_subjects.py` - Subject-level analysis

---

## Known Issues & Technical Debt

### Critical Issues (from MODEL_EVALUATION_CHECKLIST.md)
1. **Suspicious Hyperparameters:** Deep trees (17) + high min_samples constraints
2. **Missing Standard Deviations:** Results show std=0.0 (impossible for LOSO CV)
3. **Feature Selection Method:** Single RF model ranking may not generalize

### Repository Cleanup Needed
- [x] Delete empty/minimal files (`grant/test`, empty notebooks)
- [ ] Delete `grant/Feature Extraction/` (superseded)
- [ ] Delete `grant/Hyperparameter_Tuning/` (empty)
- [ ] Evaluate `tanner/` directory (merged or delete?)

### Documentation Gaps
- Need to consolidate context files into `Context/` structure
- Session handoff system needs implementation
- `/pass` and `/prime` commands need documentation

---

## Next Actions

**Last Updated:** 2026-02-16

### Immediate (This Session)
1. Complete repository cleanup - remove redundant files and directories
2. Reorganize .md files into `Context/` structure for lightweight handoff
3. Create `/pass` and `/prime` command documentation

### Short-term (Next Session)
1. Review model evaluation checklist and address critical issues
2. Verify LOSO CV standard deviation calculation bug
3. Test feature selection method alternatives

### Medium-term (This Week)
1. Finalize model selection based on evaluation checklist
2. Document model performance and hyperparameters
3. Prepare for self-testing protocol planning (February milestone)

### Long-term (This Month)
1. Define self-testing protocol (data collection procedures, stress tests)
2. Resolve model readiness criteria questions
3. Begin self-testing data collection

---

## Reference Information

### WESAD Dataset
- **Path:** `C:\Users\gloriosog\OneDrive - Milwaukee School of Engineering\Year 4 Courses\Semester 1\Senior Design\WESAD Dataset\WESAD2\WESAD`
- **Subjects:** S2-S17 (excluding S1, S12, S14)
- **Sampling Rates:** EDA=4Hz, BVP=64Hz, TEMP=4Hz, ACC=32Hz, Labels=700Hz

### Model Performance (Best Config)
- **Type:** Random Forest
- **Features:** 105 selected from 138 total
- **Metrics:** Recall=0.9325, Accuracy=0.9478, Precision=0.8852, F1=0.89
- **Validation:** LOSO CV (14 folds)

### Key Design Decisions
- TSST preparation phase excluded (first 3 windows = 180s)
- Baseline-only normalization (label=1 only for stats)
- cvxEDA-only decomposition (no fallbacks)
- Binary classification: Stress (label 2) vs Non-stress (labels 1,3,4)

---

## Session Handoff Notes

When updating this file:
- **Next Actions section (lines 260-275):** Update first 2-3 items under each timeframe
- **Last Updated date:** Update at top of Next Actions section
- **Use targeted edits:** Don't read full file, edit specific lines only

For full handoff, see `Context/HANDOFF_GUIDE.md`.
