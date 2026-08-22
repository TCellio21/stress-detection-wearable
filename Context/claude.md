# CLAUDE.md - Quick Status & Key Context

## Quick Status
**Last Updated:** 2026-02-16
**Current Phase:** Model Optimization & Repo Cleanup
**Active Directory:** `Updated Extraction/`

### Current Status
- ✅ Feature extraction pipeline complete (138 features: 46 raw + 92 normalized)
- ✅ Hyperparameter optimization complete (best config saved in `best_configs.json`)
- ✅ Model training with LOSO CV implemented
- 🔄 Repository cleanup in progress
- 📋 Next: Finalize model selection and prepare for self-testing protocol

### Next Steps
1. **Immediate:** Complete repository cleanup (remove redundant files)
2. **Short-term:** Review model evaluation checklist and address critical issues
3. **Pending:** Define self-testing protocol for February milestone

---

## Key Context

### Project Overview
This project aims to **train a model to detect stress** using physiological sensor data from the **WESAD dataset**. The long-term goal is to enable real-time stress detection for wearable systems.

### Main Directories
- `Updated Extraction/` → **ACTIVE** pipeline (merged Grant + Tanner approaches)
  - `dataset_builder.py` - Feature extraction orchestrator
  - `train_model.py` - Model training with LOSO CV
  - `optimize_hyperparameters.py` - Hyperparameter optimization
  - `all_subject_features_updated.csv` - Current feature dataset (1.1MB)
  - `best_configs.json` - Optimized hyperparameters (trial #48)
- `grant/context_reboot/` → Context documentation (TECHNICAL_STATE.md, SESSION_LOG.md, etc.)
- `docs/` → WESAD documentation and project advice
- `tanner/` → Personal working directory (may be superseded)

### Current Model State
- **Best Config:** Random Forest (trial #48)
  - Recall: 0.9325 ± 0.1487
  - Accuracy: 0.9478 ± 0.103
  - Precision: 0.8852 ± 0.1802
  - F1: 0.89 ± 0.1612
- **Features:** 105 selected features from 138 total
- **Validation:** Leave-One-Subject-Out (LOSO) cross-validation

### Key Design Decisions
- **Time continuity preserved:** Windows created on continuous time axis, then filtered by label
- **TSST preparation excluded:** First 3 stress windows (180s) dropped per subject
- **S14 excluded:** Data quality / atypical responder
- **cvxEDA-only:** No fallback methods (ensures consistency)
- **Binary labels:** Stress (label 2) vs Non-stress (labels 1,3,4)
- **Baseline-only normalization:** Stats computed from true baseline windows only

### Data Context (WESAD)
- **Subjects:** 15 (S2-S17, excluding S1, S12, S14)
- **Signals:** EDA (4 Hz), BVP (64 Hz), TEMP (4 Hz), ACC (32 Hz)
- **Window Size:** 60 seconds (non-overlapping)
- **Always use subject-wise splits** (LOSO) to prevent data leakage

---

## Workflow
When starting a session:
1. Review `Context/claude.md` Quick Status section (lines 8-36)
2. Check `Context/WORKSPACE_NOTES.md` Next Actions section (lines 260-275)
3. Summarize what you learned and outline a short plan before changing code
4. Make small, testable updates with clear comments
5. Use targeted updates for session handoff (see `/pass` command pattern)

---

## Guardrails
- ❌ Don't train or evaluate with subject data leakage
- ❌ Don't modify multiple files without summarizing the plan first
- ❌ Don't hardcode dataset paths or secrets
- ❌ Don't ignore failing tests or warnings

---

## Session Handoff Commands

### `/pass` - Lightweight Session Handoff
Updates minimal context files:
- `Context/claude.md` lines 8-36 (Quick Status section)
- `Context/WORKSPACE_NOTES.md` lines 260-275 (Next Actions section)
- `Context/PHASE2_PLAN.md` Section 15 (Changelog - append only)

### `/prime` - Bootstrap Fresh Instance
Reads minimal context:
- `Context/claude.md` lines 1-75 (Quick Status + Key Context)
- `Context/WORKSPACE_NOTES.md` lines 260-316 (Next Actions)

See `Context/HANDOFF_GUIDE.md` for detailed usage.
