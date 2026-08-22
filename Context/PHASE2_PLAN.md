# Phase 2 Plan - Model Optimization & Deployment

## Overview
Phase 2 focuses on model optimization, evaluation, and preparation for real-world testing.

## Goals
1. Optimize hyperparameters for best model performance
2. Evaluate model robustness and generalization
3. Prepare for self-testing protocol (February milestone)
4. Clean up repository structure

---

## Milestone 1: Self-Testing & Model Refinement (February)

### Testing Protocol Planning
- [ ] Define data collection procedures
- [ ] Design physical stress tests (e.g., cold exposure)
- [ ] Design emotional stress tests
- [ ] Create testing protocol document

### Open Questions
- **Data Collection Parameters:**
  - How many sessions per person?
  - How long per session?
  - What ground truth will we use?
  
- **Model Readiness Criteria:**
  - What performance metrics indicate readiness?
  - How will we know the model is ready?
  
- **Timeline:**
  - Timeline for self-testing data collection?
  - Timeline for model refinement?

### ML Model & Repo Cleanup
- [x] Choose a final model (Random Forest)
- [x] Optimize hyperparameters (trial #48 complete)
- [ ] Clean up and refine repository structure
- [ ] Document model selection and performance

---

## Milestone 2: Model Refinement & School Testing Preparation (March)

### Model Refinement
- [ ] Continue refining ML model based on self-testing results
- [ ] Validate model improvements
- [ ] Finalize model for deployment

### School Testing Outreach & Planning
- [ ] Reach out to schools/teachers for testing opportunities
- [ ] Establish testing partnerships
- [ ] Plan school testing protocol
- [ ] Prepare consent forms and documentation
- [ ] Schedule testing dates

---

## Changelog

**Section 15 - Changelog (append only)**

### 2026-02-16
- Repository cleanup initiated
- Deleted empty/minimal files (`grant/test`, empty notebooks)
- Created `Context/` directory structure for lightweight session handoff
- Reorganized documentation into standardized structure

### 2026-02-01
- Hyperparameter optimization completed (trial #48)
- Best config saved: Random Forest with 105 features
- Model performance: Recall=0.9325, Accuracy=0.9478, Precision=0.8852, F1=0.89

### 2026-01-29
- Feature extraction pipeline finalized (138 features)
- Dataset built: `all_subject_features_updated.csv`
- Model training with LOSO CV implemented

---

## Notes
- Use `/pass` command to append changelog entries (Section 15 only)
- Keep entries brief (1-2 lines per entry)
- Focus on significant changes, not minor updates
