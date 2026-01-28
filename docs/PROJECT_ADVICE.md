# Project Advice: Adapting WESAD for Adolescent Stress Detection

## Executive Summary

You're building an **AI-driven wearable for adolescent stress and hyperactivity management** using the WESAD dataset as your foundation. This document provides strategic guidance on how to approach this project effectively.

---

## Understanding the Gap: WESAD → Your Application

### What WESAD Provides ✅
- High-quality physiological signals (HRV, EDA, movement)
- Clear stress labels from controlled experiments
- Wrist-worn device data (E4) - practical for deployment
- Multiple subjects for generalization testing
- Validated protocol and ground truth

### What WESAD Lacks ❌
- **Age group**: Adults (28±3 years) vs. your target (13-18 years)
- **Context**: Lab setting vs. classroom environment
- **Activity**: TSST stress test vs. academic/social stressors
- **Hyperactivity labels**: No explicit ADHD or hyperactivity annotations
- **Environmental data**: No noise/distraction measurements

### The Strategy 🎯
**Use WESAD to develop the core algorithms, then adapt for your population.**

Think of it as:
1. WESAD = Training wheels for algorithm development
2. Teen data collection = Fine-tuning for deployment
3. Classroom integration = Real-world validation

---

## Critical Technical Decisions

### 1. Which Device Data to Use?

**Recommendation: Focus on WRIST (Empatica E4) data**

**Why?**
- ✅ Practical for teens (wrist watches are socially acceptable)
- ✅ Non-invasive (no chest strap needed)
- ✅ Contains key signals: BVP (for HRV), EDA, Accelerometer, Temperature
- ✅ Easier to wear all day in classroom

**Trade-off:**
- ❌ Slightly lower ECG quality from BVP vs. chest ECG
- ❌ No respiration data from wrist device

**Action Items:**
1. Develop models using ONLY wrist data from the start
2. Use chest data for comparison/validation only
3. Ensure all features can be computed from wrist sensors

---

### 2. Feature Engineering Strategy

**Recommendation: Multi-level feature hierarchy**

#### Level 1: Core Physiological Features (Most Important)
Focus on these first - they're well-validated for stress:

**From BVP/HRV:**
- `mean_hr`: Average heart rate (↑ in stress)
- `rmssd`: HRV time-domain metric (↓ in stress)
- `lf_hf_ratio`: Sympathetic/parasympathetic balance (↑ in stress)
- `sdnn`: Overall HRV (↓ in stress)

**From EDA:**
- `tonic_mean`: Baseline skin conductance (↑ in stress)
- `phasic_std`: SCR variability (↑ in stress)
- `scr_count`: Number of peaks (↑ in stress/arousal)

**From Accelerometer (for hyperactivity):**
- `std_mag`: Movement variability (↑ in hyperactivity)
- `activity_intensity`: Overall activity level
- `jerk_mean`: Fidgeting indicator (↑ in restlessness)

#### Level 2: Temporal Context Features
Add these after core features work:
- Trends over 5-10 minute windows
- Rate of change between windows
- Variance across recent history

#### Level 3: Personalized Features
Final layer for deployment:
- Deviation from individual baseline
- Time-of-day normalization
- Context-aware adjustments

**Why this matters:**
Start simple, validate, then add complexity. Don't build Level 3 until Level 1 works!

---

### 3. Model Development Path

**Recommendation: Progressive complexity**

#### Stage 1: Binary Classification (WEEKS 6-8)
**Task:** Stress (label=2) vs. Baseline (label=1)

**Models to try (in order):**
1. **Random Forest** - Start here
   - Pros: Fast, interpretable, robust to outliers
   - Cons: Not time-aware
   - Expected accuracy: 75-85% with good features

2. **XGBoost** - Try second
   - Pros: Often best performance, built-in feature importance
   - Cons: More hyperparameters
   - Expected accuracy: 80-90%

3. **Logistic Regression** - For baseline
   - Pros: Extremely interpretable, fast inference
   - Cons: Assumes linear separability
   - Expected accuracy: 70-80%

**Success criteria:**
- LOSO cross-validation accuracy > 75%
- Consistent performance across subjects (check std dev)

#### Stage 2: Multi-Class (WEEKS 9-10)
**Task:** Baseline vs. Stress vs. Amusement vs. Meditation

**Why this matters:**
- Tests if model distinguishes stress from other arousal states
- Amusement = high arousal, positive valence (unlike stress)
- Meditation = low arousal, positive valence

**Expected challenge:**
Stress vs. Amusement will be hardest (both have ↑ HR, ↑ EDA)

**Key differentiator:**
- HRV patterns (stress has ↓ RMSSD, amusement may not)
- EDA temporal dynamics (stress = sustained, amusement = episodic)

#### Stage 3: Deep Learning (WEEKS 11-14)
**When to use:** Only if traditional ML plateaus < 80% accuracy

**Architectures:**
1. **1D CNN** on raw BVP/EDA
   - For automatic pattern detection
   - 3-5 convolutional layers

2. **LSTM** on feature sequences
   - For temporal dependencies
   - Bidirectional LSTM recommended

3. **CNN-LSTM Hybrid**
   - CNN for feature extraction + LSTM for temporal
   - Best performance but higher complexity

**Warning:**
- Deep learning needs more data
- 15 subjects may be borderline sufficient
- Data augmentation (time warping, jittering) will be essential

---

### 4. Handling the Age Gap

**The Problem:**
Physiological responses differ between adults and teens:
- Baseline HR higher in adolescents (~70-80 vs. 60-70 bpm)
- HRV typically lower in teens
- Stress reactivity patterns may differ

**Solutions:**

#### Short-term (Use WESAD directly):
1. **Relative features instead of absolute**
   - Use z-score normalization per subject
   - Focus on "deviation from baseline" not raw values
   - Example: Instead of HR=100, use "HR is 1.5 std above subject's baseline"

2. **Feature ratios**
   - LF/HF ratio (already relative)
   - EDA phasic/tonic ratio
   - HR variability coefficients

3. **Transfer learning mindset**
   - Train on WESAD to learn "what stress looks like physiologically"
   - The relative patterns (↑ HR, ↓ HRV, ↑ EDA) should generalize

#### Long-term (For deployment):
1. **Collect small adolescent validation set**
   - Even 10-20 teen subjects helps
   - Use WESAD model as pretrained base
   - Fine-tune top layers on teen data

2. **Personalization calibration**
   - First week of use = baseline establishment
   - Continuously adapt thresholds to individual
   - Use unsupervised learning to update baselines

---

### 5. Hyperactivity Detection

**The Challenge:**
WESAD has no explicit hyperactivity labels.

**Solution: Create Proxy Labels**

#### Method 1: Movement-Based Clustering
```python
# High-level pseudocode
for each subject:
    acc_variance = std(accelerometer_magnitude)

    # Define hyperactive as high variance during "calm" periods
    baseline_windows = get_windows(label=baseline)
    high_activity_baseline = baseline_windows[acc_variance > threshold]

    # Label these as "hyperactive-like"
```

#### Method 2: Multi-Modal Activity Score
Combine:
- Accelerometer variance (primary indicator)
- Jerk magnitude (fidgeting)
- Posture changes (sitting still vs. moving)
- HR elevation (arousal + movement)

Create composite score:
```
hyperactivity_score = 0.5 * acc_std + 0.2 * jerk + 0.2 * hr_elevation + 0.1 * eda
```

**Validation:**
- Compare to self-reported "restlessness" in questionnaires
- Cross-reference with stress condition (may overlap)

#### Method 3: Unsupervised Clustering
- Extract movement features from all conditions
- Use K-means or GMM to find natural clusters
- Inspect clusters: one should be "high movement" group

**For Your Application:**
Hyperactivity detection may need additional data collection with labeled ADHD/hyperactive subjects. WESAD can establish the feature extraction pipeline, but labels will be weak.

---

## Validation Strategy

### Why Leave-One-Subject-Out (LOSO)?

**Standard k-fold CV is WRONG for this task!**

**Problem with k-fold:**
- Same subject appears in train and test
- Model learns individual patterns
- Overestimates generalization to new people

**LOSO cross-validation:**
```python
for subject in all_subjects:
    train_data = all_subjects - subject
    test_data = subject

    train_model(train_data)
    evaluate(test_data)

average_across_all_folds()
```

**This tests:** "Can the model work on a NEW person it's never seen?"

**Expected result:**
- LOSO accuracy will be 10-15% lower than k-fold
- This is the REAL performance estimate
- If LOSO < 70%, model won't generalize well

### Metrics That Matter

**For Stress Detection:**
1. **Sensitivity (Recall)** - Can't miss stress events
   - Target: > 85%
   - Missing stress = student doesn't get help

2. **Specificity** - Don't false alarm too much
   - Target: > 75%
   - Too many false alarms = user ignores device

3. **F1-Score** - Balance of both
   - Target: > 0.80

4. **Inference Time** - Must be real-time
   - Target: < 100ms per prediction
   - Measure on target hardware (e.g., Raspberry Pi)

**For Hyperactivity:**
- Harder to validate without ground truth
- Focus on face validity (does it match intuition?)
- Plan for expert validation or teacher feedback

---

## Data Augmentation (When You Need More)

With only 15 subjects, you may need augmentation for deep learning:

### Time-Domain Augmentation
1. **Jittering**: Add small Gaussian noise (σ = 0.01-0.05 × signal_std)
2. **Scaling**: Multiply by 0.9-1.1
3. **Time Warping**: Stretch/compress time axis by ±10%

### Window-Based Augmentation
4. **Sliding Windows**: Overlap windows by 75% instead of 50%
5. **Random Cropping**: Take random subwindows
6. **Mixup**: Blend features from same class

### Important
- Only augment TRAINING set, never test set
- Validate that augmented data looks realistic
- More augmentation ≠ better (can hurt if overdone)

---

## Deployment Considerations

### Real-Time Pipeline

Your system will need:

```
[Sensors] → [Signal Buffer] → [Preprocessing] → [Feature Extraction] → [ML Model] → [Prediction]
     ↓                                                                           ↓
  Hardware                                                                  Action
```

**Critical constraints:**
1. **Latency**: Prediction in < 1 second
2. **Memory**: Limited on wearable hardware
3. **Battery**: Can't drain phone/device too fast
4. **Robustness**: Must handle missing data, artifacts

**Design choices:**

#### Window Size
- Shorter (30s): Faster detection, more updates, noisier
- Longer (120s): More stable, slower to detect onset
- **Recommendation**: 60s window, 30s step (2× per minute updates)

#### Model Complexity
- Random Forest: 100 trees → ~5MB model, <50ms inference
- XGBoost: 500 trees → ~10MB model, ~100ms inference
- LSTM: Depends on size, can be 1-50MB, 50-500ms

**Recommendation**: Start with Random Forest, optimize later

#### On-Device vs. Cloud
- **On-device**: Privacy, no latency, works offline
- **Cloud**: More compute, easier updates, requires connectivity

**Recommendation**: Hybrid
- Core prediction on-device
- Model updates from cloud
- Detailed logging to cloud (opt-in)

---

## Personalization Strategy

**Why personalization matters:**
- Individual baseline HR varies 60-100 bpm
- Some people are "high EDA responders", others low
- Context matters (morning vs. afternoon, subject type)

### Calibration Period

**Week 1: Baseline Establishment**
```
Day 1-3: No predictions, just collect baseline
- Morning baseline (wake-up)
- Afternoon baseline (calm periods)
- Estimate individual thresholds

Day 4-7: Shadow mode
- Make predictions but don't show user
- Collect feedback
- Adjust thresholds

Week 2+: Active use
- Full predictions
- Continuous adaptation
```

### Adaptive Thresholds

Instead of fixed threshold:
```python
stress_detected = (current_HRV < baseline_HRV - 1.5*std)
```

Use adaptive:
```python
baseline_HRV = 0.95 * baseline_HRV + 0.05 * recent_calm_HRV  # Exponential moving average
stress_detected = (current_HRV < baseline_HRV - adaptive_threshold)
```

### User Feedback Loop

Allow user to confirm/deny:
- "Was that really stressful?" → Update model
- Active learning: Retrain on corrected labels
- Improves over time

---

## Next Steps: Your Development Timeline

### Weeks 1-2 ✅ (COMPLETED)
- [x] Set up data pipeline
- [x] Explore WESAD structure
- [x] Implement preprocessing
- [x] Build feature extraction

### Weeks 3-4 (NEXT - DO THIS)
1. **Run `quick_start.py`** - Verify everything works
2. **Open `01_data_exploration.ipynb`** - Visualize all subjects
3. **Create feature dataset**:
   ```python
   # Pseudocode
   for subject in all_subjects:
       for window in segment_subject(window_size=60s):
           features = extract_features(window)
           label = get_majority_label(window)
           save_to_dataset(features, label)
   ```
4. **Analyze feature distributions** across conditions
5. **Select top features** using:
   - Univariate tests (t-test, ANOVA)
   - Mutual information
   - Correlation analysis

### Weeks 5-6 (BASELINE MODELS)
1. **Train Random Forest** on stress vs. baseline
2. **Implement LOSO cross-validation**
3. **Feature importance analysis** (SHAP values)
4. **Try XGBoost, SVM** for comparison
5. **Tune hyperparameters** (grid search)

### Weeks 7-8 (MULTI-CLASS & HYPERACTIVITY)
1. **Extend to 4-class** problem
2. **Create hyperactivity proxy labels**
3. **Train separate hyperactivity model**
4. **Analyze failure cases**

### Weeks 9-10 (WRIST-ONLY OPTIMIZATION)
1. **Retrain using ONLY E4 wrist signals**
2. **Compare performance** to chest+wrist
3. **Optimize for inference speed**
4. **Model compression** if needed

### Weeks 11-12 (DEPLOYMENT PREP)
1. **Real-time pipeline** implementation
2. **Test on streaming data**
3. **Measure latency, memory, battery**
4. **Export model** (TensorFlow Lite, ONNX)

---

## Common Pitfalls to Avoid

### 1. Data Leakage
❌ **Wrong**: Normalize all data, then split train/test
✅ **Right**: Split first, then normalize train, apply train stats to test

### 2. Feature Engineering on Test Set
❌ **Wrong**: Use all data to select features
✅ **Right**: Feature selection only on training folds

### 3. Ignoring Class Imbalance
- WESAD has unequal class sizes (baseline > stress usually)
- Use class weights in models
- Consider SMOTE for oversampling

### 4. Overfitting to Lab Context
- Don't assume lab stress = classroom stress
- Focus on generalizable physiological responses
- Plan for domain adaptation

### 5. Not Testing Edge Cases
- What if sensor disconnects?
- What if user is exercising (high HR but not stressed)?
- What if battery low affects signal quality?

---

## Resources & Further Reading

### Datasets to Consider Later
1. **SWELL-KW**: Office stress (more realistic context)
2. **K-EmoCon**: 32 subjects, multimodal emotion
3. **CASE**: Student stress dataset

### Key Papers
1. Schmidt et al. (2018) - WESAD paper
2. Healey & Picard (2005) - Automotive stress detection
3. Gjoreski et al. (2020) - Machine learning for stress

### Tools
- **Neurokit2**: Excellent for HRV, EDA processing
- **Heartpy**: Alternative HRV analysis
- **BioPsyKit**: Physiological signal toolkit
- **SHAP**: Model interpretability

---

## Final Advice

### Start Simple, Iterate Fast
1. Get wrist-only binary classifier working first
2. Validate on WESAD (LOSO CV)
3. Then add complexity (multi-class, deep learning, personalization)

### Document Everything
- Keep a lab notebook of experiments
- Track all hyperparameters
- Save all trained models
- Version your data preprocessing

### Think About the End User
- A 15-year-old won't tolerate false alarms
- Privacy is critical (don't log sensitive data)
- Battery life matters
- UI should be simple, not overwhelming

### Plan for Real-World Testing
- WESAD gets you 70% of the way
- The last 30% requires teen data
- Budget time for IRB, data collection, iteration

---

## Questions to Ask Yourself

As you develop, regularly check:

1. **Generalization**: Does this work on held-out subjects?
2. **Interpretability**: Can I explain WHY the model predicted stress?
3. **Fairness**: Does it work equally well for all demographics?
4. **Robustness**: What happens with noisy/missing data?
5. **Usability**: Would a teen actually wear this device?

---

## Success Metrics for Your Senior Design

### Technical Success
- ✅ LOSO accuracy > 75% for binary stress detection
- ✅ Wrist-only model within 5% of chest+wrist performance
- ✅ Inference time < 100ms
- ✅ Model explainability (SHAP, feature importance)

### Project Success
- ✅ Complete ML pipeline from raw data to deployment
- ✅ Clear documentation and reproducible results
- ✅ Demonstration on real-time data
- ✅ Thoughtful analysis of limitations and future work

### Impact Success (Beyond Senior Design)
- ✅ Framework adaptable to teen population
- ✅ Ethical considerations addressed
- ✅ Path to real-world testing defined
- ✅ Potential for helping students with stress/ADHD

---

**Remember**: You're not trying to solve the complete problem in one semester. You're building a **proof-of-concept** that shows the approach is viable. WESAD gives you the data to prove the core technology works. The adaptation to teens and classrooms is the next phase.

**Good luck with your project! 🚀**
