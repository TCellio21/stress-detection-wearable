# WESAD Stress Detection - Machine Learning Pipeline

**AI-Driven Wearable Stress and Hyperactivity Detection for Adolescents**

This repository contains a comprehensive machine learning pipeline for developing stress and hyperactivity detection models using the WESAD (Wearable Stress and Affect Detection) dataset. The goal is to create models that can be adapted for an AI-driven wearable device to help adolescents (13-18 years) manage stress and hyperactivity in classroom settings.

## Project Overview

### Objective
Develop machine learning models to:
1. **Detect stress** from physiological signals (HRV, EDA, movement)
2. **Identify hyperactivity** patterns from accelerometer data
3. **Provide real-time predictions** suitable for edge deployment on wearable devices
4. **Support personalized interventions** through adaptive algorithms

### Target Application
- **Users**: Adolescents aged 13-18
- **Environment**: Classroom settings
- **Device**: Wrist-worn wearable (similar to Empatica E4)
- **Purpose**: Early detection of stress/hyperactivity to trigger coping strategies

## Dataset: WESAD

The WESAD dataset contains multimodal physiological data from 15 subjects experiencing different affective states:

### Data Collection
- **Subjects**: 15 adults (S2-S11, S13-S17)
- **Devices**:
  - RespiBAN (chest): ECG, EDA, EMG, Temperature, Respiration, 3-axis Accelerometer @ 700 Hz
  - Empatica E4 (wrist): BVP/PPG, EDA, Temperature, 3-axis Accelerometer @ various rates
- **Duration**: ~2 hours per subject
- **Conditions**: Baseline (1), Stress (2), Amusement (3), Meditation (4)

### Available Signals
- ✅ **ECG / BVP**: For heart rate variability (HRV) analysis
- ✅ **EDA**: Skin conductance (stress indicator)
- ✅ **Accelerometer**: Movement patterns (hyperactivity indicator)
- ✅ **Temperature**: Skin temperature
- ✅ **Respiration**: Breathing patterns (chest only)
- ✅ **Self-reports**: PANAS, STAI, SAM questionnaires

## Repository Structure

```
WESAD/
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Load and parse WESAD .pkl files
│   ├── preprocessing.py        # Signal filtering, normalization, segmentation
│   ├── feature_extraction.py   # Extract HRV, EDA, movement features
│   ├── models.py               # ML model implementations (to be created)
│   └── evaluation.py           # Model evaluation utilities (to be created)
├── notebooks/
│   ├── 01_data_exploration.ipynb        # Visualize and explore dataset
│   ├── 02_feature_engineering.ipynb     # Feature extraction pipeline (to be created)
│   ├── 03_baseline_models.ipynb         # Train baseline ML models (to be created)
│   ├── 04_deep_learning.ipynb           # Neural network models (to be created)
│   └── 05_wrist_only_models.ipynb       # Wrist-only deployment models (to be created)
├── data/                        # Processed datasets (to be generated)
├── models/                      # Trained models (to be saved)
├── results/                     # Evaluation results (to be saved)
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Installation

### Prerequisites
- Python 3.8+
- pip

### Setup
```bash
# Clone or navigate to the repository
cd WESAD

# Install dependencies
pip install -r requirements.txt
```

### Dependencies
Core libraries include:
- `numpy`, `pandas`, `scipy` - Data processing
- `neurokit2`, `biosppy`, `hrv-analysis` - Physiological signal processing
- `scikit-learn`, `xgboost`, `lightgbm` - Machine learning
- `torch`, `tensorflow` - Deep learning
- `matplotlib`, `seaborn`, `plotly` - Visualization
- `shap` - Model interpretability

## Usage

### 1. Data Loading

```python
from src.data_loader import WESADDataLoader

# Initialize loader
loader = WESADDataLoader('path/to/WESAD')

# Load subject data
pkl_data, metadata, questionnaire = loader.load_subject_data('S2')

# Extract signals
chest_signals = loader.extract_chest_signals(pkl_data)
wrist_signals = loader.extract_wrist_signals(pkl_data)
labels = loader.get_labels(pkl_data)
```

### 2. Preprocessing

```python
from src.preprocessing import SignalPreprocessor, RespiBAN_Converter

# Convert raw signals to SI units
ecg_mv = RespiBAN_Converter.convert_ecg(chest_signals['ECG'])
eda_us = RespiBAN_Converter.convert_eda(chest_signals['EDA'])

# Apply filters
preprocessor = SignalPreprocessor()
ecg_filtered = preprocessor.filter_ecg(ecg_mv, fs=700)
eda_filtered = preprocessor.filter_eda(eda_us, fs=700)

# Normalize
ecg_norm = preprocessor.normalize_zscore(ecg_filtered)

# Segment into windows
segments, seg_labels = preprocessor.segment_signal(
    eda_filtered, labels, window_size=60, step_size=30, fs=700
)
```

### 3. Feature Extraction

```python
from src.feature_extraction import MultimodalFeatureExtractor

# Extract features from window
signals = {
    'BVP': bvp_window,
    'EDA': eda_window,
    'ACC': acc_window,
    'TEMP': temp_window
}

fs_dict = {'BVP': 64, 'EDA': 4, 'ACC': 32, 'TEMP': 4}

features = MultimodalFeatureExtractor.extract_features_from_window(
    signals, fs_dict, device='wrist'
)

# Features include:
# - HRV: mean_hr, sdnn, rmssd, pnn50, lf, hf, lf_hf_ratio, sd1, sd2
# - EDA: tonic_mean, phasic_std, scr_count, scr_rate
# - ACC: mean_mag, std_mag, activity_intensity, jerk_mean, dominant_freq
# - TEMP: mean, std, slope
```

### 4. Model Training (Coming Soon)

```python
# Example workflow (to be implemented)
from src.models import StressClassifier
from src.evaluation import LOSOCrossValidation

# Train model
model = StressClassifier(model_type='random_forest')
model.train(X_train, y_train)

# Evaluate with Leave-One-Subject-Out CV
evaluator = LOSOCrossValidation()
results = evaluator.evaluate(model, X, y, subjects)
```

## Development Roadmap

### ✅ Phase 1: Data Infrastructure (Completed)
- [x] Data loader for .pkl files
- [x] Signal preprocessing utilities
- [x] Feature extraction modules
- [x] Data exploration notebook

### 🔄 Phase 2: Feature Engineering (In Progress)
- [ ] Complete feature extraction pipeline notebook
- [ ] Feature selection and dimensionality reduction
- [ ] Cross-subject feature analysis

### 📋 Phase 3: Baseline Models (Planned)
- [ ] Binary stress detection (Stress vs. Baseline)
- [ ] Traditional ML models (Random Forest, SVM, XGBoost)
- [ ] Leave-One-Subject-Out cross-validation
- [ ] Feature importance analysis

### 📋 Phase 4: Advanced Models (Planned)
- [ ] Multi-class classification (4 states)
- [ ] Deep learning models (CNN, LSTM, CNN-LSTM)
- [ ] Stress intensity regression
- [ ] Hyperactivity detection from accelerometer

### 📋 Phase 5: Wrist-Only Models (Planned)
- [ ] Retrain using only E4 wrist data
- [ ] Compare chest+wrist vs. wrist-only performance
- [ ] Optimize for deployment constraints

### 📋 Phase 6: Deployment Optimization (Planned)
- [ ] Real-time prediction pipeline
- [ ] Model compression (quantization, pruning)
- [ ] Edge deployment preparation (TensorFlow Lite, ONNX)
- [ ] Personalization algorithms

## Key Considerations

### Challenges & Solutions

**1. Age Mismatch**
- **Challenge**: WESAD has adult subjects (~28 years), target is teens (13-18)
- **Solution**: Use WESAD for algorithm development, plan for transfer learning and teen validation data

**2. Context Gap**
- **Challenge**: Lab-induced stress vs. real-world classroom stress
- **Solution**: Focus on generalizable physiological responses, fine-tune on classroom data later

**3. Missing Environmental Context**
- **Challenge**: WESAD lacks environmental noise detection
- **Solution**: Develop noise detection separately, integrate as additional modality

**4. Hyperactivity Labels**
- **Challenge**: WESAD doesn't explicitly label hyperactivity
- **Solution**: Use accelerometer variance as proxy, leverage movement patterns

### Technical Approach

**1. Wrist-Focused Development**
- Prioritize Empatica E4 (wrist) signals for practical deployment
- Use chest (RespiBAN) data for validation and comparison

**2. Robust Validation**
- Leave-One-Subject-Out (LOSO) cross-validation for generalization
- Per-subject analysis to understand individual variability

**3. Interpretability**
- Use SHAP values for feature importance
- Visualize attention weights in neural networks
- Ensure models are explainable for educational settings

**4. Personalization**
- Implement baseline calibration (first week of use)
- Adaptive thresholds per individual
- Continuous learning from user feedback

## Performance Metrics

Models will be evaluated using:
- **Accuracy**: Overall classification performance
- **Precision/Recall/F1**: Per-class performance
- **ROC-AUC**: Discriminative ability
- **Sensitivity**: Early detection capability
- **Inference Time**: Real-time deployment feasibility

## Citation

If you use this code or WESAD dataset, please cite:

```bibtex
@inproceedings{schmidt2018introducing,
  title={Introducing WESAD, a multimodal dataset for Wearable Stress and Affect Detection},
  author={Schmidt, Philip and Reiss, Attila and Duerichen, Robert and Marberger, Claus and Van Laerhoven, Kristof},
  booktitle={Proceedings of the 20th ACM international conference on multimodal interaction},
  pages={400--408},
  year={2018}
}
```

## License

This project is for research and educational purposes. Please refer to the WESAD dataset license for data usage terms.

## Contact

For questions about this implementation:
- **Developer**: Senior Design Team
- **Institution**: Milwaukee School of Engineering
- **Project**: AI-Driven Wearable for Adolescent Stress Management

---

**Note**: This is an active development project. Models are being developed and validated on adult data (WESAD) with the intention of adapting them for adolescent populations through transfer learning and additional data collection.
