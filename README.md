# WESAD Stress Detection - Machine Learning Pipeline

**AI-Driven Wearable Stress and Hyperactivity Detection for Adolescents**

A complete machine learning pipeline for binary stress classification using the WESAD dataset. Extracts 52 physiological features from wrist-worn sensors for real-time stress detection on wearable devices.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run feature extraction pipeline
python scripts/process_binary_labels.py
```

## Project Status

| Component | Status | Output |
|-----------|--------|--------|
| Data Loading | ✅ Complete | 15 subjects loaded |
| Preprocessing | ✅ Complete | Filtered & segmented signals |
| Feature Extraction | ✅ Complete | 52 features per window |
| Binary Classification | ✅ Complete | 1,476 labeled windows |
| Model Training | 🔄 In Progress | See `reports/` |

## Dataset: WESAD

- **Subjects**: 15 adults (S2-S11, S13-S17)
- **Devices**: RespiBAN (chest, 700 Hz) + Empatica E4 (wrist, 4-64 Hz)
- **Labels**: Baseline (1), Stress (2), Amusement (3), Meditation (4)
- **Binary Mapping**: Stress (1) vs Non-stress (0)

## Repository Structure

```
WESAD/
├── src/                         # Core pipeline modules
│   ├── __init__.py
│   ├── config.py               # Configuration loader
│   ├── data_loader.py          # Load WESAD .pkl files
│   ├── preprocessing.py        # Signal filtering & segmentation
│   └── feature_extraction.py   # Extract 52 features (HRV, EDA, ACC, TEMP)
│
├── scripts/                     # Runnable entry points
│   └── process_binary_labels.py # Main feature extraction pipeline
│
├── notebooks/                   # Exploratory analysis
│   ├── 01_data_exploration.ipynb
│   ├── 02_subject2_detailed_analysis.ipynb
│   └── 03_verify_binary_data.ipynb
│
├── data/
│   ├── raw/                    # WESAD .pkl files (S2-S17)
│   └── processed/              # Extracted features (CSV/pickle)
│
├── docs/                        # Technical documentation
│   ├── DATA_STRUCTURE_EXPLANATION.md
│   └── SAMPLING_RATE_ANALYSIS.md
│
├── config.yaml                  # Central configuration
└── requirements.txt             # Python dependencies
```

## Installation

```bash
pip install -r requirements.txt
```

**Key Dependencies:**
- `numpy`, `pandas`, `scipy` - Data processing
- `neurokit2` - HRV and EDA analysis (uses NeuroKit2 for peak detection)
- `scikit-learn`, `xgboost` - Machine learning
- `pyyaml` - Configuration loading

## Usage

### Run Full Pipeline
```bash
python scripts/process_binary_labels.py
```

**Output:** `data/processed/wesad_binary_stress_wrist_features.csv` (1,476 windows × 52 features)

### Use Individual Modules

```python
from src.data_loader import WESADDataLoader
from src.preprocessing import SignalPreprocessor, E4_Converter
from src.feature_extraction import MultimodalFeatureExtractor

# Load data
loader = WESADDataLoader('data/raw')
pkl_data = loader.load_subject_pkl('S2')
wrist_signals = loader.extract_wrist_signals(pkl_data)

# Convert to SI units
signals_si = {
    'BVP': E4_Converter.convert_bvp(wrist_signals['BVP']),
    'EDA': E4_Converter.convert_eda(wrist_signals['EDA']),
    'ACC': E4_Converter.convert_acc(wrist_signals['ACC']),
    'TEMP': E4_Converter.convert_temp(wrist_signals['TEMP'])
}

# Extract features (52 total)
fs_dict = {'BVP': 64, 'EDA': 4, 'ACC': 32, 'TEMP': 4}
features = MultimodalFeatureExtractor.extract_features_from_window(
    signals_si, fs_dict, device='wrist'
)
```

## Extracted Features (52 Total)

| Category | Count | Examples |
|----------|-------|----------|
| HRV | 14 | mean_hr, sdnn, rmssd, pnn50, lf, hf, lf_hf_ratio, sd1, sd2 |
| EDA | 17 | tonic_mean, phasic_std, scr_count, scr_rate, scr_amplitude |
| ACC | 15 | mean_mag, std_mag, activity_intensity, jerk_mean, dominant_freq |
| TEMP | 8 | mean, std, slope, range, rate_mean |

## Processed Data Summary

- **Total Windows:** 1,476 (60-second windows, 50% overlap)
- **Stress Windows:** 332 (22.5%)
- **Non-stress Windows:** 1,144 (77.5%)
- **Validation Strategy:** Leave-One-Subject-Out (LOSO)

## Files Required for Feature Extraction

To run the feature extraction pipeline, you need these **8 files**:

```
src/
├── __init__.py              # Package marker
├── config.py                # Configuration loader
├── data_loader.py           # WESADDataLoader class
├── preprocessing.py         # SignalPreprocessor, E4_Converter
└── feature_extraction.py    # All feature extractors

config.yaml                  # Parameters (window size, sampling rates)
requirements.txt             # Dependencies
scripts/process_binary_labels.py  # Main entry point
```

## Configuration

All parameters are centralized in `config.yaml`:

```yaml
preprocessing:
  window_size: 60  # seconds
  step_size: 30    # seconds (50% overlap)

sampling_rates:
  wrist_bvp: 64
  wrist_eda: 4
  wrist_acc: 32
  wrist_temp: 4

labels:
  binary_mapping:
    1: 0  # baseline -> non-stress
    2: 1  # stress -> stress
    3: 0  # amusement -> non-stress
    4: 0  # meditation -> non-stress
```

## Citation

```bibtex
@inproceedings{schmidt2018introducing,
  title={Introducing WESAD, a multimodal dataset for Wearable Stress and Affect Detection},
  author={Schmidt, Philip and Reiss, Attila and Duerichen, Robert and Marberger, Claus and Van Laerhoven, Kristof},
  booktitle={Proceedings of the 20th ACM international conference on multimodal interaction},
  pages={400--408},
  year={2018}
}
```

## Project Info

- **Team**: Senior Design, Milwaukee School of Engineering
- **Goal**: AI-driven wearable stress detection for adolescents (13-18)
- **Status**: Feature extraction complete, model training in progress
