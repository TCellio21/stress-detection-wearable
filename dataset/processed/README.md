# Processed Data

Windowed and cleaned feature data ready for machine learning.

## Contents

- `wesad_binary_stress_wrist_features.csv` - Extracted features with binary stress labels

## Data Format

- 1,476 windows (60-second windows with 50% overlap)
- 52 features (HRV, EDA, ACC, TEMP)
- Binary labels: Stress (1) vs Non-stress (0)
- Class distribution: 22.5% stress, 77.5% non-stress

## Generation

Run the feature extraction pipeline:

```bash
python tanner/process_binary_labels.py
```
