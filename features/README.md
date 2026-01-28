# Features

Feature extraction modules for physiological signals.

## Contents

- `hrv_features.py` - Heart rate variability features (SDNN, RMSSD, pNN50, LF/HF ratio, etc.)
- `eda_features.py` - Electrodermal activity features (SCR count, tonic/phasic components)
- `motion_features.py` - Accelerometer features (magnitude, jerk, activity intensity)
- `feature_config.yaml` - Feature extraction configuration

## Usage

```python
from features.hrv_features import extract_hrv_features
from features.eda_features import extract_eda_features
```

## Notes

- Currently, feature extraction code lives in `tanner/feature_extraction.py`
- Extracts 52 features across HRV, EDA, ACC, and TEMP signals
