# Preprocessing

Signal preprocessing pipeline for physiological data from wearable sensors.

## Contents

- `filtering.py` - Bandpass/lowpass/highpass filters for sensor signals
- `artifact_removal.py` - Motion artifact detection and removal
- `windowing.py` - Sliding window segmentation for feature extraction

## Usage

```python
from preprocessing.filtering import butter_filter
from preprocessing.windowing import create_windows
```

## Notes

- Currently, preprocessing code lives in `tanner/preprocessing.py`
- Code will be migrated here as it matures
