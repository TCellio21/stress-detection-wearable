# Personalization

Individual baseline calibration and normalization for stress detection.

## Contents

- `baseline_estimation.py` - Estimate individual physiological baselines
- `normalization.py` - Z-score and percentile normalization using personal baselines

## Purpose

Personalization addresses the age gap between WESAD adults (28±3 years) and target adolescents (13-18 years) by using relative features instead of absolute values.

## Usage

```python
from personalization.baseline_estimation import estimate_baseline
from personalization.normalization import normalize_to_baseline
```
