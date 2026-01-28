# Models

Machine learning model training, evaluation, and inference.

## Contents

- `train.py` - Model training scripts (Random Forest, XGBoost, etc.)
- `evaluate.py` - LOSO cross-validation and performance metrics
- `inference.py` - Real-time inference pipeline
- `saved_models/` - Trained model checkpoints

## Usage

```python
from models.train import train_random_forest
from models.evaluate import loso_cross_validation
```

## Validation Strategy

Use Leave-One-Subject-Out (LOSO) cross-validation to prevent data leakage between subjects.
