"""
V3 stress-detection inference module.

Three trained models are bundled in this directory:

    - default_hgb         (Phase 6 lock; HGB defaults; F1=0.931 LOSO)
    - tuned_hgb           (Phase 7 nested-LOSO Optuna; F1=0.917 unbiased)
    - ensemble_hgb_svm    (Phase 7 side experiment; F1=0.935; recovers S3) ← recommended

Each is a `StressModel` instance pickled to disk. Load with `load_model(name)`
and call `.predict_proba(X)` or `.predict(X)` on a feature array or DataFrame.

The expected feature list is the 16 features from Phase 5 — see
`feature_list.json` or any model's `.feature_list` attribute. Order matters
when passing a numpy array; pass a DataFrame with named columns to avoid
ordering bugs.

Usage::

    from models.inference import load_model
    model = load_model('ensemble')      # or 'default' or 'tuned'
    p = model.predict_proba(features_df)  # 1D array of P(stress)
    y = model.predict(features_df)        # 1D array of {0, 1}
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


_MODELS_DIR = Path(__file__).resolve().parent

_NAME_ALIASES = {
    'default': 'default_hgb',
    'tuned': 'tuned_hgb',
    'ensemble': 'ensemble_hgb_svm',
}


@dataclass
class StressModel:
    """A trained stress-detection model bundled with its feature list, threshold,
    and metadata. Supports both single-estimator models and probability-averaging
    ensembles (multiple estimators, predictions averaged).
    """
    name: str
    feature_list: list[str]
    estimators: list[Any] = field(default_factory=list)
    threshold: float = 0.5
    metadata: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def predict_proba(self, X) -> np.ndarray:
        """Return P(stress) for each window. 1D array, length n_samples.

        X may be a DataFrame (column names are matched to feature_list) or a
        2D numpy array (columns must be in feature_list order).
        """
        X = self._prepare_X(X)
        probas = [est.predict_proba(X)[:, 1] for est in self.estimators]
        return np.mean(probas, axis=0)

    def predict(self, X) -> np.ndarray:
        """Return binary stress label (0 = non-stress, 1 = stress) per window."""
        return (self.predict_proba(X) >= self.threshold).astype(int)

    # ------------------------------------------------------------------ #
    # Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _prepare_X(self, X) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            missing = [f for f in self.feature_list if f not in X.columns]
            if missing:
                raise ValueError(
                    f"Input DataFrame missing required features: {missing}\n"
                    f"Expected columns: {self.feature_list}")
            return X[self.feature_list].values
        arr = np.asarray(X)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2:
            raise ValueError(f"Expected 1D or 2D input, got shape {arr.shape}")
        if arr.shape[1] != len(self.feature_list):
            raise ValueError(
                f"Expected {len(self.feature_list)} features, got {arr.shape[1]}. "
                f"To avoid ordering errors, pass a DataFrame with named columns."
            )
        return arr

    def __repr__(self) -> str:
        return (f"StressModel(name={self.name!r}, "
                f"n_features={len(self.feature_list)}, "
                f"n_estimators={len(self.estimators)}, "
                f"threshold={self.threshold})")


# ====================================================================== #
# Module-level utilities                                                  #
# ====================================================================== #

def load_model(name: str = 'ensemble') -> StressModel:
    """Load a saved StressModel by name.

    Args:
        name: One of 'default_hgb' (or 'default'), 'tuned_hgb' (or 'tuned'),
              'ensemble_hgb_svm' (or 'ensemble', the default).

    Returns:
        StressModel instance ready for `.predict()` / `.predict_proba()`.
    """
    file_name = _NAME_ALIASES.get(name, name)
    path = _MODELS_DIR / f'{file_name}.pkl'
    if not path.exists():
        available = sorted(p.stem for p in _MODELS_DIR.glob('*.pkl'))
        raise FileNotFoundError(
            f"Model {name!r} not found at {path}.\n"
            f"Available: {available}\n"
            f"Aliases: {_NAME_ALIASES}"
        )
    with open(path, 'rb') as f:
        model = pickle.load(f)
    if not isinstance(model, StressModel):
        # Backward compat for older pickles in this directory
        raise TypeError(
            f"Loaded object from {path.name} is {type(model).__name__}, not StressModel. "
            f"Re-run models/build_models.py to regenerate."
        )
    return model


def list_models() -> dict:
    """Return a summary dict of all bundled models in this directory."""
    out: dict = {}
    for path in sorted(_MODELS_DIR.glob('*.pkl')):
        try:
            with open(path, 'rb') as f:
                m = pickle.load(f)
            if isinstance(m, StressModel):
                out[m.name] = {
                    'file': path.name,
                    'size_kb': path.stat().st_size / 1024,
                    'n_features': len(m.feature_list),
                    'n_estimators': len(m.estimators),
                    'threshold': m.threshold,
                    'metadata': m.metadata,
                }
        except Exception as exc:
            out[path.stem] = {'file': path.name, 'error': str(exc)}
    return out
