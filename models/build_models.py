"""
Build (or rebuild) the three V3 stress-detection models from the locked
Phase 5 dataset and Phase 7 hyperparameters. Run this whenever the upstream
dataset or hyperparameters change.

Outputs three files in this directory:
    - default_hgb.pkl        Phase 6 lock (HGB defaults)
    - tuned_hgb.pkl          Phase 7 nested-LOSO-tuned HGB (median hyperparameters)
    - ensemble_hgb_svm.pkl   Phase 7 ensemble (HGB + SVM-RBF probability averaging)

Plus feature_list.json (the locked 16 features) for portability.

Run from the repo root:
    python models/build_models.py
"""

from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / 'models'
sys.path.insert(0, str(MODELS_DIR))
from inference import StressModel  # noqa: E402

RANDOM_STATE = 42


def main():
    # ---------- Load inputs ----------
    selected_path = REPO_ROOT / 'reports' / '05_feature_selection' / 'selected_features.csv'
    LOCKED_16 = pd.read_csv(selected_path)['feature'].tolist()
    print(f'Locked 16 features (Phase 5): {len(LOCKED_16)} features')

    dataset_path = REPO_ROOT / 'Updated_Extraction_V3' / 'output' / 'dataset_W60_step60_raw.parquet'
    df = pd.read_parquet(dataset_path)
    X = df[LOCKED_16].values
    y = (df['label'].values == 'stress').astype(int)
    print(f'Training data: {df.shape}, {y.sum()} stress / {len(y)} total')

    hp_path = MODELS_DIR / 'hyperparameters.json'
    with open(hp_path) as f:
        tuned_hp = json.load(f)['hyperparameters']
    print(f'Tuned hyperparameters from Phase 7: {tuned_hp}')
    print()

    # ---------- 1. Default HGB (Phase 6 lock) ----------
    print('Building default_hgb...')
    t0 = time.time()
    default_hgb = HistGradientBoostingClassifier(
        max_iter=300, max_depth=4, learning_rate=0.05,
        class_weight='balanced', random_state=RANDOM_STATE,
    )
    default_hgb.fit(X, y)
    default_model = StressModel(
        name='default_hgb',
        feature_list=LOCKED_16,
        estimators=[default_hgb],
        threshold=0.5,
        metadata={
            'phase': 6,
            'description': 'HGB with sklearn-default-style hyperparameters (Phase 6 lock).',
            'hyperparameters': {
                'max_iter': 300, 'max_depth': 4, 'learning_rate': 0.05,
                'class_weight': 'balanced', 'random_state': RANDOM_STATE,
            },
            'loso_f1': 0.931,
            'loso_recall': 0.911,
            'loso_precision': 0.972,
            'loso_accuracy': 0.973,
            'roc_auc': 0.997,
            'min_subj_recall': 0.545,
            'inference_ms_per_window': 10.0,
            'when_to_use': 'Default choice. Highest single-LOSO F1; simplest defense.',
        },
    )
    with open(MODELS_DIR / 'default_hgb.pkl', 'wb') as f:
        pickle.dump(default_model, f)
    print(f'  saved default_hgb.pkl ({time.time()-t0:.1f}s)')

    # ---------- 2. Tuned HGB (Phase 7 median hyperparameters) ----------
    print('Building tuned_hgb...')
    t0 = time.time()
    tuned_hgb = HistGradientBoostingClassifier(
        **tuned_hp, class_weight='balanced', random_state=RANDOM_STATE,
    )
    tuned_hgb.fit(X, y)
    tuned_model = StressModel(
        name='tuned_hgb',
        feature_list=LOCKED_16,
        estimators=[tuned_hgb],
        threshold=0.5,
        metadata={
            'phase': 7,
            'description': 'HGB with median hyperparameters from Phase 7 nested-LOSO Optuna.',
            'hyperparameters': {**tuned_hp, 'class_weight': 'balanced',
                                'random_state': RANDOM_STATE},
            'nested_loso_f1': 0.917,
            'nested_loso_f1_std': 0.085,
            'nested_loso_min_recall': 0.636,
            'when_to_use': (
                'Methodologically rigorous single-model alternative. Best worst-case '
                '(min subj recall 0.636 vs 0.545 for default).'
            ),
        },
    )
    with open(MODELS_DIR / 'tuned_hgb.pkl', 'wb') as f:
        pickle.dump(tuned_model, f)
    print(f'  saved tuned_hgb.pkl ({time.time()-t0:.1f}s)')

    # ---------- 3. HGB + SVM-RBF ensemble (Phase 7 side experiment) ----------
    print('Building ensemble_hgb_svm...')
    t0 = time.time()
    hgb_for_ensemble = HistGradientBoostingClassifier(
        max_iter=300, max_depth=4, learning_rate=0.05,
        class_weight='balanced', random_state=RANDOM_STATE,
    )
    hgb_for_ensemble.fit(X, y)
    svm_pipeline = Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('scale', StandardScaler()),
        ('clf', SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced',
                    probability=True, random_state=RANDOM_STATE)),
    ])
    svm_pipeline.fit(X, y)
    ensemble_model = StressModel(
        name='ensemble_hgb_svm',
        feature_list=LOCKED_16,
        estimators=[hgb_for_ensemble, svm_pipeline],
        threshold=0.5,
        metadata={
            'phase': 7,
            'description': (
                'HGB + SVM-RBF probability-averaging ensemble (Phase 7 side experiment).'
            ),
            'hyperparameters': {
                'hgb': {'max_iter': 300, 'max_depth': 4, 'learning_rate': 0.05},
                'svm': {'kernel': 'rbf', 'C': 1.0, 'gamma': 'scale'},
            },
            'loso_f1': 0.935,
            'min_subj_recall': 0.545,
            's3_recall': 0.727,
            'inference_ms_per_window': 10.3,
            'when_to_use': (
                'RECOMMENDED. Best F1 in V3 (0.935). First method to materially '
                'recover S3 (recall 0.636 → 0.727). Costs negligible extra '
                'storage (~33 KB SVM) and inference (~0.3 ms).'
            ),
        },
    )
    with open(MODELS_DIR / 'ensemble_hgb_svm.pkl', 'wb') as f:
        pickle.dump(ensemble_model, f)
    print(f'  saved ensemble_hgb_svm.pkl ({time.time()-t0:.1f}s)')

    # ---------- Save feature list separately for portability ----------
    print('Saving feature_list.json...')
    with open(MODELS_DIR / 'feature_list.json', 'w') as f:
        json.dump({
            'features': LOCKED_16,
            'count': len(LOCKED_16),
            'description': 'Locked 16-feature set from Phase 5 combined-rank selection.',
            'source': 'reports/05_feature_selection/selected_features.csv',
        }, f, indent=2)

    # ---------- Summary ----------
    print()
    print('Built models:')
    for path in sorted(MODELS_DIR.glob('*.pkl')):
        size_kb = path.stat().st_size / 1024
        print(f'  {path.name:<28s} {size_kb:>8.1f} KB')


if __name__ == '__main__':
    main()
