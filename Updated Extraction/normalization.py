"""
Two-Pass Subject-Specific Baseline Normalization
=================================================

This module implements Grant's normalization strategy for domain adaptation:

PASS 1: Calculate subject-specific baseline statistics from TRUE baseline (label=1) ONLY
        - Amusement (label=3) and Meditation (label=4) are NOT used for baseline calculation
        - Only resting baseline provides device/person calibration

PASS 2: For ALL windows (baseline + amusement + stress), add normalized features:
        - *_percent_change: ((value - baseline_mean) / baseline_mean) * 100
        - *_z_score: (value - baseline_mean) / baseline_std

This makes the model:
1. Device-agnostic (trains on relative changes, not absolute values)
2. Robust to individual differences (each person normalized to their own baseline)
3. Stress-specific (learns stress patterns, not just general arousal)

Author: Merged pipeline (Grant + Tanner approaches)
"""

import numpy as np
import pandas as pd


def safe_percent_change(value, baseline_mean):
    """
    Calculate percent change with zero-division protection.
    
    Returns: ((value - baseline_mean) / baseline_mean) * 100
    """
    if baseline_mean == 0 or np.isnan(baseline_mean):
        return 0.0
    return ((value - baseline_mean) / baseline_mean) * 100


def safe_z_score(value, baseline_mean, baseline_std):
    """
    Calculate z-score with zero-division protection.
    
    Returns: (value - baseline_mean) / baseline_std
    """
    if baseline_std == 0 or np.isnan(baseline_std):
        return 0.0
    return (value - baseline_mean) / baseline_std


def calculate_baseline_stats(subject_df, raw_labels):
    """
    PASS 1: Calculate subject-specific baseline statistics from TRUE baseline only.
    
    Args:
        subject_df: DataFrame with features and mapped labels ('non-stress'/'stress')
        raw_labels: Original numeric labels (1=baseline, 2=stress, 3=amusement, 4=meditation)
    
    Returns:
        Dictionary with {feature_name: {'mean': X, 'std': Y}}
    
    Note:
        Only label=1 (baseline) is used for calculating statistics.
        Amusement (3) and Meditation (4) are NOT baseline - they represent
        different physiological states (arousal, relaxation).
    """
    # Filter to TRUE baseline windows only (label=1, NOT amusement=3 or meditation=4)
    baseline_mask = raw_labels == 1
    baseline_df = subject_df[baseline_mask]
    
    if len(baseline_df) == 0:
        print("  WARNING: No baseline windows found for subject")
        return {}
    
    # Get all feature columns (exclude label, subject_id, raw_label)
    exclude_cols = ['label', 'subject_id', 'raw_label', 'window_idx']
    feature_cols = [col for col in subject_df.columns 
                    if col not in exclude_cols 
                    and not col.endswith(('_percent_change', '_z_score'))]
    
    # Calculate mean and std for each feature from baseline only
    baseline_stats = {}
    for col in feature_cols:
        baseline_stats[col] = {
            'mean': baseline_df[col].mean(),
            'std': baseline_df[col].std()
        }
    
    return baseline_stats


def add_normalized_features(subject_df, baseline_stats):
    """
    PASS 2: Add normalized features (percent_change, z_score) to all windows.
    
    Args:
        subject_df: DataFrame with raw features
        baseline_stats: Dictionary from calculate_baseline_stats()
    
    Returns:
        DataFrame with raw + normalized features (46 raw → 138 total)
    """
    normalized_df = subject_df.copy()
    
    # Get all feature columns (exclude metadata)
    exclude_cols = ['label', 'subject_id', 'raw_label', 'window_idx']
    feature_cols = [col for col in subject_df.columns 
                    if col not in exclude_cols
                    and not col.endswith(('_percent_change', '_z_score'))]
    
    for col in feature_cols:
        if col not in baseline_stats:
            continue
        
        baseline_mean = baseline_stats[col]['mean']
        baseline_std = baseline_stats[col]['std']
        
        # Add percent_change column (true percent: multiply by 100)
        normalized_df[f'{col}_percent_change'] = subject_df[col].apply(
            lambda x: safe_percent_change(x, baseline_mean)
        )
        
        # Add z_score column
        normalized_df[f'{col}_z_score'] = subject_df[col].apply(
            lambda x: safe_z_score(x, baseline_mean, baseline_std)
        )
    
    return normalized_df


def normalize_subject_features(subject_df, raw_labels):
    """
    Complete normalization pipeline for a single subject.
    
    Args:
        subject_df: DataFrame with raw features for one subject
        raw_labels: Series/array of original numeric labels (1, 2, 3, 4)
    
    Returns:
        DataFrame with raw + normalized features
    """
    # Pass 1: Calculate baseline statistics
    baseline_stats = calculate_baseline_stats(subject_df, raw_labels)
    
    if not baseline_stats:
        print("  WARNING: Skipping normalization due to missing baseline")
        return subject_df
    
    # Pass 2: Add normalized features
    normalized_df = add_normalized_features(subject_df, baseline_stats)
    
    return normalized_df

