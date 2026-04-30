"""
V3 dataset builder — preprocessing → windowing → features → calibration normalization.

Design point: preprocessing is run ONCE per subject (~5 sec). Re-windowing for
different (W, step) configurations is fast (~10 sec/config) because it just
slices the cached preprocessed signals. This makes the Phase 2 (W, step) sweep
tractable in a single run instead of re-paying preprocessing 12 times.

The two public entry points are:

- ``preprocess_subject(...)`` — runs preprocessing.py end-to-end on one subject's
  PKL and returns a cacheable ``SubjectSignals`` bundle (with raw labels at
  EDA timebase already aligned).
- ``build_windowed_dataset(...)`` — given the cached bundle and a (W, step)
  config, emits a per-subject DataFrame with raw + calibration-z-scored features.

Calibration follows the user's confirmed design (audit §6 + memory): per-subject
baseline statistics come from label=1 windows only; every window (including
baseline ones) gets a `_z` companion column. This matches V1's two-pass scheme
and represents the deployment-time "wear-time calibration → fixed baseline" model.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import features as feats_mod
import preprocessing as pp
from preprocessing import EdaResult, BvpResult, TempResult, AccResult


@dataclass
class SubjectSignals:
    subject_id: str
    fs_eda: int
    fs_bvp: int
    fs_temp: int
    fs_acc: int
    eda: EdaResult
    bvp: BvpResult
    temp: TempResult
    acc: AccResult
    labels_at_eda: np.ndarray   # 700 Hz raw labels resampled (nearest) to EDA timebase
    n_eda_samples: int


def align_labels_to_eda(labels_700: np.ndarray, eda_len: int, fs_eda: int) -> np.ndarray:
    fs_label = 700
    sig_t = np.arange(eda_len) / fs_eda
    lab_t = np.arange(len(labels_700)) / fs_label
    idx = np.searchsorted(lab_t, sig_t)
    idx = np.clip(idx, 0, len(labels_700) - 1)
    return labels_700[idx]


def preprocess_subject(subject_id: str, wesad_path: str, cfg: dict) -> SubjectSignals:
    """Load and run V3 preprocessing on one subject. Cache the result."""
    fp = Path(wesad_path) / subject_id / f"{subject_id}.pkl"
    with open(fp, "rb") as f:
        data = pickle.load(f, encoding="latin1")

    eda_raw = data["signal"]["wrist"]["EDA"].flatten().astype(float)
    bvp_raw = data["signal"]["wrist"]["BVP"].flatten().astype(float)
    temp_raw = data["signal"]["wrist"]["TEMP"].flatten().astype(float)
    acc_raw = data["signal"]["wrist"]["ACC"].astype(float)
    labels_700 = data["label"].flatten()

    fs_eda = cfg["sampling_rates"]["eda"]
    fs_bvp = cfg["sampling_rates"]["bvp"]
    fs_temp = cfg["sampling_rates"]["temp"]
    fs_acc = cfg["sampling_rates"]["acc"]

    eda_r = pp.clean_eda(eda_raw, fs_eda, cfg["preprocessing"]["eda"])
    bvp_r = pp.process_bvp(bvp_raw, fs_bvp, cfg["preprocessing"]["bvp"])
    temp_r = pp.clean_temp(temp_raw, fs_temp, cfg["preprocessing"]["temp"])
    acc_r = pp.clean_acc(acc_raw, fs_acc, cfg["preprocessing"]["acc"])

    return SubjectSignals(
        subject_id=subject_id,
        fs_eda=fs_eda, fs_bvp=fs_bvp, fs_temp=fs_temp, fs_acc=fs_acc,
        eda=eda_r, bvp=bvp_r, temp=temp_r, acc=acc_r,
        labels_at_eda=align_labels_to_eda(labels_700, len(eda_raw), fs_eda),
        n_eda_samples=len(eda_raw),
    )


def get_window_label(labels_window: np.ndarray, rule: str = "majority") -> int:
    """Window-level label per the chosen labeling rule.

    Phase 2 default is ``majority`` (matches V1/V2 baseline). The rebuild plan's
    "any stress" alternative is implemented and can be selected for the
    labeling-rule sub-experiment.
    """
    if rule == "any_stress":
        return 2 if (labels_window == 2).any() else int(stats.mode(labels_window, keepdims=True).mode[0])
    if rule == "center":
        return int(labels_window[len(labels_window) // 2])
    return int(stats.mode(labels_window, keepdims=True).mode[0])  # majority


def build_windowed_dataset(
    signals: SubjectSignals,
    cfg: dict,
    window_sec: int,
    step_sec: int,
    label_rule: str = "majority",
    label_mapping: dict | None = None,
) -> pd.DataFrame:
    """Slide windows over a preprocessed subject; emit per-window features + labels.

    No normalization is applied here — `apply_calibration_normalization` does
    that across the LOSO-aware boundary. Returns one row per window with raw
    feature columns + metadata (subject_id, window_start_sec, raw_label, label).
    """
    if label_mapping is None:
        label_mapping = cfg["labels"]["mapping"]

    fs_eda = signals.fs_eda
    fs_bvp = signals.fs_bvp
    fs_temp = signals.fs_temp
    fs_acc = signals.fs_acc

    win_eda = window_sec * fs_eda
    win_bvp = window_sec * fs_bvp
    win_temp = window_sec * fs_temp
    win_acc = window_sec * fs_acc
    step_eda = step_sec * fs_eda

    # Decoupled HRV freq window (Task Force 1996): time-domain HRV uses the
    # `window_sec` slice; freq-domain HRV uses a wider lookback (default 180 s)
    # so LF (0.04 Hz, period 25 s) has ≥ 7 cycles available.
    hrv_freq_window_sec = cfg["windowing"].get("hrv_freq_window", window_sec)
    win_freq_bvp = hrv_freq_window_sec * fs_bvp

    motion_threshold = cfg["preprocessing"]["bvp"]["motion_jerk_threshold_g_per_s"]
    scr_amp_min = cfg["preprocessing"]["eda"]["scr_amplitude_min_us"]
    scr_amp_max = cfg["preprocessing"]["eda"]["scr_amplitude_max_us"]

    rows = []
    n = signals.n_eda_samples
    for s_eda in range(0, n - win_eda + 1, step_eda):
        e_eda = s_eda + win_eda
        labels_in_win = signals.labels_at_eda[s_eda:e_eda]
        raw_label = get_window_label(labels_in_win, rule=label_rule)
        if raw_label not in label_mapping:
            continue

        # Per-modality sample indexing (proportional to fs)
        s_bvp = s_eda * fs_bvp // fs_eda; e_bvp = s_bvp + win_bvp
        s_temp = s_eda * fs_temp // fs_eda; e_temp = s_temp + win_temp
        s_acc = s_eda * fs_acc // fs_eda; e_acc = s_acc + win_acc

        # HRV freq window: 180 s lookback ending at e_bvp. Clipped to start of
        # signal for early windows; extract_hrv_features NaN's freq features
        # when the available window is < 120 s.
        freq_s_bvp = max(0, e_bvp - win_freq_bvp)
        freq_e_bvp = e_bvp

        # Motion gate from ACC jerk in this window
        jerk_window = signals.acc.jerk_magnitude[s_acc:e_acc]
        motion_corrupted = pp.is_window_motion_corrupted(jerk_window, motion_threshold)

        eda_feats = feats_mod.extract_eda_features(
            signals.eda, s_eda, e_eda, fs_eda, scr_amp_min, scr_amp_max
        )
        hrv_feats = feats_mod.extract_hrv_features(
            signals.bvp,
            s_bvp, e_bvp,
            freq_s_bvp, freq_e_bvp,
            fs_bvp, motion_corrupted,
        )
        temp_feats = feats_mod.extract_temp_features(signals.temp, s_temp, e_temp, fs_temp)
        acc_feats = feats_mod.extract_acc_features(signals.acc, s_acc, e_acc, fs_acc)

        # EDA-validity gate (electrode-off): if > 50% of window flagged invalid, drop
        eda_invalid_frac = float((~signals.eda.valid[s_eda:e_eda]).mean())
        if eda_invalid_frac > 0.5:
            continue

        row = {**eda_feats, **hrv_feats, **temp_feats, **acc_feats,
               "subject_id": signals.subject_id,
               "window_start_sec": s_eda / fs_eda,
               "raw_label": int(raw_label),
               "label": label_mapping[raw_label],
               "motion_corrupted": int(motion_corrupted),
               }
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def apply_calibration_normalization(
    subject_df: pd.DataFrame, baseline_label_int: int = 1
) -> pd.DataFrame:
    """Per-subject calibration: compute baseline mean/std from raw_label==1 windows,
    add `_z` columns for every numeric feature.

    HRV columns are special: when hrv_valid==0, raw HRV is NaN. The z-score
    pass converts NaN-in to NaN-out. Phase 6 imputes; the train/eval scripts
    in this phase fill NaN per-subject median to keep RandomForest happy.
    """
    out = subject_df.copy()
    feature_cols = [c for c in feats_mod.ALL_FEATURES if c in out.columns]

    base = out[out["raw_label"] == baseline_label_int]
    if len(base) == 0:
        # No baseline windows for this subject — leave _z columns NaN; downstream sees missing
        for c in feature_cols:
            out[f"{c}_z"] = np.nan
        return out

    for c in feature_cols:
        mu = base[c].mean(skipna=True)
        sd = base[c].std(skipna=True)
        if not np.isfinite(sd) or sd == 0:
            out[f"{c}_z"] = 0.0
        else:
            out[f"{c}_z"] = (out[c] - mu) / sd
    return out


def build_full_dataset(
    subjects: list[str], wesad_path: str, cfg: dict,
    window_sec: int, step_sec: int, label_rule: str = "majority",
    cache: dict | None = None,
) -> pd.DataFrame:
    """Top-level orchestrator. ``cache`` is an optional dict {subject_id: SubjectSignals}
    so the Phase 2 sweep can reuse preprocessing across configs.
    """
    cache = cache if cache is not None else {}
    pieces = []
    for s in subjects:
        if s not in cache:
            cache[s] = preprocess_subject(s, wesad_path, cfg)
        df = build_windowed_dataset(cache[s], cfg, window_sec, step_sec, label_rule)
        if len(df) == 0:
            continue
        df = apply_calibration_normalization(df)
        pieces.append(df)
    if not pieces:
        return pd.DataFrame()
    full = pd.concat(pieces, ignore_index=True)
    return full
