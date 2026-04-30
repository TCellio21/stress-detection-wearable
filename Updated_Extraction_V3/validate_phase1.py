"""
Phase 1 empirical verification.

Three deliverables, all written to ``reports/01_preprocessing/``:

1.  cvxEDA-vs-HP-phasic divergence study: for each subject, computes
    per-60s-window features from V1's cvxEDA decomposition and V3's
    complementary-HP decomposition, correlates the resulting feature vectors,
    and plots an example baseline-window and stress-window with both
    decompositions overlaid.

2.  Raw-vs-filtered sanity plots: one baseline-window and one stress-window
    per subject, showing raw signal vs V3 preprocessed output for each of
    EDA / BVP / TEMP / ACC.

3.  Per-subject artifact-rate table: fraction of EDA samples flagged
    electrode-off, TEMP samples flagged dropout, IBIs ectopic-corrected, and
    windows flagged motion-corrupted.

Run from repo root::

    py -3.13 Updated_Extraction_V3/validate_phase1.py

Subjects sampled: S2 (the model-class-failure subject — sanity check that the
signal is clean), S10 (highest median HR — verify no motion artifact bias),
S16 (perfect-recall subject — clean reference).
"""

from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_SCRIPT_DIR))

from config_loader import load_config  # noqa: E402
import preprocessing as pp  # noqa: E402


REPORTS_DIR = _REPO_ROOT / "reports" / "01_preprocessing"
SUBJECTS = ["S2", "S10", "S16"]
WINDOW_SEC = 60


def load_subject(wesad_path: str, subject: str) -> dict:
    fp = Path(wesad_path) / subject / f"{subject}.pkl"
    with open(fp, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    return {
        "eda": data["signal"]["wrist"]["EDA"].flatten().astype(float),
        "bvp": data["signal"]["wrist"]["BVP"].flatten().astype(float),
        "temp": data["signal"]["wrist"]["TEMP"].flatten().astype(float),
        "acc": data["signal"]["wrist"]["ACC"].astype(float),
        "labels": data["label"].flatten(),
    }


def align_labels_to_eda(labels_700: np.ndarray, eda_len: int, fs_eda: int) -> np.ndarray:
    fs_label = 700
    signal_times = np.arange(eda_len) / fs_eda
    label_times = np.arange(len(labels_700)) / fs_label
    idx = np.searchsorted(label_times, signal_times)
    idx = np.clip(idx, 0, len(labels_700) - 1)
    return labels_700[idx]


def find_example_window(labels_eda: np.ndarray, target_label: int, fs_eda: int,
                        win_samples: int) -> int | None:
    """Return start-sample of the first all-target_label window, or None."""
    for start in range(0, len(labels_eda) - win_samples, win_samples):
        win = labels_eda[start:start + win_samples]
        if np.all(win == target_label):
            return start
    return None


# ---------------------------------------------------------------------------
# 1. cvxEDA vs HP comparison
# ---------------------------------------------------------------------------

def cvxeda_window_features(eda_window: np.ndarray, fs: int) -> dict:
    """Per-window features using V1-style cvxEDA decomposition.

    Returns NaN values if cvxEDA fails so the caller can correlate without
    propagating exceptions.
    """
    try:
        decomp = nk.eda_phasic(eda_window, sampling_rate=fs, method="cvxeda")
        tonic = decomp["EDA_Tonic"].values
        phasic = decomp["EDA_Phasic"].values
        try:
            _, peaks = nk.eda_peaks(phasic, sampling_rate=fs, amplitude_min=0.01)
            scr_count = int(len(peaks.get("SCR_Peaks", [])))
        except (ValueError, KeyError, IndexError, TypeError):
            scr_count = 0
        return {
            "scl_mean": float(np.mean(tonic)),
            "scl_std": float(np.std(tonic)),
            "scr_peak_count": scr_count,
        }
    except Exception:
        return {"scl_mean": np.nan, "scl_std": np.nan, "scr_peak_count": np.nan}


def hp_window_features(tonic_window: np.ndarray, phasic_window: np.ndarray,
                       fs: int) -> dict:
    """Same features computed from V3 (HP-phasic) decomposition pre-applied."""
    try:
        _, peaks = nk.eda_peaks(phasic_window, sampling_rate=fs, amplitude_min=0.01)
        scr_count = int(len(peaks.get("SCR_Peaks", [])))
    except (ValueError, KeyError, IndexError, TypeError):
        scr_count = 0
    return {
        "scl_mean": float(np.mean(tonic_window)),
        "scl_std": float(np.std(tonic_window)),
        "scr_peak_count": scr_count,
    }


def run_eda_comparison(subject: str, signals: dict, cfg: dict) -> dict:
    fs = cfg["sampling_rates"]["eda"]
    win = WINDOW_SEC * fs
    labels = align_labels_to_eda(signals["labels"], len(signals["eda"]), fs)

    eda_v3 = pp.clean_eda(signals["eda"], fs, cfg["preprocessing"]["eda"])

    rows = []
    for start in range(0, len(signals["eda"]) - win, win):
        end = start + win
        win_label = int(np.bincount(labels[start:end].astype(int)).argmax())
        if win_label not in {1, 2, 3, 4}:
            continue
        cv = cvxeda_window_features(signals["eda"][start:end], fs)
        hp = hp_window_features(eda_v3.tonic[start:end], eda_v3.phasic[start:end], fs)
        rows.append({
            "subject": subject, "start": start, "label": win_label,
            "scl_mean_cvx": cv["scl_mean"], "scl_mean_hp": hp["scl_mean"],
            "scl_std_cvx": cv["scl_std"],   "scl_std_hp": hp["scl_std"],
            "scr_count_cvx": cv["scr_peak_count"], "scr_count_hp": hp["scr_peak_count"],
        })
    df = pd.DataFrame(rows)

    correlations = {}
    for feat in ["scl_mean", "scl_std", "scr_count"]:
        cvx_vals = df[f"{feat}_cvx"].values
        hp_vals = df[f"{feat}_hp"].values
        mask = np.isfinite(cvx_vals) & np.isfinite(hp_vals)
        if mask.sum() >= 3 and np.std(cvx_vals[mask]) > 0 and np.std(hp_vals[mask]) > 0:
            correlations[feat] = float(np.corrcoef(cvx_vals[mask], hp_vals[mask])[0, 1])
        else:
            correlations[feat] = float("nan")

    plot_eda_comparison(subject, signals, eda_v3, fs, win, labels)

    return {"df": df, "correlations": correlations}


def plot_eda_comparison(subject: str, signals: dict, eda_v3: pp.EdaResult,
                        fs: int, win_samples: int, labels: np.ndarray) -> None:
    """Plot one baseline + one stress window: cvxEDA vs HP decompositions."""
    base_start = find_example_window(labels, target_label=1, fs_eda=fs, win_samples=win_samples)
    str_start = find_example_window(labels, target_label=2, fs_eda=fs, win_samples=win_samples)
    if base_start is None or str_start is None:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 7), sharex=False)
    for col, (start, title) in enumerate([(base_start, "Baseline"), (str_start, "Stress")]):
        end = start + win_samples
        t = np.arange(win_samples) / fs
        raw = signals["eda"][start:end]
        try:
            cvx = nk.eda_phasic(raw, sampling_rate=fs, method="cvxeda")
            cvx_tonic = cvx["EDA_Tonic"].values
            cvx_phasic = cvx["EDA_Phasic"].values
        except Exception:
            cvx_tonic = np.full_like(raw, np.nan)
            cvx_phasic = np.full_like(raw, np.nan)
        hp_tonic = eda_v3.tonic[start:end]
        hp_phasic = eda_v3.phasic[start:end]

        ax_t = axes[0, col]
        ax_t.plot(t, raw, color="lightgray", lw=0.8, label="raw EDA")
        ax_t.plot(t, cvx_tonic, color="C0", lw=1.5, label="cvxEDA tonic")
        ax_t.plot(t, hp_tonic, color="C3", lw=1.5, label="V3 tonic (LP 0.05 Hz)")
        ax_t.set_title(f"{subject} — {title} window — Tonic")
        ax_t.set_ylabel("EDA (µS)")
        ax_t.legend(loc="best", fontsize=8)
        ax_t.grid(alpha=0.3)

        ax_p = axes[1, col]
        ax_p.plot(t, cvx_phasic, color="C0", lw=1.0, label="cvxEDA phasic")
        ax_p.plot(t, hp_phasic, color="C3", lw=1.0, label="V3 phasic (HP 0.05 Hz)")
        ax_p.axhline(0, color="k", lw=0.4)
        ax_p.set_title(f"{subject} — {title} window — Phasic")
        ax_p.set_xlabel("time (s)")
        ax_p.set_ylabel("EDA (µS)")
        ax_p.legend(loc="best", fontsize=8)
        ax_p.grid(alpha=0.3)

    fig.suptitle(f"cvxEDA vs V3 complementary-HP — {subject}", fontsize=12)
    fig.tight_layout()
    out = REPORTS_DIR / f"cvxEDA_vs_HP_{subject}.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"  wrote {out.relative_to(_REPO_ROOT)}")


# ---------------------------------------------------------------------------
# 2. Raw-vs-filtered sanity plots
# ---------------------------------------------------------------------------

def plot_sanity(subject: str, signals: dict, cfg: dict) -> None:
    fs_eda = cfg["sampling_rates"]["eda"]
    fs_bvp = cfg["sampling_rates"]["bvp"]
    fs_temp = cfg["sampling_rates"]["temp"]
    fs_acc = cfg["sampling_rates"]["acc"]
    win_eda = WINDOW_SEC * fs_eda
    labels_eda = align_labels_to_eda(signals["labels"], len(signals["eda"]), fs_eda)

    base_start = find_example_window(labels_eda, target_label=1, fs_eda=fs_eda, win_samples=win_eda)
    str_start = find_example_window(labels_eda, target_label=2, fs_eda=fs_eda, win_samples=win_eda)
    if base_start is None or str_start is None:
        print(f"  {subject}: missing example window (skipping sanity plot)")
        return

    eda_r = pp.clean_eda(signals["eda"], fs_eda, cfg["preprocessing"]["eda"])
    bvp_r = pp.process_bvp(signals["bvp"], fs_bvp, cfg["preprocessing"]["bvp"])
    temp_r = pp.clean_temp(signals["temp"], fs_temp, cfg["preprocessing"]["temp"])
    acc_r = pp.clean_acc(signals["acc"], fs_acc, cfg["preprocessing"]["acc"])

    fig, axes = plt.subplots(4, 2, figsize=(14, 11), sharex=False)
    for col, (start_eda, title) in enumerate([(base_start, "Baseline"), (str_start, "Stress")]):
        # EDA (4 Hz)
        ax = axes[0, col]
        end = start_eda + win_eda
        t = np.arange(win_eda) / fs_eda
        ax.plot(t, signals["eda"][start_eda:end], color="lightgray", lw=0.8, label="raw")
        ax.plot(t, eda_r.smoothed[start_eda:end], color="C0", lw=1.0, label="smoothed (1 Hz LP)")
        ax.plot(t, eda_r.tonic[start_eda:end], color="C2", lw=1.2, label="tonic (SCL)")
        ax.plot(t, eda_r.phasic[start_eda:end], color="C3", lw=1.0, label="phasic (SCR)")
        ax.set_title(f"{subject} EDA — {title}")
        ax.set_ylabel("µS")
        ax.legend(loc="best", fontsize=8)
        ax.grid(alpha=0.3)

        # BVP (64 Hz) — convert eda_start to bvp samples
        start_bvp = start_eda * fs_bvp // fs_eda
        win_bvp = WINDOW_SEC * fs_bvp
        end_bvp = start_bvp + win_bvp
        ax = axes[1, col]
        t = np.arange(win_bvp) / fs_bvp
        ax.plot(t, signals["bvp"][start_bvp:end_bvp], color="lightgray", lw=0.6, label="raw")
        ax.plot(t, bvp_r.cleaned[start_bvp:end_bvp], color="C0", lw=0.8, label="cleaned (BP 0.5–8 Hz)")
        # Mark peaks within this window
        peaks_in_win = bvp_r.peaks[(bvp_r.peaks >= start_bvp) & (bvp_r.peaks < end_bvp)]
        if len(peaks_in_win) > 0:
            t_peaks = (peaks_in_win - start_bvp) / fs_bvp
            ax.plot(t_peaks, bvp_r.cleaned[peaks_in_win], "rv", ms=4, label="peaks")
        ax.set_title(f"{subject} BVP — {title}")
        ax.set_ylabel("a.u.")
        ax.legend(loc="best", fontsize=8)
        ax.grid(alpha=0.3)

        # TEMP (4 Hz) — same as EDA
        ax = axes[2, col]
        t = np.arange(win_eda) / fs_eda
        ax.plot(t, signals["temp"][start_eda:end], color="lightgray", lw=0.8, label="raw")
        ax.plot(t, temp_r.smoothed[start_eda:end], color="C0", lw=1.0, label="smoothed (0.5 Hz LP)")
        ax.set_title(f"{subject} TEMP — {title}")
        ax.set_ylabel("°C")
        ax.legend(loc="best", fontsize=8)
        ax.grid(alpha=0.3)

        # ACC (32 Hz)
        start_acc = start_eda * fs_acc // fs_eda
        win_acc = WINDOW_SEC * fs_acc
        end_acc = start_acc + win_acc
        ax = axes[3, col]
        t = np.arange(win_acc) / fs_acc
        ax.plot(t, acc_r.magnitude[start_acc:end_acc], color="lightgray", lw=0.6, label="|a| (g)")
        ax.plot(t, acc_r.magnitude_activity[start_acc:end_acc], color="C0", lw=1.0,
                label="activity-band (LP 3 Hz)")
        ax2 = ax.twinx()
        ax2.plot(t, acc_r.jerk_magnitude[start_acc:end_acc], color="C3", lw=0.6, alpha=0.6,
                 label="jerk")
        ax.set_title(f"{subject} ACC — {title}")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("g")
        ax2.set_ylabel("g/s", color="C3")
        ax.legend(loc="upper left", fontsize=8)
        ax2.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle(f"V3 raw-vs-filtered sanity check — {subject}", fontsize=12)
    fig.tight_layout()
    out = REPORTS_DIR / f"sanity_{subject}.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"  wrote {out.relative_to(_REPO_ROOT)}")


# ---------------------------------------------------------------------------
# 3. Per-subject artifact rates
# ---------------------------------------------------------------------------

def compute_artifact_rates(subject: str, signals: dict, cfg: dict) -> dict:
    fs_eda = cfg["sampling_rates"]["eda"]
    fs_bvp = cfg["sampling_rates"]["bvp"]
    fs_temp = cfg["sampling_rates"]["temp"]
    fs_acc = cfg["sampling_rates"]["acc"]

    eda_r = pp.clean_eda(signals["eda"], fs_eda, cfg["preprocessing"]["eda"])
    bvp_r = pp.process_bvp(signals["bvp"], fs_bvp, cfg["preprocessing"]["bvp"])
    temp_r = pp.clean_temp(signals["temp"], fs_temp, cfg["preprocessing"]["temp"])
    acc_r = pp.clean_acc(signals["acc"], fs_acc, cfg["preprocessing"]["acc"])

    pct_electrode_off = 100.0 * (~eda_r.valid).sum() / len(eda_r.valid)
    pct_temp_dropout = 100.0 * (~temp_r.valid).sum() / len(temp_r.valid)
    n_ibis = len(bvp_r.ibis_ms)
    n_ectopic = int((bvp_r.ibis_ms != bvp_r.ibis_corrected_ms).sum())
    pct_ectopic = 100.0 * n_ectopic / n_ibis if n_ibis else 0.0

    win_acc = WINDOW_SEC * fs_acc
    threshold = cfg["preprocessing"]["bvp"]["motion_jerk_threshold_g_per_s"]
    labels_eda = align_labels_to_eda(signals["labels"], len(signals["eda"]), fs_eda)
    fs_eda_to_acc = fs_acc // fs_eda
    n_motion = 0
    n_windows = 0
    metrics_baseline = []
    metrics_stress = []
    for s in range(0, len(acc_r.jerk_magnitude) - win_acc, win_acc):
        n_windows += 1
        m = pp.window_motion_metric(acc_r.jerk_magnitude[s:s + win_acc])
        if m > threshold:
            n_motion += 1
        # Find the corresponding majority label by looking at the EDA-rate label slice
        s_eda = s // fs_eda_to_acc
        e_eda = s_eda + (win_acc // fs_eda_to_acc)
        win_lbls = labels_eda[s_eda:e_eda]
        if len(win_lbls) == 0:
            continue
        win_label = int(np.bincount(win_lbls.astype(int)).argmax())
        if win_label == 1:
            metrics_baseline.append(m)
        elif win_label == 2:
            metrics_stress.append(m)
    pct_motion = 100.0 * n_motion / n_windows if n_windows else 0.0
    pct_baseline_high = 100.0 * sum(1 for x in metrics_baseline if x > threshold) / max(1, len(metrics_baseline))
    pct_stress_high = 100.0 * sum(1 for x in metrics_stress if x > threshold) / max(1, len(metrics_stress))

    metrics_baseline_arr = np.asarray(metrics_baseline) if metrics_baseline else np.array([np.nan])
    metrics_stress_arr = np.asarray(metrics_stress) if metrics_stress else np.array([np.nan])
    return {
        "subject": subject,
        "n_eda_samples": len(eda_r.valid),
        "pct_electrode_off": round(pct_electrode_off, 3),
        "n_temp_samples": len(temp_r.valid),
        "pct_temp_dropout": round(pct_temp_dropout, 3),
        "n_ibis": n_ibis,
        "pct_ectopic_corrected": round(pct_ectopic, 1),
        "n_windows": n_windows,
        "pct_motion_corrupted": round(pct_motion, 1),
        "median_hr_raw": round(60000.0 / float(np.median(bvp_r.ibis_ms)), 1) if n_ibis else float("nan"),
        "median_hr_corrected": round(60000.0 / float(np.median(bvp_r.ibis_corrected_ms)), 1) if n_ibis else float("nan"),
        "n_scr_peaks": len(eda_r.peaks_info.get("SCR_Peaks", [])),
        "jerk_p95_baseline_median": round(float(np.nanmedian(metrics_baseline_arr)), 2),
        "jerk_p95_stress_median": round(float(np.nanmedian(metrics_stress_arr)), 2),
        "pct_baseline_flagged": round(pct_baseline_high, 1),
        "pct_stress_flagged": round(pct_stress_high, 1),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    wesad_path = cfg["paths"]["wesad_path"]

    print("=" * 70)
    print("Phase 1 verification — V3 preprocessing")
    print("=" * 70)
    print(f"WESAD path: {wesad_path}")
    print(f"Subjects:   {SUBJECTS}")
    print(f"Outputs:    {REPORTS_DIR.relative_to(_REPO_ROOT)}")
    print()

    artifact_rows = []
    correlation_rows = []
    feature_dfs = []

    for subj in SUBJECTS:
        print(f"--- {subj} ---")
        signals = load_subject(wesad_path, subj)
        artifact_rows.append(compute_artifact_rates(subj, signals, cfg))
        plot_sanity(subj, signals, cfg)
        eda_cmp = run_eda_comparison(subj, signals, cfg)
        feature_dfs.append(eda_cmp["df"])
        correlation_rows.append({"subject": subj, **eda_cmp["correlations"]})
        print(f"  cvxEDA-vs-HP correlations: {eda_cmp['correlations']}")
        print()

    artifact_df = pd.DataFrame(artifact_rows)
    artifact_df.to_csv(REPORTS_DIR / "artifact_rates.csv", index=False)
    print("=== ARTIFACT RATES ===")
    with pd.option_context("display.max_columns", None, "display.width", 160):
        print(artifact_df.to_string(index=False))

    corr_df = pd.DataFrame(correlation_rows)
    corr_df.to_csv(REPORTS_DIR / "feature_correlations.csv", index=False)
    print()
    print("=== cvxEDA vs HP correlations (per-subject, per-feature) ===")
    with pd.option_context("display.max_columns", None, "display.width", 160):
        print(corr_df.to_string(index=False))

    full_features = pd.concat(feature_dfs, ignore_index=True)
    full_features.to_csv(REPORTS_DIR / "feature_comparison_per_window.csv", index=False)

    print()
    print("Done.")


if __name__ == "__main__":
    main()
