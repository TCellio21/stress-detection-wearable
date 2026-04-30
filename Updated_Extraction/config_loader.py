"""
Configuration loader for WESAD feature extraction pipeline.
Loads .env for WESAD_PATH, then YAML config with fallback to defaults.
Path priority: WESAD_PATH (env / .env) > config.yaml paths.wesad_path > repo-relative default.
"""

import os
from pathlib import Path

# Repo-relative default when no env or config is set (works if WESAD is at dataset/raw)
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
_DEFAULT_WESAD_PATH = str(_REPO_ROOT / "dataset" / "raw")

_REQUIRED_KEYS = (
    ("reproducibility", "random_seed"),
    ("paths", "output_dir"),
    ("paths", "output_file"),
    ("subjects", "include"),
    ("sampling_rates", "eda"),
    ("sampling_rates", "bvp"),
    ("sampling_rates", "temp"),
    ("sampling_rates", "acc"),
    ("sampling_rates", "label"),
    ("windowing", "window_size"),
    ("windowing", "step_size"),
    ("eda", "peak_threshold"),
    ("eda", "outlier_min"),
    ("eda", "outlier_max"),
    ("tsst", "prep_duration"),
    ("labels", "mapping"),
)


def get_default_config():
    """Return hardcoded default configuration (current pipeline values)."""
    return {
        "reproducibility": {"random_seed": 42},
        "paths": {
            "wesad_path": _DEFAULT_WESAD_PATH,
            "output_dir": "output",
            "output_file": "all_subject_features_updated.csv",
        },
        "subjects": {
            "include": [
                "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "S11",
                "S13", "S15", "S16", "S17",
            ],
            "exclude": ["S14"],
        },
        "sampling_rates": {
            "eda": 4,
            "bvp": 64,
            "temp": 4,
            "acc": 32,
            "label": 700,
        },
        "windowing": {"window_size": 60, "step_size": 60},
        "eda": {
            "peak_threshold": 0.01,
            "outlier_min": 0.01,
            "outlier_max": 1.0,
        },
        "tsst": {"prep_duration": 180},
        "labels": {
            "mapping": {
                1: "non-stress",
                2: "stress",
                3: "non-stress",
                4: "non-stress",
            }
        },
    }


def _deep_merge(base, override):
    """Recursively merge override into base. Lists and scalars from override replace base."""
    if not isinstance(override, dict):
        return override
    result = dict(base)
    for k, v in override.items():
        result[k] = _deep_merge(result.get(k, {}), v) if isinstance(v, dict) and isinstance(result.get(k), dict) else v
    return result


def _validate_config(cfg):
    """Raise ValueError if any required key is missing."""
    for key_path in _REQUIRED_KEYS:
        d = cfg
        for k in key_path:
            if not isinstance(d, dict) or k not in d:
                raise ValueError(f"Config missing required key: {'.'.join(key_path)}")
            d = d[k]


def _load_dotenv():
    """Load .env from repo root or Updated Extraction folder so WESAD_PATH is available."""
    try:
        from dotenv import load_dotenv
        # Try repo root first (standard place for .env), then script directory
        load_dotenv(_REPO_ROOT / ".env")
        load_dotenv(_SCRIPT_DIR / ".env")
    except ImportError:
        pass  # python-dotenv optional; env var still works if set in shell


def load_config(config_path=None):
    """
    Load configuration from .env, then YAML file with fallback to defaults.

    - Loads .env from repo root or Updated Extraction/ so WESAD_PATH can be set per machine.
    - If config_path is None, looks for config.yaml next to this module.
    - paths.wesad_path: WESAD_PATH (env / .env) > config.yaml > repo dataset/raw.

    Returns:
        dict: Merged config. paths.wesad_path is always set.
    """
    import yaml

    _load_dotenv()
    defaults = get_default_config()
    config_dir = Path(__file__).resolve().parent
    path_to_try = Path(config_path) if config_path else config_dir / "config.yaml"

    if path_to_try.is_file():
        try:
            with open(path_to_try, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
            if loaded:
                cfg = _deep_merge(defaults, loaded)
            else:
                cfg = defaults
        except Exception:
            cfg = defaults
        used_path = str(path_to_try)
    else:
        cfg = defaults
        used_path = None

    # Env override for dataset path
    wesad_env = os.environ.get("WESAD_PATH")
    if wesad_env:
        cfg["paths"]["wesad_path"] = wesad_env
    elif cfg["paths"].get("wesad_path") is None:
        cfg["paths"]["wesad_path"] = _DEFAULT_WESAD_PATH

    _validate_config(cfg)
    # Ensure label mapping keys are ints (YAML may load as str)
    mapping = cfg["labels"]["mapping"]
    cfg["labels"]["mapping"] = {int(k): v for k, v in mapping.items()}
    # Attach config source for manifest
    cfg["_config_source"] = used_path if used_path else "defaults"
    return cfg
