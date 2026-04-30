"""
Configuration loader for V3 pipeline.
Path priority: WESAD_PATH (env / .env) > config.yaml paths.wesad_path > repo-relative default.
Ported from V1 (Updated_Extraction/config_loader.py) with V3-specific required keys.
"""

import os
from pathlib import Path

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
    ("windowing", "hrv_freq_window"),
    ("preprocessing", "eda"),
    ("preprocessing", "bvp"),
    ("preprocessing", "temp"),
    ("preprocessing", "acc"),
    ("labels", "mapping"),
)


def _deep_merge(base, override):
    if not isinstance(override, dict):
        return override
    result = dict(base)
    for k, v in override.items():
        result[k] = _deep_merge(result.get(k, {}), v) if isinstance(v, dict) and isinstance(result.get(k), dict) else v
    return result


def _validate_config(cfg):
    for key_path in _REQUIRED_KEYS:
        d = cfg
        for k in key_path:
            if not isinstance(d, dict) or k not in d:
                raise ValueError(f"Config missing required key: {'.'.join(key_path)}")
            d = d[k]


def _load_dotenv():
    try:
        from dotenv import load_dotenv
        load_dotenv(_REPO_ROOT / ".env")
        load_dotenv(_SCRIPT_DIR / ".env")
    except ImportError:
        pass


def load_config(config_path=None):
    """Load V3 config from YAML, with .env-driven WESAD_PATH override."""
    import yaml

    _load_dotenv()
    config_dir = Path(__file__).resolve().parent
    path_to_try = Path(config_path) if config_path else config_dir / "config.yaml"

    if not path_to_try.is_file():
        raise FileNotFoundError(
            f"V3 config.yaml not found at {path_to_try}. V3 is config-driven; no defaults are baked in."
        )
    with open(path_to_try, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    wesad_env = os.environ.get("WESAD_PATH")
    if wesad_env:
        cfg.setdefault("paths", {})["wesad_path"] = wesad_env
    elif cfg.get("paths", {}).get("wesad_path") is None:
        cfg.setdefault("paths", {})["wesad_path"] = _DEFAULT_WESAD_PATH

    _validate_config(cfg)
    cfg["labels"]["mapping"] = {int(k): v for k, v in cfg["labels"]["mapping"].items()}
    cfg["_config_source"] = str(path_to_try)
    return cfg
