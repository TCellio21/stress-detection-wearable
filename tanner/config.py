"""
Configuration Management Module

Load and manage project configuration from config.yaml
"""

import yaml
from pathlib import Path
from typing import Dict, Any


class Config:
    """Singleton class to manage project configuration."""

    _instance = None
    _config = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._load_config()
        return cls._instance

    @classmethod
    def _load_config(cls):
        """Load configuration from config.yaml."""
        # Get project root (parent of src/)
        project_root = Path(__file__).parent.parent
        config_path = project_root / 'config.yaml'

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, 'r') as f:
            cls._config = yaml.safe_load(f)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.

        Supports nested keys with dot notation: 'data.raw_dir'

        Args:
            key: Configuration key (can be nested with dots)
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def get_path(self, key: str) -> Path:
        """
        Get configuration value as Path object.

        Args:
            key: Configuration key

        Returns:
            Path object relative to project root
        """
        project_root = Path(__file__).parent.parent
        path_str = self.get(key)

        if path_str is None:
            raise ValueError(f"Config key '{key}' not found")

        return project_root / path_str

    @property
    def all(self) -> Dict:
        """Get all configuration."""
        return self._config


# Convenience function
def load_config() -> Config:
    """Load and return configuration singleton."""
    return Config()


if __name__ == "__main__":
    # Test configuration loading
    config = load_config()

    print("Configuration loaded successfully!")
    print("\nSample values:")
    print(f"Window size: {config.get('preprocessing.window_size')} seconds")
    print(f"Chest sampling rate: {config.get('sampling_rates.chest')} Hz")
    print(f"Validation strategy: {config.get('training.validation_strategy')}")
    print(f"Raw data path: {config.get_path('data.raw_dir')}")
