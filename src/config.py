"""Configuration loader for the MLOps pipeline."""

from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str = "config.yaml") -> dict[str, Any]:
    """Load and return configuration from a YAML file.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the config file is not valid YAML.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path) as f:
        config = yaml.safe_load(f)

    return config


def get_nested(config: dict, *keys, default: Any = None) -> Any:
    """Safely retrieve a nested value from config.

    Example:
        get_nested(config, "storage", "db_path", default="storage/mlops.sqlite")
    """
    current = config
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key, default)
        if current is default:
            return default
    return current
