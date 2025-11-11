"""
Simple Config Loader

Loads configuration from a single config.yaml file.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to config file (defaults to config.yaml in project root)
        
    Returns:
        Configuration dictionary
    """
    if config_path is None:
        # Find config.yaml in project root
        project_root = Path(__file__).parent.parent.parent
        config_path = project_root / "config.yaml"
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Resolve environment variables
    config = _resolve_env_vars(config)
    
    return config


def _resolve_env_vars(config: Any) -> Any:
    """Resolve environment variables in config values."""
    if isinstance(config, dict):
        return {k: _resolve_env_vars(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [_resolve_env_vars(item) for item in config]
    elif isinstance(config, str) and config.startswith("${") and config.endswith("}"):
        # Extract env var name
        env_var = config[2:-1]
        return os.getenv(env_var, config)
    else:
        return config


def get_config_value(config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """Get configuration value by dot-separated key path.
    
    Args:
        config: Configuration dictionary
        key_path: Dot-separated key path (e.g., "engine.lookback_days")
        default: Default value if key not found
        
    Returns:
        Configuration value
    """
    keys = key_path.split(".")
    value = config
    
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    
    return value

