"""
Configuration Loader with Validation

Loads and validates configuration using Pydantic schemas.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
import structlog
from pydantic import ValidationError

from .schema import HuracanConfig
from src.shared.exceptions import ConfigurationError

logger = structlog.get_logger(__name__)


def resolve_env_vars(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively resolve environment variables in configuration.

    Supports ${VAR_NAME} syntax for environment variable substitution.

    Args:
        config_dict: Configuration dictionary

    Returns:
        Configuration with resolved environment variables
    """
    env_pattern = re.compile(r"\$\{([^}]+)\}")

    def resolve_value(value: Any) -> Any:
        """Resolve a single value."""
        if isinstance(value, str):
            # Find all environment variables in the string
            matches = env_pattern.findall(value)
            for var_name in matches:
                env_value = os.environ.get(var_name)
                if env_value is None:
                    logger.warning("env_var_not_found", var_name=var_name)
                    # Keep the placeholder if variable not found
                    continue
                # Replace the placeholder
                value = value.replace(f"${{{var_name}}}", env_value)
            return value
        elif isinstance(value, dict):
            return {k: resolve_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [resolve_value(item) for item in value]
        else:
            return value

    return resolve_value(config_dict)


def load_yaml_config(config_path: Path) -> Dict[str, Any]:
    """
    Load YAML configuration file.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary

    Raises:
        ConfigurationError: If configuration cannot be loaded
    """
    try:
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)
        logger.info("config_loaded", path=str(config_path))
        return config_dict
    except FileNotFoundError as e:
        raise ConfigurationError(
            f"Configuration file not found: {config_path}",
            context={"path": str(config_path)}
        ) from e
    except yaml.YAMLError as e:
        raise ConfigurationError(
            f"Invalid YAML in configuration file: {e}",
            context={"path": str(config_path)}
        ) from e
    except Exception as e:
        raise ConfigurationError(
            f"Failed to load configuration: {e}",
            context={"path": str(config_path)}
        ) from e


def load_config(
    config_path: Optional[Path] = None,
    validate: bool = True,
) -> HuracanConfig:
    """
    Load and validate configuration.

    Args:
        config_path: Path to configuration file (default: config.yaml in project root)
        validate: Whether to validate configuration with Pydantic schema

    Returns:
        Validated configuration

    Raises:
        ConfigurationError: If configuration is invalid

    Example:
        >>> config = load_config()
        >>> print(config.engine.lookback_days)
        180
    """
    # Default config path
    if config_path is None:
        config_path = Path("config.yaml")
        if not config_path.exists():
            # Try alternative locations
            alternatives = [
                Path("../config.yaml"),
                Path("../../config.yaml"),
                Path.home() / ".huracan" / "config.yaml",
            ]
            for alt in alternatives:
                if alt.exists():
                    config_path = alt
                    break

    if not config_path.exists():
        raise ConfigurationError(
            f"Configuration file not found: {config_path}",
            context={"searched_paths": [str(config_path)] + [str(p) for p in alternatives]}
        )

    # Load YAML
    config_dict = load_yaml_config(config_path)

    # Resolve environment variables
    config_dict = resolve_env_vars(config_dict)

    # Validate with Pydantic if requested
    if validate:
        try:
            config = HuracanConfig(**config_dict)
            logger.info("config_validated", path=str(config_path))
            return config
        except ValidationError as e:
            error_details = []
            for error in e.errors():
                loc = " -> ".join(str(x) for x in error["loc"])
                msg = error["msg"]
                error_details.append(f"{loc}: {msg}")

            raise ConfigurationError(
                f"Configuration validation failed:\n" + "\n".join(error_details),
                context={"path": str(config_path), "errors": error_details}
            ) from e
    else:
        # Return unvalidated config (for backward compatibility)
        logger.warning("config_validation_skipped", path=str(config_path))
        return config_dict  # type: ignore


def load_config_section(
    section: str,
    config_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Load a specific section of configuration.

    Args:
        section: Configuration section name (e.g., "engine", "mechanic")
        config_path: Path to configuration file

    Returns:
        Configuration section dictionary

    Raises:
        ConfigurationError: If section not found or configuration invalid
    """
    config = load_config(config_path, validate=False)

    if section not in config:
        raise ConfigurationError(
            f"Configuration section not found: {section}",
            context={"section": section, "available_sections": list(config.keys())}
        )

    return config[section]
