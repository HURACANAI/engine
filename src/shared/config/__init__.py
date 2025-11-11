"""
Configuration Module

Provides type-safe configuration loading and validation using Pydantic.
"""

from .schema import (
    HuracanConfig,
    GeneralConfig,
    EngineConfig,
    MechanicConfig,
    HamiltonConfig,
    CostsConfig,
    RegimeClassifierConfig,
    DatabaseConfig,
    S3Config,
    TelegramConfig,
    SchedulerConfig,
    ModelType,
    EncoderType,
    SchedulerMode,
)
from .loader import (
    load_config,
    load_config_section,
    load_yaml_config,
    resolve_env_vars,
)

__all__ = [
    # Config classes
    "HuracanConfig",
    "GeneralConfig",
    "EngineConfig",
    "MechanicConfig",
    "HamiltonConfig",
    "CostsConfig",
    "RegimeClassifierConfig",
    "DatabaseConfig",
    "S3Config",
    "TelegramConfig",
    "SchedulerConfig",
    # Enums
    "ModelType",
    "EncoderType",
    "SchedulerMode",
    # Loaders
    "load_config",
    "load_config_section",
    "load_yaml_config",
    "resolve_env_vars",
]
