"""Structured configuration loading for the Huracan Engine."""

from __future__ import annotations

import os
from datetime import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class SchedulerSettings(BaseModel):
    daily_run_time_utc: time = Field(..., description="HH:MM time the engine starts")
    timezone: str = "UTC"


class UniverseSettings(BaseModel):
    target_size: int = 20
    liquidity_threshold_adv_gbp: int = 10_000_000
    max_spread_bps: int = 8
    rebalance_frequency_days: int = 7


class WalkForwardSettings(BaseModel):
    train_days: int = 20
    test_days: int = 5
    min_trades: int = 300


class RLAgentSettings(BaseModel):
    enabled: bool = True
    learning_rate: float = 0.0003
    gamma: float = 0.99
    epsilon: float = 0.2
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    n_epochs: int = 10
    epochs_per_update: int = 10
    batch_size: int = 64
    hidden_dim: int = 256
    state_dim: int = 80
    device: str = "cpu"


class ShadowTradingSettings(BaseModel):
    enabled: bool = True
    position_size_gbp: float = 1000.0
    stop_loss_bps: int = 50
    take_profit_bps: int = 20
    max_hold_hours: int = 24
    max_hold_minutes: int = 120
    min_confidence_threshold: float = 0.52


class MemorySettings(BaseModel):
    vector_dim: int = 128
    similarity_threshold: float = 0.7
    max_similar_patterns: int = 10
    pattern_min_trades: int = 5


class MonitoringSettings(BaseModel):
    enabled: bool = True
    check_interval_seconds: int = 300
    win_rate_stddev_threshold: float = 2.0
    profit_stddev_threshold: float = 2.0
    volume_stddev_threshold: float = 2.5
    auto_remediation_enabled: bool = True
    pause_failing_patterns: bool = True
    pattern_failure_threshold: float = 0.45


class TrainingSettings(BaseModel):
    window_days: int = 150
    walk_forward: WalkForwardSettings = Field(default_factory=WalkForwardSettings)
    rl_agent: RLAgentSettings = Field(default_factory=RLAgentSettings)
    shadow_trading: ShadowTradingSettings = Field(default_factory=ShadowTradingSettings)
    memory: MemorySettings = Field(default_factory=MemorySettings)
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)


class CostSettings(BaseModel):
    target_net_bps: int = 15
    taker_buffer_bps: int = 9
    default_fee_bps: float = 8.0
    default_spread_bps: float = 6.0
    slippage_alpha: float = 1.1
    notional_per_trade: float = 1_000.0
    slippage_floor_bps: float = 1.0
    volatility_slippage_multiplier: float = 0.25
    adv_liquidity_breakpoints: List[float] = Field(default_factory=lambda: [0.0005, 0.002, 0.01])
    adv_penalties_bps: List[float] = Field(default_factory=lambda: [12.0, 6.0, 3.0, 1.0])


class S3Settings(BaseModel):
    bucket: str = "huracan-engine"
    prefix: str = "baselines"
    endpoint_url: Optional[str] = None
    access_key: Optional[str] = None
    secret_key: Optional[str] = None


class PostgresSettings(BaseModel):
    dsn: str = Field(..., description="SQLAlchemy connection string")


class RaySettings(BaseModel):
    address: Optional[str] = None
    namespace: str = "huracan-engine"
    runtime_env: Dict[str, Any] = Field(default_factory=dict)


class NotificationSettings(BaseModel):
    telegram_enabled: bool = False
    telegram_webhook_url: Optional[str] = None
    telegram_chat_id: Optional[str] = None


class ExchangeCredential(BaseModel):
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    api_passphrase: Optional[str] = None


class ExchangeSettings(BaseModel):
    primary: str = Field("binance", description="Primary exchange identifier")
    fallbacks: List[str] = Field(default_factory=lambda: ["binanceusdm", "coinbase"], description="Alternative exchanges compatible with the engine")
    sandbox: bool = False
    credentials: Dict[str, ExchangeCredential] = Field(default_factory=dict)

    @field_validator("primary")
    @classmethod
    def _ensure_lowercase(cls, value: str) -> str:
        return value.lower()

    @field_validator("credentials", mode="before")
    @classmethod
    def _normalise_keys(cls, value: Dict[str, Any]) -> Dict[str, Any]:
        if not value:
            return {}
        return {str(key).lower(): val for key, val in value.items()}


class EngineSettings(BaseSettings):
    """Loads layered configuration combining YAML profiles and env overrides."""

    environment: str = Field("local", alias="HURACAN_ENV")
    mode: str = Field("shadow", alias="HURACAN_MODE")
    scheduler: SchedulerSettings
    universe: UniverseSettings
    training: TrainingSettings
    costs: CostSettings
    artifacts: S3Settings
    s3: S3Settings = Field(default_factory=S3Settings)
    postgres: Optional[PostgresSettings] = None
    ray: RaySettings = Field(default_factory=RaySettings)
    notifications: NotificationSettings = Field(default_factory=NotificationSettings)
    exchange: ExchangeSettings = Field(default_factory=ExchangeSettings)

    config_dir: Path = Field(Path(__file__).resolve().parent.parent.parent.parent / "config", alias="HURACAN_CONFIG_DIR")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    @classmethod
    def load(cls, environment: Optional[str] = None, config_dir: Optional[Path] = None) -> "EngineSettings":
        base_dir = config_dir or Path(__file__).resolve().parent.parent.parent.parent / "config"
        env_name = environment or os.getenv("HURACAN_ENV", "local")
        merged: Dict[str, Any] = cls._load_yaml(base_dir / "base.yaml")
        env_file = base_dir / f"{env_name}.yaml"
        if env_file.exists():
            merged = cls._deep_merge(merged, cls._load_yaml(env_file))
        merged.pop("inherit_from", None)
        merged["environment"] = env_name
        merged.setdefault("mode", os.getenv("HURACAN_MODE", "shadow"))
        explicit_env: Dict[str, Any] = {}
        return cls.model_validate(cls._deep_merge(merged, explicit_env))

    @field_validator("mode")
    @classmethod
    def _validate_mode(cls, value: str) -> str:
        norm = value.lower()
        if norm not in {"shadow", "live"}:
            raise ValueError("mode must be either 'shadow' or 'live'")
        return norm

    @staticmethod
    def _load_yaml(path: Path) -> Dict[str, Any]:
        with path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}

    @staticmethod
    def _deep_merge(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
        result = dict(base)
        for key, value in overrides.items():
            if isinstance(value, dict) and isinstance(result.get(key), dict):
                result[key] = EngineSettings._deep_merge(result[key], value)
            else:
                result[key] = value
        return result
