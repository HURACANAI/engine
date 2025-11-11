"""
Configuration Schema using Pydantic

Provides type-safe configuration validation for the Huracan trading engine.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

from pydantic import BaseModel, Field, field_validator, ConfigDict
from pydantic_settings import BaseSettings


class ModelType(str, Enum):
    """Supported model types."""
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    CATBOOST = "catboost"
    RANDOM_FOREST = "random_forest"


class EncoderType(str, Enum):
    """Shared encoder types."""
    PCA = "pca"
    AUTOENCODER = "autoencoder"


class SchedulerMode(str, Enum):
    """Scheduler execution modes."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    HYBRID = "hybrid"


class GeneralConfig(BaseModel):
    """General configuration."""
    version: str = Field(default="2.0", description="Configuration version")
    timezone: str = Field(default="UTC", description="System timezone")
    dropbox_root: str = Field(default="/Huracan/", description="Dropbox root folder")
    s3_bucket: str = Field(default="huracan", description="S3 bucket name")
    symbols: List[str] = Field(default=["BTCUSDT", "ETHUSDT", "SOLUSDT"], description="Trading symbols")


class SharedEncoderConfig(BaseModel):
    """Shared encoder configuration."""
    type: EncoderType = Field(default=EncoderType.PCA, description="Encoder type")
    n_components: int = Field(default=50, ge=1, le=1000, description="Number of components")
    enabled: bool = Field(default=True, description="Enable shared encoder")


class DataGatesConfig(BaseModel):
    """Data quality gates."""
    min_volume_gbp: float = Field(default=10_000_000, ge=0, description="Minimum volume in GBP")
    max_spread_bps: float = Field(default=8.0, ge=0, description="Maximum spread in bps")
    max_gap_pct: float = Field(default=0.10, ge=0, le=1, description="Maximum gap percentage")


class SlippageCalibrationConfig(BaseModel):
    """Slippage calibration configuration."""
    enabled: bool = Field(default=True, description="Enable slippage calibration")
    lookback_days: int = Field(default=30, ge=1, le=365, description="Lookback days for calibration")


class GuardrailsConfig(BaseModel):
    """Trading guardrails."""
    net_edge_floor_bps: float = Field(default=3.0, ge=0, description="Minimum net edge to trade")
    spread_threshold_bps: float = Field(default=50.0, ge=0, description="Maximum spread to trade")
    sample_size_min: int = Field(default=100, ge=1, description="Minimum sample size for promotion")


class EngineConfig(BaseModel):
    """Engine configuration."""
    lookback_days: int = Field(default=180, ge=1, le=1000, description="Lookback days for training")
    features: List[str] = Field(default=["rsi", "ema", "volatility", "momentum"], description="Feature list")
    model_type: ModelType = Field(default=ModelType.XGBOOST, description="Model type")
    parallel_tasks: int = Field(default=8, ge=1, le=128, description="Parallel tasks")
    shared_encoder: SharedEncoderConfig = Field(default_factory=SharedEncoderConfig)
    target_symbols: int = Field(default=400, ge=1, description="Target number of symbols")
    start_with_symbols: int = Field(default=150, ge=1, description="Starting number of symbols")
    data_gates: DataGatesConfig = Field(default_factory=DataGatesConfig)
    slippage_calibration: SlippageCalibrationConfig = Field(default_factory=SlippageCalibrationConfig)
    guardrails: GuardrailsConfig = Field(default_factory=GuardrailsConfig)


class MechanicGuardrailsConfig(BaseModel):
    """Mechanic guardrails."""
    min_hours_live_before_promotion: int = Field(default=24, ge=0, description="Minimum hours live before promotion")
    min_trades_before_promotion: int = Field(default=50, ge=0, description="Minimum trades before promotion")


class RollbackRulesConfig(BaseModel):
    """Rollback rules."""
    drawdown_floor_pct: float = Field(default=10.0, ge=0, le=100, description="Drawdown floor percentage")
    win_rate_floor_pct: float = Field(default=40.0, ge=0, le=100, description="Win rate floor percentage")


class StaggerWorkConfig(BaseModel):
    """Stagger work configuration."""
    enabled: bool = Field(default=True, description="Enable stagger work")
    round_robin_symbols: bool = Field(default=True, description="Round robin symbols")
    max_concurrent_challengers: int = Field(default=2, ge=1, description="Max concurrent challengers")


class MechanicConfig(BaseModel):
    """Mechanic configuration."""
    fine_tune_hours: int = Field(default=6, ge=1, le=168, description="Fine tune hours")
    challengers_per_symbol: int = Field(default=3, ge=1, le=10, description="Challengers per symbol")
    promote_if_net_pnl_above_pct: float = Field(default=1.0, ge=0, description="Promotion PnL threshold")
    rollback_drawdown_pct: float = Field(default=5.0, ge=0, le=100, description="Rollback drawdown percentage")
    guardrails: MechanicGuardrailsConfig = Field(default_factory=MechanicGuardrailsConfig)
    rollback_rules: RollbackRulesConfig = Field(default_factory=RollbackRulesConfig)
    stagger_work: StaggerWorkConfig = Field(default_factory=StaggerWorkConfig)
    shadow_types: List[str] = Field(default=["fine_tune", "threshold_tweak"], description="Shadow types")


class PreTradeChecksConfig(BaseModel):
    """Pre-trade checks configuration."""
    min_balance_usd: float = Field(default=100, ge=0, description="Minimum balance in USD")
    min_notional_multiplier: float = Field(default=1.0, ge=0, description="Minimum notional multiplier")
    funding_rate_threshold_pct: float = Field(default=0.1, ge=0, description="Funding rate threshold percentage")


class SessionLimitsConfig(BaseModel):
    """Session limits configuration."""
    daily_loss_cap_pct: float = Field(default=1.0, ge=0, le=100, description="Daily loss cap percentage")
    daily_trade_count_cap: int = Field(default=50, ge=0, description="Daily trade count cap")


class LatencyMeterConfig(BaseModel):
    """Latency meter configuration."""
    enabled: bool = Field(default=True, description="Enable latency meter")
    threshold_ms: int = Field(default=200, ge=0, description="Latency threshold in milliseconds")


class HamiltonConfig(BaseModel):
    """Hamilton (live trading) configuration."""
    edge_threshold_bps: float = Field(default=10, ge=0, description="Edge threshold in bps")
    max_positions: int = Field(default=3, ge=1, le=50, description="Maximum positions")
    cooldown_sec: int = Field(default=900, ge=0, description="Cooldown seconds")
    daily_loss_cap_pct: float = Field(default=1.0, ge=0, le=100, description="Daily loss cap percentage")
    weekly_loss_cap_pct: float = Field(default=3.0, ge=0, le=100, description="Weekly loss cap percentage")
    pre_trade_checks: PreTradeChecksConfig = Field(default_factory=PreTradeChecksConfig)
    session_limits: SessionLimitsConfig = Field(default_factory=SessionLimitsConfig)
    latency_meter: LatencyMeterConfig = Field(default_factory=LatencyMeterConfig)


class CostsConfig(BaseModel):
    """Trading costs configuration."""
    taker_fee_bps: float = Field(default=4.0, ge=0, description="Taker fee in bps")
    maker_fee_bps: float = Field(default=2.0, ge=0, description="Maker fee in bps")
    median_spread_bps: float = Field(default=5.0, ge=0, description="Median spread in bps")
    slippage_bps_per_sigma: float = Field(default=2.0, ge=0, description="Slippage in bps per sigma")
    per_symbol_costs: Dict[str, Dict[str, float]] = Field(default_factory=dict, description="Per-symbol cost overrides")


class RegimeClassifierConfig(BaseModel):
    """Regime classifier configuration."""
    trend_threshold: float = Field(default=0.6, ge=0, le=1, description="Trend strength threshold")
    volatility_threshold: float = Field(default=0.05, ge=0, description="Volatility threshold for panic")
    volume_threshold: float = Field(default=0.5, ge=0, description="Volume threshold for illiquid")
    panic_allows_swing: bool = Field(default=False, description="Panic regime allows swing trading")
    panic_allows_position: bool = Field(default=False, description="Panic regime allows position trading")
    illiquid_allows_swing: bool = Field(default=False, description="Illiquid regime allows swing trading")
    illiquid_allows_position: bool = Field(default=False, description="Illiquid regime allows position trading")
    panic_risk_multiplier: float = Field(default=2.0, ge=1, description="Risk multiplier in panic")
    illiquid_risk_multiplier: float = Field(default=1.5, ge=1, description="Risk multiplier in illiquid")
    high_volatility_risk_multiplier: float = Field(default=1.3, ge=1, description="Risk multiplier in high volatility")
    volatility_threshold_high: float = Field(default=0.05, ge=0, description="High volatility threshold")
    volatility_threshold_low: float = Field(default=0.01, ge=0, description="Low volatility threshold")


class DatabaseConfig(BaseModel):
    """Database configuration."""
    connection_string: str = Field(..., description="Database connection string")
    pool_size: int = Field(default=10, ge=1, le=100, description="Connection pool size")
    max_overflow: int = Field(default=20, ge=0, le=100, description="Maximum overflow connections")


class S3Config(BaseModel):
    """S3 configuration."""
    bucket: str = Field(default="huracan", description="S3 bucket name")
    region: str = Field(default="us-east-1", description="AWS region")
    endpoint_url: Optional[str] = Field(default=None, description="S3-compatible endpoint URL")
    access_key: str = Field(..., description="AWS access key ID")
    secret_key: str = Field(..., description="AWS secret access key")


class TelegramConfig(BaseModel):
    """Telegram configuration."""
    token: str = Field(..., description="Telegram bot token")
    chat_id: str = Field(..., description="Telegram chat ID")
    symbols_selector_file: str = Field(default="symbols_selector.json", description="Symbols selector file")


class SchedulerConfig(BaseModel):
    """Scheduler configuration."""
    mode: SchedulerMode = Field(default=SchedulerMode.HYBRID, description="Scheduler mode")
    max_concurrent: int = Field(default=12, ge=1, le=128, description="Maximum concurrent tasks")
    timeout_minutes: int = Field(default=45, ge=1, le=1440, description="Timeout in minutes")
    retry_count: int = Field(default=2, ge=0, le=10, description="Retry count")
    retry_backoff_seconds: int = Field(default=2, ge=0, le=300, description="Retry backoff seconds")


class HuracanConfig(BaseSettings):
    """
    Main Huracan configuration with validation.

    This model validates all configuration settings and provides type-safe access.
    Environment variables are automatically resolved.
    """
    model_config = ConfigDict(
        env_prefix="HURACAN_",
        case_sensitive=False,
        validate_assignment=True,
        extra="allow",  # Allow extra fields for backward compatibility
    )

    general: GeneralConfig = Field(default_factory=GeneralConfig)
    engine: EngineConfig = Field(default_factory=EngineConfig)
    mechanic: MechanicConfig = Field(default_factory=MechanicConfig)
    hamilton: HamiltonConfig = Field(default_factory=HamiltonConfig)
    costs: CostsConfig = Field(default_factory=CostsConfig)
    regime_classifier: RegimeClassifierConfig = Field(default_factory=RegimeClassifierConfig)
    database: DatabaseConfig
    s3: S3Config
    telegram: TelegramConfig
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)

    @field_validator("general")
    @classmethod
    def validate_general(cls, v: GeneralConfig) -> GeneralConfig:
        """Validate general configuration."""
        if not v.symbols:
            raise ValueError("At least one symbol must be configured")
        return v

    @field_validator("engine")
    @classmethod
    def validate_engine(cls, v: EngineConfig) -> EngineConfig:
        """Validate engine configuration."""
        if v.start_with_symbols > v.target_symbols:
            raise ValueError("start_with_symbols cannot exceed target_symbols")
        return v

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self.model_dump()
