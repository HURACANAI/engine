"""Structured configuration loading for the Huracan Engine."""

from __future__ import annotations

import os
from datetime import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml  # type: ignore[reportMissingImports]
from pydantic import BaseModel, Field, field_validator  # type: ignore[reportMissingImports]
from pydantic_settings import BaseSettings, SettingsConfigDict  # type: ignore[reportMissingImports]


class SchedulerSettings(BaseModel):
    daily_run_time_utc: time = Field(..., description="HH:MM time the engine starts")
    timezone: str = "UTC"


class UniverseSettings(BaseModel):
    target_size: int = 20
    liquidity_threshold_adv_gbp: int = 10_000_000
    max_spread_bps: int = 8
    rebalance_frequency_days: int = 7


class WalkForwardSettings(BaseModel):
    train_days: int = 45  # Increased from 20 to 45 days for more variety per split
    test_days: int = 10  # Increased from 5 to 10 days
    min_trades: int = 300


class ModelTrainingSettings(BaseModel):
    """LightGBM model training hyperparameters."""
    n_estimators: int = 2000  # Increased from 300 for proper training
    learning_rate: float = 0.01  # Lower learning rate for better learning
    max_depth: int = 8  # Increased depth for more complex patterns
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_samples: int = 32
    early_stopping_rounds: int = 100  # Enable early stopping
    verbose: int = -1  # Suppress verbose output
    n_jobs: int = -1  # Use all CPU cores
    random_state: int = 42


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
    stop_loss_bps: int = 200  # Hard stop at 2% (Grok AI recommendation)
    take_profit_bps: int = 20
    max_hold_hours: int = 24
    max_hold_minutes: int = 120
    min_confidence_threshold: float = 0.40  # Raised from 0.20 (Grok AI recommendation)


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


class MandatoryOOSSettings(BaseModel):
    enabled: bool = True
    min_oos_sharpe: float = 1.0
    min_oos_win_rate: float = 0.55
    max_train_test_gap: float = 0.3
    max_sharpe_std: float = 0.2
    min_test_trades: int = 100
    min_windows: int = 5


class OverfittingDetectionSettings(BaseModel):
    enabled: bool = True
    train_test_gap_threshold: float = 0.5
    cv_stability_threshold: float = 0.3
    degradation_threshold: float = -0.2


class DataValidationSettings(BaseModel):
    enabled: bool = True
    outlier_z_threshold: float = 3.0
    max_missing_pct: float = 0.05
    max_age_hours: int = 24
    min_coverage: float = 0.95


class PaperTradingSettings(BaseModel):
    enabled: bool = False
    min_duration_days: int = 14
    min_trades: int = 100
    min_win_rate: float = 0.55
    min_sharpe: float = 1.0
    max_backtest_deviation: float = 0.20


class StressTestingSettings(BaseModel):
    enabled: bool = False
    max_drawdown_threshold: float = 0.30
    min_survival_rate: float = 0.70


class ValidationSettings(BaseModel):
    enabled: bool = True
    mandatory_oos: MandatoryOOSSettings = Field(default_factory=MandatoryOOSSettings)
    overfitting_detection: OverfittingDetectionSettings = Field(default_factory=OverfittingDetectionSettings)
    data_validation: DataValidationSettings = Field(default_factory=DataValidationSettings)
    paper_trading: PaperTradingSettings = Field(default_factory=PaperTradingSettings)
    stress_testing: StressTestingSettings = Field(default_factory=StressTestingSettings)


class ParallelProcessingSettings(BaseModel):
    enabled: bool = True
    num_workers: int = 6
    use_ray: bool = True


class CachingSettings(BaseModel):
    enabled: bool = True
    max_size: int = 1000
    default_ttl: int = 3600


class QueryOptimizationSettings(BaseModel):
    enabled: bool = True
    enable_caching: bool = True
    cache_ttl: int = 3600


class OptimizationSettings(BaseModel):
    parallel_processing: ParallelProcessingSettings = Field(default_factory=ParallelProcessingSettings)
    caching: CachingSettings = Field(default_factory=CachingSettings)
    query_optimization: QueryOptimizationSettings = Field(default_factory=QueryOptimizationSettings)


class PerCoinGatesSettings(BaseModel):
    """Gate thresholds for per-coin model promotion."""
    min_sharpe: float = 1.0
    min_hit_rate: float = 0.50  # 50% hit rate
    max_drawdown_pct: float = 15.0  # 15% max drawdown
    min_net_pnl_pct: float = 1.0  # 1% net P&L improvement over champion
    min_sample_size: int = 100  # Minimum sample size for validation


class PerCoinPromotionRulesSettings(BaseModel):
    """Promotion rules for per-coin challenger promotion."""
    min_hit_rate_improvement: float = 0.01  # 1% improvement
    min_sharpe_improvement: float = 0.2  # 0.2 improvement
    max_drawdown_tolerance: float = 0.0  # No tolerance (must be better or equal)
    min_net_pnl_improvement: float = 0.01  # 1% improvement


class PerCoinTrainingSettings(BaseModel):
    """Per-coin training configuration."""
    symbols_allowed: List[str] = Field(default_factory=list)  # Empty list = all symbols
    per_symbol_costs: Dict[str, Dict[str, float]] = Field(default_factory=dict)  # Per-symbol cost overrides
    parallel_tasks: int = 2  # Number of symbols to train in parallel
    time_budget_per_symbol_minutes: int = 30  # Time budget per symbol in minutes
    gates: PerCoinGatesSettings = Field(default_factory=PerCoinGatesSettings)
    promotion_rules: PerCoinPromotionRulesSettings = Field(default_factory=PerCoinPromotionRulesSettings)


class AdvancedTrainingSettings(BaseModel):
    """Advanced training features configuration."""
    use_multi_model_ensemble: bool = True  # Use ensemble of multiple models
    ensemble_techniques: List[str] = Field(default_factory=lambda: ["xgboost", "random_forest", "lightgbm"])
    ensemble_method: str = "weighted_voting"  # "weighted_voting", "stacking", "dynamic"
    ensemble_weights: Optional[Dict[str, float]] = None  # Fixed ensemble weights (e.g., {"xgboost": 0.50, "lightgbm": 0.30, "random_forest": 0.20})
    use_fixed_ensemble_weights: bool = False  # Use fixed weights instead of performance-based
    edge_threshold_override_bps: Optional[float] = None  # Override recommended edge threshold (lower to capture more trades)
    
    use_progressive_training: bool = True  # Train on full coin history
    progressive_initial_epoch_days: int = 730  # 2 years for first epoch
    progressive_subsequent_epoch_days: int = 365  # 1 year for subsequent epochs
    progressive_max_epochs: Optional[int] = None  # None = train until inception
    
    use_enhanced_rl: bool = True  # Use enhanced RL pipeline with Phase 1 features
    use_rl_v2_pipeline: bool = True  # Use V2 pipeline (triple-barrier, meta-labeling)
    enable_advanced_rewards: bool = True  # Advanced reward shaping
    enable_higher_order_features: bool = True  # Higher-order feature engineering
    enable_granger_causality: bool = True  # Granger causality for cross-asset timing
    enable_regime_prediction: bool = True  # Regime transition prediction
    
    use_meta_labeling: bool = True  # Meta-labeling for profitable trades
    meta_label_cost_threshold: float = 0.0  # Cost threshold for meta-labeling
    use_triple_barrier: bool = True  # Triple-barrier labeling (no lookahead bias)
    use_recency_weighting: bool = True  # Weight recent data higher
    
    use_v2_data_quality: bool = True  # V2 data quality pipeline
    enable_auto_window_selection: bool = False
    candidate_window_days: List[int] = Field(default_factory=lambda: [180, 270, 365])
    enable_hyperparam_search: bool = False


class TrainingSettings(BaseModel):
    window_days: int = 365  # Increased from 150 to 365+ days for broader training data
    walk_forward: WalkForwardSettings = Field(default_factory=WalkForwardSettings)
    model_training: ModelTrainingSettings = Field(default_factory=ModelTrainingSettings)
    advanced: AdvancedTrainingSettings = Field(default_factory=AdvancedTrainingSettings)
    rl_agent: RLAgentSettings = Field(default_factory=RLAgentSettings)
    shadow_trading: ShadowTradingSettings = Field(default_factory=ShadowTradingSettings)
    memory: MemorySettings = Field(default_factory=MemorySettings)
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)
    validation: ValidationSettings = Field(default_factory=ValidationSettings)
    optimization: OptimizationSettings = Field(default_factory=OptimizationSettings)
    per_coin: PerCoinTrainingSettings = Field(default_factory=PerCoinTrainingSettings)


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
    grok_api_key: Optional[str] = Field(default=None, description="Grok API key (can also be set via GROK_API_KEY env var)")
    grok_enabled: bool = True  # Enable Grok explanations by default if API key is set
    
    def __init__(self, **data: Any) -> None:
        # If grok_api_key not provided in data, try environment variable
        if "grok_api_key" not in data or data.get("grok_api_key") is None:
            data["grok_api_key"] = os.getenv("GROK_API_KEY") or data.get("grok_api_key")
        super().__init__(**data)


class DropboxSettings(BaseModel):
    enabled: bool = True  # Enabled by default
    # Token can be set via environment variable DROPBOX_ACCESS_TOKEN or hardcoded here
    access_token: Optional[str] = os.getenv("DROPBOX_ACCESS_TOKEN") or "sl.u.AGEfLXv_9XC-CMSpJ-ICWQXMwTk22GdAmQRu73hCE6ObsZv6hztL9DKQlbsrQq0xlp5I3Y9Kxf_xtfPRGHt48Dw79QT4GgMwsAUfTlBFPIynodyEXLymk1OdG9UyBNwBYKSVikWgrjehD5b4Tz1UiOqx4mkwphtx4o_CfkF4zMWKfS-TVq50iMqrhtUeb4_cD4bQXAAP6oBqgWWLlCmyYdqkx9Ba2Xd-FX_Pg66nwQvijMhiq0n7twIqBXe6PJ9xYO7ok8KEeV_9IQlPUwfJbW4z7jbg1gnhPvZ86krPvTj3Zj--qQXRm1G5Eu48tjKeiJAmyXUQBg-582ntphrS7YPl97IhyKQR2Vc5P8XzWYKe5SBQXgWuudTS-NZ784LoB8pEl3ToB2ol2_AymY1es9g0bRZ-TjNFtaUhdO-zssa2tkJc-stIs6hZrJnMJCnQYFNDw4HoG1QnFY1paK_2uoeHuOavScNgg8C1kMMWw7O_yER8sMW-Xbqs3zJMrq9BbGL4hAgqyvogNdF1H70HC_UR2jn_3Is8TbIu-_jq61RJ2u9jxHGOQUO--B5idNR5v0YMga1V8Bq9JE3hf1RuKIkMJt4WZFelvfoEsx8ipFqYOOs4vnXfJ5cTyAjzc36jArfOgG4C44DKQvx0TaLeWk_b6e-16Rv3zie8mPiBSCLMcD3lF0TK_6IUbQiRsBxregtw7LaxsQNMVhRjf-jNHIOVw1fH_fCh0CEbZXcYoUotQfHk0frfo-krSSwS8GOogj3TMfKIMs73QoOurpzwEz1ijUkQBAn8oj93exQ3S1cp4m31jJ5-POFSxKhs949v88gBbrPJ-SR04gysO_jJyuvvVqHY6P839jS-hgcVC94LtKvLf4JhmBRlfan9P_KHU1KfVxjCrfGEX68IiE6NtMVnRT1R5i6__6Vw6162wwzsGIFpb_Pxryu77KwEbYpbJDRH4X5X3R1QUhkVT3Vj9HdxDuLuC2Y1solrB1G4-I8JbNpe0n-QaY2b_9PlZyaSN5BJCV0BSeUZIpVBbwJXt-XSIVLwjQqiSj8JK-zY6XzDAc98iOUj0F8rOdIJ06dyMhz1hLRLxwvCUYBLP-1pCfbwi9CGondWA17BTu_n0dEnGbHvcZD2GxnvwcT-y7G0DMNX1EjJoV4HQem9kxDOwHfduL6vdRZH1ygmspcSnwt3lf-hG0yo-08V_fVxMMSp8QpdEBCmDGjcg2kJhN4aAs6EliYj4mDkKh_SkHOw9lmX9aFnshB9Eyho7EvjUhib767y0PRHHY9VUfkXE5LDb2XZ"
    app_folder: str = "Runpodhuracan"
    sync_logs: bool = True
    sync_models: bool = True
    sync_monitoring: bool = True
    sync_learning: bool = True  # Sync learning data (patterns, insights, etc.)
    sync_data_cache: bool = True  # Sync historical coin data (parquet files)
    # Sync intervals in seconds
    # Learning data: sync frequently (every 5 min) to capture insights quickly
    sync_interval_learning_seconds: int = 300  # 5 minutes
    # Logs & monitoring: sync frequently (every 5 min) for real-time monitoring
    sync_interval_logs_seconds: int = 300  # 5 minutes
    # Models: sync less frequently (every 30 min) - models don't change as often
    sync_interval_models_seconds: int = 1800  # 30 minutes
    # Historical data cache: sync less frequently (every 2 hours) - large files, don't change often
    sync_interval_data_cache_seconds: int = 7200  # 2 hours
    # Enable automatic restore of historical data from Dropbox on startup
    restore_data_cache_on_startup: bool = True


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

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
    )

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
    dropbox: DropboxSettings = Field(default_factory=DropboxSettings)
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
        base_dir = config_dir or Path(__file__).resolve().parent.parent.parent.parent.parent / "config"
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
