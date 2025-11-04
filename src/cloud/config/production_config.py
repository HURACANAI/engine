"""
Production Configuration

Complete production-ready configuration for the Huracan Engine.
Includes all Phase 1 and Phase 2 settings optimized for live trading.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import structlog

logger = structlog.get_logger()


@dataclass
class Phase1Config:
    """Phase 1 feature configuration."""

    # Regime Detection
    regime_panic_threshold: float = 0.7
    regime_trend_threshold: float = 0.6
    regime_range_threshold: float = 0.6
    regime_lookback_period: int = 30

    # Confidence Scoring
    min_confidence_threshold: float = 0.52
    sample_threshold: int = 20
    strong_alignment_threshold: float = 0.7

    # Feature Importance Learning
    feature_importance_ema_alpha: float = 0.05
    feature_importance_min_samples: int = 30
    feature_importance_top_k: int = 10

    # Recency Penalties
    recency_half_life_days: float = 30.0
    recency_max_age_days: float = 365.0
    recency_min_similarity: float = 0.3
    recency_enabled: bool = True


@dataclass
class Phase2Config:
    """Phase 2 feature configuration."""

    # Multi-Symbol Coordination
    max_portfolio_heat: float = 0.7  # Max 70% capital deployed
    max_correlated_exposure: float = 0.4  # Max 40% in correlated assets
    correlation_threshold: float = 0.7  # Consider correlated if > 0.7
    correlation_lookback_days: int = 30

    # Enhanced Risk Management
    base_position_size_gbp: float = 100.0
    max_position_multiplier: float = 2.0  # Max 2x base size
    min_position_multiplier: float = 0.25  # Min 0.25x base size
    kelly_fraction: float = 0.25  # Use 25% of full Kelly
    volatility_target_bps: float = 200.0  # Target 200 bps volatility
    max_portfolio_volatility: float = 0.03  # Max 3% daily portfolio vol

    # Advanced Pattern Recognition
    min_pattern_quality: float = 0.6
    pattern_lookback_periods: int = 50

    # Portfolio-Level Learning
    portfolio_ema_alpha: float = 0.05
    cross_symbol_threshold: int = 3  # Pattern must appear in 3+ symbols


@dataclass
class TradingConfig:
    """Core trading configuration."""

    # Capital Management
    total_capital_gbp: float = 10000.0
    max_positions: int = 5
    max_position_size_pct: float = 0.20  # Max 20% per position

    # Risk Parameters
    default_stop_loss_bps: float = 100.0  # 100 bps = 1%
    default_take_profit_bps: float = 200.0  # 200 bps = 2%
    max_drawdown_pct: float = 20.0  # Stop trading at 20% drawdown

    # Execution
    slippage_bps: float = 5.0  # Assume 5 bps slippage
    maker_fee_bps: float = 2.0
    taker_fee_bps: float = 5.0

    # Timing
    trading_enabled: bool = True
    paper_trading_mode: bool = True  # Safe default
    max_trades_per_day: int = 20


@dataclass
class BacktestConfig:
    """Backtesting configuration."""

    lookback_days: int = 90
    lookback_for_optimal_exit: int = 50
    min_candles_for_shadow_trading: int = 60


@dataclass
class DatabaseConfig:
    """Database configuration."""

    host: str = "localhost"
    port: int = 5432
    database: str = "huracan_engine"
    user: str = "postgres"
    password: str = ""  # Should be loaded from environment
    min_pool_size: int = 2
    max_pool_size: int = 10


@dataclass
class LoggingConfig:
    """Logging configuration."""

    level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR
    json_logs: bool = True
    log_to_file: bool = True
    log_file_path: str = "./logs/huracan_engine.log"
    max_log_size_mb: int = 100
    backup_count: int = 10


@dataclass
class PersistenceConfig:
    """Model persistence configuration."""

    enabled: bool = True
    persistence_dir: Path = Path("./model_state")
    backup_frequency_hours: int = 6
    max_backups: int = 10
    auto_save_on_exit: bool = True


@dataclass
class ProductionConfig:
    """
    Complete production configuration.

    This is the master configuration that brings together all subsystems.
    """

    # Environment
    environment: str = "production"  # "development", "staging", "production"

    # Sub-configurations
    phase1: Phase1Config = field(default_factory=Phase1Config)
    phase2: Phase2Config = field(default_factory=Phase2Config)
    trading: TradingConfig = field(default_factory=TradingConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    persistence: PersistenceConfig = field(default_factory=PersistenceConfig)

    # Feature flags
    enable_phase1_features: bool = True
    enable_phase2_features: bool = True
    enable_regime_detection: bool = True
    enable_confidence_scoring: bool = True
    enable_feature_importance: bool = True
    enable_recency_penalties: bool = True
    enable_multi_symbol_coordination: bool = True
    enable_enhanced_risk_management: bool = True
    enable_pattern_detection: bool = True
    enable_portfolio_learning: bool = True

    @classmethod
    def development(cls) -> "ProductionConfig":
        """Create development configuration."""
        config = cls()
        config.environment = "development"
        config.trading.paper_trading_mode = True
        config.trading.total_capital_gbp = 1000.0
        config.logging.level = "DEBUG"
        return config

    @classmethod
    def staging(cls) -> "ProductionConfig":
        """Create staging configuration."""
        config = cls()
        config.environment = "staging"
        config.trading.paper_trading_mode = True
        config.trading.total_capital_gbp = 5000.0
        config.logging.level = "INFO"
        return config

    @classmethod
    def production(cls) -> "ProductionConfig":
        """Create production configuration."""
        config = cls()
        config.environment = "production"
        config.trading.paper_trading_mode = False  # Live trading!
        config.trading.total_capital_gbp = 10000.0
        config.logging.level = "INFO"
        config.logging.log_to_file = True
        return config

    def validate(self) -> tuple[bool, list[str]]:
        """
        Validate configuration.

        Returns:
            (is_valid, errors) tuple
        """
        errors = []

        # Validate capital
        if self.trading.total_capital_gbp <= 0:
            errors.append("Total capital must be positive")

        # Validate position sizing
        if self.phase2.base_position_size_gbp > self.trading.total_capital_gbp * 0.5:
            errors.append("Base position size too large relative to capital")

        # Validate risk parameters
        if self.trading.max_drawdown_pct <= 0 or self.trading.max_drawdown_pct > 50:
            errors.append("Max drawdown must be between 0-50%")

        # Validate portfolio heat
        if self.phase2.max_portfolio_heat > 1.0 or self.phase2.max_portfolio_heat <= 0:
            errors.append("Max portfolio heat must be between 0-1")

        # Validate database
        if not self.database.host:
            errors.append("Database host not configured")

        # Validate persistence
        if self.persistence.enabled and not self.persistence.persistence_dir:
            errors.append("Persistence enabled but no directory configured")

        # Warn about production settings
        if self.environment == "production" and self.trading.paper_trading_mode:
            logger.warning("Production environment but paper trading enabled")

        is_valid = len(errors) == 0

        if is_valid:
            logger.info("configuration_validated", environment=self.environment)
        else:
            logger.error("configuration_validation_failed", errors=errors)

        return (is_valid, errors)

    def get_summary(self) -> dict:
        """Get human-readable configuration summary."""
        return {
            "environment": self.environment,
            "capital_gbp": self.trading.total_capital_gbp,
            "paper_trading": self.trading.paper_trading_mode,
            "max_positions": self.trading.max_positions,
            "max_portfolio_heat": f"{self.phase2.max_portfolio_heat:.0%}",
            "min_confidence": f"{self.phase1.min_confidence_threshold:.0%}",
            "phase1_enabled": self.enable_phase1_features,
            "phase2_enabled": self.enable_phase2_features,
            "persistence_enabled": self.persistence.enabled,
            "database": f"{self.database.host}:{self.database.port}/{self.database.database}",
        }


# Convenience functions
def load_config(environment: str = "development") -> ProductionConfig:
    """
    Load configuration for specified environment.

    Args:
        environment: "development", "staging", or "production"

    Returns:
        ProductionConfig instance
    """
    if environment == "development":
        config = ProductionConfig.development()
    elif environment == "staging":
        config = ProductionConfig.staging()
    elif environment == "production":
        config = ProductionConfig.production()
    else:
        raise ValueError(f"Unknown environment: {environment}")

    # Validate
    is_valid, errors = config.validate()
    if not is_valid:
        raise ValueError(f"Invalid configuration: {errors}")

    logger.info("configuration_loaded", **config.get_summary())

    return config
