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

    # ===== PHASE 1 ENGINE ENHANCEMENTS (NEW!) =====

    # Multi-Timeframe Analysis
    enable_multi_timeframe: bool = True
    timeframes: list = field(default_factory=lambda: ['5m', '15m', '1h'])
    min_confluence_score: float = 0.67  # Require 2/3 timeframes to agree
    multi_tf_trend_threshold: float = 0.3
    multi_tf_momentum_threshold: float = 0.2

    # Volume Validation
    enable_volume_validation: bool = True
    volume_breakout_min_ratio: float = 1.5  # Breakouts need 1.5x avg volume
    volume_trend_min_ratio: float = 0.8  # Trends need 0.8x avg volume
    volume_range_max_ratio: float = 1.3  # Range trades max 1.3x avg volume

    # Pattern Memory Check
    enable_pattern_memory: bool = True
    pattern_min_samples: int = 5  # Need 5+ historical samples
    pattern_min_similarity: float = 0.70  # Fairly strict similarity matching


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

    # ===== PHASE 2 SMART EXITS (NEW!) =====

    # Adaptive Trailing Stops
    enable_adaptive_trailing: bool = True
    trail_stage_1_profit_bps: float = 50.0  # First stage at +50 bps
    trail_stage_1_distance_bps: float = 25.0  # Trail 25 bps
    trail_stage_2_profit_bps: float = 100.0  # Second stage at +100 bps
    trail_stage_2_distance_bps: float = 50.0  # Trail 50 bps
    trail_stage_3_profit_bps: float = 200.0  # Third stage at +200 bps
    trail_stage_3_distance_bps: float = 100.0  # Trail 100 bps
    trail_stage_4_profit_bps: float = 400.0  # Fourth stage at +400 bps
    trail_stage_4_distance_bps: float = 200.0  # Trail 200 bps
    trail_volatility_multiplier_high: float = 1.5  # Widen in high vol
    trail_volatility_multiplier_low: float = 0.7  # Tighten in low vol
    trail_volatility_high_threshold: float = 200.0  # High vol threshold (bps)
    trail_volatility_low_threshold: float = 80.0  # Low vol threshold (bps)
    trail_momentum_tighten_threshold: float = 0.3  # Tighten if momentum < 0.3
    trail_momentum_tighten_factor: float = 0.8  # Tighten by 20%

    # Exit Signal Detection
    enable_exit_signals: bool = True
    exit_momentum_reversal_threshold: float = -0.2  # Momentum < -0.2 = reversal
    exit_volume_climax_threshold: float = 2.5  # Volume > 2.5x avg = climax
    exit_divergence_threshold: float = 0.15  # Divergence strength threshold
    exit_overbought_rsi: float = 75.0  # RSI > 75 = overbought
    exit_oversold_rsi: float = 25.0  # RSI < 25 = oversold
    exit_profit_target_min_bps: float = 100.0  # Min profit for P3 exits

    # Regime Exit Management
    enable_regime_exits: bool = True
    regime_exit_profit_threshold_bps: float = 50.0  # Min profit for protective exits
    regime_panic_stop_tighten_bps: float = 50.0  # Tighten stops by 50 bps in panic
    regime_range_stop_tighten_bps: float = 30.0  # Tighten stops by 30 bps in range
    regime_trend_stop_relax_bps: float = 20.0  # Relax stops by 20 bps in favorable trend


@dataclass
class Phase3Config:
    """Phase 3 feature configuration - Engine Intelligence."""

    # ===== PHASE 3 ENGINE INTELLIGENCE =====

    # Engine Consensus System
    enable_engine_consensus: bool = True
    consensus_min_participating_engines: int = 3  # Need 3+ engines with opinions
    consensus_unanimous_boost: float = 0.10  # +10% confidence for unanimous
    consensus_strong_boost: float = 0.05  # +5% confidence for 75%+ agreement
    consensus_moderate_penalty: float = 0.0  # No change for 60-75% agreement
    consensus_weak_penalty: float = -0.05  # -5% confidence for 50-60% agreement
    consensus_divided_penalty: float = -0.15  # -15% confidence for <50% agreement
    consensus_min_confidence_after_adjustment: float = 0.55  # Min after adjustment

    # Regime-specific consensus requirements
    consensus_trend_regime_min_agreement: float = 0.60  # 60% in TREND
    consensus_range_regime_min_agreement: float = 0.60  # 60% in RANGE
    consensus_panic_regime_min_agreement: float = 0.75  # 75% in PANIC (stricter)

    # Confidence Calibration System
    enable_confidence_calibration: bool = True
    calibration_target_win_rate: float = 0.60  # Target 60% win rate
    calibration_target_profit_factor: float = 2.0  # Target 2.0 profit factor
    calibration_overconfident_threshold: float = 0.50  # <50% win rate = overconfident
    calibration_underconfident_threshold: float = 0.75  # >75% win rate = underconfident
    calibration_max_adjustment: float = 0.10  # Max ±10% adjustment per calibration
    calibration_min_trades: int = 30  # Need 30 trades for calibration
    calibration_ema_alpha: float = 0.3  # EMA smoothing for adjustments
    calibration_lookback_trades: int = 50  # Analyze last 50 trades

    # Regime-specific calibration
    calibration_per_regime: bool = True  # Separate calibration per regime


@dataclass
class Phase4Config:
    """Phase 4 feature configuration - Advanced Market Intelligence."""

    # ===== PHASE 4 WAVE 1: MARKET CONTEXT INTELLIGENCE =====

    # Cross-Asset Correlation Analyzer
    enable_correlation_analyzer: bool = True
    correlation_lookback_periods: int = 100  # Historical periods for correlation
    correlation_rolling_window: int = 20  # Recent periods for rolling correlation
    correlation_high_threshold: float = 0.70  # High correlation threshold
    correlation_very_high_threshold: float = 0.90  # Very high correlation
    correlation_min_periods: int = 30  # Minimum data for valid correlation
    correlation_systemic_threshold: float = 0.80  # Market-wide event threshold
    correlation_diversification_min: float = 0.40  # Min diversification ratio

    # Win/Loss Pattern Analyzer
    enable_pattern_analyzer: bool = True
    pattern_failure_threshold: float = 0.35  # Below 35% = failure pattern
    pattern_success_threshold: float = 0.70  # Above 70% = success pattern
    pattern_min_size: int = 10  # Minimum trades per pattern
    pattern_min_confidence: float = 0.65  # Minimum confidence to report
    pattern_max_trades_stored: int = 5000  # Maximum trade history

    # Take-Profit Ladder
    enable_tp_ladder: bool = True
    tp_ladder_style: str = "default"  # 'default' or 'aggressive'

    # Default ladder settings
    tp_level_1_target_bps: float = 100.0  # TP1 at +100 bps
    tp_level_1_exit_pct: float = 0.30  # Exit 30%
    tp_level_2_target_bps: float = 200.0  # TP2 at +200 bps
    tp_level_2_exit_pct: float = 0.40  # Exit 40%
    tp_level_3_target_bps: float = 400.0  # TP3 at +400 bps
    tp_level_3_exit_pct: float = 0.20  # Exit 20%
    tp_final_trail_bps: float = 200.0  # Trail remaining 10%
    tp_final_trail_multiplier: float = 1.5  # 300 bps trail (200 * 1.5)

    # Strategy Performance Tracker
    enable_strategy_tracker: bool = True
    strategy_min_win_rate: float = 0.50  # Minimum acceptable win rate
    strategy_min_profit_factor: float = 1.5  # Minimum profit factor
    strategy_min_trades_to_evaluate: int = 20  # Min trades before evaluation
    strategy_recent_window: int = 20  # Recent performance window
    strategy_auto_disable: bool = True  # Auto-disable underperforming strategies

    # ===== PHASE 4 WAVE 2: ADVANCED LEARNING =====

    # Adaptive Position Sizing 2.0
    enable_adaptive_sizing: bool = True
    adaptive_sizing_base_usd: float = 1000.0  # Base position size
    adaptive_sizing_min_multiplier: float = 0.25  # Min 0.25x base
    adaptive_sizing_max_multiplier: float = 2.50  # Max 2.5x base
    adaptive_sizing_confidence_weight: float = 0.30
    adaptive_sizing_consensus_weight: float = 0.25
    adaptive_sizing_regime_weight: float = 0.20
    adaptive_sizing_risk_weight: float = 0.15
    adaptive_sizing_pattern_weight: float = 0.10

    # Liquidity Depth Analyzer
    enable_liquidity_analyzer: bool = True
    liquidity_min_depth_ratio: float = 3.0  # Need 3x position size in depth
    liquidity_slippage_tolerance_bps: float = 15.0  # Max 15 bps slippage
    liquidity_spread_warning_multiplier: float = 2.0  # Warn if spread > 2x normal
    liquidity_min_score: float = 0.50  # Min acceptable liquidity score

    # Regime Transition Anticipator
    enable_regime_anticipator: bool = True
    regime_vol_spike_threshold: float = 2.0  # 2x vol = spike
    regime_vol_surge_threshold: float = 2.5  # 2.5x volume = surge
    regime_adx_collapse_threshold: float = 0.6  # 40% ADX drop = collapse
    regime_transition_confidence_min: float = 0.60
    regime_transition_action_min_confidence: float = 0.70

    # Ensemble Exit Strategy
    enable_ensemble_exits: bool = True
    ensemble_p1_danger_votes: int = 3  # P1 DANGER = 3 votes
    ensemble_p2_warning_votes: int = 2  # P2 WARNING = 2 votes
    ensemble_p3_profit_votes: int = 1  # P3 PROFIT = 1 vote
    ensemble_scale_25_threshold: int = 3  # 3-4 votes = 25% exit
    ensemble_scale_50_threshold: int = 5  # 5-6 votes = 50% exit
    ensemble_scale_75_threshold: int = 7  # 7-8 votes = 75% exit
    ensemble_exit_all_threshold: int = 9  # 9+ votes = 100% exit

    # ===== PHASE 4 WAVE 3: POLISH & OPTIMIZATION =====

    # Smart Order Executor
    enable_smart_execution: bool = True
    smart_exec_maker_fee_bps: float = 2.0
    smart_exec_taker_fee_bps: float = 5.0
    smart_exec_large_size_threshold_usd: float = 10000.0  # Split orders > $10k
    smart_exec_twap_slice_count: int = 10
    smart_exec_twap_window_seconds: int = 300  # 5 minutes
    smart_exec_limit_order_offset_bps: float = 1.0  # 1 bps better than mid

    # Multi-Horizon Predictor
    enable_multi_horizon: bool = True
    multi_horizon_weight_5m: float = 0.10  # 5m weight
    multi_horizon_weight_15m: float = 0.20  # 15m weight
    multi_horizon_weight_1h: float = 0.30  # 1h weight
    multi_horizon_weight_4h: float = 0.40  # 4h weight (most important)
    multi_horizon_alignment_excellent: float = 0.85  # Excellent alignment threshold
    multi_horizon_alignment_good: float = 0.70  # Good alignment threshold
    multi_horizon_alignment_moderate: float = 0.55  # Moderate alignment threshold
    multi_horizon_alignment_poor: float = 0.40  # Poor alignment threshold

    # Macro Event Detector
    enable_macro_detector: bool = True
    macro_normal_volatility_bps: float = 100.0
    macro_normal_spread_bps: float = 5.0
    macro_vol_spike_threshold: float = 3.0  # 3x vol = spike
    macro_volume_surge_threshold: float = 3.0  # 3x volume = surge
    macro_spread_widening_threshold: float = 2.0  # 2x spread = widening
    macro_rapid_move_threshold_bps: float = 500.0  # 500 bps in 5m = rapid
    macro_correlation_breakdown_threshold: float = 0.50
    macro_liquidation_threshold: float = 0.60
    macro_history_window_minutes: int = 60

    # Hyperparameter Auto-Tuner
    enable_hyperparameter_tuning: bool = True
    tuning_degradation_threshold: float = 0.15  # 15% drop triggers tuning
    tuning_min_trades_baseline: int = 50  # Need 50 trades for baseline
    tuning_min_trades_per_test: int = 30  # Test each param for 30 trades
    tuning_improvement_threshold: float = 0.05  # Need 5% improvement to switch
    tuning_performance_window: int = 100  # Track last 100 trades


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
    phase3: Phase3Config = field(default_factory=Phase3Config)
    phase4: Phase4Config = field(default_factory=Phase4Config)
    trading: TradingConfig = field(default_factory=TradingConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    persistence: PersistenceConfig = field(default_factory=PersistenceConfig)

    # Feature flags
    enable_phase1_features: bool = True
    enable_phase2_features: bool = True
    enable_phase3_features: bool = True
    enable_phase4_features: bool = True
    enable_regime_detection: bool = True
    enable_confidence_scoring: bool = True
    enable_feature_importance: bool = True
    enable_recency_penalties: bool = True
    enable_multi_symbol_coordination: bool = True
    enable_enhanced_risk_management: bool = True
    enable_pattern_detection: bool = True
    enable_portfolio_learning: bool = True
    enable_engine_consensus: bool = True
    enable_confidence_calibration: bool = True

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
