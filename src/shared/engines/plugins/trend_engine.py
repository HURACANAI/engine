"""
Trend Engine Plugin

Engine #1: Detects and trades trends with momentum breakout filters.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict

import structlog

from ..enhanced_engine_interface import (
    BaseEnhancedEngine,
    EnhancedEngineInput,
    EnhancedEngineOutput,
    TradingHorizon,
    Direction,
)

logger = structlog.get_logger(__name__)


class TrendEngine(BaseEnhancedEngine):
    """Trend engine with momentum breakout filters."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize trend engine.
        
        Args:
            config: Engine configuration
        """
        super().__init__(
            engine_id="trend_engine",
            name="Trend Engine",
            supported_regimes=["TREND", "RANGE"],
            supported_horizons=[TradingHorizon.SWING, TradingHorizon.POSITION],
            default_horizon=TradingHorizon.SWING,
        )
        
        self.config = config
        self.window = config.get("window", 20)
        self.momentum_threshold = config.get("momentum_threshold", 0.02)  # 2% momentum
        self.volatility_threshold = config.get("volatility_threshold", 0.05)  # 5% volatility
        self.trailing_entry = config.get("trailing_entry", True)
        
        logger.info(
            "trend_engine_initialized",
            window=self.window,
            momentum_threshold=self.momentum_threshold,
        )
    
    def infer(self, input_data: EnhancedEngineInput) -> EnhancedEngineOutput:
        """Run trend detection inference.
        
        Args:
            input_data: Enhanced engine input data
            
        Returns:
            Enhanced engine output with trend signal
        """
        features = input_data.features
        regime = input_data.regime
        
        # Extract features
        momentum = features.get("momentum", 0.0)
        volatility = features.get("volatility", 0.0)
        trend_strength = features.get("trend_strength", 0.0)
        rsi = features.get("rsi", 50.0)
        ema_slope = features.get("ema_slope", 0.0)
        
        # Trend detection logic
        direction = Direction.WAIT
        edge_bps = 0.0
        confidence = 0.0
        
        # Check if trend is strong enough
        if trend_strength > 0.6 and abs(momentum) > self.momentum_threshold:
            # Bullish trend
            if momentum > 0 and ema_slope > 0 and rsi < 70:
                direction = Direction.BUY
                edge_bps = momentum * 10000.0  # Convert to basis points
                confidence = min(trend_strength, 0.9)
            
            # Bearish trend
            elif momentum < 0 and ema_slope < 0 and rsi > 30:
                direction = Direction.SELL
                edge_bps = abs(momentum) * 10000.0
                confidence = min(trend_strength, 0.9)
        
        # Volatility breakout trigger
        if volatility > self.volatility_threshold and trend_strength > 0.7:
            # Strong trend with high volatility = breakout
            if direction == Direction.BUY:
                edge_bps *= 1.2  # Boost edge for breakout
                confidence = min(confidence * 1.1, 1.0)
            elif direction == Direction.SELL:
                edge_bps *= 1.2
                confidence = min(confidence * 1.1, 1.0)
        
        # Determine horizon based on trend strength
        horizon_type = TradingHorizon.SWING
        horizon_minutes = 24 * 60  # 1 day default
        
        if trend_strength > 0.8:
            # Very strong trend = position trade
            horizon_type = TradingHorizon.POSITION
            horizon_minutes = 7 * 24 * 60  # 1 week
        
        # Stop loss and take profit
        stop_loss_bps = 200.0  # 2% stop loss
        take_profit_bps = 400.0  # 4% take profit
        
        if horizon_type == TradingHorizon.POSITION:
            stop_loss_bps = 300.0  # 3% for position trades
            take_profit_bps = 600.0  # 6% for position trades
        
        # Trailing stop
        trailing_stop_bps = 100.0  # 1% trailing stop
        
        # Position size multiplier based on confidence
        position_size_multiplier = confidence
        
        # Max holding hours
        max_holding_hours = horizon_minutes / 60.0
        
        # Funding cost estimate (for swing/position trades)
        funding_cost_estimate_bps = 0.0
        if horizon_type in [TradingHorizon.SWING, TradingHorizon.POSITION]:
            holding_hours = horizon_minutes / 60.0
            funding_cost_estimate_bps = (holding_hours / 8.0) * input_data.costs.get("funding_bps", 1.0)
        
        return EnhancedEngineOutput(
            direction=direction,
            edge_bps_before_costs=edge_bps,
            confidence_0_1=confidence,
            horizon_minutes=horizon_minutes,
            horizon_type=horizon_type,
            stop_loss_bps=stop_loss_bps,
            take_profit_bps=take_profit_bps,
            trailing_stop_bps=trailing_stop_bps,
            position_size_multiplier=position_size_multiplier,
            max_holding_hours=max_holding_hours,
            funding_cost_estimate_bps=funding_cost_estimate_bps,
            metadata={
                "trend_strength": trend_strength,
                "momentum": momentum,
                "volatility": volatility,
                "rsi": rsi,
                "ema_slope": ema_slope,
                "regime": regime,
            },
        )

