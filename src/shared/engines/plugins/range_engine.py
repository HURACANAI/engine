"""
Range Engine Plugin

Engine #2: Trades range-bound markets with adaptive support/resistance detection.
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


class RangeEngine(BaseEnhancedEngine):
    """Range engine with adaptive support/resistance detection."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize range engine.
        
        Args:
            config: Engine configuration
        """
        super().__init__(
            engine_id="range_engine",
            name="Range Engine",
            supported_regimes=["RANGE", "LOW_VOLATILITY"],
            supported_horizons=[TradingHorizon.SCALP, TradingHorizon.SWING],
            default_horizon=TradingHorizon.SCALP,
        )
        
        self.config = config
        self.window = config.get("window", 30)
        self.range_threshold = config.get("range_threshold", 0.02)  # 2% range
        self.support_resistance_tolerance = config.get("support_resistance_tolerance", 0.005)  # 0.5% tolerance
        self.adaptive_detection = config.get("adaptive_detection", True)
        
        logger.info(
            "range_engine_initialized",
            window=self.window,
            range_threshold=self.range_threshold,
        )
    
    def infer(self, input_data: EnhancedEngineInput) -> EnhancedEngineOutput:
        """Run range detection inference.
        
        Args:
            input_data: Enhanced engine input data
            
        Returns:
            Enhanced engine output with range signal
        """
        features = input_data.features
        regime = input_data.regime
        
        # Extract features
        volatility = features.get("volatility", 0.0)
        trend_strength = features.get("trend_strength", 0.0)
        rsi = features.get("rsi", 50.0)
        support_level = features.get("support_level", 0.0)
        resistance_level = features.get("resistance_level", 0.0)
        current_price = features.get("current_price", 0.0)
        distance_to_support = features.get("distance_to_support", 0.0)
        distance_to_resistance = features.get("distance_to_resistance", 0.0)
        
        # Range detection logic
        direction = Direction.WAIT
        edge_bps = 0.0
        confidence = 0.0
        
        # Check if market is in range
        is_in_range = volatility < self.range_threshold and trend_strength < 0.4
        
        if is_in_range and support_level > 0 and resistance_level > 0:
            # Near support = buy
            if distance_to_support < self.support_resistance_tolerance and rsi < 40:
                direction = Direction.BUY
                edge_bps = (self.support_resistance_tolerance - distance_to_support) * 10000.0
                confidence = 0.7 - (distance_to_support / self.support_resistance_tolerance) * 0.3
            
            # Near resistance = sell
            elif distance_to_resistance < self.support_resistance_tolerance and rsi > 60:
                direction = Direction.SELL
                edge_bps = (self.support_resistance_tolerance - distance_to_resistance) * 10000.0
                confidence = 0.7 - (distance_to_resistance / self.support_resistance_tolerance) * 0.3
        
        # Determine horizon based on range characteristics
        horizon_type = TradingHorizon.SCALP
        horizon_minutes = 60  # 1 hour default
        
        if volatility < 0.01 and is_in_range:
            # Very tight range = swing trade
            horizon_type = TradingHorizon.SWING
            horizon_minutes = 12 * 60  # 12 hours
        
        # Stop loss and take profit (tighter for range trades)
        stop_loss_bps = 100.0  # 1% stop loss
        take_profit_bps = 150.0  # 1.5% take profit
        
        if horizon_type == TradingHorizon.SWING:
            stop_loss_bps = 150.0  # 1.5% for swing trades
            take_profit_bps = 200.0  # 2% for swing trades
        
        # Trailing stop (tighter for range trades)
        trailing_stop_bps = 50.0  # 0.5% trailing stop
        
        # Position size multiplier based on confidence
        position_size_multiplier = confidence * 0.8  # Reduce size for range trades
        
        # Max holding hours
        max_holding_hours = horizon_minutes / 60.0
        
        # Funding cost estimate
        funding_cost_estimate_bps = 0.0
        if horizon_type == TradingHorizon.SWING:
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
                "volatility": volatility,
                "trend_strength": trend_strength,
                "rsi": rsi,
                "support_level": support_level,
                "resistance_level": resistance_level,
                "distance_to_support": distance_to_support,
                "distance_to_resistance": distance_to_resistance,
                "is_in_range": is_in_range,
                "regime": regime,
            },
        )

