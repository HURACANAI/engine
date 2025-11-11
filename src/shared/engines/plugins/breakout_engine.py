"""
Breakout Engine Plugin

Engine #3: Detects and trades breakouts with volume-flow confirmation.
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


class BreakoutEngine(BaseEnhancedEngine):
    """Breakout engine with volume-flow confirmation."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize breakout engine.
        
        Args:
            config: Engine configuration
        """
        super().__init__(
            engine_id="breakout_engine",
            name="Breakout Engine",
            supported_regimes=["TREND", "RANGE"],
            supported_horizons=[TradingHorizon.SWING, TradingHorizon.POSITION],
            default_horizon=TradingHorizon.SWING,
        )
        
        self.config = config
        self.window = config.get("window", 20)
        self.breakout_threshold = config.get("breakout_threshold", 0.02)  # 2% breakout
        self.volume_multiplier = config.get("volume_multiplier", 1.5)  # 1.5x volume
        self.volatility_contraction_threshold = config.get("volatility_contraction_threshold", 0.01)  # 1% contraction
        self.depth_breakout = config.get("depth_breakout", True)
        
        logger.info(
            "breakout_engine_initialized",
            window=self.window,
            breakout_threshold=self.breakout_threshold,
        )
    
    def infer(self, input_data: EnhancedEngineInput) -> EnhancedEngineOutput:
        """Run breakout detection inference.
        
        Args:
            input_data: Enhanced engine input data
            
        Returns:
            Enhanced engine output with breakout signal
        """
        features = input_data.features
        regime = input_data.regime
        
        # Extract features
        volatility = features.get("volatility", 0.0)
        volume_ratio = features.get("volume_ratio", 1.0)
        price_change = features.get("price_change", 0.0)
        resistance_level = features.get("resistance_level", 0.0)
        support_level = features.get("support_level", 0.0)
        current_price = features.get("current_price", 0.0)
        order_book_imbalance = features.get("order_book_imbalance", 0.0)
        depth_breakout = features.get("depth_breakout", False)
        volatility_contraction = features.get("volatility_contraction", 0.0)
        
        # Breakout detection logic
        direction = Direction.WAIT
        edge_bps = 0.0
        confidence = 0.0
        
        # Check for volatility contraction → expansion
        is_contraction = volatility_contraction < self.volatility_contraction_threshold
        
        # Check for volume confirmation
        is_volume_confirmed = volume_ratio > self.volume_multiplier
        
        # Check for depth breakout (if enabled)
        is_depth_breakout = depth_breakout if self.depth_breakout else True
        
        # Bullish breakout
        if (price_change > self.breakout_threshold and
            is_volume_confirmed and
            is_depth_breakout and
            (is_contraction or volatility > 0.03)):
            
            # Breakout above resistance
            if resistance_level > 0 and current_price > resistance_level:
                direction = Direction.BUY
                edge_bps = price_change * 10000.0  # Convert to basis points
                confidence = min(0.7 + (volume_ratio - 1.0) * 0.2, 0.95)
                
                # Boost confidence with order book imbalance
                if order_book_imbalance > 0.1:
                    confidence = min(confidence * 1.1, 1.0)
                    edge_bps *= 1.15
        
        # Bearish breakout
        elif (price_change < -self.breakout_threshold and
              is_volume_confirmed and
              is_depth_breakout and
              (is_contraction or volatility > 0.03)):
            
            # Breakout below support
            if support_level > 0 and current_price < support_level:
                direction = Direction.SELL
                edge_bps = abs(price_change) * 10000.0
                confidence = min(0.7 + (volume_ratio - 1.0) * 0.2, 0.95)
                
                # Boost confidence with order book imbalance
                if order_book_imbalance < -0.1:
                    confidence = min(confidence * 1.1, 1.0)
                    edge_bps *= 1.15
        
        # Determine horizon based on breakout strength
        horizon_type = TradingHorizon.SWING
        horizon_minutes = 24 * 60  # 1 day default
        
        if confidence > 0.85 and volume_ratio > 2.0:
            # Very strong breakout = position trade
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
        
        # Funding cost estimate
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
                "volatility": volatility,
                "volume_ratio": volume_ratio,
                "price_change": price_change,
                "resistance_level": resistance_level,
                "support_level": support_level,
                "order_book_imbalance": order_book_imbalance,
                "depth_breakout": depth_breakout,
                "volatility_contraction": volatility_contraction,
                "is_contraction": is_contraction,
                "is_volume_confirmed": is_volume_confirmed,
                "regime": regime,
            },
        )

