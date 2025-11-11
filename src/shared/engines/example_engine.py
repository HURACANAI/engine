"""
Example Engine Implementation

Example implementation of a simple engine to demonstrate the interface.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, Any

import structlog

from .engine_interface import BaseEngine, Direction, EngineInput, EngineOutput

logger = structlog.get_logger(__name__)


class ExampleTrendEngine(BaseEngine):
    """Example trend-following engine."""
    
    def __init__(self):
        """Initialize example trend engine."""
        super().__init__(
            engine_id="example_trend",
            name="Example Trend Engine",
            supported_regimes=["TREND", "RANGE"],
        )
    
    def infer(self, input_data: EngineInput) -> EngineOutput:
        """Run inference on input data.
        
        Args:
            input_data: Engine input data
            
        Returns:
            Engine output with direction, edge, confidence, horizon
        """
        # Simple trend-following logic
        features = input_data.features
        
        # Get trend indicators
        sma_20 = features.get("sma_20", 0.0)
        price_vs_sma_20 = features.get("price_vs_sma_20", 0.0)
        rsi_14 = features.get("rsi_14", 50.0)
        
        # Determine direction
        if price_vs_sma_20 > 2.0 and rsi_14 < 70.0:
            direction = Direction.BUY
            edge_bps = 10.0
            confidence = 0.7
        elif price_vs_sma_20 < -2.0 and rsi_14 > 30.0:
            direction = Direction.SELL
            edge_bps = 10.0
            confidence = 0.7
        else:
            direction = Direction.WAIT
            edge_bps = 0.0
            confidence = 0.3
        
        return EngineOutput(
            direction=direction,
            edge_bps_before_costs=edge_bps,
            confidence_0_1=confidence,
            horizon_minutes=60,
            metadata={
                "sma_20": sma_20,
                "price_vs_sma_20": price_vs_sma_20,
                "rsi_14": rsi_14,
                "regime": input_data.regime,
            },
        )


class ExampleRangeEngine(BaseEngine):
    """Example range-trading engine."""
    
    def __init__(self):
        """Initialize example range engine."""
        super().__init__(
            engine_id="example_range",
            name="Example Range Engine",
            supported_regimes=["RANGE"],
        )
    
    def infer(self, input_data: EngineInput) -> EngineOutput:
        """Run inference on input data.
        
        Args:
            input_data: Engine input data
            
        Returns:
            Engine output with direction, edge, confidence, horizon
        """
        # Simple range-trading logic
        features = input_data.features
        
        # Get range indicators
        rsi_14 = features.get("rsi_14", 50.0)
        volatility_20 = features.get("volatility_20", 0.0)
        
        # Determine direction
        if rsi_14 < 30.0 and volatility_20 < 2.0:
            direction = Direction.BUY
            edge_bps = 8.0
            confidence = 0.6
        elif rsi_14 > 70.0 and volatility_20 < 2.0:
            direction = Direction.SELL
            edge_bps = 8.0
            confidence = 0.6
        else:
            direction = Direction.WAIT
            edge_bps = 0.0
            confidence = 0.2
        
        return EngineOutput(
            direction=direction,
            edge_bps_before_costs=edge_bps,
            confidence_0_1=confidence,
            horizon_minutes=120,
            metadata={
                "rsi_14": rsi_14,
                "volatility_20": volatility_20,
                "regime": input_data.regime,
            },
        )

