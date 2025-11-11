"""
Regime Classifier

Regime gate. Only allow engines in the regimes where they work.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
import structlog

from ..engines.engine_interface import BaseEngine

logger = structlog.get_logger(__name__)


class Regime(Enum):
    """Market regime."""
    TREND = "TREND"
    RANGE = "RANGE"
    PANIC = "PANIC"
    ILLIQUID = "ILLIQUID"


@dataclass
class RegimeClassification:
    """Regime classification result."""
    regime: Regime
    confidence: float  # Confidence score (0.0 to 1.0)
    metadata: Dict[str, Any]  # Additional metadata


class RegimeClassifier:
    """Regime classifier for tagging each bar."""
    
    def __init__(self):
        """Initialize regime classifier."""
        logger.info("regime_classifier_initialized")
    
    def classify(self, candles_df: pd.DataFrame, symbol: str) -> RegimeClassification:
        """Classify market regime for a symbol.
        
        Args:
            candles_df: DataFrame with OHLCV data
            symbol: Trading symbol
            
        Returns:
            Regime classification
        """
        if candles_df.empty or len(candles_df) < 20:
            # Default to RANGE if insufficient data
            return RegimeClassification(
                regime=Regime.RANGE,
                confidence=0.5,
                metadata={"reason": "insufficient_data"},
            )
        
        # Calculate regime indicators
        trend_strength = self._calculate_trend_strength(candles_df)
        volatility = self._calculate_volatility(candles_df)
        volume = self._calculate_volume(candles_df)
        
        # Classify regime
        if volatility > 0.05 and trend_strength > 0.7:
            regime = Regime.PANIC
            confidence = min(volatility * 10, 1.0)
        elif trend_strength > 0.6:
            regime = Regime.TREND
            confidence = trend_strength
        elif volume < 0.5:
            regime = Regime.ILLIQUID
            confidence = 1.0 - volume
        else:
            regime = Regime.RANGE
            confidence = 1.0 - trend_strength
        
        return RegimeClassification(
            regime=regime,
            confidence=confidence,
            metadata={
                "trend_strength": trend_strength,
                "volatility": volatility,
                "volume": volume,
            },
        )
    
    def _calculate_trend_strength(self, df: pd.DataFrame) -> float:
        """Calculate trend strength.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Trend strength (0.0 to 1.0)
        """
        if len(df) < 20:
            return 0.5
        
        # Use moving average slope as trend indicator
        sma_20 = df["close"].rolling(window=20).mean()
        sma_50 = df["close"].rolling(window=50).mean() if len(df) >= 50 else sma_20
        
        # Calculate slope
        slope = (sma_20.iloc[-1] - sma_20.iloc[-10]) / sma_20.iloc[-10] if len(sma_20) >= 10 else 0.0
        
        # Normalize to 0-1
        trend_strength = min(abs(slope) * 100, 1.0)
        
        return trend_strength
    
    def _calculate_volatility(self, df: pd.DataFrame) -> float:
        """Calculate volatility.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Volatility (0.0 to 1.0)
        """
        if len(df) < 20:
            return 0.5
        
        returns = df["close"].pct_change()
        volatility = returns.tail(20).std()
        
        # Normalize to 0-1 (assuming max volatility of 0.1)
        normalized_volatility = min(volatility * 10, 1.0)
        
        return normalized_volatility
    
    def _calculate_volume(self, df: pd.DataFrame) -> float:
        """Calculate volume indicator.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Volume indicator (0.0 to 1.0)
        """
        if len(df) < 20:
            return 0.5
        
        avg_volume = df["volume"].tail(20).mean()
        current_volume = df["volume"].iloc[-1]
        
        # Normalize to 0-1
        volume_ratio = min(current_volume / avg_volume if avg_volume > 0 else 1.0, 2.0) / 2.0
        
        return volume_ratio
    
    def filter_engines_by_regime(
        self,
        engines: List[BaseEngine],
        regime: Regime,
    ) -> List[BaseEngine]:
        """Filter engines by regime support.
        
        Args:
            engines: List of engines
            regime: Market regime
            
        Returns:
            List of engines that support the regime
        """
        return [engine for engine in engines if engine.is_supported_regime(regime.value)]

