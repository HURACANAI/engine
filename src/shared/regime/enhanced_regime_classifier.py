"""
Enhanced Regime Classifier

Regime gating for swing trading: only allow engines in regimes where they work.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
import structlog

from ..engines.enhanced_engine_interface import BaseEnhancedEngine, TradingHorizon

logger = structlog.get_logger(__name__)


class Regime(Enum):
    """Market regime."""
    TREND = "TREND"
    RANGE = "RANGE"
    PANIC = "PANIC"
    ILLIQUID = "ILLIQUID"
    RECOVERY = "RECOVERY"  # New regime for recovery after panic
    HIGH_VOLATILITY = "HIGH_VOLATILITY"  # High volatility but not panic
    LOW_VOLATILITY = "LOW_VOLATILITY"  # Low volatility, stable


@dataclass
class RegimeClassification:
    """Enhanced regime classification result."""
    regime: Regime
    confidence: float  # Confidence score (0.0 to 1.0)
    metadata: Dict[str, Any]  # Additional metadata
    # Swing trading specific
    allows_swing_trading: bool = True  # Whether swing trading is allowed
    allows_position_trading: bool = True  # Whether position trading is allowed
    allows_core_holding: bool = True  # Whether core holding is allowed
    risk_multiplier: float = 1.0  # Risk multiplier for this regime
    
    def is_safe_for_horizon(self, horizon: TradingHorizon) -> bool:
        """Check if regime is safe for a given horizon.
        
        Args:
            horizon: Trading horizon
            
        Returns:
            True if safe for the horizon
        """
        if horizon == TradingHorizon.SCALP:
            return True  # Scalps can trade in most regimes
        elif horizon == TradingHorizon.SWING:
            return self.allows_swing_trading
        elif horizon == TradingHorizon.POSITION:
            return self.allows_position_trading
        elif horizon == TradingHorizon.CORE:
            return self.allows_core_holding
        else:
            return False


@dataclass
class RegimeGatingConfig:
    """Configuration for regime gating."""
    # Regime-specific gating rules
    panic_allows_swing: bool = False  # Panic regime blocks swing trading
    panic_allows_position: bool = False  # Panic regime blocks position trading
    illiquid_allows_swing: bool = False  # Illiquid regime blocks swing trading
    illiquid_allows_position: bool = False  # Illiquid regime blocks position trading
    # Risk multipliers
    panic_risk_multiplier: float = 2.0  # 2x risk in panic
    illiquid_risk_multiplier: float = 1.5  # 1.5x risk in illiquid
    high_volatility_risk_multiplier: float = 1.3  # 1.3x risk in high volatility
    # Thresholds
    volatility_threshold_high: float = 0.05  # High volatility threshold
    volatility_threshold_low: float = 0.01  # Low volatility threshold
    volume_threshold_low: float = 0.5  # Low volume threshold
    trend_strength_threshold: float = 0.6  # Trend strength threshold


class EnhancedRegimeClassifier:
    """Enhanced regime classifier with swing trading gating."""
    
    def __init__(self, config: Optional[RegimeGatingConfig] = None):
        """Initialize enhanced regime classifier.
        
        Args:
            config: Regime gating configuration (uses defaults if None)
        """
        self.config = config or RegimeGatingConfig()
        logger.info("enhanced_regime_classifier_initialized")
    
    def classify(self, candles_df: pd.DataFrame, symbol: str) -> RegimeClassification:
        """Classify market regime for a symbol.
        
        Args:
            candles_df: DataFrame with OHLCV data
            symbol: Trading symbol
            
        Returns:
            Enhanced regime classification
        """
        if candles_df.empty or len(candles_df) < 20:
            # Default to RANGE if insufficient data
            return RegimeClassification(
                regime=Regime.RANGE,
                confidence=0.5,
                metadata={"reason": "insufficient_data"},
                allows_swing_trading=True,
                allows_position_trading=True,
                allows_core_holding=True,
                risk_multiplier=1.0,
            )
        
        # Calculate regime indicators
        trend_strength = self._calculate_trend_strength(candles_df)
        volatility = self._calculate_volatility(candles_df)
        volume = self._calculate_volume(candles_df)
        
        # Classify regime
        regime = self._classify_regime(trend_strength, volatility, volume)
        confidence = self._calculate_confidence(trend_strength, volatility, volume, regime)
        
        # Determine swing trading permissions
        allows_swing_trading = self._allows_swing_trading(regime)
        allows_position_trading = self._allows_position_trading(regime)
        allows_core_holding = self._allows_core_holding(regime)
        risk_multiplier = self._get_risk_multiplier(regime)
        
        return RegimeClassification(
            regime=regime,
            confidence=confidence,
            metadata={
                "trend_strength": trend_strength,
                "volatility": volatility,
                "volume": volume,
            },
            allows_swing_trading=allows_swing_trading,
            allows_position_trading=allows_position_trading,
            allows_core_holding=allows_core_holding,
            risk_multiplier=risk_multiplier,
        )
    
    def _classify_regime(
        self,
        trend_strength: float,
        volatility: float,
        volume: float,
    ) -> Regime:
        """Classify regime from indicators.
        
        Args:
            trend_strength: Trend strength (0.0 to 1.0)
            volatility: Volatility (0.0 to 1.0)
            volume: Volume indicator (0.0 to 1.0)
            
        Returns:
            Market regime
        """
        # Panic: High volatility + trend
        if volatility > self.config.volatility_threshold_high and trend_strength > 0.7:
            return Regime.PANIC
        
        # Illiquid: Low volume
        if volume < self.config.volume_threshold_low:
            return Regime.ILLIQUID
        
        # High volatility: High volatility but not panic
        if volatility > self.config.volatility_threshold_high:
            return Regime.HIGH_VOLATILITY
        
        # Low volatility: Low volatility
        if volatility < self.config.volatility_threshold_low:
            return Regime.LOW_VOLATILITY
        
        # Trend: Strong trend
        if trend_strength > self.config.trend_strength_threshold:
            return Regime.TREND
        
        # Range: Default
        return Regime.RANGE
    
    def _calculate_confidence(
        self,
        trend_strength: float,
        volatility: float,
        volume: float,
        regime: Regime,
    ) -> float:
        """Calculate confidence in regime classification.
        
        Args:
            trend_strength: Trend strength (0.0 to 1.0)
            volatility: Volatility (0.0 to 1.0)
            volume: Volume indicator (0.0 to 1.0)
            regime: Classified regime
            
        Returns:
            Confidence score (0.0 to 1.0)
        """
        # Higher confidence when indicators are extreme
        if regime == Regime.PANIC:
            return min(volatility * 10, 1.0)
        elif regime == Regime.ILLIQUID:
            return 1.0 - volume
        elif regime == Regime.TREND:
            return trend_strength
        elif regime == Regime.HIGH_VOLATILITY:
            return min(volatility * 8, 1.0)
        elif regime == Regime.LOW_VOLATILITY:
            return 1.0 - min(volatility * 10, 1.0)
        else:  # RANGE
            return 1.0 - trend_strength
    
    def _allows_swing_trading(self, regime: Regime) -> bool:
        """Check if regime allows swing trading.
        
        Args:
            regime: Market regime
            
        Returns:
            True if swing trading is allowed
        """
        if regime == Regime.PANIC:
            return self.config.panic_allows_swing
        elif regime == Regime.ILLIQUID:
            return self.config.illiquid_allows_swing
        else:
            return True
    
    def _allows_position_trading(self, regime: Regime) -> bool:
        """Check if regime allows position trading.
        
        Args:
            regime: Market regime
            
        Returns:
            True if position trading is allowed
        """
        if regime == Regime.PANIC:
            return self.config.panic_allows_position
        elif regime == Regime.ILLIQUID:
            return self.config.illiquid_allows_position
        else:
            return True
    
    def _allows_core_holding(self, regime: Regime) -> bool:
        """Check if regime allows core holding.
        
        Args:
            regime: Market regime
            
        Returns:
            True if core holding is allowed (usually always True)
        """
        # Core holding is usually always allowed (DCA strategy)
        return True
    
    def _get_risk_multiplier(self, regime: Regime) -> float:
        """Get risk multiplier for regime.
        
        Args:
            regime: Market regime
            
        Returns:
            Risk multiplier
        """
        multipliers = {
            Regime.PANIC: self.config.panic_risk_multiplier,
            Regime.ILLIQUID: self.config.illiquid_risk_multiplier,
            Regime.HIGH_VOLATILITY: self.config.high_volatility_risk_multiplier,
            Regime.TREND: 1.0,
            Regime.RANGE: 1.0,
            Regime.LOW_VOLATILITY: 0.8,  # Lower risk in low volatility
            Regime.RECOVERY: 1.2,  # Slightly higher risk in recovery
        }
        return multipliers.get(regime, 1.0)
    
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
        engines: List[BaseEnhancedEngine],
        regime: Regime,
        horizon: Optional[TradingHorizon] = None,
    ) -> List[BaseEnhancedEngine]:
        """Filter engines by regime support and horizon.
        
        Args:
            engines: List of engines
            regime: Market regime
            horizon: Trading horizon (optional)
            
        Returns:
            List of engines that support the regime and horizon
        """
        filtered = [engine for engine in engines if engine.is_supported_regime(regime.value)]
        
        if horizon:
            filtered = [engine for engine in filtered if engine.is_supported_horizon(horizon)]
        
        return filtered
    
    def should_trade_horizon(
        self,
        regime: Regime,
        horizon: TradingHorizon,
    ) -> bool:
        """Check if a horizon should trade in a given regime.
        
        Args:
            regime: Market regime
            horizon: Trading horizon
            
        Returns:
            True if horizon should trade
        """
        classification = RegimeClassification(
            regime=regime,
            confidence=1.0,
            metadata={},
            allows_swing_trading=self._allows_swing_trading(regime),
            allows_position_trading=self._allows_position_trading(regime),
            allows_core_holding=self._allows_core_holding(regime),
            risk_multiplier=self._get_risk_multiplier(regime),
        )
        
        return classification.is_safe_for_horizon(horizon)

