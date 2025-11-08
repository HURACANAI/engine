"""Adaptive regime detection for market conditions."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import structlog  # type: ignore[reportMissingImports]

logger = structlog.get_logger(__name__)


class RegimeDetector:
    """
    Detects market regimes:
    - Volatility: low, normal, high, extreme
    - Market: trending, ranging, volatile
    
    Uses:
    - Rolling volatility (24h std)
    - ATR ratio
    - RSI slope
    - Liquidation volume
    """

    def __init__(
        self,
        volatility_window: int = 24,  # 24 hours
        atr_period: int = 14,
        rsi_period: int = 14,
    ) -> None:
        """
        Initialize regime detector.
        
        Args:
            volatility_window: Window for volatility calculation (hours)
            atr_period: ATR calculation period
            rsi_period: RSI calculation period
        """
        self.volatility_window = volatility_window
        self.atr_period = atr_period
        self.rsi_period = rsi_period
        
        logger.info(
            "regime_detector_initialized",
            volatility_window=volatility_window,
            atr_period=atr_period,
            rsi_period=rsi_period,
        )

    def calculate_volatility(self, data: pd.DataFrame, window: Optional[int] = None) -> pd.Series:
        """
        Calculate rolling volatility.
        
        Args:
            data: Price data DataFrame
            window: Rolling window (default: self.volatility_window)
            
        Returns:
            Volatility series
        """
        if window is None:
            window = self.volatility_window
        
        if 'close' not in data.columns:
            raise ValueError("Data must have 'close' column")
        
        # Calculate returns
        returns = data['close'].pct_change()
        
        # Calculate rolling volatility (std)
        volatility = returns.rolling(window=window).std()
        
        return volatility

    def calculate_atr(self, data: pd.DataFrame, period: Optional[int] = None) -> pd.Series:
        """
        Calculate Average True Range (ATR).
        
        Args:
            data: OHLC data DataFrame
            period: ATR period (default: self.atr_period)
            
        Returns:
            ATR series
        """
        if period is None:
            period = self.atr_period
        
        required_columns = ['high', 'low', 'close']
        if not all(col in data.columns for col in required_columns):
            raise ValueError(f"Data must have columns: {required_columns}")
        
        # Calculate True Range
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        
        # Calculate ATR (SMA of True Range)
        atr = true_range.rolling(window=period).mean()
        
        return atr

    def calculate_rsi(self, data: pd.DataFrame, period: Optional[int] = None) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI).
        
        Args:
            data: Price data DataFrame
            period: RSI period (default: self.rsi_period)
            
        Returns:
            RSI series
        """
        if period is None:
            period = self.rsi_period
        
        if 'close' not in data.columns:
            raise ValueError("Data must have 'close' column")
        
        # Calculate price changes
        delta = data['close'].diff()
        
        # Separate gains and losses
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        
        # Calculate average gains and losses
        avg_gains = gains.rolling(window=period).mean()
        avg_losses = losses.rolling(window=period).mean()
        
        # Calculate RS and RSI
        rs = avg_gains / (avg_losses + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi

    def classify_volatility(
        self,
        volatility: pd.Series,
        percentiles: Optional[Dict[str, float]] = None,
    ) -> pd.Series:
        """
        Classify volatility regime.
        
        Args:
            volatility: Volatility series
            percentiles: Percentile thresholds (default: 25th, 50th, 75th)
            
        Returns:
            Volatility regime series ('low', 'normal', 'high', 'extreme')
        """
        if percentiles is None:
            percentiles = {
                "low": 25.0,
                "normal": 50.0,
                "high": 75.0,
            }
        
        # Calculate percentiles
        p25 = np.percentile(volatility.dropna(), percentiles["low"])
        p50 = np.percentile(volatility.dropna(), percentiles["normal"])
        p75 = np.percentile(volatility.dropna(), percentiles["high"])
        
        # Classify
        def classify_vol(value):
            if pd.isna(value):
                return "unknown"
            elif value < p25:
                return "low"
            elif value < p50:
                return "normal"
            elif value < p75:
                return "high"
            else:
                return "extreme"
        
        regime = volatility.apply(classify_vol)
        
        return regime

    def classify_market(
        self,
        data: pd.DataFrame,
        rsi: Optional[pd.Series] = None,
        atr: Optional[pd.Series] = None,
    ) -> pd.Series:
        """
        Classify market regime (trending, ranging, volatile).
        
        Args:
            data: Price data DataFrame
            rsi: Optional RSI series
            atr: Optional ATR series
            
        Returns:
            Market regime series
        """
        if 'close' not in data.columns:
            raise ValueError("Data must have 'close' column")
        
        # Calculate RSI if not provided
        if rsi is None:
            rsi = self.calculate_rsi(data)
        
        # Calculate ATR if not provided
        if atr is None:
            atr = self.calculate_atr(data)
        
        # Calculate price trend (slope of moving average)
        ma_short = data['close'].rolling(window=10).mean()
        ma_long = data['close'].rolling(window=50).mean()
        trend = ma_short - ma_long
        
        # Calculate RSI slope
        rsi_slope = rsi.diff()
        
        # Classify
        def classify_market_regime(idx):
            if pd.isna(rsi.iloc[idx]) or pd.isna(atr.iloc[idx]) or pd.isna(trend.iloc[idx]):
                return "unknown"
            
            rsi_val = rsi.iloc[idx]
            rsi_slope_val = rsi_slope.iloc[idx]
            atr_val = atr.iloc[idx]
            trend_val = trend.iloc[idx]
            
            # High volatility
            if atr_val > atr.quantile(0.75):
                return "volatile"
            
            # Strong trend
            if abs(trend_val) > abs(trend.quantile(0.75)):
                return "trending"
            
            # Ranging
            return "ranging"
        
        regime = pd.Series([classify_market_regime(i) for i in range(len(data))], index=data.index)
        
        return regime

    def detect_regime(
        self,
        symbol: str,
        data: pd.DataFrame,
        liquidation_data: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        Detect market regime.
        
        Args:
            symbol: Trading symbol
            data: Price data DataFrame
            liquidation_data: Optional liquidation data
            
        Returns:
            Regime detection results
        """
        logger.info("detecting_regime", symbol=symbol)
        
        # Calculate volatility
        volatility = self.calculate_volatility(data)
        
        # Classify volatility regime
        volatility_regime = self.classify_volatility(volatility)
        
        # Classify market regime
        market_regime = self.classify_market(data)
        
        # Get current regime
        current_volatility_regime = volatility_regime.iloc[-1] if len(volatility_regime) > 0 else "unknown"
        current_market_regime = market_regime.iloc[-1] if len(market_regime) > 0 else "unknown"
        
        result = {
            "symbol": symbol,
            "volatility_regime": current_volatility_regime,
            "market_regime": current_market_regime,
            "volatility_series": volatility,
            "volatility_regime_series": volatility_regime,
            "market_regime_series": market_regime,
        }
        
        logger.info(
            "regime_detected",
            symbol=symbol,
            volatility_regime=current_volatility_regime,
            market_regime=current_market_regime,
        )
        
        return result

