"""
Slippage Calibration Service

Fits slippage_bps_per_sigma per symbol from last 30 days of data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional, Tuple

import structlog

logger = structlog.get_logger(__name__)


class SlippageCalibrator:
    """Calibrates slippage per symbol from historical data."""
    
    def __init__(self, lookback_days: int = 30):
        """Initialize slippage calibrator.
        
        Args:
            lookback_days: Number of days to look back for calibration
        """
        self.lookback_days = lookback_days
        logger.info("slippage_calibrator_initialized", lookback_days=lookback_days)
    
    def calibrate_slippage(
        self,
        symbol: str,
        candles_df: pd.DataFrame,
        trades_df: Optional[pd.DataFrame] = None,
    ) -> Tuple[float, datetime]:
        """Calibrate slippage_bps_per_sigma for a symbol.
        
        Args:
            symbol: Trading symbol
            candles_df: DataFrame with OHLCV data (must have 'timestamp', 'close', 'volume', 'high', 'low')
            trades_df: Optional DataFrame with actual trade fills (if available)
            
        Returns:
            Tuple of (slippage_bps_per_sigma, fit_date)
        """
        if candles_df.empty:
            logger.warning("empty_candles_for_slippage", symbol=symbol)
            return 2.0, datetime.now(timezone.utc)  # Default slippage
        
        # Calculate returns and volatility
        candles_df = candles_df.copy()
        candles_df['returns'] = candles_df['close'].pct_change()
        candles_df['volatility'] = candles_df['returns'].rolling(window=20).std()
        
        # Calculate spread (high - low) as proxy for slippage
        candles_df['spread_bps'] = ((candles_df['high'] - candles_df['low']) / candles_df['close']) * 10000
        
        # Filter to recent data
        if 'timestamp' in candles_df.columns:
            candles_df['timestamp'] = pd.to_datetime(candles_df['timestamp'])
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=self.lookback_days)
            recent_df = candles_df[candles_df['timestamp'] >= cutoff_date]
        else:
            recent_df = candles_df.tail(self.lookback_days * 24)  # Assume hourly data
        
        if recent_df.empty:
            logger.warning("insufficient_recent_data_for_slippage", symbol=symbol)
            return 2.0, datetime.now(timezone.utc)
        
        # Calculate slippage per sigma
        # Slippage tends to scale with volatility (sigma)
        # Fit: slippage_bps = alpha * volatility + beta
        
        # Remove outliers
        recent_df = recent_df[
            (recent_df['spread_bps'] > 0) & 
            (recent_df['spread_bps'] < 100) &  # Max 1% spread
            (recent_df['volatility'].notna())
        ]
        
        if len(recent_df) < 10:
            logger.warning("insufficient_data_points_for_slippage", symbol=symbol, points=len(recent_df))
            # Use median spread as fallback
            median_slippage = recent_df['spread_bps'].median() if not recent_df.empty else 2.0
            return float(median_slippage), datetime.now(timezone.utc)
        
        # Fit linear model: slippage_bps = alpha * volatility + beta
        # We want slippage_bps_per_sigma, which is alpha
        X = recent_df['volatility'].values
        y = recent_df['spread_bps'].values
        
        # Simple linear regression
        X_mean = np.mean(X)
        y_mean = np.mean(y)
        
        numerator = np.sum((X - X_mean) * (y - y_mean))
        denominator = np.sum((X - X_mean) ** 2)
        
        if denominator == 0:
            # No volatility variation, use median
            median_slippage = np.median(y)
            return float(median_slippage), datetime.now(timezone.utc)
        
        alpha = numerator / denominator
        beta = y_mean - alpha * X_mean
        
        # Clamp to reasonable range (0.5 to 10 bps per sigma)
        slippage_bps_per_sigma = max(0.5, min(10.0, alpha))
        
        fit_date = datetime.now(timezone.utc)
        
        logger.info(
            "slippage_calibrated",
            symbol=symbol,
            slippage_bps_per_sigma=slippage_bps_per_sigma,
            fit_date=fit_date.isoformat(),
            sample_size=len(recent_df),
            median_spread=float(np.median(y)),
        )
        
        return float(slippage_bps_per_sigma), fit_date
    
    def calibrate_from_trades(
        self,
        symbol: str,
        trades_df: pd.DataFrame,
        candles_df: pd.DataFrame,
    ) -> Tuple[float, datetime]:
        """Calibrate slippage from actual trade fills (more accurate).
        
        Args:
            symbol: Trading symbol
            trades_df: DataFrame with trade fills (must have 'entry_price', 'fill_price', 'size', 'timestamp')
            candles_df: DataFrame with OHLCV data for volatility calculation
            
        Returns:
            Tuple of (slippage_bps_per_sigma, fit_date)
        """
        if trades_df.empty:
            logger.warning("empty_trades_for_slippage", symbol=symbol)
            return self.calibrate_slippage(symbol, candles_df)
        
        # Calculate actual slippage from trades
        trades_df = trades_df.copy()
        trades_df['slippage_bps'] = abs((trades_df['fill_price'] - trades_df['entry_price']) / trades_df['entry_price']) * 10000
        
        # Calculate volatility at time of trade
        candles_df = candles_df.copy()
        candles_df['returns'] = candles_df['close'].pct_change()
        candles_df['volatility'] = candles_df['returns'].rolling(window=20).std()
        
        # Merge trades with volatility
        if 'timestamp' in trades_df.columns and 'timestamp' in candles_df.columns:
            trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
            candles_df['timestamp'] = pd.to_datetime(candles_df['timestamp'])
            merged = pd.merge_asof(
                trades_df.sort_values('timestamp'),
                candles_df[['timestamp', 'volatility']].sort_values('timestamp'),
                on='timestamp',
                direction='nearest',
            )
        else:
            logger.warning("missing_timestamp_for_slippage_calibration", symbol=symbol)
            return self.calibrate_slippage(symbol, candles_df)
        
        # Filter to recent data
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=self.lookback_days)
        recent_df = merged[merged['timestamp'] >= cutoff_date]
        
        if len(recent_df) < 10:
            logger.warning("insufficient_trades_for_slippage", symbol=symbol, trades=len(recent_df))
            return self.calibrate_slippage(symbol, candles_df)
        
        # Fit linear model
        X = recent_df['volatility'].values
        y = recent_df['slippage_bps'].values
        
        X_mean = np.mean(X)
        y_mean = np.mean(y)
        
        numerator = np.sum((X - X_mean) * (y - y_mean))
        denominator = np.sum((X - X_mean) ** 2)
        
        if denominator == 0:
            median_slippage = np.median(y)
            return float(median_slippage), datetime.now(timezone.utc)
        
        alpha = numerator / denominator
        slippage_bps_per_sigma = max(0.5, min(10.0, alpha))
        
        fit_date = datetime.now(timezone.utc)
        
        logger.info(
            "slippage_calibrated_from_trades",
            symbol=symbol,
            slippage_bps_per_sigma=slippage_bps_per_sigma,
            fit_date=fit_date.isoformat(),
            sample_size=len(recent_df),
        )
        
        return float(slippage_bps_per_sigma), fit_date

