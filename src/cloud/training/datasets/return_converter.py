"""
Return Converter - Price-to-Return Conversion Layer

Converts raw price series to total return series (includes dividends, splits, etc.)
and cleans missing data. Normalizes data across all tickers for fair HTF comparison.

Key Features:
- Fetches adjusted close prices (via exchange wrapper or yfinance)
- Cleans NaN data with dropna(inplace=True)
- Calculates percent change (df.pct_change())
- Stores both raw returns and log returns in Brain Library

Author: Huracan Engine Team
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
import polars as pl  # type: ignore[reportMissingImports]
import structlog  # type: ignore[reportMissingImports]

from ..brain.brain_library import BrainLibrary

logger = structlog.get_logger(__name__)


class ReturnConverter:
    """
    Return Converter node that converts price series to return series.
    
    This normalizes the data across all tickers for fair HTF comparison.
    
    Usage:
        converter = ReturnConverter(brain_library=brain)
        
        # Convert price data to returns
        returns_df = converter.convert(
            price_data=df,
            price_column='close',
            symbol='BTC/USDT'
        )
        
        # Returns DataFrame with:
        # - raw_returns: Percent change returns
        # - log_returns: Log returns
        # - adjusted_close: Adjusted close prices (if available)
    """
    
    def __init__(
        self,
        brain_library: Optional[BrainLibrary] = None,
        use_adjusted_prices: bool = True,
        fill_method: str = 'forward'
    ):
        """
        Initialize Return Converter.
        
        Args:
            brain_library: Brain Library instance for storing returns (optional)
            use_adjusted_prices: Whether to use adjusted close prices (default: True)
            fill_method: Method for filling NaN values ('forward', 'backward', 'drop')
        """
        self.brain_library = brain_library
        self.use_adjusted_prices = use_adjusted_prices
        self.fill_method = fill_method
        
        logger.info(
            "return_converter_initialized",
            use_adjusted_prices=use_adjusted_prices,
            fill_method=fill_method
        )
    
    def convert(
        self,
        price_data: pl.DataFrame,
        price_column: str = 'close',
        symbol: Optional[str] = None,
        timestamp_column: str = 'timestamp'
    ) -> pl.DataFrame:
        """
        Convert price series to return series.
        
        Args:
            price_data: DataFrame with price data (must have price_column and timestamp_column)
            price_column: Name of the price column (default: 'close')
            symbol: Trading symbol (optional, for Brain Library storage)
            timestamp_column: Name of the timestamp column (default: 'timestamp')
        
        Returns:
            DataFrame with:
            - All original columns
            - raw_returns: Percent change returns
            - log_returns: Log returns
            - adjusted_close: Adjusted close prices (if use_adjusted_prices=True)
        """
        if price_column not in price_data.columns:
            raise ValueError(f"Price column '{price_column}' not found in DataFrame")
        
        if timestamp_column not in price_data.columns:
            raise ValueError(f"Timestamp column '{timestamp_column}' not found in DataFrame")
        
        # Convert to pandas for easier manipulation
        df = price_data.to_pandas().copy()
        
        # Sort by timestamp
        df = df.sort_values(timestamp_column).reset_index(drop=True)
        
        # Get price series
        prices = df[price_column].copy()
        
        # Use adjusted close if available and requested
        if self.use_adjusted_prices and 'adjusted_close' in df.columns:
            prices = df['adjusted_close'].copy()
            logger.debug("using_adjusted_close", symbol=symbol)
        elif self.use_adjusted_prices and 'adj_close' in df.columns:
            prices = df['adj_close'].copy()
            logger.debug("using_adj_close", symbol=symbol)
        else:
            # Use regular close prices
            prices = df[price_column].copy()
            logger.debug("using_regular_close", symbol=symbol)
        
        # Clean NaN data
        original_length = len(prices)
        
        if self.fill_method == 'drop':
            # Drop NaN values
            df = df.dropna(subset=[price_column]).reset_index(drop=True)
            prices = prices.dropna()
            logger.debug("dropped_nan_values", dropped=original_length - len(prices))
        elif self.fill_method == 'forward':
            # Forward fill (using .ffill() instead of deprecated fillna(method='ffill'))
            prices = prices.ffill()
            df[price_column] = prices
            logger.debug("forward_filled_nan_values")
        elif self.fill_method == 'backward':
            # Backward fill (using .bfill() instead of deprecated fillna(method='bfill'))
            prices = prices.bfill()
            df[price_column] = prices
            logger.debug("backward_filled_nan_values")
        else:
            # Drop remaining NaN
            df = df.dropna(subset=[price_column]).reset_index(drop=True)
            prices = prices.dropna()
        
        # Calculate percent change (raw returns)
        raw_returns = prices.pct_change()
        
        # Calculate log returns: log(price_t / price_{t-1}) = log(1 + return)
        log_returns = np.log(1 + raw_returns)
        
        # Add returns to DataFrame
        df['raw_returns'] = raw_returns
        df['log_returns'] = log_returns
        
        # Store adjusted close if we used it
        if self.use_adjusted_prices and 'adjusted_close' not in df.columns:
            if 'adj_close' in df.columns:
                df['adjusted_close'] = df['adj_close']
            else:
                df['adjusted_close'] = prices
        
        # Store in Brain Library if available
        if self.brain_library and symbol:
            self._store_in_brain_library(df, symbol, timestamp_column)
        
        logger.info(
            "returns_converted",
            symbol=symbol,
            original_rows=original_length,
            final_rows=len(df),
            mean_return=float(raw_returns.mean()),
            std_return=float(raw_returns.std())
        )
        
        # Convert back to polars
        return pl.from_pandas(df)
    
    def _store_in_brain_library(
        self,
        df: pd.DataFrame,
        symbol: str,
        timestamp_column: str
    ) -> None:
        """
        Store returns in Brain Library.
        """
        if not self.brain_library:
            logger.debug("brain_library_not_available", message="Skipping returns storage")
            return
        
        try:
            # Get price column if available
            price_col = None
            if 'close' in df.columns:
                price_col = df['close']
            elif 'adjusted_close' in df.columns:
                price_col = df['adjusted_close']
            
            # Store returns
            rows_stored = self.brain_library.store_returns(
                symbol=symbol,
                timestamps=df[timestamp_column],
                raw_returns=df['raw_returns'],
                log_returns=df['log_returns'],
                prices=price_col
            )
            
            logger.info(
                "returns_stored_in_brain_library",
                symbol=symbol,
                rows=rows_stored,
                timestamp_column=timestamp_column
            )
        except Exception as e:
            logger.warning(
                "returns_storage_failed",
                symbol=symbol,
                error=str(e),
                error_type=type(e).__name__
            )

