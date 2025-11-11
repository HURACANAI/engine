"""
Mechanic Utilities - Financial Analysis Functions

Provides utility functions for financial analysis including:
- Geometric linking of returns (compound returns)
- Annualization of returns and volatility
- Wealth index calculation
- Drawdown analysis

Author: Huracan Engine Team
"""

from __future__ import annotations

from typing import Optional, Tuple
from enum import Enum
import numpy as np
import pandas as pd
import polars as pl
import structlog

logger = structlog.get_logger(__name__)


class Frequency(Enum):
    """Data frequency for annualization."""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"


class Mechanic:
    """
    Mechanic utilities for financial analysis.
    
    Provides functions for:
    - Geometric linking of returns (compound returns)
    - Annualization calculations
    - Wealth index creation
    - Drawdown analysis
    """
    
    @staticmethod
    def geometric_link(
        returns: pl.DataFrame | pd.DataFrame | np.ndarray,
        return_column: str = 'returns'
    ) -> pl.DataFrame | pd.DataFrame | np.ndarray:
        """
        Compute multi-period growth using geometric linking.
        
        This corrects for variance drag when adding returns directly.
        
        Formula: geometric_return = (1 + r1) * (1 + r2) * ... * (1 + rn) - 1
        
        Args:
            returns: DataFrame or array with returns
            return_column: Name of returns column (if DataFrame)
        
        Returns:
            DataFrame or array with geometric_return column added
        
        Usage:
            df = Mechanic.geometric_link(df)
            # df now has 'geometric_return' column
        """
        if isinstance(returns, pl.DataFrame):
            # Polars DataFrame
            returns_array = returns[return_column].to_numpy()
            geometric_return = (1 + returns_array).prod() - 1
            
            # Add as new column
            return returns.with_columns([
                pl.lit(geometric_return).alias('geometric_return')
            ])
        
        elif isinstance(returns, pd.DataFrame):
            # Pandas DataFrame
            returns_array = returns[return_column].values
            geometric_return = (1 + returns_array).prod() - 1
            
            # Add as new column
            result = returns.copy()
            result['geometric_return'] = geometric_return
            return result
        
        else:
            # NumPy array
            geometric_return = (1 + returns).prod() - 1
            return np.array([geometric_return])
    
    @staticmethod
    def annualize_return(
        returns: np.ndarray | pd.Series | pl.Series,
        periods_per_year: Optional[int] = None,
        frequency: Optional[Frequency] = None
    ) -> float:
        """
        Annualize return for comparison across assets or timeframes.
        
        Formula: (1 + r).prod() ** (periods_per_year / len(r)) - 1
        
        Args:
            returns: Array or Series of returns
            periods_per_year: Number of periods per year (e.g., 252 for daily)
            frequency: Data frequency (auto-detects if not provided)
        
        Returns:
            Annualized return
        
        Usage:
            annual_return = Mechanic.annualize_return(returns, periods_per_year=252)
        """
        # Convert to numpy array
        if isinstance(returns, pl.Series):
            returns_array = returns.to_numpy()
        elif isinstance(returns, pd.Series):
            returns_array = returns.values
        else:
            returns_array = np.array(returns)
        
        if len(returns_array) == 0:
            return 0.0
        
        # Auto-detect frequency if not provided
        if periods_per_year is None:
            if frequency:
                periods_per_year = Mechanic._get_periods_per_year(frequency)
            else:
                # Default to daily (252 trading days)
                periods_per_year = 252
                logger.warning("using_default_periods_per_year", periods=252)
        
        # Calculate annualized return
        total_return = (1 + returns_array).prod()
        n_periods = len(returns_array)
        
        if n_periods == 0:
            return 0.0
        
        annualized = total_return ** (periods_per_year / n_periods) - 1
        
        return float(annualized)
    
    @staticmethod
    def annualize_volatility(
        returns: np.ndarray | pd.Series | pl.Series,
        periods_per_year: Optional[int] = None,
        frequency: Optional[Frequency] = None
    ) -> float:
        """
        Annualize volatility for comparison across assets or timeframes.
        
        Formula: std(r) * sqrt(periods_per_year)
        
        Args:
            returns: Array or Series of returns
            periods_per_year: Number of periods per year (e.g., 252 for daily)
            frequency: Data frequency (auto-detects if not provided)
        
        Returns:
            Annualized volatility
        
        Usage:
            annual_vol = Mechanic.annualize_volatility(returns, periods_per_year=252)
        """
        # Convert to numpy array
        if isinstance(returns, pl.Series):
            returns_array = returns.to_numpy()
        elif isinstance(returns, pd.Series):
            returns_array = returns.values
        else:
            returns_array = np.array(returns)
        
        if len(returns_array) == 0:
            return 0.0
        
        # Auto-detect frequency if not provided
        if periods_per_year is None:
            if frequency:
                periods_per_year = Mechanic._get_periods_per_year(frequency)
            else:
                # Default to daily (252 trading days)
                periods_per_year = 252
                logger.warning("using_default_periods_per_year", periods=252)
        
        # Calculate annualized volatility
        volatility = np.std(returns_array)
        annualized = volatility * np.sqrt(periods_per_year)
        
        return float(annualized)
    
    @staticmethod
    def create_wealth_index(
        returns: pl.DataFrame | pd.DataFrame | np.ndarray,
        return_column: str = 'returns',
        initial_value: float = 1.0
    ) -> pl.DataFrame | pd.DataFrame | np.ndarray:
        """
        Create wealth index showing cumulative growth.
        
        Formula: wealth_index = (1 + returns).cumprod() * initial_value
        
        Args:
            returns: DataFrame or array with returns
            return_column: Name of returns column (if DataFrame)
            initial_value: Starting value (default: 1.0)
        
        Returns:
            DataFrame or array with wealth_index column added
        
        Usage:
            df = Mechanic.create_wealth_index(df, initial_value=1000.0)
            # df now has 'wealth_index' column
        """
        if isinstance(returns, pl.DataFrame):
            # Polars DataFrame
            returns_array = returns[return_column].to_numpy()
            wealth_index = (1 + returns_array).cumprod() * initial_value
            
            # Add as new column
            return returns.with_columns([
                pl.Series('wealth_index', wealth_index)
            ])
        
        elif isinstance(returns, pd.DataFrame):
            # Pandas DataFrame
            returns_array = returns[return_column].values
            wealth_index = (1 + returns_array).cumprod() * initial_value
            
            # Add as new column
            result = returns.copy()
            result['wealth_index'] = wealth_index
            return result
        
        else:
            # NumPy array
            wealth_index = (1 + returns).cumprod() * initial_value
            return wealth_index
    
    @staticmethod
    def calc_drawdowns(
        data: pl.DataFrame | pd.DataFrame,
        wealth_column: str = 'wealth_index'
    ) -> pl.DataFrame | pd.DataFrame:
        """
        Calculate drawdowns from wealth index.
        
        Drawdown shows risk exposure and recovery time.
        Formula: drawdown = (wealth_index - previous_peak) / previous_peak
        
        Args:
            data: DataFrame with wealth_index column
            wealth_column: Name of wealth index column (default: 'wealth_index')
        
        Returns:
            DataFrame with drawdown and previous_peak columns added
        
        Usage:
            df = Mechanic.calc_drawdowns(df)
            # df now has 'drawdown' and 'previous_peak' columns
        """
        if isinstance(data, pl.DataFrame):
            # Polars DataFrame
            wealth = data[wealth_column].to_numpy()
            previous_peaks = np.maximum.accumulate(wealth)
            drawdown = (wealth - previous_peaks) / previous_peaks
            
            # Add as new columns
            return data.with_columns([
                pl.Series('previous_peak', previous_peaks),
                pl.Series('drawdown', drawdown)
            ])
        
        else:
            # Pandas DataFrame
            wealth = data[wealth_column].values
            previous_peaks = np.maximum.accumulate(wealth)
            drawdown = (wealth - previous_peaks) / previous_peaks
            
            # Add as new columns
            result = data.copy()
            result['previous_peak'] = previous_peaks
            result['drawdown'] = drawdown
            return result
    
    @staticmethod
    def get_max_drawdown(
        data: pl.DataFrame | pd.DataFrame,
        wealth_column: str = 'wealth_index'
    ) -> Tuple[float, int, int]:
        """
        Get maximum drawdown and recovery information.
        
        Args:
            data: DataFrame with wealth_index column
            wealth_column: Name of wealth index column
        
        Returns:
            Tuple of (max_drawdown, drawdown_start_idx, recovery_idx)
        """
        if isinstance(data, pl.DataFrame):
            wealth = data[wealth_column].to_numpy()
        else:
            wealth = data[wealth_column].values
        
        previous_peaks = np.maximum.accumulate(wealth)
        drawdown = (wealth - previous_peaks) / previous_peaks
        
        max_drawdown_idx = np.argmin(drawdown)
        max_drawdown = float(drawdown[max_drawdown_idx])
        
        # Find recovery point (when drawdown returns to 0 or above)
        recovery_idx = max_drawdown_idx
        for i in range(max_drawdown_idx, len(drawdown)):
            if drawdown[i] >= 0:
                recovery_idx = i
                break
        
        # Find start of drawdown (when wealth was at peak)
        drawdown_start_idx = max_drawdown_idx
        for i in range(max_drawdown_idx, -1, -1):
            if wealth[i] >= previous_peaks[max_drawdown_idx]:
                drawdown_start_idx = i
                break
        
        return (abs(max_drawdown), drawdown_start_idx, recovery_idx)
    
    @staticmethod
    def _get_periods_per_year(frequency: Frequency) -> int:
        """Get periods per year for a given frequency."""
        mapping = {
            Frequency.DAILY: 252,      # Trading days
            Frequency.WEEKLY: 52,
            Frequency.MONTHLY: 12,
            Frequency.QUARTERLY: 4,
            Frequency.YEARLY: 1
        }
        return mapping.get(frequency, 252)  # Default to daily

