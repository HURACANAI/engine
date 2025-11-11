"""
Data Gates Service

Strict data gates to skip symbols with low volume, large gaps, abnormal spreads.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional

import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class DataGateResult:
    """Result of data gate checks."""
    passed: bool
    skip_reasons: List[str]
    warnings: List[str]


class DataGates:
    """Data gates for symbol filtering."""
    
    def __init__(
        self,
        min_volume_usd: float = 1_000_000.0,  # $1M daily volume
        max_gap_pct: float = 5.0,  # 5% price gap
        max_spread_bps: float = 50.0,  # 50 bps max spread
        min_data_coverage: float = 0.95,  # 95% data coverage
        max_missing_hours: int = 24,  # Max 24 hours missing
    ):
        """Initialize data gates.
        
        Args:
            min_volume_usd: Minimum daily volume in USD
            max_gap_pct: Maximum price gap percentage
            max_spread_bps: Maximum spread in basis points
            min_data_coverage: Minimum data coverage (0-1)
            max_missing_hours: Maximum missing hours
        """
        self.min_volume_usd = min_volume_usd
        self.max_gap_pct = max_gap_pct
        self.max_spread_bps = max_spread_bps
        self.min_data_coverage = min_data_coverage
        self.max_missing_hours = max_missing_hours
        logger.info("data_gates_initialized", min_volume_usd=min_volume_usd)
    
    def check_symbol(
        self,
        symbol: str,
        candles_df: pd.DataFrame,
        price_usd: Optional[float] = None,
    ) -> DataGateResult:
        """Check if symbol passes data gates.
        
        Args:
            symbol: Trading symbol
            candles_df: DataFrame with OHLCV data
            price_usd: Current price in USD (for volume calculation)
            
        Returns:
            DataGateResult with pass/fail and reasons
        """
        skip_reasons = []
        warnings = []
        
        if candles_df.empty:
            skip_reasons.append("empty_data")
            return DataGateResult(passed=False, skip_reasons=skip_reasons, warnings=warnings)
        
        # Check data coverage
        if 'timestamp' in candles_df.columns:
            candles_df['timestamp'] = pd.to_datetime(candles_df['timestamp'])
            expected_hours = (candles_df['timestamp'].max() - candles_df['timestamp'].min()).total_seconds() / 3600
            actual_hours = len(candles_df)
            coverage = actual_hours / expected_hours if expected_hours > 0 else 0.0
            
            if coverage < self.min_data_coverage:
                skip_reasons.append(f"low_data_coverage_{coverage:.2%}")
            
            missing_hours = expected_hours - actual_hours
            if missing_hours > self.max_missing_hours:
                skip_reasons.append(f"too_many_missing_hours_{missing_hours:.0f}")
        else:
            warnings.append("missing_timestamp_column")
        
        # Check volume
        if 'volume' in candles_df.columns:
            avg_volume = candles_df['volume'].mean()
            
            # Convert to USD if price provided
            if price_usd:
                avg_volume_usd = avg_volume * price_usd
            else:
                # Assume volume is already in USD or use close price
                if 'close' in candles_df.columns:
                    avg_price = candles_df['close'].mean()
                    avg_volume_usd = avg_volume * avg_price
                else:
                    avg_volume_usd = avg_volume
            
            if avg_volume_usd < self.min_volume_usd:
                skip_reasons.append(f"low_volume_{avg_volume_usd:.0f}_usd")
        else:
            warnings.append("missing_volume_column")
        
        # Check price gaps
        if 'close' in candles_df.columns:
            candles_df = candles_df.sort_values('timestamp' if 'timestamp' in candles_df.columns else candles_df.index)
            candles_df['price_change_pct'] = candles_df['close'].pct_change().abs() * 100
            
            max_gap = candles_df['price_change_pct'].max()
            if max_gap > self.max_gap_pct:
                skip_reasons.append(f"large_price_gap_{max_gap:.2f}%")
            
            # Check for zero or negative prices
            if (candles_df['close'] <= 0).any():
                skip_reasons.append("invalid_prices")
        else:
            warnings.append("missing_close_column")
        
        # Check spreads
        if 'high' in candles_df.columns and 'low' in candles_df.columns and 'close' in candles_df.columns:
            candles_df['spread_bps'] = ((candles_df['high'] - candles_df['low']) / candles_df['close']) * 10000
            max_spread = candles_df['spread_bps'].max()
            median_spread = candles_df['spread_bps'].median()
            
            if max_spread > self.max_spread_bps:
                skip_reasons.append(f"abnormal_spread_{max_spread:.1f}_bps")
            
            if median_spread > self.max_spread_bps * 0.5:
                warnings.append(f"high_median_spread_{median_spread:.1f}_bps")
        else:
            warnings.append("missing_ohlc_columns")
        
        # Check for constant prices (no movement)
        if 'close' in candles_df.columns:
            price_std = candles_df['close'].std()
            price_mean = candles_df['close'].mean()
            if price_mean > 0 and (price_std / price_mean) < 0.001:  # Less than 0.1% variation
                skip_reasons.append("constant_price")
        
        passed = len(skip_reasons) == 0
        
        logger.info(
            "data_gates_check",
            symbol=symbol,
            passed=passed,
            skip_reasons=skip_reasons,
            warnings=warnings,
        )
        
        return DataGateResult(passed=passed, skip_reasons=skip_reasons, warnings=warnings)

