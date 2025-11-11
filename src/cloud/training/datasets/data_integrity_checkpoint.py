"""
Data Integrity Checkpoint - Validation Before Training

Data quality = model quality. Adds a data validation checkpoint before every
Engine training cycle to prevent training corruption.

Features:
- Remove outliers (flash crashes, missing volume)
- Compare multiple data sources (Binance, Kraken) for consistency
- Store data integrity scores in the Log Book

Author: Huracan Engine Team
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import polars as pl
import structlog

logger = structlog.get_logger(__name__)


class IntegrityLevel(Enum):
    """Data integrity levels."""
    PASS = "pass"
    WARNING = "warning"
    FAIL = "fail"


@dataclass
class IntegrityCheck:
    """Result of a single integrity check."""
    check_name: str
    level: IntegrityLevel
    score: float  # 0-1, higher is better
    message: str
    details: Dict[str, any]


@dataclass
class IntegrityReport:
    """Complete data integrity report."""
    timestamp: datetime
    symbol: str
    total_rows: int
    passed_rows: int
    overall_score: float  # 0-1
    level: IntegrityLevel
    checks: List[IntegrityCheck]
    outliers_removed: int
    missing_data_points: int
    inconsistencies: int


class DataIntegrityCheckpoint:
    """
    Data validation checkpoint before training.
    
    Usage:
        checkpoint = DataIntegrityCheckpoint()
        
        # Validate data
        report = checkpoint.validate(data, symbol="BTC/USDT")
        
        if report.level == IntegrityLevel.FAIL:
            raise ValueError("Data integrity check failed")
        
        # Get cleaned data
        cleaned_data = checkpoint.clean_data(data, report)
    """
    
    def __init__(
        self,
        outlier_threshold_std: float = 5.0,
        min_volume_threshold: float = 0.0,
        max_price_change_pct: float = 0.50,  # 50% max change per candle
        require_volume: bool = True
    ):
        """
        Initialize data integrity checkpoint.
        
        Args:
            outlier_threshold_std: Standard deviations for outlier detection
            min_volume_threshold: Minimum volume threshold
            max_price_change_pct: Maximum allowed price change per candle (as fraction)
            require_volume: Whether to require non-zero volume
        """
        self.outlier_threshold_std = outlier_threshold_std
        self.min_volume_threshold = min_volume_threshold
        self.max_price_change_pct = max_price_change_pct
        self.require_volume = require_volume
        
        logger.info(
            "data_integrity_checkpoint_initialized",
            outlier_threshold_std=outlier_threshold_std,
            max_price_change_pct=max_price_change_pct
        )
    
    def validate(
        self,
        data: pl.DataFrame | pd.DataFrame,
        symbol: str,
        compare_sources: Optional[List[pl.DataFrame | pd.DataFrame]] = None
    ) -> IntegrityReport:
        """
        Validate data integrity.
        
        Args:
            data: Primary data source
            symbol: Trading symbol
            compare_sources: Optional list of other data sources for comparison
        
        Returns:
            IntegrityReport with validation results
        """
        # Convert to pandas for easier manipulation
        if isinstance(data, pl.DataFrame):
            df = data.to_pandas()
        else:
            df = data.copy()
        
        original_rows = len(df)
        checks = []
        
        # Check 1: Missing data
        missing_check = self._check_missing_data(df)
        checks.append(missing_check)
        
        # Check 2: Outliers (flash crashes)
        outlier_check, outliers_count = self._check_outliers(df)
        checks.append(outlier_check)
        
        # Check 3: Volume validation
        volume_check = self._check_volume(df)
        checks.append(volume_check)
        
        # Check 4: Price consistency (OHLC)
        price_check = self._check_price_consistency(df)
        checks.append(price_check)
        
        # Check 5: Timestamp continuity
        timestamp_check = self._check_timestamp_continuity(df)
        checks.append(timestamp_check)
        
        # Check 6: Cross-source comparison (if provided)
        if compare_sources:
            comparison_check, inconsistencies = self._check_cross_source_consistency(
                df, compare_sources
            )
            checks.append(comparison_check)
        else:
            inconsistencies = 0
        
        # Calculate overall score
        scores = [c.score for c in checks]
        overall_score = np.mean(scores) if scores else 0.0
        
        # Determine level
        if overall_score >= 0.9:
            level = IntegrityLevel.PASS
        elif overall_score >= 0.7:
            level = IntegrityLevel.WARNING
        else:
            level = IntegrityLevel.FAIL
        
        # Count missing data points
        missing_count = df.isnull().sum().sum()
        
        report = IntegrityReport(
            timestamp=datetime.now(),
            symbol=symbol,
            total_rows=original_rows,
            passed_rows=original_rows - outliers_count,
            overall_score=overall_score,
            level=level,
            checks=checks,
            outliers_removed=outliers_count,
            missing_data_points=int(missing_count),
            inconsistencies=inconsistencies
        )
        
        logger.info(
            "data_integrity_validated",
            symbol=symbol,
            overall_score=overall_score,
            level=level.value,
            outliers_removed=outliers_count,
            missing_data_points=missing_count
        )
        
        return report
    
    def clean_data(
        self,
        data: pl.DataFrame | pd.DataFrame,
        report: IntegrityReport
    ) -> pl.DataFrame | pd.DataFrame:
        """
        Clean data based on integrity report.
        
        Args:
            data: Original data
            report: Integrity report with issues identified
        
        Returns:
            Cleaned data
        """
        # Convert to pandas
        if isinstance(data, pl.DataFrame):
            df = data.to_pandas()
            return_polars = True
        else:
            df = data.copy()
            return_polars = False
        
        original_rows = len(df)
        
        # Remove outliers
        if 'close' in df.columns:
            returns = df['close'].pct_change()
            z_scores = np.abs((returns - returns.mean()) / (returns.std() + 1e-8))
            df = df[z_scores < self.outlier_threshold_std]
        
        # Remove rows with missing critical data
        critical_columns = ['open', 'high', 'low', 'close']
        if all(col in df.columns for col in critical_columns):
            df = df.dropna(subset=critical_columns)
        
        # Remove rows with zero/negative volume if required
        if self.require_volume and 'volume' in df.columns:
            df = df[df['volume'] > self.min_volume_threshold]
        
        # Remove flash crashes (extreme price changes)
        if 'close' in df.columns and 'open' in df.columns:
            price_change = abs((df['close'] - df['open']) / df['open'])
            df = df[price_change <= self.max_price_change_pct]
        
        cleaned_rows = len(df)
        removed = original_rows - cleaned_rows
        
        logger.info(
            "data_cleaned",
            original_rows=original_rows,
            cleaned_rows=cleaned_rows,
            removed=removed
        )
        
        if return_polars:
            return pl.from_pandas(df)
        else:
            return df
    
    def _check_missing_data(self, df: pd.DataFrame) -> IntegrityCheck:
        """Check for missing data."""
        missing_pct = df.isnull().sum().sum() / (len(df) * len(df.columns))
        score = 1.0 - missing_pct
        
        if score >= 0.95:
            level = IntegrityLevel.PASS
            message = "No significant missing data"
        elif score >= 0.90:
            level = IntegrityLevel.WARNING
            message = f"Some missing data: {missing_pct:.2%}"
        else:
            level = IntegrityLevel.FAIL
            message = f"Excessive missing data: {missing_pct:.2%}"
        
        return IntegrityCheck(
            check_name="missing_data",
            level=level,
            score=score,
            message=message,
            details={"missing_percentage": missing_pct}
        )
    
    def _check_outliers(self, df: pd.DataFrame) -> Tuple[IntegrityCheck, int]:
        """Check for outliers (flash crashes)."""
        if 'close' not in df.columns:
            return IntegrityCheck(
                check_name="outliers",
                level=IntegrityLevel.WARNING,
                score=0.5,
                message="Cannot check outliers without close price",
                details={}
            ), 0
        
        returns = df['close'].pct_change().dropna()
        z_scores = np.abs((returns - returns.mean()) / (returns.std() + 1e-8))
        outliers = (z_scores > self.outlier_threshold_std).sum()
        outlier_pct = outliers / len(returns) if len(returns) > 0 else 0.0
        score = 1.0 - min(1.0, outlier_pct * 10)  # Penalize heavily
        
        if score >= 0.9:
            level = IntegrityLevel.PASS
            message = f"Few outliers: {outliers}"
        elif score >= 0.7:
            level = IntegrityLevel.WARNING
            message = f"Moderate outliers: {outliers} ({outlier_pct:.2%})"
        else:
            level = IntegrityLevel.FAIL
            message = f"Excessive outliers: {outliers} ({outlier_pct:.2%})"
        
        return IntegrityCheck(
            check_name="outliers",
            level=level,
            score=score,
            message=message,
            details={"outlier_count": int(outliers), "outlier_percentage": outlier_pct}
        ), int(outliers)
    
    def _check_volume(self, df: pd.DataFrame) -> IntegrityCheck:
        """Check volume data."""
        if 'volume' not in df.columns:
            return IntegrityCheck(
                check_name="volume",
                level=IntegrityLevel.WARNING,
                score=0.5,
                message="No volume column",
                details={}
            )
        
        zero_volume = (df['volume'] <= self.min_volume_threshold).sum()
        zero_volume_pct = zero_volume / len(df) if len(df) > 0 else 0.0
        score = 1.0 - zero_volume_pct
        
        if score >= 0.95:
            level = IntegrityLevel.PASS
            message = "Volume data looks good"
        elif score >= 0.90:
            level = IntegrityLevel.WARNING
            message = f"Some zero volume: {zero_volume} ({zero_volume_pct:.2%})"
        else:
            level = IntegrityLevel.FAIL
            message = f"Excessive zero volume: {zero_volume} ({zero_volume_pct:.2%})"
        
        return IntegrityCheck(
            check_name="volume",
            level=level,
            score=score,
            message=message,
            details={"zero_volume_count": int(zero_volume), "zero_volume_percentage": zero_volume_pct}
        )
    
    def _check_price_consistency(self, df: pd.DataFrame) -> IntegrityCheck:
        """Check OHLC price consistency."""
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_cols):
            return IntegrityCheck(
                check_name="price_consistency",
                level=IntegrityLevel.WARNING,
                score=0.5,
                message="Missing OHLC columns",
                details={}
            )
        
        # Check: high >= low, high >= open, high >= close, low <= open, low <= close
        invalid = (
            (df['high'] < df['low']) |
            (df['high'] < df['open']) |
            (df['high'] < df['close']) |
            (df['low'] > df['open']) |
            (df['low'] > df['close'])
        ).sum()
        
        invalid_pct = invalid / len(df) if len(df) > 0 else 0.0
        score = 1.0 - invalid_pct
        
        if score >= 0.99:
            level = IntegrityLevel.PASS
            message = "Price data is consistent"
        elif score >= 0.95:
            level = IntegrityLevel.WARNING
            message = f"Some price inconsistencies: {invalid}"
        else:
            level = IntegrityLevel.FAIL
            message = f"Excessive price inconsistencies: {invalid} ({invalid_pct:.2%})"
        
        return IntegrityCheck(
            check_name="price_consistency",
            level=level,
            score=score,
            message=message,
            details={"invalid_count": int(invalid), "invalid_percentage": invalid_pct}
        )
    
    def _check_timestamp_continuity(self, df: pd.DataFrame) -> IntegrityCheck:
        """Check timestamp continuity."""
        if 'timestamp' not in df.columns:
            return IntegrityCheck(
                check_name="timestamp_continuity",
                level=IntegrityLevel.WARNING,
                score=0.5,
                message="No timestamp column",
                details={}
            )
        
        timestamps = pd.to_datetime(df['timestamp'])
        timestamps_sorted = timestamps.sort_values()
        gaps = (timestamps_sorted.diff() > timestamps_sorted.diff().median() * 2).sum()
        gaps_pct = gaps / len(timestamps) if len(timestamps) > 0 else 0.0
        score = 1.0 - gaps_pct
        
        if score >= 0.95:
            level = IntegrityLevel.PASS
            message = "Timestamps are continuous"
        elif score >= 0.90:
            level = IntegrityLevel.WARNING
            message = f"Some timestamp gaps: {gaps}"
        else:
            level = IntegrityLevel.FAIL
            message = f"Excessive timestamp gaps: {gaps} ({gaps_pct:.2%})"
        
        return IntegrityCheck(
            check_name="timestamp_continuity",
            level=level,
            score=score,
            message=message,
            details={"gap_count": int(gaps), "gap_percentage": gaps_pct}
        )
    
    def _check_cross_source_consistency(
        self,
        primary: pd.DataFrame,
        compare_sources: List[pd.DataFrame]
    ) -> Tuple[IntegrityCheck, int]:
        """Check consistency across multiple data sources."""
        if 'close' not in primary.columns:
            return IntegrityCheck(
                check_name="cross_source_consistency",
                level=IntegrityLevel.WARNING,
                score=0.5,
                message="Cannot compare without close price",
                details={}
            ), 0
        
        inconsistencies = 0
        total_comparisons = 0
        
        for compare_df in compare_sources:
            if isinstance(compare_df, pl.DataFrame):
                compare_df = compare_df.to_pandas()
            
            if 'close' not in compare_df.columns:
                continue
            
            # Align timestamps if possible
            if 'timestamp' in primary.columns and 'timestamp' in compare_df.columns:
                primary_ts = pd.to_datetime(primary['timestamp'])
                compare_ts = pd.to_datetime(compare_df['timestamp'])
                
                # Find overlapping timestamps
                common_ts = set(primary_ts) & set(compare_ts)
                if len(common_ts) == 0:
                    continue
                
                primary_aligned = primary[primary_ts.isin(common_ts)]['close']
                compare_aligned = compare_df[compare_ts.isin(common_ts)]['close']
                
                # Compare prices (allow small differences for different exchanges)
                price_diff = abs(primary_aligned.values - compare_aligned.values) / primary_aligned.values
                inconsistencies += (price_diff > 0.01).sum()  # 1% threshold
                total_comparisons += len(price_diff)
        
        inconsistency_pct = inconsistencies / total_comparisons if total_comparisons > 0 else 0.0
        score = 1.0 - inconsistency_pct
        
        if score >= 0.95:
            level = IntegrityLevel.PASS
            message = "Data sources are consistent"
        elif score >= 0.90:
            level = IntegrityLevel.WARNING
            message = f"Some inconsistencies: {inconsistencies}/{total_comparisons}"
        else:
            level = IntegrityLevel.FAIL
            message = f"Excessive inconsistencies: {inconsistencies}/{total_comparisons}"
        
        return IntegrityCheck(
            check_name="cross_source_consistency",
            level=level,
            score=score,
            message=message,
            details={
                "inconsistencies": int(inconsistencies),
                "total_comparisons": int(total_comparisons),
                "inconsistency_percentage": inconsistency_pct
            }
        ), int(inconsistencies)

