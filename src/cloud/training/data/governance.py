"""
Data Governance - Rules and validation.

Implements:
- Vendor reconciliation
- Missing data policy
- Time alignment
- Outlier filtering
- Survivorship bias handling
- Universe selection rules
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class DataQualityReport:
    """Data quality report."""
    symbol: str
    start_date: datetime
    end_date: datetime
    total_rows: int
    missing_rows: int
    missing_pct: float
    gaps: List[Tuple[datetime, datetime]]
    outliers: int
    vendor_mismatches: int
    quality_score: float  # 0 to 1


class DataGovernance:
    """
    Data governance system.
    
    Features:
    - Vendor reconciliation (cross-check 2+ sources)
    - Missing data policy (forward fill small gaps, drop large)
    - Time alignment (snap to exchange timestamps)
    - Outlier filtering (robust z-score)
    - Survivorship bias handling (include delisted coins)
    - Universe selection rules (liquidity, age filters)
    """
    
    def __init__(
        self,
        max_gap_minutes: int = 60,  # Forward fill gaps < 1 hour
        outlier_z_threshold: float = 3.0,  # Z-score threshold
        min_liquidity_usd: float = 1000000.0,  # $1M daily volume
        min_age_days: int = 30,  # 30 days old
    ) -> None:
        """
        Initialize data governance.
        
        Args:
            max_gap_minutes: Maximum gap to forward fill (default: 60)
            outlier_z_threshold: Z-score threshold for outliers (default: 3.0)
            min_liquidity_usd: Minimum daily liquidity (default: $1M)
            min_age_days: Minimum coin age in days (default: 30)
        """
        self.max_gap_minutes = max_gap_minutes
        self.outlier_z_threshold = outlier_z_threshold
        self.min_liquidity_usd = min_liquidity_usd
        self.min_age_days = min_age_days
        
        logger.info(
            "data_governance_initialized",
            max_gap_minutes=max_gap_minutes,
            outlier_z_threshold=outlier_z_threshold
        )
    
    def reconcile_vendors(
        self,
        data_sources: Dict[str, pd.DataFrame],
        price_column: str = 'close'
    ) -> Tuple[pd.DataFrame, int]:
        """
        Reconcile data from multiple vendors.
        
        Cross-checks at least 2 sources and flags mismatches.
        
        Args:
            data_sources: Dictionary of vendor_name -> DataFrame
            price_column: Price column name
        
        Returns:
            (reconciled_dataframe, num_mismatches)
        """
        if len(data_sources) < 2:
            logger.warning("insufficient_vendors", count=len(data_sources))
            # Return first source if only one
            if data_sources:
                return list(data_sources.values())[0], 0
            return pd.DataFrame(), 0
        
        # Align all dataframes by timestamp
        aligned = {}
        for vendor, df in data_sources.items():
            if 'timestamp' in df.columns:
                df = df.set_index('timestamp')
            aligned[vendor] = df[price_column] if price_column in df.columns else pd.Series()
        
        # Combine into single dataframe
        combined = pd.DataFrame(aligned)
        
        # Calculate mismatches (prices differ by > 0.1%)
        mismatches = 0
        if len(combined.columns) >= 2:
            for col1 in combined.columns:
                for col2 in combined.columns:
                    if col1 != col2:
                        diff_pct = abs((combined[col1] - combined[col2]) / combined[col1]) * 100
                        mismatches += (diff_pct > 0.1).sum()
        
        # Use median price across vendors
        reconciled = combined.median(axis=1).to_frame(name=price_column)
        reconciled = reconciled.reset_index()
        
        logger.info(
            "vendor_reconciliation_complete",
            vendors=list(data_sources.keys()),
            mismatches=mismatches,
            rows=len(reconciled)
        )
        
        return reconciled, mismatches
    
    def handle_missing_data(
        self,
        data: pd.DataFrame,
        timestamp_column: str = 'timestamp'
    ) -> Tuple[pd.DataFrame, List[Tuple[datetime, datetime]]]:
        """
        Handle missing data according to policy.
        
        Forward fill only small gaps (< max_gap_minutes), drop large gaps.
        
        Args:
            data: DataFrame with data
            timestamp_column: Timestamp column name
        
        Returns:
            (cleaned_dataframe, list_of_gaps)
        """
        if timestamp_column not in data.columns:
            return data, []
        
        data = data.sort_values(timestamp_column)
        data = data.set_index(timestamp_column)
        
        # Detect gaps
        gaps = []
        if len(data) > 1:
            time_diffs = data.index.to_series().diff()
            gap_threshold = timedelta(minutes=self.max_gap_minutes)
            
            large_gaps = time_diffs[time_diffs > gap_threshold]
            for gap_start, gap_duration in large_gaps.items():
                gap_end = gap_start + gap_duration
                gaps.append((gap_start, gap_end))
        
        # Forward fill small gaps
        data = data.fillna(method='ffill', limit=self.max_gap_minutes)
        
        # Drop rows with remaining NaNs (large gaps)
        data = data.dropna()
        
        data = data.reset_index()
        
        logger.info(
            "missing_data_handled",
            original_rows=len(data),
            gaps_found=len(gaps),
            final_rows=len(data)
        )
        
        return data, gaps
    
    def align_to_exchange_timestamps(
        self,
        data: pd.DataFrame,
        exchange_timestamps: List[datetime],
        timestamp_column: str = 'timestamp'
    ) -> pd.DataFrame:
        """
        Align data to exchange timestamps (not local time).
        
        Args:
            data: DataFrame with data
            exchange_timestamps: List of exchange timestamps
            timestamp_column: Timestamp column name
        
        Returns:
            Aligned dataframe
        """
        if timestamp_column not in data.columns:
            return data
        
        # Create exchange timestamp index
        exchange_df = pd.DataFrame({timestamp_column: exchange_timestamps})
        exchange_df = exchange_df.set_index(timestamp_column)
        
        # Set timestamp as index
        data = data.set_index(timestamp_column)
        
        # Reindex to exchange timestamps (forward fill)
        aligned = data.reindex(exchange_df.index, method='ffill')
        
        aligned = aligned.reset_index()
        
        logger.debug("data_aligned_to_exchange_timestamps", rows=len(aligned))
        
        return aligned
    
    def filter_outliers(
        self,
        data: pd.DataFrame,
        return_column: str = 'returns',
        clip_tails: bool = True
    ) -> Tuple[pd.DataFrame, int]:
        """
        Filter outliers using robust z-score.
        
        Args:
            data: DataFrame with returns
            return_column: Return column name
            clip_tails: Whether to clip tails after investigation
        
        Returns:
            (filtered_dataframe, num_outliers)
        """
        if return_column not in data.columns:
            return data, 0
        
        returns = data[return_column].dropna()
        
        # Robust z-score using median and MAD
        median = returns.median()
        mad = (returns - median).abs().median()
        
        if mad == 0:
            return data, 0
        
        z_scores = (returns - median) / (1.4826 * mad)  # 1.4826 makes MAD consistent with std
        
        outliers = (z_scores.abs() > self.outlier_z_threshold).sum()
        
        if clip_tails:
            # Clip outliers to threshold
            lower_bound = median - self.outlier_z_threshold * 1.4826 * mad
            upper_bound = median + self.outlier_z_threshold * 1.4826 * mad
            data[return_column] = data[return_column].clip(lower=lower_bound, upper=upper_bound)
        else:
            # Remove outliers
            outlier_mask = z_scores.abs() <= self.outlier_z_threshold
            data = data[outlier_mask]
        
        logger.info(
            "outliers_filtered",
            outliers=outliers,
            method="clip" if clip_tails else "remove"
        )
        
        return data, outliers
    
    def check_universe_selection(
        self,
        symbol: str,
        daily_volume_usd: float,
        coin_age_days: int,
        is_delisted: bool = False
    ) -> Tuple[bool, str]:
        """
        Check if symbol meets universe selection criteria.
        
        Args:
            symbol: Trading symbol
            daily_volume_usd: Daily volume in USD
            coin_age_days: Coin age in days
            is_delisted: Whether coin is delisted
        
        Returns:
            (meets_criteria, reason)
        """
        # Include delisted coins for survivorship bias handling
        if is_delisted:
            return True, "Delisted coin included for survivorship bias"
        
        # Check liquidity
        if daily_volume_usd < self.min_liquidity_usd:
            return False, f"Volume {daily_volume_usd:,.0f} below minimum {self.min_liquidity_usd:,.0f}"
        
        # Check age
        if coin_age_days < self.min_age_days:
            return False, f"Age {coin_age_days} days below minimum {self.min_age_days} days"
        
        return True, "Meets all criteria"
    
    def generate_quality_report(
        self,
        data: pd.DataFrame,
        symbol: str,
        timestamp_column: str = 'timestamp'
    ) -> DataQualityReport:
        """
        Generate data quality report.
        
        Args:
            data: DataFrame to analyze
            symbol: Symbol name
            timestamp_column: Timestamp column
        
        Returns:
            DataQualityReport
        """
        total_rows = len(data)
        missing_rows = data.isnull().sum().sum()
        missing_pct = (missing_rows / (total_rows * len(data.columns))) * 100 if total_rows > 0 else 0.0
        
        # Detect gaps
        _, gaps = self.handle_missing_data(data.copy(), timestamp_column)
        
        # Count outliers (if returns column exists)
        outliers = 0
        if 'returns' in data.columns:
            _, outliers = self.filter_outliers(data.copy(), 'returns', clip_tails=False)
        
        # Quality score (0 to 1)
        quality_score = 1.0
        quality_score -= min(missing_pct / 10.0, 0.5)  # Penalize missing data
        quality_score -= min(len(gaps) / 10.0, 0.3)  # Penalize gaps
        quality_score -= min(outliers / total_rows if total_rows > 0 else 0, 0.2)  # Penalize outliers
        quality_score = max(0.0, quality_score)
        
        start_date = data[timestamp_column].min() if timestamp_column in data.columns else datetime.now()
        end_date = data[timestamp_column].max() if timestamp_column in data.columns else datetime.now()
        
        return DataQualityReport(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            total_rows=total_rows,
            missing_rows=missing_rows,
            missing_pct=missing_pct,
            gaps=gaps,
            outliers=outliers,
            vendor_mismatches=0,  # Would be set by reconcile_vendors
            quality_score=quality_score
        )

