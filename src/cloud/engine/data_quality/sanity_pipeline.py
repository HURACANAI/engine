"""
Data Sanity Pipeline - Clean Historical Data

Main data cleaning orchestrator that:
1. Deduplicates trades
2. Fixes timestamp issues
3. Removes outliers (bad prints, flash crashes)
4. Handles gaps (exchange outages)
5. Applies historical fees

This is the FIRST step in any training pipeline. Clean data = better models.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np
import polars as pl
import structlog

from .fee_schedule import HistoricalFeeManager
from .gap_handler import DataGap, GapHandler

logger = structlog.get_logger(__name__)


@dataclass
class SanityReport:
    """Report from data sanity pipeline."""

    original_rows: int
    cleaned_rows: int
    duplicates_removed: int
    bad_timestamps_removed: int
    outliers_removed: int
    gaps_detected: int
    gaps_filled: int
    fees_applied: bool
    processing_time_seconds: float


class DataSanityPipeline:
    """
    Complete data cleaning pipeline.

    Usage:
        pipeline = DataSanityPipeline(exchange='binance')
        clean_data, report = pipeline.clean(raw_data)

        print(report.summary())
    """

    def __init__(
        self,
        exchange: str = 'binance',
        outlier_threshold_pct: float = 0.10,  # 10% move in 1 candle
        max_gap_minutes: int = 5,
        apply_fees: bool = True
    ):
        """
        Initialize data sanity pipeline.

        Args:
            exchange: Exchange name for fee lookups
            outlier_threshold_pct: Remove candles with >X% range
            max_gap_minutes: Gap detection threshold
            apply_fees: Whether to add historical fee column
        """
        self.exchange = exchange.lower()
        self.outlier_threshold = outlier_threshold_pct
        self.max_gap_minutes = max_gap_minutes
        self.apply_fees = apply_fees

        # Initialize components
        self.fee_manager = HistoricalFeeManager() if apply_fees else None
        self.gap_handler = GapHandler(max_gap_minutes=max_gap_minutes)

        logger.info(
            "data_sanity_pipeline_initialized",
            exchange=exchange,
            outlier_threshold_pct=outlier_threshold_pct,
            max_gap_minutes=max_gap_minutes
        )

    def clean(self, df: pl.DataFrame) -> tuple[pl.DataFrame, SanityReport]:
        """
        Run complete cleaning pipeline.

        Args:
            df: Raw candle data with columns:
                - timestamp (datetime)
                - open, high, low, close (float)
                - volume (float)

        Returns:
            (cleaned_dataframe, sanity_report)
        """
        start_time = datetime.now()
        original_rows = len(df)

        logger.info("starting_data_sanity", original_rows=original_rows)

        # Step 1: Remove duplicates
        df, dups_removed = self._remove_duplicates(df)

        # Step 2: Fix timestamps
        df, bad_ts_removed = self._fix_timestamps(df)

        # Step 3: Remove outliers
        df, outliers_removed = self._remove_outliers(df)

        # Step 4: Handle gaps (marks rows, does NOT forward-fill OHLCV)
        df, gaps = self.gap_handler.process(df)
        
        # Step 4b: Filter out rows with gaps BEFORE applying fees
        # This prevents training on synthetic/flat candles that cause perfect hindsight
        rows_before_gap_filter = len(df)
        if 'gap_flag' in df.columns:
            df = df.filter(~pl.col('gap_flag'))
            rows_removed = rows_before_gap_filter - len(df)
            if rows_removed > 0:
                logger.warning(
                    "gap_rows_excluded",
                    rows_removed=rows_removed,
                    pct=rows_removed / rows_before_gap_filter * 100 if rows_before_gap_filter > 0 else 0,
                    note="Rows touching gaps excluded to prevent perfect hindsight from flat candles"
                )

        # Step 5: Apply historical fees (only to non-gap rows)
        if self.apply_fees and self.fee_manager:
            df = self.fee_manager.apply_fees_to_dataframe(df, self.exchange)

        # Generate report
        processing_time = (datetime.now() - start_time).total_seconds()

        report = SanityReport(
            original_rows=original_rows,
            cleaned_rows=len(df),
            duplicates_removed=dups_removed,
            bad_timestamps_removed=bad_ts_removed,
            outliers_removed=outliers_removed,
            gaps_detected=len(gaps),
            gaps_filled=0,  # Gaps are NOT filled - rows are excluded instead
            fees_applied=self.apply_fees,
            processing_time_seconds=processing_time
        )

        logger.info(
            "data_sanity_complete",
            original_rows=original_rows,
            cleaned_rows=len(df),
            removed=original_rows - len(df),
            processing_seconds=processing_time
        )

        return df, report

    def _remove_duplicates(self, df: pl.DataFrame) -> tuple[pl.DataFrame, int]:
        """
        Remove duplicate candles.

        Duplicates occur due to:
        - Exchange API glitches
        - Retry logic double-fetching
        - Data feed overlap

        Strategy: Keep first occurrence, drop rest
        """
        original_len = len(df)

        # Define uniqueness by timestamp + price + volume
        # (same candle at same time with same values = duplicate)
        df = df.unique(subset=['timestamp', 'close', 'volume'])

        removed = original_len - len(df)

        if removed > 0:
            logger.warning(
                "duplicates_removed",
                count=removed,
                pct=removed / original_len * 100 if original_len > 0 else 0
            )

        return df, removed

    def _fix_timestamps(self, df: pl.DataFrame) -> tuple[pl.DataFrame, int]:
        """
        Fix timestamp issues.

        Issues:
        - Timestamps jump backwards (clock drift)
        - Non-monotonic ordering
        - Duplicate timestamps

        Strategy: Enforce monotonically increasing timestamps
        """
        original_len = len(df)

        # Sort by timestamp first
        df = df.sort('timestamp')

        # Find backwards jumps
        df = df.with_columns([
            (pl.col('timestamp').shift(1) >= pl.col('timestamp')).alias('is_bad_timestamp')
        ])

        # Remove rows with bad timestamps
        bad_ts_count = df['is_bad_timestamp'].sum()
        df = df.filter(~pl.col('is_bad_timestamp').fill_null(False))
        df = df.drop('is_bad_timestamp')

        removed = original_len - len(df)

        if removed > 0:
            logger.warning(
                "bad_timestamps_removed",
                count=removed,
                backwards_jumps=bad_ts_count
            )

        return df, removed

    def _remove_outliers(self, df: pl.DataFrame) -> tuple[pl.DataFrame, int]:
        """
        Remove outlier candles.

        Outliers include:
        - Flash crashes (>10% move in 1 candle)
        - Fat finger trades
        - Exchange bugs (negative prices, zero volume)

        Strategy: Remove candles with suspicious characteristics
        """
        original_len = len(df)

        # Calculate candle range
        df = df.with_columns([
            ((pl.col('high') / pl.col('low')) - 1).alias('candle_range_pct')
        ])

        # Mark outliers
        df = df.with_columns([
            (
                # Excessive range
                (pl.col('candle_range_pct') > self.outlier_threshold) |
                # Zero or negative prices
                (pl.col('low') <= 0) |
                (pl.col('high') <= 0) |
                # High > Low violation
                (pl.col('high') < pl.col('low')) |
                # Zero volume (dead candle)
                (pl.col('volume') <= 0)
            ).alias('is_outlier')
        ])

        # Count and remove outliers
        outliers = df['is_outlier'].sum()
        df = df.filter(~pl.col('is_outlier'))
        df = df.drop('is_outlier', 'candle_range_pct')

        removed = original_len - len(df)

        if removed > 0:
            logger.warning(
                "outliers_removed",
                count=removed,
                threshold_pct=self.outlier_threshold * 100
            )

        return df, removed

    def validate_schema(self, df: pl.DataFrame) -> bool:
        """
        Validate that dataframe has required columns.

        Required:
        - timestamp (datetime)
        - open, high, low, close, volume (float)
        """
        required_columns = {'timestamp', 'open', 'high', 'low', 'close', 'volume'}
        actual_columns = set(df.columns)

        missing = required_columns - actual_columns

        if missing:
            logger.error(
                "invalid_schema",
                missing_columns=list(missing),
                actual_columns=list(actual_columns)
            )
            return False

        # Check data types
        if not df['timestamp'].dtype in [pl.Datetime, pl.Datetime('ms'), pl.Datetime('us')]:
            logger.error("timestamp_not_datetime", dtype=df['timestamp'].dtype)
            return False

        for col in ['open', 'high', 'low', 'close', 'volume']:
            if not df[col].dtype in [pl.Float64, pl.Float32]:
                logger.warning(
                    "incorrect_dtype_will_cast",
                    column=col,
                    current_dtype=df[col].dtype,
                    target_dtype=pl.Float64
                )

        return True


def format_sanity_report(report: SanityReport) -> str:
    """
    Format sanity report as human-readable text.

    Usage:
        clean_data, report = pipeline.clean(raw_data)
        print(format_sanity_report(report))
    """
    removal_rate = (
        (report.original_rows - report.cleaned_rows) / report.original_rows * 100
        if report.original_rows > 0 else 0
    )

    report_text = f"""
╔═══════════════════════════════════════════╗
║     DATA SANITY PIPELINE REPORT           ║
╚═══════════════════════════════════════════╝

📊 Summary
──────────────────────────────────────────────
Original Rows:     {report.original_rows:,}
Cleaned Rows:      {report.cleaned_rows:,}
Rows Removed:      {report.original_rows - report.cleaned_rows:,} ({removal_rate:.2f}%)

🧹 Cleaning Steps
──────────────────────────────────────────────
Duplicates:        {report.duplicates_removed:,} removed
Bad Timestamps:    {report.bad_timestamps_removed:,} removed
Outliers:          {report.outliers_removed:,} removed
Gaps Detected:     {report.gaps_detected}
Gaps Filled:       {report.gaps_filled}
Fees Applied:      {'✓' if report.fees_applied else '✗'}

⏱️ Performance
──────────────────────────────────────────────
Processing Time:   {report.processing_time_seconds:.2f} seconds
Rows/Second:       {report.original_rows / report.processing_time_seconds:,.0f}

{'✅ Data is clean and ready for training!' if removal_rate < 1.0 else '⚠️ High removal rate - check data quality'}
"""

    return report_text
