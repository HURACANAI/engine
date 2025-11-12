"""
Gap Handler for Exchange Outages

Detects and handles missing data due to:
- Exchange downtime
- API rate limits
- Network issues
- Data feed interruptions

Strategy: Detect gaps, log them, and forward-fill price (conservative approach).
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List

import polars as pl
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class DataGap:
    """Represents a detected data gap."""

    start_time: datetime
    end_time: datetime
    duration_minutes: float
    candles_missing: int
    reason: str = "unknown"


class GapHandler:
    """
    Detects and handles gaps in candle data.

    Philosophy:
    - Gaps corrupt training data (model sees impossible price jumps)
    - Better to forward-fill than to ignore
    - But track gaps so you know when model trains on synthetic data
    """

    def __init__(
        self,
        max_gap_minutes: int = 5,
        forward_fill: bool = False  # Changed default: DO NOT forward-fill OHLCV
    ):
        """
        Initialize gap handler.

        Args:
            max_gap_minutes: Gap > this triggers detection
            forward_fill: DEPRECATED - Always False. Gaps are marked, not filled.
                          Forward-filling OHLCV creates flat candles and perfect hindsight.
        """
        self.max_gap_minutes = max_gap_minutes
        self.forward_fill = False  # Always False - never forward-fill OHLCV

        if forward_fill:
            logger.warning(
                "forward_fill_deprecated",
                message="forward_fill=True is deprecated. Gaps are marked, not filled, to prevent perfect hindsight."
            )

        logger.info(
            "gap_handler_initialized",
            max_gap_minutes=max_gap_minutes,
            note="Gaps will be marked with flags, NOT forward-filled"
        )

    def detect_gaps(self, df: pl.DataFrame) -> List[DataGap]:
        """
        Detect gaps in the data.

        Args:
            df: DataFrame with 'timestamp' column (sorted)

        Returns:
            List of detected gaps
        """
        if len(df) < 2:
            return []

        gaps = []

        # Calculate time differences
        # Polars returns duration as int64 (milliseconds) when subtracting datetimes
        df = df.with_columns([
            (pl.col('timestamp') - pl.col('timestamp').shift(1))
            .alias('gap_duration_ms')
        ])

        # Find rows where gap > threshold
        gap_threshold_ms = self.max_gap_minutes * 60 * 1000  # Convert minutes to milliseconds

        for row in df.iter_rows(named=True):
            gap_dur_ms = row.get('gap_duration_ms')

            if gap_dur_ms is not None and gap_dur_ms > gap_threshold_ms:
                # Convert milliseconds to timedelta for calculations
                gap_dur = timedelta(milliseconds=gap_dur_ms)
                # Calculate missing candles (assuming 1-minute candles)
                minutes_missing = gap_dur.total_seconds() / 60
                candles_missing = int(minutes_missing) - 1

                # Convert timestamp from milliseconds to datetime
                timestamp_ms = row['timestamp']
                if isinstance(timestamp_ms, int):
                    end_time = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
                else:
                    end_time = timestamp_ms
                start_time = end_time - gap_dur

                gap = DataGap(
                    start_time=start_time,
                    end_time=end_time,
                    duration_minutes=minutes_missing,
                    candles_missing=candles_missing,
                    reason="exchange_outage_or_api_limit"
                )

                gaps.append(gap)

                logger.warning(
                    "gap_detected",
                    start=gap.start_time,
                    end=gap.end_time,
                    duration_min=gap.duration_minutes,
                    candles_missing=candles_missing
                )

        return gaps

    def mark_gaps(self, df: pl.DataFrame, gaps: List[DataGap]) -> pl.DataFrame:
        """
        Mark rows that touch gaps with gap flags instead of forward-filling OHLCV.
        
        CRITICAL: We do NOT forward-fill OHLCV because it creates flat candles
        that lead to perfect hindsight labels (100% hit rate). Instead, we mark
        rows that touch gaps so they can be excluded from training.

        Args:
            df: DataFrame with gaps
            gaps: List of detected gaps

        Returns:
            DataFrame with gap_flag and gap_span_minutes columns added
        """
        if not gaps:
            # No gaps - add columns with False/0
            df = df.with_columns([
                pl.lit(False).alias('gap_flag'),
                pl.lit(0.0).alias('gap_span_minutes')
            ])
            return df

        # Create gap flags: mark rows that are within or immediately after a gap
        # We mark rows that are within the gap period or the first row after a gap
        gap_flags = pl.lit(False).alias('gap_flag')
        gap_spans = pl.lit(0.0).alias('gap_span_minutes')
        
        for gap in gaps:
            # Mark rows within the gap period
            within_gap = (
                (pl.col('timestamp') >= gap.start_time) &
                (pl.col('timestamp') <= gap.end_time)
            )
            # Also mark the first row immediately after the gap (it may have stale data)
            after_gap = (
                (pl.col('timestamp') > gap.end_time) &
                (pl.col('timestamp') <= gap.end_time + timedelta(minutes=1))
            )
            
            gap_flags = gap_flags | within_gap | after_gap
            # Set gap_span_minutes for rows within or after the gap
            gap_spans = pl.when(within_gap | after_gap).then(
                pl.lit(gap.duration_minutes)
            ).otherwise(gap_spans)

        df = df.with_columns([
            gap_flags,
            gap_spans
        ])

        gap_count = df['gap_flag'].sum()
        logger.info(
            "gaps_marked",
            method="gap_flagging",
            gaps_detected=len(gaps),
            rows_marked=gap_count,
            note="OHLCV NOT forward-filled - rows with gaps will be excluded from training"
        )

        return df

    def process(self, df: pl.DataFrame) -> tuple[pl.DataFrame, List[DataGap]]:
        """
        Detect gaps and mark rows (DO NOT forward-fill OHLCV).

        Returns:
            (processed_dataframe_with_gap_flags, detected_gaps)
        """
        # Ensure sorted by timestamp
        df = df.sort('timestamp')

        # Detect gaps
        gaps = self.detect_gaps(df)

        # Mark gaps (do NOT forward-fill OHLCV - this causes perfect hindsight)
        df = self.mark_gaps(df, gaps)

        logger.info(
            "gap_processing_complete",
            total_gaps=len(gaps),
            rows_marked=df['gap_flag'].sum() if 'gap_flag' in df.columns else 0,
            note="OHLCV preserved - gap rows marked for exclusion"
        )

        return df, gaps

    def generate_gap_report(self, gaps: List[DataGap]) -> str:
        """
        Generate human-readable gap report.

        Useful for understanding data quality issues.
        """
        if not gaps:
            return "✅ No gaps detected - clean data!"

        total_duration = sum(g.duration_minutes for g in gaps)
        total_candles_missing = sum(g.candles_missing for g in gaps)

        report = f"""
📊 Data Gap Report
═══════════════════

Total Gaps: {len(gaps)}
Total Duration: {total_duration:.1f} minutes
Missing Candles: {total_candles_missing}

Largest Gaps:
"""

        # Sort by duration
        sorted_gaps = sorted(gaps, key=lambda g: g.duration_minutes, reverse=True)

        for i, gap in enumerate(sorted_gaps[:5], 1):
            report += f"\n{i}. {gap.start_time} → {gap.end_time}"
            report += f"\n   Duration: {gap.duration_minutes:.1f} min ({gap.candles_missing} candles)"
            report += f"\n   Reason: {gap.reason}\n"

        return report
