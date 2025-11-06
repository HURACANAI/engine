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
from datetime import datetime, timedelta
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
        forward_fill: bool = True
    ):
        """
        Initialize gap handler.

        Args:
            max_gap_minutes: Gap > this triggers detection
            forward_fill: Whether to fill gaps (vs mark as NaN)
        """
        self.max_gap_minutes = max_gap_minutes
        self.forward_fill = forward_fill

        logger.info(
            "gap_handler_initialized",
            max_gap_minutes=max_gap_minutes,
            forward_fill=forward_fill
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
        df = df.with_columns([
            (pl.col('timestamp') - pl.col('timestamp').shift(1))
            .alias('gap_duration')
        ])

        # Find rows where gap > threshold
        gap_threshold = timedelta(minutes=self.max_gap_minutes)

        for row in df.iter_rows(named=True):
            gap_dur = row.get('gap_duration')

            if gap_dur and gap_dur > gap_threshold:
                # Calculate missing candles (assuming 1-minute candles)
                minutes_missing = gap_dur.total_seconds() / 60
                candles_missing = int(minutes_missing) - 1

                gap = DataGap(
                    start_time=row['timestamp'] - gap_dur,
                    end_time=row['timestamp'],
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

    def fill_gaps(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Fill gaps using forward-fill strategy.

        Args:
            df: DataFrame with gaps

        Returns:
            DataFrame with gaps filled
        """
        if not self.forward_fill:
            return df

        # Forward-fill all columns except timestamp
        fill_columns = [col for col in df.columns if col != 'timestamp']

        for col in fill_columns:
            df = df.with_columns([
                pl.col(col).forward_fill().alias(col)
            ])

        logger.info(
            "gaps_filled",
            method="forward_fill",
            columns=len(fill_columns)
        )

        return df

    def process(self, df: pl.DataFrame) -> tuple[pl.DataFrame, List[DataGap]]:
        """
        Detect and fill gaps.

        Returns:
            (processed_dataframe, detected_gaps)
        """
        # Ensure sorted by timestamp
        df = df.sort('timestamp')

        # Detect gaps
        gaps = self.detect_gaps(df)

        # Fill if configured
        if self.forward_fill and gaps:
            df = self.fill_gaps(df)

        logger.info(
            "gap_processing_complete",
            total_gaps=len(gaps),
            filled=self.forward_fill
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
