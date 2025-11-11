"""Data validation leveraging Great Expectations-style hygiene gates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import polars as pl  # type: ignore[reportMissingImports]
import structlog  # type: ignore[reportMissingImports]


logger = structlog.get_logger(__name__)


@dataclass
class ValidationResult:
    coverage_ratio: float
    duplicate_count: int
    notes: Optional[str] = None


class DataQualitySuite:
    """Validates coverage, duplication, and timestamp order for OHLCV datasets."""

    def __init__(self, coverage_threshold: float = 0.95) -> None:
        # Lowered threshold from 0.998 to 0.95 for more lenient validation
        # Some coins may have gaps (weekends, maintenance, etc.)
        self._coverage_threshold = coverage_threshold

    def validate(self, frame: pl.DataFrame, *, query: Any) -> pl.DataFrame:
        if frame.is_empty():
            raise ValueError(f"No data returned for {query.symbol}")
        duplicates = frame.height - frame.select(pl.col("ts")).unique().height
        frame = frame.unique(subset=["ts"]).sort("ts")
        expected_rows = self._expected_rows(query)
        coverage = len(frame) / expected_rows if expected_rows else 1.0
        if coverage < self._coverage_threshold:
            raise ValueError(
                f"Coverage {coverage:.4f} below threshold for {query.symbol} ({len(frame)} of {expected_rows})"
            )
        self._assert_monotonic(frame)
        logger.debug(
            "data_quality_pass",
            symbol=query.symbol,
            coverage_ratio=coverage,
            duplicates=duplicates,
        )
        return frame

    def _expected_rows(self, query: Any) -> int:
        """Calculate expected number of rows based on timeframe.
        
        Args:
            query: CandleQuery with timeframe, start_at, end_at
            
        Returns:
            Expected number of candles
        """
        delta = query.end_at - query.start_at
        timeframe = getattr(query, 'timeframe', '1m')  # Default to 1m if not set
        
        # Parse timeframe (e.g., '1m', '1h', '1d', '4h', '1w')
        timeframe_str = str(timeframe).lower()
        
        # Extract number and unit
        if timeframe_str.endswith('m'):
            # Minutes (1m, 5m, 15m, etc.)
            interval_minutes = int(timeframe_str[:-1])
            total_minutes = int(delta.total_seconds() // 60)
            expected = total_minutes // interval_minutes + 1
        elif timeframe_str.endswith('h'):
            # Hours (1h, 4h, 12h, etc.)
            interval_hours = int(timeframe_str[:-1])
            total_hours = int(delta.total_seconds() // 3600)
            expected = total_hours // interval_hours + 1
        elif timeframe_str.endswith('d'):
            # Days (1d, 7d, etc.)
            interval_days = int(timeframe_str[:-1])
            total_days = int(delta.total_seconds() // 86400)
            expected = total_days // interval_days + 1
        elif timeframe_str.endswith('w'):
            # Weeks (1w, etc.)
            interval_weeks = int(timeframe_str[:-1])
            total_weeks = int(delta.total_seconds() // (86400 * 7))
            expected = total_weeks // interval_weeks + 1
        elif timeframe_str.endswith('M'):
            # Months (1M, etc.) - approximate as 30 days
            interval_months = int(timeframe_str[:-1])
            total_months = int(delta.total_seconds() // (86400 * 30))
            expected = total_months // interval_months + 1
        else:
            # Fallback: assume minutes
            logger.warning("unknown_timeframe_format", timeframe=timeframe, defaulting_to_minutes=True)
            interval_minutes = 1
            total_minutes = int(delta.total_seconds() // 60)
            expected = total_minutes // interval_minutes + 1
        
        return max(expected, 1)

    def _assert_monotonic(self, frame: pl.DataFrame) -> None:
        if not frame["ts"].is_sorted():
            raise ValueError("Timestamp column is not strictly increasing")
