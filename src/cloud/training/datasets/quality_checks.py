"""Data validation leveraging Great Expectations-style hygiene gates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import polars as pl
import structlog


logger = structlog.get_logger(__name__)


@dataclass
class ValidationResult:
    coverage_ratio: float
    duplicate_count: int
    notes: Optional[str] = None


class DataQualitySuite:
    """Validates coverage, duplication, and timestamp order for OHLCV datasets."""

    def __init__(self, coverage_threshold: float = 0.998) -> None:
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
        delta = query.end_at - query.start_at
        minutes = int(delta.total_seconds() // 60)
        return max(minutes + 1, 1)

    def _assert_monotonic(self, frame: pl.DataFrame) -> None:
        if not frame["ts"].is_sorted():
            raise ValueError("Timestamp column is not strictly increasing")
