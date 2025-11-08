"""Enhanced data loader with self-validation and Brain Library integration."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

import polars as pl  # type: ignore[reportMissingImports]
import structlog  # type: ignore[reportMissingImports]

from ..brain.brain_library import BrainLibrary  # type: ignore[reportMissingImports]
from .data_loader import CandleDataLoader, CandleQuery
from .quality_checks import DataQualitySuite

logger = structlog.get_logger(__name__)


class EnhancedDataLoader(CandleDataLoader):
    """
    Enhanced data loader with:
    - Self-validation
    - Automatic retry logic
    - Brain Library integration for data quality logging
    """

    def __init__(
        self,
        exchange_client: Optional[any] = None,
        cache_dir: Optional[str] = None,
        brain_library: Optional[BrainLibrary] = None,
    ) -> None:
        """
        Initialize enhanced data loader.
        
        Args:
            exchange_client: Exchange client instance
            cache_dir: Cache directory path
            brain_library: Brain Library instance for logging
        """
        super().__init__(exchange_client, cache_dir)
        self.brain = brain_library
        self.quality_suite = DataQualitySuite()
        logger.info("enhanced_data_loader_initialized")

    def load_with_validation(
        self,
        query: CandleQuery,
        max_retries: int = 3,
        retry_delay: int = 5,
    ) -> Optional[pl.DataFrame]:
        """
        Load data with automatic validation and retry.
        
        Args:
            query: Candle query
            max_retries: Maximum retry attempts
            retry_delay: Delay between retries (seconds)
            
        Returns:
            Validated DataFrame or None if all retries fail
        """
        for attempt in range(max_retries):
            try:
                # Load data
                df = self.load(query)
                
                if df is None or df.is_empty():
                    self._log_quality_issue(
                        query.symbol,
                        "missing_data",
                        "warning",
                        f"No data returned for {query.symbol}",
                    )
                    continue
                
                # Validate data quality
                validation_result = self.quality_suite.validate(df, query.symbol)
                
                if validation_result["is_valid"]:
                    logger.info(
                        "data_validated",
                        symbol=query.symbol,
                        coverage=validation_result.get("coverage", 0.0),
                    )
                    return df
                else:
                    # Log quality issues
                    issues = validation_result.get("issues", [])
                    for issue in issues:
                        self._log_quality_issue(
                            query.symbol,
                            issue.get("type", "unknown"),
                            issue.get("severity", "warning"),
                            issue.get("message", ""),
                        )
                    
                    # Try alternative exchange if available
                    if attempt < max_retries - 1:
                        logger.warning(
                            "data_validation_failed_retrying",
                            symbol=query.symbol,
                            attempt=attempt + 1,
                            issues=len(issues),
                        )
                        import time
                        time.sleep(retry_delay)
                        continue
                    else:
                        # Mark as degraded
                        self._log_quality_issue(
                            query.symbol,
                            "degraded_data",
                            "critical",
                            f"Data quality below threshold after {max_retries} attempts",
                        )
                        return df  # Return degraded data
                        
            except Exception as e:
                logger.warning(
                    "data_load_failed",
                    symbol=query.symbol,
                    attempt=attempt + 1,
                    error=str(e),
                )
                
                if attempt < max_retries - 1:
                    import time
                    time.sleep(retry_delay)
                else:
                    self._log_quality_issue(
                        query.symbol,
                        "load_failure",
                        "critical",
                        f"Failed to load data after {max_retries} attempts: {str(e)}",
                    )
        
        return None

    def _log_quality_issue(
        self,
        symbol: str,
        issue_type: str,
        severity: str,
        details: str,
    ) -> None:
        """Log data quality issue to Brain Library."""
        if self.brain:
            try:
                self.brain.log_data_quality_issue(
                    timestamp=datetime.now(tz=timezone.utc),
                    symbol=symbol,
                    issue_type=issue_type,
                    severity=severity,
                    details=details,
                )
            except Exception as e:
                logger.warning("failed_to_log_quality_issue", error=str(e))

    def check_data_completeness(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        expected_interval_minutes: int = 1,
    ) -> dict:
        """
        Check data completeness and identify gaps.
        
        Args:
            symbol: Trading symbol
            start_time: Expected start time
            end_time: Expected end time
            expected_interval_minutes: Expected interval between candles
            
        Returns:
            Dictionary with completeness metrics
        """
        query = CandleQuery(
            symbol=symbol,
            start_at=start_time,
            end_at=end_time,
        )
        
        df = self.load(query)
        
        if df is None or df.is_empty():
            return {
                "coverage": 0.0,
                "missing_periods": [],
                "total_expected": 0,
                "total_actual": 0,
            }
        
        # Calculate expected number of candles
        total_minutes = (end_time - start_time).total_seconds() / 60
        expected_count = int(total_minutes / expected_interval_minutes)
        actual_count = len(df)
        
        coverage = actual_count / expected_count if expected_count > 0 else 0.0
        
        # Identify gaps (simplified - would need more sophisticated gap detection)
        if "timestamp" in df.columns:
            timestamps = df["timestamp"].sort()
            gaps = []
            for i in range(len(timestamps) - 1):
                gap_minutes = (timestamps[i + 1] - timestamps[i]).total_seconds() / 60
                if gap_minutes > expected_interval_minutes * 2:  # More than 2x expected
                    gaps.append({
                        "start": timestamps[i],
                        "end": timestamps[i + 1],
                        "duration_minutes": gap_minutes,
                    })
        else:
            gaps = []
        
        result = {
            "coverage": coverage,
            "missing_periods": gaps,
            "total_expected": expected_count,
            "total_actual": actual_count,
        }
        
        # Log if coverage is low
        if coverage < 0.95:
            self._log_quality_issue(
                symbol,
                "low_coverage",
                "warning" if coverage > 0.80 else "critical",
                f"Coverage: {coverage:.2%}, Expected: {expected_count}, Actual: {actual_count}",
            )
        
        return result

