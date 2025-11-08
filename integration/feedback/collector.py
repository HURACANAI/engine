"""
Execution Feedback Collector

Collects trade execution results from Hamilton (live trading).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import polars as pl
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class ExecutionResult:
    """
    Single trade execution result

    Captures everything needed to learn from live performance.
    """
    # Identity
    trade_id: str
    model_id: str
    symbol: str

    # Timing
    signal_time: datetime
    entry_time: datetime
    exit_time: datetime
    duration_seconds: float

    # Signal details
    signal_direction: str  # long, short
    signal_confidence: float

    # Expected vs Actual PnL
    expected_pnl_bps: float
    actual_pnl_bps: float
    pnl_error_bps: float  # actual - expected

    # Slippage
    expected_slippage_bps: float
    actual_slippage_bps: float
    slippage_error_bps: float  # actual - expected

    # Execution quality
    entry_fill_ratio: float  # 0-1, 1.0 = fully filled
    exit_fill_ratio: float
    timing_delay_ms: float  # Signal to entry delay

    # Market context
    regime_at_entry: str
    regime_at_exit: str
    volatility_at_entry: Optional[float] = None
    spread_at_entry_bps: Optional[float] = None

    # Fees
    total_fees_bps: Optional[float] = None

    # Metadata
    metadata: Optional[dict] = None


class ExecutionFeedbackCollector:
    """
    Execution Feedback Collector

    Used by Hamilton to record trade execution results.

    Example (in Hamilton):
        collector = ExecutionFeedbackCollector(
            output_path="data/execution_results.parquet"
        )

        # After trade closes
        collector.record_trade(
            trade_id="trade_xyz",
            model_id="btc_trend_v47",
            symbol="BTC",
            signal_time=signal_ts,
            entry_time=entry_ts,
            exit_time=exit_ts,
            expected_pnl_bps=20,
            actual_pnl_bps=18,
            expected_slippage_bps=2,
            actual_slippage_bps=3.5,
            ...
        )

        # Flush to disk
        collector.save()
    """

    def __init__(self, output_path: str | Path):
        """
        Initialize collector

        Args:
            output_path: Path to parquet file (will append)
        """
        self.output_path = Path(output_path)
        self.results: list[ExecutionResult] = []

        # Load existing results if file exists
        if self.output_path.exists():
            try:
                existing_df = pl.read_parquet(self.output_path)
                logger.info(
                    "loaded_existing_results",
                    path=str(self.output_path),
                    num_results=len(existing_df)
                )
            except Exception as e:
                logger.warning(
                    "failed_to_load_existing_results",
                    error=str(e)
                )

    def record_trade(
        self,
        trade_id: str,
        model_id: str,
        symbol: str,
        signal_time: datetime,
        entry_time: datetime,
        exit_time: datetime,
        signal_direction: str,
        signal_confidence: float,
        expected_pnl_bps: float,
        actual_pnl_bps: float,
        expected_slippage_bps: float,
        actual_slippage_bps: float,
        entry_fill_ratio: float = 1.0,
        exit_fill_ratio: float = 1.0,
        timing_delay_ms: float = 0.0,
        regime_at_entry: str = "unknown",
        regime_at_exit: str = "unknown",
        volatility_at_entry: Optional[float] = None,
        spread_at_entry_bps: Optional[float] = None,
        total_fees_bps: Optional[float] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        """
        Record a trade execution result

        Args:
            trade_id: Unique trade identifier
            model_id: Model that generated signal
            symbol: Trading symbol
            signal_time: When signal was generated
            entry_time: When trade was entered
            exit_time: When trade was exited
            signal_direction: "long" or "short"
            signal_confidence: Model confidence [0-1]
            expected_pnl_bps: Expected PnL in bps
            actual_pnl_bps: Actual realized PnL in bps
            expected_slippage_bps: Expected slippage
            actual_slippage_bps: Actual slippage
            entry_fill_ratio: Entry fill ratio [0-1]
            exit_fill_ratio: Exit fill ratio [0-1]
            timing_delay_ms: Signal to entry delay (ms)
            regime_at_entry: Market regime at entry
            regime_at_exit: Market regime at exit
            volatility_at_entry: Volatility at entry
            spread_at_entry_bps: Spread at entry (bps)
            total_fees_bps: Total fees in bps
            metadata: Optional metadata dict
        """
        # Calculate derived fields
        duration_seconds = (exit_time - entry_time).total_seconds()
        pnl_error_bps = actual_pnl_bps - expected_pnl_bps
        slippage_error_bps = actual_slippage_bps - expected_slippage_bps

        result = ExecutionResult(
            trade_id=trade_id,
            model_id=model_id,
            symbol=symbol,
            signal_time=signal_time,
            entry_time=entry_time,
            exit_time=exit_time,
            duration_seconds=duration_seconds,
            signal_direction=signal_direction,
            signal_confidence=signal_confidence,
            expected_pnl_bps=expected_pnl_bps,
            actual_pnl_bps=actual_pnl_bps,
            pnl_error_bps=pnl_error_bps,
            expected_slippage_bps=expected_slippage_bps,
            actual_slippage_bps=actual_slippage_bps,
            slippage_error_bps=slippage_error_bps,
            entry_fill_ratio=entry_fill_ratio,
            exit_fill_ratio=exit_fill_ratio,
            timing_delay_ms=timing_delay_ms,
            regime_at_entry=regime_at_entry,
            regime_at_exit=regime_at_exit,
            volatility_at_entry=volatility_at_entry,
            spread_at_entry_bps=spread_at_entry_bps,
            total_fees_bps=total_fees_bps,
            metadata=metadata
        )

        self.results.append(result)

        logger.info(
            "trade_recorded",
            trade_id=trade_id,
            model_id=model_id,
            symbol=symbol,
            actual_pnl_bps=actual_pnl_bps,
            slippage_error_bps=slippage_error_bps
        )

    def save(self) -> None:
        """
        Save results to parquet file

        Appends to existing file if present.
        """
        if not self.results:
            logger.warning("no_results_to_save")
            return

        # Convert to DataFrame
        records = [asdict(r) for r in self.results]
        new_df = pl.DataFrame(records)

        # Append to existing file
        if self.output_path.exists():
            try:
                existing_df = pl.read_parquet(self.output_path)
                combined_df = pl.concat([existing_df, new_df])
            except Exception as e:
                logger.warning("failed_to_load_existing", error=str(e))
                combined_df = new_df
        else:
            combined_df = new_df

        # Ensure parent directory exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write to parquet
        combined_df.write_parquet(self.output_path)

        logger.info(
            "results_saved",
            path=str(self.output_path),
            num_new=len(self.results),
            num_total=len(combined_df)
        )

        # Clear in-memory results
        self.results = []

    def get_stats(self) -> dict:
        """
        Get summary statistics

        Returns:
            Dict of stats
        """
        if not self.results:
            return {}

        df = pd.DataFrame([asdict(r) for r in self.results])

        return {
            "num_trades": len(df),
            "avg_pnl_bps": df['actual_pnl_bps'].mean(),
            "avg_pnl_error_bps": df['pnl_error_bps'].mean(),
            "avg_slippage_bps": df['actual_slippage_bps'].mean(),
            "avg_slippage_error_bps": df['slippage_error_bps'].mean(),
            "avg_timing_delay_ms": df['timing_delay_ms'].mean(),
            "win_rate": (df['actual_pnl_bps'] > 0).mean(),
        }
