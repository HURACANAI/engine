"""
Execution Feedback Loop

Captures trade execution results from Hamilton and feeds them back into Engine training.

Critical for:
- Learning from live performance
- Slippage model calibration
- Regime-specific performance tracking
- Model quality assessment

Flow:
    Hamilton executes trades → writes to execution_results.parquet
    ↓
    Engine reads on startup → processes feedback
    ↓
    Feeds into: RL rewards, slippage model, performance tracking

Usage:
    from src.cloud.training.integrations.feedback import ExecutionFeedbackCollector, FeedbackProcessor

    # In Hamilton: Collect execution results
    collector = ExecutionFeedbackCollector(output_path="execution_results.parquet")
    collector.record_trade(
        model_id="btc_v47",
        signal_time=signal_ts,
        entry_time=entry_ts,
        exit_time=exit_ts,
        expected_pnl_bps=20,
        actual_pnl_bps=18,
        expected_slippage_bps=2,
        actual_slippage_bps=3.5
    )

    # In Engine: Process feedback
    processor = FeedbackProcessor()
    feedback = processor.load_feedback("execution_results.parquet")
    processor.integrate_into_training(feedback)
"""

from .collector import ExecutionFeedbackCollector, ExecutionResult
from .processor import FeedbackProcessor, FeedbackStats
from .schema import EXECUTION_RESULT_SCHEMA

__all__ = [
    "ExecutionFeedbackCollector",
    "ExecutionResult",
    "FeedbackProcessor",
    "FeedbackStats",
    "EXECUTION_RESULT_SCHEMA",
]
