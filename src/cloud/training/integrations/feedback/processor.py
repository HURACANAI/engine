"""
Execution Feedback Processor

Processes feedback from Hamilton and integrates into Engine training.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import polars as pl
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class FeedbackStats:
    """Aggregated feedback statistics"""
    num_trades: int
    avg_pnl_bps: float
    avg_slippage_error_bps: float
    avg_timing_delay_ms: float
    win_rate: float

    # By model
    model_stats: Dict[str, dict]

    # By regime
    regime_stats: Dict[str, dict]

    # By symbol
    symbol_stats: Dict[str, dict]


class FeedbackProcessor:
    """
    Execution Feedback Processor

    Loads feedback from Hamilton and processes for Engine training.

    Example (in Engine):
        processor = FeedbackProcessor()

        # Load feedback
        feedback_df = processor.load_feedback("data/execution_results.parquet")

        # Get statistics
        stats = processor.compute_stats(feedback_df)

        # Integrate into training
        processor.update_slippage_model(feedback_df)
        processor.update_model_performance(feedback_df)
        processor.enrich_training_rewards(feedback_df)
    """

    def __init__(self):
        """Initialize processor"""
        pass

    def load_feedback(
        self,
        path: str | Path,
        days_back: Optional[int] = None,
        model_id: Optional[str] = None
    ) -> pl.DataFrame:
        """
        Load execution feedback

        Args:
            path: Path to execution_results.parquet
            days_back: Only load last N days (None = all)
            model_id: Filter by model ID (None = all)

        Returns:
            Polars DataFrame with feedback
        """
        path = Path(path)

        if not path.exists():
            logger.warning("feedback_file_not_found", path=str(path))
            return pl.DataFrame()

        df = pl.read_parquet(path)

        # Filter by date if requested
        if days_back is not None:
            cutoff = datetime.now() - timedelta(days=days_back)
            df = df.filter(pl.col("exit_time") >= cutoff)

        # Filter by model if requested
        if model_id is not None:
            df = df.filter(pl.col("model_id") == model_id)

        logger.info(
            "feedback_loaded",
            path=str(path),
            num_trades=len(df),
            days_back=days_back,
            model_id=model_id
        )

        return df

    def compute_stats(self, feedback_df: pl.DataFrame) -> FeedbackStats:
        """
        Compute aggregated statistics

        Args:
            feedback_df: Feedback DataFrame

        Returns:
            FeedbackStats
        """
        if len(feedback_df) == 0:
            return FeedbackStats(
                num_trades=0,
                avg_pnl_bps=0.0,
                avg_slippage_error_bps=0.0,
                avg_timing_delay_ms=0.0,
                win_rate=0.0,
                model_stats={},
                regime_stats={},
                symbol_stats={}
            )

        # Overall stats
        num_trades = len(feedback_df)
        avg_pnl_bps = feedback_df['actual_pnl_bps'].mean()
        avg_slippage_error_bps = feedback_df['slippage_error_bps'].mean()
        avg_timing_delay_ms = feedback_df['timing_delay_ms'].mean()
        win_rate = (feedback_df['actual_pnl_bps'] > 0).sum() / num_trades

        # Stats by model
        model_stats = {}
        for model_id in feedback_df['model_id'].unique():
            model_df = feedback_df.filter(pl.col("model_id") == model_id)
            model_stats[model_id] = {
                "num_trades": len(model_df),
                "avg_pnl_bps": model_df['actual_pnl_bps'].mean(),
                "win_rate": (model_df['actual_pnl_bps'] > 0).sum() / len(model_df),
                "avg_slippage_error": model_df['slippage_error_bps'].mean()
            }

        # Stats by regime
        regime_stats = {}
        for regime in feedback_df['regime_at_entry'].unique():
            regime_df = feedback_df.filter(pl.col("regime_at_entry") == regime)
            regime_stats[regime] = {
                "num_trades": len(regime_df),
                "avg_pnl_bps": regime_df['actual_pnl_bps'].mean(),
                "win_rate": (regime_df['actual_pnl_bps'] > 0).sum() / len(regime_df)
            }

        # Stats by symbol
        symbol_stats = {}
        for symbol in feedback_df['symbol'].unique():
            symbol_df = feedback_df.filter(pl.col("symbol") == symbol)
            symbol_stats[symbol] = {
                "num_trades": len(symbol_df),
                "avg_pnl_bps": symbol_df['actual_pnl_bps'].mean(),
                "win_rate": (symbol_df['actual_pnl_bps'] > 0).sum() / len(symbol_df)
            }

        return FeedbackStats(
            num_trades=num_trades,
            avg_pnl_bps=avg_pnl_bps,
            avg_slippage_error_bps=avg_slippage_error_bps,
            avg_timing_delay_ms=avg_timing_delay_ms,
            win_rate=win_rate,
            model_stats=model_stats,
            regime_stats=regime_stats,
            symbol_stats=symbol_stats
        )

    def update_slippage_model(
        self,
        feedback_df: pl.DataFrame
    ) -> Dict[str, float]:
        """
        Update slippage model from feedback

        Args:
            feedback_df: Feedback DataFrame

        Returns:
            Dict of slippage parameters by regime
        """
        if len(feedback_df) == 0:
            logger.warning("no_feedback_for_slippage_update")
            return {}

        # Calculate slippage by regime
        slippage_params = {}

        for regime in feedback_df['regime_at_entry'].unique():
            regime_df = feedback_df.filter(pl.col("regime_at_entry") == regime)

            slippage_params[regime] = {
                "mean_slippage_bps": regime_df['actual_slippage_bps'].mean(),
                "std_slippage_bps": regime_df['actual_slippage_bps'].std(),
                "mean_error_bps": regime_df['slippage_error_bps'].mean(),
                "num_samples": len(regime_df)
            }

        logger.info(
            "slippage_model_updated",
            regimes=list(slippage_params.keys()),
            total_samples=len(feedback_df)
        )

        return slippage_params

    def update_model_performance(
        self,
        feedback_df: pl.DataFrame,
        registry_dsn: Optional[str] = None
    ) -> None:
        """
        Update model performance tracking in registry

        Args:
            feedback_df: Feedback DataFrame
            registry_dsn: PostgreSQL DSN for model registry
        """
        if len(feedback_df) == 0:
            logger.warning("no_feedback_for_performance_update")
            return

        if registry_dsn is None:
            logger.warning("no_registry_dsn_provided_skipping_update")
            return

        from src.shared.model_registry import UnifiedModelRegistry

        registry = UnifiedModelRegistry(dsn=registry_dsn)

        # Group by model
        for model_id in feedback_df['model_id'].unique():
            model_df = feedback_df.filter(pl.col("model_id") == model_id)

            # Calculate live performance
            live_trades = len(model_df)
            live_pnl_gbp = model_df['actual_pnl_bps'].sum() * 0.01  # Approximate
            live_win_rate = (model_df['actual_pnl_bps'] > 0).sum() / live_trades
            avg_slippage = model_df['actual_slippage_bps'].mean()

            # Record in registry
            try:
                registry.record_performance(
                    model_id=model_id,
                    live_trades=live_trades,
                    live_pnl_gbp=live_pnl_gbp,
                    live_win_rate=live_win_rate,
                    avg_slippage_bps=avg_slippage
                )

                logger.info(
                    "model_performance_updated",
                    model_id=model_id,
                    live_trades=live_trades,
                    live_pnl_gbp=live_pnl_gbp
                )
            except Exception as e:
                logger.error(
                    "failed_to_update_model_performance",
                    model_id=model_id,
                    error=str(e)
                )

        registry.close()

    def enrich_training_rewards(
        self,
        feedback_df: pl.DataFrame,
        symbol: str
    ) -> Dict[str, float]:
        """
        Enrich RL training rewards with live feedback

        Args:
            feedback_df: Feedback DataFrame
            symbol: Symbol to enrich

        Returns:
            Dict of reward adjustments
        """
        if len(feedback_df) == 0:
            return {}

        # Filter to symbol
        symbol_df = feedback_df.filter(pl.col("symbol") == symbol)

        if len(symbol_df) == 0:
            return {}

        # Calculate reward adjustments based on live performance
        reward_adjustments = {
            "live_pnl_bonus": symbol_df['actual_pnl_bps'].mean() * 0.1,
            "slippage_penalty": -abs(symbol_df['slippage_error_bps'].mean()) * 0.5,
            "timing_penalty": -symbol_df['timing_delay_ms'].mean() * 0.001,
        }

        logger.info(
            "training_rewards_enriched",
            symbol=symbol,
            num_trades=len(symbol_df),
            adjustments=reward_adjustments
        )

        return reward_adjustments

    def generate_feedback_report(
        self,
        feedback_df: pl.DataFrame,
        output_path: Optional[Path] = None
    ) -> str:
        """
        Generate human-readable feedback report

        Args:
            feedback_df: Feedback DataFrame
            output_path: Optional path to save report

        Returns:
            Report string
        """
        stats = self.compute_stats(feedback_df)

        report_lines = [
            "=" * 80,
            "EXECUTION FEEDBACK REPORT",
            "=" * 80,
            "",
            f"Total Trades: {stats.num_trades}",
            f"Average PnL: {stats.avg_pnl_bps:.2f} bps",
            f"Win Rate: {stats.win_rate * 100:.1f}%",
            f"Avg Slippage Error: {stats.avg_slippage_error_bps:.2f} bps",
            f"Avg Timing Delay: {stats.avg_timing_delay_ms:.1f} ms",
            "",
            "BY MODEL:",
            "-" * 80,
        ]

        for model_id, model_stat in stats.model_stats.items():
            report_lines.append(
                f"  {model_id}: {model_stat['num_trades']} trades, "
                f"{model_stat['avg_pnl_bps']:.2f} bps avg, "
                f"{model_stat['win_rate']*100:.1f}% win rate"
            )

        report_lines.extend([
            "",
            "BY REGIME:",
            "-" * 80,
        ])

        for regime, regime_stat in stats.regime_stats.items():
            report_lines.append(
                f"  {regime}: {regime_stat['num_trades']} trades, "
                f"{regime_stat['avg_pnl_bps']:.2f} bps avg, "
                f"{regime_stat['win_rate']*100:.1f}% win rate"
            )

        report_lines.append("=" * 80)

        report = "\n".join(report_lines)

        if output_path:
            output_path.write_text(report)
            logger.info("feedback_report_saved", path=str(output_path))

        return report
