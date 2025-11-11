"""
V2 Pipeline Integration Adapter

Bridges the V2 data pipeline (triple-barrier labeling, multi-window training)
with the existing RLTrainingPipeline.

This allows gradual migration from naive labeling to production-grade labeling
without breaking existing systems.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional

import polars as pl
import structlog

from ...engine.data_quality import DataSanityPipeline
from ...engine.data_quality.sanity_pipeline import format_sanity_report
from ...engine.labeling import (
    TripleBarrierLabeler,
    MetaLabeler,
    ScalpLabelConfig,
    RunnerLabelConfig,
    LabeledTrade
)
from ...engine.costs import CostEstimator
from ...engine.weighting import RecencyWeighter, create_mode_specific_weighter
from ...engine.walk_forward import WalkForwardValidator
from ..services.costs import CostModel, CostBreakdown

logger = structlog.get_logger(__name__)


@dataclass
class V2PipelineConfig:
    """Configuration for V2 pipeline adapter."""

    # Data quality
    enable_data_quality: bool = True
    outlier_threshold_pct: float = 0.10
    max_gap_minutes: int = 5

    # Labeling mode
    trading_mode: str = 'scalp'  # 'scalp' or 'runner'

    # Scalp settings
    scalp_tp_bps: float = 15.0
    scalp_sl_bps: float = 10.0
    scalp_timeout_minutes: int = 30

    # Runner settings
    runner_tp_bps: float = 80.0
    runner_sl_bps: float = 40.0
    runner_timeout_minutes: int = 10080  # 7 days

    # Meta-labeling
    enable_meta_labeling: bool = True
    meta_cost_threshold_bps: float = 5.0

    # Recency weighting
    enable_recency_weighting: bool = True
    recency_halflife_days: Optional[float] = None  # Auto from mode if None

    # Walk-forward validation
    enable_walk_forward: bool = False  # Optional validation
    walk_forward_train_days: int = 30
    walk_forward_test_days: int = 1
    embargo_minutes: int = 30

    # Performance
    max_labels: Optional[int] = 10000  # Limit labels for speed


class V2PipelineAdapter:
    """
    Adapter that wraps V2 pipeline for use in existing training flow.

    Usage in RLTrainingPipeline:
        adapter = V2PipelineAdapter(config=V2PipelineConfig())

        # Replace naive labeling:
        # OLD: labels = naive_label_builder.build(data)
        # NEW: labeled_trades, weights = adapter.process(data, symbol='BTC/USDT')

        # Use labeled_trades for training
        for trade in labeled_trades:
            agent.add_experience(trade, weight=weights[i])
    """

    def __init__(
        self,
        config: Optional[V2PipelineConfig] = None,
        exchange: str = 'binance'
    ):
        """
        Initialize V2 pipeline adapter.

        Args:
            config: Pipeline configuration
            exchange: Exchange name for fee schedules
        """
        self.config = config or V2PipelineConfig()
        self.exchange = exchange

        # Initialize components
        if self.config.enable_data_quality:
            self.sanity_pipeline = DataSanityPipeline(
                exchange=exchange,
                outlier_threshold_pct=self.config.outlier_threshold_pct,
                max_gap_minutes=self.config.max_gap_minutes
            )
        else:
            self.sanity_pipeline = None

        # Create cost estimator
        self.cost_estimator = CostEstimator(exchange=exchange)

        # Create labeler based on mode
        if self.config.trading_mode == 'scalp':
            label_config = ScalpLabelConfig(
                tp_bps=self.config.scalp_tp_bps,
                sl_bps=self.config.scalp_sl_bps,
                timeout_minutes=self.config.scalp_timeout_minutes
            )
        else:  # runner
            label_config = RunnerLabelConfig(
                tp_bps=self.config.runner_tp_bps,
                sl_bps=self.config.runner_sl_bps,
                timeout_minutes=self.config.runner_timeout_minutes
            )

        self.labeler = TripleBarrierLabeler(
            config=label_config,
            cost_estimator=self.cost_estimator
        )

        # Create meta-labeler
        if self.config.enable_meta_labeling:
            self.meta_labeler = MetaLabeler(
                cost_threshold_bps=self.config.meta_cost_threshold_bps,
                min_pnl_bps=0.0
            )
        else:
            self.meta_labeler = None

        # Create recency weighter
        if self.config.enable_recency_weighting:
            if self.config.recency_halflife_days is not None:
                self.weighter = RecencyWeighter(
                    halflife_days=self.config.recency_halflife_days
                )
            else:
                self.weighter = create_mode_specific_weighter(
                    mode=self.config.trading_mode
                )
        else:
            self.weighter = None

        # Create validator
        if self.config.enable_walk_forward:
            self.validator = WalkForwardValidator(
                train_days=self.config.walk_forward_train_days,
                test_days=self.config.walk_forward_test_days,
                embargo_minutes=self.config.embargo_minutes
            )
        else:
            self.validator = None

        logger.info(
            "v2_pipeline_adapter_initialized",
            mode=self.config.trading_mode,
            exchange=exchange,
            data_quality=self.config.enable_data_quality,
            meta_labeling=self.config.enable_meta_labeling,
            recency_weighting=self.config.enable_recency_weighting
        )

    def process(
        self,
        data: pl.DataFrame,
        symbol: str
    ) -> tuple[List[LabeledTrade], Optional[list]]:
        """
        Process raw candle data through V2 pipeline.

        Args:
            data: Raw OHLCV data (with 'ts' or 'timestamp' column)
            symbol: Trading symbol (e.g., 'BTC/USDT')

        Returns:
            Tuple of (labeled_trades, weights)
            - labeled_trades: List of LabeledTrade objects
            - weights: Sample weights (None if recency weighting disabled)

        Example:
            labeled_trades, weights = adapter.process(
                data=candles_df,
                symbol='BTC/USDT'
            )

            # Use in training
            for i, trade in enumerate(labeled_trades):
                weight = weights[i] if weights else 1.0
                agent.add_experience(trade, weight=weight)
        """
        logger.info(
            "v2_pipeline_processing",
            symbol=symbol,
            rows=len(data),
            mode=self.config.trading_mode
        )

        # Step 1: Data quality pipeline
        if self.sanity_pipeline:
            logger.debug("running_data_quality_pipeline", symbol=symbol)
            clean_data, sanity_report = self.sanity_pipeline.clean(data)

            logger.info(
                "data_quality_complete",
                symbol=symbol,
                original_rows=sanity_report.original_rows,
                cleaned_rows=sanity_report.cleaned_rows,
                duplicates=sanity_report.duplicates_removed,
                outliers=sanity_report.outliers_removed,
                gaps_filled=sanity_report.gaps_filled
            )
        else:
            clean_data = data

        # Step 2: Triple-barrier labeling
        logger.debug("triple_barrier_labeling", symbol=symbol)
        labeled_trades = self.labeler.label_dataframe(
            df=clean_data,
            symbol=symbol,
            max_labels=self.config.max_labels
        )

        if not labeled_trades:
            logger.warning("no_labeled_trades", symbol=symbol)
            return [], None

        logger.info(
            "labeling_complete",
            symbol=symbol,
            total_labels=len(labeled_trades)
        )

        # Step 3: Meta-labeling (filter to profitable)
        if self.meta_labeler:
            logger.debug("applying_meta_labeling", symbol=symbol)
            labeled_trades = self.meta_labeler.apply(labeled_trades)

            profitable = sum(1 for t in labeled_trades if t.meta_label == 1)
            logger.info(
                "meta_labeling_complete",
                symbol=symbol,
                total_trades=len(labeled_trades),
                profitable_trades=profitable,
                profitable_pct=profitable / len(labeled_trades) * 100 if labeled_trades else 0
            )

        # Step 4: Recency weighting
        weights = None
        if self.weighter:
            logger.debug("calculating_recency_weights", symbol=symbol)
            weights = self.weighter.calculate_weights_from_labels(labeled_trades)

            ess = self.weighter.get_effective_sample_size(weights)
            logger.info(
                "recency_weighting_complete",
                symbol=symbol,
                total_samples=len(weights),
                effective_samples=ess,
                weight_range=(weights.min(), weights.max())
            )

        return labeled_trades, weights

    def validate_trades(
        self,
        labeled_trades: List[LabeledTrade]
    ) -> dict:
        """
        Optional: Run walk-forward validation on labeled trades.

        Args:
            labeled_trades: List of labeled trades

        Returns:
            Validation results dictionary
        """
        if not self.validator:
            logger.warning("walk_forward_validator_not_enabled")
            return {"error": "validator_not_enabled"}

        logger.info("running_walk_forward_validation", trades=len(labeled_trades))

        results = self.validator.validate_with_labels(labeled_trades)

        logger.info(
            "walk_forward_validation_complete",
            windows=len(results.windows),
            train_sharpe=results.overall_train_sharpe,
            test_sharpe=results.overall_test_sharpe
        )

        return {
            'windows': len(results.windows),
            'train_sharpe': results.overall_train_sharpe,
            'test_sharpe': results.overall_test_sharpe,
            'train_win_rate': results.overall_train_win_rate,
            'test_win_rate': results.overall_test_win_rate,
            'overfitting_detected': results.overall_test_sharpe < results.overall_train_sharpe * 0.7
        }

    def convert_to_legacy_format(
        self,
        labeled_trades: List[LabeledTrade],
        weights: Optional[list] = None
    ) -> pl.DataFrame:
        """
        Convert labeled trades to legacy DataFrame format for compatibility.

        This allows using V2 labels with existing code that expects DataFrames.

        Args:
            labeled_trades: List of LabeledTrade objects
            weights: Optional sample weights

        Returns:
            DataFrame with legacy columns
        """
        if not labeled_trades:
            return pl.DataFrame()

        records = []
        for i, trade in enumerate(labeled_trades):
            records.append({
                'timestamp': trade.entry_time,
                'entry_price': trade.entry_price,
                'exit_price': trade.exit_price,
                'exit_time': trade.exit_time,
                'exit_reason': trade.exit_reason,
                'duration_minutes': trade.duration_minutes,
                'gross_pnl_bps': trade.pnl_gross_bps,
                'costs_bps': trade.costs_bps,
                'net_pnl_bps': trade.pnl_net_bps,
                'meta_label': trade.meta_label,
                'weight': weights[i] if weights else 1.0,
                'is_profitable': trade.meta_label == 1
            })

        df = pl.DataFrame(records)

        logger.debug(
            "converted_to_legacy_format",
            rows=len(df)
        )

        return df

    def get_cost_breakdown(
        self,
        entry_row: pl.DataFrame,
        exit_time: datetime,
        duration_minutes: int,
        mode: str = 'scalp'
    ) -> CostBreakdown:
        """
        Get detailed cost breakdown compatible with existing CostModel.

        Args:
            entry_row: Entry candle row
            exit_time: Exit timestamp
            duration_minutes: Trade duration
            mode: 'scalp' or 'runner'

        Returns:
            CostBreakdown compatible with existing training code
        """
        # Use V2 cost estimator
        tca_report = self.cost_estimator.estimate_detailed(
            entry_row=entry_row,
            exit_time=exit_time,
            duration_minutes=duration_minutes,
            mode=mode
        )

        # Convert to legacy CostBreakdown
        cost_breakdown = CostBreakdown(
            maker_fee_bps=tca_report.fee_entry_bps,
            taker_fee_bps=tca_report.fee_exit_bps,
            spread_bps=tca_report.spread_bps,
            slippage_bps=tca_report.slippage_total_bps,
            total_costs_bps=tca_report.total_cost_bps
        )

        return cost_breakdown


def create_v2_scalp_adapter(exchange: str = 'binance') -> V2PipelineAdapter:
    """
    Create V2 adapter configured for scalp trading.

    Returns:
        Configured V2PipelineAdapter for scalp mode
    """
    config = V2PipelineConfig(
        trading_mode='scalp',
        scalp_tp_bps=15.0,
        scalp_sl_bps=10.0,
        scalp_timeout_minutes=30,
        enable_data_quality=True,
        enable_meta_labeling=True,
        enable_recency_weighting=True,
        recency_halflife_days=10.0  # Scalp mode: 10-day halflife
    )

    return V2PipelineAdapter(config=config, exchange=exchange)


def create_v2_runner_adapter(exchange: str = 'binance') -> V2PipelineAdapter:
    """
    Create V2 adapter configured for runner trading.

    Returns:
        Configured V2PipelineAdapter for runner mode
    """
    config = V2PipelineConfig(
        trading_mode='runner',
        runner_tp_bps=80.0,
        runner_sl_bps=40.0,
        runner_timeout_minutes=10080,  # 7 days
        enable_data_quality=True,
        enable_meta_labeling=True,
        enable_recency_weighting=True,
        recency_halflife_days=20.0  # Runner mode: 20-day halflife
    )

    return V2PipelineAdapter(config=config, exchange=exchange)
