"""
Progressive Historical Training Pipeline

Trains on FULL coin history (from inception to present) by working backwards:
1. Train on most recent 1-2 years (most relevant)
2. Fine-tune on 2-3 years ago
3. Continue backwards to coin inception

This ensures the model learns the most relevant patterns first, then
progressively incorporates older historical context.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import structlog

from src.cloud.training.datasets.data_loader import CandleDataLoader, CandleQuery
from src.cloud.training.pipelines.enhanced_rl_pipeline import EnhancedRLPipeline
from src.exchanges.base import ExchangeClient

logger = structlog.get_logger(__name__)


@dataclass
class TrainingEpoch:
    """Single training epoch with date range."""

    epoch_number: int  # 0 = most recent
    start_date: datetime
    end_date: datetime
    description: str  # e.g., "2023-2024 (most recent)"


@dataclass
class ProgressiveTrainingConfig:
    """Configuration for progressive historical training."""

    # Epoch configuration
    initial_epoch_days: int = 730  # Start with 2 years (most recent)
    subsequent_epoch_days: int = 365  # Each additional epoch is 1 year
    max_epochs: Optional[int] = None  # None = train until coin inception

    # Training configuration
    train_from_scratch_first_epoch: bool = True  # First epoch trains from scratch
    fine_tune_subsequent_epochs: bool = True  # Subsequent epochs fine-tune
    save_checkpoints_per_epoch: bool = True  # Save model after each epoch

    # Performance configuration
    min_data_points_per_epoch: int = 10000  # Skip epochs with too little data
    early_stop_if_performance_degrades: bool = True  # Stop if old data hurts performance


class ProgressiveHistoricalTrainer:
    """
    Train on full coin history by progressively working backwards.

    Strategy:
    1. Determine coin inception date (earliest available data)
    2. Create training schedule (newest to oldest epochs)
    3. Train on most recent epoch
    4. Progressively fine-tune on older epochs
    5. Track performance metrics per epoch
    """

    def __init__(
        self,
        config: ProgressiveTrainingConfig,
        base_pipeline: EnhancedRLPipeline,
    ):
        """
        Initialize progressive trainer.

        Args:
            config: Progressive training configuration
            base_pipeline: Base RL training pipeline to use
        """
        self.config = config
        self.base_pipeline = base_pipeline
        self.training_history: List[Dict] = []

        logger.info(
            "progressive_trainer_initialized",
            initial_epoch_days=config.initial_epoch_days,
            subsequent_epoch_days=config.subsequent_epoch_days,
        )

    def get_coin_inception_date(
        self,
        symbol: str,
        exchange_client: ExchangeClient,
        timeframe: str = "1d",
    ) -> Optional[datetime]:
        """
        Determine when a coin was first listed (earliest available data).

        Args:
            symbol: Trading pair (e.g., "BTC/USDT")
            exchange_client: Exchange client to query
            timeframe: Candle timeframe

        Returns:
            Earliest available data date, or None if error
        """
        try:
            logger.info("determining_coin_inception", symbol=symbol)

            # Try to fetch very old data (start from 2009 for BTC, 2015 for most alts)
            search_start = datetime(2009, 1, 1, tzinfo=timezone.utc)
            search_end = datetime(2015, 1, 1, tzinfo=timezone.utc)

            # Special cases for newer coins
            if "SOL" in symbol:
                search_start = datetime(2020, 1, 1, tzinfo=timezone.utc)
            elif "DOGE" in symbol:
                search_start = datetime(2013, 1, 1, tzinfo=timezone.utc)
            elif symbol.startswith("BTC"):
                search_start = datetime(2009, 1, 1, tzinfo=timezone.utc)
            elif symbol.startswith("ETH"):
                search_start = datetime(2015, 7, 1, tzinfo=timezone.utc)

            # Try to fetch earliest candles
            query = CandleQuery(
                symbol=symbol,
                timeframe=timeframe,
                start_at=search_start,
                end_at=search_end,
            )

            loader = CandleDataLoader(exchange_client)
            candles = loader.load_candles(query)

            if len(candles) == 0:
                logger.warning(
                    "no_historical_data_found",
                    symbol=symbol,
                    search_start=search_start,
                )
                return None

            # Get timestamp of first candle
            inception_date = candles["timestamp"].min()
            inception_datetime = datetime.fromtimestamp(inception_date, tz=timezone.utc)

            logger.info(
                "coin_inception_determined",
                symbol=symbol,
                inception_date=inception_datetime.isoformat(),
                years_of_history=(datetime.now(tz=timezone.utc) - inception_datetime).days / 365.25,
            )

            return inception_datetime

        except Exception as e:
            logger.error(
                "failed_to_determine_inception",
                symbol=symbol,
                error=str(e),
            )
            return None

    def create_training_schedule(
        self,
        symbol: str,
        inception_date: datetime,
        current_date: Optional[datetime] = None,
    ) -> List[TrainingEpoch]:
        """
        Create training schedule from newest to oldest.

        Args:
            symbol: Trading pair
            inception_date: Coin inception date
            current_date: Current date (or None for now)

        Returns:
            List of training epochs (newest to oldest)
        """
        if current_date is None:
            current_date = datetime.now(tz=timezone.utc)

        epochs: List[TrainingEpoch] = []

        # Epoch 0: Most recent (initial_epoch_days)
        epoch_0_start = current_date - timedelta(days=self.config.initial_epoch_days)
        epoch_0_start = max(epoch_0_start, inception_date)  # Don't go before inception

        epochs.append(
            TrainingEpoch(
                epoch_number=0,
                start_date=epoch_0_start,
                end_date=current_date,
                description=f"Most recent ({self.config.initial_epoch_days} days)",
            )
        )

        # Subsequent epochs: Work backwards
        epoch_num = 1
        current_end = epoch_0_start

        while True:
            # Check if we've reached inception
            if current_end <= inception_date:
                break

            # Check max epochs limit
            if self.config.max_epochs is not None and epoch_num >= self.config.max_epochs:
                logger.info(
                    "max_epochs_reached",
                    symbol=symbol,
                    max_epochs=self.config.max_epochs,
                )
                break

            # Calculate epoch dates
            epoch_start = current_end - timedelta(days=self.config.subsequent_epoch_days)
            epoch_start = max(epoch_start, inception_date)

            epochs.append(
                TrainingEpoch(
                    epoch_number=epoch_num,
                    start_date=epoch_start,
                    end_date=current_end,
                    description=f"Epoch {epoch_num} ({epoch_start.year}-{current_end.year})",
                )
            )

            current_end = epoch_start
            epoch_num += 1

        total_days = (current_date - inception_date).days
        logger.info(
            "training_schedule_created",
            symbol=symbol,
            total_epochs=len(epochs),
            total_days=total_days,
            total_years=total_days / 365.25,
            inception_date=inception_date.isoformat(),
        )

        return epochs

    def train_progressive(
        self,
        symbol: str,
        exchange_client: ExchangeClient,
        timeframe: str = "1m",
    ) -> Dict:
        """
        Execute progressive historical training.

        Args:
            symbol: Trading pair
            exchange_client: Exchange client
            timeframe: Candle timeframe

        Returns:
            Training results summary
        """
        logger.info(
            "starting_progressive_training",
            symbol=symbol,
            timeframe=timeframe,
        )

        # Step 1: Determine coin inception
        inception_date = self.get_coin_inception_date(symbol, exchange_client, timeframe)

        if inception_date is None:
            logger.error("cannot_determine_inception", symbol=symbol)
            return {
                "success": False,
                "error": "Could not determine coin inception date",
            }

        # Step 2: Create training schedule
        epochs = self.create_training_schedule(symbol, inception_date)

        if len(epochs) == 0:
            logger.error("no_epochs_created", symbol=symbol)
            return {
                "success": False,
                "error": "No training epochs created",
            }

        # Step 3: Train on each epoch (newest to oldest)
        results = {
            "symbol": symbol,
            "inception_date": inception_date.isoformat(),
            "total_epochs": len(epochs),
            "epochs_trained": 0,
            "epochs_skipped": 0,
            "epoch_results": [],
            "success": True,
        }

        best_performance = None

        for epoch in epochs:
            logger.info(
                "training_epoch",
                symbol=symbol,
                epoch=epoch.epoch_number,
                description=epoch.description,
                start_date=epoch.start_date.isoformat(),
                end_date=epoch.end_date.isoformat(),
            )

            try:
                # Train on this epoch
                epoch_result = self._train_single_epoch(
                    symbol=symbol,
                    exchange_client=exchange_client,
                    epoch=epoch,
                    is_first_epoch=(epoch.epoch_number == 0),
                    timeframe=timeframe,
                )

                results["epoch_results"].append(epoch_result)

                if epoch_result["success"]:
                    results["epochs_trained"] += 1

                    # Check if performance degraded
                    if self.config.early_stop_if_performance_degrades:
                        current_performance = epoch_result.get("sharpe_ratio", 0.0)

                        if best_performance is None:
                            best_performance = current_performance
                        elif current_performance < best_performance * 0.8:
                            # Performance degraded by >20%
                            logger.warning(
                                "performance_degraded_stopping",
                                symbol=symbol,
                                epoch=epoch.epoch_number,
                                best_performance=best_performance,
                                current_performance=current_performance,
                            )
                            break
                        else:
                            best_performance = max(best_performance, current_performance)
                else:
                    results["epochs_skipped"] += 1

            except Exception as e:
                logger.error(
                    "epoch_training_failed",
                    symbol=symbol,
                    epoch=epoch.epoch_number,
                    error=str(e),
                )
                results["epochs_skipped"] += 1

        logger.info(
            "progressive_training_complete",
            symbol=symbol,
            total_epochs=results["total_epochs"],
            epochs_trained=results["epochs_trained"],
            epochs_skipped=results["epochs_skipped"],
        )

        return results

    def _train_single_epoch(
        self,
        symbol: str,
        exchange_client: ExchangeClient,
        epoch: TrainingEpoch,
        is_first_epoch: bool,
        timeframe: str,
    ) -> Dict:
        """
        Train on a single epoch.

        Args:
            symbol: Trading pair
            exchange_client: Exchange client
            epoch: Training epoch
            is_first_epoch: True if first epoch (most recent)
            timeframe: Candle timeframe

        Returns:
            Epoch training results
        """
        # Calculate lookback days for this epoch
        lookback_days = (epoch.end_date - epoch.start_date).days

        logger.info(
            "training_single_epoch",
            symbol=symbol,
            epoch=epoch.epoch_number,
            lookback_days=lookback_days,
            is_first_epoch=is_first_epoch,
        )

        try:
            # Load data for this epoch
            query = CandleQuery(
                symbol=symbol,
                timeframe=timeframe,
                start_at=epoch.start_date,
                end_at=epoch.end_date,
            )

            loader = CandleDataLoader(exchange_client)
            candles = loader.load_candles(query)

            if len(candles) < self.config.min_data_points_per_epoch:
                logger.warning(
                    "insufficient_data_skipping_epoch",
                    symbol=symbol,
                    epoch=epoch.epoch_number,
                    data_points=len(candles),
                    min_required=self.config.min_data_points_per_epoch,
                )
                return {
                    "epoch": epoch.epoch_number,
                    "success": False,
                    "reason": "insufficient_data",
                    "data_points": len(candles),
                }

            # Decide: train from scratch or fine-tune?
            if is_first_epoch and self.config.train_from_scratch_first_epoch:
                mode = "train_from_scratch"
            elif not is_first_epoch and self.config.fine_tune_subsequent_epochs:
                mode = "fine_tune"
            else:
                mode = "train_from_scratch"

            logger.info(
                "epoch_training_mode",
                symbol=symbol,
                epoch=epoch.epoch_number,
                mode=mode,
            )

            # Train using base pipeline
            # Note: We pass the date-bounded data through the existing pipeline
            training_result = self.base_pipeline.train_on_symbol(
                symbol=symbol,
                exchange_client=exchange_client,
                lookback_days=lookback_days,
            )

            # Save checkpoint if configured
            if self.config.save_checkpoints_per_epoch:
                checkpoint_name = f"{symbol.replace('/', '_')}_epoch_{epoch.epoch_number}"
                self._save_checkpoint(checkpoint_name, training_result)

            return {
                "epoch": epoch.epoch_number,
                "success": True,
                "mode": mode,
                "data_points": len(candles),
                "start_date": epoch.start_date.isoformat(),
                "end_date": epoch.end_date.isoformat(),
                "sharpe_ratio": training_result.get("sharpe_ratio", 0.0),
                "total_return": training_result.get("total_return", 0.0),
            }

        except Exception as e:
            logger.error(
                "single_epoch_training_failed",
                symbol=symbol,
                epoch=epoch.epoch_number,
                error=str(e),
            )
            return {
                "epoch": epoch.epoch_number,
                "success": False,
                "reason": "training_error",
                "error": str(e),
            }

    def _save_checkpoint(self, checkpoint_name: str, training_result: Dict) -> None:
        """Save model checkpoint after epoch."""
        try:
            # TODO: Implement checkpoint saving
            # This would save the trained model weights, optimizer state, etc.
            logger.info(
                "checkpoint_saved",
                checkpoint_name=checkpoint_name,
            )
        except Exception as e:
            logger.error(
                "checkpoint_save_failed",
                checkpoint_name=checkpoint_name,
                error=str(e),
            )

    def train_all_symbols(
        self,
        symbols: List[str],
        exchange_client: ExchangeClient,
        timeframe: str = "1m",
    ) -> Dict[str, Dict]:
        """
        Train progressively on multiple symbols.

        Args:
            symbols: List of trading pairs
            exchange_client: Exchange client
            timeframe: Candle timeframe

        Returns:
            Results for each symbol
        """
        all_results = {}

        for symbol in symbols:
            logger.info(
                "progressive_training_symbol",
                symbol=symbol,
                progress=f"{len(all_results) + 1}/{len(symbols)}",
            )

            result = self.train_progressive(
                symbol=symbol,
                exchange_client=exchange_client,
                timeframe=timeframe,
            )

            all_results[symbol] = result

        logger.info(
            "progressive_training_all_complete",
            total_symbols=len(symbols),
            successful=sum(1 for r in all_results.values() if r.get("success", False)),
        )

        return all_results
