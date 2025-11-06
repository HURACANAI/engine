"""
Incremental Labeling System

Enables efficient hourly updates by:
1. Caching previously labeled trades
2. Only labeling NEW candles since last run
3. Updating incomplete labels (e.g., timeout not hit yet)
4. Maintaining rolling window (e.g., keep last 90 days)

This is critical for Mechanic - avoids re-labeling 90 days every hour.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional
import pickle

import polars as pl
import structlog

from ..labeling import TripleBarrierLabeler, LabeledTrade, ScalpLabelConfig
from ..costs import CostEstimator

logger = structlog.get_logger(__name__)


@dataclass
class LabelCache:
    """
    Cache of previously labeled trades.

    Stored in pickle for fast load/save.
    """
    labeled_trades: List[LabeledTrade]
    last_update: datetime
    last_candle_timestamp: datetime
    symbol: str
    config_hash: str  # Detect config changes


class IncrementalLabeler:
    """
    Incremental labeling for efficient hourly updates.

    Usage (Mechanic flow):
        labeler = IncrementalLabeler(
            base_labeler=TripleBarrierLabeler(...),
            cache_dir='./cache/labels'
        )

        # First run: labels all data, caches
        labeled_trades = labeler.label_incremental(
            new_candles=df,
            symbol='BTC/USDT'
        )

        # Hourly: only labels new candles
        new_candles = fetch_last_hour()
        labeled_trades = labeler.label_incremental(
            new_candles=new_candles,
            symbol='BTC/USDT'
        )
        # → Returns cached + new labels
    """

    def __init__(
        self,
        base_labeler: TripleBarrierLabeler,
        cache_dir: str | Path,
        rolling_window_days: int = 90
    ):
        """
        Initialize incremental labeler.

        Args:
            base_labeler: Underlying triple-barrier labeler
            cache_dir: Directory to store label cache
            rolling_window_days: Keep last N days of labels
        """
        self.base_labeler = base_labeler
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.rolling_window_days = rolling_window_days

        logger.info(
            "incremental_labeler_initialized",
            cache_dir=str(self.cache_dir),
            rolling_window_days=rolling_window_days
        )

    def label_incremental(
        self,
        new_candles: pl.DataFrame,
        symbol: str,
        force_full_relabel: bool = False
    ) -> List[LabeledTrade]:
        """
        Label new candles incrementally.

        Flow:
        1. Load cached labels
        2. Check if new candles exist
        3. Label only new candles
        4. Update incomplete labels (timeout not hit)
        5. Combine cached + new
        6. Apply rolling window
        7. Save cache

        Args:
            new_candles: New candle data (may include overlap)
            symbol: Trading symbol
            force_full_relabel: Force full relabeling (config changed)

        Returns:
            Complete list of labeled trades (cached + new)
        """
        logger.info(
            "incremental_labeling_start",
            symbol=symbol,
            new_candles=len(new_candles),
            force_full=force_full_relabel
        )

        # Load cache
        cache = self._load_cache(symbol)

        # Check if config changed
        current_hash = self._config_hash()
        if cache and cache.config_hash != current_hash:
            logger.warning(
                "config_changed_forcing_full_relabel",
                symbol=symbol,
                old_hash=cache.config_hash,
                new_hash=current_hash
            )
            force_full_relabel = True

        # Full relabel if forced or no cache
        if force_full_relabel or cache is None:
            logger.info("full_labeling", symbol=symbol)
            labeled_trades = self.base_labeler.label_dataframe(
                df=new_candles,
                symbol=symbol
            )

            self._save_cache(
                symbol=symbol,
                labeled_trades=labeled_trades,
                last_candle_timestamp=new_candles['timestamp'].max()
            )

            return labeled_trades

        # Incremental labeling
        logger.info(
            "incremental_labeling",
            symbol=symbol,
            cached_trades=len(cache.labeled_trades),
            last_cache_timestamp=cache.last_candle_timestamp
        )

        # Find new candles (after last cache timestamp)
        new_candles_only = new_candles.filter(
            pl.col('timestamp') > cache.last_candle_timestamp
        )

        if len(new_candles_only) == 0:
            logger.info("no_new_candles", symbol=symbol)
            return cache.labeled_trades

        logger.info(
            "new_candles_found",
            symbol=symbol,
            new_candles_count=len(new_candles_only)
        )

        # Label new candles
        # Need historical context, so include some lookback
        lookback_candles = self._get_lookback_candles(
            all_candles=new_candles,
            new_candle_start=cache.last_candle_timestamp
        )

        new_labels = self.base_labeler.label_dataframe(
            df=lookback_candles,
            symbol=symbol
        )

        # Filter to only labels that start AFTER cache timestamp
        new_labels_filtered = [
            t for t in new_labels
            if t.entry_time > cache.last_candle_timestamp
        ]

        logger.info(
            "new_labels_generated",
            symbol=symbol,
            new_labels=len(new_labels_filtered)
        )

        # Update incomplete labels from cache
        updated_cache_labels = self._update_incomplete_labels(
            cached_labels=cache.labeled_trades,
            all_candles=new_candles
        )

        # Combine
        all_labels = updated_cache_labels + new_labels_filtered

        # Apply rolling window
        all_labels = self._apply_rolling_window(all_labels)

        # Save cache
        self._save_cache(
            symbol=symbol,
            labeled_trades=all_labels,
            last_candle_timestamp=new_candles['timestamp'].max()
        )

        logger.info(
            "incremental_labeling_complete",
            symbol=symbol,
            total_labels=len(all_labels),
            new_labels=len(new_labels_filtered)
        )

        return all_labels

    def _load_cache(self, symbol: str) -> Optional[LabelCache]:
        """Load cached labels from disk."""
        cache_path = self._cache_path(symbol)

        if not cache_path.exists():
            logger.debug("no_cache_found", symbol=symbol)
            return None

        try:
            with open(cache_path, 'rb') as f:
                cache = pickle.load(f)

            logger.info(
                "cache_loaded",
                symbol=symbol,
                cached_trades=len(cache.labeled_trades),
                last_update=cache.last_update
            )

            return cache

        except Exception as e:
            logger.error(
                "cache_load_failed",
                symbol=symbol,
                error=str(e)
            )
            return None

    def _save_cache(
        self,
        symbol: str,
        labeled_trades: List[LabeledTrade],
        last_candle_timestamp: datetime
    ):
        """Save labels to cache."""
        cache = LabelCache(
            labeled_trades=labeled_trades,
            last_update=datetime.now(),
            last_candle_timestamp=last_candle_timestamp,
            symbol=symbol,
            config_hash=self._config_hash()
        )

        cache_path = self._cache_path(symbol)

        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(cache, f)

            logger.info(
                "cache_saved",
                symbol=symbol,
                trades=len(labeled_trades),
                path=str(cache_path)
            )

        except Exception as e:
            logger.error(
                "cache_save_failed",
                symbol=symbol,
                error=str(e)
            )

    def _cache_path(self, symbol: str) -> Path:
        """Get cache file path for symbol."""
        safe_symbol = symbol.replace('/', '-')
        return self.cache_dir / f"{safe_symbol}_labels.pkl"

    def _config_hash(self) -> str:
        """
        Generate hash of labeler config.

        If config changes, cache is invalidated.
        """
        config = self.base_labeler.config
        hash_str = f"{config.tp_bps}_{config.sl_bps}_{config.timeout_minutes}"
        return hash_str

    def _get_lookback_candles(
        self,
        all_candles: pl.DataFrame,
        new_candle_start: datetime
    ) -> pl.DataFrame:
        """
        Get candles with lookback for context.

        Need historical candles before new_candle_start to properly
        label trades that may span the boundary.
        """
        # Get timeout window worth of lookback
        timeout_minutes = self.base_labeler.config.timeout_minutes
        lookback_start = new_candle_start - timedelta(minutes=timeout_minutes * 2)

        lookback_candles = all_candles.filter(
            pl.col('timestamp') >= lookback_start
        )

        logger.debug(
            "lookback_candles_prepared",
            total_candles=len(lookback_candles),
            lookback_minutes=timeout_minutes * 2
        )

        return lookback_candles

    def _update_incomplete_labels(
        self,
        cached_labels: List[LabeledTrade],
        all_candles: pl.DataFrame
    ) -> List[LabeledTrade]:
        """
        Update incomplete labels with new data.

        Some cached labels may have hit timeout but not TP/SL.
        With new data, they might now hit TP/SL.

        For simplicity, we keep cached labels as-is.
        (Full implementation would re-check exit conditions)
        """
        # Simple implementation: keep cached as-is
        # Advanced: re-check trades that exited on timeout

        timeout_exits = [
            t for t in cached_labels
            if t.exit_reason == 'timeout'
        ]

        if timeout_exits:
            logger.debug(
                "incomplete_labels_found",
                timeout_count=len(timeout_exits)
            )
            # TODO: Re-check these with new data
            # For now, keep as-is

        return cached_labels

    def _apply_rolling_window(
        self,
        labeled_trades: List[LabeledTrade]
    ) -> List[LabeledTrade]:
        """
        Apply rolling window to keep last N days.

        Prevents cache from growing infinitely.
        """
        if not labeled_trades:
            return labeled_trades

        cutoff_date = datetime.now() - timedelta(days=self.rolling_window_days)

        windowed_trades = [
            t for t in labeled_trades
            if t.entry_time >= cutoff_date
        ]

        removed = len(labeled_trades) - len(windowed_trades)

        if removed > 0:
            logger.info(
                "rolling_window_applied",
                total_trades=len(labeled_trades),
                kept_trades=len(windowed_trades),
                removed_trades=removed,
                cutoff_date=cutoff_date
            )

        return windowed_trades

    def clear_cache(self, symbol: Optional[str] = None):
        """
        Clear label cache.

        Args:
            symbol: Specific symbol to clear, or None for all
        """
        if symbol:
            cache_path = self._cache_path(symbol)
            if cache_path.exists():
                cache_path.unlink()
                logger.info("cache_cleared", symbol=symbol)
        else:
            # Clear all caches
            for cache_file in self.cache_dir.glob("*_labels.pkl"):
                cache_file.unlink()
                logger.info("cache_cleared", file=cache_file.name)


def create_incremental_labeler_for_mechanic(
    trading_mode: str = 'scalp',
    exchange: str = 'binance',
    cache_dir: str = './cache/mechanic/labels',
    rolling_window_days: int = 90
) -> IncrementalLabeler:
    """
    Create incremental labeler configured for Mechanic.

    Args:
        trading_mode: 'scalp' or 'runner'
        exchange: Exchange name
        cache_dir: Where to cache labels
        rolling_window_days: Keep last N days

    Returns:
        Configured IncrementalLabeler

    Usage:
        labeler = create_incremental_labeler_for_mechanic(
            trading_mode='scalp',
            exchange='binance'
        )

        # Hourly update
        new_candles = fetch_last_hour()
        labeled_trades = labeler.label_incremental(
            new_candles=new_candles,
            symbol='BTC/USDT'
        )
    """
    from ..labeling import ScalpLabelConfig, RunnerLabelConfig, TripleBarrierLabeler
    from ..costs import CostEstimator

    # Create config
    if trading_mode == 'scalp':
        config = ScalpLabelConfig()
    else:
        config = RunnerLabelConfig()

    # Create base labeler
    cost_estimator = CostEstimator(exchange=exchange)
    base_labeler = TripleBarrierLabeler(
        config=config,
        cost_estimator=cost_estimator
    )

    # Create incremental labeler
    incremental_labeler = IncrementalLabeler(
        base_labeler=base_labeler,
        cache_dir=cache_dir,
        rolling_window_days=rolling_window_days
    )

    logger.info(
        "mechanic_incremental_labeler_created",
        trading_mode=trading_mode,
        exchange=exchange,
        rolling_window_days=rolling_window_days
    )

    return incremental_labeler
