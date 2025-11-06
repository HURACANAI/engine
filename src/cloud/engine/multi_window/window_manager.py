"""
Training Window Manager

Manages data windows for different components with appropriate:
1. Historical cutoff (e.g., scalp = 60 days, regime = 365 days)
2. Recency weighting (e.g., scalp halflife = 10 days, regime = 60 days)
3. Walk-forward splits

This ensures each component trains on the RIGHT data, not ALL data.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional

import numpy as np
import polars as pl
import structlog

from .component_configs import ComponentConfig
from ..weighting import RecencyWeighter
from ..walk_forward import WalkForwardValidator, WalkForwardWindow

logger = structlog.get_logger(__name__)


@dataclass
class ComponentDataWindow:
    """Data window for a specific component."""

    component_name: str
    config: ComponentConfig

    # Data
    data: pl.DataFrame
    weights: np.ndarray

    # Window info
    start_date: datetime
    end_date: datetime
    total_samples: int
    effective_sample_size: float

    # Walk-forward splits (if requested)
    wf_windows: Optional[List[WalkForwardWindow]] = None


class TrainingWindowManager:
    """
    Manages training windows for multiple components.

    Usage:
        manager = TrainingWindowManager()

        # Prepare data for scalp component
        window = manager.prepare_component_window(
            data=labeled_trades_df,
            config=ScalpCoreConfig()
        )

        # Use window for training
        X_train = window.data
        sample_weights = window.weights
    """

    def __init__(self, validate_windows: bool = True):
        """
        Initialize window manager.

        Args:
            validate_windows: Whether to validate walk-forward windows
        """
        self.validate_windows = validate_windows

        logger.info("training_window_manager_initialized")

    def prepare_component_window(
        self,
        data: pl.DataFrame,
        config: ComponentConfig,
        create_walk_forward_splits: bool = False
    ) -> ComponentDataWindow:
        """
        Prepare training window for a component.

        Args:
            data: Full dataset (must have 'timestamp' column)
            config: Component configuration
            create_walk_forward_splits: Whether to create WF splits

        Returns:
            ComponentDataWindow with filtered data and weights
        """
        logger.info(
            "preparing_component_window",
            component=config.name,
            lookback_days=config.lookback_days,
            total_rows=len(data)
        )

        # Step 1: Cut data to lookback window
        windowed_data = self._apply_lookback_window(data, config)

        if len(windowed_data) == 0:
            logger.error(
                "no_data_in_window",
                component=config.name,
                lookback_days=config.lookback_days
            )
            raise ValueError(f"No data available for {config.name} window")

        # Step 2: Calculate recency weights
        weighter = RecencyWeighter(
            halflife_days=config.recency_halflife_days,
            min_weight=0.01  # Don't completely discard old data
        )
        weights = weighter.calculate_weights(windowed_data)

        # Step 3: Validate sample count
        if len(windowed_data) < config.min_samples:
            logger.warning(
                "insufficient_samples",
                component=config.name,
                available=len(windowed_data),
                required=config.min_samples
            )

        # Step 4: Create walk-forward splits if requested
        wf_windows = None
        if create_walk_forward_splits:
            wf_windows = self._create_walk_forward_splits(windowed_data, config)

        # Get date range
        timestamps = windowed_data['timestamp']
        if hasattr(timestamps, 'to_numpy'):
            timestamps_array = timestamps.to_numpy()
        else:
            timestamps_array = np.array(timestamps)

        start_date = timestamps_array.min()
        end_date = timestamps_array.max()

        # Calculate effective sample size
        effective_n = weighter.get_effective_sample_size(weights)

        logger.info(
            "component_window_prepared",
            component=config.name,
            samples=len(windowed_data),
            effective_samples=effective_n,
            start_date=start_date,
            end_date=end_date,
            days_span=(end_date - start_date).days if hasattr(end_date - start_date, 'days') else 0
        )

        return ComponentDataWindow(
            component_name=config.name,
            config=config,
            data=windowed_data,
            weights=weights,
            start_date=start_date,
            end_date=end_date,
            total_samples=len(windowed_data),
            effective_sample_size=effective_n,
            wf_windows=wf_windows
        )

    def _apply_lookback_window(
        self,
        data: pl.DataFrame,
        config: ComponentConfig
    ) -> pl.DataFrame:
        """
        Filter data to lookback window.

        Args:
            data: Full dataset
            config: Component config with lookback_days

        Returns:
            Filtered dataframe
        """
        # Find latest timestamp
        latest_timestamp = data['timestamp'].max()

        # Calculate cutoff
        cutoff_date = latest_timestamp - timedelta(days=config.lookback_days)

        # Filter
        windowed = data.filter(pl.col('timestamp') >= cutoff_date)

        logger.debug(
            "lookback_window_applied",
            component=config.name,
            original_rows=len(data),
            windowed_rows=len(windowed),
            cutoff_date=cutoff_date,
            latest_date=latest_timestamp
        )

        return windowed

    def _create_walk_forward_splits(
        self,
        data: pl.DataFrame,
        config: ComponentConfig
    ) -> List[WalkForwardWindow]:
        """
        Create walk-forward validation splits.

        Args:
            data: Windowed data
            config: Component config with WF parameters

        Returns:
            List of walk-forward windows
        """
        validator = WalkForwardValidator(
            train_days=config.walk_forward_train_days,
            test_days=config.walk_forward_test_days,
            embargo_minutes=config.embargo_minutes
        )

        windows = validator.create_windows(data)

        logger.info(
            "walk_forward_splits_created",
            component=config.name,
            num_windows=len(windows),
            train_days=config.walk_forward_train_days,
            test_days=config.walk_forward_test_days
        )

        return windows

    def prepare_all_components(
        self,
        data: pl.DataFrame,
        configs: dict[str, ComponentConfig],
        create_walk_forward_splits: bool = False
    ) -> dict[str, ComponentDataWindow]:
        """
        Prepare windows for all components.

        Args:
            data: Full dataset
            configs: Dict mapping component names to configs
            create_walk_forward_splits: Whether to create WF splits

        Returns:
            Dict mapping component names to prepared windows

        Usage:
            configs = create_all_component_configs()
            windows = manager.prepare_all_components(
                data=labeled_trades_df,
                configs=configs
            )

            scalp_window = windows['scalp_core']
            regime_window = windows['regime_classifier']
        """
        windows = {}

        logger.info(
            "preparing_all_component_windows",
            components=list(configs.keys()),
            total_rows=len(data)
        )

        for name, config in configs.items():
            try:
                window = self.prepare_component_window(
                    data=data,
                    config=config,
                    create_walk_forward_splits=create_walk_forward_splits
                )
                windows[name] = window

            except Exception as e:
                logger.error(
                    "component_window_preparation_failed",
                    component=name,
                    error=str(e)
                )
                # Continue with other components
                continue

        logger.info(
            "all_component_windows_prepared",
            successful=len(windows),
            total=len(configs)
        )

        return windows

    def get_training_summary(
        self,
        windows: dict[str, ComponentDataWindow]
    ) -> dict:
        """
        Get summary statistics for all windows.

        Returns:
            Dictionary with summary stats
        """
        summary = {
            'components': {},
            'total_effective_samples': 0,
            'date_range': None
        }

        earliest_date = None
        latest_date = None

        for name, window in windows.items():
            summary['components'][name] = {
                'samples': window.total_samples,
                'effective_samples': window.effective_sample_size,
                'start_date': window.start_date,
                'end_date': window.end_date,
                'lookback_days': window.config.lookback_days,
                'timeframe': window.config.timeframe
            }

            summary['total_effective_samples'] += window.effective_sample_size

            # Track overall date range
            if earliest_date is None or window.start_date < earliest_date:
                earliest_date = window.start_date
            if latest_date is None or window.end_date > latest_date:
                latest_date = window.end_date

        summary['date_range'] = {
            'start': earliest_date,
            'end': latest_date,
            'days': (latest_date - earliest_date).days if earliest_date and latest_date else 0
        }

        return summary


def print_window_summary(windows: dict[str, ComponentDataWindow]) -> None:
    """
    Pretty-print summary of component windows.

    Usage:
        windows = manager.prepare_all_components(data, configs)
        print_window_summary(windows)
    """
    print("\n" + "="*80)
    print("COMPONENT TRAINING WINDOWS SUMMARY")
    print("="*80 + "\n")

    print(f"{'Component':<20} {'Samples':<12} {'Effective':<12} {'Date Range':<30}")
    print("-"*80)

    for name, window in windows.items():
        date_range = f"{window.start_date.strftime('%Y-%m-%d')} to {window.end_date.strftime('%Y-%m-%d')}"

        print(
            f"{window.component_name:<20} "
            f"{window.total_samples:<12,} "
            f"{int(window.effective_sample_size):<12,} "
            f"{date_range:<30}"
        )

    print("\n" + "="*80)

    # Overall stats
    total_samples = sum(w.total_samples for w in windows.values())
    total_effective = sum(w.effective_sample_size for w in windows.values())

    print(f"\nTotal samples: {total_samples:,}")
    print(f"Total effective samples: {int(total_effective):,}")
    print(f"Components: {len(windows)}")

    print("\n" + "="*80 + "\n")
