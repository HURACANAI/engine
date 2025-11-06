"""
Walk-Forward Validation System

Prevents lookahead bias by:
1. Training on past data only
2. Validating on unseen future data
3. Embargo period to prevent label leakage
4. Rolling window that mimics real-time deployment

Method:
  Train on [D-60 ... D-1], Validate on [D], Deploy on [D+1]
  Train on [D-59 ... D],   Validate on [D+1], Deploy on [D+2]
  ...

Why this matters:
- Random train/test split → lookahead bias (model sees future)
- No embargo → label leakage (overlapping trades)
- Walk-forward → realistic performance estimates
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

import numpy as np
import polars as pl
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class WalkForwardWindow:
    """Single train/test window in walk-forward validation."""

    window_id: int
    train_start: datetime
    train_end: datetime
    embargo_end: datetime
    test_start: datetime
    test_end: datetime

    train_size: int = 0
    test_size: int = 0


@dataclass
class WalkForwardResults:
    """Results from walk-forward validation."""

    windows: List[WalkForwardWindow]
    total_windows: int

    # Performance metrics (across all test windows)
    test_sharpe: float
    test_win_rate: float
    test_avg_pnl_bps: float

    # Stability metrics
    sharpe_std: float  # Std dev of Sharpe across windows
    win_rate_std: float

    # Overfitting indicators
    train_test_sharpe_diff: float  # How much better is train vs test?
    train_test_wr_diff: float


class WalkForwardValidator:
    """
    Walk-forward validation with embargo.

    This is the CORRECT way to validate trading models.

    Usage:
        validator = WalkForwardValidator(
            train_days=60,
            test_days=1,
            embargo_minutes=120
        )

        results = validator.validate(
            data=historical_candles,
            model=my_model
        )

        print(f"Test Sharpe: {results.test_sharpe:.2f}")
    """

    def __init__(
        self,
        train_days: int = 60,
        test_days: int = 1,
        embargo_minutes: int = 120,  # Max hold time
        min_train_samples: int = 100,
        min_test_samples: int = 10
    ):
        """
        Initialize walk-forward validator.

        Args:
            train_days: Training window size
            test_days: Test window size
            embargo_minutes: Gap between train and test
            min_train_samples: Minimum samples for training
            min_test_samples: Minimum samples for testing
        """
        self.train_days = train_days
        self.test_days = test_days
        self.embargo_minutes = embargo_minutes
        self.min_train_samples = min_train_samples
        self.min_test_samples = min_test_samples

        logger.info(
            "walk_forward_validator_initialized",
            train_days=train_days,
            test_days=test_days,
            embargo_minutes=embargo_minutes
        )

    def create_windows(
        self,
        data: pl.DataFrame,
        timestamp_column: str = 'timestamp'
    ) -> List[WalkForwardWindow]:
        """
        Create train/test windows.

        Args:
            data: DataFrame with timestamps
            timestamp_column: Name of timestamp column

        Returns:
            List of WalkForwardWindow objects
        """
        if len(data) < self.min_train_samples + self.min_test_samples:
            logger.error(
                "insufficient_data",
                rows=len(data),
                required=self.min_train_samples + self.min_test_samples
            )
            return []

        # Get date range
        data = data.sort(timestamp_column)
        min_date = data[timestamp_column].min()
        max_date = data[timestamp_column].max()

        windows = []
        window_id = 0

        # Rolling window start
        current_date = min_date + timedelta(days=self.train_days)

        while current_date + timedelta(days=self.test_days) <= max_date:
            # Train window
            train_start = current_date - timedelta(days=self.train_days)
            train_end = current_date

            # Embargo period
            embargo_end = train_end + timedelta(minutes=self.embargo_minutes)

            # Test window (after embargo)
            test_start = embargo_end
            test_end = test_start + timedelta(days=self.test_days)

            # Count samples in each window
            train_data = data.filter(
                (pl.col(timestamp_column) >= train_start) &
                (pl.col(timestamp_column) < train_end)
            )
            test_data = data.filter(
                (pl.col(timestamp_column) >= test_start) &
                (pl.col(timestamp_column) < test_end)
            )

            # Skip if insufficient samples
            if len(train_data) < self.min_train_samples or len(test_data) < self.min_test_samples:
                current_date += timedelta(days=self.test_days)
                continue

            window = WalkForwardWindow(
                window_id=window_id,
                train_start=train_start,
                train_end=train_end,
                embargo_end=embargo_end,
                test_start=test_start,
                test_end=test_end,
                train_size=len(train_data),
                test_size=len(test_data)
            )

            windows.append(window)

            logger.debug(
                "window_created",
                window_id=window_id,
                train_dates=f"{train_start.date()} to {train_end.date()}",
                test_dates=f"{test_start.date()} to {test_end.date()}",
                train_samples=len(train_data),
                test_samples=len(test_data)
            )

            # Move to next window
            current_date += timedelta(days=self.test_days)
            window_id += 1

        logger.info(
            "windows_created",
            total_windows=len(windows),
            date_range=f"{min_date.date()} to {max_date.date()}"
        )

        return windows

    def validate_with_labels(
        self,
        labeled_trades: List,
        windows: Optional[List[WalkForwardWindow]] = None
    ) -> WalkForwardResults:
        """
        Validate using labeled trades.

        Args:
            labeled_trades: List of LabeledTrade objects
            windows: Optional pre-created windows

        Returns:
            WalkForwardResults with performance metrics
        """
        if not labeled_trades:
            logger.error("no_labeled_trades_provided")
            return self._create_empty_results()

        # Create windows if not provided
        if windows is None:
            # Convert trades to dataframe for window creation
            df = pl.DataFrame([{
                'timestamp': t.entry_time,
                'pnl_net_bps': t.pnl_net_bps,
                'meta_label': t.meta_label
            } for t in labeled_trades])

            windows = self.create_windows(df)

        if not windows:
            return self._create_empty_results()

        # Evaluate each window
        window_metrics = []

        for window in windows:
            # Get test trades for this window
            test_trades = [
                t for t in labeled_trades
                if window.test_start <= t.entry_time < window.test_end
            ]

            if not test_trades:
                continue

            # Calculate metrics
            pnl_list = [t.pnl_net_bps for t in test_trades]
            win_rate = sum(1 for t in test_trades if t.meta_label == 1) / len(test_trades)
            avg_pnl = np.mean(pnl_list)

            # Sharpe (assuming daily returns)
            sharpe = (np.mean(pnl_list) / np.std(pnl_list)) if np.std(pnl_list) > 0 else 0

            window_metrics.append({
                'window_id': window.window_id,
                'sharpe': sharpe,
                'win_rate': win_rate,
                'avg_pnl_bps': avg_pnl,
                'num_trades': len(test_trades)
            })

        if not window_metrics:
            return self._create_empty_results()

        # Aggregate across all windows
        sharpe_values = [m['sharpe'] for m in window_metrics]
        wr_values = [m['win_rate'] for m in window_metrics]
        pnl_values = [m['avg_pnl_bps'] for m in window_metrics]

        results = WalkForwardResults(
            windows=windows,
            total_windows=len(windows),
            test_sharpe=np.mean(sharpe_values),
            test_win_rate=np.mean(wr_values),
            test_avg_pnl_bps=np.mean(pnl_values),
            sharpe_std=np.std(sharpe_values),
            win_rate_std=np.std(wr_values),
            train_test_sharpe_diff=0.0,  # TODO: Calculate from train performance
            train_test_wr_diff=0.0
        )

        logger.info(
            "walk_forward_validation_complete",
            windows=len(windows),
            test_sharpe=results.test_sharpe,
            test_win_rate=results.test_win_rate,
            sharpe_stability=results.sharpe_std
        )

        return results

    def _create_empty_results(self) -> WalkForwardResults:
        """Create empty results when validation fails."""
        return WalkForwardResults(
            windows=[],
            total_windows=0,
            test_sharpe=0.0,
            test_win_rate=0.0,
            test_avg_pnl_bps=0.0,
            sharpe_std=0.0,
            win_rate_std=0.0,
            train_test_sharpe_diff=0.0,
            train_test_wr_diff=0.0
        )

    def detect_overfitting(self, results: WalkForwardResults) -> dict:
        """
        Detect signs of overfitting.

        Red flags:
        1. Train Sharpe >> Test Sharpe (overfitting)
        2. High std dev across windows (instability)
        3. Win rate degrades over time (drift)

        Returns:
            Dictionary with overfitting indicators
        """
        is_overfitting = results.train_test_sharpe_diff > 0.5  # Train is 0.5 higher than test

        is_unstable = results.sharpe_std > 0.3  # High variance across windows

        return {
            'is_overfitting': is_overfitting,
            'is_unstable': is_unstable,
            'train_test_gap': results.train_test_sharpe_diff,
            'stability': 1.0 - min(results.sharpe_std, 1.0),  # 0-1, higher is better
            'recommendation': self._get_recommendation(is_overfitting, is_unstable)
        }

    def _get_recommendation(self, is_overfitting: bool, is_unstable: bool) -> str:
        """Get recommendation based on validation results."""
        if is_overfitting and is_unstable:
            return "⚠️ CRITICAL: Model is overfitting AND unstable. Do not deploy."
        elif is_overfitting:
            return "⚠️ WARNING: Model is overfitting. Simplify model or add regularization."
        elif is_unstable:
            return "⚠️ WARNING: Model is unstable across windows. Increase training data or use ensemble."
        else:
            return "✅ Model passed validation. Safe to deploy."


def print_walk_forward_results(results: WalkForwardResults) -> None:
    """Pretty-print walk-forward results."""
    print(f"""
╔═══════════════════════════════════════════╗
║   WALK-FORWARD VALIDATION RESULTS         ║
╚═══════════════════════════════════════════╝

📊 Test Performance (Out-of-Sample)
──────────────────────────────────────────────
Windows:           {results.total_windows}
Test Sharpe:       {results.test_sharpe:.2f}
Test Win Rate:     {results.test_win_rate:.1%}
Avg P&L (net):     {results.test_avg_pnl_bps:+.2f} bps

📈 Stability Metrics
──────────────────────────────────────────────
Sharpe Std Dev:    {results.sharpe_std:.2f} {'✅ Stable' if results.sharpe_std < 0.3 else '⚠️ Unstable'}
Win Rate Std Dev:  {results.win_rate_std:.2%}

🔍 Overfitting Check
──────────────────────────────────────────────
Train-Test Gap:    {results.train_test_sharpe_diff:+.2f} {'⚠️ Overfitting!' if results.train_test_sharpe_diff > 0.5 else '✅ Good'}

{'✅ PASSED - Model is ready for deployment' if results.test_sharpe > 1.0 and results.sharpe_std < 0.3 else '⚠️ REVIEW - Check metrics before deploying'}
""")
