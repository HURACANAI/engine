"""
Purged Walk-Forward Testing - Prevents data leakage.

Implements:
- Expanding/rolling windows
- Purge gap to remove label overlap
- Combinatorial purged k-fold for model selection
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional, Tuple, Callable, Any
import numpy as np
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class WalkForwardWindow:
    """Walk-forward test window."""
    train_start: datetime
    train_end: datetime
    purge_start: datetime
    purge_end: datetime
    test_start: datetime
    test_end: datetime
    window_type: str  # "expanding" or "rolling"


@dataclass
class WalkForwardResult:
    """Result from walk-forward test."""
    window: WalkForwardWindow
    train_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    model_id: Optional[str] = None
    error: Optional[str] = None


class PurgedWalkForwardTester:
    """
    Purged walk-forward testing to prevent data leakage.
    
    Features:
    - Train window: 120 days
    - Purge gap: 3-5 days
    - Test window: 30 days
    - Expanding or rolling windows
    - Combinatorial purged k-fold for model selection
    """
    
    def __init__(
        self,
        train_days: int = 120,
        purge_days: int = 5,
        test_days: int = 30,
        window_type: str = "expanding",  # "expanding" or "rolling"
        min_train_samples: int = 100,
    ) -> None:
        """
        Initialize purged walk-forward tester.
        
        Args:
            train_days: Training window size in days (default: 120)
            purge_days: Purge gap in days (default: 5)
            test_days: Test window size in days (default: 30)
            window_type: "expanding" or "rolling" (default: "expanding")
            min_train_samples: Minimum samples required for training
        """
        self.train_days = train_days
        self.purge_days = purge_days
        self.test_days = test_days
        self.window_type = window_type
        self.min_train_samples = min_train_samples
        
        logger.info(
            "purged_walk_forward_initialized",
            train_days=train_days,
            purge_days=purge_days,
            test_days=test_days,
            window_type=window_type
        )
    
    def generate_windows(
        self,
        start_date: datetime,
        end_date: datetime,
        step_days: Optional[int] = None
    ) -> List[WalkForwardWindow]:
        """
        Generate walk-forward test windows.
        
        Args:
            start_date: Start of available data
            end_date: End of available data
            step_days: Days to step forward (default: test_days)
        
        Returns:
            List of walk-forward windows
        """
        if step_days is None:
            step_days = self.test_days
        
        windows = []
        current_start = start_date
        
        while True:
            # Training window
            train_start = current_start
            train_end = train_start + timedelta(days=self.train_days)
            
            # Purge gap
            purge_start = train_end
            purge_end = purge_start + timedelta(days=self.purge_days)
            
            # Test window
            test_start = purge_end
            test_end = test_start + timedelta(days=self.test_days)
            
            # Check if we have enough data
            if test_end > end_date:
                break
            
            # Check minimum samples
            train_duration = (train_end - train_start).days
            if train_duration < self.train_days:
                break
            
            window = WalkForwardWindow(
                train_start=train_start,
                train_end=train_end,
                purge_start=purge_start,
                purge_end=purge_end,
                test_start=test_start,
                test_end=test_end,
                window_type=self.window_type
            )
            windows.append(window)
            
            # Move to next window
            if self.window_type == "expanding":
                # Expanding: train_start stays same, test moves forward
                current_start = train_start
            else:
                # Rolling: both move forward
                current_start = current_start + timedelta(days=step_days)
        
        logger.info(
            "walk_forward_windows_generated",
            count=len(windows),
            first_window_start=windows[0].train_start.isoformat() if windows else None,
            last_window_end=windows[-1].test_end.isoformat() if windows else None
        )
        
        return windows
    
    def run_walk_forward(
        self,
        windows: List[WalkForwardWindow],
        train_fn: Callable[[datetime, datetime], Any],
        test_fn: Callable[[Any, datetime, datetime], Dict[str, float]],
        data_loader: Callable[[datetime, datetime], Any],
    ) -> List[WalkForwardResult]:
        """
        Run walk-forward testing.
        
        Args:
            windows: List of walk-forward windows
            train_fn: Function to train model (start, end) -> model
            test_fn: Function to test model (model, start, end) -> metrics
            data_loader: Function to load data (start, end) -> data
        
        Returns:
            List of walk-forward results
        """
        results = []
        
        for i, window in enumerate(windows):
            logger.info(
                "walk_forward_window_start",
                window_num=i + 1,
                total=len(windows),
                train_start=window.train_start.isoformat(),
                test_end=window.test_end.isoformat()
            )
            
            try:
                # Load training data
                train_data = data_loader(window.train_start, window.train_end)
                
                if len(train_data) < self.min_train_samples:
                    logger.warning(
                        "insufficient_train_samples",
                        window_num=i + 1,
                        samples=len(train_data),
                        required=self.min_train_samples
                    )
                    continue
                
                # Train model
                model = train_fn(window.train_start, window.train_end)
                
                # Load test data
                test_data = data_loader(window.test_start, window.test_end)
                
                # Test model
                test_metrics = test_fn(model, window.test_start, window.test_end)
                
                # Calculate training metrics (optional)
                train_metrics = {}  # Can be calculated if needed
                
                result = WalkForwardResult(
                    window=window,
                    train_metrics=train_metrics,
                    test_metrics=test_metrics
                )
                results.append(result)
                
                logger.info(
                    "walk_forward_window_complete",
                    window_num=i + 1,
                    test_sharpe=test_metrics.get('sharpe_ratio', 0.0),
                    test_win_rate=test_metrics.get('win_rate', 0.0)
                )
                
            except Exception as e:
                logger.error(
                    "walk_forward_window_failed",
                    window_num=i + 1,
                    error=str(e)
                )
                results.append(WalkForwardResult(
                    window=window,
                    train_metrics={},
                    test_metrics={},
                    error=str(e)
                ))
        
        return results
    
    def aggregate_results(
        self,
        results: List[WalkForwardResult]
    ) -> Dict[str, Any]:
        """
        Aggregate walk-forward results.
        
        Args:
            results: List of walk-forward results
        
        Returns:
            Aggregated metrics with mean and dispersion
        """
        if not results:
            return {}
        
        # Extract test metrics
        test_metrics_list = [r.test_metrics for r in results if r.error is None]
        
        if not test_metrics_list:
            return {"error": "No valid results"}
        
        # Aggregate each metric
        aggregated = {}
        metric_names = set()
        for metrics in test_metrics_list:
            metric_names.update(metrics.keys())
        
        for metric_name in metric_names:
            values = [
                metrics.get(metric_name, 0.0)
                for metrics in test_metrics_list
                if metric_name in metrics
            ]
            
            if values:
                aggregated[f"{metric_name}_mean"] = float(np.mean(values))
                aggregated[f"{metric_name}_std"] = float(np.std(values))
                aggregated[f"{metric_name}_min"] = float(np.min(values))
                aggregated[f"{metric_name}_max"] = float(np.max(values))
                aggregated[f"{metric_name}_median"] = float(np.median(values))
        
        aggregated["num_windows"] = len(results)
        aggregated["num_valid"] = len(test_metrics_list)
        
        logger.info(
            "walk_forward_results_aggregated",
            num_windows=len(results),
            num_valid=len(test_metrics_list)
        )
        
        return aggregated
    
    def combinatorial_purged_kfold(
        self,
        train_start: datetime,
        train_end: datetime,
        k: int = 5,
        purge_days: int = 5
    ) -> List[Tuple[datetime, datetime, datetime, datetime]]:
        """
        Generate combinatorial purged k-fold splits.
        
        Reduces data snooping by using multiple train/validation splits.
        
        Args:
            train_start: Training period start
            train_end: Training period end
            k: Number of folds
            purge_days: Purge gap in days
        
        Returns:
            List of (train_start, train_end, val_start, val_end) tuples
        """
        total_days = (train_end - train_start).days
        fold_size = total_days // k
        
        splits = []
        
        for i in range(k):
            # Validation fold
            val_start = train_start + timedelta(days=i * fold_size)
            val_end = val_start + timedelta(days=fold_size)
            
            # Training folds (all other folds)
            # Split into before and after validation
            train_before_end = val_start - timedelta(days=purge_days)
            train_after_start = val_end + timedelta(days=purge_days)
            
            # Create two training periods
            if train_before_end > train_start:
                splits.append((
                    train_start,
                    train_before_end,
                    val_start,
                    val_end
                ))
            
            if train_after_start < train_end:
                splits.append((
                    train_after_start,
                    train_end,
                    val_start,
                    val_end
                ))
        
        logger.info(
            "combinatorial_kfold_generated",
            k=k,
            splits=len(splits),
            purge_days=purge_days
        )
        
        return splits

