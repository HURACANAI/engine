"""
Walk-Forward Purged Cross-Validation

Implements walk-forward validation with purged gaps to prevent data leakage.
Ensures no overlap between training and testing periods.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class WalkForwardWindow:
    """Single walk-forward window."""
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    purge_start: datetime  # Start of purge gap
    purge_end: datetime  # End of purge gap
    window_index: int


@dataclass
class WalkForwardResult:
    """Walk-forward validation result."""
    window_index: int
    train_metrics: Dict[str, float] = field(default_factory=dict)
    test_metrics: Dict[str, float] = field(default_factory=dict)
    model_path: Optional[str] = None
    leakage_detected: bool = False
    leakage_score: float = 0.0


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward validation."""
    train_days: int = 20
    test_days: int = 5
    purge_days: int = 2  # Gap between train and test
    min_windows: int = 5
    min_test_trades: int = 100
    leakage_threshold: float = 0.3  # Max allowed train-test gap
    min_train_samples: int = 1000


class WalkForwardPurgedCV:
    """
    Walk-forward purged cross-validation.
    
    Features:
    - Multiple walk-forward windows
    - Purged gaps between train/test sets
    - Leakage detection
    - Performance tracking across windows
    """
    
    def __init__(
        self,
        config: WalkForwardConfig,
    ) -> None:
        """
        Initialize walk-forward validator.
        
        Args:
            config: Walk-forward configuration
        """
        self.config = config
        
        logger.info(
            "walk_forward_initialized",
            train_days=config.train_days,
            test_days=config.test_days,
            purge_days=config.purge_days,
            min_windows=config.min_windows,
        )
    
    def generate_windows(
        self,
        data_start: datetime,
        data_end: datetime,
    ) -> List[WalkForwardWindow]:
        """
        Generate walk-forward windows.
        
        Args:
            data_start: Start of available data
            data_end: End of available data
        
        Returns:
            List of walk-forward windows
        """
        windows = []
        window_index = 0
        
        current_start = data_start
        
        while True:
            # Training period
            train_start = current_start
            train_end = train_start + timedelta(days=self.config.train_days)
            
            # Purge gap
            purge_start = train_end
            purge_end = purge_start + timedelta(days=self.config.purge_days)
            
            # Test period
            test_start = purge_end
            test_end = test_start + timedelta(days=self.config.test_days)
            
            # Check if we have enough data
            if test_end > data_end:
                break
            
            window = WalkForwardWindow(
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                purge_start=purge_start,
                purge_end=purge_end,
                window_index=window_index,
            )
            
            windows.append(window)
            
            # Move to next window (step forward by test_days)
            current_start = test_end
            window_index += 1
        
        # Check minimum windows requirement
        if len(windows) < self.config.min_windows:
            logger.warning(
                "insufficient_windows",
                generated=len(windows),
                required=self.config.min_windows,
            )
        
        logger.info(
            "walk_forward_windows_generated",
            num_windows=len(windows),
            first_window_start=windows[0].train_start.isoformat() if windows else None,
            last_window_end=windows[-1].test_end.isoformat() if windows else None,
        )
        
        return windows
    
    def split_data(
        self,
        data: pd.DataFrame,
        window: WalkForwardWindow,
        timestamp_column: str = "timestamp",
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into train and test sets for a window.
        
        Args:
            data: Full dataset
            window: Walk-forward window
            timestamp_column: Name of timestamp column
        
        Returns:
            Tuple of (train_data, test_data)
        """
        # Ensure timestamp column is datetime
        if not pd.api.types.is_datetime64_any_dtype(data[timestamp_column]):
            data[timestamp_column] = pd.to_datetime(data[timestamp_column])
        
        # Training data
        train_mask = (
            (data[timestamp_column] >= window.train_start) &
            (data[timestamp_column] < window.train_end)
        )
        train_data = data[train_mask].copy()
        
        # Test data (after purge gap)
        test_mask = (
            (data[timestamp_column] >= window.test_start) &
            (data[timestamp_column] < window.test_end)
        )
        test_data = data[test_mask].copy()
        
        # Verify no overlap
        if len(train_data) > 0 and len(test_data) > 0:
            max_train_time = train_data[timestamp_column].max()
            min_test_time = test_data[timestamp_column].min()
            
            if min_test_time <= max_train_time:
                logger.warning(
                    "data_overlap_detected",
                    window_index=window.window_index,
                    max_train=max_train_time.isoformat(),
                    min_test=min_test_time.isoformat(),
                )
        
        logger.debug(
            "data_split",
            window_index=window.window_index,
            train_samples=len(train_data),
            test_samples=len(test_data),
        )
        
        return train_data, test_data
    
    def detect_leakage(
        self,
        train_metrics: Dict[str, float],
        test_metrics: Dict[str, float],
    ) -> Tuple[bool, float]:
        """
        Detect data leakage by comparing train and test metrics.
        
        Large gaps between train and test performance may indicate leakage.
        
        Args:
            train_metrics: Training set metrics
            test_metrics: Test set metrics
        
        Returns:
            Tuple of (leakage_detected, leakage_score)
        """
        # Key metrics to compare
        key_metrics = ["sharpe_ratio", "hit_rate", "edge_after_cost_bps"]
        
        gaps = []
        
        for metric in key_metrics:
            train_val = train_metrics.get(metric, 0.0)
            test_val = test_metrics.get(metric, 0.0)
            
            if train_val == 0.0:
                continue
            
            # Calculate relative gap
            gap = abs(train_val - test_val) / (abs(train_val) + 1e-8)
            gaps.append(gap)
        
        if not gaps:
            return False, 0.0
        
        # Average gap
        avg_gap = np.mean(gaps)
        
        # Leakage detected if gap exceeds threshold
        leakage_detected = avg_gap > self.config.leakage_threshold
        
        return leakage_detected, avg_gap
    
    def validate_window(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        window: WalkForwardWindow,
        train_fn: Any,  # Training function
        evaluate_fn: Any,  # Evaluation function
    ) -> WalkForwardResult:
        """
        Validate a single window.
        
        Args:
            train_data: Training data
            test_data: Test data
            window: Walk-forward window
            train_fn: Function to train model (returns model and train_metrics)
            evaluate_fn: Function to evaluate model (returns test_metrics)
        
        Returns:
            Walk-forward result
        """
        # Check minimum samples
        if len(train_data) < self.config.min_train_samples:
            logger.warning(
                "insufficient_train_samples",
                window_index=window.window_index,
                samples=len(train_data),
                required=self.config.min_train_samples,
            )
            return WalkForwardResult(
                window_index=window.window_index,
                leakage_detected=True,
                leakage_score=1.0,
            )
        
        if len(test_data) < self.config.min_test_trades:
            logger.warning(
                "insufficient_test_samples",
                window_index=window.window_index,
                samples=len(test_data),
                required=self.config.min_test_trades,
            )
            return WalkForwardResult(
                window_index=window.window_index,
                leakage_detected=True,
                leakage_score=1.0,
            )
        
        # Train model
        try:
            model, train_metrics = train_fn(train_data)
        except Exception as e:
            logger.error(
                "training_failed",
                window_index=window.window_index,
                error=str(e),
            )
            return WalkForwardResult(
                window_index=window.window_index,
                leakage_detected=True,
                leakage_score=1.0,
            )
        
        # Evaluate on test set
        try:
            test_metrics = evaluate_fn(model, test_data)
        except Exception as e:
            logger.error(
                "evaluation_failed",
                window_index=window.window_index,
                error=str(e),
            )
            return WalkForwardResult(
                window_index=window.window_index,
                train_metrics=train_metrics,
                leakage_detected=True,
                leakage_score=1.0,
            )
        
        # Detect leakage
        leakage_detected, leakage_score = self.detect_leakage(
            train_metrics,
            test_metrics,
        )
        
        result = WalkForwardResult(
            window_index=window.window_index,
            train_metrics=train_metrics,
            test_metrics=test_metrics,
            leakage_detected=leakage_detected,
            leakage_score=leakage_score,
        )
        
        if leakage_detected:
            logger.warning(
                "leakage_detected",
                window_index=window.window_index,
                leakage_score=leakage_score,
                threshold=self.config.leakage_threshold,
            )
        
        return result
    
    def run_validation(
        self,
        data: pd.DataFrame,
        data_start: datetime,
        data_end: datetime,
        train_fn: Any,
        evaluate_fn: Any,
        timestamp_column: str = "timestamp",
    ) -> List[WalkForwardResult]:
        """
        Run full walk-forward validation.
        
        Args:
            data: Full dataset
            data_start: Start of available data
            data_end: End of available data
            train_fn: Training function
            evaluate_fn: Evaluation function
            timestamp_column: Name of timestamp column
        
        Returns:
            List of walk-forward results
        """
        # Generate windows
        windows = self.generate_windows(data_start, data_end)
        
        if len(windows) < self.config.min_windows:
            logger.error(
                "insufficient_windows_for_validation",
                generated=len(windows),
                required=self.config.min_windows,
            )
            return []
        
        results = []
        
        for window in windows:
            # Split data
            train_data, test_data = self.split_data(
                data,
                window,
                timestamp_column,
            )
            
            # Validate window
            result = self.validate_window(
                train_data,
                test_data,
                window,
                train_fn,
                evaluate_fn,
            )
            
            results.append(result)
        
        # Summary statistics
        num_windows = len(results)
        num_leakage = sum(1 for r in results if r.leakage_detected)
        avg_test_sharpe = np.mean([
            r.test_metrics.get("sharpe_ratio", 0.0)
            for r in results
            if not r.leakage_detected
        ])
        
        logger.info(
            "walk_forward_validation_complete",
            num_windows=num_windows,
            num_leakage_detected=num_leakage,
            avg_test_sharpe=avg_test_sharpe,
        )
        
        return results
    
    def aggregate_results(
        self,
        results: List[WalkForwardResult],
    ) -> Dict[str, Any]:
        """
        Aggregate results across all windows.
        
        Args:
            results: List of walk-forward results
        
        Returns:
            Aggregated statistics
        """
        # Filter out windows with leakage
        valid_results = [r for r in results if not r.leakage_detected]
        
        if not valid_results:
            return {
                "num_windows": len(results),
                "num_valid_windows": 0,
                "leakage_rate": 1.0,
            }
        
        # Aggregate test metrics
        test_sharpes = [r.test_metrics.get("sharpe_ratio", 0.0) for r in valid_results]
        test_hit_rates = [r.test_metrics.get("hit_rate", 0.0) for r in valid_results]
        test_edges = [r.test_metrics.get("edge_after_cost_bps", 0.0) for r in valid_results]
        
        return {
            "num_windows": len(results),
            "num_valid_windows": len(valid_results),
            "leakage_rate": (len(results) - len(valid_results)) / len(results),
            "avg_test_sharpe": np.mean(test_sharpes),
            "std_test_sharpe": np.std(test_sharpes),
            "avg_test_hit_rate": np.mean(test_hit_rates),
            "avg_test_edge_bps": np.mean(test_edges),
            "min_test_sharpe": np.min(test_sharpes),
            "max_test_sharpe": np.max(test_sharpes),
        }

