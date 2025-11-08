"""
Walk-Forward Testing Automation

Automates walk-forward testing with rolling in-sample and out-of-sample segments.
Logs every segment's metrics to track stability across time.

Key Features:
- Rolling window backtests
- In-sample / out-of-sample splitting
- Segment metrics logging
- Stability tracking
- Integration with Log Book

Author: Huracan Engine Team
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional, Tuple, Callable, Any
from enum import Enum

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class WalkForwardSegment:
    """A single walk-forward segment"""
    segment_id: int
    in_sample_start: datetime
    in_sample_end: datetime
    out_of_sample_start: datetime
    out_of_sample_end: datetime
    in_sample_metrics: Dict[str, float] = field(default_factory=dict)
    out_of_sample_metrics: Dict[str, float] = field(default_factory=dict)
    stability_score: float = 0.0
    is_stable: bool = False


@dataclass
class WalkForwardResult:
    """Complete walk-forward test result"""
    model_id: str
    segments: List[WalkForwardSegment]
    overall_metrics: Dict[str, float]
    stability_metrics: Dict[str, float]
    is_robust: bool
    recommendation: str
    test_duration_seconds: float


class WalkForwardTester:
    """
    Automated walk-forward testing system.
    
    Splits data into rolling windows, trains on in-sample, tests on out-of-sample.
    Tracks stability across segments to validate model robustness.
    
    Usage:
        tester = WalkForwardTester(
            in_sample_days=180,
            out_of_sample_days=30,
            step_size_days=30
        )
        
        result = tester.test_model(
            model=my_model,
            data=train_data,
            train_fn=my_train_function,
            test_fn=my_test_function
        )
    """
    
    def __init__(
        self,
        in_sample_days: int = 180,
        out_of_sample_days: int = 30,
        step_size_days: int = 30,
        min_segments: int = 3
    ):
        """
        Initialize walk-forward tester.
        
        Args:
            in_sample_days: Number of days for in-sample training
            out_of_sample_days: Number of days for out-of-sample testing
            step_size_days: Step size for rolling window
            min_segments: Minimum number of segments required
        """
        self.in_sample_days = in_sample_days
        self.out_of_sample_days = out_of_sample_days
        self.step_size_days = step_size_days
        self.min_segments = min_segments
        
        logger.info(
            "walk_forward_tester_initialized",
            in_sample_days=in_sample_days,
            out_of_sample_days=out_of_sample_days,
            step_size_days=step_size_days,
            min_segments=min_segments
        )
    
    def test_model(
        self,
        model_id: str,
        data: Any,  # DataFrame or similar
        train_fn: Callable[[Any, datetime, datetime], Any],  # (data, start, end) -> model
        test_fn: Callable[[Any, Any, datetime, datetime], Dict[str, float]],  # (model, data, start, end) -> metrics
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> WalkForwardResult:
        """
        Run walk-forward test on model.
        
        Args:
            model_id: Model identifier
            data: Training data (must have timestamp column)
            train_fn: Function to train model on in-sample data
            test_fn: Function to test model on out-of-sample data
            start_date: Start date for testing (optional)
            end_date: End date for testing (optional)
        
        Returns:
            WalkForwardResult with all segments and metrics
        """
        import time
        start_time = time.time()
        
        logger.info("walk_forward_test_start", model_id=model_id)
        
        # Determine date range
        if start_date is None or end_date is None:
            start_date, end_date = self._get_date_range(data)
        
        # Generate segments
        segments = self._generate_segments(start_date, end_date)
        
        if len(segments) < self.min_segments:
            raise ValueError(
                f"Insufficient data for walk-forward test. "
                f"Need at least {self.min_segments} segments, got {len(segments)}"
            )
        
        # Test each segment
        for segment in segments:
            try:
                # Train on in-sample
                logger.info(
                    "walk_forward_segment_train",
                    segment_id=segment.segment_id,
                    in_sample_start=segment.in_sample_start,
                    in_sample_end=segment.in_sample_end
                )
                
                model = train_fn(
                    data,
                    segment.in_sample_start,
                    segment.in_sample_end
                )
                
                # Test on in-sample (for comparison)
                in_sample_metrics = test_fn(
                    model,
                    data,
                    segment.in_sample_start,
                    segment.in_sample_end
                )
                segment.in_sample_metrics = in_sample_metrics
                
                # Test on out-of-sample
                logger.info(
                    "walk_forward_segment_test",
                    segment_id=segment.segment_id,
                    out_of_sample_start=segment.out_of_sample_start,
                    out_of_sample_end=segment.out_of_sample_end
                )
                
                out_of_sample_metrics = test_fn(
                    model,
                    data,
                    segment.out_of_sample_start,
                    segment.out_of_sample_end
                )
                segment.out_of_sample_metrics = out_of_sample_metrics
                
                # Calculate stability score for this segment
                segment.stability_score = self._calculate_segment_stability(
                    in_sample_metrics,
                    out_of_sample_metrics
                )
                segment.is_stable = segment.stability_score >= 0.7
                
            except Exception as e:
                logger.error(
                    "walk_forward_segment_failed",
                    segment_id=segment.segment_id,
                    error=str(e)
                )
                continue
        
        # Calculate overall metrics
        overall_metrics = self._calculate_overall_metrics(segments)
        stability_metrics = self._calculate_stability_metrics(segments)
        
        # Determine if model is robust
        is_robust = self._is_robust(segments, stability_metrics)
        recommendation = self._get_recommendation(is_robust, stability_metrics)
        
        test_duration = time.time() - start_time
        
        result = WalkForwardResult(
            model_id=model_id,
            segments=segments,
            overall_metrics=overall_metrics,
            stability_metrics=stability_metrics,
            is_robust=is_robust,
            recommendation=recommendation,
            test_duration_seconds=test_duration
        )
        
        logger.info(
            "walk_forward_test_complete",
            model_id=model_id,
            num_segments=len(segments),
            is_robust=is_robust,
            recommendation=recommendation,
            test_duration=test_duration
        )
        
        return result
    
    def _generate_segments(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> List[WalkForwardSegment]:
        """Generate walk-forward segments"""
        segments = []
        segment_id = 0
        
        current_start = start_date
        
        while current_start < end_date:
            # In-sample period
            in_sample_end = current_start + timedelta(days=self.in_sample_days)
            
            # Out-of-sample period
            out_of_sample_start = in_sample_end
            out_of_sample_end = out_of_sample_start + timedelta(days=self.out_of_sample_days)
            
            # Check if we have enough data
            if out_of_sample_end > end_date:
                break
            
            segment = WalkForwardSegment(
                segment_id=segment_id,
                in_sample_start=current_start,
                in_sample_end=in_sample_end,
                out_of_sample_start=out_of_sample_start,
                out_of_sample_end=out_of_sample_end
            )
            
            segments.append(segment)
            segment_id += 1
            
            # Move to next segment
            current_start += timedelta(days=self.step_size_days)
        
        return segments
    
    def _get_date_range(self, data: Any) -> Tuple[datetime, datetime]:
        """Get date range from data"""
        # Assume data has a timestamp column
        # This is a placeholder - implement based on your data structure
        if hasattr(data, 'timestamp'):
            timestamps = data.timestamp
            start_date = min(timestamps)
            end_date = max(timestamps)
        elif hasattr(data, 'ts'):
            timestamps = data.ts
            start_date = min(timestamps)
            end_date = max(timestamps)
        else:
            # Default: use current date and go back
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=365)
        
        return start_date, end_date
    
    def _calculate_segment_stability(
        self,
        in_sample_metrics: Dict[str, float],
        out_of_sample_metrics: Dict[str, float]
    ) -> float:
        """Calculate stability score for a segment"""
        # Compare in-sample vs out-of-sample metrics
        # Lower degradation = higher stability
        
        key_metrics = ["sharpe_ratio", "total_return", "win_rate"]
        
        stability_scores = []
        for metric in key_metrics:
            if metric in in_sample_metrics and metric in out_of_sample_metrics:
                in_sample_val = in_sample_metrics[metric]
                out_of_sample_val = out_of_sample_metrics[metric]
                
                if in_sample_val != 0:
                    degradation = abs((out_of_sample_val - in_sample_val) / in_sample_val)
                    stability = max(0.0, 1.0 - degradation)
                    stability_scores.append(stability)
        
        return np.mean(stability_scores) if stability_scores else 0.5
    
    def _calculate_overall_metrics(
        self,
        segments: List[WalkForwardSegment]
    ) -> Dict[str, float]:
        """Calculate overall metrics across all segments"""
        # Aggregate out-of-sample metrics
        all_sharpes = [s.out_of_sample_metrics.get("sharpe_ratio", 0.0) for s in segments if s.out_of_sample_metrics]
        all_returns = [s.out_of_sample_metrics.get("total_return", 0.0) for s in segments if s.out_of_sample_metrics]
        all_win_rates = [s.out_of_sample_metrics.get("win_rate", 0.0) for s in segments if s.out_of_sample_metrics]
        
        return {
            "mean_sharpe": np.mean(all_sharpes) if all_sharpes else 0.0,
            "std_sharpe": np.std(all_sharpes) if all_sharpes else 0.0,
            "mean_return": np.mean(all_returns) if all_returns else 0.0,
            "mean_win_rate": np.mean(all_win_rates) if all_win_rates else 0.0,
            "num_segments": len(segments)
        }
    
    def _calculate_stability_metrics(
        self,
        segments: List[WalkForwardSegment]
    ) -> Dict[str, float]:
        """Calculate stability metrics"""
        stability_scores = [s.stability_score for s in segments]
        
        return {
            "mean_stability": np.mean(stability_scores) if stability_scores else 0.0,
            "std_stability": np.std(stability_scores) if stability_scores else 0.0,
            "min_stability": np.min(stability_scores) if stability_scores else 0.0,
            "max_stability": np.max(stability_scores) if stability_scores else 0.0,
            "stable_segments": sum(1 for s in segments if s.is_stable),
            "total_segments": len(segments)
        }
    
    def _is_robust(
        self,
        segments: List[WalkForwardSegment],
        stability_metrics: Dict[str, float]
    ) -> bool:
        """Determine if model is robust"""
        # Model is robust if:
        # 1. Mean stability >= 0.7
        # 2. At least 70% of segments are stable
        # 3. Std stability is low (consistent performance)
        
        mean_stability = stability_metrics.get("mean_stability", 0.0)
        stable_ratio = stability_metrics.get("stable_segments", 0) / stability_metrics.get("total_segments", 1)
        std_stability = stability_metrics.get("std_stability", 1.0)
        
        is_robust = (
            mean_stability >= 0.7 and
            stable_ratio >= 0.7 and
            std_stability <= 0.2
        )
        
        return is_robust
    
    def _get_recommendation(
        self,
        is_robust: bool,
        stability_metrics: Dict[str, float]
    ) -> str:
        """Get recommendation based on walk-forward results"""
        if is_robust:
            return "APPROVE"
        elif stability_metrics.get("mean_stability", 0.0) >= 0.6:
            return "CONDITIONAL_APPROVE"
        elif stability_metrics.get("mean_stability", 0.0) >= 0.4:
            return "MONITOR"
        else:
            return "REJECT"

