"""
Data Drift Detector - Retraining Trigger

Since "good data > good model," the Engine should prioritize retraining when
data drift exceeds a threshold. Compares statistical profile (mean, variance,
autocorrelation) of new data vs last training batch. If deviation > threshold,
trigger Mechanic retrain request automatically.

Author: Huracan Engine Team
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import polars as pl
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class DriftMetrics:
    """Data drift metrics."""
    mean_drift: float
    variance_drift: float
    autocorrelation_drift: float
    distribution_drift: float  # Kolmogorov-Smirnov or similar
    overall_drift_score: float  # 0-1, higher = more drift
    should_retrain: bool


@dataclass
class DriftReport:
    """Data drift detection report."""
    timestamp: datetime
    symbol: str
    metrics: DriftMetrics
    baseline_stats: Dict[str, float]
    current_stats: Dict[str, float]
    threshold: float


class DataDriftDetector:
    """
    Detects data drift and triggers retraining when needed.
    
    Usage:
        detector = DataDriftDetector(drift_threshold=0.2)
        
        # Set baseline from last training batch
        detector.set_baseline(training_data, symbol="BTC/USDT")
        
        # Check for drift in new data
        report = detector.detect_drift(new_data, symbol="BTC/USDT")
        
        if report.metrics.should_retrain:
            # Trigger Mechanic retrain request
            trigger_retrain(symbol)
    """
    
    def __init__(
        self,
        drift_threshold: float = 0.2,
        lookback_window: int = 100
    ):
        """
        Initialize data drift detector.
        
        Args:
            drift_threshold: Threshold for triggering retrain (0-1)
            lookback_window: Window size for statistical comparison
        """
        self.drift_threshold = drift_threshold
        self.lookback_window = lookback_window
        self.baselines: Dict[str, Dict[str, float]] = {}  # symbol -> stats
        
        logger.info(
            "data_drift_detector_initialized",
            drift_threshold=drift_threshold,
            lookback_window=lookback_window
        )
    
    def set_baseline(
        self,
        data: pl.DataFrame | pd.DataFrame,
        symbol: str,
        price_column: str = 'close'
    ) -> None:
        """
        Set baseline statistics from training data.
        
        Args:
            data: Training data used for last model training
            symbol: Trading symbol
            price_column: Name of price column
        """
        # Convert to pandas
        if isinstance(data, pl.DataFrame):
            df = data.to_pandas()
        else:
            df = data.copy()
        
        if price_column not in df.columns:
            raise ValueError(f"Price column '{price_column}' not found")
        
        # Calculate returns
        prices = df[price_column].values
        returns = np.diff(prices) / prices[:-1]
        
        # Calculate statistics
        stats = self._calculate_statistics(returns)
        
        self.baselines[symbol] = stats
        
        logger.info(
            "baseline_set",
            symbol=symbol,
            stats=stats
        )
    
    def detect_drift(
        self,
        data: pl.DataFrame | pd.DataFrame,
        symbol: str,
        price_column: str = 'close'
    ) -> DriftReport:
        """
        Detect data drift in new data.
        
        Args:
            data: New data to check
            symbol: Trading symbol
            price_column: Name of price column
        
        Returns:
            DriftReport with drift metrics and retrain recommendation
        """
        if symbol not in self.baselines:
            logger.warning("no_baseline_set", symbol=symbol)
            # Create baseline from current data
            self.set_baseline(data, symbol, price_column)
            return DriftReport(
                timestamp=datetime.now(),
                symbol=symbol,
                metrics=DriftMetrics(
                    mean_drift=0.0,
                    variance_drift=0.0,
                    autocorrelation_drift=0.0,
                    distribution_drift=0.0,
                    overall_drift_score=0.0,
                    should_retrain=False
                ),
                baseline_stats=self.baselines[symbol],
                current_stats=self.baselines[symbol],
                threshold=self.drift_threshold
            )
        
        # Convert to pandas
        if isinstance(data, pl.DataFrame):
            df = data.to_pandas()
        else:
            df = data.copy()
        
        if price_column not in df.columns:
            raise ValueError(f"Price column '{price_column}' not found")
        
        # Use recent window
        prices = df[price_column].tail(self.lookback_window).values
        returns = np.diff(prices) / prices[:-1]
        
        # Calculate current statistics
        current_stats = self._calculate_statistics(returns)
        baseline_stats = self.baselines[symbol]
        
        # Calculate drift metrics
        mean_drift = abs(current_stats['mean'] - baseline_stats['mean']) / (abs(baseline_stats['mean']) + 1e-8)
        variance_drift = abs(current_stats['variance'] - baseline_stats['variance']) / (baseline_stats['variance'] + 1e-8)
        autocorr_drift = abs(current_stats['autocorr'] - baseline_stats['autocorr'])
        
        # Distribution drift (Kolmogorov-Smirnov test)
        # Simplified: compare percentiles
        distribution_drift = self._calculate_distribution_drift(
            baseline_stats.get('percentiles', []),
            current_stats.get('percentiles', [])
        )
        
        # Overall drift score (weighted average)
        overall_drift = (
            0.3 * mean_drift +
            0.3 * variance_drift +
            0.2 * autocorr_drift +
            0.2 * distribution_drift
        )
        
        should_retrain = overall_drift > self.drift_threshold
        
        metrics = DriftMetrics(
            mean_drift=mean_drift,
            variance_drift=variance_drift,
            autocorrelation_drift=autocorr_drift,
            distribution_drift=distribution_drift,
            overall_drift_score=overall_drift,
            should_retrain=should_retrain
        )
        
        report = DriftReport(
            timestamp=datetime.now(),
            symbol=symbol,
            metrics=metrics,
            baseline_stats=baseline_stats,
            current_stats=current_stats,
            threshold=self.drift_threshold
        )
        
        logger.info(
            "drift_detected",
            symbol=symbol,
            overall_drift=overall_drift,
            should_retrain=should_retrain
        )
        
        return report
    
    def _calculate_statistics(self, returns: np.ndarray) -> Dict[str, float]:
        """Calculate statistical profile."""
        if len(returns) == 0:
            return {
                'mean': 0.0,
                'variance': 0.0,
                'std': 0.0,
                'autocorr': 0.0,
                'skewness': 0.0,
                'kurtosis': 0.0,
                'percentiles': []
            }
        
        mean = float(np.mean(returns))
        variance = float(np.var(returns))
        std = float(np.std(returns))
        
        # Autocorrelation (lag 1)
        if len(returns) > 1:
            autocorr = float(np.corrcoef(returns[:-1], returns[1:])[0, 1])
            if np.isnan(autocorr):
                autocorr = 0.0
        else:
            autocorr = 0.0
        
        # Skewness and kurtosis
        skewness = float(self._skewness(returns))
        kurtosis = float(self._kurtosis(returns))
        
        # Percentiles for distribution comparison
        percentiles = [float(np.percentile(returns, p)) for p in [10, 25, 50, 75, 90]]
        
        return {
            'mean': mean,
            'variance': variance,
            'std': std,
            'autocorr': autocorr,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'percentiles': percentiles
        }
    
    def _skewness(self, data: np.ndarray) -> float:
        """Calculate skewness."""
        if len(data) < 3:
            return 0.0
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 3)
    
    def _kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis."""
        if len(data) < 4:
            return 0.0
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 4) - 3.0
    
    def _calculate_distribution_drift(
        self,
        baseline_percentiles: List[float],
        current_percentiles: List[float]
    ) -> float:
        """Calculate distribution drift using percentile comparison."""
        if not baseline_percentiles or not current_percentiles:
            return 0.0
        
        if len(baseline_percentiles) != len(current_percentiles):
            return 1.0  # Maximum drift if lengths don't match
        
        # Calculate average absolute difference in percentiles
        differences = [
            abs(b - c) / (abs(b) + 1e-8)
            for b, c in zip(baseline_percentiles, current_percentiles)
        ]
        
        return float(np.mean(differences))
    
    def update_baseline(
        self,
        data: pl.DataFrame | pd.DataFrame,
        symbol: str,
        price_column: str = 'close'
    ) -> None:
        """Update baseline with new training data."""
        self.set_baseline(data, symbol, price_column)
        logger.info("baseline_updated", symbol=symbol)

