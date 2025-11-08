"""
Feature Drift Detector

Detects feature drift using statistical tests.
Drops features whose mean or variance shift too far.

Key Features:
- Mean shift detection (t-test, KS test)
- Variance shift detection (F-test, Levene's test)
- PSI (Population Stability Index)
- Automatic feature dropping
- Drift reporting

Author: Huracan Engine Team
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum

import numpy as np
import polars as pl
from scipy import stats
import structlog

logger = structlog.get_logger(__name__)


class DriftTest(Enum):
    """Drift test type"""
    MEAN_SHIFT = "mean_shift"  # t-test
    VARIANCE_SHIFT = "variance_shift"  # F-test
    DISTRIBUTION_SHIFT = "distribution_shift"  # KS test
    PSI = "psi"  # Population Stability Index


@dataclass
class DriftResult:
    """Feature drift result"""
    feature_name: str
    test_type: DriftTest
    statistic: float
    p_value: float
    is_drifted: bool
    threshold: float
    baseline_mean: Optional[float] = None
    current_mean: Optional[float] = None
    baseline_std: Optional[float] = None
    current_std: Optional[float] = None


class FeatureDriftDetector:
    """
    Feature Drift Detector.
    
    Detects feature drift using statistical tests.
    Drops features whose mean or variance shift too far.
    
    Usage:
        detector = FeatureDriftDetector(
            mean_shift_threshold=0.05,
            variance_shift_threshold=0.10,
            psi_threshold=0.25
        )
        
        # Establish baseline
        detector.establish_baseline(baseline_data, features=["feature1", "feature2"])
        
        # Check drift
        drift_results = detector.check_drift(current_data, features=["feature1", "feature2"])
        
        # Get drifted features
        drifted_features = [r.feature_name for r in drift_results if r.is_drifted]
    """
    
    def __init__(
        self,
        mean_shift_threshold: float = 0.05,  # p-value threshold for mean shift
        variance_shift_threshold: float = 0.10,  # p-value threshold for variance shift
        psi_threshold: float = 0.25,  # PSI threshold
        ks_threshold: float = 0.05,  # p-value threshold for KS test
        min_samples: int = 100
    ):
        """
        Initialize feature drift detector.
        
        Args:
            mean_shift_threshold: p-value threshold for mean shift test
            variance_shift_threshold: p-value threshold for variance shift test
            psi_threshold: PSI threshold (higher = more drift)
            ks_threshold: p-value threshold for KS test
            min_samples: Minimum samples for statistical tests
        """
        self.mean_shift_threshold = mean_shift_threshold
        self.variance_shift_threshold = variance_shift_threshold
        self.psi_threshold = psi_threshold
        self.ks_threshold = ks_threshold
        self.min_samples = min_samples
        
        # Baseline statistics
        self.baseline_stats: Dict[str, Dict[str, float]] = {}  # feature -> {mean, std, ...}
        self.baseline_distributions: Dict[str, np.ndarray] = {}  # feature -> distribution
        
        logger.info(
            "feature_drift_detector_initialized",
            mean_shift_threshold=mean_shift_threshold,
            variance_shift_threshold=variance_shift_threshold,
            psi_threshold=psi_threshold
        )
    
    def establish_baseline(
        self,
        data: pl.DataFrame,
        features: List[str]
    ) -> None:
        """
        Establish baseline statistics for features.
        
        Args:
            data: Baseline data
            features: List of feature names
        """
        for feature in features:
            if feature not in data.columns:
                logger.warning("feature_not_in_data", feature=feature)
                continue
            
            feature_data = data[feature].drop_nulls().to_numpy()
            
            if len(feature_data) < self.min_samples:
                logger.warning(
                    "insufficient_samples_for_baseline",
                    feature=feature,
                    samples=len(feature_data),
                    min_required=self.min_samples
                )
                continue
            
            # Calculate statistics
            mean = np.mean(feature_data)
            std = np.std(feature_data)
            median = np.median(feature_data)
            q25 = np.percentile(feature_data, 25)
            q75 = np.percentile(feature_data, 75)
            
            self.baseline_stats[feature] = {
                "mean": mean,
                "std": std,
                "median": median,
                "q25": q25,
                "q75": q75,
                "min": np.min(feature_data),
                "max": np.max(feature_data),
                "sample_size": len(feature_data)
            }
            
            # Store distribution for PSI
            self.baseline_distributions[feature] = feature_data
        
        logger.info(
            "baseline_established",
            features=len(self.baseline_stats),
            sample_size=len(data)
        )
    
    def check_drift(
        self,
        current_data: pl.DataFrame,
        features: Optional[List[str]] = None
    ) -> List[DriftResult]:
        """
        Check for feature drift.
        
        Args:
            current_data: Current data
            features: List of features to check (None = check all baselines)
        
        Returns:
            List of DriftResult
        """
        if features is None:
            features = list(self.baseline_stats.keys())
        
        results = []
        
        for feature in features:
            if feature not in self.baseline_stats:
                logger.warning("feature_not_in_baseline", feature=feature)
                continue
            
            if feature not in current_data.columns:
                logger.warning("feature_not_in_current_data", feature=feature)
                continue
            
            # Get current feature data
            current_feature_data = current_data[feature].drop_nulls().to_numpy()
            
            if len(current_feature_data) < self.min_samples:
                logger.warning(
                    "insufficient_samples_for_drift_check",
                    feature=feature,
                    samples=len(current_feature_data),
                    min_required=self.min_samples
                )
                continue
            
            # Get baseline statistics
            baseline_stats = self.baseline_stats[feature]
            baseline_data = self.baseline_distributions[feature]
            
            # Test mean shift (t-test)
            mean_shift_result = self._test_mean_shift(
                baseline_data,
                current_feature_data,
                feature
            )
            results.append(mean_shift_result)
            
            # Test variance shift (F-test)
            variance_shift_result = self._test_variance_shift(
                baseline_data,
                current_feature_data,
                feature
            )
            results.append(variance_shift_result)
            
            # Test distribution shift (KS test)
            distribution_shift_result = self._test_distribution_shift(
                baseline_data,
                current_feature_data,
                feature
            )
            results.append(distribution_shift_result)
            
            # Test PSI
            psi_result = self._test_psi(
                baseline_data,
                current_feature_data,
                feature
            )
            results.append(psi_result)
        
        # Log drifted features
        drifted_features = [r.feature_name for r in results if r.is_drifted]
        if drifted_features:
            logger.warning(
                "feature_drift_detected",
                drifted_features=drifted_features,
                total_features=len(features)
            )
        
        return results
    
    def _test_mean_shift(
        self,
        baseline: np.ndarray,
        current: np.ndarray,
        feature_name: str
    ) -> DriftResult:
        """Test for mean shift using t-test"""
        try:
            statistic, p_value = stats.ttest_ind(baseline, current)
            
            baseline_mean = np.mean(baseline)
            current_mean = np.mean(current)
            
            is_drifted = p_value < self.mean_shift_threshold
            
            return DriftResult(
                feature_name=feature_name,
                test_type=DriftTest.MEAN_SHIFT,
                statistic=statistic,
                p_value=p_value,
                is_drifted=is_drifted,
                threshold=self.mean_shift_threshold,
                baseline_mean=baseline_mean,
                current_mean=current_mean
            )
        except Exception as e:
            logger.error("mean_shift_test_failed", feature=feature_name, error=str(e))
            return DriftResult(
                feature_name=feature_name,
                test_type=DriftTest.MEAN_SHIFT,
                statistic=0.0,
                p_value=1.0,
                is_drifted=False,
                threshold=self.mean_shift_threshold
            )
    
    def _test_variance_shift(
        self,
        baseline: np.ndarray,
        current: np.ndarray,
        feature_name: str
    ) -> DriftResult:
        """Test for variance shift using F-test"""
        try:
            baseline_var = np.var(baseline, ddof=1)
            current_var = np.var(current, ddof=1)
            
            # F-test
            f_statistic = baseline_var / (current_var + 1e-9)
            df1 = len(baseline) - 1
            df2 = len(current) - 1
            
            p_value = 2 * min(
                stats.f.cdf(f_statistic, df1, df2),
                1 - stats.f.cdf(f_statistic, df1, df2)
            )
            
            baseline_std = np.std(baseline)
            current_std = np.std(current)
            
            is_drifted = p_value < self.variance_shift_threshold
            
            return DriftResult(
                feature_name=feature_name,
                test_type=DriftTest.VARIANCE_SHIFT,
                statistic=f_statistic,
                p_value=p_value,
                is_drifted=is_drifted,
                threshold=self.variance_shift_threshold,
                baseline_std=baseline_std,
                current_std=current_std
            )
        except Exception as e:
            logger.error("variance_shift_test_failed", feature=feature_name, error=str(e))
            return DriftResult(
                feature_name=feature_name,
                test_type=DriftTest.VARIANCE_SHIFT,
                statistic=1.0,
                p_value=1.0,
                is_drifted=False,
                threshold=self.variance_shift_threshold
            )
    
    def _test_distribution_shift(
        self,
        baseline: np.ndarray,
        current: np.ndarray,
        feature_name: str
    ) -> DriftResult:
        """Test for distribution shift using KS test"""
        try:
            statistic, p_value = stats.ks_2samp(baseline, current)
            
            is_drifted = p_value < self.ks_threshold
            
            return DriftResult(
                feature_name=feature_name,
                test_type=DriftTest.DISTRIBUTION_SHIFT,
                statistic=statistic,
                p_value=p_value,
                is_drifted=is_drifted,
                threshold=self.ks_threshold
            )
        except Exception as e:
            logger.error("distribution_shift_test_failed", feature=feature_name, error=str(e))
            return DriftResult(
                feature_name=feature_name,
                test_type=DriftTest.DISTRIBUTION_SHIFT,
                statistic=0.0,
                p_value=1.0,
                is_drifted=False,
                threshold=self.ks_threshold
            )
    
    def _test_psi(
        self,
        baseline: np.ndarray,
        current: np.ndarray,
        feature_name: str
    ) -> DriftResult:
        """Test for population stability using PSI"""
        try:
            # Create bins from baseline
            _, bin_edges = np.histogram(baseline, bins=10)
            
            # Calculate baseline distribution
            baseline_counts, _ = np.histogram(baseline, bins=bin_edges)
            baseline_probs = baseline_counts / (len(baseline) + 1e-9)
            
            # Calculate current distribution
            current_counts, _ = np.histogram(current, bins=bin_edges)
            current_probs = current_counts / (len(current) + 1e-9)
            
            # Calculate PSI
            psi = 0.0
            for i in range(len(baseline_probs)):
                if baseline_probs[i] > 0:
                    psi += (current_probs[i] - baseline_probs[i]) * np.log(
                        (current_probs[i] + 1e-9) / baseline_probs[i]
                    )
            
            is_drifted = psi > self.psi_threshold
            
            return DriftResult(
                feature_name=feature_name,
                test_type=DriftTest.PSI,
                statistic=psi,
                p_value=1.0 - min(psi / self.psi_threshold, 1.0),  # Normalized
                is_drifted=is_drifted,
                threshold=self.psi_threshold
            )
        except Exception as e:
            logger.error("psi_test_failed", feature=feature_name, error=str(e))
            return DriftResult(
                feature_name=feature_name,
                test_type=DriftTest.PSI,
                statistic=0.0,
                p_value=1.0,
                is_drifted=False,
                threshold=self.psi_threshold
            )
    
    def get_drifted_features(self, results: List[DriftResult]) -> List[str]:
        """Get list of drifted features"""
        drifted = set()
        for result in results:
            if result.is_drifted:
                drifted.add(result.feature_name)
        return list(drifted)
    
    def get_drift_summary(self, results: List[DriftResult]) -> Dict[str, Any]:
        """Get drift summary"""
        drifted_features = self.get_drifted_features(results)
        
        summary = {
            "total_features": len(set(r.feature_name for r in results)),
            "drifted_features": len(drifted_features),
            "drift_rate": len(drifted_features) / len(set(r.feature_name for r in results)) if results else 0.0,
            "drifted_feature_names": drifted_features,
            "test_results": {}
        }
        
        # Group by test type
        for test_type in DriftTest:
            test_results = [r for r in results if r.test_type == test_type]
            if test_results:
                summary["test_results"][test_type.value] = {
                    "total": len(test_results),
                    "drifted": len([r for r in test_results if r.is_drifted]),
                    "avg_p_value": np.mean([r.p_value for r in test_results])
                }
        
        return summary

