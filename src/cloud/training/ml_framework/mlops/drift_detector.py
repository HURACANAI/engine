"""
Drift Detection - Data and Concept Drift Detection

Detects data drift and concept drift for automated retraining.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import structlog
from scipy import stats

logger = structlog.get_logger(__name__)


class DriftDetector:
    """
    Drift detector for data and concept drift.
    
    Features:
    - Data distribution drift detection
    - Concept drift detection
    - Statistical tests (KS test, etc.)
    - Automated retraining triggers
    """
    
    def __init__(self, threshold: float = 0.05):
        """
        Initialize drift detector.
        
        Args:
            threshold: P-value threshold for drift detection
        """
        self.threshold = threshold
        self.reference_data: Optional[pd.DataFrame] = None
        
        logger.info("drift_detector_initialized", threshold=threshold)
    
    def set_reference(self, data: pd.DataFrame) -> None:
        """
        Set reference data distribution.
        
        Args:
            data: Reference data
        """
        self.reference_data = data.copy()
        logger.info("reference_data_set", shape=data.shape)
    
    def detect_data_drift(
        self,
        current_data: pd.DataFrame,
        method: str = "ks_test",
    ) -> Dict[str, Any]:
        """
        Detect data distribution drift.
        
        Args:
            current_data: Current data to test
            method: Detection method ("ks_test", "psi")
            
        Returns:
            Dictionary with drift detection results
        """
        if self.reference_data is None:
            raise ValueError("Reference data not set. Call set_reference() first.")
        
        logger.info("detecting_data_drift", method=method)
        
        numeric_cols = current_data.select_dtypes(include=[np.number]).columns
        drift_results = {}
        
        for col in numeric_cols:
            if col not in self.reference_data.columns:
                continue
            
            ref_values = self.reference_data[col].dropna()
            current_values = current_data[col].dropna()
            
            if method == "ks_test":
                # Kolmogorov-Smirnov test
                statistic, p_value = stats.ks_2samp(ref_values, current_values)
                is_drift = p_value < self.threshold
                
                drift_results[col] = {
                    "statistic": float(statistic),
                    "p_value": float(p_value),
                    "is_drift": is_drift,
                    "method": method,
                }
            
            elif method == "psi":
                # Population Stability Index
                psi = self._calculate_psi(ref_values, current_values)
                is_drift = psi > 0.2  # Threshold for PSI
                
                drift_results[col] = {
                    "psi": float(psi),
                    "is_drift": is_drift,
                    "method": method,
                }
        
        # Overall drift detection
        num_drifted_features = sum(1 for r in drift_results.values() if r["is_drift"])
        overall_drift = num_drifted_features > len(drift_results) * 0.1  # 10% threshold
        
        logger.info(
            "data_drift_detection_complete",
            num_drifted_features=num_drifted_features,
            overall_drift=overall_drift,
        )
        
        return {
            "drift_results": drift_results,
            "num_drifted_features": num_drifted_features,
            "overall_drift": overall_drift,
        }
    
    def _calculate_psi(self, reference: np.ndarray, current: np.ndarray) -> float:
        """Calculate Population Stability Index."""
        # Create bins
        bins = np.histogram_bin_edges(np.concatenate([reference, current]), bins=10)
        
        # Calculate distributions
        ref_dist, _ = np.histogram(reference, bins=bins)
        current_dist, _ = np.histogram(current, bins=bins)
        
        # Normalize
        ref_dist = ref_dist / len(reference)
        current_dist = current_dist / len(current)
        
        # Avoid division by zero
        ref_dist = ref_dist + 1e-8
        current_dist = current_dist + 1e-8
        
        # Calculate PSI
        psi = np.sum((current_dist - ref_dist) * np.log(current_dist / ref_dist))
        
        return psi
    
    def detect_concept_drift(
        self,
        y_pred: np.ndarray,
        y_true: np.ndarray,
        reference_performance: float,
    ) -> Dict[str, Any]:
        """
        Detect concept drift (performance degradation).
        
        Args:
            y_pred: Current predictions
            y_true: Current true values
            reference_performance: Reference performance metric
            
        Returns:
            Dictionary with concept drift detection results
        """
        logger.info("detecting_concept_drift")
        
        # Calculate current performance
        from sklearn.metrics import mean_squared_error, accuracy_score
        
        if len(np.unique(y_true)) > 2:
            # Regression
            current_performance = -mean_squared_error(y_true, y_pred)  # Negative for consistency
        else:
            # Classification
            current_performance = accuracy_score(y_true, y_pred)
        
        # Check for significant degradation
        performance_degradation = reference_performance - current_performance
        is_drift = performance_degradation > 0.1  # 10% threshold
        
        logger.info(
            "concept_drift_detection_complete",
            performance_degradation=performance_degradation,
            is_drift=is_drift,
        )
        
        return {
            "reference_performance": reference_performance,
            "current_performance": current_performance,
            "performance_degradation": performance_degradation,
            "is_drift": is_drift,
        }
    
    def should_retrain(
        self,
        current_data: pd.DataFrame,
        y_pred: Optional[np.ndarray] = None,
        y_true: Optional[np.ndarray] = None,
        reference_performance: Optional[float] = None,
    ) -> bool:
        """
        Determine if model should be retrained.
        
        Args:
            current_data: Current data
            y_pred: Current predictions (optional)
            y_true: Current true values (optional)
            reference_performance: Reference performance (optional)
            
        Returns:
            True if model should be retrained
        """
        # Check data drift
        data_drift = self.detect_data_drift(current_data)
        data_drift_detected = data_drift["overall_drift"]
        
        # Check concept drift
        concept_drift_detected = False
        if y_pred is not None and y_true is not None and reference_performance is not None:
            concept_drift = self.detect_concept_drift(y_pred, y_true, reference_performance)
            concept_drift_detected = concept_drift["is_drift"]
        
        should_retrain = data_drift_detected or concept_drift_detected
        
        logger.info(
            "retrain_decision",
            data_drift=data_drift_detected,
            concept_drift=concept_drift_detected,
            should_retrain=should_retrain,
        )
        
        return should_retrain

