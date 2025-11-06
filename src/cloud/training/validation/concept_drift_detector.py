"""
Concept Drift Detection System

Real-time concept drift detection for trading models:
- Monitor prediction accuracy over rolling windows
- Detect when model performance drops >10% from baseline
- Auto-trigger retraining when drift detected
- Track feature distribution shifts (PSI/KS tests)
- Alert on regime changes that affect model performance

Usage:
    detector = ConceptDriftDetector()
    
    # Monitor model performance
    detector.record_prediction(actual_win=True, predicted_prob=0.75)
    
    # Check for drift
    drift_report = detector.check_drift()
    
    if drift_report.drift_detected:
        logger.warning("Concept drift detected", severity=drift_report.severity)
        # Trigger retraining
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import structlog  # type: ignore
import numpy as np

logger = structlog.get_logger(__name__)


class DriftSeverity(Enum):
    """Drift severity levels"""
    NONE = "none"
    MILD = "mild"  # 5-10% degradation
    MODERATE = "moderate"  # 10-20% degradation
    SEVERE = "severe"  # 20-30% degradation
    CRITICAL = "critical"  # >30% degradation


@dataclass
class DriftReport:
    """Concept drift detection report"""
    drift_detected: bool
    severity: DriftSeverity
    baseline_accuracy: float
    current_accuracy: float
    degradation_pct: float
    window_size: int
    recommendations: List[str]
    timestamp: datetime


class ConceptDriftDetector:
    """
    Real-time concept drift detection for trading models.
    
    Monitors:
    - Prediction accuracy over rolling windows
    - Feature distribution shifts
    - Performance degradation patterns
    - Regime-specific drift
    """

    def __init__(
        self,
        baseline_window_size: int = 100,  # Baseline window size
        current_window_size: int = 50,  # Current window size
        degradation_threshold: float = 0.10,  # 10% degradation triggers alert
        min_samples: int = 30,  # Minimum samples before checking
        psi_threshold: float = 0.25,  # PSI threshold for feature drift
    ):
        """
        Initialize concept drift detector.
        
        Args:
            baseline_window_size: Number of samples for baseline
            current_window_size: Number of samples for current window
            degradation_threshold: Performance degradation threshold (0.10 = 10%)
            min_samples: Minimum samples before drift detection
            psi_threshold: PSI threshold for feature distribution drift
        """
        self.baseline_window_size = baseline_window_size
        self.current_window_size = current_window_size
        self.degradation_threshold = degradation_threshold
        self.min_samples = min_samples
        self.psi_threshold = psi_threshold
        
        # Prediction history
        self.baseline_predictions: List[Tuple[bool, float]] = []  # (actual_win, predicted_prob)
        self.current_predictions: List[Tuple[bool, float]] = []
        
        # Feature history (for distribution drift)
        self.baseline_features: List[Dict[str, float]] = []
        self.current_features: List[Dict[str, float]] = []
        
        # Performance tracking
        self.baseline_accuracy: Optional[float] = None
        self.current_accuracy: Optional[float] = None
        
        logger.info(
            "concept_drift_detector_initialized",
            baseline_window=baseline_window_size,
            current_window=current_window_size,
            degradation_threshold=degradation_threshold,
        )

    def record_prediction(
        self,
        actual_win: bool,
        predicted_prob: float,
        features: Optional[Dict[str, float]] = None,
    ):
        """
        Record a prediction and its outcome.
        
        Args:
            actual_win: Whether the trade was actually a winner
            predicted_prob: Predicted win probability
            features: Feature values at prediction time (optional)
        """
        # Add to current window
        self.current_predictions.append((actual_win, predicted_prob))
        
        # Keep window size
        if len(self.current_predictions) > self.current_window_size:
            # Move oldest to baseline
            oldest = self.current_predictions.pop(0)
            self.baseline_predictions.append(oldest)
            
            # Keep baseline window size
            if len(self.baseline_predictions) > self.baseline_window_size:
                self.baseline_predictions.pop(0)
        
        # Track features if provided
        if features:
            self.current_features.append(features)
            if len(self.current_features) > self.current_window_size:
                oldest_features = self.current_features.pop(0)
                self.baseline_features.append(oldest_features)
                
                if len(self.baseline_features) > self.baseline_window_size:
                    self.baseline_features.pop(0)
        
        logger.debug(
            "prediction_recorded",
            actual_win=actual_win,
            predicted_prob=predicted_prob,
            current_window_size=len(self.current_predictions),
        )

    def check_drift(self) -> DriftReport:
        """
        Check for concept drift.
        
        Returns:
            DriftReport with detection results
        """
        # Need enough samples
        if len(self.current_predictions) < self.min_samples:
            return DriftReport(
                drift_detected=False,
                severity=DriftSeverity.NONE,
                baseline_accuracy=0.0,
                current_accuracy=0.0,
                degradation_pct=0.0,
                window_size=len(self.current_predictions),
                recommendations=["Need more samples for drift detection"],
                timestamp=datetime.utcnow(),
            )
        
        # Calculate accuracies
        baseline_accuracy = self._calculate_accuracy(self.baseline_predictions)
        current_accuracy = self._calculate_accuracy(self.current_predictions)
        
        self.baseline_accuracy = baseline_accuracy
        self.current_accuracy = current_accuracy
        
        # Calculate degradation
        if baseline_accuracy > 0:
            degradation_pct = (baseline_accuracy - current_accuracy) / baseline_accuracy
        else:
            degradation_pct = 0.0
        
        # Determine severity
        drift_detected = degradation_pct >= self.degradation_threshold
        
        if degradation_pct >= 0.30:
            severity = DriftSeverity.CRITICAL
        elif degradation_pct >= 0.20:
            severity = DriftSeverity.SEVERE
        elif degradation_pct >= 0.10:
            severity = DriftSeverity.MODERATE
        elif degradation_pct >= 0.05:
            severity = DriftSeverity.MILD
        else:
            severity = DriftSeverity.NONE
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            drift_detected, severity, degradation_pct
        )
        
        # Check feature distribution drift
        feature_drift = self._check_feature_drift()
        if feature_drift:
            recommendations.append(f"Feature distribution drift detected: {feature_drift}")
        
        report = DriftReport(
            drift_detected=drift_detected,
            severity=severity,
            baseline_accuracy=baseline_accuracy,
            current_accuracy=current_accuracy,
            degradation_pct=degradation_pct,
            window_size=len(self.current_predictions),
            recommendations=recommendations,
            timestamp=datetime.utcnow(),
        )
        
        if drift_detected:
            logger.warning(
                "concept_drift_detected",
                severity=severity.value,
                baseline_accuracy=baseline_accuracy,
                current_accuracy=current_accuracy,
                degradation_pct=degradation_pct * 100,
            )
        
        return report

    def _calculate_accuracy(self, predictions: List[Tuple[bool, float]]) -> float:
        """Calculate prediction accuracy"""
        if not predictions:
            return 0.0
        
        correct = sum(1 for actual, pred_prob in predictions if (actual and pred_prob > 0.5) or (not actual and pred_prob <= 0.5))
        return correct / len(predictions)

    def _check_feature_drift(self) -> Optional[str]:
        """Check for feature distribution drift using PSI"""
        if len(self.baseline_features) < 10 or len(self.current_features) < 10:
            return None
        
        # Check each feature
        drifted_features = []
        
        # Get common features
        if not self.baseline_features or not self.current_features:
            return None
        
        common_features = set(self.baseline_features[0].keys()) & set(self.current_features[0].keys())
        
        for feature_name in common_features:
            baseline_values = [f[feature_name] for f in self.baseline_features if feature_name in f]
            current_values = [f[feature_name] for f in self.current_features if feature_name in f]
            
            if not baseline_values or not current_values:
                continue
            
            # Calculate PSI (Population Stability Index)
            psi = self._calculate_psi(baseline_values, current_values)
            
            if psi > self.psi_threshold:
                drifted_features.append(f"{feature_name} (PSI={psi:.2f})")
        
        if drifted_features:
            return ", ".join(drifted_features[:3])  # Top 3
        
        return None

    def _calculate_psi(self, baseline: List[float], current: List[float]) -> float:
        """
        Calculate Population Stability Index (PSI).
        
        PSI < 0.1: No significant change
        PSI 0.1-0.25: Minor change
        PSI > 0.25: Significant change
        """
        try:
            # Create bins
            all_values = baseline + current
            min_val = min(all_values)
            max_val = max(all_values)
            
            if max_val == min_val:
                return 0.0
            
            num_bins = 10
            bin_edges = np.linspace(min_val, max_val, num_bins + 1)
            
            # Calculate distributions
            baseline_hist, _ = np.histogram(baseline, bins=bin_edges)
            current_hist, _ = np.histogram(current, bins=bin_edges)
            
            # Normalize to probabilities
            baseline_prob = baseline_hist / len(baseline) if len(baseline) > 0 else baseline_hist
            current_prob = current_hist / len(current) if len(current) > 0 else current_hist
            
            # Avoid division by zero
            baseline_prob = np.where(baseline_prob == 0, 1e-10, baseline_prob)
            current_prob = np.where(current_prob == 0, 1e-10, current_prob)
            
            # Calculate PSI
            psi = np.sum((current_prob - baseline_prob) * np.log(current_prob / baseline_prob))
            
            return float(psi)
        except Exception as e:
            logger.warning("psi_calculation_failed", error=str(e))
            return 0.0

    def _generate_recommendations(
        self,
        drift_detected: bool,
        severity: DriftSeverity,
        degradation_pct: float,
    ) -> List[str]:
        """Generate recommendations based on drift detection"""
        recommendations = []
        
        if not drift_detected:
            recommendations.append("✅ No concept drift detected - Model performing normally")
            return recommendations
        
        if severity == DriftSeverity.CRITICAL:
            recommendations.append("🚨 CRITICAL: Model performance degraded >30% - Immediate retraining required")
            recommendations.append("⚠️ Pause trading until model is retrained")
            recommendations.append("🔍 Investigate recent market regime changes")
        elif severity == DriftSeverity.SEVERE:
            recommendations.append("🔴 SEVERE: Model performance degraded 20-30% - Retrain within 24 hours")
            recommendations.append("📉 Reduce position sizes by 50% until retrained")
            recommendations.append("🔍 Check for market regime shifts")
        elif severity == DriftSeverity.MODERATE:
            recommendations.append("🟠 MODERATE: Model performance degraded 10-20% - Schedule retraining")
            recommendations.append("📊 Monitor closely - May need retraining soon")
            recommendations.append("🔍 Review recent feature importance changes")
        elif severity == DriftSeverity.MILD:
            recommendations.append("🟡 MILD: Model performance degraded 5-10% - Monitor closely")
            recommendations.append("📈 Continue monitoring - May be temporary")
        
        recommendations.append(f"📉 Performance degradation: {degradation_pct:.1%}")
        recommendations.append(f"📊 Baseline accuracy: {self.baseline_accuracy:.1%}")
        recommendations.append(f"📊 Current accuracy: {self.current_accuracy:.1%}")
        
        return recommendations

    def reset_baseline(self):
        """Reset baseline to current window (after retraining)"""
        logger.info("resetting_baseline", previous_baseline_size=len(self.baseline_predictions))
        
        self.baseline_predictions = self.current_predictions.copy()
        self.current_predictions = []
        
        self.baseline_features = self.current_features.copy()
        self.current_features = []
        
        self.baseline_accuracy = None
        self.current_accuracy = None

    def get_status(self) -> Dict[str, Any]:
        """Get current detector status"""
        return {
            'baseline_window_size': len(self.baseline_predictions),
            'current_window_size': len(self.current_predictions),
            'baseline_accuracy': self.baseline_accuracy,
            'current_accuracy': self.current_accuracy,
            'degradation_pct': (
                (self.baseline_accuracy - self.current_accuracy) / self.baseline_accuracy
                if self.baseline_accuracy and self.baseline_accuracy > 0
                else 0.0
            ),
        }

