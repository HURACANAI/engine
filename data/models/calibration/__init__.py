"""
Confidence Calibration

Calibrates model confidence scores to match actual probabilities.

Methods:
- Isotonic Regression: Non-parametric, monotonic calibration
- Platt Scaling: Logistic regression calibration
- Temperature Scaling: Single parameter scaling

Metrics:
- Brier Score: Mean squared error of probabilities
- Calibration Curves: Visual assessment
- Expected Calibration Error (ECE)

Usage:
    from models.calibration import ConfidenceCalibrator

    calibrator = ConfidenceCalibrator(method='isotonic')

    # Fit on validation data
    calibrator.fit(
        confidence_scores=model.predict_proba(X_val),
        actual_outcomes=y_val
    )

    # Calibrate new predictions
    calibrated_probs = calibrator.calibrate(
        confidence_scores=model.predict_proba(X_test)
    )

    # Track performance
    brier_score = calibrator.calculate_brier_score(
        calibrated_probs, y_test
    )
"""

from .calibrator import (
    ConfidenceCalibrator,
    RegimeSpecificCalibrator,
    CalibrationMethod
)
from .metrics import (
    calculate_brier_score,
    calculate_ece,
    calculate_calibration_curve,
    calculate_calibration_by_regime
)
from .tracker import (
    CalibrationTracker,
    CalibrationSnapshot
)

__all__ = [
    # Calibrators
    "ConfidenceCalibrator",
    "RegimeSpecificCalibrator",
    "CalibrationMethod",

    # Metrics
    "calculate_brier_score",
    "calculate_ece",
    "calculate_calibration_curve",
    "calculate_calibration_by_regime",

    # Tracking
    "CalibrationTracker",
    "CalibrationSnapshot",
]
