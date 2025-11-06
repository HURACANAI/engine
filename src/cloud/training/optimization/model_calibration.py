"""
Model Calibration System

Calibrates model probabilities to be well-calibrated.
P(win) = 0.75 means 75% chance of winning (not just ranking).

Source: scikit-learn calibration best practices
Expected Impact: Better confidence estimates, improved decision-making
"""

from dataclasses import dataclass
from typing import Optional, Any
import structlog  # type: ignore
import numpy as np
import pandas as pd

try:
    from sklearn.calibration import (
        CalibratedClassifierCV,
        calibration_curve,
    )
    from sklearn.metrics import brier_score_loss
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

logger = structlog.get_logger(__name__)


@dataclass
class CalibrationResult:
    """Result of model calibration."""
    is_calibrated: bool
    calibration_method: str
    brier_score_before: float
    brier_score_after: float
    improvement_pct: float
    calibration_curve: Optional[Any] = None


class ModelCalibrator:
    """
    Calibrates model probabilities for better confidence estimates.
    
    Methods:
    - Isotonic: Non-parametric, more flexible
    - Sigmoid: Parametric, faster, works well for well-behaved models
    """

    def __init__(
        self,
        method: str = 'isotonic',  # 'isotonic' or 'sigmoid'
        cv: int = 5,
    ):
        """
        Initialize model calibrator.
        
        Args:
            method: Calibration method ('isotonic' or 'sigmoid')
            cv: Number of cross-validation folds
        """
        if not HAS_SKLEARN:
            raise ImportError("scikit-learn required: pip install scikit-learn")
        
        self.method = method
        self.cv = cv
        
        logger.info("model_calibrator_initialized", method=method, cv=cv)

    def calibrate(
        self,
        model: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> tuple[Any, CalibrationResult]:
        """
        Calibrate model probabilities.
        
        Args:
            model: Uncalibrated model
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            
        Returns:
            (calibrated_model, CalibrationResult)
        """
        # Check if model supports predict_proba
        if not hasattr(model, 'predict_proba'):
            logger.warning("model_does_not_support_calibration", model_type=type(model).__name__)
            return model, CalibrationResult(
                is_calibrated=False,
                calibration_method='none',
                brier_score_before=0.0,
                brier_score_after=0.0,
                improvement_pct=0.0,
            )
        
        # Calculate Brier score before calibration
        if X_val is not None and y_val is not None:
            y_proba_before = model.predict_proba(X_val)[:, 1]
            brier_before = brier_score_loss(y_val, y_proba_before)
        else:
            # Use training data
            y_proba_before = model.predict_proba(X_train)[:, 1]
            brier_before = brier_score_loss(y_train, y_proba_before)
        
        # Calibrate
        calibrated_model = CalibratedClassifierCV(
            base_estimator=model,
            method=self.method,
            cv=self.cv,
        )
        
        calibrated_model.fit(X_train, y_train)
        
        # Calculate Brier score after calibration
        if X_val is not None and y_val is not None:
            y_proba_after = calibrated_model.predict_proba(X_val)[:, 1]
            brier_after = brier_score_loss(y_val, y_proba_after)
        else:
            y_proba_after = calibrated_model.predict_proba(X_train)[:, 1]
            brier_after = brier_score_loss(y_train, y_proba_after)
        
        improvement_pct = ((brier_before - brier_after) / brier_before) * 100 if brier_before > 0 else 0.0
        
        # Calculate calibration curve
        if X_val is not None and y_val is not None:
            prob_true, prob_pred = calibration_curve(y_val, y_proba_after, n_bins=10)
        else:
            prob_true, prob_pred = calibration_curve(y_train, y_proba_after, n_bins=10)
        
        result = CalibrationResult(
            is_calibrated=True,
            calibration_method=self.method,
            brier_score_before=brier_before,
            brier_score_after=brier_after,
            improvement_pct=improvement_pct,
            calibration_curve=(prob_true, prob_pred),
        )
        
        logger.info(
            "model_calibration_complete",
            brier_before=brier_before,
            brier_after=brier_after,
            improvement_pct=improvement_pct,
        )
        
        return calibrated_model, result

