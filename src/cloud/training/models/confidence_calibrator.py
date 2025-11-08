"""
Confidence Calibration

Isotonic scaling by regime. Rejects poorly calibrated models at Council.

Key Features:
- Isotonic regression for calibration
- Regime-specific calibration
- Calibration metrics (Brier score, reliability)
- Model rejection for poor calibration
- Integration with Council

Author: Huracan Engine Team
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


class CalibrationStatus(Enum):
    """Calibration status"""
    WELL_CALIBRATED = "well_calibrated"
    MODERATELY_CALIBRATED = "moderately_calibrated"
    POORLY_CALIBRATED = "poorly_calibrated"
    REJECTED = "rejected"


@dataclass
class CalibrationResult:
    """Calibration result"""
    model_id: str
    regime: str
    brier_score: float
    reliability: float
    calibration_status: CalibrationStatus
    calibration_curve: Dict[float, float] = field(default_factory=dict)
    is_approved: bool = False


class ConfidenceCalibrator:
    """
    Confidence Calibration with Isotonic Scaling.
    
    Calibrates model confidence predictions using isotonic regression.
    Rejects poorly calibrated models.
    
    Usage:
        calibrator = ConfidenceCalibrator()
        
        # Calibrate model
        result = calibrator.calibrate(
            model_id="model_1",
            regime="trend",
            predictions=[0.8, 0.6, 0.9, ...],
            actuals=[1, 0, 1, ...]
        )
        
        if result.is_approved:
            # Use calibrated model
            calibrated_prediction = calibrator.predict(
                model_id="model_1",
                regime="trend",
                raw_confidence=0.75
            )
    """
    
    def __init__(
        self,
        min_brier_score: float = 0.25,  # Maximum Brier score for approval
        min_reliability: float = 0.7,  # Minimum reliability for approval
        num_bins: int = 10,  # Number of bins for calibration curve
        use_isotonic: bool = True  # Use isotonic regression
    ):
        """
        Initialize confidence calibrator.
        
        Args:
            min_brier_score: Maximum Brier score for approval (lower is better)
            min_reliability: Minimum reliability for approval (higher is better)
            num_bins: Number of bins for calibration curve
            use_isotonic: Use isotonic regression for calibration
        """
        self.min_brier_score = min_brier_score
        self.min_reliability = min_reliability
        self.num_bins = num_bins
        self.use_isotonic = use_isotonic
        
        # Store calibration models per model+regime
        self.calibration_models: Dict[str, Dict[str, any]] = {}  # model_id -> regime -> model
        
        # Store calibration data
        self.calibration_data: Dict[str, Dict[str, List[Tuple[float, float]]]] = {}  # model_id -> regime -> [(pred, actual)]
        
        logger.info(
            "confidence_calibrator_initialized",
            min_brier_score=min_brier_score,
            min_reliability=min_reliability,
            use_isotonic=use_isotonic
        )
    
    def calibrate(
        self,
        model_id: str,
        regime: str,
        predictions: List[float],
        actuals: List[int],  # 0 or 1
        fit_calibration: bool = True
    ) -> CalibrationResult:
        """
        Calibrate model confidence.
        
        Args:
            model_id: Model ID
            regime: Market regime
            predictions: Confidence predictions (0-1)
            actuals: Actual outcomes (0 or 1)
            fit_calibration: Fit calibration model
        
        Returns:
            CalibrationResult
        """
        if len(predictions) != len(actuals):
            raise ValueError("Predictions and actuals must have same length")
        
        if not predictions:
            return CalibrationResult(
                model_id=model_id,
                regime=regime,
                brier_score=1.0,
                reliability=0.0,
                calibration_status=CalibrationStatus.REJECTED,
                is_approved=False
            )
        
        # Calculate Brier score
        brier_score = self._calculate_brier_score(predictions, actuals)
        
        # Calculate reliability (calibration)
        reliability = self._calculate_reliability(predictions, actuals)
        
        # Build calibration curve
        calibration_curve = self._build_calibration_curve(predictions, actuals)
        
        # Determine calibration status
        calibration_status = self._determine_calibration_status(brier_score, reliability)
        
        # Check if approved
        is_approved = (
            calibration_status != CalibrationStatus.REJECTED and
            brier_score <= self.min_brier_score and
            reliability >= self.min_reliability
        )
        
        # Store calibration data
        if model_id not in self.calibration_data:
            self.calibration_data[model_id] = {}
        if regime not in self.calibration_data[model_id]:
            self.calibration_data[model_id][regime] = []
        
        self.calibration_data[model_id][regime].extend(list(zip(predictions, actuals)))
        
        # Fit calibration model if requested
        if fit_calibration and is_approved:
            self._fit_calibration_model(model_id, regime, predictions, actuals)
        
        result = CalibrationResult(
            model_id=model_id,
            regime=regime,
            brier_score=brier_score,
            reliability=reliability,
            calibration_status=calibration_status,
            calibration_curve=calibration_curve,
            is_approved=is_approved
        )
        
        logger.info(
            "model_calibrated",
            model_id=model_id,
            regime=regime,
            brier_score=brier_score,
            reliability=reliability,
            is_approved=is_approved
        )
        
        return result
    
    def predict(
        self,
        model_id: str,
        regime: str,
        raw_confidence: float
    ) -> float:
        """
        Predict calibrated confidence.
        
        Args:
            model_id: Model ID
            regime: Market regime
            raw_confidence: Raw confidence prediction
        
        Returns:
            Calibrated confidence
        """
        # Check if calibration model exists
        if model_id in self.calibration_models and regime in self.calibration_models[model_id]:
            calibration_model = self.calibration_models[model_id][regime]
            
            # Apply isotonic regression transformation
            if self.use_isotonic and calibration_model is not None:
                # Simplified: use linear interpolation from calibration curve
                calibrated = self._apply_isotonic_transform(raw_confidence, calibration_model)
                return float(np.clip(calibrated, 0.0, 1.0))
        
        # No calibration model: return raw confidence
        return float(np.clip(raw_confidence, 0.0, 1.0))
    
    def _calculate_brier_score(self, predictions: List[float], actuals: List[int]) -> float:
        """Calculate Brier score (lower is better)"""
        predictions_array = np.array(predictions)
        actuals_array = np.array(actuals)
        
        brier_score = np.mean((predictions_array - actuals_array) ** 2)
        return float(brier_score)
    
    def _calculate_reliability(self, predictions: List[float], actuals: List[int]) -> float:
        """Calculate reliability (calibration) score (higher is better)"""
        # Bin predictions
        bins = np.linspace(0, 1, self.num_bins + 1)
        bin_indices = np.digitize(predictions, bins) - 1
        bin_indices = np.clip(bin_indices, 0, self.num_bins - 1)
        
        # Calculate observed frequency per bin
        reliability_scores = []
        for i in range(self.num_bins):
            bin_mask = bin_indices == i
            if np.sum(bin_mask) > 0:
                predicted_freq = np.mean(predictions[bin_mask])
                observed_freq = np.mean(actuals[bin_mask])
                
                # Reliability: how close predicted frequency is to observed frequency
                reliability = 1.0 - abs(predicted_freq - observed_freq)
                reliability_scores.append(reliability)
        
        # Average reliability across bins
        if reliability_scores:
            return float(np.mean(reliability_scores))
        else:
            return 0.0
    
    def _build_calibration_curve(
        self,
        predictions: List[float],
        actuals: List[int]
    ) -> Dict[float, float]:
        """Build calibration curve"""
        # Bin predictions
        bins = np.linspace(0, 1, self.num_bins + 1)
        bin_indices = np.digitize(predictions, bins) - 1
        bin_indices = np.clip(bin_indices, 0, self.num_bins - 1)
        
        calibration_curve = {}
        for i in range(self.num_bins):
            bin_mask = bin_indices == i
            if np.sum(bin_mask) > 0:
                bin_center = (bins[i] + bins[i + 1]) / 2
                predicted_freq = np.mean(predictions[bin_mask])
                observed_freq = np.mean(actuals[bin_mask])
                calibration_curve[bin_center] = observed_freq
        
        return calibration_curve
    
    def _determine_calibration_status(
        self,
        brier_score: float,
        reliability: float
    ) -> CalibrationStatus:
        """Determine calibration status"""
        if brier_score > self.min_brier_score or reliability < self.min_reliability:
            return CalibrationStatus.REJECTED
        elif brier_score > self.min_brier_score * 0.8 or reliability < self.min_reliability * 1.1:
            return CalibrationStatus.POORLY_CALIBRATED
        elif brier_score > self.min_brier_score * 0.6 or reliability < self.min_reliability * 1.2:
            return CalibrationStatus.MODERATELY_CALIBRATED
        else:
            return CalibrationStatus.WELL_CALIBRATED
    
    def _fit_calibration_model(
        self,
        model_id: str,
        regime: str,
        predictions: List[float],
        actuals: List[int]
    ) -> None:
        """Fit isotonic regression model for calibration"""
        # Simplified isotonic regression
        # In production, would use sklearn.isotonic.IsotonicRegression
        
        # Sort by predictions
        sorted_indices = np.argsort(predictions)
        sorted_predictions = np.array(predictions)[sorted_indices]
        sorted_actuals = np.array(actuals)[sorted_indices]
        
        # Calculate cumulative means (isotonic regression)
        cumulative_actuals = np.cumsum(sorted_actuals)
        cumulative_counts = np.arange(1, len(sorted_actuals) + 1)
        isotonic_predictions = cumulative_actuals / cumulative_counts
        
        # Store calibration model (simplified: store mapping)
        if model_id not in self.calibration_models:
            self.calibration_models[model_id] = {}
        
        self.calibration_models[model_id][regime] = {
            "predictions": sorted_predictions.tolist(),
            "calibrated": isotonic_predictions.tolist()
        }
    
    def _apply_isotonic_transform(
        self,
        raw_confidence: float,
        calibration_model: Dict[str, List[float]]
    ) -> float:
        """Apply isotonic transformation"""
        predictions = calibration_model["predictions"]
        calibrated = calibration_model["calibrated"]
        
        # Linear interpolation
        if raw_confidence <= predictions[0]:
            return calibrated[0]
        elif raw_confidence >= predictions[-1]:
            return calibrated[-1]
        else:
            # Find interpolation point
            idx = np.searchsorted(predictions, raw_confidence)
            if idx == 0:
                return calibrated[0]
            elif idx >= len(predictions):
                return calibrated[-1]
            else:
                # Linear interpolation
                x0, x1 = predictions[idx - 1], predictions[idx]
                y0, y1 = calibrated[idx - 1], calibrated[idx]
                return y0 + (y1 - y0) * (raw_confidence - x0) / (x1 - x0)
    
    def get_calibration_stats(self) -> Dict[str, Dict[str, CalibrationResult]]:
        """Get calibration statistics for all models"""
        stats = {}
        
        for model_id, regime_data in self.calibration_data.items():
            stats[model_id] = {}
            for regime, data in regime_data.items():
                if data:
                    predictions, actuals = zip(*data)
                    result = self.calibrate(
                        model_id=model_id,
                        regime=regime,
                        predictions=list(predictions),
                        actuals=list(actuals),
                        fit_calibration=False
                    )
                    stats[model_id][regime] = result
        
        return stats
