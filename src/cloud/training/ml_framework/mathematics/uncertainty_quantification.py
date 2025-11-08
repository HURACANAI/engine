"""
Uncertainty Quantification

Provides confidence scores and uncertainty measures for every prediction.
Uses statistical and probabilistic methods.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import structlog
from scipy import stats

logger = structlog.get_logger(__name__)


class UncertaintyQuantifier:
    """
    Uncertainty quantification for predictions.
    
    Methods:
    - Prediction intervals (using residuals)
    - Confidence intervals (using statistical tests)
    - Bootstrap intervals
    - Ensemble uncertainty (variance across models)
    """
    
    def __init__(self):
        """Initialize uncertainty quantifier."""
        logger.info("uncertainty_quantifier_initialized")
    
    def prediction_interval(
        self,
        y_pred: np.ndarray,
        residuals: np.ndarray,
        confidence_level: float = 0.95,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate prediction intervals using residuals.
        
        Args:
            y_pred: Predictions
            residuals: Prediction errors (y_true - y_pred)
            confidence_level: Confidence level (e.g., 0.95)
            
        Returns:
            Tuple of (lower_bound, upper_bound) arrays
        """
        # Calculate residual standard error
        residual_std = np.std(residuals)
        n = len(residuals)
        
        # Use t-distribution for prediction intervals
        t_critical = stats.t.ppf(0.5 + confidence_level / 2, n - 1)
        
        # Prediction interval: y_pred ± t * sqrt(MSE * (1 + 1/n))
        margin = t_critical * residual_std * np.sqrt(1 + 1 / n)
        
        lower_bound = y_pred - margin
        upper_bound = y_pred + margin
        
        logger.info(
            "prediction_interval_calculated",
            confidence_level=confidence_level,
            margin_mean=np.mean(margin),
        )
        
        return lower_bound, upper_bound
    
    def confidence_interval(
        self,
        values: np.ndarray,
        confidence_level: float = 0.95,
        method: str = "t_test",
    ) -> Tuple[float, float]:
        """
        Calculate confidence interval for a set of values.
        
        Args:
            values: Array of values
            confidence_level: Confidence level
            method: Method ("t_test", "bootstrap", "normal")
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        if method == "t_test":
            # Using t-distribution
            n = len(values)
            mean_val = np.mean(values)
            std_val = np.std(values, ddof=1)  # Sample standard deviation
            
            t_critical = stats.t.ppf(0.5 + confidence_level / 2, n - 1)
            margin = t_critical * std_val / np.sqrt(n)
            
            lower_bound = mean_val - margin
            upper_bound = mean_val + margin
        
        elif method == "normal":
            # Using normal distribution
            mean_val = np.mean(values)
            std_val = np.std(values)
            z_critical = stats.norm.ppf(0.5 + confidence_level / 2)
            margin = z_critical * std_val / np.sqrt(len(values))
            
            lower_bound = mean_val - margin
            upper_bound = mean_val + margin
        
        else:
            # Default to t-test
            return self.confidence_interval(values, confidence_level, "t_test")
        
        logger.info(
            "confidence_interval_calculated",
            method=method,
            confidence_level=confidence_level,
            interval=(lower_bound, upper_bound),
        )
        
        return (float(lower_bound), float(upper_bound))
    
    def bootstrap_confidence_interval(
        self,
        values: np.ndarray,
        confidence_level: float = 0.95,
        n_bootstrap: int = 1000,
    ) -> Tuple[float, float]:
        """
        Calculate confidence interval using bootstrap.
        
        Args:
            values: Array of values
            confidence_level: Confidence level
            n_bootstrap: Number of bootstrap samples
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        bootstrap_means = []
        
        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(values, size=len(values), replace=True)
            bootstrap_means.append(np.mean(bootstrap_sample))
        
        alpha = 1 - confidence_level
        lower_bound = np.percentile(bootstrap_means, 100 * alpha / 2)
        upper_bound = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
        
        logger.info(
            "bootstrap_confidence_interval_calculated",
            n_bootstrap=n_bootstrap,
            interval=(lower_bound, upper_bound),
        )
        
        return (float(lower_bound), float(upper_bound))
    
    def ensemble_uncertainty(
        self,
        predictions: List[np.ndarray],
    ) -> Dict[str, float]:
        """
        Calculate uncertainty from ensemble predictions.
        
        Args:
            predictions: List of prediction arrays from different models
            
        Returns:
            Dictionary with uncertainty metrics
        """
        predictions_array = np.array(predictions)
        
        # Mean prediction
        mean_pred = np.mean(predictions_array, axis=0)
        
        # Variance across models (epistemic uncertainty)
        model_variance = np.var(predictions_array, axis=0)
        avg_model_variance = np.mean(model_variance)
        
        # Standard deviation
        std_pred = np.std(mean_pred)
        
        # Coefficient of variation
        cv = std_pred / (np.abs(np.mean(mean_pred)) + 1e-8)
        
        uncertainty_metrics = {
            "mean_prediction": float(np.mean(mean_pred)),
            "std_prediction": float(std_pred),
            "model_variance": float(avg_model_variance),
            "coefficient_of_variation": float(cv),
            "uncertainty_score": float(avg_model_variance + std_pred),  # Combined uncertainty
        }
        
        logger.info("ensemble_uncertainty_calculated", **uncertainty_metrics)
        
        return uncertainty_metrics
    
    def quantify_uncertainty(
        self,
        y_pred: np.ndarray,
        y_true: Optional[np.ndarray] = None,
        ensemble_predictions: Optional[List[np.ndarray]] = None,
    ) -> Dict[str, Any]:
        """
        Quantify uncertainty comprehensively.
        
        Args:
            y_pred: Predictions
            y_true: True values (optional)
            ensemble_predictions: Ensemble predictions (optional)
            
        Returns:
            Dictionary with all uncertainty metrics
        """
        uncertainty_results = {
            "prediction_mean": float(np.mean(y_pred)),
            "prediction_std": float(np.std(y_pred)),
            "prediction_variance": float(np.var(y_pred)),
        }
        
        # Prediction intervals (if true values available)
        if y_true is not None:
            residuals = y_true - y_pred
            lower_bound, upper_bound = self.prediction_interval(y_pred, residuals)
            uncertainty_results["prediction_interval"] = {
                "lower": lower_bound.tolist(),
                "upper": upper_bound.tolist(),
            }
            uncertainty_results["residual_std"] = float(np.std(residuals))
        
        # Confidence interval
        ci_lower, ci_upper = self.confidence_interval(y_pred)
        uncertainty_results["confidence_interval"] = {
            "lower": ci_lower,
            "upper": ci_upper,
            "width": ci_upper - ci_lower,
        }
        
        # Ensemble uncertainty (if available)
        if ensemble_predictions:
            ensemble_uncertainty = self.ensemble_uncertainty(ensemble_predictions)
            uncertainty_results["ensemble_uncertainty"] = ensemble_uncertainty
        
        # Overall uncertainty score (0-1, higher = more uncertain)
        uncertainty_score = min(1.0, uncertainty_results["prediction_std"] / (abs(uncertainty_results["prediction_mean"]) + 1e-8))
        uncertainty_results["uncertainty_score"] = uncertainty_score
        
        logger.info("uncertainty_quantified", uncertainty_score=uncertainty_score)
        
        return uncertainty_results

