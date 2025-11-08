"""
Mathematical Validation Framework

Validates models using mathematical principles:
- Cross-validation
- Statistical tests
- Bias-variance decomposition
- Generalization error estimation
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import structlog
from scipy import stats

from ..validation import BiasVarianceDiagnostics, CrossValidator

logger = structlog.get_logger(__name__)


class MathematicalValidator:
    """
    Mathematical validation framework.
    
    Validates models using:
    - Statistical hypothesis testing
    - Cross-validation
    - Bias-variance decomposition
    - Generalization error estimation
    """
    
    def __init__(self):
        """Initialize mathematical validator."""
        self.cross_validator = CrossValidator(cv_folds=5, use_time_series_split=True)
        logger.info("mathematical_validator_initialized")
    
    def validate_model_mathematically(
        self,
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Validate model using mathematical principles.
        
        Args:
            model: Trained model
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            X_test: Test features (optional)
            y_test: Test targets (optional)
            
        Returns:
            Dictionary with validation results
        """
        logger.info("validating_model_mathematically")
        
        validation_results = {}
        
        # 1. Cross-validation
        cv_results = self.cross_validator.cross_validate(model, X_train, y_train)
        validation_results["cross_validation"] = cv_results
        
        # 2. Bias-variance diagnostics
        bias_variance = self.cross_validator.bias_variance_diagnosis(
            model, X_train, y_train, X_val, y_val, X_test, y_test
        )
        validation_results["bias_variance"] = bias_variance.to_dict()
        
        # 3. Statistical tests
        statistical_tests = self._perform_statistical_tests(model, X_val, y_val)
        validation_results["statistical_tests"] = statistical_tests
        
        # 4. Generalization error estimation
        generalization_error = self._estimate_generalization_error(
            bias_variance.bias_score,
            bias_variance.variance_score,
        )
        validation_results["generalization_error"] = generalization_error
        
        # 5. Model stability test
        stability = self._test_model_stability(model, X_val, y_val)
        validation_results["stability"] = stability
        
        logger.info("mathematical_validation_complete")
        
        return validation_results
    
    def _perform_statistical_tests(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Dict[str, Any]:
        """Perform statistical tests on model predictions."""
        # Get predictions
        y_pred = model.predict(X)
        residuals = y - y_pred
        
        tests = {}
        
        # 1. Normality test (Shapiro-Wilk)
        if len(residuals) <= 5000:  # Shapiro-Wilk works for n <= 5000
            stat, p_value = stats.shapiro(residuals)
            tests["residual_normality"] = {
                "test": "shapiro_wilk",
                "statistic": float(stat),
                "p_value": float(p_value),
                "is_normal": p_value > 0.05,
            }
        else:
            # Use D'Agostino's test for larger samples
            stat, p_value = stats.normaltest(residuals)
            tests["residual_normality"] = {
                "test": "dagostino",
                "statistic": float(stat),
                "p_value": float(p_value),
                "is_normal": p_value > 0.05,
            }
        
        # 2. Homoscedasticity test (Levene's test on residuals)
        # Split residuals into groups
        n_groups = 5
        group_size = len(residuals) // n_groups
        groups = [residuals[i * group_size:(i + 1) * group_size] for i in range(n_groups)]
        
        stat, p_value = stats.levene(*groups)
        tests["homoscedasticity"] = {
            "test": "levene",
            "statistic": float(stat),
            "p_value": float(p_value),
            "is_homoscedastic": p_value > 0.05,
        }
        
        # 3. Independence test (autocorrelation)
        if len(residuals) > 1:
            autocorr = np.corrcoef(residuals[:-1], residuals[1:])[0, 1]
            tests["independence"] = {
                "test": "autocorrelation",
                "autocorrelation": float(autocorr),
                "is_independent": abs(autocorr) < 0.1,
            }
        
        logger.info("statistical_tests_complete")
        
        return tests
    
    def _estimate_generalization_error(
        self,
        bias: float,
        variance: float,
    ) -> Dict[str, float]:
        """Estimate generalization error using bias-variance decomposition."""
        # Expected squared error = Bias² + Variance + Irreducible Error
        irreducible_error = 0.01  # Can be estimated from data
        
        generalization_error = bias + variance + irreducible_error
        
        return {
            "bias": float(bias),
            "variance": float(variance),
            "irreducible_error": irreducible_error,
            "generalization_error": float(generalization_error),
        }
    
    def _test_model_stability(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        n_perturbations: int = 10,
    ) -> Dict[str, Any]:
        """Test model stability to small perturbations."""
        predictions = []
        
        for _ in range(n_perturbations):
            # Add small random noise
            epsilon = 1e-5
            X_perturbed = X + np.random.normal(0, epsilon, X.shape)
            
            try:
                y_pred = model.predict(X_perturbed)
                predictions.append(y_pred)
            except Exception:
                continue
        
        if len(predictions) > 0:
            predictions_array = np.array(predictions)
            
            # Calculate stability (lower variance = more stable)
            prediction_variance = np.var(predictions_array, axis=0)
            stability_score = 1.0 / (1.0 + np.mean(prediction_variance))
            
            return {
                "stability_score": float(stability_score),
                "prediction_variance": float(np.mean(prediction_variance)),
                "is_stable": stability_score > 0.9,
            }
        
        return {
            "stability_score": 0.0,
            "prediction_variance": 0.0,
            "is_stable": False,
        }

