"""
Continuous Learning Cycle

Implements the continuous learning cycle with mathematical reasoning:
- What assumption am I making mathematically?
- Does data support that assumption?
- Is my variance increasing or bias accumulating?
- How can I minimize the loss function better?
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import structlog

from .reasoning_engine import MathematicalReasoningEngine, PredictionWithReasoning

logger = structlog.get_logger(__name__)


class ContinuousLearningCycle:
    """
    Continuous learning cycle with mathematical reasoning.
    
    At every iteration, the system asks:
    1. What assumption am I making mathematically?
    2. Does data support that assumption?
    3. Is my variance increasing or bias accumulating?
    4. How can I minimize the loss function better?
    """
    
    def __init__(self):
        """Initialize continuous learning cycle."""
        self.reasoning_engine = MathematicalReasoningEngine()
        self.learning_history: List[Dict[str, Any]] = []
        
        logger.info("continuous_learning_cycle_initialized")
    
    def iterate(
        self,
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        iteration: int = 0,
    ) -> Dict[str, Any]:
        """
        Perform one iteration of continuous learning cycle.
        
        Args:
            model: Current model
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            iteration: Current iteration number
            
        Returns:
            Dictionary with learning cycle results
        """
        logger.info("continuous_learning_iteration", iteration=iteration)
        
        # Get predictions
        y_pred_train = model.predict(X_train)
        y_pred_val = model.predict(X_val)
        
        # 1. What assumption am I making mathematically?
        assumptions = self._identify_assumptions(model, X_train, y_train)
        
        # 2. Does data support that assumption?
        data_support = self._check_data_support(assumptions, X_train, y_train, y_pred_train)
        
        # 3. Is my variance increasing or bias accumulating?
        bias_variance_analysis = self._analyze_bias_variance(
            model, X_train, y_train, X_val, y_val, y_pred_train, y_pred_val
        )
        
        # 4. How can I minimize the loss function better?
        optimization_suggestions = self._suggest_optimizations(
            model, bias_variance_analysis, data_support
        )
        
        # Generate mathematical reasoning
        prediction_reasoning = self.reasoning_engine.reason_about_prediction(
            model, X_val, y_pred_val, y_val
        )
        
        # Compile results
        iteration_results = {
            "iteration": iteration,
            "assumptions": assumptions,
            "data_support": data_support,
            "bias_variance": bias_variance_analysis,
            "optimization_suggestions": optimization_suggestions,
            "prediction_reasoning": {
                "confidence_score": prediction_reasoning.confidence_score,
                "uncertainty_measure": prediction_reasoning.uncertainty_measure,
                "bias_estimate": prediction_reasoning.bias_estimate,
                "variance_estimate": prediction_reasoning.variance_estimate,
                "generalization_score": prediction_reasoning.generalization_score,
            },
            "explanation": self.reasoning_engine.explain_prediction(prediction_reasoning),
        }
        
        self.learning_history.append(iteration_results)
        
        logger.info(
            "continuous_learning_iteration_complete",
            iteration=iteration,
            generalization_score=prediction_reasoning.generalization_score,
        )
        
        return iteration_results
    
    def _identify_assumptions(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
    ) -> List[Dict[str, Any]]:
        """Identify mathematical assumptions made by the model."""
        assumptions = []
        
        # Assumption 1: Linearity (for linear models)
        if hasattr(model, "coef_") or "linear" in str(type(model)).lower():
            assumptions.append({
                "assumption": "Linear relationship between features and target",
                "mathematical_principle": "linear_algebra",
                "equation": "y = w^T * x + b",
            })
        
        # Assumption 2: Normal distribution of errors
        assumptions.append({
            "assumption": "Prediction errors are normally distributed",
            "mathematical_principle": "statistics",
            "equation": "ε ~ N(0, σ²)",
        })
        
        # Assumption 3: Independent and identically distributed data
        assumptions.append({
            "assumption": "Data points are independent and identically distributed",
            "mathematical_principle": "statistics",
            "equation": "P(X₁, X₂, ..., Xₙ) = ∏ P(Xᵢ)",
        })
        
        # Assumption 4: Stationarity (for time series)
        if X.ndim == 2 and X.shape[0] > 1:
            assumptions.append({
                "assumption": "Time series is stationary (for sequential models)",
                "mathematical_principle": "statistics",
                "equation": "E[X_t] = μ, Var(X_t) = σ² (constant)",
            })
        
        logger.info("assumptions_identified", num_assumptions=len(assumptions))
        
        return assumptions
    
    def _check_data_support(
        self,
        assumptions: List[Dict[str, Any]],
        X: np.ndarray,
        y: np.ndarray,
        y_pred: np.ndarray,
    ) -> Dict[str, bool]:
        """Check if data supports the assumptions."""
        data_support = {}
        
        for assumption in assumptions:
            assumption_text = assumption["assumption"]
            
            if "normally distributed" in assumption_text.lower():
                # Test normality of residuals
                residuals = y - y_pred
                from scipy import stats
                if len(residuals) <= 5000:
                    _, p_value = stats.shapiro(residuals)
                else:
                    _, p_value = stats.normaltest(residuals)
                data_support[assumption_text] = p_value > 0.05
            
            elif "independent" in assumption_text.lower():
                # Test independence (autocorrelation)
                residuals = y - y_pred
                if len(residuals) > 1:
                    autocorr = np.corrcoef(residuals[:-1], residuals[1:])[0, 1]
                    data_support[assumption_text] = abs(autocorr) < 0.1
                else:
                    data_support[assumption_text] = True
            
            elif "stationary" in assumption_text.lower():
                # Test stationarity (simplified)
                mean_first_half = np.mean(y[:len(y)//2])
                mean_second_half = np.mean(y[len(y)//2:])
                std_first_half = np.std(y[:len(y)//2])
                std_second_half = np.std(y[len(y)//2:])
                
                # Check if means and stds are similar
                mean_diff = abs(mean_first_half - mean_second_half)
                std_diff = abs(std_first_half - std_second_half)
                
                data_support[assumption_text] = (
                    mean_diff < 0.1 * (abs(mean_first_half) + abs(mean_second_half)) and
                    std_diff < 0.1 * (std_first_half + std_second_half)
                )
            
            else:
                # Default: assume supported
                data_support[assumption_text] = True
        
        logger.info("data_support_checked", **data_support)
        
        return data_support
    
    def _analyze_bias_variance(
        self,
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        y_pred_train: np.ndarray,
        y_pred_val: np.ndarray,
    ) -> Dict[str, Any]:
        """Analyze bias and variance."""
        # Calculate errors
        train_error = np.mean((y_train - y_pred_train) ** 2)
        val_error = np.mean((y_val - y_pred_val) ** 2)
        
        # Bias estimate (training error)
        bias_estimate = train_error
        
        # Variance estimate (difference between train and val error)
        variance_estimate = val_error - train_error
        
        # Check trends (compared to previous iteration)
        bias_trend = "stable"
        variance_trend = "stable"
        
        if len(self.learning_history) > 0:
            prev_bias = self.learning_history[-1]["bias_variance"].get("bias_estimate", 0.0)
            prev_variance = self.learning_history[-1]["bias_variance"].get("variance_estimate", 0.0)
            
            if bias_estimate > prev_bias * 1.1:
                bias_trend = "increasing"
            elif bias_estimate < prev_bias * 0.9:
                bias_trend = "decreasing"
            
            if variance_estimate > prev_variance * 1.1:
                variance_trend = "increasing"
            elif variance_estimate < prev_variance * 0.9:
                variance_trend = "decreasing"
        
        analysis = {
            "train_error": float(train_error),
            "val_error": float(val_error),
            "bias_estimate": float(bias_estimate),
            "variance_estimate": float(variance_estimate),
            "bias_trend": bias_trend,
            "variance_trend": variance_trend,
            "is_overfitting": variance_estimate > 0.1 and val_error > train_error * 1.2,
            "is_underfitting": bias_estimate > 0.5 and variance_estimate < 0.05,
        }
        
        logger.info(
            "bias_variance_analysis_complete",
            bias=bias_estimate,
            variance=variance_estimate,
            bias_trend=bias_trend,
            variance_trend=variance_trend,
        )
        
        return analysis
    
    def _suggest_optimizations(
        self,
        model: Any,
        bias_variance: Dict[str, Any],
        data_support: Dict[str, bool],
    ) -> List[str]:
        """Suggest optimizations to minimize loss function."""
        suggestions = []
        
        # Check for unsupported assumptions
        unsupported_assumptions = [k for k, v in data_support.items() if not v]
        if unsupported_assumptions:
            suggestions.append(
                f"Data does not support assumptions: {', '.join(unsupported_assumptions)}. "
                "Consider transforming data or using different model assumptions."
            )
        
        # Check for overfitting
        if bias_variance["is_overfitting"]:
            suggestions.append(
                "Model is overfitting (high variance). "
                "Consider: adding regularization, reducing model complexity, increasing training data, or adding dropout."
            )
        
        # Check for underfitting
        if bias_variance["is_underfitting"]:
            suggestions.append(
                "Model is underfitting (high bias). "
                "Consider: increasing model complexity, adding more features, reducing regularization, or using a different model architecture."
            )
        
        # Check bias trend
        if bias_variance["bias_trend"] == "increasing":
            suggestions.append(
                "Bias is increasing. Consider: increasing model complexity or adding more features."
            )
        
        # Check variance trend
        if bias_variance["variance_trend"] == "increasing":
            suggestions.append(
                "Variance is increasing. Consider: adding regularization, reducing model complexity, or increasing training data."
            )
        
        # General optimization
        if not suggestions:
            suggestions.append(
                "Model performance is stable. Consider: fine-tuning hyperparameters or ensemble methods for further improvement."
            )
        
        logger.info("optimization_suggestions_generated", num_suggestions=len(suggestions))
        
        return suggestions
    
    def get_learning_summary(self) -> Dict[str, Any]:
        """Get summary of learning history."""
        if not self.learning_history:
            return {}
        
        # Get latest iteration
        latest = self.learning_history[-1]
        
        # Calculate trends
        if len(self.learning_history) > 1:
            first = self.learning_history[0]
            latest_gen_score = latest["prediction_reasoning"]["generalization_score"]
            first_gen_score = first["prediction_reasoning"]["generalization_score"]
            
            improvement = first_gen_score - latest_gen_score  # Lower is better
        else:
            improvement = 0.0
        
        return {
            "total_iterations": len(self.learning_history),
            "current_generalization_score": latest["prediction_reasoning"]["generalization_score"],
            "improvement": float(improvement),
            "current_bias": latest["bias_variance"]["bias_estimate"],
            "current_variance": latest["bias_variance"]["variance_estimate"],
            "optimization_suggestions": latest["optimization_suggestions"],
        }

