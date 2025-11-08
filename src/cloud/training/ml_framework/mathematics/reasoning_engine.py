"""
Mathematical Reasoning Engine

Every prediction, decision, and output is traceable to mathematical principles:
- Statistics: probability distributions, confidence intervals, z-scores
- Linear Algebra: embeddings, vector transformations, feature interactions
- Calculus: gradient optimization, sensitivity analysis
- Bias-Variance: automatic balance between overfitting and underfitting
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import structlog
from scipy import stats

logger = structlog.get_logger(__name__)


@dataclass
class MathematicalReasoning:
    """Mathematical reasoning trace for a prediction."""
    
    assumption: str
    mathematical_principle: str  # "statistics", "linear_algebra", "calculus", "bias_variance"
    equation: str
    confidence_score: float
    uncertainty_measure: float
    validation_method: str
    data_support: bool
    bias_estimate: float
    variance_estimate: float


@dataclass
class PredictionWithReasoning:
    """Prediction with full mathematical reasoning."""
    
    prediction: float
    confidence_interval: Tuple[float, float]
    confidence_score: float
    uncertainty_measure: float
    mathematical_reasoning: List[MathematicalReasoning]
    bias_estimate: float
    variance_estimate: float
    generalization_score: float


class MathematicalReasoningEngine:
    """
    Mathematical reasoning engine for the Huracan Engine.
    
    Purpose: Ensure every prediction is traceable to mathematical principles.
    Method: Use statistics, linear algebra, calculus, and bias-variance theory.
    """
    
    def __init__(self):
        """Initialize mathematical reasoning engine."""
        self.reasoning_history: List[MathematicalReasoning] = []
        logger.info("mathematical_reasoning_engine_initialized")
    
    def reason_about_prediction(
        self,
        model: Any,
        X: np.ndarray,
        y_pred: np.ndarray,
        y_true: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
    ) -> PredictionWithReasoning:
        """
        Generate mathematical reasoning for a prediction.
        
        Args:
            model: Trained model
            X: Input features
            y_pred: Predictions
            y_true: True values (optional, for validation)
            feature_names: Names of features
            
        Returns:
            PredictionWithReasoning with full mathematical trace
        """
        logger.info("reasoning_about_prediction", samples=len(y_pred))
        
        reasoning_list = []
        
        # 1. Statistical Reasoning
        statistical_reasoning = self._statistical_reasoning(y_pred, y_true)
        reasoning_list.extend(statistical_reasoning)
        
        # 2. Linear Algebra Reasoning
        linear_algebra_reasoning = self._linear_algebra_reasoning(model, X, feature_names)
        reasoning_list.extend(linear_algebra_reasoning)
        
        # 3. Calculus Reasoning (if model supports gradients)
        calculus_reasoning = self._calculus_reasoning(model, X, y_pred)
        reasoning_list.extend(calculus_reasoning)
        
        # 4. Bias-Variance Reasoning
        bias_variance_reasoning = self._bias_variance_reasoning(model, X, y_pred, y_true)
        reasoning_list.extend(bias_variance_reasoning)
        
        # Calculate overall confidence and uncertainty
        confidence_score = self._calculate_confidence_score(reasoning_list, y_pred, y_true)
        uncertainty_measure = self._calculate_uncertainty(y_pred, y_true)
        confidence_interval = self._calculate_confidence_interval(y_pred, confidence_score)
        
        # Calculate bias and variance estimates
        bias_estimate = bias_variance_reasoning[0].bias_estimate if bias_variance_reasoning else 0.0
        variance_estimate = bias_variance_reasoning[0].variance_estimate if bias_variance_reasoning else 0.0
        
        # Calculate generalization score (lower is better)
        generalization_score = bias_estimate + variance_estimate
        
        prediction_with_reasoning = PredictionWithReasoning(
            prediction=float(np.mean(y_pred)),
            confidence_interval=confidence_interval,
            confidence_score=confidence_score,
            uncertainty_measure=uncertainty_measure,
            mathematical_reasoning=reasoning_list,
            bias_estimate=bias_estimate,
            variance_estimate=variance_estimate,
            generalization_score=generalization_score,
        )
        
        self.reasoning_history.extend(reasoning_list)
        
        logger.info(
            "prediction_reasoning_complete",
            confidence_score=confidence_score,
            uncertainty=uncertainty_measure,
            bias=bias_estimate,
            variance=variance_estimate,
        )
        
        return prediction_with_reasoning
    
    def _statistical_reasoning(
        self,
        y_pred: np.ndarray,
        y_true: Optional[np.ndarray],
    ) -> List[MathematicalReasoning]:
        """Generate statistical reasoning."""
        reasoning_list = []
        
        # Z-score analysis
        mean_pred = np.mean(y_pred)
        std_pred = np.std(y_pred)
        
        reasoning_list.append(MathematicalReasoning(
            assumption="Predictions follow a normal distribution",
            mathematical_principle="statistics",
            equation=f"Z = (X - μ) / σ, where μ = {mean_pred:.4f}, σ = {std_pred:.4f}",
            confidence_score=0.95 if std_pred > 0 else 0.0,
            uncertainty_measure=float(std_pred),
            validation_method="z-score_test",
            data_support=std_pred > 0,
            bias_estimate=0.0,
            variance_estimate=float(std_pred ** 2),
        ))
        
        # Confidence interval reasoning
        if y_true is not None:
            errors = y_pred - y_true
            mean_error = np.mean(errors)
            std_error = np.std(errors)
            
            # Calculate confidence interval using t-distribution
            n = len(errors)
            t_critical = stats.t.ppf(0.975, n - 1)  # 95% confidence
            margin_of_error = t_critical * std_error / np.sqrt(n)
            
            reasoning_list.append(MathematicalReasoning(
                assumption="Prediction errors are normally distributed",
                mathematical_principle="statistics",
                equation=f"CI = μ_error ± t_α/2 * (σ_error / √n), margin = {margin_of_error:.4f}",
                confidence_score=0.95,
                uncertainty_measure=float(margin_of_error),
                validation_method="confidence_interval",
                data_support=abs(mean_error) < 2 * std_error,
                bias_estimate=float(mean_error ** 2),
                variance_estimate=float(std_error ** 2),
            ))
        
        return reasoning_list
    
    def _linear_algebra_reasoning(
        self,
        model: Any,
        X: np.ndarray,
        feature_names: Optional[List[str]],
    ) -> List[MathematicalReasoning]:
        """Generate linear algebra reasoning."""
        reasoning_list = []
        
        # Feature space analysis
        if X.ndim == 2:
            n_samples, n_features = X.shape
            
            # Calculate covariance matrix
            if n_samples > 1:
                X_centered = X - np.mean(X, axis=0)
                covariance_matrix = np.cov(X_centered.T)
                
                # Eigenvalue decomposition (PCA principle)
                eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
                explained_variance = eigenvalues / np.sum(eigenvalues)
                
                reasoning_list.append(MathematicalReasoning(
                    assumption="Features can be represented in a lower-dimensional space",
                    mathematical_principle="linear_algebra",
                    equation=f"C = X^T * X / (n-1), λ_1 / Σλ = {explained_variance[0]:.4f}",
                    confidence_score=float(explained_variance[0]),
                    uncertainty_measure=float(1 - explained_variance[0]),
                    validation_method="pca_decomposition",
                    data_support=explained_variance[0] > 0.5,
                    bias_estimate=0.0,
                    variance_estimate=float(np.sum(explained_variance[1:])),
                ))
        
        # Feature interactions
        if hasattr(model, "get_feature_importance"):
            try:
                importance = model.get_feature_importance()
                if importance:
                    max_importance = max(importance.values())
                    reasoning_list.append(MathematicalReasoning(
                        assumption="Feature importance reflects true relationship strength",
                        mathematical_principle="linear_algebra",
                        equation=f"w^T * x, max(|w_i|) = {max_importance:.4f}",
                        confidence_score=float(max_importance),
                        uncertainty_measure=float(1 - max_importance),
                        validation_method="feature_importance",
                        data_support=True,
                        bias_estimate=0.0,
                        variance_estimate=0.0,
                    ))
            except Exception:
                pass
        
        return reasoning_list
    
    def _calculus_reasoning(
        self,
        model: Any,
        X: np.ndarray,
        y_pred: np.ndarray,
    ) -> List[MathematicalReasoning]:
        """Generate calculus reasoning (gradients, sensitivity)."""
        reasoning_list = []
        
        # Sensitivity analysis (numerical gradients)
        if X.ndim == 2 and len(X) > 0:
            # Calculate sensitivity: how much does prediction change with input?
            epsilon = 1e-5
            sensitivity_scores = []
            
            for i in range(min(10, X.shape[1])):  # Sample first 10 features
                X_perturbed = X.copy()
                X_perturbed[:, i] += epsilon
                
                try:
                    if hasattr(model, "predict"):
                        y_perturbed = model.predict(X_perturbed)
                        sensitivity = np.mean(np.abs(y_perturbed - y_pred) / epsilon)
                        sensitivity_scores.append(sensitivity)
                except Exception:
                    continue
            
            if sensitivity_scores:
                avg_sensitivity = np.mean(sensitivity_scores)
                max_sensitivity = np.max(sensitivity_scores)
                
                reasoning_list.append(MathematicalReasoning(
                    assumption="Model is differentiable and stable to small perturbations",
                    mathematical_principle="calculus",
                    equation=f"∂f/∂x_i ≈ (f(x + ε) - f(x)) / ε, avg = {avg_sensitivity:.4f}",
                    confidence_score=1.0 / (1.0 + avg_sensitivity),  # Lower sensitivity = higher confidence
                    uncertainty_measure=float(avg_sensitivity),
                    validation_method="sensitivity_analysis",
                    data_support=avg_sensitivity < 1.0,
                    bias_estimate=0.0,
                    variance_estimate=float(max_sensitivity ** 2),
                ))
        
        return reasoning_list
    
    def _bias_variance_reasoning(
        self,
        model: Any,
        X: np.ndarray,
        y_pred: np.ndarray,
        y_true: Optional[np.ndarray],
    ) -> List[MathematicalReasoning]:
        """Generate bias-variance reasoning."""
        reasoning_list = []
        
        if y_true is not None:
            errors = y_pred - y_true
            
            # Bias: E[error]
            bias = np.mean(errors)
            
            # Variance: Var(error)
            variance = np.var(errors)
            
            # Expected squared error = Bias² + Variance + Irreducible Error
            mse = np.mean(errors ** 2)
            irreducible_error = mse - bias ** 2 - variance
            
            reasoning_list.append(MathematicalReasoning(
                assumption="Total error decomposes into bias, variance, and irreducible error",
                mathematical_principle="bias_variance",
                equation=f"E[(y - ŷ)²] = Bias² + Variance + σ², Bias = {bias:.4f}, Var = {variance:.4f}",
                confidence_score=1.0 / (1.0 + abs(bias) + variance),
                uncertainty_measure=float(variance),
                validation_method="bias_variance_decomposition",
                data_support=abs(bias) < np.sqrt(variance),
                bias_estimate=float(bias ** 2),
                variance_estimate=float(variance),
            ))
            
            # Check for overfitting/underfitting
            if abs(bias) > np.sqrt(variance):
                issue = "underfitting" if bias ** 2 > variance else "overfitting"
                reasoning_list.append(MathematicalReasoning(
                    assumption=f"Model is {issue}",
                    mathematical_principle="bias_variance",
                    equation=f"Bias² ({bias**2:.4f}) {'>' if bias**2 > variance else '<'} Variance ({variance:.4f})",
                    confidence_score=0.5,
                    uncertainty_measure=float(abs(bias ** 2 - variance)),
                    validation_method="bias_variance_balance",
                    data_support=False,
                    bias_estimate=float(bias ** 2),
                    variance_estimate=float(variance),
                ))
        
        return reasoning_list
    
    def _calculate_confidence_score(
        self,
        reasoning_list: List[MathematicalReasoning],
        y_pred: np.ndarray,
        y_true: Optional[np.ndarray],
    ) -> float:
        """Calculate overall confidence score."""
        if not reasoning_list:
            return 0.5
        
        # Weighted average of confidence scores
        confidence_scores = [r.confidence_score for r in reasoning_list]
        return float(np.mean(confidence_scores))
    
    def _calculate_uncertainty(
        self,
        y_pred: np.ndarray,
        y_true: Optional[np.ndarray],
    ) -> float:
        """Calculate uncertainty measure."""
        if y_true is not None:
            # Uncertainty based on prediction error
            errors = y_pred - y_true
            return float(np.std(errors))
        else:
            # Uncertainty based on prediction variance
            return float(np.std(y_pred))
    
    def _calculate_confidence_interval(
        self,
        y_pred: np.ndarray,
        confidence_score: float,
    ) -> Tuple[float, float]:
        """Calculate confidence interval."""
        mean_pred = np.mean(y_pred)
        std_pred = np.std(y_pred)
        
        # Use z-score for confidence interval
        z_score = stats.norm.ppf(0.5 + confidence_score / 2)
        margin = z_score * std_pred
        
        return (float(mean_pred - margin), float(mean_pred + margin))
    
    def explain_prediction(self, prediction_with_reasoning: PredictionWithReasoning) -> str:
        """
        Generate plain-language explanation of prediction.
        
        Args:
            prediction_with_reasoning: Prediction with mathematical reasoning
            
        Returns:
            Plain-language explanation
        """
        explanation_parts = []
        
        explanation_parts.append(
            f"Prediction: {prediction_with_reasoning.prediction:.4f} "
            f"(confidence: {prediction_with_reasoning.confidence_score:.2%}, "
            f"uncertainty: ±{prediction_with_reasoning.uncertainty_measure:.4f})"
        )
        
        explanation_parts.append(
            f"Confidence interval: [{prediction_with_reasoning.confidence_interval[0]:.4f}, "
            f"{prediction_with_reasoning.confidence_interval[1]:.4f}]"
        )
        
        explanation_parts.append(
            f"Bias estimate: {prediction_with_reasoning.bias_estimate:.4f}, "
            f"Variance estimate: {prediction_with_reasoning.variance_estimate:.4f}"
        )
        
        explanation_parts.append(
            f"Generalization score: {prediction_with_reasoning.generalization_score:.4f} "
            f"(lower is better)"
        )
        
        # Add mathematical reasoning
        explanation_parts.append("\nMathematical Reasoning:")
        for i, reasoning in enumerate(prediction_with_reasoning.mathematical_reasoning, 1):
            explanation_parts.append(
                f"{i}. [{reasoning.mathematical_principle.upper()}] {reasoning.assumption}"
            )
            explanation_parts.append(f"   Equation: {reasoning.equation}")
            explanation_parts.append(
                f"   Confidence: {reasoning.confidence_score:.2%}, "
                f"Data supports: {reasoning.data_support}"
            )
        
        return "\n".join(explanation_parts)
    
    def validate_assumptions(self, reasoning: MathematicalReasoning) -> bool:
        """
        Validate mathematical assumptions.
        
        Args:
            reasoning: Mathematical reasoning to validate
            
        Returns:
            True if assumptions are valid
        """
        return reasoning.data_support and reasoning.confidence_score > 0.5
    
    def get_continuous_learning_questions(self) -> List[str]:
        """Get questions for continuous learning cycle."""
        return [
            "What assumption am I making mathematically?",
            "Does data support that assumption?",
            "Is my variance increasing or bias accumulating?",
            "How can I minimize the loss function better?",
        ]

