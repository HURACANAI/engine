"""
Huracan Core - Mathematical Reasoning System

The core mathematical reasoning system for the Huracan Engine.
Integrates all mathematical components for reasoning, validation, and continuous learning.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import structlog

from .continuous_learning import ContinuousLearningCycle
from .data_understanding import DataUnderstanding
from .reasoning_engine import MathematicalReasoningEngine, PredictionWithReasoning
from .uncertainty_quantification import UncertaintyQuantifier
from .validation_framework import MathematicalValidator

logger = structlog.get_logger(__name__)


class HuracanCore:
    """
    Huracan Core - Mathematical Reasoning System
    
    Purpose: Detect, learn, and act on market patterns using mathematical reasoning.
    Method: Statistics, probability, linear algebra, and calculus.
    Goal: Maximum generalization and stability in dynamic financial markets.
    """
    
    def __init__(self):
        """Initialize Huracan Core."""
        self.reasoning_engine = MathematicalReasoningEngine()
        self.data_understanding = DataUnderstanding()
        self.uncertainty_quantifier = UncertaintyQuantifier()
        self.validator = MathematicalValidator()
        self.learning_cycle = ContinuousLearningCycle()
        
        logger.info("huracan_core_initialized")
    
    def understand_data(
        self,
        X: pd.DataFrame | np.ndarray,
        y: Optional[pd.Series | np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Understand data using mathematical principles.
        
        Args:
            X: Feature matrix
            y: Target vector (optional)
            feature_names: Names of features
            
        Returns:
            Dictionary with data understanding results
        """
        logger.info("understanding_data_mathematically")
        
        if isinstance(X, pd.DataFrame):
            X_array = X.values
            if feature_names is None:
                feature_names = list(X.columns)
        else:
            X_array = X
        
        if isinstance(y, pd.Series):
            y_array = y.values
        else:
            y_array = y
        
        # Comprehensive analysis
        analysis = self.data_understanding.comprehensive_analysis(
            X_array, y_array, feature_names
        )
        
        logger.info("data_understanding_complete")
        
        return analysis
    
    def train_with_reasoning(
        self,
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Train model with mathematical reasoning.
        
        Args:
            model: Model to train
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            
        Returns:
            Dictionary with training results and reasoning
        """
        logger.info("training_with_mathematical_reasoning")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Get predictions
        y_pred_train = model.predict(X_train)
        y_pred_val = model.predict(X_val)
        
        # Mathematical reasoning
        prediction_reasoning = self.reasoning_engine.reason_about_prediction(
            model, X_val, y_pred_val, y_val
        )
        
        # Uncertainty quantification
        uncertainty = self.uncertainty_quantifier.quantify_uncertainty(
            y_pred_val, y_val
        )
        
        # Mathematical validation
        validation = self.validator.validate_model_mathematically(
            model, X_train, y_train, X_val, y_val
        )
        
        # Generate explanation
        explanation = self.reasoning_engine.explain_prediction(prediction_reasoning)
        
        results = {
            "model": model,
            "prediction_reasoning": {
                "confidence_score": prediction_reasoning.confidence_score,
                "uncertainty_measure": prediction_reasoning.uncertainty_measure,
                "bias_estimate": prediction_reasoning.bias_estimate,
                "variance_estimate": prediction_reasoning.variance_estimate,
                "generalization_score": prediction_reasoning.generalization_score,
            },
            "uncertainty": uncertainty,
            "validation": validation,
            "explanation": explanation,
        }
        
        logger.info("training_with_reasoning_complete", **results["prediction_reasoning"])
        
        return results
    
    def predict_with_reasoning(
        self,
        model: Any,
        X: np.ndarray,
        y_true: Optional[np.ndarray] = None,
    ) -> PredictionWithReasoning:
        """
        Make prediction with full mathematical reasoning.
        
        Args:
            model: Trained model
            X: Features to predict on
            y_true: True values (optional, for validation)
            
        Returns:
            PredictionWithReasoning with full mathematical trace
        """
        logger.info("predicting_with_mathematical_reasoning")
        
        # Get predictions
        y_pred = model.predict(X)
        
        # Generate mathematical reasoning
        prediction_reasoning = self.reasoning_engine.reason_about_prediction(
            model, X, y_pred, y_true
        )
        
        # Quantify uncertainty
        uncertainty = self.uncertainty_quantifier.quantify_uncertainty(y_pred, y_true)
        prediction_reasoning.uncertainty_measure = uncertainty["uncertainty_score"]
        
        logger.info(
            "prediction_with_reasoning_complete",
            prediction=prediction_reasoning.prediction,
            confidence=prediction_reasoning.confidence_score,
            uncertainty=prediction_reasoning.uncertainty_measure,
        )
        
        return prediction_reasoning
    
    def continuous_learning_iteration(
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
        return self.learning_cycle.iterate(model, X_train, y_train, X_val, y_val, iteration)
    
    def explain_decision(
        self,
        prediction_reasoning: PredictionWithReasoning,
    ) -> str:
        """
        Explain decision in plain reasoning.
        
        Args:
            prediction_reasoning: Prediction with mathematical reasoning
            
        Returns:
            Plain-language explanation
        """
        explanation = self.reasoning_engine.explain_prediction(prediction_reasoning)
        
        # Add continuous learning questions
        questions = self.reasoning_engine.get_continuous_learning_questions()
        explanation += "\n\nContinuous Learning Questions:\n"
        for i, question in enumerate(questions, 1):
            explanation += f"{i}. {question}\n"
        
        return explanation
    
    def get_mathematical_trace(
        self,
        prediction_reasoning: PredictionWithReasoning,
    ) -> List[Dict[str, Any]]:
        """
        Get full mathematical trace for a prediction.
        
        Args:
            prediction_reasoning: Prediction with mathematical reasoning
            
        Returns:
            List of mathematical reasoning steps
        """
        trace = []
        
        for reasoning in prediction_reasoning.mathematical_reasoning:
            trace.append({
                "assumption": reasoning.assumption,
                "mathematical_principle": reasoning.mathematical_principle,
                "equation": reasoning.equation,
                "confidence_score": reasoning.confidence_score,
                "uncertainty_measure": reasoning.uncertainty_measure,
                "data_support": reasoning.data_support,
                "bias_estimate": reasoning.bias_estimate,
                "variance_estimate": reasoning.variance_estimate,
            })
        
        return trace
    
    def validate_assumptions(
        self,
        prediction_reasoning: PredictionWithReasoning,
    ) -> Dict[str, bool]:
        """
        Validate all mathematical assumptions.
        
        Args:
            prediction_reasoning: Prediction with mathematical reasoning
            
        Returns:
            Dictionary of assumption validation results
        """
        validation_results = {}
        
        for reasoning in prediction_reasoning.mathematical_reasoning:
            is_valid = self.reasoning_engine.validate_assumptions(reasoning)
            validation_results[reasoning.assumption] = is_valid
        
        return validation_results

