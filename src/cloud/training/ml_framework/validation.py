"""
Validation and Cross-Validation Utilities

Implements cross-validation, bias-variance diagnostics, and validation helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import structlog
from sklearn.model_selection import (
    KFold,
    TimeSeriesSplit,
    cross_val_score,
    train_test_split,
)

from .base import BaseModel, ModelMetrics

logger = structlog.get_logger(__name__)


@dataclass
class BiasVarianceDiagnostics:
    """Bias-variance diagnostics results."""
    
    train_error: float
    validation_error: float
    test_error: float
    bias_score: float  # Lower is better
    variance_score: float  # Lower is better
    overfitting_detected: bool
    underfitting_detected: bool
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "train_error": self.train_error,
            "validation_error": self.validation_error,
            "test_error": self.test_error,
            "bias_score": self.bias_score,
            "variance_score": self.variance_score,
            "overfitting_detected": self.overfitting_detected,
            "underfitting_detected": self.underfitting_detected,
        }


class CrossValidator:
    """Cross-validation utilities for model evaluation."""
    
    def __init__(self, cv_folds: int = 5, use_time_series_split: bool = True):
        """
        Initialize cross-validator.
        
        Args:
            cv_folds: Number of cross-validation folds
            use_time_series_split: Use time-series split (preserves temporal order)
        """
        self.cv_folds = cv_folds
        self.use_time_series_split = use_time_series_split
        
        if use_time_series_split:
            self.cv = TimeSeriesSplit(n_splits=cv_folds)
        else:
            self.cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        logger.info(
            "cross_validator_initialized",
            cv_folds=cv_folds,
            use_time_series_split=use_time_series_split,
        )
    
    def cross_validate(
        self,
        model: BaseModel,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        scoring: str = "neg_mean_squared_error",
    ) -> Dict[str, Any]:
        """
        Perform cross-validation.
        
        Args:
            model: Model to validate
            X: Features
            y: Targets
            scoring: Scoring metric
            
        Returns:
            Dictionary with CV results
        """
        logger.info("starting_cross_validation", model_name=model.config.name)
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        
        # For sklearn-compatible models, use cross_val_score
        if hasattr(model.model, "fit") and hasattr(model.model, "predict"):
            try:
                scores = cross_val_score(
                    model.model,
                    X,
                    y,
                    cv=self.cv,
                    scoring=scoring,
                    n_jobs=-1,
                )
                
                results = {
                    "mean_score": float(np.mean(scores)),
                    "std_score": float(np.std(scores)),
                    "scores": scores.tolist(),
                    "cv_folds": self.cv_folds,
                }
                
                logger.info("cross_validation_complete", **results)
                return results
            except Exception as e:
                logger.warning("sklearn_cv_failed_falling_back", error=str(e))
        
        # Manual CV for custom models
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(self.cv.split(X)):
            logger.info("cv_fold", fold=fold + 1, train_size=len(train_idx), val_size=len(val_idx))
            
            X_train_fold = X[train_idx]
            y_train_fold = y[train_idx]
            X_val_fold = X[val_idx]
            y_val_fold = y[val_idx]
            
            # Train model
            model.fit(X_train_fold, y_train_fold)
            
            # Evaluate
            metrics = model.evaluate(X_val_fold, y_val_fold)
            
            # Get score (use RMSE for regression, accuracy for classification)
            if model.config.model_type == "regression":
                score = -metrics.rmse  # Negative because sklearn uses negative scores
            else:
                score = metrics.accuracy
            
            cv_scores.append(score)
        
        results = {
            "mean_score": float(np.mean(cv_scores)),
            "std_score": float(np.std(cv_scores)),
            "scores": cv_scores,
            "cv_folds": self.cv_folds,
        }
        
        logger.info("cross_validation_complete", **results)
        return results
    
    def bias_variance_diagnosis(
        self,
        model: BaseModel,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        X_val: pd.DataFrame | np.ndarray,
        y_val: pd.Series | np.ndarray,
        X_test: Optional[pd.DataFrame | np.ndarray] = None,
        y_test: Optional[pd.Series | np.ndarray] = None,
    ) -> BiasVarianceDiagnostics:
        """
        Diagnose bias-variance tradeoff.
        
        Args:
            model: Model to diagnose
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            X_test: Test features (optional)
            y_test: Test targets (optional)
            
        Returns:
            BiasVarianceDiagnostics with diagnosis results
        """
        logger.info("starting_bias_variance_diagnosis", model_name=model.config.name)
        
        # Train model
        model.fit(X_train, y_train, X_val, y_val)
        
        # Evaluate on training set
        train_metrics = model.evaluate(X_train, y_train)
        train_error = train_metrics.rmse if model.config.model_type == "regression" else 1.0 - train_metrics.accuracy
        
        # Evaluate on validation set
        val_metrics = model.evaluate(X_val, y_val)
        val_error = val_metrics.rmse if model.config.model_type == "regression" else 1.0 - val_metrics.accuracy
        
        # Evaluate on test set if provided
        if X_test is not None and y_test is not None:
            test_metrics = model.evaluate(X_test, y_test)
            test_error = test_metrics.rmse if model.config.model_type == "regression" else 1.0 - test_metrics.accuracy
        else:
            test_error = val_error
        
        # Calculate bias and variance
        # Bias: difference between train error and optimal error (approximate with validation error)
        bias_score = train_error
        
        # Variance: difference between validation error and train error
        variance_score = val_error - train_error
        
        # Detect overfitting (high variance)
        overfitting_detected = variance_score > 0.1 and val_error > train_error * 1.2
        
        # Detect underfitting (high bias)
        underfitting_detected = train_error > 0.5 and variance_score < 0.05
        
        diagnostics = BiasVarianceDiagnostics(
            train_error=train_error,
            validation_error=val_error,
            test_error=test_error,
            bias_score=bias_score,
            variance_score=variance_score,
            overfitting_detected=overfitting_detected,
            underfitting_detected=underfitting_detected,
        )
        
        logger.info("bias_variance_diagnosis_complete", **diagnostics.to_dict())
        
        if overfitting_detected:
            logger.warning("overfitting_detected", model_name=model.config.name, recommendations=["Add regularization", "Reduce model complexity", "Add dropout"])
        
        if underfitting_detected:
            logger.warning("underfitting_detected", model_name=model.config.name, recommendations=["Increase model complexity", "Add more features", "Reduce regularization"])
        
        return diagnostics


def create_train_val_test_split(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    shuffle: bool = True,
) -> Tuple[pd.DataFrame | np.ndarray, pd.DataFrame | np.ndarray, pd.DataFrame | np.ndarray, pd.Series | np.ndarray, pd.Series | np.ndarray, pd.Series | np.ndarray]:
    """
    Create train/validation/test split.
    
    Args:
        X: Features
        y: Targets
        test_size: Proportion of data for test set
        val_size: Proportion of data for validation set (from remaining after test)
        random_state: Random seed
        shuffle: Whether to shuffle data
        
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=shuffle
    )
    
    # Second split: train vs val
    val_size_adjusted = val_size / (1 - test_size)  # Adjust for remaining data
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, shuffle=shuffle
    )
    
    logger.info(
        "train_val_test_split_created",
        train_size=len(X_train),
        val_size=len(X_val),
        test_size=len(X_test),
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test

