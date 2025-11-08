"""Baseline comparison system for model evaluation."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import structlog  # type: ignore[reportMissingImports]

from .comprehensive_evaluation import ComprehensiveEvaluation

logger = structlog.get_logger(__name__)


class BaselineComparison:
    """
    Compares every model against Random Forest baseline.
    
    If model underperforms baseline, flags as failed iteration.
    Stores comparison results in Brain Library.
    """

    def __init__(
        self,
        evaluator: Optional[ComprehensiveEvaluation] = None,
    ) -> None:
        """
        Initialize baseline comparison.
        
        Args:
            evaluator: Comprehensive evaluation instance
        """
        self.evaluator = evaluator or ComprehensiveEvaluation()
        logger.info("baseline_comparison_initialized")

    def create_baseline_model(
        self,
        random_state: int = 42,
        n_estimators: int = 100,
        max_depth: int = 10,
    ) -> Any:
        """
        Create Random Forest baseline model.
        
        Args:
            random_state: Random seed
            n_estimators: Number of trees
            max_depth: Maximum tree depth
            
        Returns:
            Random Forest model
        """
        try:
            from sklearn.ensemble import RandomForestRegressor  # type: ignore[reportMissingImports]
        except ImportError:
            logger.warning("sklearn_not_available", message="sklearn not installed - cannot create baseline")
            return None
        
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
        )
        
        return model

    def train_baseline(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        baseline_model: Optional[Any] = None,
    ) -> Any:
        """
        Train baseline model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            baseline_model: Optional pre-created baseline model
            
        Returns:
            Trained baseline model
        """
        if baseline_model is None:
            baseline_model = self.create_baseline_model()
        
        if baseline_model is None:
            logger.error("baseline_model_creation_failed")
            return None
        
        try:
            baseline_model.fit(X_train, y_train)
            logger.info("baseline_model_trained", samples=len(X_train))
            return baseline_model
        except Exception as e:
            logger.error("baseline_training_failed", error=str(e))
            return None

    def compare_with_baseline(
        self,
        model: Any,
        baseline_model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        improvement_threshold: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Compare model with baseline.
        
        Args:
            model: Model to compare
            baseline_model: Baseline model
            X_test: Test features
            y_test: Test targets
            improvement_threshold: Minimum improvement required (default: 0.0)
            
        Returns:
            Comparison results dictionary
        """
        try:
            # Get predictions
            model_predictions = model.predict(X_test)
            baseline_predictions = baseline_model.predict(X_test)
            
            # Calculate returns (prediction error as proxy)
            model_returns = model_predictions - y_test
            baseline_returns = baseline_predictions - y_test
            
            # Evaluate both models
            model_metrics = self.evaluator.evaluate_model(
                predictions=model_predictions,
                actuals=y_test,
                returns=model_returns,
            )
            
            baseline_metrics = self.evaluator.evaluate_model(
                predictions=baseline_predictions,
                actuals=y_test,
                returns=baseline_returns,
            )
            
            # Compare key metrics
            sharpe_diff = model_metrics.get("sharpe_ratio", 0.0) - baseline_metrics.get("sharpe_ratio", 0.0)
            rmse_diff = baseline_metrics.get("rmse", float('inf')) - model_metrics.get("rmse", float('inf'))
            r2_diff = model_metrics.get("r_squared", -float('inf')) - baseline_metrics.get("r_squared", -float('inf'))
            
            # Determine status
            passed = (
                sharpe_diff >= improvement_threshold and
                rmse_diff >= 0 and  # Lower RMSE is better
                r2_diff >= 0  # Higher R² is better
            )
            
            result = {
                "status": "passed" if passed else "failed",
                "model_metrics": model_metrics,
                "baseline_metrics": baseline_metrics,
                "improvement": {
                    "sharpe_diff": sharpe_diff,
                    "rmse_improvement": rmse_diff,
                    "r2_improvement": r2_diff,
                },
                "reason": "meets_requirements" if passed else "underperforms_baseline",
            }
            
            if not passed:
                logger.warning(
                    "model_failed_baseline_comparison",
                    sharpe_diff=sharpe_diff,
                    rmse_diff=rmse_diff,
                    r2_diff=r2_diff,
                )
            else:
                logger.info(
                    "model_passed_baseline_comparison",
                    sharpe_improvement=sharpe_diff,
                    rmse_improvement=rmse_diff,
                )
            
            return result
            
        except Exception as e:
            logger.error("baseline_comparison_failed", error=str(e))
            return {
                "status": "error",
                "error": str(e),
            }

    def compare_with_baseline_from_data(
        self,
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        improvement_threshold: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Compare model with baseline (trains baseline automatically).
        
        Args:
            model: Model to compare
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            y_test: Test targets
            improvement_threshold: Minimum improvement required
            
        Returns:
            Comparison results dictionary
        """
        # Train baseline
        baseline_model = self.train_baseline(X_train, y_train)
        
        if baseline_model is None:
            return {
                "status": "error",
                "error": "Failed to create baseline model",
            }
        
        # Compare
        return self.compare_with_baseline(
            model=model,
            baseline_model=baseline_model,
            X_test=X_test,
            y_test=y_test,
            improvement_threshold=improvement_threshold,
        )

