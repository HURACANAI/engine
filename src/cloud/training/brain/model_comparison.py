"""Model comparison framework for multi-model evaluation."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import structlog  # type: ignore[reportMissingImports]

from .brain_library import BrainLibrary

logger = structlog.get_logger(__name__)


class ModelComparisonFramework:
    """
    Compares multiple model types (LSTM, CNN, XGBoost, Transformer)
    and selects the best model per asset.
    """

    def __init__(
        self,
        brain_library: BrainLibrary,
    ) -> None:
        """
        Initialize model comparison framework.
        
        Args:
            brain_library: Brain Library instance for storage
        """
        self.brain = brain_library
        self.model_types = ['lstm', 'cnn', 'xgboost', 'transformer']
        logger.info("model_comparison_framework_initialized")

    def compare_models(
        self,
        symbol: str,
        models: Dict[str, Any],
        X_test: np.ndarray,
        y_test: np.ndarray,
        comparison_date: Optional[datetime] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare multiple models and store results.
        
        Args:
            symbol: Trading symbol
            models: Dictionary mapping model_type to model instance
            X_test: Test features
            y_test: Test targets
            comparison_date: Date of comparison (default: today)
            
        Returns:
            Dictionary mapping model_type to metrics
        """
        if comparison_date is None:
            comparison_date = datetime.now(tz=timezone.utc)
        
        all_metrics = {}
        
        for model_type, model in models.items():
            try:
                metrics = self._evaluate_model(
                    model,
                    X_test,
                    y_test,
                )
                
                all_metrics[model_type] = metrics
                
                # Store in Brain Library
                self.brain.store_model_comparison(
                    comparison_date=comparison_date,
                    symbol=symbol,
                    model_type=model_type,
                    metrics=metrics,
                )
                
                logger.info(
                    "model_evaluated",
                    symbol=symbol,
                    model_type=model_type,
                    sharpe=metrics.get("sharpe_ratio", 0.0),
                )
            except Exception as e:
                logger.warning(
                    "model_evaluation_failed",
                    symbol=symbol,
                    model_type=model_type,
                    error=str(e),
                )
        
        # Select best model
        best_model = self._select_best_model(all_metrics)
        
        if best_model:
            logger.info(
                "best_model_selected",
                symbol=symbol,
                model_type=best_model,
                metrics=all_metrics[best_model],
            )
        
        return all_metrics

    def _evaluate_model(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, float]:
        """
        Evaluate a model and return metrics.
        
        Args:
            model: Model instance
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary of metrics
        """
        # Get predictions
        if hasattr(model, 'predict'):
            predictions = model.predict(X_test)
        else:
            raise ValueError("Model must have predict method")
        
        # Calculate returns (assuming predictions are returns)
        returns = predictions - y_test  # Prediction error as proxy for returns
        
        # Calculate metrics
        accuracy = float(np.mean(np.abs(returns) < 0.01))  # Within 1% accuracy
        
        # Sharpe ratio
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(252))  # Annualized
        else:
            sharpe = 0.0
        
        # Sortino ratio (downside deviation only)
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 1 and np.std(downside_returns) > 0:
            sortino = float(np.mean(returns) / np.std(downside_returns) * np.sqrt(252))
        else:
            sortino = 0.0
        
        # Max drawdown
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = cumulative - running_max
        max_drawdown = float(np.min(drawdown)) if len(drawdown) > 0 else 0.0
        
        # Profit factor (simplified)
        profits = returns[returns > 0]
        losses = returns[returns < 0]
        if len(losses) > 0 and abs(np.sum(losses)) > 0:
            profit_factor = float(np.sum(profits) / abs(np.sum(losses)))
        else:
            profit_factor = 0.0
        
        return {
            "accuracy": accuracy,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown": abs(max_drawdown),  # Positive value
            "profit_factor": profit_factor,
        }

    def _select_best_model(
        self,
        all_metrics: Dict[str, Dict[str, float]],
    ) -> Optional[str]:
        """
        Select best model based on composite score.
        
        Args:
            all_metrics: Dictionary mapping model_type to metrics
            
        Returns:
            Best model type
        """
        if not all_metrics:
            return None
        
        best_model = None
        best_score = float('-inf')
        
        for model_type, metrics in all_metrics.items():
            # Composite score
            score = (
                0.4 * metrics.get("sharpe_ratio", 0.0) +
                0.3 * metrics.get("profit_factor", 0.0) +
                0.2 * (1.0 - abs(metrics.get("max_drawdown", 1.0))) +
                0.1 * metrics.get("accuracy", 0.0)
            )
            
            if score > best_score:
                best_score = score
                best_model = model_type
        
        return best_model

    def get_best_model_for_symbol(
        self,
        symbol: str,
        comparison_date: Optional[datetime] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Get best model for a symbol from Brain Library.
        
        Args:
            symbol: Trading symbol
            comparison_date: Date to check (default: latest)
            
        Returns:
            Best model information or None
        """
        return self.brain.get_best_model(symbol, comparison_date)

