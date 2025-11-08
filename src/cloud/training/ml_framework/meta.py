"""
Meta-Layer: Ensemble Blending

Combines predictions from multiple models using weighted voting or stacking.
Weights adapt dynamically based on recent performance.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import structlog
from sklearn.linear_model import LinearRegression

from .base import BaseModel, ModelMetrics

logger = structlog.get_logger(__name__)


@dataclass
class EnsembleConfig:
    """Configuration for ensemble blending."""
    
    method: str = "weighted_voting"  # "weighted_voting", "stacking", "averaging"
    performance_window_days: int = 7  # Days to look back for performance weighting
    min_performance_samples: int = 10  # Minimum samples needed for weighting
    reweight_frequency: str = "daily"  # "daily", "weekly", "manual"
    use_sharpe_for_weighting: bool = True  # Use Sharpe ratio for weighting (vs RMSE)


@dataclass
class ModelPerformance:
    """Track performance of a single model."""
    
    model_name: str
    sharpe_ratio: float = 0.0
    rmse: float = 0.0
    mae: float = 0.0
    win_rate: float = 0.0
    accuracy: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    sample_count: int = 0


class EnsembleBlender:
    """
    Ensemble blending system that combines predictions from multiple models.
    
    Supports:
    - Weighted voting based on recent performance
    - Stacking with meta-learner
    - Simple averaging
    - Dynamic weight adjustment
    """
    
    def __init__(self, config: EnsembleConfig):
        """
        Initialize ensemble blender.
        
        Args:
            config: Ensemble configuration
        """
        self.config = config
        self.models: Dict[str, BaseModel] = {}
        self.weights: Dict[str, float] = {}
        self.performance_history: Dict[str, List[ModelPerformance]] = {}
        self.meta_model: Optional[LinearRegression] = None  # For stacking
        
        logger.info(
            "ensemble_blender_initialized",
            method=config.method,
            performance_window_days=config.performance_window_days,
        )
    
    def add_model(self, name: str, model: BaseModel) -> None:
        """
        Add a model to the ensemble.
        
        Args:
            name: Model name
            model: Trained model instance
        """
        self.models[name] = model
        self.weights[name] = 1.0 / len(self.models) if len(self.models) > 0 else 1.0
        self.performance_history[name] = []
        
        # Normalize weights
        self._normalize_weights()
        
        logger.info("model_added_to_ensemble", name=name, total_models=len(self.models))
    
    def update_performance(
        self,
        model_name: str,
        metrics: ModelMetrics,
        sample_count: int = 1,
    ) -> None:
        """
        Update performance tracking for a model.
        
        Args:
            model_name: Name of the model
            metrics: Performance metrics
            sample_count: Number of samples this metrics is based on
        """
        if model_name not in self.performance_history:
            self.performance_history[model_name] = []
        
        performance = ModelPerformance(
            model_name=model_name,
            sharpe_ratio=metrics.sharpe_ratio,
            rmse=metrics.rmse,
            mae=metrics.mae,
            win_rate=metrics.win_rate,
            accuracy=metrics.accuracy,
            timestamp=datetime.now(),
            sample_count=sample_count,
        )
        
        self.performance_history[model_name].append(performance)
        
        # Keep only recent performance (within window)
        cutoff_date = datetime.now() - timedelta(days=self.config.performance_window_days)
        self.performance_history[model_name] = [
            p for p in self.performance_history[model_name] if p.timestamp >= cutoff_date
        ]
        
        # Recalculate weights if enough samples
        if self._has_enough_samples():
            self._calculate_weights()
        
        logger.info(
            "performance_updated",
            model_name=model_name,
            sharpe=metrics.sharpe_ratio,
            rmse=metrics.rmse,
        )
    
    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Make ensemble prediction.
        
        Args:
            X: Features to predict on
            
        Returns:
            Ensemble predictions
        """
        if not self.models:
            raise ValueError("No models in ensemble")
        
        # Get predictions from all models
        predictions = {}
        for name, model in self.models.items():
            if model.is_trained:
                try:
                    pred = model.predict(X)
                    predictions[name] = pred
                except Exception as e:
                    logger.warning("model_prediction_failed", model_name=name, error=str(e))
                    continue
        
        if not predictions:
            raise ValueError("No models could make predictions")
        
        # Combine predictions based on method
        if self.config.method == "weighted_voting":
            return self._weighted_voting(predictions)
        elif self.config.method == "stacking":
            return self._stacking(predictions, X)
        elif self.config.method == "averaging":
            return self._averaging(predictions)
        else:
            raise ValueError(f"Unknown ensemble method: {self.config.method}")
    
    def _weighted_voting(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Weighted voting ensemble."""
        if not predictions:
            raise ValueError("No predictions to combine")
        
        # Normalize weights to sum to 1
        model_names = list(predictions.keys())
        weights = np.array([self.weights.get(name, 1.0 / len(model_names)) for name in model_names])
        weights = weights / weights.sum()
        
        # Weighted average
        weighted_pred = np.zeros_like(list(predictions.values())[0])
        for i, name in enumerate(model_names):
            weighted_pred += weights[i] * predictions[name]
        
        return weighted_pred
    
    def _averaging(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Simple averaging ensemble."""
        if not predictions:
            raise ValueError("No predictions to combine")
        
        pred_array = np.array(list(predictions.values()))
        return np.mean(pred_array, axis=0)
    
    def _stacking(self, predictions: Dict[str, np.ndarray], X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Stacking ensemble with meta-learner.
        
        Note: Meta-learner should be trained separately on validation data.
        For now, uses simple weighted average if meta-model not trained.
        """
        if self.meta_model is None:
            logger.warning("meta_model_not_trained_using_weighted_voting")
            return self._weighted_voting(predictions)
        
        # Create meta-features from base predictions
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        meta_features = np.column_stack(list(predictions.values()))
        
        # Predict with meta-model
        return self.meta_model.predict(meta_features)
    
    def train_meta_model(
        self,
        base_predictions: Dict[str, np.ndarray],
        y_true: np.ndarray,
    ) -> None:
        """
        Train stacking meta-model.
        
        Args:
            base_predictions: Dictionary of base model predictions
            y_true: True targets
        """
        if not base_predictions:
            raise ValueError("No base predictions provided")
        
        # Create meta-features
        meta_X = np.column_stack(list(base_predictions.values()))
        
        # Train meta-model
        self.meta_model = LinearRegression()
        self.meta_model.fit(meta_X, y_true)
        
        logger.info("meta_model_trained", num_base_models=len(base_predictions))
    
    def _calculate_weights(self) -> None:
        """Calculate weights based on recent performance."""
        if not self.models:
            return
        
        weights = {}
        
        for model_name in self.models.keys():
            if model_name not in self.performance_history:
                weights[model_name] = 1.0 / len(self.models)
                continue
            
            recent_performance = self.performance_history[model_name]
            if not recent_performance:
                weights[model_name] = 1.0 / len(self.models)
                continue
            
            # Calculate average performance
            if self.config.use_sharpe_for_weighting:
                # Use Sharpe ratio (higher is better)
                avg_performance = np.mean([p.sharpe_ratio for p in recent_performance])
                # Normalize to positive values
                avg_performance = max(avg_performance, 0.0) + 0.1  # Add small value to avoid zero
            else:
                # Use RMSE (lower is better, so invert)
                avg_rmse = np.mean([p.rmse for p in recent_performance])
                avg_performance = 1.0 / (avg_rmse + 0.1)  # Invert and add small value
            
            weights[model_name] = avg_performance
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            self.weights = {name: w / total_weight for name, w in weights.items()}
        else:
            # Equal weights if all performance is zero
            self.weights = {name: 1.0 / len(self.models) for name in self.models.keys()}
        
        logger.info("weights_recalculated", weights=self.weights)
    
    def _normalize_weights(self) -> None:
        """Normalize weights to sum to 1."""
        if not self.weights:
            return
        
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {name: w / total for name, w in self.weights.items()}
        else:
            # Equal weights
            self.weights = {name: 1.0 / len(self.models) for name in self.models.keys()}
    
    def _has_enough_samples(self) -> bool:
        """Check if we have enough samples to calculate weights."""
        for model_name in self.models.keys():
            if model_name in self.performance_history:
                total_samples = sum(p.sample_count for p in self.performance_history[model_name])
                if total_samples >= self.config.min_performance_samples:
                    return True
        return False
    
    def get_weights(self) -> Dict[str, float]:
        """Get current model weights."""
        return self.weights.copy()
    
    def save(self, path: Path) -> None:
        """Save ensemble blender."""
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save weights and performance history
        save_data = {
            "weights": self.weights,
            "performance_history": {
                name: [
                    {
                        "sharpe_ratio": p.sharpe_ratio,
                        "rmse": p.rmse,
                        "mae": p.mae,
                        "win_rate": p.win_rate,
                        "accuracy": p.accuracy,
                        "timestamp": p.timestamp.isoformat(),
                        "sample_count": p.sample_count,
                    }
                    for p in perf_list
                ]
                for name, perf_list in self.performance_history.items()
            },
            "config": self.config.__dict__,
        }
        
        with open(path, "wb") as f:
            pickle.dump(save_data, f)
        
        logger.info("ensemble_blender_saved", path=str(path))
    
    def load(self, path: Path) -> None:
        """Load ensemble blender."""
        with open(path, "rb") as f:
            save_data = pickle.load(f)
        
        self.weights = save_data["weights"]
        
        # Restore performance history
        self.performance_history = {}
        for name, perf_list in save_data["performance_history"].items():
            self.performance_history[name] = [
                ModelPerformance(
                    model_name=name,
                    sharpe_ratio=p["sharpe_ratio"],
                    rmse=p["rmse"],
                    mae=p["mae"],
                    win_rate=p["win_rate"],
                    accuracy=p["accuracy"],
                    timestamp=datetime.fromisoformat(p["timestamp"]),
                    sample_count=p["sample_count"],
                )
                for p in perf_list
            ]
        
        logger.info("ensemble_blender_loaded", path=str(path))

