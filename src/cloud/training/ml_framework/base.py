"""
Base Model Interface and Abstract Classes

Provides unified interface for all ML models in the framework.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class ModelConfig:
    """Configuration for a model."""
    
    name: str
    enabled: bool = True
    model_type: str = "regression"  # "regression" or "classification"
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    save_path: Optional[Path] = None
    load_path: Optional[Path] = None


@dataclass
class ModelMetrics:
    """Metrics for model evaluation."""
    
    mae: float = 0.0
    mse: float = 0.0
    rmse: float = 0.0
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    r2_score: float = 0.0
    auc: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "mae": self.mae,
            "mse": self.mse,
            "rmse": self.rmse,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "sharpe_ratio": self.sharpe_ratio,
            "win_rate": self.win_rate,
            "r2_score": self.r2_score,
            "auc": self.auc,
            "created_at": self.created_at.isoformat(),
        }


class BaseModel(ABC):
    """
    Base class for all ML models in the framework.
    
    All models must implement:
    - fit(): Train the model
    - predict(): Make predictions
    - evaluate(): Evaluate model performance
    - save(): Save model to disk
    - load(): Load model from disk
    """
    
    def __init__(self, config: ModelConfig):
        """
        Initialize base model.
        
        Args:
            config: Model configuration
        """
        self.config = config
        self.model: Any = None
        self.is_trained = False
        self.metrics: Optional[ModelMetrics] = None
        self.feature_names: Optional[List[str]] = None
        
        logger.info(
            "model_initialized",
            name=config.name,
            model_type=config.model_type,
            enabled=config.enabled,
        )
    
    @abstractmethod
    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        X_val: Optional[pd.DataFrame | np.ndarray] = None,
        y_val: Optional[pd.Series | np.ndarray] = None,
    ) -> ModelMetrics:
        """
        Train the model.
        
        Args:
            X: Training features
            y: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            
        Returns:
            ModelMetrics: Training metrics
        """
        pass
    
    @abstractmethod
    def predict(
        self,
        X: pd.DataFrame | np.ndarray,
    ) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Features to predict on
            
        Returns:
            Predictions array
        """
        pass
    
    @abstractmethod
    def evaluate(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
    ) -> ModelMetrics:
        """
        Evaluate model performance.
        
        Args:
            X: Features
            y: True targets
            
        Returns:
            ModelMetrics: Evaluation metrics
        """
        pass
    
    @abstractmethod
    def save(self, path: Path) -> None:
        """
        Save model to disk.
        
        Args:
            path: Path to save model
        """
        pass
    
    @abstractmethod
    def load(self, path: Path) -> None:
        """
        Load model from disk.
        
        Args:
            path: Path to load model from
        """
        pass
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance (if available).
        
        Returns:
            Dictionary mapping feature names to importance scores, or None
        """
        return None
    
    def _calculate_regression_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> ModelMetrics:
        """Calculate regression metrics."""
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        # Calculate Sharpe ratio (assuming returns)
        if len(y_pred) > 1:
            sharpe = np.mean(y_pred) / (np.std(y_pred) + 1e-9) * np.sqrt(252) if np.std(y_pred) > 0 else 0.0
        else:
            sharpe = 0.0
        
        # Calculate win rate (positive predictions)
        win_rate = float(np.mean(y_pred > 0)) if len(y_pred) > 0 else 0.0
        
        return ModelMetrics(
            mae=mae,
            mse=mse,
            rmse=rmse,
            r2_score=r2,
            sharpe_ratio=sharpe,
            win_rate=win_rate,
        )
    
    def _calculate_classification_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
    ) -> ModelMetrics:
        """Calculate classification metrics."""
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            roc_auc_score,
        )
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average="weighted", zero_division="warn")
        recall = recall_score(y_true, y_pred, average="weighted", zero_division="warn")
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division="warn")
        
        auc = 0.0
        if y_proba is not None and len(np.unique(y_true)) == 2:
            try:
                auc = float(roc_auc_score(y_true, y_proba))
            except ValueError:
                auc = 0.0
        
        # Calculate win rate (positive class predictions)
        win_rate = np.mean(y_pred == 1) if len(y_pred) > 0 else 0.0
        
        return ModelMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            auc=auc,
            win_rate=win_rate,
        )

