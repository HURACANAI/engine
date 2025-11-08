"""
Baseline Models

Implements Linear Regression, Logistic Regression, KNN, and SVM for baseline performance.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import structlog
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, SVR

from .base import BaseModel, ModelConfig, ModelMetrics

logger = structlog.get_logger(__name__)


class LinearRegressionModel(BaseModel):
    """Linear Regression model for continuous targets."""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        if config.model_type != "regression":
            logger.warning("linear_regression_is_regression_only", model_type=config.model_type)
    
    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        X_val: Optional[pd.DataFrame | np.ndarray] = None,
        y_val: Optional[pd.Series | np.ndarray] = None,
    ) -> ModelMetrics:
        """Train linear regression model."""
        logger.info("training_linear_regression", samples=len(X))
        
        # Convert to numpy if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        
        # Create pipeline
        self.model = Pipeline([
            ("regressor", LinearRegression(**self.config.hyperparameters))
        ])
        
        # Train
        self.model.fit(X, y)
        self.is_trained = True
        
        # Evaluate
        if X_val is not None and y_val is not None:
            metrics = self.evaluate(X_val, y_val)
        else:
            metrics = self.evaluate(X, y)
        
        self.metrics = metrics
        logger.info("linear_regression_trained", **metrics.to_dict())
        return metrics
    
    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        return self.model.predict(X)
    
    def evaluate(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> ModelMetrics:
        """Evaluate model."""
        y_pred = self.predict(X)
        
        if isinstance(y, pd.Series):
            y = y.values
        
        return self._calculate_regression_metrics(y, y_pred)
    
    def save(self, path: Path) -> None:
        """Save model."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.model, f)
        logger.info("linear_regression_saved", path=str(path))
    
    def load(self, path: Path) -> None:
        """Load model."""
        with open(path, "rb") as f:
            self.model = pickle.load(f)
        self.is_trained = True
        logger.info("linear_regression_loaded", path=str(path))


class LogisticRegressionModel(BaseModel):
    """Logistic Regression model for binary classification."""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        if config.model_type != "classification":
            logger.warning("logistic_regression_is_classification_only", model_type=config.model_type)
    
    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        X_val: Optional[pd.DataFrame | np.ndarray] = None,
        y_val: Optional[pd.Series | np.ndarray] = None,
    ) -> ModelMetrics:
        """Train logistic regression model."""
        logger.info("training_logistic_regression", samples=len(X))
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        
        # Ensure binary classification
        if len(np.unique(y)) > 2:
            logger.warning("logistic_regression_multi_class", num_classes=len(np.unique(y)))
        
        self.model = Pipeline([
            ("classifier", LogisticRegression(**self.config.hyperparameters, max_iter=1000))
        ])
        
        self.model.fit(X, y)
        self.is_trained = True
        
        if X_val is not None and y_val is not None:
            metrics = self.evaluate(X_val, y_val)
        else:
            metrics = self.evaluate(X, y)
        
        self.metrics = metrics
        logger.info("logistic_regression_trained", **metrics.to_dict())
        return metrics
    
    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        return self.model.predict_proba(X)
    
    def evaluate(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> ModelMetrics:
        """Evaluate model."""
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)
        
        if isinstance(y, pd.Series):
            y = y.values
        
        # Get probabilities for positive class
        if y_proba.shape[1] == 2:
            y_proba_positive = y_proba[:, 1]
        else:
            y_proba_positive = None
        
        return self._calculate_classification_metrics(y, y_pred, y_proba_positive)
    
    def save(self, path: Path) -> None:
        """Save model."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.model, f)
        logger.info("logistic_regression_saved", path=str(path))
    
    def load(self, path: Path) -> None:
        """Load model."""
        with open(path, "rb") as f:
            self.model = pickle.load(f)
        self.is_trained = True
        logger.info("logistic_regression_loaded", path=str(path))


class KNNModel(BaseModel):
    """K-Nearest Neighbors model for classification or regression."""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        # Default hyperparameters
        if "n_neighbors" not in config.hyperparameters:
            config.hyperparameters["n_neighbors"] = 5
    
    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        X_val: Optional[pd.DataFrame | np.ndarray] = None,
        y_val: Optional[pd.Series | np.ndarray] = None,
    ) -> ModelMetrics:
        """Train KNN model."""
        logger.info("training_knn", samples=len(X), model_type=self.config.model_type)
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        
        # Select appropriate KNN model
        if self.config.model_type == "classification":
            self.model = Pipeline([
                ("classifier", KNeighborsClassifier(**self.config.hyperparameters))
            ])
        else:
            self.model = Pipeline([
                ("regressor", KNeighborsRegressor(**self.config.hyperparameters))
            ])
        
        self.model.fit(X, y)
        self.is_trained = True
        
        if X_val is not None and y_val is not None:
            metrics = self.evaluate(X_val, y_val)
        else:
            metrics = self.evaluate(X, y)
        
        self.metrics = metrics
        logger.info("knn_trained", **metrics.to_dict())
        return metrics
    
    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> Optional[np.ndarray]:
        """Predict probabilities (classification only)."""
        if self.config.model_type != "classification":
            return None
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        return None
    
    def evaluate(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> ModelMetrics:
        """Evaluate model."""
        y_pred = self.predict(X)
        
        if isinstance(y, pd.Series):
            y = y.values
        
        if self.config.model_type == "classification":
            y_proba = self.predict_proba(X)
            if y_proba is not None and y_proba.shape[1] == 2:
                y_proba_positive = y_proba[:, 1]
            else:
                y_proba_positive = None
            return self._calculate_classification_metrics(y, y_pred, y_proba_positive)
        else:
            return self._calculate_regression_metrics(y, y_pred)
    
    def save(self, path: Path) -> None:
        """Save model."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.model, f)
        logger.info("knn_saved", path=str(path))
    
    def load(self, path: Path) -> None:
        """Load model."""
        with open(path, "rb") as f:
            self.model = pickle.load(f)
        self.is_trained = True
        logger.info("knn_loaded", path=str(path))


class SVMModel(BaseModel):
    """Support Vector Machine model for classification or regression."""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        # Default hyperparameters
        if "C" not in config.hyperparameters:
            config.hyperparameters["C"] = 1.0
        if "kernel" not in config.hyperparameters:
            config.hyperparameters["kernel"] = "rbf"
    
    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        X_val: Optional[pd.DataFrame | np.ndarray] = None,
        y_val: Optional[pd.Series | np.ndarray] = None,
    ) -> ModelMetrics:
        """Train SVM model."""
        logger.info("training_svm", samples=len(X), model_type=self.config.model_type)
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        
        # Select appropriate SVM model
        if self.config.model_type == "classification":
            self.model = Pipeline([
                ("classifier", SVC(**self.config.hyperparameters, probability=True))
            ])
        else:
            self.model = Pipeline([
                ("regressor", SVR(**self.config.hyperparameters))
            ])
        
        self.model.fit(X, y)
        self.is_trained = True
        
        if X_val is not None and y_val is not None:
            metrics = self.evaluate(X_val, y_val)
        else:
            metrics = self.evaluate(X, y)
        
        self.metrics = metrics
        logger.info("svm_trained", **metrics.to_dict())
        return metrics
    
    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> Optional[np.ndarray]:
        """Predict probabilities (classification only)."""
        if self.config.model_type != "classification":
            return None
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        return None
    
    def evaluate(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> ModelMetrics:
        """Evaluate model."""
        y_pred = self.predict(X)
        
        if isinstance(y, pd.Series):
            y = y.values
        
        if self.config.model_type == "classification":
            y_proba = self.predict_proba(X)
            if y_proba is not None and y_proba.shape[1] == 2:
                y_proba_positive = y_proba[:, 1]
            else:
                y_proba_positive = None
            return self._calculate_classification_metrics(y, y_pred, y_proba_positive)
        else:
            return self._calculate_regression_metrics(y, y_pred)
    
    def save(self, path: Path) -> None:
        """Save model."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.model, f)
        logger.info("svm_saved", path=str(path))
    
    def load(self, path: Path) -> None:
        """Load model."""
        with open(path, "rb") as f:
            self.model = pickle.load(f)
        self.is_trained = True
        logger.info("svm_loaded", path=str(path))

