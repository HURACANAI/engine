"""
Core Learners

Implements Decision Trees, Random Forest, and XGBoost for non-linear modeling.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import structlog
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from .base import BaseModel, ModelConfig, ModelMetrics

logger = structlog.get_logger(__name__)

# Try to import XGBoost (optional)
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    logger.warning("xgboost_not_available", message="XGBoost not installed - XGBoostModel will not work")


class DecisionTreeModel(BaseModel):
    """Decision Tree model for classification or regression."""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        # Default hyperparameters
        if "max_depth" not in config.hyperparameters:
            config.hyperparameters["max_depth"] = 10
        if "min_samples_split" not in config.hyperparameters:
            config.hyperparameters["min_samples_split"] = 2
        if "min_samples_leaf" not in config.hyperparameters:
            config.hyperparameters["min_samples_leaf"] = 1
    
    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        X_val: Optional[pd.DataFrame | np.ndarray] = None,
        y_val: Optional[pd.Series | np.ndarray] = None,
    ) -> ModelMetrics:
        """Train decision tree model."""
        logger.info("training_decision_tree", samples=len(X), model_type=self.config.model_type)
        
        if isinstance(X, pd.DataFrame):
            X = X.values
            self.feature_names = list(X.columns) if hasattr(X, 'columns') else None
        if isinstance(y, pd.Series):
            y = y.values
        
        # Select appropriate model
        if self.config.model_type == "classification":
            self.model = DecisionTreeClassifier(**self.config.hyperparameters)
        else:
            self.model = DecisionTreeRegressor(**self.config.hyperparameters)
        
        self.model.fit(X, y)
        self.is_trained = True
        
        if X_val is not None and y_val is not None:
            metrics = self.evaluate(X_val, y_val)
        else:
            metrics = self.evaluate(X, y)
        
        self.metrics = metrics
        logger.info("decision_tree_trained", **metrics.to_dict())
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
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance."""
        if not self.is_trained or not hasattr(self.model, "feature_importances_"):
            return None
        
        if self.feature_names:
            return {name: float(imp) for name, imp in zip(self.feature_names, self.model.feature_importances_)}
        else:
            return {f"feature_{i}": float(imp) for i, imp in enumerate(self.model.feature_importances_)}
    
    def save(self, path: Path) -> None:
        """Save model."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.model, f)
        logger.info("decision_tree_saved", path=str(path))
    
    def load(self, path: Path) -> None:
        """Load model."""
        with open(path, "rb") as f:
            self.model = pickle.load(f)
        self.is_trained = True
        logger.info("decision_tree_loaded", path=str(path))


class RandomForestModel(BaseModel):
    """Random Forest model for classification or regression."""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        # Default hyperparameters
        if "n_estimators" not in config.hyperparameters:
            config.hyperparameters["n_estimators"] = 100
        if "max_depth" not in config.hyperparameters:
            config.hyperparameters["max_depth"] = 10
        if "min_samples_split" not in config.hyperparameters:
            config.hyperparameters["min_samples_split"] = 2
        if "min_samples_leaf" not in config.hyperparameters:
            config.hyperparameters["min_samples_leaf"] = 1
    
    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        X_val: Optional[pd.DataFrame | np.ndarray] = None,
        y_val: Optional[pd.Series | np.ndarray] = None,
    ) -> ModelMetrics:
        """Train random forest model."""
        logger.info("training_random_forest", samples=len(X), model_type=self.config.model_type)
        
        if isinstance(X, pd.DataFrame):
            X = X.values
            self.feature_names = list(X.columns) if hasattr(X, 'columns') else None
        if isinstance(y, pd.Series):
            y = y.values
        
        # Select appropriate model
        if self.config.model_type == "classification":
            self.model = RandomForestClassifier(**self.config.hyperparameters, n_jobs=-1)
        else:
            self.model = RandomForestRegressor(**self.config.hyperparameters, n_jobs=-1)
        
        self.model.fit(X, y)
        self.is_trained = True
        
        if X_val is not None and y_val is not None:
            metrics = self.evaluate(X_val, y_val)
        else:
            metrics = self.evaluate(X, y)
        
        self.metrics = metrics
        logger.info("random_forest_trained", **metrics.to_dict())
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
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance."""
        if not self.is_trained or not hasattr(self.model, "feature_importances_"):
            return None
        
        if self.feature_names:
            return {name: float(imp) for name, imp in zip(self.feature_names, self.model.feature_importances_)}
        else:
            return {f"feature_{i}": float(imp) for i, imp in enumerate(self.model.feature_importances_)}
    
    def save(self, path: Path) -> None:
        """Save model."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.model, f)
        logger.info("random_forest_saved", path=str(path))
    
    def load(self, path: Path) -> None:
        """Load model."""
        with open(path, "rb") as f:
            self.model = pickle.load(f)
        self.is_trained = True
        logger.info("random_forest_loaded", path=str(path))


class XGBoostModel(BaseModel):
    """XGBoost model for classification or regression."""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        if not HAS_XGBOOST:
            raise ImportError("XGBoost is not installed. Install it with: pip install xgboost")
        
        # Default hyperparameters
        if "n_estimators" not in config.hyperparameters:
            config.hyperparameters["n_estimators"] = 100
        if "max_depth" not in config.hyperparameters:
            config.hyperparameters["max_depth"] = 6
        if "learning_rate" not in config.hyperparameters:
            config.hyperparameters["learning_rate"] = 0.1
        if "objective" not in config.hyperparameters:
            if config.model_type == "classification":
                config.hyperparameters["objective"] = "binary:logistic"
            else:
                config.hyperparameters["objective"] = "reg:squarederror"
    
    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        X_val: Optional[pd.DataFrame | np.ndarray] = None,
        y_val: Optional[pd.Series | np.ndarray] = None,
    ) -> ModelMetrics:
        """Train XGBoost model."""
        logger.info("training_xgboost", samples=len(X), model_type=self.config.model_type)
        
        if isinstance(X, pd.DataFrame):
            X = X.values
            self.feature_names = list(X.columns) if hasattr(X, 'columns') else None
        if isinstance(y, pd.Series):
            y = y.values
        
        # Select appropriate model
        if self.config.model_type == "classification":
            self.model = xgb.XGBClassifier(**self.config.hyperparameters, n_jobs=-1)
        else:
            self.model = xgb.XGBRegressor(**self.config.hyperparameters, n_jobs=-1)
        
        # Train with early stopping if validation data provided
        if X_val is not None and y_val is not None:
            if isinstance(X_val, pd.DataFrame):
                X_val = X_val.values
            if isinstance(y_val, pd.Series):
                y_val = y_val.values
            
            self.model.fit(
                X, y,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=10,
                verbose=False,
            )
        else:
            self.model.fit(X, y)
        
        self.is_trained = True
        
        if X_val is not None and y_val is not None:
            metrics = self.evaluate(X_val, y_val)
        else:
            metrics = self.evaluate(X, y)
        
        self.metrics = metrics
        logger.info("xgboost_trained", **metrics.to_dict())
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
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance."""
        if not self.is_trained or not hasattr(self.model, "feature_importances_"):
            return None
        
        if self.feature_names:
            return {name: float(imp) for name, imp in zip(self.feature_names, self.model.feature_importances_)}
        else:
            return {f"feature_{i}": float(imp) for i, imp in enumerate(self.model.feature_importances_)}
    
    def save(self, path: Path) -> None:
        """Save model."""
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(path))
        logger.info("xgboost_saved", path=str(path))
    
    def load(self, path: Path) -> None:
        """Load model."""
        if self.config.model_type == "classification":
            self.model = xgb.XGBClassifier()
        else:
            self.model = xgb.XGBRegressor()
        self.model.load_model(str(path))
        self.is_trained = True
        logger.info("xgboost_loaded", path=str(path))

