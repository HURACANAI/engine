"""
Feature Selection Utilities

Implements feature selection based on importance, correlation, and statistical tests.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import structlog
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import (
    SelectKBest,
    SelectPercentile,
    f_regression,
    mutual_info_regression,
)

from .base import BaseModel

logger = structlog.get_logger(__name__)


class FeatureSelector:
    """Feature selection based on importance and statistical tests."""
    
    def __init__(
        self,
        method: str = "importance",
        n_features: Optional[int] = None,
        percentile: Optional[int] = None,
        threshold: Optional[float] = None,
    ):
        """
        Initialize feature selector.
        
        Args:
            method: Selection method ("importance", "correlation", "mutual_info", "f_test")
            n_features: Number of features to select (for SelectKBest)
            percentile: Percentile of features to select (for SelectPercentile)
            threshold: Threshold for selection (for threshold-based methods)
        """
        self.method = method
        self.n_features = n_features
        self.percentile = percentile
        self.threshold = threshold
        self.selector: Optional[Any] = None
        self.selected_features: Optional[List[str]] = None
        self.feature_scores: Optional[Dict[str, float]] = None
        
        logger.info(
            "feature_selector_initialized",
            method=method,
            n_features=n_features,
            percentile=percentile,
        )
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
        model: Optional[BaseModel] = None,
    ) -> FeatureSelector:
        """
        Fit feature selector.
        
        Args:
            X: Features
            y: Targets
            model: Optional model to use for importance-based selection
            
        Returns:
            Self for chaining
        """
        logger.info("fitting_feature_selector", method=self.method, n_features=len(X.columns))
        
        if isinstance(y, pd.Series):
            y = y.values
        
        if self.method == "importance":
            # Use model feature importance
            if model is None:
                # Use Random Forest as default
                rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                rf.fit(X, y)
                importances = rf.feature_importances_
            else:
                importance_dict = model.get_feature_importance()
                if importance_dict is None:
                    raise ValueError("Model does not support feature importance")
                importances = np.array([importance_dict.get(col, 0.0) for col in X.columns])
            
            # Create scores dictionary
            self.feature_scores = {col: float(imp) for col, imp in zip(X.columns, importances)}
            
            # Select features
            if self.n_features is not None:
                top_indices = np.argsort(importances)[-self.n_features:]
                self.selected_features = [X.columns[i] for i in top_indices]
            elif self.percentile is not None:
                n_select = max(1, int(len(X.columns) * self.percentile / 100))
                top_indices = np.argsort(importances)[-n_select:]
                self.selected_features = [X.columns[i] for i in top_indices]
            elif self.threshold is not None:
                self.selected_features = [col for col, imp in self.feature_scores.items() if imp >= self.threshold]
            else:
                # Select top 50% by default
                n_select = max(1, len(X.columns) // 2)
                top_indices = np.argsort(importances)[-n_select:]
                self.selected_features = [X.columns[i] for i in top_indices]
        
        elif self.method == "correlation":
            # Select features based on correlation with target
            correlations = X.corrwith(pd.Series(y, index=X.index)).abs()
            self.feature_scores = correlations.to_dict()
            
            if self.n_features is not None:
                top_features = correlations.nlargest(self.n_features)
                self.selected_features = top_features.index.tolist()
            elif self.threshold is not None:
                self.selected_features = correlations[correlations >= self.threshold].index.tolist()
            else:
                # Select top 50% by default
                n_select = max(1, len(X.columns) // 2)
                top_features = correlations.nlargest(n_select)
                self.selected_features = top_features.index.tolist()
        
        elif self.method == "mutual_info":
            # Use mutual information
            if self.n_features is not None:
                self.selector = SelectKBest(score_func=mutual_info_regression, k=self.n_features)
            elif self.percentile is not None:
                self.selector = SelectPercentile(score_func=mutual_info_regression, percentile=self.percentile)
            else:
                n_select = max(1, len(X.columns) // 2)
                self.selector = SelectKBest(score_func=mutual_info_regression, k=n_select)
            
            self.selector.fit(X, y)
            self.selected_features = X.columns[self.selector.get_support()].tolist()
            self.feature_scores = {col: float(score) for col, score in zip(X.columns, self.selector.scores_)}
        
        elif self.method == "f_test":
            # Use F-test
            if self.n_features is not None:
                self.selector = SelectKBest(score_func=f_regression, k=self.n_features)
            elif self.percentile is not None:
                self.selector = SelectPercentile(score_func=f_regression, percentile=self.percentile)
            else:
                n_select = max(1, len(X.columns) // 2)
                self.selector = SelectKBest(score_func=f_regression, k=n_select)
            
            self.selector.fit(X, y)
            self.selected_features = X.columns[self.selector.get_support()].tolist()
            self.feature_scores = {col: float(score) for col, score in zip(X.columns, self.selector.scores_)}
        
        else:
            raise ValueError(f"Unknown selection method: {self.method}")
        
        logger.info(
            "feature_selection_complete",
            n_selected=len(self.selected_features),
            selected_features=self.selected_features[:10],  # Log first 10
        )
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform features to selected subset."""
        if self.selected_features is None:
            raise ValueError("Feature selector must be fitted before transformation")
        
        return X[self.selected_features]
    
    def fit_transform(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
        model: Optional[BaseModel] = None,
    ) -> pd.DataFrame:
        """Fit and transform features."""
        return self.fit(X, y, model).transform(X)
    
    def get_selected_features(self) -> List[str]:
        """Get list of selected features."""
        if self.selected_features is None:
            raise ValueError("Feature selector must be fitted")
        return self.selected_features.copy()
    
    def get_feature_scores(self) -> Dict[str, float]:
        """Get feature importance scores."""
        if self.feature_scores is None:
            raise ValueError("Feature selector must be fitted")
        return self.feature_scores.copy()

