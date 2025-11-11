"""
Model Introspection Layer - SHAP Impact Timeline

Deep learning explains predictions poorly, but SHAP/correlation analysis helps.
On each Engine retrain, calculate SHAP values for top 20 features and store
them in Brain Library as "feature impact timeline." Council can later reference
these for strategy voting.

Author: Huracan Engine Team
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import structlog

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

from ..brain.brain_library import BrainLibrary

logger = structlog.get_logger(__name__)


@dataclass
class FeatureImpact:
    """Feature impact from SHAP analysis."""
    feature_name: str
    shap_value: float
    importance_rank: int
    mean_abs_shap: float
    contribution_pct: float  # Percentage of total impact


@dataclass
class IntrospectionReport:
    """Model introspection report."""
    model_id: str
    timestamp: datetime
    symbol: str
    top_features: List[FeatureImpact]
    shap_values: np.ndarray
    feature_names: List[str]
    correlation_matrix: Optional[np.ndarray] = None


class ModelIntrospection:
    """
    Model introspection using SHAP and correlation analysis.
    
    Usage:
        introspection = ModelIntrospection(brain_library=brain)
        
        # Analyze model
        report = introspection.analyze(
            model=trained_model,
            X=X_test,
            y=y_test,
            feature_names=feature_names,
            model_id="model_123",
            symbol="BTC/USDT"
        )
        
        # Get top features
        top_features = introspection.get_top_features("model_123", top_n=20)
    """
    
    def __init__(
        self,
        brain_library: Optional[BrainLibrary] = None,
        top_n_features: int = 20
    ):
        """
        Initialize model introspection.
        
        Args:
            brain_library: Brain Library for storing feature impact timeline
            top_n_features: Number of top features to track (default: 20)
        """
        self.brain_library = brain_library
        self.top_n_features = top_n_features
        
        logger.info(
            "model_introspection_initialized",
            top_n_features=top_n_features,
            has_shap=HAS_SHAP
        )
    
    def analyze(
        self,
        model: any,
        X: np.ndarray | pd.DataFrame,
        y: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        model_id: str = "unknown",
        symbol: str = "unknown",
        method: str = "shap"
    ) -> IntrospectionReport:
        """
        Analyze model and calculate feature impacts.
        
        Args:
            model: Trained model (must have predict method)
            X: Feature matrix
            y: Target values (optional, for correlation)
            feature_names: List of feature names
            model_id: Model identifier
            symbol: Trading symbol
            method: Analysis method ('shap', 'permutation', 'correlation')
        
        Returns:
            IntrospectionReport with feature impacts
        """
        # Convert to numpy
        if isinstance(X, pd.DataFrame):
            X_array = X.values
            if feature_names is None:
                feature_names = list(X.columns)
        else:
            X_array = X
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(X_array.shape[1])]
        
        # Calculate SHAP values
        if method == "shap" and HAS_SHAP:
            shap_values = self._calculate_shap(model, X_array)
        elif method == "permutation":
            shap_values = self._calculate_permutation_importance(model, X_array, y)
        else:
            # Fallback to correlation
            shap_values = self._calculate_correlation_importance(X_array, y)
        
        # Get top features
        top_features = self._get_top_features(
            feature_names,
            shap_values,
            top_n=self.top_n_features
        )
        
        # Calculate correlation matrix if y is provided
        correlation_matrix = None
        if y is not None:
            correlation_matrix = self._calculate_correlation_matrix(X_array, y)
        
        report = IntrospectionReport(
            model_id=model_id,
            timestamp=datetime.now(),
            symbol=symbol,
            top_features=top_features,
            shap_values=shap_values,
            feature_names=feature_names,
            correlation_matrix=correlation_matrix
        )
        
        # Store in Brain Library
        if self.brain_library:
            self._store_in_brain_library(report)
        
        logger.info(
            "model_introspection_complete",
            model_id=model_id,
            symbol=symbol,
            top_features_count=len(top_features)
        )
        
        return report
    
    def _calculate_shap(self, model: any, X: np.ndarray) -> np.ndarray:
        """Calculate SHAP values."""
        if not HAS_SHAP:
            logger.warning("shap_not_available_using_fallback")
            return np.abs(X).mean(axis=0)  # Fallback
        
        try:
            # Try TreeExplainer for tree-based models
            if hasattr(model, 'predict_proba'):
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X)
                # Handle multi-class output
                if isinstance(shap_values, list):
                    shap_values = shap_values[0]  # Use first class
                return np.abs(shap_values).mean(axis=0)
            else:
                # Use KernelExplainer for other models
                explainer = shap.KernelExplainer(model.predict, X[:100])  # Sample
                shap_values = explainer.shap_values(X[:1000])  # Limit for speed
                return np.abs(shap_values).mean(axis=0)
        except Exception as e:
            logger.warning("shap_calculation_failed", error=str(e))
            return np.abs(X).mean(axis=0)  # Fallback
    
    def _calculate_permutation_importance(
        self,
        model: any,
        X: np.ndarray,
        y: Optional[np.ndarray]
    ) -> np.ndarray:
        """Calculate permutation importance."""
        if y is None:
            return np.abs(X).mean(axis=0)  # Fallback
        
        baseline_score = self._score_model(model, X, y)
        importances = []
        
        for i in range(X.shape[1]):
            X_permuted = X.copy()
            np.random.shuffle(X_permuted[:, i])
            permuted_score = self._score_model(model, X_permuted, y)
            importance = baseline_score - permuted_score
            importances.append(importance)
        
        return np.array(importances)
    
    def _calculate_correlation_importance(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray]
    ) -> np.ndarray:
        """Calculate correlation-based importance."""
        if y is None:
            return np.abs(X).mean(axis=0)  # Fallback
        
        importances = []
        for i in range(X.shape[1]):
            corr = np.corrcoef(X[:, i], y)[0, 1]
            importances.append(abs(corr) if not np.isnan(corr) else 0.0)
        
        return np.array(importances)
    
    def _calculate_correlation_matrix(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> np.ndarray:
        """Calculate correlation matrix between features and target."""
        correlations = []
        for i in range(X.shape[1]):
            corr = np.corrcoef(X[:, i], y)[0, 1]
            correlations.append(corr if not np.isnan(corr) else 0.0)
        return np.array(correlations)
    
    def _score_model(self, model: any, X: np.ndarray, y: np.ndarray) -> float:
        """Score model (R² or accuracy)."""
        try:
            predictions = model.predict(X)
            if predictions.ndim > 1:
                predictions = predictions.flatten()
            
            # R² score
            ss_res = np.sum((y - predictions) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            return 1 - (ss_res / (ss_tot + 1e-8))
        except:
            return 0.0
    
    def _get_top_features(
        self,
        feature_names: List[str],
        shap_values: np.ndarray,
        top_n: int
    ) -> List[FeatureImpact]:
        """Get top N features by importance."""
        # Calculate mean absolute SHAP values
        mean_abs_shap = np.abs(shap_values)
        total_impact = mean_abs_shap.sum()
        
        # Get top indices
        top_indices = np.argsort(mean_abs_shap)[::-1][:top_n]
        
        top_features = []
        for rank, idx in enumerate(top_indices, 1):
            feature_name = feature_names[idx]
            shap_value = float(shap_values[idx])
            mean_abs = float(mean_abs_shap[idx])
            contribution_pct = float(mean_abs / (total_impact + 1e-8))
            
            top_features.append(FeatureImpact(
                feature_name=feature_name,
                shap_value=shap_value,
                importance_rank=rank,
                mean_abs_shap=mean_abs,
                contribution_pct=contribution_pct
            ))
        
        return top_features
    
    def _store_in_brain_library(self, report: IntrospectionReport) -> None:
        """Store feature impact timeline in Brain Library."""
        if not self.brain_library:
            return
        
        # Convert to Brain Library format
        feature_rankings = [
            {
                "feature_name": feat.feature_name,
                "importance_score": feat.mean_abs_shap,
                "rank": feat.importance_rank
            }
            for feat in report.top_features
        ]
        
        # Store using existing Brain Library method
        self.brain_library.store_feature_importance(
            analysis_date=report.timestamp,
            symbol=report.symbol,
            feature_rankings=feature_rankings,
            method="shap"
        )
        
        logger.debug(
            "feature_impact_stored",
            model_id=report.model_id,
            symbol=report.symbol,
            features_count=len(feature_rankings)
        )
    
    def get_top_features(
        self,
        model_id: str,
        top_n: Optional[int] = None
    ) -> List[str]:
        """Get top features for a model from Brain Library."""
        if not self.brain_library:
            return []
        
        top_n = top_n or self.top_n_features
        return self.brain_library.get_top_features(
            symbol="unknown",  # Would need model_id lookup
            top_n=top_n,
            method="shap"
        )

