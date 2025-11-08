"""Automated feature importance analysis for the Mechanic."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import polars as pl  # type: ignore[reportMissingImports]
import structlog  # type: ignore[reportMissingImports]

from .brain_library import BrainLibrary

logger = structlog.get_logger(__name__)


class FeatureImportanceAnalyzer:
    """
    Analyzes feature importance using multiple methods:
    - SHAP values
    - Permutation importance
    - Correlation analysis
    
    Stores rankings in Brain Library for Engine to use.
    """

    def __init__(
        self,
        brain_library: BrainLibrary,
    ) -> None:
        """
        Initialize feature importance analyzer.
        
        Args:
            brain_library: Brain Library instance for storage
        """
        self.brain = brain_library
        logger.info("feature_importance_analyzer_initialized")

    def analyze_feature_importance(
        self,
        symbol: str,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        methods: Optional[List[str]] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Analyze feature importance using multiple methods.
        
        Args:
            symbol: Trading symbol
            model: Trained model
            X: Feature matrix
            y: Target values
            feature_names: List of feature names
            methods: Methods to use (default: ['shap', 'permutation', 'correlation'])
            
        Returns:
            Dictionary mapping method names to feature rankings
        """
        methods = methods or ['shap', 'permutation', 'correlation']
        results = {}
        
        for method in methods:
            try:
                rankings = self._analyze_with_method(
                    method,
                    model,
                    X,
                    y,
                    feature_names,
                )
                results[method] = rankings
                
                # Store in Brain Library
                self.brain.store_feature_importance(
                    analysis_date=datetime.now(tz=timezone.utc),
                    symbol=symbol,
                    feature_rankings=rankings,
                    method=method,
                )
                
                logger.info(
                    "feature_importance_analyzed",
                    symbol=symbol,
                    method=method,
                    num_features=len(rankings),
                )
            except Exception as e:
                logger.warning(
                    "feature_importance_method_failed",
                    symbol=symbol,
                    method=method,
                    error=str(e),
                )
        
        return results

    def _analyze_with_method(
        self,
        method: str,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
    ) -> List[Dict[str, Any]]:
        """Analyze feature importance with a specific method."""
        if method == 'shap':
            return self._shap_importance(model, X, feature_names)
        elif method == 'permutation':
            return self._permutation_importance(model, X, y, feature_names)
        elif method == 'correlation':
            return self._correlation_importance(X, y, feature_names)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _shap_importance(
        self,
        model: Any,
        X: np.ndarray,
        feature_names: List[str],
    ) -> List[Dict[str, Any]]:
        """Calculate SHAP importance values."""
        try:
            import shap  # type: ignore[reportMissingImports]
            
            # Use TreeExplainer for tree-based models, otherwise KernelExplainer
            if hasattr(model, 'predict_proba'):
                explainer = shap.TreeExplainer(model)
            else:
                explainer = shap.KernelExplainer(model.predict, X[:100])  # Sample for speed
            
            shap_values = explainer.shap_values(X[:1000])  # Sample for speed
            
            # Calculate mean absolute SHAP values
            if isinstance(shap_values, list):
                shap_values = np.array(shap_values)
            
            importance = np.abs(shap_values).mean(axis=0)
            
            # Create rankings
            rankings = [
                {
                    "feature_name": feature_names[i],
                    "importance_score": float(importance[i]),
                }
                for i in range(len(feature_names))
            ]
            
            # Sort by importance
            rankings.sort(key=lambda x: x["importance_score"], reverse=True)
            
            return rankings
        except ImportError:
            logger.warning("shap_not_available", message="SHAP library not installed")
            return self._fallback_importance(X, feature_names)

    def _permutation_importance(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
    ) -> List[Dict[str, Any]]:
        """Calculate permutation importance."""
        try:
            from sklearn.inspection import permutation_importance  # type: ignore[reportMissingImports]
            
            # Calculate permutation importance
            perm_importance = permutation_importance(
                model,
                X,
                y,
                n_repeats=10,
                random_state=42,
                n_jobs=-1,
            )
            
            # Create rankings
            rankings = [
                {
                    "feature_name": feature_names[i],
                    "importance_score": float(perm_importance.importances_mean[i]),
                }
                for i in range(len(feature_names))
            ]
            
            # Sort by importance
            rankings.sort(key=lambda x: x["importance_score"], reverse=True)
            
            return rankings
        except ImportError:
            logger.warning("sklearn_not_available", message="sklearn not installed")
            return self._fallback_importance(X, feature_names)

    def _correlation_importance(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
    ) -> List[Dict[str, Any]]:
        """Calculate correlation-based importance."""
        # Calculate correlation between each feature and target
        correlations = []
        for i in range(X.shape[1]):
            corr = np.abs(np.corrcoef(X[:, i], y)[0, 1])
            correlations.append(corr)
        
        # Create rankings
        rankings = [
            {
                "feature_name": feature_names[i],
                "importance_score": float(correlations[i]),
            }
            for i in range(len(feature_names))
        ]
        
        # Sort by importance
        rankings.sort(key=lambda x: x["importance_score"], reverse=True)
        
        return rankings

    def _fallback_importance(
        self,
        X: np.ndarray,
        feature_names: List[str],
    ) -> List[Dict[str, Any]]:
        """Fallback importance calculation using variance."""
        # Use variance as a simple importance metric
        variances = np.var(X, axis=0)
        
        rankings = [
            {
                "feature_name": feature_names[i],
                "importance_score": float(variances[i]),
            }
            for i in range(len(feature_names))
        ]
        
        rankings.sort(key=lambda x: x["importance_score"], reverse=True)
        
        return rankings

    def get_top_features_for_symbol(
        self,
        symbol: str,
        top_n: int = 20,
        method: str = 'shap',
    ) -> List[str]:
        """
        Get top N features for a symbol from Brain Library.
        
        Args:
            symbol: Trading symbol
            top_n: Number of top features to return
            method: Method to use for ranking
            
        Returns:
            List of top feature names
        """
        return self.brain.get_top_features(symbol, top_n, method)

