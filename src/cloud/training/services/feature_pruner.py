"""Automatic feature pruning based on importance rankings."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import structlog  # type: ignore[reportMissingImports]

logger = structlog.get_logger(__name__)


class FeaturePruner:
    """
    Automatically prunes low-value features.
    
    Uses SHAP, permutation importance, and correlation rankings.
    Drops lowest 20-30% of features.
    Compares performance pre/post pruning.
    """

    def __init__(
        self,
        prune_percentage: float = 0.2,  # Drop lowest 20%
        min_features: int = 10,  # Minimum number of features to keep
    ) -> None:
        """
        Initialize feature pruner.
        
        Args:
            prune_percentage: Percentage of features to prune (0.0-1.0)
            min_features: Minimum number of features to keep
        """
        self.prune_percentage = prune_percentage
        self.min_features = min_features
        
        logger.info(
            "feature_pruner_initialized",
            prune_percentage=prune_percentage,
            min_features=min_features,
        )

    def calculate_feature_importance(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        method: str = "permutation",
    ) -> Dict[str, float]:
        """
        Calculate feature importance.
        
        Args:
            model: Trained model
            X: Features
            y: Targets
            feature_names: List of feature names
            method: Method to use ('permutation', 'shap', 'correlation')
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if method == "permutation":
            return self._permutation_importance(model, X, y, feature_names)
        elif method == "shap":
            return self._shap_importance(model, X, y, feature_names)
        elif method == "correlation":
            return self._correlation_importance(X, y, feature_names)
        else:
            logger.warning("unknown_importance_method", method=method, using="permutation")
            return self._permutation_importance(model, X, y, feature_names)

    def _permutation_importance(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
    ) -> Dict[str, float]:
        """Calculate permutation importance."""
        try:
            from sklearn.inspection import permutation_importance  # type: ignore[reportMissingImports]
        except ImportError:
            logger.warning("sklearn_not_available", using="correlation")
            return self._correlation_importance(X, y, feature_names)
        
        # Get baseline score
        baseline_score = model.score(X, y)
        
        # Calculate permutation importance
        result = permutation_importance(
            model, X, y,
            n_repeats=10,
            random_state=42,
            n_jobs=-1,
        )
        
        # Create importance dictionary
        importance_dict = {
            feature_names[i]: float(result.importances_mean[i])
            for i in range(len(feature_names))
        }
        
        return importance_dict

    def _shap_importance(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
    ) -> Dict[str, float]:
        """Calculate SHAP importance."""
        try:
            import shap  # type: ignore[reportMissingImports]
        except ImportError:
            logger.warning("shap_not_available", using="permutation")
            return self._permutation_importance(model, X, y, feature_names)
        
        # Calculate SHAP values
        explainer = shap.Explainer(model, X)
        shap_values = explainer(X)
        
        # Calculate mean absolute SHAP value per feature
        importance_dict = {
            feature_names[i]: float(np.abs(shap_values.values[:, i]).mean())
            for i in range(len(feature_names))
        }
        
        return importance_dict

    def _correlation_importance(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
    ) -> Dict[str, float]:
        """Calculate correlation importance."""
        # Calculate correlation with target
        correlations = []
        for i in range(X.shape[1]):
            corr = np.abs(np.corrcoef(X[:, i], y)[0, 1])
            if np.isnan(corr):
                corr = 0.0
            correlations.append(corr)
        
        # Create importance dictionary
        importance_dict = {
            feature_names[i]: float(correlations[i])
            for i in range(len(feature_names))
        }
        
        return importance_dict

    def prune_features(
        self,
        feature_importance: Dict[str, float],
        feature_names: Optional[List[str]] = None,
    ) -> Tuple[List[str], Dict[str, Any]]:
        """
        Prune low-importance features.
        
        Args:
            feature_importance: Dictionary mapping feature names to importance scores
            feature_names: Optional list of all feature names (if importance dict is incomplete)
            
        Returns:
            Tuple of (kept_features, prune_report)
        """
        if feature_names is None:
            feature_names = list(feature_importance.keys())
        
        # Get importance scores for all features
        importance_scores = [
            feature_importance.get(feat, 0.0) for feat in feature_names
        ]
        
        # Calculate threshold (drop lowest percentage)
        num_to_keep = max(
            self.min_features,
            int(len(feature_names) * (1.0 - self.prune_percentage))
        )
        
        # Get indices of features to keep (top N by importance)
        feature_indices = np.argsort(importance_scores)[::-1]  # Sort descending
        kept_indices = feature_indices[:num_to_keep]
        pruned_indices = feature_indices[num_to_keep:]
        
        # Get feature names
        kept_features = [feature_names[i] for i in kept_indices]
        pruned_features = [feature_names[i] for i in pruned_indices]
        
        # Create report
        prune_report = {
            "original_count": len(feature_names),
            "kept_count": len(kept_features),
            "pruned_count": len(pruned_features),
            "prune_percentage": (len(pruned_features) / len(feature_names)) * 100,
            "pruned_features": pruned_features,
            "kept_features": kept_features,
            "min_importance_kept": float(min(importance_scores[i] for i in kept_indices)),
            "max_importance_pruned": float(max(importance_scores[i] for i in pruned_indices)) if pruned_indices.size > 0 else 0.0,
        }
        
        logger.info(
            "features_pruned",
            original=len(feature_names),
            kept=len(kept_features),
            pruned=len(pruned_features),
        )
        
        return kept_features, prune_report

    def compare_pre_post_pruning(
        self,
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        feature_names: List[str],
        importance_method: str = "permutation",
    ) -> Dict[str, Any]:
        """
        Compare model performance before and after pruning.
        
        Args:
            model: Model to test
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            y_test: Test targets
            feature_names: List of feature names
            importance_method: Method to calculate importance
            
        Returns:
            Comparison results dictionary
        """
        # Train model on all features
        model_all = model.__class__(**model.get_params() if hasattr(model, 'get_params') else {})
        model_all.fit(X_train, y_train)
        
        # Evaluate before pruning
        from .comprehensive_evaluation import ComprehensiveEvaluation
        evaluator = ComprehensiveEvaluation()
        
        predictions_all = model_all.predict(X_test)
        returns_all = predictions_all - y_test
        metrics_all = evaluator.evaluate_model(
            predictions=predictions_all,
            actuals=y_test,
            returns=returns_all,
        )
        
        # Calculate importance and prune
        importance = self.calculate_feature_importance(
            model_all, X_train, y_train, feature_names, method=importance_method
        )
        kept_features, prune_report = self.prune_features(importance, feature_names)
        
        # Get indices of kept features
        kept_indices = [feature_names.index(feat) for feat in kept_features]
        
        # Train model on pruned features
        X_train_pruned = X_train[:, kept_indices]
        X_test_pruned = X_test[:, kept_indices]
        
        model_pruned = model.__class__(**model.get_params() if hasattr(model, 'get_params') else {})
        model_pruned.fit(X_train_pruned, y_train)
        
        # Evaluate after pruning
        predictions_pruned = model_pruned.predict(X_test_pruned)
        returns_pruned = predictions_pruned - y_test
        metrics_pruned = evaluator.evaluate_model(
            predictions=predictions_pruned,
            actuals=y_test,
            returns=returns_pruned,
        )
        
        # Compare
        comparison = {
            "before_pruning": {
                "features": len(feature_names),
                "metrics": metrics_all,
            },
            "after_pruning": {
                "features": len(kept_features),
                "metrics": metrics_pruned,
            },
            "improvement": {
                "sharpe_diff": metrics_pruned.get("sharpe_ratio", 0.0) - metrics_all.get("sharpe_ratio", 0.0),
                "rmse_improvement": metrics_all.get("rmse", float('inf')) - metrics_pruned.get("rmse", float('inf')),
                "r2_improvement": metrics_pruned.get("r_squared", -float('inf')) - metrics_all.get("r_squared", -float('inf')),
            },
            "prune_report": prune_report,
            "recommendation": "keep_pruned" if metrics_pruned.get("sharpe_ratio", 0.0) >= metrics_all.get("sharpe_ratio", 0.0) else "revert_to_all",
        }
        
        logger.info(
            "pruning_comparison_complete",
            before_sharpe=metrics_all.get("sharpe_ratio", 0.0),
            after_sharpe=metrics_pruned.get("sharpe_ratio", 0.0),
            recommendation=comparison["recommendation"],
        )
        
        return comparison

