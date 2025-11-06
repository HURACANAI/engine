"""
Automated Feature Selection System

Automatically selects best features using multiple methods:
1. Remove low-variance features
2. Remove highly correlated features
3. Select top K by mutual information
4. Recursive feature elimination (RFE)
5. Model-based selection

Source: scikit-learn feature selection best practices
Expected Impact: Faster training, less overfitting, better generalization
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import structlog  # type: ignore
import numpy as np
import pandas as pd

try:
    from sklearn.feature_selection import (
        VarianceThreshold,
        SelectKBest,
        mutual_info_classif,
        mutual_info_regression,
        RFE,
        SelectFromModel,
    )
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import LassoCV
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import xgboost as xgb  # type: ignore
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

logger = structlog.get_logger(__name__)


@dataclass
class FeatureSelectionResult:
    """Result of feature selection."""
    selected_features: List[str]
    n_features_before: int
    n_features_after: int
    reduction_pct: float
    selection_method: str
    feature_scores: Optional[Dict[str, float]] = None


class AutomatedFeatureSelector:
    """
    Automated feature selection using multiple methods.
    
    Pipeline:
    1. Remove low-variance features
    2. Remove highly correlated features
    3. Select top K by mutual information
    4. Recursive feature elimination (RFE)
    5. Model-based selection (optional)
    """

    def __init__(
        self,
        variance_threshold: float = 0.01,
        correlation_threshold: float = 0.95,
        n_features_mutual_info: int = 50,
        n_features_rfe: int = 30,
        use_model_selection: bool = True,
    ):
        """
        Initialize feature selector.
        
        Args:
            variance_threshold: Minimum variance to keep feature
            correlation_threshold: Maximum correlation between features
            n_features_mutual_info: Number of features to select by mutual information
            n_features_rfe: Number of features to select by RFE
            use_model_selection: Whether to use model-based selection
        """
        if not HAS_SKLEARN:
            raise ImportError("scikit-learn required: pip install scikit-learn")
        
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold
        self.n_features_mutual_info = n_features_mutual_info
        self.n_features_rfe = n_features_rfe
        self.use_model_selection = use_model_selection
        
        logger.info(
            "automated_feature_selector_initialized",
            variance_threshold=variance_threshold,
            correlation_threshold=correlation_threshold,
            n_features_mutual_info=n_features_mutual_info,
            n_features_rfe=n_features_rfe,
        )

    def select_features(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        is_classification: bool = True,
        method: str = 'full',  # 'full', 'mutual_info', 'rfe', 'model'
    ) -> FeatureSelectionResult:
        """
        Select best features using automated pipeline.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            is_classification: True for classification, False for regression
            method: Selection method ('full', 'mutual_info', 'rfe', 'model')
            
        Returns:
            FeatureSelectionResult with selected features
        """
        original_features = list(X.columns)
        n_before = len(original_features)
        X_selected = X.copy()
        
        # Step 1: Remove low-variance features
        logger.info("removing_low_variance_features", n_features=n_before)
        selector = VarianceThreshold(threshold=self.variance_threshold)
        X_selected = pd.DataFrame(
            selector.fit_transform(X_selected),
            columns=X_selected.columns[selector.get_support()],
            index=X_selected.index,
        )
        n_after_variance = X_selected.shape[1]
        logger.info("variance_selection_complete", n_features=n_after_variance)
        
        # Step 2: Remove highly correlated features
        logger.info("removing_highly_correlated_features", n_features=n_after_variance)
        X_selected = self._remove_correlated_features(X_selected, self.correlation_threshold)
        n_after_correlation = X_selected.shape[1]
        logger.info("correlation_removal_complete", n_features=n_after_correlation)
        
        # Step 3-5: Apply selection method
        if method == 'full':
            # Full pipeline: mutual info → RFE → model
            X_selected, feature_scores = self._select_by_mutual_info(
                X_selected, y, is_classification, self.n_features_mutual_info
            )
            X_selected = self._select_by_rfe(
                X_selected, y, is_classification, self.n_features_rfe
            )
            if self.use_model_selection:
                X_selected = self._select_by_model(
                    X_selected, y, is_classification, min(20, X_selected.shape[1])
                )
        elif method == 'mutual_info':
            X_selected, feature_scores = self._select_by_mutual_info(
                X_selected, y, is_classification, self.n_features_mutual_info
            )
        elif method == 'rfe':
            X_selected = self._select_by_rfe(
                X_selected, y, is_classification, self.n_features_rfe
            )
            feature_scores = None
        elif method == 'model':
            X_selected = self._select_by_model(
                X_selected, y, is_classification, min(20, X_selected.shape[1])
            )
            feature_scores = None
        else:
            raise ValueError(f"Unknown method: {method}")
        
        n_after = X_selected.shape[1]
        reduction_pct = ((n_before - n_after) / n_before) * 100
        
        logger.info(
            "feature_selection_complete",
            n_before=n_before,
            n_after=n_after,
            reduction_pct=reduction_pct,
            method=method,
        )
        
        return FeatureSelectionResult(
            selected_features=list(X_selected.columns),
            n_features_before=n_before,
            n_features_after=n_after,
            reduction_pct=reduction_pct,
            selection_method=method,
            feature_scores=feature_scores,
        )

    def _remove_correlated_features(
        self,
        X: pd.DataFrame,
        threshold: float,
    ) -> pd.DataFrame:
        """Remove highly correlated features."""
        corr_matrix = X.corr().abs()
        
        # Find pairs with correlation > threshold
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Find features to drop
        to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]
        
        # Drop features
        X_cleaned = X.drop(columns=to_drop)
        
        if to_drop:
            logger.info("correlated_features_removed", n_removed=len(to_drop), features=to_drop[:5])
        
        return X_cleaned

    def _select_by_mutual_info(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        is_classification: bool,
        n_features: int,
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """Select features by mutual information."""
        if is_classification:
            selector = SelectKBest(mutual_info_classif, k=min(n_features, X.shape[1]))
        else:
            selector = SelectKBest(mutual_info_regression, k=min(n_features, X.shape[1]))
        
        X_selected = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()]
        
        # Get scores
        feature_scores = dict(zip(X.columns, selector.scores_))
        
        return pd.DataFrame(X_selected, columns=selected_features, index=X.index), feature_scores

    def _select_by_rfe(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        is_classification: bool,
        n_features: int,
    ) -> pd.DataFrame:
        """Select features by recursive feature elimination."""
        # Use simple model for RFE
        if is_classification:
            estimator = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=1)
        else:
            estimator = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=1)
        
        rfe = RFE(estimator=estimator, n_features_to_select=min(n_features, X.shape[1]), step=1)
        X_selected = rfe.fit_transform(X, y)
        selected_features = X.columns[rfe.support_]
        
        return pd.DataFrame(X_selected, columns=selected_features, index=X.index)

    def _select_by_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        is_classification: bool,
        n_features: int,
    ) -> pd.DataFrame:
        """Select features using model-based selection."""
        if is_classification:
            if HAS_XGBOOST:
                estimator = xgb.XGBClassifier(n_estimators=50, random_state=42, n_jobs=1)
            else:
                estimator = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=1)
        else:
            if HAS_XGBOOST:
                estimator = xgb.XGBRegressor(n_estimators=50, random_state=42, n_jobs=1)
            else:
                estimator = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=1)
        
        selector = SelectFromModel(estimator, max_features=min(n_features, X.shape[1]), threshold='median')
        X_selected = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()]
        
        return pd.DataFrame(X_selected, columns=selected_features, index=X.index)

