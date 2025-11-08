"""
Trade Attribution System

SHAP and permutation importance for each prediction.
Analyzes which features drove each trading decision.

Key Features:
- SHAP values for feature attribution
- Permutation importance for feature ranking
- Error type classification
- Feature importance by trade outcome
- Attribution visualization

Author: Huracan Engine Team
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from datetime import datetime

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


class AttributionMethod(Enum):
    """Attribution method"""
    SHAP = "shap"
    PERMUTATION = "permutation"
    GRADIENT = "gradient"
    INTEGRATED_GRADIENT = "integrated_gradient"


@dataclass
class FeatureAttribution:
    """Feature attribution result"""
    feature_name: str
    importance: float
    shap_value: Optional[float] = None
    permutation_importance: Optional[float] = None
    contribution: float = 0.0  # Contribution to prediction


@dataclass
class TradeAttribution:
    """Trade attribution result"""
    trade_id: str
    pred_id: str
    method: AttributionMethod
    feature_attributions: List[FeatureAttribution]
    top_features: List[Tuple[str, float]]  # (feature_name, importance)
    error_type: Optional[str] = None
    prediction_value: float = 0.0
    actual_outcome: float = 0.0
    attribution_confidence: float = 1.0


class TradeAttributionSystem:
    """
    Trade Attribution System.
    
    Computes SHAP and permutation importance for each prediction.
    
    Usage:
        attribution_system = TradeAttributionSystem()
        
        # Compute attribution
        attribution = attribution_system.compute_attribution(
            model=model,
            features=features,
            prediction=prediction,
            actual_outcome=actual_outcome,
            method=AttributionMethod.SHAP
        )
    """
    
    def __init__(
        self,
        top_k_features: int = 10,
        num_permutations: int = 100
    ):
        """
        Initialize trade attribution system.
        
        Args:
            top_k_features: Number of top features to return
            num_permutations: Number of permutations for permutation importance
        """
        self.top_k_features = top_k_features
        self.num_permutations = num_permutations
        
        logger.info(
            "trade_attribution_system_initialized",
            top_k_features=top_k_features,
            num_permutations=num_permutations
        )
    
    def compute_attribution(
        self,
        model: Any,
        features: Dict[str, float],
        prediction: float,
        actual_outcome: Optional[float] = None,
        method: AttributionMethod = AttributionMethod.PERMUTATION,
        baseline_features: Optional[Dict[str, float]] = None
    ) -> TradeAttribution:
        """
        Compute attribution for a prediction.
        
        Args:
            model: Trained model
            features: Feature dictionary
            prediction: Model prediction
            actual_outcome: Actual outcome (for error classification)
            method: Attribution method
            baseline_features: Baseline features for SHAP (optional)
        
        Returns:
            TradeAttribution
        """
        if method == AttributionMethod.SHAP:
            return self._compute_shap_attribution(
                model, features, prediction, baseline_features
            )
        elif method == AttributionMethod.PERMUTATION:
            return self._compute_permutation_attribution(
                model, features, prediction, actual_outcome
            )
        else:
            # Default to permutation
            return self._compute_permutation_attribution(
                model, features, prediction, actual_outcome
            )
    
    def _compute_shap_attribution(
        self,
        model: Any,
        features: Dict[str, float],
        prediction: float,
        baseline_features: Optional[Dict[str, float]] = None
    ) -> TradeAttribution:
        """
        Compute SHAP attribution.
        
        Note: This is a simplified implementation. In production, would use
        the SHAP library for more accurate attribution.
        """
        try:
            # Convert features to array
            feature_names = list(features.keys())
            feature_values = np.array([features[f] for f in feature_names])
            
            # Use baseline if provided, otherwise use mean
            if baseline_features:
                baseline_values = np.array([baseline_features.get(f, 0.0) for f in feature_names])
            else:
                baseline_values = np.zeros_like(feature_values)
            
            # Simplified SHAP: difference from baseline
            shap_values = feature_values - baseline_values
            
            # Create feature attributions
            feature_attributions = [
                FeatureAttribution(
                    feature_name=name,
                    importance=abs(shap_value),
                    shap_value=shap_value,
                    contribution=shap_value
                )
                for name, shap_value in zip(feature_names, shap_values)
            ]
            
            # Sort by importance
            feature_attributions.sort(key=lambda x: x.importance, reverse=True)
            
            # Get top features
            top_features = [
                (fa.feature_name, fa.importance)
                for fa in feature_attributions[:self.top_k_features]
            ]
            
            # Create attribution record
            attribution = TradeAttribution(
                trade_id="",  # Will be set by caller
                pred_id="",  # Will be set by caller
                method=AttributionMethod.SHAP,
                feature_attributions=feature_attributions,
                top_features=top_features,
                prediction_value=prediction,
                attribution_confidence=1.0
            )
            
            return attribution
            
        except Exception as e:
            logger.error("shap_attribution_failed", error=str(e))
            # Return empty attribution on error
            return TradeAttribution(
                trade_id="",
                pred_id="",
                method=AttributionMethod.SHAP,
                feature_attributions=[],
                top_features=[],
                prediction_value=prediction,
                attribution_confidence=0.0
            )
    
    def _compute_permutation_attribution(
        self,
        model: Any,
        features: Dict[str, float],
        prediction: float,
        actual_outcome: Optional[float] = None
    ) -> TradeAttribution:
        """
        Compute permutation importance attribution.
        
        Permutes each feature and measures impact on prediction.
        """
        try:
            feature_names = list(features.keys())
            feature_values = np.array([features[f] for f in feature_names])
            
            # Get baseline prediction (would use actual model prediction)
            baseline_prediction = prediction
            
            # Compute permutation importance for each feature
            permutation_importances = []
            
            for feature_idx, feature_name in enumerate(feature_names):
                # Permute this feature and measure impact
                permuted_predictions = []
                
                for _ in range(self.num_permutations):
                    # Create permuted features
                    permuted_features = features.copy()
                    
                    # Permute this feature (random value from distribution)
                    # In production, would use actual feature distribution
                    permuted_value = np.random.normal(
                        feature_values[feature_idx],
                        abs(feature_values[feature_idx]) * 0.1  # 10% noise
                    )
                    permuted_features[feature_name] = permuted_value
                    
                    # Get prediction with permuted feature
                    # In production, would use actual model prediction
                    permuted_prediction = self._predict_with_features(model, permuted_features)
                    permuted_predictions.append(permuted_prediction)
                
                # Calculate importance as change in prediction
                importance = abs(baseline_prediction - np.mean(permuted_predictions))
                permutation_importances.append(importance)
            
            # Create feature attributions
            feature_attributions = [
                FeatureAttribution(
                    feature_name=name,
                    importance=importance,
                    permutation_importance=importance,
                    contribution=importance * np.sign(feature_values[i])
                )
                for i, (name, importance) in enumerate(zip(feature_names, permutation_importances))
            ]
            
            # Sort by importance
            feature_attributions.sort(key=lambda x: x.importance, reverse=True)
            
            # Get top features
            top_features = [
                (fa.feature_name, fa.importance)
                for fa in feature_attributions[:self.top_k_features]
            ]
            
            # Classify error type if actual outcome provided
            error_type = None
            if actual_outcome is not None:
                error_type = self._classify_error_type(prediction, actual_outcome)
            
            # Create attribution record
            attribution = TradeAttribution(
                trade_id="",  # Will be set by caller
                pred_id="",  # Will be set by caller
                method=AttributionMethod.PERMUTATION,
                feature_attributions=feature_attributions,
                top_features=top_features,
                error_type=error_type,
                prediction_value=prediction,
                actual_outcome=actual_outcome or 0.0,
                attribution_confidence=1.0
            )
            
            return attribution
            
        except Exception as e:
            logger.error("permutation_attribution_failed", error=str(e))
            # Return empty attribution on error
            return TradeAttribution(
                trade_id="",
                pred_id="",
                method=AttributionMethod.PERMUTATION,
                feature_attributions=[],
                top_features=[],
                prediction_value=prediction,
                attribution_confidence=0.0
            )
    
    def _predict_with_features(self, model: Any, features: Dict[str, float]) -> float:
        """
        Get prediction from model with features.
        
        Placeholder - would use actual model prediction.
        """
        # In production, would call model.predict(features)
        # For now, return a dummy value
        return 0.0
    
    def _classify_error_type(
        self,
        prediction: float,
        actual_outcome: float
    ) -> str:
        """Classify error type based on prediction vs actual outcome"""
        if prediction > 0 and actual_outcome <= 0:
            return "direction_wrong"
        elif prediction <= 0 and actual_outcome > 0:
            return "direction_wrong"
        elif abs(prediction - actual_outcome) > abs(actual_outcome) * 0.5:
            return "magnitude_wrong"
        else:
            return "none"
    
    def compute_batch_attributions(
        self,
        model: Any,
        trades: List[Dict[str, Any]],
        predictions: List[Dict[str, Any]],
        method: AttributionMethod = AttributionMethod.PERMUTATION
    ) -> List[TradeAttribution]:
        """
        Compute attributions for a batch of trades.
        
        Args:
            model: Trained model
            trades: List of trade records
            predictions: List of prediction records
            method: Attribution method
        
        Returns:
            List of TradeAttribution
        """
        attributions = []
        
        # Create prediction lookup
        pred_dict = {p.get("pred_id"): p for p in predictions}
        
        for trade in trades:
            pred_id = trade.get("pred_id")
            pred = pred_dict.get(pred_id)
            
            if not pred:
                continue
            
            # Get features and prediction
            features = pred.get("features", {})
            prediction = pred.get("predicted_label", 0.0)
            actual_outcome = trade.get("pnl_after_costs", 0.0)
            
            # Compute attribution
            attribution = self.compute_attribution(
                model=model,
                features=features,
                prediction=prediction,
                actual_outcome=actual_outcome,
                method=method
            )
            
            # Set trade and prediction IDs
            attribution.trade_id = trade.get("trade_id", "")
            attribution.pred_id = pred_id
            
            attributions.append(attribution)
        
        return attributions
    
    def get_feature_importance_summary(
        self,
        attributions: List[TradeAttribution]
    ) -> Dict[str, Dict[str, float]]:
        """
        Get feature importance summary across all attributions.
        
        Args:
            attributions: List of trade attributions
        
        Returns:
            Dictionary mapping feature names to importance statistics
        """
        feature_importances: Dict[str, List[float]] = {}
        
        for attribution in attributions:
            for fa in attribution.feature_attributions:
                if fa.feature_name not in feature_importances:
                    feature_importances[fa.feature_name] = []
                feature_importances[fa.feature_name].append(fa.importance)
        
        # Calculate statistics
        summary = {}
        for feature_name, importances in feature_importances.items():
            summary[feature_name] = {
                "mean": np.mean(importances),
                "std": np.std(importances),
                "median": np.median(importances),
                "max": np.max(importances),
                "min": np.min(importances),
                "count": len(importances)
            }
        
        return summary
    
    def get_attribution_by_error_type(
        self,
        attributions: List[TradeAttribution]
    ) -> Dict[str, List[TradeAttribution]]:
        """Group attributions by error type"""
        by_error_type: Dict[str, List[TradeAttribution]] = {}
        
        for attribution in attributions:
            error_type = attribution.error_type or "none"
            if error_type not in by_error_type:
                by_error_type[error_type] = []
            by_error_type[error_type].append(attribution)
        
        return by_error_type

