"""Model selection service for Hamilton dynamic model switching."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

import structlog  # type: ignore[reportMissingImports]

from ..brain.brain_library import BrainLibrary

logger = structlog.get_logger(__name__)


class ModelSelector:
    """
    Model selection service for Hamilton.
    
    Selects appropriate model based on:
    - Volatility regime
    - Model performance metrics
    - Market conditions
    """

    def __init__(
        self,
        brain_library: BrainLibrary,
    ) -> None:
        """
        Initialize model selector.
        
        Args:
            brain_library: Brain Library instance
        """
        self.brain = brain_library
        logger.info("model_selector_initialized")

    def select_model_for_symbol(
        self,
        symbol: str,
        volatility_regime: str,
        model_type_preference: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Select best model for a symbol based on volatility regime.
        
        Args:
            symbol: Trading symbol
            volatility_regime: Current volatility regime ('low', 'normal', 'high', 'extreme')
            model_type_preference: Optional preferred model type
            
        Returns:
            Selected model information or None
        """
        logger.info(
            "selecting_model",
            symbol=symbol,
            volatility_regime=volatility_regime,
            model_type_preference=model_type_preference,
        )
        
        # Get best model from Brain Library
        best_model = self.brain.get_best_model(symbol)
        
        if not best_model:
            logger.warning("no_model_found", symbol=symbol)
            return None
        
        # Get active model (may be different from best model if rollback occurred)
        active_model = self.brain.get_active_model(symbol, model_type_preference)
        
        if not active_model:
            logger.warning("no_active_model_found", symbol=symbol)
            return None
        
        # Adjust model selection based on volatility regime
        recommended_model_type = self._recommend_model_type_for_regime(volatility_regime)
        
        # If recommended model type differs from active, check if it's available
        if recommended_model_type != active_model.get("model_type"):
            recommended_model = self.brain.get_active_model(symbol, recommended_model_type)
            if recommended_model:
                logger.info(
                    "regime_based_model_selection",
                    symbol=symbol,
                    regime=volatility_regime,
                    recommended_type=recommended_model_type,
                    active_type=active_model.get("model_type"),
                )
                return recommended_model
        
        return active_model

    def _recommend_model_type_for_regime(self, volatility_regime: str) -> str:
        """
        Recommend model type based on volatility regime.
        
        Args:
            volatility_regime: Volatility regime ('low', 'normal', 'high', 'extreme')
            
        Returns:
            Recommended model type
        """
        regime = volatility_regime.lower()
        
        # Model type recommendations by regime
        recommendations = {
            "low": "xgboost",  # Better for stable trends
            "normal": "lightgbm",  # Default good performer
            "high": "lstm",  # Better for complex patterns
            "extreme": "lightgbm",  # Conservative, stable
        }
        
        return recommendations.get(regime, "lightgbm")

    def get_model_confidence(
        self,
        symbol: str,
        model_id: Optional[str] = None,
    ) -> float:
        """
        Get model confidence score based on recent performance.
        
        Args:
            symbol: Trading symbol
            model_id: Optional specific model ID
            
        Returns:
            Confidence score (0.0 - 1.0)
        """
        if model_id:
            metrics_df = self.brain.get_model_metrics(model_id, symbol)
        else:
            active_model = self.brain.get_active_model(symbol)
            if not active_model:
                return 0.0
            model_id = active_model["model_id"]
            metrics_df = self.brain.get_model_metrics(model_id, symbol)
        
        if metrics_df.is_empty():
            return 0.5  # Default confidence
        
        # Calculate confidence based on recent metrics
        # Use Sharpe ratio as proxy for confidence
        latest_metrics = metrics_df.head(1)
        sharpe = latest_metrics["sharpe_ratio"][0] if "sharpe_ratio" in latest_metrics.columns else 0.0
        
        # Normalize Sharpe to 0-1 range (assuming Sharpe typically ranges from -2 to 4)
        confidence = max(0.0, min(1.0, (sharpe + 2.0) / 6.0))
        
        return float(confidence)

    def should_switch_model(
        self,
        symbol: str,
        current_model_id: str,
        volatility_regime: str,
    ) -> bool:
        """
        Determine if model should be switched based on regime change.
        
        Args:
            symbol: Trading symbol
            current_model_id: Current model ID
            volatility_regime: Current volatility regime
            
        Returns:
            True if model should be switched
        """
        # Get recommended model type for regime
        recommended_type = self._recommend_model_type_for_regime(volatility_regime)
        
        # Get current model
        current_model = self.brain.get_model_manifest(current_model_id)
        if not current_model:
            return False
        
        current_type = current_model.get("model_type", "lightgbm")
        
        # Switch if recommended type differs
        if recommended_type != current_type:
            # Check if recommended model exists and is better
            recommended_model = self.brain.get_active_model(symbol, recommended_type)
            if recommended_model:
                # Compare performance
                current_metrics = self.brain.get_model_metrics(current_model_id, symbol)
                recommended_metrics = self.brain.get_model_metrics(
                    recommended_model["model_id"],
                    symbol,
                )
                
                if not current_metrics.is_empty() and not recommended_metrics.is_empty():
                    current_sharpe = current_metrics["sharpe_ratio"][0] if "sharpe_ratio" in current_metrics.columns else 0.0
                    recommended_sharpe = recommended_metrics["sharpe_ratio"][0] if "sharpe_ratio" in recommended_metrics.columns else 0.0
                    
                    # Switch if recommended model is significantly better (10% improvement)
                    if recommended_sharpe > current_sharpe * 1.1:
                        return True
        
        return False

