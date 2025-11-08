"""
Integration example showing how to use Brain Library components
with Engine, Mechanic, and Hamilton.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List

import numpy as np
import structlog  # type: ignore[reportMissingImports]

from .brain_library import BrainLibrary
from .feature_importance_analyzer import FeatureImportanceAnalyzer
from .liquidation_collector import LiquidationCollector
from .model_comparison import ModelComparisonFramework
from .model_versioning import ModelVersioning
from .rl_agent import RLAgent

logger = structlog.get_logger(__name__)


class HuracanMLIntegration:
    """
    Example integration of Brain Library components with Huracan modules.
    
    This demonstrates how Engine, Mechanic, and Hamilton can use
    the Brain Library for enhanced ML trading capabilities.
    """

    def __init__(
        self,
        brain_library: BrainLibrary,
    ) -> None:
        """Initialize integration."""
        self.brain = brain_library
        
        # Initialize components
        self.liquidation_collector = LiquidationCollector(brain_library)
        self.feature_analyzer = FeatureImportanceAnalyzer(brain_library)
        self.model_comparison = ModelComparisonFramework(brain_library)
        self.model_versioning = ModelVersioning(brain_library)
        self.rl_agent = RLAgent()
        
        logger.info("huracan_ml_integration_initialized")

    def engine_training_workflow(
        self,
        symbol: str,
        models: Dict[str, Any],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        feature_names: List[str],
    ) -> Dict[str, Any]:
        """
        Engine training workflow with Brain Library integration.
        
        Steps:
        1. Compare multiple models
        2. Select best model
        3. Register model version
        4. Check for rollback
        """
        # Step 1: Compare models
        comparison_results = self.model_comparison.compare_models(
            symbol=symbol,
            models=models,
            X_test=X_test,
            y_test=y_test,
        )
        
        # Step 2: Get best model
        best_model_info = self.model_comparison.get_best_model_for_symbol(symbol)
        
        if not best_model_info:
            logger.warning("no_best_model_found", symbol=symbol)
            return {"status": "failed", "reason": "No best model found"}
        
        best_model_type = best_model_info["model_type"]
        best_model = models[best_model_type]
        
        # Step 3: Register model version
        model_id = f"{symbol}_{best_model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.model_versioning.register_model_version(
            model_id=model_id,
            symbol=symbol,
            version=1,
            hyperparameters=best_model.get_params() if hasattr(best_model, 'get_params') else {},
            dataset_id=f"dataset_{symbol}_{datetime.now().strftime('%Y%m%d')}",
            feature_set=feature_names,
            training_metrics=comparison_results[best_model_type],
            validation_metrics=comparison_results[best_model_type],
        )
        
        # Step 4: Check for rollback (if previous model exists)
        rollback_occurred = self.model_versioning.check_and_rollback(
            model_id=model_id,
            symbol=symbol,
            new_metrics=comparison_results[best_model_type],
        )
        
        if rollback_occurred:
            logger.warning("model_rollback_occurred", symbol=symbol, model_id=model_id)
            return {
                "status": "rollback",
                "model_id": model_id,
                "best_model_type": best_model_type,
            }
        
        # Step 5: Register in model registry
        self.brain.register_model(
            model_id=model_id,
            symbol=symbol,
            model_type=best_model_type,
            version=1,
            composite_score=best_model_info["composite_score"],
            hyperparameters=best_model.get_params() if hasattr(best_model, 'get_params') else {},
            dataset_id=f"dataset_{symbol}_{datetime.now().strftime('%Y%m%d')}",
            feature_set=feature_names,
        )
        
        return {
            "status": "success",
            "model_id": model_id,
            "best_model_type": best_model_type,
            "metrics": comparison_results[best_model_type],
        }

    def mechanic_feature_analysis_workflow(
        self,
        symbol: str,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
    ) -> Dict[str, Any]:
        """
        Mechanic feature analysis workflow.
        
        Steps:
        1. Analyze feature importance
        2. Get top features
        3. Return feature rankings
        """
        # Step 1: Analyze feature importance
        importance_results = self.feature_analyzer.analyze_feature_importance(
            symbol=symbol,
            model=model,
            X=X,
            y=y,
            feature_names=feature_names,
        )
        
        # Step 2: Get top features from Brain Library
        top_features = self.feature_analyzer.get_top_features_for_symbol(
            symbol=symbol,
            top_n=20,
            method='shap',
        )
        
        return {
            "top_features": top_features,
            "importance_results": importance_results,
        }

    def hamilton_execution_workflow(
        self,
        symbol: str,
        base_predictions: Dict[str, float],
        portfolio_allocation: Dict[str, float],
        model_confidences: Dict[str, float],
        volatility_regime: str,
        recent_pnl: float,
        current_drawdown: float,
    ) -> Dict[str, Any]:
        """
        Hamilton execution workflow with RL agent.
        
        Steps:
        1. Get active model from Brain Library
        2. Construct state vector
        3. Get RL agent suggestions
        4. Combine with base predictions
        """
        # Step 1: Get active model
        active_model = self.brain.get_active_model(symbol)
        
        if not active_model:
            logger.warning("no_active_model", symbol=symbol)
            return {"status": "failed", "reason": "No active model"}
        
        # Step 2: Construct state vector
        state = self.rl_agent.get_state_vector(
            portfolio_allocation=portfolio_allocation,
            model_confidences=model_confidences,
            volatility_regime=volatility_regime,
            recent_pnl=recent_pnl,
            current_drawdown=current_drawdown,
        )
        
        # Step 3: Get RL agent suggestions
        rl_suggestions = self.rl_agent.suggest_allocation(
            state=state,
            base_predictions=base_predictions,
        )
        
        return {
            "status": "success",
            "active_model": active_model,
            "rl_suggestions": rl_suggestions,
            "final_allocation": rl_suggestions["allocations"],
        }

    def liquidation_data_workflow(
        self,
        symbol: str,
        hours: int = 24,
    ) -> Dict[str, Any]:
        """
        Collect and analyze liquidation data.
        
        Steps:
        1. Collect liquidations
        2. Detect cascades
        3. Label volatility clusters
        """
        from datetime import timedelta
        
        end_time = datetime.now(tz=timezone.utc)
        start_time = end_time - timedelta(hours=hours)
        
        # Step 1: Collect liquidations
        collected = self.liquidation_collector.collect_liquidations(
            symbol=symbol,
            start_time=start_time,
            end_time=end_time,
        )
        
        # Step 2: Detect cascades
        cascades = self.liquidation_collector.detect_cascades(symbol)
        
        # Step 3: Label volatility clusters
        clusters = self.liquidation_collector.label_volatility_clusters(symbol)
        
        return {
            "collected_liquidations": collected,
            "cascades": cascades,
            "volatility_clusters": clusters,
        }

