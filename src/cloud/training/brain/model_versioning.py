"""Model versioning and rollback system."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

import structlog  # type: ignore[reportMissingImports]

from .brain_library import BrainLibrary

logger = structlog.get_logger(__name__)


class ModelVersioning:
    """
    Manages model versioning and automatic rollback.
    
    Stores model manifests and automatically rolls back
    if new model underperforms.
    """

    def __init__(
        self,
        brain_library: BrainLibrary,
    ) -> None:
        """
        Initialize model versioning system.
        
        Args:
            brain_library: Brain Library instance for storage
        """
        self.brain = brain_library
        logger.info("model_versioning_initialized")

    def register_model_version(
        self,
        model_id: str,
        symbol: str,
        version: int,
        hyperparameters: Dict[str, Any],
        dataset_id: str,
        feature_set: List[str],
        training_metrics: Optional[Dict[str, Any]] = None,
        validation_metrics: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Register a new model version.
        
        Args:
            model_id: Unique model identifier
            symbol: Trading symbol
            version: Version number
            hyperparameters: Model hyperparameters
            dataset_id: Dataset identifier used for training
            feature_set: List of features used
            training_metrics: Training metrics
            validation_metrics: Validation metrics
            
        Returns:
            Manifest ID
        """
        manifest_id = self.brain.store_model_manifest(
            model_id=model_id,
            version=version,
            symbol=symbol,
            hyperparameters=hyperparameters,
            dataset_id=dataset_id,
            feature_set=feature_set,
            training_metrics=training_metrics,
            validation_metrics=validation_metrics,
        )
        
        logger.info(
            "model_version_registered",
            model_id=model_id,
            symbol=symbol,
            version=version,
        )
        
        return manifest_id

    def check_and_rollback(
        self,
        model_id: str,
        symbol: str,
        new_metrics: Dict[str, float],
        rollback_threshold: float = 0.05,  # 5% performance drop triggers rollback
    ) -> bool:
        """
        Check if new model underperforms and rollback if needed.
        
        Args:
            model_id: New model ID
            symbol: Trading symbol
            new_metrics: New model metrics
            rollback_threshold: Performance drop threshold (default: 5%)
            
        Returns:
            True if rollback occurred, False otherwise
        """
        # Get previous model metrics
        previous_model = self.brain.get_active_model(symbol)
        
        if not previous_model:
            # No previous model, accept new one
            logger.info("no_previous_model", symbol=symbol, message="Accepting first model")
            return False
        
        # Get previous model metrics
        previous_metrics = self.brain.get_model_metrics(
            previous_model["model_id"],
            symbol,
        )
        
        if previous_metrics.is_empty():
            # No previous metrics, accept new one
            logger.info("no_previous_metrics", symbol=symbol, message="Accepting new model")
            return False
        
        # Get latest previous metrics
        latest_previous = previous_metrics.head(1)
        prev_sharpe = latest_previous["sharpe_ratio"][0] if "sharpe_ratio" in latest_previous.columns else 0.0
        new_sharpe = new_metrics.get("sharpe_ratio", 0.0)
        
        # Check if new model underperforms
        performance_drop = (prev_sharpe - new_sharpe) / (abs(prev_sharpe) + 1e-6)
        
        if performance_drop > rollback_threshold:
            # Rollback to previous model
            logger.warning(
                "model_rollback_triggered",
                symbol=symbol,
                model_id=model_id,
                previous_sharpe=prev_sharpe,
                new_sharpe=new_sharpe,
                performance_drop=performance_drop,
            )
            
            # Log rollback
            self.brain.log_rollback(
                model_id=model_id,
                previous_version=previous_model["version"],
                reason=f"Performance drop: {performance_drop:.2%} (threshold: {rollback_threshold:.2%})",
            )
            
            # Deactivate new model, reactivate previous
            # This is handled by the registry when we don't register the new model
            return True
        else:
            logger.info(
                "model_accepted",
                symbol=symbol,
                model_id=model_id,
                new_sharpe=new_sharpe,
                previous_sharpe=prev_sharpe,
            )
            return False

    def get_model_manifest(
        self,
        model_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Get model manifest.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Model manifest or None
        """
        return self.brain.get_model_manifest(model_id)

