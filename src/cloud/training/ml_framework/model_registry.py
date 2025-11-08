"""
Model Registry - Unified Model Information for Mechanic

Provides unified interface for all models with purpose, dataset shape, 
feature requirements, and output schema for dynamic selection/retraining.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import structlog

from .base import BaseModel

logger = structlog.get_logger(__name__)


@dataclass
class ModelMetadata:
    """Model metadata for Mechanic."""
    
    name: str
    purpose: str
    ideal_dataset_shape: str
    feature_requirements: List[str]
    output_schema: Dict[str, Any]
    model_type: str  # "regression", "classification", "clustering", "rl"
    market_regimes: List[str]  # Market regimes where model performs well
    hyperparameters: Dict[str, Any]
    performance_metrics: Dict[str, float]


class ModelRegistry:
    """
    Model registry for unified model management.
    
    Provides information about all models for the Mechanic to:
    - Dynamically select models based on market regime
    - Understand feature requirements
    - Know output schemas
    - Trigger retraining
    """
    
    def __init__(self):
        """Initialize model registry."""
        self.models: Dict[str, BaseModel] = {}
        self.metadata: Dict[str, ModelMetadata] = {}
        
        logger.info("model_registry_initialized")
    
    def register_model(
        self,
        model: BaseModel,
        metadata: ModelMetadata,
    ) -> None:
        """
        Register a model with metadata.
        
        Args:
            model: Model instance
            metadata: Model metadata
        """
        self.models[metadata.name] = model
        self.metadata[metadata.name] = metadata
        
        logger.info("model_registered", name=metadata.name, purpose=metadata.purpose)
    
    def get_model(self, name: str) -> Optional[BaseModel]:
        """Get model by name."""
        return self.models.get(name)
    
    def get_metadata(self, name: str) -> Optional[ModelMetadata]:
        """Get model metadata by name."""
        return self.metadata.get(name)
    
    def get_models_by_regime(self, regime: str) -> List[str]:
        """
        Get models suitable for a market regime.
        
        Args:
            regime: Market regime ("trending", "ranging", "volatile", etc.)
            
        Returns:
            List of model names
        """
        suitable_models = [
            name for name, metadata in self.metadata.items()
            if regime in metadata.market_regimes
        ]
        
        logger.info("models_for_regime", regime=regime, num_models=len(suitable_models))
        return suitable_models
    
    def get_all_models_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all models."""
        info = {}
        
        for name, metadata in self.metadata.items():
            info[name] = {
                "purpose": metadata.purpose,
                "ideal_dataset_shape": metadata.ideal_dataset_shape,
                "feature_requirements": metadata.feature_requirements,
                "output_schema": metadata.output_schema,
                "model_type": metadata.model_type,
                "market_regimes": metadata.market_regimes,
                "performance_metrics": metadata.performance_metrics,
            }
        
        return info
    
    def create_metadata_from_model(self, model: BaseModel) -> ModelMetadata:
        """
        Create metadata from model (if model has get_model_info method).
        
        Args:
            model: Model instance
            
        Returns:
            ModelMetadata
        """
        # Try to get model info from model itself
        if hasattr(model, "get_model_info"):
            model_info = model.get_model_info()
            return ModelMetadata(
                name=model.config.name,
                purpose=model_info.get("purpose", "Unknown"),
                ideal_dataset_shape=model_info.get("ideal_dataset_shape", "Unknown"),
                feature_requirements=model_info.get("feature_requirements", []),
                output_schema=model_info.get("output_schema", {}),
                model_type=model.config.model_type,
                market_regimes=model_info.get("market_regimes", []),
                hyperparameters=model.config.hyperparameters,
                performance_metrics={},
            )
        else:
            # Default metadata
            return ModelMetadata(
                name=model.config.name,
                purpose="Machine learning model",
                ideal_dataset_shape="(samples, features)",
                feature_requirements=[],
                output_schema={"predictions": "array"},
                model_type=model.config.model_type,
                market_regimes=[],
                hyperparameters=model.config.hyperparameters,
                performance_metrics={},
            )


# Global registry instance
_registry = ModelRegistry()


def get_registry() -> ModelRegistry:
    """Get global model registry."""
    return _registry

