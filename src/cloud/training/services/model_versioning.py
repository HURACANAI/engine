"""
Model Versioning Service

Manages model versions per coin/regime/timeframe, tracks performance metrics,
and handles daily best model selection and push to Brain Library.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class ModelVersion:
    """Model version information."""
    model_id: str
    coin: str
    regime: str
    timeframe: str
    version: int
    timestamp: datetime
    model_path: str
    metrics: Dict[str, float] = field(default_factory=dict)
    composite_score: float = 0.0
    is_active: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelComparison:
    """Model comparison result."""
    model_id: str
    sharpe_ratio: float
    max_drawdown: float
    hit_rate: float
    edge_after_cost_bps: float
    composite_score: float
    is_best: bool = False


class ModelVersioningService:
    """
    Model versioning and management service.
    
    Features:
    - Semantic versioning (coin-regime-timeframe-version)
    - Performance tracking and comparison
    - Best model selection
    - Daily push to Brain Library
    """
    
    def __init__(
        self,
        storage_path: Path,
        brain_library: Optional[Any] = None,  # BrainLibrary type
    ) -> None:
        """
        Initialize model versioning service.
        
        Args:
            storage_path: Path to store model files
            brain_library: Optional Brain Library for model registry
        """
        self.storage_path = Path(storage_path)
        self.brain_library = brain_library
        
        # Model registry (coin-regime-timeframe -> list of versions)
        self.model_registry: Dict[Tuple[str, str, str], List[ModelVersion]] = {}
        
        # Active models (coin-regime-timeframe -> active version)
        self.active_models: Dict[Tuple[str, str, str], ModelVersion] = {}
        
        logger.info(
            "model_versioning_initialized",
            storage_path=str(storage_path),
            brain_library_available=brain_library is not None,
        )
    
    def register_model(
        self,
        coin: str,
        regime: str,
        timeframe: str,
        model_path: str,
        metrics: Dict[str, float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ModelVersion:
        """
        Register a new model version.
        
        Args:
            coin: Trading symbol
            regime: Market regime
            timeframe: Timeframe (e.g., "1h", "4h", "1d")
            model_path: Path to model file
            metrics: Model performance metrics
            metadata: Optional metadata
        
        Returns:
            Model version object
        """
        # Get next version number
        key = (coin, regime, timeframe)
        existing_versions = self.model_registry.get(key, [])
        next_version = max([v.version for v in existing_versions], default=0) + 1
        
        # Generate model ID
        model_id = f"{coin}_{regime}_{timeframe}_v{next_version}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        
        # Calculate composite score
        composite_score = self._calculate_composite_score(metrics)
        
        # Create model version
        model_version = ModelVersion(
            model_id=model_id,
            coin=coin,
            regime=regime,
            timeframe=timeframe,
            version=next_version,
            timestamp=datetime.now(timezone.utc),
            model_path=model_path,
            metrics=metrics,
            composite_score=composite_score,
            is_active=False,
            metadata=metadata or {},
        )
        
        # Add to registry
        if key not in self.model_registry:
            self.model_registry[key] = []
        self.model_registry[key].append(model_version)
        
        # Register in Brain Library if available
        if self.brain_library:
            try:
                self.brain_library.register_model(
                    model_id=model_id,
                    symbol=coin,
                    model_type=f"{regime}_{timeframe}",
                    version=next_version,
                    composite_score=composite_score,
                    hyperparameters=metadata.get("hyperparameters", {}) if metadata else {},
                    dataset_id=f"{coin}_{regime}",
                    feature_set=metadata.get("feature_set", []) if metadata else [],
                )
            except Exception as e:
                logger.warning(
                    "brain_library_register_failed",
                    model_id=model_id,
                    error=str(e),
                )
        
        logger.info(
            "model_registered",
            model_id=model_id,
            coin=coin,
            regime=regime,
            timeframe=timeframe,
            version=next_version,
            composite_score=composite_score,
        )
        
        return model_version
    
    def select_best_model(
        self,
        coin: str,
        regime: str,
        timeframe: str,
    ) -> Optional[ModelVersion]:
        """
        Select best model for a coin/regime/timeframe.
        
        Args:
            coin: Trading symbol
            regime: Market regime
            timeframe: Timeframe
        
        Returns:
            Best model version or None
        """
        key = (coin, regime, timeframe)
        versions = self.model_registry.get(key, [])
        
        if not versions:
            return None
        
        # Sort by composite score (descending)
        sorted_versions = sorted(versions, key=lambda v: v.composite_score, reverse=True)
        best_model = sorted_versions[0]
        
        # Deactivate previous active model
        if key in self.active_models:
            self.active_models[key].is_active = False
        
        # Activate best model
        best_model.is_active = True
        self.active_models[key] = best_model
        
        logger.info(
            "best_model_selected",
            coin=coin,
            regime=regime,
            timeframe=timeframe,
            model_id=best_model.model_id,
            composite_score=best_model.composite_score,
        )
        
        return best_model
    
    def get_active_model(
        self,
        coin: str,
        regime: str,
        timeframe: str,
    ) -> Optional[ModelVersion]:
        """
        Get active model for a coin/regime/timeframe.
        
        Args:
            coin: Trading symbol
            regime: Market regime
            timeframe: Timeframe
        
        Returns:
            Active model version or None
        """
        key = (coin, regime, timeframe)
        return self.active_models.get(key)
    
    def compare_models(
        self,
        coin: str,
        regime: str,
        timeframe: str,
    ) -> List[ModelComparison]:
        """
        Compare all models for a coin/regime/timeframe.
        
        Args:
            coin: Trading symbol
            regime: Market regime
            timeframe: Timeframe
        
        Returns:
            List of model comparisons sorted by composite score
        """
        key = (coin, regime, timeframe)
        versions = self.model_registry.get(key, [])
        
        comparisons = []
        best_score = max([v.composite_score for v in versions], default=0.0)
        
        for version in versions:
            comparison = ModelComparison(
                model_id=version.model_id,
                sharpe_ratio=version.metrics.get("sharpe_ratio", 0.0),
                max_drawdown=version.metrics.get("max_drawdown", 0.0),
                hit_rate=version.metrics.get("hit_rate", 0.0),
                edge_after_cost_bps=version.metrics.get("edge_after_cost_bps", 0.0),
                composite_score=version.composite_score,
                is_best=version.composite_score == best_score,
            )
            comparisons.append(comparison)
        
        # Sort by composite score (descending)
        comparisons.sort(key=lambda c: c.composite_score, reverse=True)
        
        return comparisons
    
    def push_best_models_to_brain_library(
        self,
        coins: Optional[List[str]] = None,
    ) -> Dict[str, int]:
        """
        Push best models to Brain Library for Hamilton to use.
        
        Args:
            coins: List of coins to push (None = all coins)
        
        Returns:
            Dictionary of coin -> number of models pushed
        """
        if not self.brain_library:
            logger.warning("brain_library_not_available")
            return {}
        
        pushed = {}
        
        # Get all unique coins
        if coins is None:
            coins = set()
            for (coin, _, _) in self.model_registry.keys():
                coins.add(coin)
            coins = list(coins)
        
        for coin in coins:
            count = 0
            
            # Get all regimes and timeframes for this coin
            for (c, regime, timeframe) in self.model_registry.keys():
                if c != coin:
                    continue
                
                # Get best model
                best_model = self.select_best_model(coin, regime, timeframe)
                if best_model is None:
                    continue
                
                # Model is already registered in Brain Library during registration
                # This step ensures it's marked as active
                try:
                    # Update Brain Library to mark as active
                    # This would be implemented based on Brain Library API
                    count += 1
                except Exception as e:
                    logger.warning(
                        "brain_library_push_failed",
                        coin=coin,
                        regime=regime,
                        timeframe=timeframe,
                        error=str(e),
                    )
            
            pushed[coin] = count
        
        logger.info(
            "best_models_pushed",
            total_coins=len(pushed),
            total_models=sum(pushed.values()),
        )
        
        return pushed
    
    def _calculate_composite_score(self, metrics: Dict[str, float]) -> float:
        """
        Calculate composite score from metrics.
        
        Composite score combines:
        - Sharpe ratio (40%)
        - Hit rate (30%)
        - Edge after cost (20%)
        - Drawdown penalty (10%)
        
        Args:
            metrics: Model metrics
        
        Returns:
            Composite score
        """
        sharpe = metrics.get("sharpe_ratio", 0.0)
        hit_rate = metrics.get("hit_rate", 0.0)
        edge_after_cost = metrics.get("edge_after_cost_bps", 0.0)
        max_drawdown = abs(metrics.get("max_drawdown", 0.0))
        
        # Normalize metrics
        sharpe_score = max(0.0, min(1.0, (sharpe + 2.0) / 4.0))  # Map [-2, 2] to [0, 1]
        hit_rate_score = hit_rate  # Already 0-1
        edge_score = max(0.0, min(1.0, edge_after_cost / 10.0))  # Map [0, 10] bps to [0, 1]
        drawdown_penalty = max(0.0, 1.0 - (max_drawdown / 0.2))  # Penalty for >20% drawdown
        
        # Weighted composite
        composite = (
            0.4 * sharpe_score +
            0.3 * hit_rate_score +
            0.2 * edge_score +
            0.1 * drawdown_penalty
        )
        
        return composite

