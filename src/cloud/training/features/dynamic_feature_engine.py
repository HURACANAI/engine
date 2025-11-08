"""
Dynamic Feature Engineering System

Dynamically generates features per symbol and stores them in Brain Library.
Integrates with FeatureRecipe for consistent feature generation.

Key Features:
- Dynamic feature generation per symbol
- Brain Library storage
- Feature versioning
- Feature dependency tracking
- Automated feature generation
- Feature caching

Author: Huracan Engine Team
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any, Set
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json

import polars as pl
import structlog

from shared.features.recipe import FeatureRecipe

logger = structlog.get_logger(__name__)


@dataclass
class FeatureDefinition:
    """Feature definition"""
    name: str
    description: str
    dependencies: List[str]  # Required input columns
    generator: Callable[[pl.DataFrame], pl.Expr]  # Feature generator function
    version: int = 1
    tags: List[str] = field(default_factory=list)
    cached: bool = False


@dataclass
class FeatureSet:
    """Feature set metadata"""
    name: str
    description: str
    features: List[str]  # Feature names
    version: int = 1
    symbol: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)


class DynamicFeatureEngine:
    """
    Dynamic Feature Engineering System.
    
    Generates features dynamically per symbol and stores them in Brain Library.
    
    Usage:
        engine = DynamicFeatureEngine(brain_library_path="brain_library")
        
        # Register feature
        engine.register_feature(
            name="custom_feature",
            description="Custom feature",
            dependencies=["close", "volume"],
            generator=lambda df: (pl.col("close") / pl.col("volume")).alias("custom_feature")
        )
        
        # Generate features for symbol
        features = engine.generate_features(data, symbol="BTC-USD")
    """
    
    def __init__(
        self,
        brain_library_path: Optional[Path] = None,
        use_cache: bool = True
    ):
        """
        Initialize dynamic feature engine.
        
        Args:
            brain_library_path: Path to Brain Library storage
            use_cache: Whether to use feature caching
        """
        self.brain_library_path = brain_library_path or Path("brain_library/features")
        self.brain_library_path.mkdir(parents=True, exist_ok=True)
        self.use_cache = use_cache
        
        # Feature registry
        self.feature_definitions: Dict[str, FeatureDefinition] = {}
        
        # Feature sets
        self.feature_sets: Dict[str, FeatureSet] = {}
        
        # Feature cache
        self.feature_cache: Dict[str, pl.DataFrame] = {}
        
        # Initialize with FeatureRecipe
        self.feature_recipe = FeatureRecipe()
        
        logger.info(
            "dynamic_feature_engine_initialized",
            brain_library_path=str(self.brain_library_path),
            use_cache=use_cache
        )
    
    def register_feature(
        self,
        name: str,
        description: str,
        dependencies: List[str],
        generator: Callable[[pl.DataFrame], pl.Expr],
        version: int = 1,
        tags: Optional[List[str]] = None,
        cached: bool = False
    ) -> None:
        """
        Register a feature definition.
        
        Args:
            name: Feature name
            description: Feature description
            dependencies: Required input columns
            generator: Feature generator function
            version: Feature version
            tags: Feature tags
            cached: Whether to cache this feature
        """
        feature_def = FeatureDefinition(
            name=name,
            description=description,
            dependencies=dependencies,
            generator=generator,
            version=version,
            tags=tags or [],
            cached=cached
        )
        
        self.feature_definitions[name] = feature_def
        
        logger.info(
            "feature_registered",
            name=name,
            version=version,
            dependencies=dependencies
        )
    
    def generate_features(
        self,
        data: pl.DataFrame,
        symbol: str,
        feature_set: Optional[str] = None,
        force_regenerate: bool = False
    ) -> pl.DataFrame:
        """
        Generate features for a symbol.
        
        Args:
            data: Input data
            symbol: Symbol name
            feature_set: Feature set name (optional)
            force_regenerate: Force regeneration even if cached
        
        Returns:
            DataFrame with generated features
        """
        # Check cache
        cache_key = self._get_cache_key(symbol, feature_set)
        
        if self.use_cache and not force_regenerate and cache_key in self.feature_cache:
            logger.debug("features_loaded_from_cache", symbol=symbol, cache_key=cache_key)
            return self.feature_cache[cache_key]
        
        # Get feature set
        if feature_set:
            features_to_generate = self._get_feature_set_features(feature_set)
        else:
            # Generate all registered features
            features_to_generate = list(self.feature_definitions.keys())
        
        # Start with input data
        result_df = data.clone()
        
        # Generate features in dependency order
        generated_features = set(data.columns)
        remaining_features = set(features_to_generate)
        
        while remaining_features:
            progress_made = False
            
            for feature_name in list(remaining_features):
                feature_def = self.feature_definitions.get(feature_name)
                
                if not feature_def:
                    logger.warning("feature_not_found", feature=feature_name)
                    remaining_features.remove(feature_name)
                    continue
                
                # Check if dependencies are available
                if all(dep in generated_features for dep in feature_def.dependencies):
                    try:
                        # Generate feature
                        feature_expr = feature_def.generator(result_df)
                        result_df = result_df.with_columns([feature_expr])
                        
                        generated_features.add(feature_name)
                        remaining_features.remove(feature_name)
                        progress_made = True
                        
                        logger.debug(
                            "feature_generated",
                            feature=feature_name,
                            symbol=symbol
                        )
                    except Exception as e:
                        logger.error(
                            "feature_generation_failed",
                            feature=feature_name,
                            symbol=symbol,
                            error=str(e)
                        )
                        remaining_features.remove(feature_name)
            
            if not progress_made:
                # Circular dependency or missing dependencies
                logger.warning(
                    "feature_generation_stalled",
                    remaining_features=list(remaining_features),
                    symbol=symbol
                )
                break
        
        # Also generate features from FeatureRecipe
        try:
            recipe_features = self.feature_recipe.build(result_df)
            result_df = recipe_features
        except Exception as e:
            logger.warning("feature_recipe_failed", error=str(e))
        
        # Cache result
        if self.use_cache:
            self.feature_cache[cache_key] = result_df
            self._save_feature_cache(symbol, feature_set, result_df)
        
        logger.info(
            "features_generated",
            symbol=symbol,
            feature_count=len(result_df.columns) - len(data.columns),
            total_features=len(result_df.columns)
        )
        
        return result_df
    
    def create_feature_set(
        self,
        name: str,
        description: str,
        features: List[str],
        symbol: Optional[str] = None,
        version: int = 1,
        metadata: Optional[Dict[str, Any]] = None
    ) -> FeatureSet:
        """
        Create a feature set.
        
        Args:
            name: Feature set name
            description: Feature set description
            features: List of feature names
            symbol: Symbol (optional, for symbol-specific feature sets)
            version: Feature set version
            metadata: Additional metadata
        
        Returns:
            FeatureSet
        """
        feature_set = FeatureSet(
            name=name,
            description=description,
            features=features,
            version=version,
            symbol=symbol,
            metadata=metadata or {}
        )
        
        self.feature_sets[name] = feature_set
        
        # Save to Brain Library
        self._save_feature_set(feature_set)
        
        logger.info(
            "feature_set_created",
            name=name,
            feature_count=len(features),
            symbol=symbol
        )
        
        return feature_set
    
    def get_feature_set(self, name: str) -> Optional[FeatureSet]:
        """Get feature set by name"""
        return self.feature_sets.get(name)
    
    def list_feature_sets(self, symbol: Optional[str] = None) -> List[FeatureSet]:
        """List feature sets"""
        if symbol:
            return [
                fs for fs in self.feature_sets.values()
                if fs.symbol == symbol or fs.symbol is None
            ]
        return list(self.feature_sets.values())
    
    def _get_feature_set_features(self, feature_set_name: str) -> List[str]:
        """Get features in a feature set"""
        feature_set = self.feature_sets.get(feature_set_name)
        if not feature_set:
            return []
        return feature_set.features
    
    def _get_cache_key(self, symbol: str, feature_set: Optional[str]) -> str:
        """Get cache key for symbol and feature set"""
        key = f"{symbol}_{feature_set or 'default'}"
        return hashlib.md5(key.encode()).hexdigest()
    
    def _save_feature_cache(
        self,
        symbol: str,
        feature_set: Optional[str],
        features: pl.DataFrame
    ) -> None:
        """Save feature cache to disk"""
        cache_dir = self.brain_library_path / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        cache_key = self._get_cache_key(symbol, feature_set)
        cache_path = cache_dir / f"{cache_key}.parquet"
        
        try:
            features.write_parquet(cache_path)
            logger.debug("feature_cache_saved", path=str(cache_path))
        except Exception as e:
            logger.warning("feature_cache_save_failed", error=str(e))
    
    def _load_feature_cache(
        self,
        symbol: str,
        feature_set: Optional[str]
    ) -> Optional[pl.DataFrame]:
        """Load feature cache from disk"""
        cache_dir = self.brain_library_path / "cache"
        cache_key = self._get_cache_key(symbol, feature_set)
        cache_path = cache_dir / f"{cache_key}.parquet"
        
        if cache_path.exists():
            try:
                return pl.read_parquet(cache_path)
            except Exception as e:
                logger.warning("feature_cache_load_failed", error=str(e))
        
        return None
    
    def _save_feature_set(self, feature_set: FeatureSet) -> None:
        """Save feature set to Brain Library"""
        feature_sets_dir = self.brain_library_path / "feature_sets"
        feature_sets_dir.mkdir(parents=True, exist_ok=True)
        
        feature_set_path = feature_sets_dir / f"{feature_set.name}_v{feature_set.version}.json"
        
        feature_set_data = {
            "name": feature_set.name,
            "description": feature_set.description,
            "features": feature_set.features,
            "version": feature_set.version,
            "symbol": feature_set.symbol,
            "created_at": feature_set.created_at.isoformat(),
            "metadata": feature_set.metadata
        }
        
        try:
            with open(feature_set_path, 'w') as f:
                json.dump(feature_set_data, f, indent=2)
            logger.debug("feature_set_saved", path=str(feature_set_path))
        except Exception as e:
            logger.warning("feature_set_save_failed", error=str(e))
    
    def _load_feature_sets(self) -> None:
        """Load feature sets from Brain Library"""
        feature_sets_dir = self.brain_library_path / "feature_sets"
        
        if not feature_sets_dir.exists():
            return
        
        for feature_set_path in feature_sets_dir.glob("*.json"):
            try:
                with open(feature_set_path, 'r') as f:
                    feature_set_data = json.load(f)
                
                feature_set = FeatureSet(
                    name=feature_set_data["name"],
                    description=feature_set_data["description"],
                    features=feature_set_data["features"],
                    version=feature_set_data["version"],
                    symbol=feature_set_data.get("symbol"),
                    created_at=datetime.fromisoformat(feature_set_data["created_at"]),
                    metadata=feature_set_data.get("metadata", {})
                )
                
                self.feature_sets[feature_set.name] = feature_set
            except Exception as e:
                logger.warning("feature_set_load_failed", path=str(feature_set_path), error=str(e))
    
    def get_feature_dependencies(self, feature_name: str) -> Set[str]:
        """Get feature dependencies"""
        feature_def = self.feature_definitions.get(feature_name)
        if not feature_def:
            return set()
        
        dependencies = set(feature_def.dependencies)
        
        # Recursively get dependencies of dependencies
        for dep in feature_def.dependencies:
            dependencies.update(self.get_feature_dependencies(dep))
        
        return dependencies
    
    def validate_feature_set(self, feature_set_name: str) -> Dict[str, Any]:
        """Validate a feature set"""
        feature_set = self.feature_sets.get(feature_set_name)
        if not feature_set:
            return {"valid": False, "error": "Feature set not found"}
        
        validation_result = {
            "valid": True,
            "missing_features": [],
            "circular_dependencies": [],
            "missing_dependencies": []
        }
        
        # Check if all features exist
        for feature_name in feature_set.features:
            if feature_name not in self.feature_definitions:
                validation_result["missing_features"].append(feature_name)
                validation_result["valid"] = False
        
        # Check dependencies
        for feature_name in feature_set.features:
            feature_def = self.feature_definitions.get(feature_name)
            if feature_def:
                for dep in feature_def.dependencies:
                    if dep not in feature_set.features and dep not in ["open", "high", "low", "close", "volume"]:
                        validation_result["missing_dependencies"].append((feature_name, dep))
                        validation_result["valid"] = False
        
        return validation_result

