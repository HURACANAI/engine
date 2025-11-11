"""
Feature Store with Versioning

Manages feature registration, versioning, and feature set pinning.
Ensures reproducibility by pinning feature sets to run manifests.

Author: Huracan Engine Team
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Any
from enum import Enum

import structlog

logger = structlog.get_logger(__name__)


class FeatureStatus(Enum):
    """Feature status"""
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    EXPERIMENTAL = "experimental"


@dataclass
class FeatureDefinition:
    """Definition of a single feature"""
    name: str
    description: str
    version: str
    status: FeatureStatus = FeatureStatus.ACTIVE
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "status": self.status.value,
            "dependencies": self.dependencies,
            "metadata": self.metadata,
            "created_at": self.created_at
        }


@dataclass
class FeatureSet:
    """A set of features with a unique ID"""
    feature_set_id: str
    features: List[str]  # List of feature names
    version: str
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "feature_set_id": self.feature_set_id,
            "features": sorted(self.features),  # Sort for consistency
            "version": self.version,
            "created_at": self.created_at,
            "metadata": self.metadata
        }
    
    def to_hash(self) -> str:
        """Generate hash for feature set"""
        data = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(data.encode()).hexdigest()[:16]


class FeatureStore:
    """
    Centralized feature store with versioning.
    
    Features:
    - Register features with metadata
    - Version features
    - Create feature sets
    - Pin feature sets to run manifests
    - Track feature dependencies
    - Validate feature availability
    """
    
    def __init__(self, store_path: Optional[Path] = None):
        """
        Initialize feature store.
        
        Args:
            store_path: Path to store metadata (default: .feature_store/)
        """
        self.store_path = store_path or Path(".feature_store")
        self.store_path.mkdir(parents=True, exist_ok=True)
        
        self.features: Dict[str, FeatureDefinition] = {}
        self.feature_sets: Dict[str, FeatureSet] = {}
        
        # Load existing features
        self._load_features()
        
        logger.info(
            "feature_store_initialized",
            store_path=str(self.store_path),
            feature_count=len(self.features),
            feature_set_count=len(self.feature_sets)
        )
    
    def register_feature(
        self,
        name: str,
        description: str,
        version: str = "1.0.0",
        status: FeatureStatus = FeatureStatus.ACTIVE,
        dependencies: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> FeatureDefinition:
        """
        Register a new feature.
        
        Args:
            name: Feature name (e.g., "book_imbalance")
            description: Feature description
            version: Feature version
            status: Feature status
            dependencies: List of feature names this depends on
            metadata: Additional metadata
        
        Returns:
            FeatureDefinition
        """
        if name in self.features:
            existing = self.features[name]
            if existing.version != version:
                logger.warning(
                    "feature_version_change",
                    feature=name,
                    old_version=existing.version,
                    new_version=version
                )
        
        feature = FeatureDefinition(
            name=name,
            description=description,
            version=version,
            status=status,
            dependencies=dependencies or [],
            metadata=metadata or {}
        )
        
        self.features[name] = feature
        self._save_features()
        
        logger.info(
            "feature_registered",
            feature=name,
            version=version,
            status=status.value
        )
        
        return feature
    
    def get_feature(self, name: str) -> Optional[FeatureDefinition]:
        """Get feature definition by name"""
        return self.features.get(name)
    
    def list_features(self, status: Optional[FeatureStatus] = None) -> List[FeatureDefinition]:
        """List all features, optionally filtered by status"""
        features = list(self.features.values())
        if status:
            features = [f for f in features if f.status == status]
        return features
    
    def create_feature_set(
        self,
        features: List[str],
        version: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> FeatureSet:
        """
        Create a new feature set.
        
        Args:
            features: List of feature names
            version: Optional version string (auto-generated if None)
            metadata: Optional metadata
        
        Returns:
            FeatureSet with unique ID
        """
        # Validate all features exist
        missing = [f for f in features if f not in self.features]
        if missing:
            raise ValueError(f"Features not registered: {missing}")
        
        # Generate feature set ID
        if version is None:
            version = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        
        # Create hash-based ID
        features_sorted = sorted(features)
        features_str = ",".join(features_sorted)
        feature_set_id = f"fs_{hashlib.sha256(features_str.encode()).hexdigest()[:12]}_{version}"
        
        feature_set = FeatureSet(
            feature_set_id=feature_set_id,
            features=features,
            version=version,
            metadata=metadata or {}
        )
        
        self.feature_sets[feature_set_id] = feature_set
        self._save_feature_sets()
        
        logger.info(
            "feature_set_created",
            feature_set_id=feature_set_id,
            feature_count=len(features),
            version=version
        )
        
        return feature_set
    
    def get_feature_set(self, feature_set_id: str) -> Optional[FeatureSet]:
        """Get feature set by ID"""
        return self.feature_sets.get(feature_set_id)
    
    def pin_feature_set(self, feature_set_id: str, run_id: str) -> None:
        """
        Pin a feature set to a run manifest.
        
        Args:
            feature_set_id: Feature set ID
            run_id: Run ID to pin to
        """
        if feature_set_id not in self.feature_sets:
            raise ValueError(f"Feature set not found: {feature_set_id}")
        
        # Save pinning to manifest
        manifest_path = self.store_path / "manifests" / f"{run_id}.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        
        manifest = {
            "run_id": run_id,
            "feature_set_id": feature_set_id,
            "pinned_at": datetime.now(timezone.utc).isoformat(),
            "feature_set": self.feature_sets[feature_set_id].to_dict()
        }
        
        manifest_path.write_text(json.dumps(manifest, indent=2))
        
        logger.info(
            "feature_set_pinned",
            run_id=run_id,
            feature_set_id=feature_set_id
        )
    
    def get_pinned_feature_set(self, run_id: str) -> Optional[FeatureSet]:
        """Get feature set pinned to a run"""
        manifest_path = self.store_path / "manifests" / f"{run_id}.json"
        if not manifest_path.exists():
            return None
        
        manifest = json.loads(manifest_path.read_text())
        feature_set_id = manifest.get("feature_set_id")
        return self.get_feature_set(feature_set_id) if feature_set_id else None
    
    def validate_features(self, feature_names: List[str]) -> tuple[bool, List[str]]:
        """
        Validate that all features are available.
        
        Args:
            feature_names: List of feature names to validate
        
        Returns:
            (is_valid, missing_features)
        """
        missing = [f for f in feature_names if f not in self.features]
        return len(missing) == 0, missing
    
    def _load_features(self) -> None:
        """Load features from disk"""
        features_path = self.store_path / "features.json"
        if features_path.exists():
            try:
                data = json.loads(features_path.read_text())
                for name, feature_data in data.items():
                    self.features[name] = FeatureDefinition(
                        name=feature_data["name"],
                        description=feature_data["description"],
                        version=feature_data["version"],
                        status=FeatureStatus(feature_data.get("status", "active")),
                        dependencies=feature_data.get("dependencies", []),
                        metadata=feature_data.get("metadata", {}),
                        created_at=feature_data.get("created_at", datetime.now(timezone.utc).isoformat())
                    )
            except Exception as e:
                logger.warning("feature_load_failed", error=str(e))
    
    def _save_features(self) -> None:
        """Save features to disk"""
        features_path = self.store_path / "features.json"
        data = {name: feature.to_dict() for name, feature in self.features.items()}
        features_path.write_text(json.dumps(data, indent=2))
    
    def _save_feature_sets(self) -> None:
        """Save feature sets to disk"""
        feature_sets_path = self.store_path / "feature_sets.json"
        data = {fs_id: fs.to_dict() for fs_id, fs in self.feature_sets.items()}
        feature_sets_path.write_text(json.dumps(data, indent=2))
    
    def _load_feature_sets(self) -> None:
        """Load feature sets from disk"""
        feature_sets_path = self.store_path / "feature_sets.json"
        if feature_sets_path.exists():
            try:
                data = json.loads(feature_sets_path.read_text())
                for fs_id, fs_data in data.items():
                    self.feature_sets[fs_id] = FeatureSet(
                        feature_set_id=fs_data["feature_set_id"],
                        features=fs_data["features"],
                        version=fs_data["version"],
                        created_at=fs_data.get("created_at", datetime.now(timezone.utc).isoformat()),
                        metadata=fs_data.get("metadata", {})
                    )
            except Exception as e:
                logger.warning("feature_set_load_failed", error=str(e))

