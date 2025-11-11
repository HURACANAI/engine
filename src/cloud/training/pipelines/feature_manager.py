"""
Feature Manager - Graceful Degradation System

Tracks which features are working and which have failed.
If a feature fails, it's disabled but the engine continues running.
Only truly critical features (like exchange client) will stop the engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Set
from enum import Enum

import structlog

logger = structlog.get_logger(__name__)


class FeatureStatus(Enum):
    """Status of a feature."""
    WORKING = "working"
    FAILED = "failed"
    DISABLED = "disabled"
    CRITICAL_FAILED = "critical_failed"  # This will stop the engine


@dataclass
class FeatureInfo:
    """Information about a feature."""
    name: str
    status: FeatureStatus
    error: Optional[str] = None
    enabled: bool = True
    critical: bool = False  # If True, failure stops the engine
    last_error_time: Optional[float] = None


class FeatureManager:
    """Manages feature status and graceful degradation."""
    
    def __init__(self):
        """Initialize feature manager."""
        self.features: Dict[str, FeatureInfo] = {}
        self.critical_failures: Set[str] = set()
        logger.info("feature_manager_initialized")
    
    def register_feature(
        self,
        name: str,
        critical: bool = False,
        enabled: bool = True,
    ) -> None:
        """
        Register a feature.
        
        Args:
            name: Feature name
            critical: If True, failure stops the engine
            enabled: If False, feature is disabled from start
        """
        status = FeatureStatus.DISABLED if not enabled else FeatureStatus.WORKING
        self.features[name] = FeatureInfo(
            name=name,
            status=status,
            enabled=enabled,
            critical=critical,
        )
        logger.info("feature_registered", name=name, critical=critical, enabled=enabled)
    
    def mark_feature_working(self, name: str) -> None:
        """Mark a feature as working."""
        if name in self.features:
            self.features[name].status = FeatureStatus.WORKING
            self.features[name].error = None
            if name in self.critical_failures:
                self.critical_failures.remove(name)
            logger.info("feature_working", name=name)
    
    def mark_feature_failed(
        self,
        name: str,
        error: str,
        stop_engine: bool = False,
    ) -> None:
        """
        Mark a feature as failed.
        
        Args:
            name: Feature name
            error: Error message
            stop_engine: If True, mark as critical failure (will stop engine)
        """
        import time
        
        if name not in self.features:
            # Auto-register if not registered
            self.register_feature(name, critical=stop_engine)
        
        feature = self.features[name]
        feature.status = FeatureStatus.CRITICAL_FAILED if stop_engine else FeatureStatus.FAILED
        feature.error = error
        feature.last_error_time = time.time()
        
        if stop_engine or feature.critical:
            self.critical_failures.add(name)
            logger.error(
                "critical_feature_failed",
                name=name,
                error=error,
                message="This feature is critical - engine may stop",
            )
        else:
            logger.warning(
                "feature_failed_non_critical",
                name=name,
                error=error,
                message="Feature failed but engine will continue",
            )
    
    def is_feature_working(self, name: str) -> bool:
        """Check if a feature is working."""
        if name not in self.features:
            return True  # Unknown features assumed working
        feature = self.features[name]
        return (
            feature.enabled
            and feature.status == FeatureStatus.WORKING
        )
    
    def should_stop_engine(self) -> bool:
        """Check if engine should stop due to critical failures."""
        return len(self.critical_failures) > 0
    
    def get_status_report(self) -> Dict[str, Any]:
        """Get status report of all features."""
        working = [f for f in self.features.values() if f.status == FeatureStatus.WORKING]
        failed = [f for f in self.features.values() if f.status == FeatureStatus.FAILED]
        critical_failed = [
            f for f in self.features.values()
            if f.status == FeatureStatus.CRITICAL_FAILED
        ]
        disabled = [f for f in self.features.values() if f.status == FeatureStatus.DISABLED]
        
        return {
            "total_features": len(self.features),
            "working": len(working),
            "failed": len(failed),
            "critical_failed": len(critical_failed),
            "disabled": len(disabled),
            "working_features": [f.name for f in working],
            "failed_features": [f.name for f in failed],
            "critical_failed_features": [f.name for f in critical_failed],
            "disabled_features": [f.name for f in disabled],
            "should_stop_engine": self.should_stop_engine(),
        }
    
    def get_feature_info(self, name: str) -> Optional[FeatureInfo]:
        """Get information about a specific feature."""
        return self.features.get(name)


# Global feature manager instance
_feature_manager: Optional[FeatureManager] = None


def get_feature_manager() -> FeatureManager:
    """Get the global feature manager instance."""
    global _feature_manager
    if _feature_manager is None:
        _feature_manager = FeatureManager()
    return _feature_manager






