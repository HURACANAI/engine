"""
Model Bundle Contract

Minimal contract for model bundle: model.bin, config.json, metrics.json, sha256.txt
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class ModelBundle:
    """Model bundle contract."""
    model_id: str
    symbol: str
    model_path: str  # Path to model.bin
    config_path: str  # Path to config.json
    metrics_path: str  # Path to metrics.json
    sha256_path: str  # Path to sha256.txt
    created_at: datetime
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_id": self.model_id,
            "symbol": self.symbol,
            "model_path": self.model_path,
            "config_path": self.config_path,
            "metrics_path": self.metrics_path,
            "sha256_path": self.sha256_path,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata or {},
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ModelBundle:
        """Create from dictionary."""
        return cls(
            model_id=data["model_id"],
            symbol=data["symbol"],
            model_path=data["model_path"],
            config_path=data["config_path"],
            metrics_path=data["metrics_path"],
            sha256_path=data["sha256_path"],
            created_at=datetime.fromisoformat(data["created_at"]),
            metadata=data.get("metadata"),
        )


@dataclass
class ChampionPointer:
    """Champion pointer contract."""
    symbol: str
    model_id: str
    s3_path: str  # S3 path to model bundle
    updated_at: datetime
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "model_id": self.model_id,
            "s3_path": self.s3_path,
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata or {},
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ChampionPointer:
        """Create from dictionary."""
        return cls(
            symbol=data["symbol"],
            model_id=data["model_id"],
            s3_path=data["s3_path"],
            updated_at=datetime.fromisoformat(data["updated_at"]),
            metadata=data.get("metadata"),
        )

