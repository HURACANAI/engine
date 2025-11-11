"""
Work Item for Training Scheduler

Represents a single training work item for a symbol.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional

import structlog  # type: ignore[import-untyped]

logger = structlog.get_logger(__name__)


class WorkStatus(Enum):
    """Work item status."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    TIMEOUT = "timeout"


@dataclass
class TrainResult:
    """Result of training a symbol."""
    symbol: str
    status: str  # "success", "failed", "skipped", "timeout"
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    wall_minutes: float = 0.0
    output_path: str = ""
    metrics_path: Optional[str] = None
    error: Optional[str] = None
    error_type: Optional[str] = None
    retry_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "status": self.status,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "wall_minutes": self.wall_minutes,
            "output_path": self.output_path,
            "metrics_path": self.metrics_path,
            "error": self.error,
            "error_type": self.error_type,
            "retry_count": self.retry_count,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TrainResult:
        """Create from dictionary."""
        return cls(
            symbol=data["symbol"],
            status=data["status"],
            started_at=datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None,
            ended_at=datetime.fromisoformat(data["ended_at"]) if data.get("ended_at") else None,
            wall_minutes=data.get("wall_minutes", 0.0),
            output_path=data.get("output_path", ""),
            metrics_path=data.get("metrics_path"),
            error=data.get("error"),
            error_type=data.get("error_type"),
            retry_count=data.get("retry_count", 0),
        )


@dataclass
class WorkItem:
    """Work item for training a symbol."""
    symbol: str
    status: WorkStatus = WorkStatus.PENDING
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    result: Optional[TrainResult] = None
    retry_count: int = 0
    max_retries: int = 2
    
    def start(self) -> None:
        """Mark work item as started."""
        self.status = WorkStatus.RUNNING
        self.started_at = datetime.now(timezone.utc)
        logger.info("work_item_started", symbol=self.symbol)
    
    def complete(self, result: TrainResult) -> None:
        """Mark work item as completed."""
        self.status = WorkStatus.SUCCESS if result.status == "success" else WorkStatus.FAILED
        self.ended_at = datetime.now(timezone.utc)
        self.result = result
        logger.info("work_item_completed", symbol=self.symbol, status=result.status)
    
    def fail(self, error: str, error_type: Optional[str] = None) -> None:
        """Mark work item as failed."""
        self.status = WorkStatus.FAILED
        self.ended_at = datetime.now(timezone.utc)
        self.result = TrainResult(
            symbol=self.symbol,
            status="failed",
            started_at=self.started_at,
            ended_at=self.ended_at,
            error=error,
            error_type=error_type,
            retry_count=self.retry_count,
        )
        logger.error("work_item_failed", symbol=self.symbol, error=error, error_type=error_type)
    
    def skip(self, reason: str) -> None:
        """Mark work item as skipped."""
        self.status = WorkStatus.SKIPPED
        self.ended_at = datetime.now(timezone.utc)
        self.result = TrainResult(
            symbol=self.symbol,
            status="skipped",
            started_at=self.started_at,
            ended_at=self.ended_at,
            error=reason,
            retry_count=self.retry_count,
        )
        logger.info("work_item_skipped", symbol=self.symbol, reason=reason)
    
    def should_retry(self) -> bool:
        """Check if work item should be retried."""
        return self.retry_count < self.max_retries and self.status == WorkStatus.FAILED
    
    def retry(self) -> None:
        """Retry work item."""
        self.retry_count += 1
        self.status = WorkStatus.PENDING
        self.started_at = None
        self.ended_at = None
        self.result = None
        logger.info("work_item_retry", symbol=self.symbol, retry_count=self.retry_count)

