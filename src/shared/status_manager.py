"""
Status Manager

Manages engine_status.json with phase, symbols_done, symbols_failed, ETA.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog

logger = structlog.get_logger(__name__)


class StatusManager:
    """Manages engine status."""
    
    def __init__(self, status_file: str = "engine_status.json"):
        """Initialize status manager.
        
        Args:
            status_file: Path to status file
        """
        self.status_file = Path(status_file)
        logger.info("status_manager_initialized", status_file=str(self.status_file))
    
    def update_status(
        self,
        phase: str,
        symbols_done: int = 0,
        symbols_failed: int = 0,
        symbols_total: int = 0,
        current_symbol: Optional[str] = None,
        eta_seconds: Optional[int] = None,
        last_error: Optional[str] = None,
    ) -> bool:
        """Update engine status.
        
        Args:
            phase: Current phase ("loading", "training", "validating", "publishing", "complete")
            symbols_done: Number of symbols completed
            symbols_failed: Number of symbols failed
            symbols_total: Total number of symbols
            current_symbol: Current symbol being processed
            eta_seconds: Estimated time to completion in seconds
            last_error: Last error message
            
        Returns:
            True if successful
        """
        status = {
            "phase": phase,
            "symbols_done": symbols_done,
            "symbols_failed": symbols_failed,
            "symbols_total": symbols_total,
            "current_symbol": current_symbol,
            "eta_seconds": eta_seconds,
            "last_error": last_error,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        
        try:
            self.status_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.status_file, 'w') as f:
                json.dump(status, f, indent=2)
            
            logger.debug("status_updated", phase=phase, symbols_done=symbols_done, symbols_total=symbols_total)
            return True
        except Exception as e:
            logger.error("status_update_failed", error=str(e))
            return False
    
    def load_status(self) -> Dict[str, Any]:
        """Load engine status.
        
        Returns:
            Status dictionary
        """
        if not self.status_file.exists():
            return {
                "phase": "idle",
                "symbols_done": 0,
                "symbols_failed": 0,
                "symbols_total": 0,
                "current_symbol": None,
                "eta_seconds": None,
                "last_error": None,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
        
        with open(self.status_file, 'r') as f:
            status = json.load(f)
        
        return status

