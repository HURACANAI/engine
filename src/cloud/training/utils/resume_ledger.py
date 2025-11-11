"""
Resume Ledger for Training Scheduler

Tracks per-symbol status in runs/YYYYMMDDZ/status.json for resumability.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import structlog

from ..pipelines.work_item import WorkStatus, TrainResult

logger = structlog.get_logger(__name__)


class ResumeLedger:
    """Resume ledger for tracking training status."""
    
    def __init__(
        self,
        run_date: Optional[datetime] = None,
        ledger_dir: str = "runs",
    ):
        """Initialize resume ledger.
        
        Args:
            run_date: Run date (defaults to today)
            ledger_dir: Directory for ledger files
        """
        if run_date is None:
            run_date = datetime.now(timezone.utc)
        
        self.run_date = run_date
        self.ledger_dir = Path(ledger_dir)
        self.ledger_dir.mkdir(parents=True, exist_ok=True)
        
        # Create run directory
        run_date_str = run_date.strftime("%Y%m%dZ")
        self.run_dir = self.ledger_dir / run_date_str
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        self.status_file = self.run_dir / "status.json"
        self.status: Dict[str, Dict[str, Any]] = {}
        
        # Load existing status if available
        self._load_status()
        
        logger.info("resume_ledger_initialized", run_date=run_date_str, status_file=str(self.status_file))
    
    def _load_status(self) -> None:
        """Load status from file."""
        if self.status_file.exists():
            try:
                with open(self.status_file, 'r') as f:
                    self.status = json.load(f)
                logger.info("status_loaded", symbols=len(self.status))
            except Exception as e:
                logger.warning("status_load_failed", error=str(e))
                self.status = {}
        else:
            self.status = {}
    
    def _save_status(self) -> bool:
        """Save status to file."""
        try:
            status_data = {
                "run_date": self.run_date.isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "symbols": self.status,
            }
            
            with open(self.status_file, 'w') as f:
                json.dump(status_data, f, indent=2)
            
            return True
        except Exception as e:
            logger.error("status_save_failed", error=str(e))
            return False
    
    def get_symbol_status(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get status for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Status dictionary, or None if not found
        """
        return self.status.get(symbol)
    
    def mark_pending(self, symbol: str) -> None:
        """Mark symbol as pending.
        
        Args:
            symbol: Trading symbol
        """
        self.status[symbol] = {
            "status": WorkStatus.PENDING.value,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self._save_status()
        logger.debug("symbol_marked_pending", symbol=symbol)
    
    def mark_running(self, symbol: str) -> None:
        """Mark symbol as running.
        
        Args:
            symbol: Trading symbol
        """
        self.status[symbol] = {
            "status": WorkStatus.RUNNING.value,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self._save_status()
        logger.debug("symbol_marked_running", symbol=symbol)
    
    def mark_success(self, symbol: str, result: TrainResult) -> None:
        """Mark symbol as successful.
        
        Args:
            symbol: Trading symbol
            result: Training result
        """
        self.status[symbol] = {
            "status": WorkStatus.SUCCESS.value,
            "started_at": result.started_at.isoformat() if result.started_at else None,
            "ended_at": result.ended_at.isoformat() if result.ended_at else None,
            "wall_minutes": result.wall_minutes,
            "output_path": result.output_path,
            "metrics_path": result.metrics_path,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self._save_status()
        logger.debug("symbol_marked_success", symbol=symbol)
    
    def mark_failed(self, symbol: str, result: TrainResult) -> None:
        """Mark symbol as failed.
        
        Args:
            symbol: Trading symbol
            result: Training result
        """
        self.status[symbol] = {
            "status": WorkStatus.FAILED.value,
            "started_at": result.started_at.isoformat() if result.started_at else None,
            "ended_at": result.ended_at.isoformat() if result.ended_at else None,
            "wall_minutes": result.wall_minutes,
            "error": result.error,
            "error_type": result.error_type,
            "retry_count": result.retry_count,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self._save_status()
        logger.debug("symbol_marked_failed", symbol=symbol, error=result.error)
    
    def mark_skipped(self, symbol: str, reason: str) -> None:
        """Mark symbol as skipped.
        
        Args:
            symbol: Trading symbol
            reason: Skip reason
        """
        self.status[symbol] = {
            "status": WorkStatus.SKIPPED.value,
            "reason": reason,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self._save_status()
        logger.debug("symbol_marked_skipped", symbol=symbol, reason=reason)
    
    def is_completed(self, symbol: str) -> bool:
        """Check if symbol is completed.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            True if completed (success or skipped)
        """
        status = self.status.get(symbol, {}).get("status")
        return status in [WorkStatus.SUCCESS.value, WorkStatus.SKIPPED.value]
    
    def is_running(self, symbol: str) -> bool:
        """Check if symbol is running.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            True if running
        """
        status = self.status.get(symbol, {}).get("status")
        return status == WorkStatus.RUNNING.value
    
    def get_all_symbols(self) -> List[str]:
        """Get all symbols in ledger.
        
        Returns:
            List of symbols
        """
        return list(self.status.keys())
    
    def get_summary(self) -> Dict[str, int]:
        """Get summary of status counts.
        
        Returns:
            Dictionary with status counts
        """
        summary = {
            "pending": 0,
            "running": 0,
            "success": 0,
            "failed": 0,
            "skipped": 0,
            "total": len(self.status),
        }
        
        for symbol_status in self.status.values():
            status = symbol_status.get("status")
            if status in summary:
                summary[status] = summary.get(status, 0) + 1
        
        return summary

