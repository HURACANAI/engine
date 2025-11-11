"""
Download Progress Tracker

Tracks the progress of downloading historical data for all symbols during training.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Optional
from enum import Enum

import structlog

logger = structlog.get_logger(__name__)


class DownloadStatus(Enum):
    """Status of a download."""
    PENDING = "pending"
    DOWNLOADING = "downloading"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class SymbolDownloadProgress:
    """Progress information for a single symbol."""
    symbol: str
    status: DownloadStatus
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    rows_downloaded: int = 0
    error: Optional[str] = None
    batch_num: Optional[int] = None
    task_num: Optional[int] = None
    total_tasks: Optional[int] = None


@dataclass
class DownloadProgress:
    """Overall download progress."""
    total_symbols: int = 0
    completed: int = 0
    downloading: int = 0
    failed: int = 0
    pending: int = 0
    symbols: Dict[str, SymbolDownloadProgress] = field(default_factory=dict)
    start_time: Optional[float] = None
    batch_num: Optional[int] = None
    total_batches: Optional[int] = None
    
    def get_progress_percent(self) -> float:
        """Get overall progress percentage."""
        if self.total_symbols == 0:
            return 0.0
        return (self.completed + self.failed) / self.total_symbols * 100.0
    
    def get_elapsed_time(self) -> Optional[float]:
        """Get elapsed time in seconds."""
        if self.start_time:
            return time.time() - self.start_time
        return None
    
    def get_estimated_remaining_time(self) -> Optional[float]:
        """Estimate remaining time in seconds."""
        elapsed = self.get_elapsed_time()
        if elapsed is None or self.completed == 0:
            return None
        
        avg_time_per_symbol = elapsed / self.completed
        remaining_symbols = self.total_symbols - self.completed - self.failed
        return avg_time_per_symbol * remaining_symbols


class DownloadProgressTracker:
    """Tracks download progress for all symbols."""
    
    def __init__(self):
        """Initialize progress tracker."""
        self._progress: Optional[DownloadProgress] = None
        logger.info("download_progress_tracker_initialized")
    
    def start_download_session(
        self,
        total_symbols: int,
        batch_num: Optional[int] = None,
        total_batches: Optional[int] = None,
    ) -> None:
        """
        Start a new download session.
        
        Args:
            total_symbols: Total number of symbols to download
            batch_num: Current batch number (optional)
            total_batches: Total number of batches (optional)
        """
        self._progress = DownloadProgress(
            total_symbols=total_symbols,
            completed=0,
            downloading=0,
            failed=0,
            pending=total_symbols,
            start_time=time.time(),
            batch_num=batch_num,
            total_batches=total_batches,
        )
        logger.info(
            "download_session_started",
            total_symbols=total_symbols,
            batch_num=batch_num,
            total_batches=total_batches,
        )
    
    def start_symbol_download(
        self,
        symbol: str,
        batch_num: Optional[int] = None,
        task_num: Optional[int] = None,
        total_tasks: Optional[int] = None,
    ) -> None:
        """
        Mark a symbol as starting to download.
        
        Args:
            symbol: Symbol name
            batch_num: Batch number (optional)
            task_num: Task number within batch (optional)
            total_tasks: Total tasks in batch (optional)
        """
        if not self._progress:
            logger.warning("download_session_not_started", symbol=symbol)
            return
        
        if symbol not in self._progress.symbols:
            self._progress.symbols[symbol] = SymbolDownloadProgress(
                symbol=symbol,
                status=DownloadStatus.PENDING,
            )
        
        symbol_progress = self._progress.symbols[symbol]
        symbol_progress.status = DownloadStatus.DOWNLOADING
        symbol_progress.start_time = time.time()
        symbol_progress.batch_num = batch_num
        symbol_progress.task_num = task_num
        symbol_progress.total_tasks = total_tasks
        
        # Update counts
        if symbol_progress.status == DownloadStatus.PENDING:
            self._progress.pending = max(0, self._progress.pending - 1)
        self._progress.downloading += 1
        
        logger.info(
            "symbol_download_started",
            symbol=symbol,
            batch_num=batch_num,
            task_num=task_num,
            total_tasks=total_tasks,
        )
    
    def complete_symbol_download(
        self,
        symbol: str,
        rows_downloaded: int = 0,
    ) -> None:
        """
        Mark a symbol download as complete.
        
        Args:
            symbol: Symbol name
            rows_downloaded: Number of rows downloaded
        """
        if not self._progress:
            logger.warning("download_session_not_started", symbol=symbol)
            return
        
        if symbol not in self._progress.symbols:
            self._progress.symbols[symbol] = SymbolDownloadProgress(
                symbol=symbol,
                status=DownloadStatus.DOWNLOADING,
            )
        
        symbol_progress = self._progress.symbols[symbol]
        symbol_progress.status = DownloadStatus.COMPLETED
        symbol_progress.end_time = time.time()
        symbol_progress.rows_downloaded = rows_downloaded
        
        # Update counts
        self._progress.downloading = max(0, self._progress.downloading - 1)
        self._progress.completed += 1
        
        logger.info(
            "symbol_download_completed",
            symbol=symbol,
            rows_downloaded=rows_downloaded,
            duration_seconds=symbol_progress.end_time - symbol_progress.start_time if symbol_progress.start_time else None,
        )
    
    def fail_symbol_download(
        self,
        symbol: str,
        error: str,
    ) -> None:
        """
        Mark a symbol download as failed.
        
        Args:
            symbol: Symbol name
            error: Error message
        """
        if not self._progress:
            logger.warning("download_session_not_started", symbol=symbol)
            return
        
        if symbol not in self._progress.symbols:
            self._progress.symbols[symbol] = SymbolDownloadProgress(
                symbol=symbol,
                status=DownloadStatus.DOWNLOADING,
            )
        
        symbol_progress = self._progress.symbols[symbol]
        symbol_progress.status = DownloadStatus.FAILED
        symbol_progress.end_time = time.time()
        symbol_progress.error = error
        
        # Update counts
        self._progress.downloading = max(0, self._progress.downloading - 1)
        self._progress.failed += 1
        
        logger.info(
            "symbol_download_failed",
            symbol=symbol,
            error=error,
        )
    
    def get_progress(self) -> Optional[DownloadProgress]:
        """Get current download progress."""
        return self._progress
    
    def clear(self) -> None:
        """Clear progress (for testing or new session)."""
        self._progress = None
        logger.info("download_progress_cleared")


# Global progress tracker instance
_progress_tracker: Optional[DownloadProgressTracker] = None


def get_progress_tracker() -> DownloadProgressTracker:
    """Get the global download progress tracker instance."""
    global _progress_tracker
    if _progress_tracker is None:
        _progress_tracker = DownloadProgressTracker()
    return _progress_tracker





