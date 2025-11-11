"""
Job Queue for Per-Coin Training

Builds a job queue that loops symbols and runs train → validate → export.
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import structlog  # type: ignore[import-untyped]

logger = structlog.get_logger(__name__)


class JobStatus(Enum):
    """Job status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class TrainingJob:
    """Training job for a single symbol."""
    symbol: str
    status: JobStatus = JobStatus.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    skip_reason: Optional[str] = None


class JobQueue:
    """Job queue for per-coin training."""
    
    def __init__(
        self,
        symbols: List[str],
        train_func: Callable[[str], Dict[str, Any]],
        max_workers: int = 8,
    ):
        """Initialize job queue.
        
        Args:
            symbols: List of symbols to train
            train_func: Training function that takes symbol and returns result dict
            max_workers: Maximum number of parallel workers
        """
        self.symbols = symbols
        self.train_func = train_func
        self.max_workers = max_workers
        
        self.jobs: Dict[str, TrainingJob] = {
            symbol: TrainingJob(symbol=symbol) for symbol in symbols
        }
        
        self.job_queue = queue.Queue()
        self.results: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()
        
        logger.info("job_queue_initialized", total_symbols=len(symbols), max_workers=max_workers)
    
    def add_job(self, symbol: str) -> None:
        """Add job to queue.
        
        Args:
            symbol: Trading symbol
        """
        if symbol in self.jobs:
            self.job_queue.put(symbol)
    
    def process_job(self, symbol: str) -> Dict[str, Any]:
        """Process a single training job.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Result dictionary
        """
        job = self.jobs[symbol]
        job.status = JobStatus.RUNNING
        job.started_at = datetime.now(timezone.utc)
        
        logger.info("job_started", symbol=symbol)
        
        try:
            # Run training function
            result = self.train_func(symbol)
            
            job.status = JobStatus.COMPLETED
            job.completed_at = datetime.now(timezone.utc)
            job.result = result
            
            with self.lock:
                self.results[symbol] = result
            
            logger.info("job_completed", symbol=symbol, duration_seconds=(job.completed_at - job.started_at).total_seconds())
            return result
            
        except Exception as e:
            job.status = JobStatus.FAILED
            job.completed_at = datetime.now(timezone.utc)
            job.error = str(e)
            
            logger.error("job_failed", symbol=symbol, error=str(e))
            raise
    
    def skip_job(self, symbol: str, reason: str) -> None:
        """Skip a job with reason.
        
        Args:
            symbol: Trading symbol
            reason: Skip reason
        """
        job = self.jobs[symbol]
        job.status = JobStatus.SKIPPED
        job.skip_reason = reason
        job.completed_at = datetime.now(timezone.utc)
        
        logger.info("job_skipped", symbol=symbol, reason=reason)
    
    def worker(self) -> None:
        """Worker thread that processes jobs."""
        while True:
            try:
                symbol = self.job_queue.get(timeout=1)
                if symbol is None:
                    break
                
                self.process_job(symbol)
                self.job_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error("worker_error", error=str(e))
                self.job_queue.task_done()
    
    def run(self, skip_symbols: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        """Run all jobs in parallel.
        
        Args:
            skip_symbols: List of symbols to skip
            
        Returns:
            Dictionary mapping symbol to result
        """
        # Skip symbols if provided
        if skip_symbols:
            for symbol in skip_symbols:
                self.skip_job(symbol, "user_skipped")
        
        # Add all jobs to queue
        for symbol in self.symbols:
            if symbol not in (skip_symbols or []):
                self.add_job(symbol)
        
        # Start workers
        workers = []
        for _ in range(self.max_workers):
            worker = threading.Thread(target=self.worker, daemon=True)
            worker.start()
            workers.append(worker)
        
        # Wait for all jobs to complete
        self.job_queue.join()
        
        # Stop workers
        for _ in range(self.max_workers):
            self.job_queue.put(None)
        
        for worker in workers:
            worker.join()
        
        # Return results
        return self.results
    
    def get_status(self) -> Dict[str, Any]:
        """Get current queue status.
        
        Returns:
            Status dictionary
        """
        with self.lock:
            status = {
                "total": len(self.jobs),
                "pending": sum(1 for j in self.jobs.values() if j.status == JobStatus.PENDING),
                "running": sum(1 for j in self.jobs.values() if j.status == JobStatus.RUNNING),
                "completed": sum(1 for j in self.jobs.values() if j.status == JobStatus.COMPLETED),
                "failed": sum(1 for j in self.jobs.values() if j.status == JobStatus.FAILED),
                "skipped": sum(1 for j in self.jobs.values() if j.status == JobStatus.SKIPPED),
            }
        
        return status

