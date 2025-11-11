"""
Hybrid Training Scheduler

Schedules coin training in parallel batches with safe I/O, partial outputs, and resumability.
"""

from __future__ import annotations

import os
import queue
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import structlog

from ..pipelines.work_item import WorkItem, WorkStatus, TrainResult
from ..utils.resume_ledger import ResumeLedger
from ..services.storage import StorageClient, create_storage_client
from ..services.telegram import TelegramService

logger = structlog.get_logger(__name__)


class TrainingMode(Enum):
    """Training mode."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    HYBRID = "hybrid"


@dataclass
class SchedulerConfig:
    """Scheduler configuration."""
    mode: TrainingMode = TrainingMode.HYBRID
    max_concurrent: int = 12
    timeout_minutes: int = 45
    force: bool = False
    driver: str = "dropbox"
    dry_run: bool = False
    storage_client: Optional[StorageClient] = None
    telegram_service: Optional[TelegramService] = None
    
    def __post_init__(self):
        """Set defaults based on GPU availability."""
        # Detect GPU and set defaults
        try:
            import torch
            has_gpu = torch.cuda.is_available()
        except ImportError:
            has_gpu = os.getenv("CUDA_VISIBLE_DEVICES") is not None
        
        if self.max_concurrent == 12 and not has_gpu:
            # CPU-only node, reduce concurrency
            self.max_concurrent = 2
            logger.info("gpu_not_available", message="Reducing max_concurrent to 2 for CPU-only node")
        
        if self.mode == TrainingMode.HYBRID and self.max_concurrent == 1:
            # Hybrid with max_concurrent=1 is essentially sequential
            self.mode = TrainingMode.SEQUENTIAL
            logger.info("mode_switched_to_sequential", reason="max_concurrent=1")


class HybridTrainingScheduler:
    """Hybrid training scheduler with parallel batches."""
    
    def __init__(
        self,
        config: SchedulerConfig,
        train_func: Callable[[str, Dict[str, Any]], TrainResult],
        resume_ledger: Optional[ResumeLedger] = None,
        app_config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize hybrid training scheduler.
        
        Args:
            config: Scheduler configuration
            train_func: Training function that takes symbol and config, returns TrainResult
            resume_ledger: Resume ledger for tracking status
            app_config: Application configuration dictionary (optional)
        """
        self.config = config
        self.train_func = train_func
        self.resume_ledger = resume_ledger or ResumeLedger()
        self._config = app_config  # Store app config for passing to train_func
        
        self.work_queue: queue.Queue[str] = queue.Queue()
        self.active_workers: Dict[str, WorkItem] = {}
        self.completed_results: List[TrainResult] = []
        self.lock = threading.Lock()
        
        logger.info("scheduler_initialized", mode=config.mode.value, max_concurrent=config.max_concurrent)
    
    def schedule_symbols(self, symbols: List[str]) -> List[TrainResult]:
        """Schedule symbols for training.
        
        Args:
            symbols: List of symbols to train
            
        Returns:
            List of training results
        """
        logger.info("scheduling_symbols", total_symbols=len(symbols), mode=self.config.mode.value)
        
        # Filter symbols based on resume ledger
        symbols_to_train = self._filter_symbols(symbols)
        
        if not symbols_to_train:
            logger.info("no_symbols_to_train", message="All symbols already completed")
            return self.completed_results
        
        # Send start notification
        if self.config.telegram_service:
            self.config.telegram_service.send_start_summary(len(symbols), self.config.mode.value)
        
        # Run based on mode
        if self.config.mode == TrainingMode.SEQUENTIAL:
            results = self._run_sequential(symbols_to_train)
        elif self.config.mode == TrainingMode.PARALLEL:
            results = self._run_parallel(symbols_to_train)
        elif self.config.mode == TrainingMode.HYBRID:
            results = self._run_hybrid(symbols_to_train)
        else:
            raise ValueError(f"Unknown training mode: {self.config.mode}")
        
        # Send completion notification
        if self.config.telegram_service:
            succeeded = sum(1 for r in results if r.status == "success")
            failed = sum(1 for r in results if r.status == "failed")
            skipped = sum(1 for r in results if r.status == "skipped")
            total_wall_minutes = sum(r.wall_minutes for r in results)
            
            self.config.telegram_service.send_completion_summary(
                total_symbols=len(symbols),
                succeeded=succeeded,
                failed=failed,
                skipped=skipped,
                total_wall_minutes=total_wall_minutes,
            )
        
        return results
    
    def _filter_symbols(self, symbols: List[str]) -> List[str]:
        """Filter symbols based on resume ledger.
        
        Args:
            symbols: List of symbols
            
        Returns:
            Filtered list of symbols to train
        """
        if self.config.force:
            return symbols
        
        symbols_to_train = []
        for symbol in symbols:
            if self.resume_ledger.is_completed(symbol):
                # Load existing result
                status = self.resume_ledger.get_symbol_status(symbol)
                if status:
                    result = TrainResult(
                        symbol=symbol,
                        status=status.get("status", "success"),
                        output_path=status.get("output_path", ""),
                        metrics_path=status.get("metrics_path"),
                    )
                    self.completed_results.append(result)
                    logger.info("symbol_skipped_completed", symbol=symbol)
                else:
                    symbols_to_train.append(symbol)
            else:
                symbols_to_train.append(symbol)
        
        return symbols_to_train
    
    def _run_sequential(self, symbols: List[str]) -> List[TrainResult]:
        """Run training sequentially.
        
        Args:
            symbols: List of symbols to train
            
        Returns:
            List of training results
        """
        results = []
        
        for symbol in symbols:
            work_item = WorkItem(symbol=symbol)
            result = self._train_symbol(work_item)
            results.append(result)
        
        return results
    
    def _run_parallel(self, symbols: List[str]) -> List[TrainResult]:
        """Run training in parallel using Ray or multiprocessing.
        
        Args:
            symbols: List of symbols to train
            
        Returns:
            List of training results
        """
        # Try Ray first
        try:
            import ray
            if ray.is_initialized():
                return self._run_parallel_ray(symbols)
        except ImportError:
            pass
        
        # Fall back to multiprocessing
        return self._run_parallel_multiprocessing(symbols)
    
    def _run_parallel_ray(self, symbols: List[str]) -> List[TrainResult]:
        """Run training in parallel using Ray.
        
        Args:
            symbols: List of symbols to train
            
        Returns:
            List of training results
        """
        import ray
        
        @ray.remote
        def train_symbol_ray(symbol: str, config_dict: Dict[str, Any]) -> TrainResult:
            """Ray remote function for training."""
            return self.train_func(symbol, config_dict)
        
        # Convert config to dict
        config_dict = {
            "timeout_minutes": self.config.timeout_minutes,
            "dry_run": self.config.dry_run,
        }
        
        # Submit all tasks
        futures = [train_symbol_ray.remote(symbol, config_dict) for symbol in symbols]
        
        # Wait for all tasks
        results = ray.get(futures)
        
        return list(results)
    
    def _run_parallel_multiprocessing(self, symbols: List[str]) -> List[TrainResult]:
        """Run training in parallel using multiprocessing.
        
        Args:
            symbols: List of symbols to train
            
        Returns:
            List of training results
        """
        from multiprocessing import Pool
        
        def train_symbol_mp(symbol: str) -> TrainResult:
            """Multiprocessing function for training."""
            config_dict = {
                "timeout_minutes": self.config.timeout_minutes,
                "dry_run": self.config.dry_run,
            }
            return self.train_func(symbol, config_dict)
        
        with Pool(processes=self.config.max_concurrent) as pool:
            results = pool.map(train_symbol_mp, symbols)
        
        return results
    
    def _run_hybrid(self, symbols: List[str]) -> List[TrainResult]:
        """Run training in hybrid mode (batched parallel).
        
        Args:
            symbols: List of symbols to train
            
        Returns:
            List of training results
        """
        # Add all symbols to queue
        for symbol in symbols:
            self.work_queue.put(symbol)
        
        # Start workers
        workers = []
        for _ in range(self.config.max_concurrent):
            worker = threading.Thread(target=self._worker, daemon=True)
            worker.start()
            workers.append(worker)
        
        # Wait for all work to complete
        self.work_queue.join()
        
        # Stop workers
        for _ in range(self.config.max_concurrent):
            self.work_queue.put(None)
        
        for worker in workers:
            worker.join()
        
        return self.completed_results
    
    def _worker(self) -> None:
        """Worker thread that processes symbols from queue."""
        while True:
            try:
                symbol = self.work_queue.get(timeout=1)
                if symbol is None:
                    break
                
                work_item = WorkItem(symbol=symbol)
                result = self._train_symbol(work_item)
                
                with self.lock:
                    self.completed_results.append(result)
                
                self.work_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error("worker_error", error=str(e))
                self.work_queue.task_done()
    
    def _train_symbol(self, work_item: WorkItem) -> TrainResult:
        """Train a single symbol with timeout and retries.
        
        Args:
            work_item: Work item for symbol
            
        Returns:
            Training result
        """
        symbol = work_item.symbol
        max_retries = 2
        
        # Retry loop
        for attempt in range(max_retries + 1):
            if attempt > 0:
                # Wait with jittered backoff
                import random
                backoff_seconds = (2 ** attempt) + random.uniform(0, 1)
                logger.info("retrying_symbol", symbol=symbol, attempt=attempt, backoff_seconds=backoff_seconds)
                time.sleep(backoff_seconds)
                
                work_item.retry()
            
            # Mark as running
            work_item.start()
            self.resume_ledger.mark_running(symbol)
            
            logger.info("coin_started", symbol=symbol, attempt=attempt + 1)
            
            try:
                # Train symbol with timeout
                config_dict = {
                    "timeout_minutes": self.config.timeout_minutes,
                    "dry_run": self.config.dry_run,
                    "storage_client": self.config.storage_client,
                    "config": getattr(self, "_config", None),  # Pass config if available
                }
                
                # Run with timeout
                import signal
                import threading
                
                result_container = {"result": None, "exception": None}
                
                def train_with_timeout():
                    try:
                        result_container["result"] = self.train_func(symbol, config_dict)
                    except Exception as e:
                        result_container["exception"] = e
                
                thread = threading.Thread(target=train_with_timeout, daemon=True)
                thread.start()
                thread.join(timeout=self.config.timeout_minutes * 60)
                
                if thread.is_alive():
                    # Timeout
                    error_msg = f"Training timeout after {self.config.timeout_minutes} minutes"
                    error_type = "TimeoutError"
                    
                    work_item.fail(error_msg, error_type)
                    self.resume_ledger.mark_failed(symbol, work_item.result or TrainResult(
                        symbol=symbol,
                        status="timeout",
                        error=error_msg,
                        error_type=error_type,
                        retry_count=attempt,
                    ))
                    
                    logger.error("coin_timeout", symbol=symbol, timeout_minutes=self.config.timeout_minutes)
                    
                    if attempt < max_retries:
                        continue
                    else:
                        return work_item.result or TrainResult(
                            symbol=symbol,
                            status="timeout",
                            error=error_msg,
                            error_type=error_type,
                            retry_count=attempt,
                        )
                
                if result_container["exception"]:
                    raise result_container["exception"]
                
                result = result_container["result"]
                if result is None:
                    raise ValueError("Training function returned None")
                
                # Mark as completed
                work_item.complete(result)
                result.retry_count = attempt
                
                if result.status == "success":
                    self.resume_ledger.mark_success(symbol, result)
                    logger.info("coin_succeeded", symbol=symbol, wall_minutes=result.wall_minutes, attempt=attempt + 1)
                    return result
                else:
                    # Failed, but check if we should retry
                    if attempt < max_retries and work_item.should_retry():
                        self.resume_ledger.mark_failed(symbol, result)
                        logger.warning("coin_failed_retrying", symbol=symbol, error=result.error, attempt=attempt + 1)
                        continue
                    else:
                        self.resume_ledger.mark_failed(symbol, result)
                        logger.error("coin_failed", symbol=symbol, error=result.error, attempt=attempt + 1)
                        
                        # Send failure alert
                        if self.config.telegram_service and result.status == "failed":
                            self.config.telegram_service.send_coin_failure_alert(
                                symbol=symbol,
                                error=result.error or "Unknown error",
                                error_type=result.error_type,
                            )
                        
                        return result
                
            except Exception as e:
                # Mark as failed
                error_msg = str(e)
                error_type = type(e).__name__
                
                work_item.fail(error_msg, error_type)
                result = TrainResult(
                    symbol=symbol,
                    status="failed",
                    error=error_msg,
                    error_type=error_type,
                    retry_count=attempt,
                )
                
                logger.error("coin_training_exception", symbol=symbol, error=error_msg, error_type=error_type, attempt=attempt + 1)
                
                # Check if we should retry
                if attempt < max_retries:
                    continue
                else:
                    self.resume_ledger.mark_failed(symbol, result)
                    
                    # Send failure alert
                    if self.config.telegram_service:
                        self.config.telegram_service.send_coin_failure_alert(symbol, error_msg, error_type)
                    
                    return result
        
        # Should not reach here
        return work_item.result or TrainResult(
            symbol=symbol,
            status="failed",
            error="Max retries exceeded",
            error_type="MaxRetriesExceeded",
        )

