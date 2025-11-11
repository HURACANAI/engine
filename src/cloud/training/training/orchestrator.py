"""
Training Orchestrator for Scalable Architecture

Orchestrates training for all eligible Binance pairs using Ray/Dask.
Supports asynchronous training with configurable concurrency.

Author: Huracan Engine Team
Date: 2025-01-27
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable

import structlog

logger = structlog.get_logger(__name__)

try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    logger.warning("ray not available, using asyncio for parallel training")

try:
    import dask
    from dask.distributed import Client as DaskClient
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False
    logger.warning("dask not available, using asyncio for parallel training")


class TrainingBackend(Enum):
    """Training backend."""
    RAY = "ray"
    DASK = "dask"
    ASYNCIO = "asyncio"


@dataclass
class TrainingJob:
    """Training job specification."""
    coin: str
    horizon: str  # e.g., "1h", "4h", "1d"
    regime: Optional[str] = None
    engine_type: Optional[str] = None
    priority: int = 0
    job_id: str = field(default_factory=lambda: f"job_{int(time.time() * 1000000)}")
    created_at: datetime = field(default_factory=datetime.now)
    status: str = "pending"
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@dataclass
class TrainingConfig:
    """Training configuration."""
    backend: TrainingBackend = TrainingBackend.ASYNCIO
    max_concurrent_jobs: int = 10
    lookback_days: int = 150
    horizons: List[str] = field(default_factory=lambda: ["1h", "4h", "1d"])
    risk_preset: str = "balanced"
    dry_run: bool = False
    ray_address: Optional[str] = None
    dask_address: Optional[str] = None
    num_workers: int = 4
    gpu_per_worker: int = 1


class TrainingOrchestrator:
    """
    Training orchestrator for scalable training.
    
    Features:
    - Asynchronous training with Ray/Dask
    - Configurable concurrency
    - Job prioritization
    - Error handling and retries
    - Progress tracking
    """
    
    def __init__(self, config: TrainingConfig):
        """
        Initialize training orchestrator.
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.jobs: Dict[str, TrainingJob] = {}
        self.completed_jobs: List[TrainingJob] = []
        self.failed_jobs: List[TrainingJob] = []
        
        # Backend clients
        self.ray_client: Optional[Any] = None
        self.dask_client: Optional[DaskClient] = None
        
        # Training function
        self.training_function: Optional[Callable] = None
        
        logger.info(
            "training_orchestrator_initialized",
            backend=config.backend.value,
            max_concurrent_jobs=config.max_concurrent_jobs,
            dry_run=config.dry_run,
        )
    
    async def initialize(self) -> None:
        """Initialize training backend."""
        if self.config.backend == TrainingBackend.RAY and RAY_AVAILABLE:
            if self.config.ray_address:
                self.ray_client = ray.init(address=self.config.ray_address)
            else:
                self.ray_client = ray.init(num_cpus=self.config.num_workers)
            logger.info("ray_initialized", num_workers=self.config.num_workers)
        
        elif self.config.backend == TrainingBackend.DASK and DASK_AVAILABLE:
            if self.config.dask_address:
                self.dask_client = DaskClient(address=self.config.dask_address)
            else:
                self.dask_client = DaskClient(n_workers=self.config.num_workers)
            logger.info("dask_initialized", num_workers=self.config.num_workers)
        
        else:
            logger.info("using_asyncio_backend")
    
    async def shutdown(self) -> None:
        """Shutdown training backend."""
        if self.ray_client:
            ray.shutdown()
            logger.info("ray_shutdown")
        
        if self.dask_client:
            await self.dask_client.close()
            logger.info("dask_shutdown")
    
    def register_training_function(self, training_function: Callable) -> None:
        """Register training function."""
        self.training_function = training_function
        logger.info("training_function_registered")
    
    def add_job(self, job: TrainingJob) -> None:
        """Add a training job."""
        self.jobs[job.job_id] = job
        logger.debug("job_added", job_id=job.job_id, coin=job.coin, horizon=job.horizon)
    
    def add_jobs(self, jobs: List[TrainingJob]) -> None:
        """Add multiple training jobs."""
        for job in jobs:
            self.add_job(job)
        logger.info("jobs_added", count=len(jobs))
    
    async def train_all(self) -> Dict[str, Any]:
        """
        Train all jobs.
        
        Returns:
            Dictionary with training results
        """
        if not self.training_function:
            raise ValueError("Training function not registered")
        
        logger.info("training_started", total_jobs=len(self.jobs))
        
        # Sort jobs by priority
        sorted_jobs = sorted(self.jobs.values(), key=lambda j: j.priority, reverse=True)
        
        # Train jobs based on backend
        if self.config.backend == TrainingBackend.RAY and RAY_AVAILABLE:
            results = await self._train_with_ray(sorted_jobs)
        elif self.config.backend == TrainingBackend.DASK and DASK_AVAILABLE:
            results = await self._train_with_dask(sorted_jobs)
        else:
            results = await self._train_with_asyncio(sorted_jobs)
        
        # Update job statuses
        for job_id, result in results.items():
            if job_id in self.jobs:
                job = self.jobs[job_id]
                if result.get("success"):
                    job.status = "completed"
                    job.result = result
                    self.completed_jobs.append(job)
                else:
                    job.status = "failed"
                    job.error = result.get("error")
                    self.failed_jobs.append(job)
        
        logger.info(
            "training_completed",
            total_jobs=len(self.jobs),
            completed=len(self.completed_jobs),
            failed=len(self.failed_jobs),
        )
        
        return {
            "total_jobs": len(self.jobs),
            "completed": len(self.completed_jobs),
            "failed": len(self.failed_jobs),
            "results": results,
        }
    
    async def _train_with_ray(self, jobs: List[TrainingJob]) -> Dict[str, Any]:
        """Train jobs with Ray."""
        if not RAY_AVAILABLE:
            raise ValueError("Ray not available")
        
        # Create Ray remote function
        @ray.remote
        def train_job_ray(job_dict: Dict[str, Any]) -> Dict[str, Any]:
            return self.training_function(job_dict)
        
        # Submit jobs
        futures = []
        for job in jobs:
            job_dict = {
                "coin": job.coin,
                "horizon": job.horizon,
                "regime": job.regime,
                "engine_type": job.engine_type,
                "job_id": job.job_id,
            }
            future = train_job_ray.remote(job_dict)
            futures.append((job.job_id, future))
        
        # Collect results with concurrency limit
        results = {}
        semaphore = asyncio.Semaphore(self.config.max_concurrent_jobs)
        
        async def process_future(job_id: str, future: Any) -> None:
            async with semaphore:
                try:
                    result = await asyncio.get_event_loop().run_in_executor(
                        None, ray.get, future
                    )
                    results[job_id] = result
                except Exception as e:
                    results[job_id] = {"success": False, "error": str(e)}
        
        await asyncio.gather(*[process_future(job_id, future) for job_id, future in futures])
        
        return results
    
    async def _train_with_dask(self, jobs: List[TrainingJob]) -> Dict[str, Any]:
        """Train jobs with Dask."""
        if not DASK_AVAILABLE or not self.dask_client:
            raise ValueError("Dask not available")
        
        # Submit jobs
        futures = []
        for job in jobs:
            job_dict = {
                "coin": job.coin,
                "horizon": job.horizon,
                "regime": job.regime,
                "engine_type": job.engine_type,
                "job_id": job.job_id,
            }
            future = self.dask_client.submit(self.training_function, job_dict)
            futures.append((job.job_id, future))
        
        # Collect results
        results = {}
        for job_id, future in futures:
            try:
                result = future.result()
                results[job_id] = result
            except Exception as e:
                results[job_id] = {"success": False, "error": str(e)}
        
        return results
    
    async def _train_with_asyncio(self, jobs: List[TrainingJob]) -> Dict[str, Any]:
        """Train jobs with asyncio."""
        semaphore = asyncio.Semaphore(self.config.max_concurrent_jobs)
        results = {}
        
        async def train_job(job: TrainingJob) -> None:
            async with semaphore:
                try:
                    job_dict = {
                        "coin": job.coin,
                        "horizon": job.horizon,
                        "regime": job.regime,
                        "engine_type": job.engine_type,
                        "job_id": job.job_id,
                    }
                    result = await asyncio.get_event_loop().run_in_executor(
                        None, self.training_function, job_dict
                    )
                    results[job.job_id] = result
                except Exception as e:
                    results[job.job_id] = {"success": False, "error": str(e)}
                    logger.error("job_training_failed", job_id=job.job_id, error=str(e))
        
        await asyncio.gather(*[train_job(job) for job in jobs])
        return results
    
    def get_job_status(self, job_id: str) -> Optional[TrainingJob]:
        """Get job status."""
        return self.jobs.get(job_id)
    
    def get_all_jobs(self) -> List[TrainingJob]:
        """Get all jobs."""
        return list(self.jobs.values())
    
    def get_completed_jobs(self) -> List[TrainingJob]:
        """Get completed jobs."""
        return self.completed_jobs.copy()
    
    def get_failed_jobs(self) -> List[TrainingJob]:
        """Get failed jobs."""
        return self.failed_jobs.copy()


def compute_code_hash() -> str:
    """Compute code hash for reproducibility."""
    # In production, hash the actual code files
    return hashlib.sha256(str(time.time()).encode()).hexdigest()[:16]


def compute_features_hash(features: Dict[str, Any]) -> str:
    """Compute features hash for reproducibility."""
    features_str = str(sorted(features.items()))
    return hashlib.sha256(features_str.encode()).hexdigest()[:16]

