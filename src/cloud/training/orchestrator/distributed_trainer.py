"""
Distributed Training Orchestrator

Coordinates distributed, asynchronous training of 400+ coins using Ray or Dask
on RunPod GPUs. Handles GPU allocation, job queuing, progress tracking, and
failure recovery.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)


class TrainingBackend(str, Enum):
    """Training backend options."""
    RAY = "ray"
    DASK = "dask"
    SEQUENTIAL = "sequential"  # Fallback for testing


@dataclass
class TrainingJob:
    """Individual training job specification."""
    job_id: str
    coin: str
    regime: str
    timeframe: str
    priority: int = 0  # Higher = more priority
    retry_count: int = 0
    max_retries: int = 3
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: str = "pending"  # pending, running, completed, failed
    gpu_id: Optional[int] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingResult:
    """Training job result."""
    job_id: str
    coin: str
    regime: str
    timeframe: str
    success: bool
    model_path: Optional[str] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    training_time_seconds: float = 0.0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DistributedTrainingConfig:
    """Configuration for distributed training."""
    backend: TrainingBackend = TrainingBackend.RAY
    num_workers: int = 8
    gpus_per_worker: int = 1
    max_concurrent_jobs: int = 16
    retry_attempts: int = 3
    timeout_seconds: int = 3600
    ray_address: Optional[str] = None  # Ray cluster address
    dask_scheduler_address: Optional[str] = None  # Dask scheduler address
    gpu_memory_gb: int = 24  # GPU memory per worker
    checkpoint_interval_seconds: int = 300


class DistributedTrainer:
    """
    Distributed training orchestrator for 400+ coins.
    
    Features:
    - Async job queue management
    - GPU allocation and cleanup
    - Progress tracking
    - Failure recovery with retries
    - Ray/Dask backend support
    """
    
    def __init__(
        self,
        config: DistributedTrainingConfig,
        model_storage_path: Path,
        brain_library: Optional[Any] = None,  # BrainLibrary type
    ) -> None:
        """
        Initialize distributed trainer.
        
        Args:
            config: Distributed training configuration
            model_storage_path: Path to store trained models
            brain_library: Optional Brain Library instance for model storage
        """
        self.config = config
        self.model_storage_path = Path(model_storage_path)
        self.brain_library = brain_library
        
        # Job management
        self.job_queue: List[TrainingJob] = []
        self.running_jobs: Dict[str, TrainingJob] = {}
        self.completed_jobs: Dict[str, TrainingResult] = {}
        self.failed_jobs: Dict[str, TrainingJob] = {}
        
        # GPU management
        self.available_gpus: Set[int] = set(range(config.num_workers * config.gpus_per_worker))
        self.gpu_allocations: Dict[int, str] = {}  # gpu_id -> job_id
        
        # Backend initialization
        self.backend_initialized = False
        self._initialize_backend()
        
        logger.info(
            "distributed_trainer_initialized",
            backend=config.backend.value,
            num_workers=config.num_workers,
            max_concurrent_jobs=config.max_concurrent_jobs,
        )
    
    def _initialize_backend(self) -> None:
        """Initialize distributed backend (Ray or Dask)."""
        if self.config.backend == TrainingBackend.RAY:
            self._initialize_ray()
        elif self.config.backend == TrainingBackend.DASK:
            self._initialize_dask()
        else:
            logger.info("using_sequential_backend", backend="sequential")
            self.backend_initialized = True
    
    def _initialize_ray(self) -> None:
        """Initialize Ray cluster."""
        try:
            import ray
            
            if not ray.is_initialized():
                init_kwargs = {
                    "ignore_reinit_error": True,
                    "log_to_driver": True,
                }
                
                if self.config.ray_address:
                    init_kwargs["address"] = self.config.ray_address
                else:
                    # Local Ray cluster
                    init_kwargs["num_cpus"] = self.config.num_workers
                    init_kwargs["num_gpus"] = self.config.num_workers * self.config.gpus_per_worker
                
                ray.init(**init_kwargs)
            
            self.backend_initialized = True
            logger.info("ray_initialized", address=self.config.ray_address or "local")
            
        except ImportError:
            logger.warning("ray_not_available", message="Ray not installed, falling back to sequential")
            self.config.backend = TrainingBackend.SEQUENTIAL
            self.backend_initialized = True
        except Exception as e:
            logger.error("ray_initialization_failed", error=str(e))
            self.config.backend = TrainingBackend.SEQUENTIAL
            self.backend_initialized = True
    
    def _initialize_dask(self) -> None:
        """Initialize Dask cluster."""
        try:
            from dask.distributed import Client
            
            if self.config.dask_scheduler_address:
                self.dask_client = Client(self.config.dask_scheduler_address)
            else:
                # Local Dask cluster
                self.dask_client = Client(
                    n_workers=self.config.num_workers,
                    threads_per_worker=1,
                )
            
            self.backend_initialized = True
            logger.info("dask_initialized", address=self.config.dask_scheduler_address or "local")
            
        except ImportError:
            logger.warning("dask_not_available", message="Dask not installed, falling back to sequential")
            self.config.backend = TrainingBackend.SEQUENTIAL
            self.backend_initialized = True
        except Exception as e:
            logger.error("dask_initialization_failed", error=str(e))
            self.config.backend = TrainingBackend.SEQUENTIAL
            self.backend_initialized = True
    
    def add_training_jobs(
        self,
        jobs: List[Tuple[str, str, str]],  # List of (coin, regime, timeframe)
        priority: int = 0,
    ) -> List[str]:
        """
        Add training jobs to the queue.
        
        Args:
            jobs: List of (coin, regime, timeframe) tuples
            priority: Priority level (higher = more priority)
        
        Returns:
            List of job IDs
        """
        job_ids = []
        
        for coin, regime, timeframe in jobs:
            job_id = f"{coin}_{regime}_{timeframe}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
            
            job = TrainingJob(
                job_id=job_id,
                coin=coin,
                regime=regime,
                timeframe=timeframe,
                priority=priority,
                max_retries=self.config.retry_attempts,
            )
            
            self.job_queue.append(job)
            job_ids.append(job_id)
        
        # Sort queue by priority (higher first)
        self.job_queue.sort(key=lambda j: j.priority, reverse=True)
        
        logger.info(
            "training_jobs_added",
            num_jobs=len(jobs),
            total_queue_size=len(self.job_queue),
        )
        
        return job_ids
    
    async def run_training_loop(self) -> Dict[str, TrainingResult]:
        """
        Run the main training loop.
        
        Processes jobs from the queue, allocates GPUs, and tracks progress.
        
        Returns:
            Dictionary of job_id -> TrainingResult
        """
        logger.info("training_loop_started", queue_size=len(self.job_queue))
        
        while self.job_queue or self.running_jobs:
            # Start new jobs if capacity available
            await self._start_pending_jobs()
            
            # Check running jobs for completion
            await self._check_running_jobs()
            
            # Retry failed jobs if retries available
            await self._retry_failed_jobs()
            
            # Wait a bit before next iteration
            await asyncio.sleep(1.0)
        
        logger.info(
            "training_loop_completed",
            total_completed=len(self.completed_jobs),
            total_failed=len(self.failed_jobs),
        )
        
        return self.completed_jobs
    
    async def _start_pending_jobs(self) -> None:
        """Start pending jobs if capacity is available."""
        while (
            self.job_queue
            and len(self.running_jobs) < self.config.max_concurrent_jobs
            and self.available_gpus
        ):
            job = self.job_queue.pop(0)
            
            # Allocate GPU
            gpu_id = self.available_gpus.pop()
            job.gpu_id = gpu_id
            job.status = "running"
            job.started_at = datetime.now(timezone.utc)
            self.gpu_allocations[gpu_id] = job.job_id
            self.running_jobs[job.job_id] = job
            
            # Start training task
            if self.config.backend == TrainingBackend.RAY:
                asyncio.create_task(self._run_ray_job(job))
            elif self.config.backend == TrainingBackend.DASK:
                asyncio.create_task(self._run_dask_job(job))
            else:
                asyncio.create_task(self._run_sequential_job(job))
            
            logger.info(
                "training_job_started",
                job_id=job.job_id,
                coin=job.coin,
                regime=job.regime,
                timeframe=job.timeframe,
                gpu_id=gpu_id,
            )
    
    async def _check_running_jobs(self) -> None:
        """Check running jobs for completion."""
        # In a real implementation, this would check task futures
        # For now, this is a placeholder that would be implemented
        # based on the actual backend (Ray/Dask) task tracking
        pass
    
    async def _retry_failed_jobs(self) -> None:
        """Retry failed jobs if retries are available."""
        retry_jobs = []
        
        for job_id, job in list(self.failed_jobs.items()):
            if job.retry_count < job.max_retries:
                job.retry_count += 1
                job.status = "pending"
                job.error = None
                retry_jobs.append(job)
                del self.failed_jobs[job_id]
        
        if retry_jobs:
            self.job_queue.extend(retry_jobs)
            self.job_queue.sort(key=lambda j: j.priority, reverse=True)
            logger.info("failed_jobs_retried", num_retries=len(retry_jobs))
    
    async def _run_ray_job(self, job: TrainingJob) -> None:
        """Run training job using Ray."""
        try:
            import ray
            
            # This would call the actual training function as a Ray remote
            # For now, this is a placeholder
            result = await self._execute_training(job)
            
            # Mark job as completed
            self._complete_job(job, result)
            
        except Exception as e:
            logger.error(
                "ray_job_failed",
                job_id=job.job_id,
                error=str(e),
            )
            self._fail_job(job, str(e))
    
    async def _run_dask_job(self, job: TrainingJob) -> None:
        """Run training job using Dask."""
        try:
            # This would submit the training function to Dask
            # For now, this is a placeholder
            result = await self._execute_training(job)
            
            # Mark job as completed
            self._complete_job(job, result)
            
        except Exception as e:
            logger.error(
                "dask_job_failed",
                job_id=job.job_id,
                error=str(e),
            )
            self._fail_job(job, str(e))
    
    async def _run_sequential_job(self, job: TrainingJob) -> None:
        """Run training job sequentially (fallback)."""
        try:
            result = await self._execute_training(job)
            self._complete_job(job, result)
            
        except Exception as e:
            logger.error(
                "sequential_job_failed",
                job_id=job.job_id,
                error=str(e),
            )
            self._fail_job(job, str(e))
    
    async def _execute_training(self, job: TrainingJob) -> TrainingResult:
        """
        Execute the actual training (placeholder - would call real training function).
        
        This is where the actual model training would happen. In a real implementation,
        this would:
        1. Load data for the coin/regime/timeframe
        2. Run walk-forward validation
        3. Train the model
        4. Evaluate metrics
        5. Save model to storage
        6. Store in Brain Library
        
        Args:
            job: Training job specification
        
        Returns:
            Training result
        """
        start_time = datetime.now(timezone.utc)
        
        # Placeholder: In real implementation, this would call the actual training pipeline
        # For now, simulate training time
        await asyncio.sleep(1.0)  # Simulate training
        
        training_time = (datetime.now(timezone.utc) - start_time).total_seconds()
        
        # Generate model path
        model_filename = f"{job.coin}_{job.regime}_{job.timeframe}_v1.pkl"
        model_path = str(self.model_storage_path / model_filename)
        
        # Placeholder metrics
        metrics = {
            "sharpe_ratio": 1.5,
            "max_drawdown": -0.05,
            "hit_rate": 0.55,
            "edge_after_cost_bps": 5.0,
        }
        
        result = TrainingResult(
            job_id=job.job_id,
            coin=job.coin,
            regime=job.regime,
            timeframe=job.timeframe,
            success=True,
            model_path=model_path,
            metrics=metrics,
            training_time_seconds=training_time,
        )
        
        # Store in Brain Library if available
        if self.brain_library:
            try:
                self.brain_library.register_model(
                    model_id=job.job_id,
                    symbol=job.coin,
                    model_type=f"{job.regime}_{job.timeframe}",
                    version=1,
                    composite_score=metrics.get("sharpe_ratio", 0.0),
                    hyperparameters={},
                    dataset_id=f"{job.coin}_{job.regime}",
                    feature_set=[],
                )
            except Exception as e:
                logger.warning(
                    "brain_library_store_failed",
                    job_id=job.job_id,
                    error=str(e),
                )
        
        return result
    
    def _complete_job(self, job: TrainingJob, result: TrainingResult) -> None:
        """Mark job as completed and free GPU."""
        job.status = "completed"
        job.completed_at = datetime.now(timezone.utc)
        
        # Free GPU
        if job.gpu_id is not None:
            self.available_gpus.add(job.gpu_id)
            if job.gpu_id in self.gpu_allocations:
                del self.gpu_allocations[job.gpu_id]
        
        # Move to completed
        del self.running_jobs[job.job_id]
        self.completed_jobs[job.job_id] = result
        
        logger.info(
            "training_job_completed",
            job_id=job.job_id,
            coin=job.coin,
            training_time_seconds=result.training_time_seconds,
            sharpe_ratio=result.metrics.get("sharpe_ratio", 0.0),
        )
    
    def _fail_job(self, job: TrainingJob, error: str) -> None:
        """Mark job as failed and free GPU."""
        job.status = "failed"
        job.error = error
        
        # Free GPU
        if job.gpu_id is not None:
            self.available_gpus.add(job.gpu_id)
            if job.gpu_id in self.gpu_allocations:
                del self.gpu_allocations[job.gpu_id]
        
        # Move to failed (will retry if retries available)
        del self.running_jobs[job.job_id]
        self.failed_jobs[job.job_id] = job
        
        logger.error(
            "training_job_failed",
            job_id=job.job_id,
            coin=job.coin,
            error=error,
            retry_count=job.retry_count,
            max_retries=job.max_retries,
        )
    
    def get_progress(self) -> Dict[str, Any]:
        """
        Get current training progress.
        
        Returns:
            Progress dictionary with queue size, running jobs, etc.
        """
        return {
            "queue_size": len(self.job_queue),
            "running_jobs": len(self.running_jobs),
            "completed_jobs": len(self.completed_jobs),
            "failed_jobs": len(self.failed_jobs),
            "available_gpus": len(self.available_gpus),
            "total_gpus": self.config.num_workers * self.config.gpus_per_worker,
        }
    
    def shutdown(self) -> None:
        """Shutdown distributed backend."""
        if self.config.backend == TrainingBackend.RAY:
            try:
                import ray
                if ray.is_initialized():
                    ray.shutdown()
            except Exception as e:
                logger.warning("ray_shutdown_failed", error=str(e))
        
        elif self.config.backend == TrainingBackend.DASK:
            try:
                if hasattr(self, 'dask_client'):
                    self.dask_client.close()
            except Exception as e:
                logger.warning("dask_shutdown_failed", error=str(e))
        
        logger.info("distributed_trainer_shutdown")

