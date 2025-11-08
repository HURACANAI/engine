"""
Continuous Learning System

Hourly retrain loop with model versioning.
Rebuilds models as new data comes in.

Key Features:
- Hourly retrain loop
- Model versioning
- Automatic retraining triggers
- Performance-based retraining
- Model rollback on degradation
- Integration with Log Book

Author: Huracan Engine Team
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
from enum import Enum
from datetime import datetime, timezone, timedelta
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)


class RetrainTrigger(Enum):
    """Retrain trigger"""
    SCHEDULED = "scheduled"  # Hourly schedule
    PERFORMANCE_DEGRADATION = "performance_degradation"
    DATA_DRIFT = "data_drift"
    MANUAL = "manual"
    NEW_DATA = "new_data"


@dataclass
class ModelVersion:
    """Model version"""
    version_id: str
    model_id: str
    created_at: datetime
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    is_current: bool = False
    model_path: Optional[Path] = None
    metadata: Dict[str, any] = field(default_factory=dict)


@dataclass
class RetrainResult:
    """Retrain result"""
    retrain_id: str
    model_id: str
    trigger: RetrainTrigger
    success: bool
    new_version: Optional[ModelVersion] = None
    previous_version: Optional[ModelVersion] = None
    performance_improvement: float = 0.0
    rolled_back: bool = False
    error: Optional[str] = None
    duration_seconds: float = 0.0


class ContinuousLearningSystem:
    """
    Continuous Learning System.
    
    Hourly retrain loop with model versioning.
    Automatically retrains models as new data comes in.
    
    Usage:
        system = ContinuousLearningSystem(
            retrain_interval_hours=1,
            min_performance_improvement=0.05
        )
        
        # Register model
        system.register_model(
            model_id="model_1",
            train_fn=my_train_function,
            data_loader=my_data_loader
        )
        
        # Start continuous learning
        asyncio.run(system.start())
    """
    
    def __init__(
        self,
        retrain_interval_hours: float = 1.0,
        min_performance_improvement: float = 0.05,  # 5% improvement required
        performance_degradation_threshold: float = 0.10,  # 10% degradation triggers retrain
        model_storage_path: Optional[Path] = None,
        max_versions_per_model: int = 10
    ):
        """
        Initialize continuous learning system.
        
        Args:
            retrain_interval_hours: Retrain interval in hours
            min_performance_improvement: Minimum performance improvement to keep new model
            performance_degradation_threshold: Performance degradation threshold
            model_storage_path: Path to store model versions
            max_versions_per_model: Maximum versions to keep per model
        """
        self.retrain_interval_hours = retrain_interval_hours
        self.min_performance_improvement = min_performance_improvement
        self.performance_degradation_threshold = performance_degradation_threshold
        self.model_storage_path = model_storage_path or Path("models/versions")
        self.model_storage_path.mkdir(parents=True, exist_ok=True)
        self.max_versions_per_model = max_versions_per_model
        
        # Model registry
        self.models: Dict[str, Dict[str, any]] = {}  # model_id -> {train_fn, data_loader, ...}
        self.model_versions: Dict[str, List[ModelVersion]] = {}  # model_id -> [versions]
        self.current_versions: Dict[str, ModelVersion] = {}  # model_id -> current_version
        
        # Retrain tracking
        self.retrain_history: List[RetrainResult] = []
        self.running = False
        
        logger.info(
            "continuous_learning_system_initialized",
            retrain_interval_hours=retrain_interval_hours,
            min_performance_improvement=min_performance_improvement
        )
    
    def register_model(
        self,
        model_id: str,
        train_fn: Callable[[Any, Any], Any],  # (data, config) -> model
        data_loader: Callable[[], Any],  # () -> data
        evaluation_fn: Optional[Callable[[Any], Dict[str, float]]] = None,  # model -> metrics
        config: Optional[Dict[str, any]] = None
    ) -> None:
        """
        Register model for continuous learning.
        
        Args:
            model_id: Model identifier
            train_fn: Training function
            data_loader: Data loader function
            evaluation_fn: Evaluation function (optional)
            config: Model configuration (optional)
        """
        self.models[model_id] = {
            "train_fn": train_fn,
            "data_loader": data_loader,
            "evaluation_fn": evaluation_fn,
            "config": config or {},
            "registered_at": datetime.now(timezone.utc)
        }
        
        self.model_versions[model_id] = []
        self.current_versions[model_id] = None
        
        logger.info(
            "model_registered",
            model_id=model_id,
            retrain_interval_hours=self.retrain_interval_hours
        )
    
    async def retrain_model(
        self,
        model_id: str,
        trigger: RetrainTrigger = RetrainTrigger.SCHEDULED
    ) -> RetrainResult:
        """
        Retrain a model.
        
        Args:
            model_id: Model identifier
            trigger: Retrain trigger
        
        Returns:
            RetrainResult
        """
        import uuid
        
        start_time = time.time()
        retrain_id = str(uuid.uuid4())
        
        if model_id not in self.models:
            return RetrainResult(
                retrain_id=retrain_id,
                model_id=model_id,
                trigger=trigger,
                success=False,
                error=f"Model not registered: {model_id}",
                duration_seconds=0.0
            )
        
        logger.info(
            "model_retrain_started",
            retrain_id=retrain_id,
            model_id=model_id,
            trigger=trigger.value
        )
        
        try:
            # Get current version
            current_version = self.current_versions.get(model_id)
            
            # Load data
            model_info = self.models[model_id]
            data = await self._load_data_async(model_info["data_loader"])
            
            # Train new model
            new_model = await self._train_model_async(
                model_info["train_fn"],
                data,
                model_info["config"]
            )
            
            # Evaluate new model
            if model_info["evaluation_fn"]:
                new_metrics = await self._evaluate_model_async(
                    model_info["evaluation_fn"],
                    new_model
                )
            else:
                new_metrics = {}
            
            # Compare with current version
            performance_improvement = 0.0
            if current_version and current_version.performance_metrics:
                current_sharpe = current_version.performance_metrics.get("sharpe_ratio", 0.0)
                new_sharpe = new_metrics.get("sharpe_ratio", 0.0)
                
                if current_sharpe > 0:
                    performance_improvement = (new_sharpe - current_sharpe) / current_sharpe
                else:
                    performance_improvement = 1.0 if new_sharpe > 0 else 0.0
            
            # Create new version
            version_id = str(uuid.uuid4())
            new_version = ModelVersion(
                version_id=version_id,
                model_id=model_id,
                created_at=datetime.now(timezone.utc),
                performance_metrics=new_metrics,
                is_current=False,
                model_path=self._save_model_version(model_id, version_id, new_model),
                metadata={
                    "trigger": trigger.value,
                    "retrain_id": retrain_id
                }
            )
            
            # Decide if to keep new version
            rolled_back = False
            if performance_improvement >= self.min_performance_improvement:
                # Keep new version
                if current_version:
                    current_version.is_current = False
                new_version.is_current = True
                self.current_versions[model_id] = new_version
            else:
                # Roll back: keep current version
                rolled_back = True
                if current_version:
                    current_version.is_current = True
                logger.warning(
                    "model_retrain_rolled_back",
                    model_id=model_id,
                    performance_improvement=performance_improvement,
                    min_required=self.min_performance_improvement
                )
            
            # Store version
            self.model_versions[model_id].append(new_version)
            
            # Cleanup old versions
            self._cleanup_old_versions(model_id)
            
            duration = time.time() - start_time
            
            result = RetrainResult(
                retrain_id=retrain_id,
                model_id=model_id,
                trigger=trigger,
                success=True,
                new_version=new_version,
                previous_version=current_version,
                performance_improvement=performance_improvement,
                rolled_back=rolled_back,
                duration_seconds=duration
            )
            
            self.retrain_history.append(result)
            
            logger.info(
                "model_retrain_complete",
                retrain_id=retrain_id,
                model_id=model_id,
                performance_improvement=performance_improvement,
                rolled_back=rolled_back,
                duration_seconds=duration
            )
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            
            result = RetrainResult(
                retrain_id=retrain_id,
                model_id=model_id,
                trigger=trigger,
                success=False,
                error=str(e),
                duration_seconds=duration
            )
            
            self.retrain_history.append(result)
            
            logger.error(
                "model_retrain_failed",
                retrain_id=retrain_id,
                model_id=model_id,
                error=str(e)
            )
            
            return result
    
    async def _load_data_async(self, data_loader: Callable) -> Any:
        """Load data asynchronously"""
        # If data loader is async, await it
        if asyncio.iscoroutinefunction(data_loader):
            return await data_loader()
        else:
            # Run in executor for sync functions
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, data_loader)
    
    async def _train_model_async(
        self,
        train_fn: Callable,
        data: Any,
        config: Dict[str, any]
    ) -> Any:
        """Train model asynchronously"""
        # If train function is async, await it
        if asyncio.iscoroutinefunction(train_fn):
            return await train_fn(data, config)
        else:
            # Run in executor for sync functions
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, train_fn, data, config)
    
    async def _evaluate_model_async(
        self,
        evaluation_fn: Callable,
        model: Any
    ) -> Dict[str, float]:
        """Evaluate model asynchronously"""
        # If evaluation function is async, await it
        if asyncio.iscoroutinefunction(evaluation_fn):
            return await evaluation_fn(model)
        else:
            # Run in executor for sync functions
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, evaluation_fn, model)
    
    def _save_model_version(
        self,
        model_id: str,
        version_id: str,
        model: Any
    ) -> Path:
        """Save model version to disk"""
        model_dir = self.model_storage_path / model_id
        model_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = model_dir / f"{version_id}.pkl"
        
        # Save model (simplified - would use actual model serialization)
        # In production, would use pickle, joblib, or model-specific serialization
        import pickle
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        return model_path
    
    def _cleanup_old_versions(self, model_id: str) -> None:
        """Cleanup old model versions"""
        if model_id not in self.model_versions:
            return
        
        versions = self.model_versions[model_id]
        
        # Keep only recent versions
        if len(versions) > self.max_versions_per_model:
            # Sort by creation time (newest first)
            versions.sort(key=lambda v: v.created_at, reverse=True)
            
            # Remove old versions (keep current and recent)
            versions_to_remove = versions[self.max_versions_per_model:]
            for version in versions_to_remove:
                if version.model_path and version.model_path.exists():
                    version.model_path.unlink()
            
            # Update versions list
            self.model_versions[model_id] = versions[:self.max_versions_per_model]
    
    async def retrain_loop(self) -> None:
        """Continuous retrain loop"""
        logger.info("continuous_retrain_loop_started")
        
        while self.running:
            try:
                # Wait for retrain interval
                await asyncio.sleep(self.retrain_interval_hours * 3600)
                
                # Retrain all registered models
                for model_id in self.models.keys():
                    await self.retrain_model(model_id, RetrainTrigger.SCHEDULED)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("retrain_loop_error", error=str(e))
        
        logger.info("continuous_retrain_loop_stopped")
    
    async def start(self) -> None:
        """Start continuous learning system"""
        self.running = True
        
        # Start retrain loop
        retrain_task = asyncio.create_task(self.retrain_loop())
        
        logger.info("continuous_learning_system_started")
        
        # Wait for retrain loop (runs until stopped)
        await retrain_task
    
    async def stop(self) -> None:
        """Stop continuous learning system"""
        self.running = False
        
        # Cancel retrain loop
        tasks = [t for t in asyncio.all_tasks() if not t.done()]
        for task in tasks:
            task.cancel()
        
        await asyncio.gather(*tasks, return_exceptions=True)
        
        logger.info("continuous_learning_system_stopped")
    
    def check_performance_degradation(self, model_id: str) -> bool:
        """Check if model performance has degraded"""
        if model_id not in self.current_versions:
            return False
        
        current_version = self.current_versions[model_id]
        if not current_version or not current_version.performance_metrics:
            return False
        
        # Get recent performance (would compare with historical)
        # Simplified: check if current performance is below threshold
        current_sharpe = current_version.performance_metrics.get("sharpe_ratio", 0.0)
        
        # Get baseline performance (from previous versions)
        versions = self.model_versions.get(model_id, [])
        if len(versions) < 2:
            return False
        
        # Compare with previous version
        previous_version = versions[-2] if len(versions) >= 2 else None
        if previous_version and previous_version.performance_metrics:
            previous_sharpe = previous_version.performance_metrics.get("sharpe_ratio", 0.0)
            
            if previous_sharpe > 0:
                degradation = (previous_sharpe - current_sharpe) / previous_sharpe
                return degradation >= self.performance_degradation_threshold
        
        return False
    
    def get_model_versions(self, model_id: str) -> List[ModelVersion]:
        """Get all versions of a model"""
        return self.model_versions.get(model_id, [])
    
    def get_current_version(self, model_id: str) -> Optional[ModelVersion]:
        """Get current version of a model"""
        return self.current_versions.get(model_id)
    
    def get_retrain_history(self, model_id: Optional[str] = None) -> List[RetrainResult]:
        """Get retrain history"""
        if model_id:
            return [r for r in self.retrain_history if r.model_id == model_id]
        return self.retrain_history

