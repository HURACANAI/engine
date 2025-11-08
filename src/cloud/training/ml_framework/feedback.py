"""
Feedback Loop: Performance Tracking and Auto-Tuning

Tracks model performance and automatically fine-tunes underperforming models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import structlog

from .base import BaseModel, ModelMetrics

logger = structlog.get_logger(__name__)


@dataclass
class FeedbackConfig:
    """Configuration for feedback loop."""
    
    # Performance thresholds
    sharpe_threshold: float = 0.5  # Minimum Sharpe ratio
    rmse_threshold: float = 100.0  # Maximum RMSE
    win_rate_threshold: float = 0.45  # Minimum win rate
    
    # Auto-tuning
    auto_retrain_enabled: bool = True
    auto_reweight_enabled: bool = True
    auto_prune_enabled: bool = True
    
    # Retraining
    retrain_threshold_days: int = 7  # Retrain if performance drops for N days
    min_samples_for_retrain: int = 100  # Minimum samples needed for retrain
    
    # Pruning
    prune_threshold_days: int = 14  # Prune if underperforming for N days
    min_models_to_keep: int = 3  # Minimum number of models to keep
    
    # Database
    db_path: Optional[str] = None  # Path to database (PostgreSQL DSN or SQLite path)
    use_s3: bool = False
    s3_bucket: Optional[str] = None


@dataclass
class ModelPerformanceRecord:
    """Record of model performance at a specific time."""
    
    model_name: str
    timestamp: datetime
    metrics: ModelMetrics
    sample_count: int
    status: str = "active"  # "active", "underperforming", "pruned", "retraining"


class ModelFeedback:
    """
    Feedback loop for tracking model performance and auto-tuning.
    
    Features:
    - Track model performance over time
    - Detect underperforming models
    - Automatically retrain or reweight models
    - Prune consistently poor performers
    - Store metrics in database (PostgreSQL or SQLite)
    """
    
    def __init__(self, config: FeedbackConfig):
        """
        Initialize feedback system.
        
        Args:
            config: Feedback configuration
        """
        self.config = config
        self.performance_records: Dict[str, List[ModelPerformanceRecord]] = {}
        self.model_status: Dict[str, str] = {}  # "active", "underperforming", "pruned", "retraining"
        self.retrain_queue: List[str] = []
        self.prune_candidates: List[str] = []
        
        logger.info(
            "model_feedback_initialized",
            auto_retrain=config.auto_retrain_enabled,
            auto_reweight=config.auto_reweight_enabled,
            auto_prune=config.auto_prune_enabled,
        )
    
    def record_performance(
        self,
        model_name: str,
        metrics: ModelMetrics,
        sample_count: int = 1,
    ) -> None:
        """
        Record model performance.
        
        Args:
            model_name: Name of the model
            metrics: Performance metrics
            sample_count: Number of samples this metrics is based on
        """
        if model_name not in self.performance_records:
            self.performance_records[model_name] = []
            self.model_status[model_name] = "active"
        
        record = ModelPerformanceRecord(
            model_name=model_name,
            timestamp=datetime.now(),
            metrics=metrics,
            sample_count=sample_count,
            status=self.model_status.get(model_name, "active"),
        )
        
        self.performance_records[model_name].append(record)
        
        # Check if model is underperforming
        if self._is_underperforming(metrics):
            self.model_status[model_name] = "underperforming"
            record.status = "underperforming"
            
            logger.warning(
                "model_underperforming",
                model_name=model_name,
                sharpe=metrics.sharpe_ratio,
                rmse=metrics.rmse,
                win_rate=metrics.win_rate,
            )
            
            # Add to retrain queue if enabled
            if self.config.auto_retrain_enabled and model_name not in self.retrain_queue:
                if self._should_retrain(model_name):
                    self.retrain_queue.append(model_name)
                    logger.info("model_added_to_retrain_queue", model_name=model_name)
        else:
            # Model is performing well
            if self.model_status.get(model_name) == "underperforming":
                self.model_status[model_name] = "active"
                record.status = "active"
                logger.info("model_performance_recovered", model_name=model_name)
        
        # Check for pruning candidates
        if self.config.auto_prune_enabled:
            if self._should_prune(model_name):
                if model_name not in self.prune_candidates:
                    self.prune_candidates.append(model_name)
                    logger.warning("model_added_to_prune_candidates", model_name=model_name)
    
    def _is_underperforming(self, metrics: ModelMetrics) -> bool:
        """Check if metrics indicate underperformance."""
        return (
            metrics.sharpe_ratio < self.config.sharpe_threshold
            or metrics.rmse > self.config.rmse_threshold
            or metrics.win_rate < self.config.win_rate_threshold
        )
    
    def _should_retrain(self, model_name: str) -> bool:
        """Check if model should be retrained."""
        if model_name not in self.performance_records:
            return False
        
        records = self.performance_records[model_name]
        if len(records) < self.config.min_samples_for_retrain:
            return False
        
        # Check if underperforming for threshold days
        cutoff_date = datetime.now() - timedelta(days=self.config.retrain_threshold_days)
        recent_records = [r for r in records if r.timestamp >= cutoff_date]
        
        if len(recent_records) < self.config.min_samples_for_retrain:
            return False
        
        # Check if all recent records are underperforming
        underperforming_count = sum(1 for r in recent_records if self._is_underperforming(r.metrics))
        return underperforming_count == len(recent_records)
    
    def _should_prune(self, model_name: str) -> bool:
        """Check if model should be pruned."""
        if model_name not in self.performance_records:
            return False
        
        records = self.performance_records[model_name]
        if len(records) < self.config.min_samples_for_retrain:
            return False
        
        # Check if underperforming for prune threshold days
        cutoff_date = datetime.now() - timedelta(days=self.config.prune_threshold_days)
        recent_records = [r for r in records if r.timestamp >= cutoff_date]
        
        if len(recent_records) < self.config.min_samples_for_retrain:
            return False
        
        # Check if consistently underperforming
        underperforming_count = sum(1 for r in recent_records if self._is_underperforming(r.metrics))
        underperforming_ratio = underperforming_count / len(recent_records) if recent_records else 0.0
        
        return underperforming_ratio >= 0.8  # 80% of recent records are underperforming
    
    def get_retrain_queue(self) -> List[str]:
        """Get list of models that should be retrained."""
        return self.retrain_queue.copy()
    
    def get_prune_candidates(self) -> List[str]:
        """Get list of models that should be pruned."""
        return self.prune_candidates.copy()
    
    def mark_retrained(self, model_name: str) -> None:
        """Mark model as retrained."""
        if model_name in self.retrain_queue:
            self.retrain_queue.remove(model_name)
        self.model_status[model_name] = "active"
        logger.info("model_retrained", model_name=model_name)
    
    def prune_model(self, model_name: str) -> None:
        """Prune (remove) a model."""
        if model_name in self.prune_candidates:
            self.prune_candidates.remove(model_name)
        self.model_status[model_name] = "pruned"
        logger.warning("model_pruned", model_name=model_name)
    
    def get_model_status(self, model_name: str) -> str:
        """Get current status of a model."""
        return self.model_status.get(model_name, "unknown")
    
    def get_performance_summary(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get performance summary for a model."""
        if model_name not in self.performance_records:
            return None
        
        records = self.performance_records[model_name]
        if not records:
            return None
        
        # Get recent records (last 7 days)
        cutoff_date = datetime.now() - timedelta(days=7)
        recent_records = [r for r in records if r.timestamp >= cutoff_date]
        
        if not recent_records:
            return None
        
        # Calculate averages
        avg_sharpe = np.mean([r.metrics.sharpe_ratio for r in recent_records])
        avg_rmse = np.mean([r.metrics.rmse for r in recent_records])
        avg_win_rate = np.mean([r.metrics.win_rate for r in recent_records])
        total_samples = sum(r.sample_count for r in recent_records)
        
        return {
            "model_name": model_name,
            "status": self.model_status.get(model_name, "unknown"),
            "avg_sharpe": avg_sharpe,
            "avg_rmse": avg_rmse,
            "avg_win_rate": avg_win_rate,
            "total_samples": total_samples,
            "num_records": len(recent_records),
            "latest_timestamp": max(r.timestamp for r in recent_records).isoformat(),
        }
    
    def get_all_performance_summaries(self) -> Dict[str, Dict[str, Any]]:
        """Get performance summaries for all models."""
        summaries = {}
        for model_name in self.performance_records.keys():
            summary = self.get_performance_summary(model_name)
            if summary:
                summaries[model_name] = summary
        return summaries
    
    def save_to_database(self) -> None:
        """Save performance records to database."""
        if not self.config.db_path:
            logger.warning("no_database_path_configured")
            return
        
        # TODO: Implement database saving
        # This would connect to PostgreSQL or SQLite and save performance records
        logger.info("saving_performance_to_database", db_path=self.config.db_path)
    
    def load_from_database(self) -> None:
        """Load performance records from database."""
        if not self.config.db_path:
            logger.warning("no_database_path_configured")
            return
        
        # TODO: Implement database loading
        logger.info("loading_performance_from_database", db_path=self.config.db_path)

