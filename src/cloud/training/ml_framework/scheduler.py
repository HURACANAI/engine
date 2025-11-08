"""
Learning Rate Schedulers

Implements various learning rate scheduling strategies for neural network training.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import structlog
import torch
import torch.optim as optim

logger = structlog.get_logger(__name__)

# Check if PyTorch is available
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.warning("pytorch_not_available_for_schedulers")


class LearningRateScheduler:
    """Learning rate scheduler for neural network training."""
    
    def __init__(
        self,
        optimizer: Optional[Any] = None,
        scheduler_type: str = "step",
        **kwargs: Any,
    ):
        """
        Initialize learning rate scheduler.
        
        Args:
            optimizer: PyTorch optimizer
            scheduler_type: Type of scheduler ("step", "cosine", "exponential", "plateau")
            **kwargs: Scheduler-specific parameters
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for learning rate scheduling")
        
        self.optimizer = optimizer
        self.scheduler_type = scheduler_type
        self.scheduler: Optional[Any] = None
        
        if optimizer is not None:
            self._create_scheduler(**kwargs)
        
        logger.info(
            "learning_rate_scheduler_initialized",
            scheduler_type=scheduler_type,
        )
    
    def _create_scheduler(self, **kwargs: Any) -> None:
        """Create scheduler based on type."""
        if self.scheduler_type == "step":
            step_size = kwargs.get("step_size", 10)
            gamma = kwargs.get("gamma", 0.1)
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=step_size,
                gamma=gamma,
            )
        
        elif self.scheduler_type == "cosine":
            T_max = kwargs.get("T_max", 50)
            eta_min = kwargs.get("eta_min", 0.0)
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=T_max,
                eta_min=eta_min,
            )
        
        elif self.scheduler_type == "exponential":
            gamma = kwargs.get("gamma", 0.95)
            self.scheduler = optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=gamma,
            )
        
        elif self.scheduler_type == "plateau":
            mode = kwargs.get("mode", "min")
            factor = kwargs.get("factor", 0.1)
            patience = kwargs.get("patience", 10)
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=mode,
                factor=factor,
                patience=patience,
            )
        
        else:
            raise ValueError(f"Unknown scheduler type: {self.scheduler_type}")
    
    def step(self, metrics: Optional[float] = None) -> None:
        """
        Step scheduler.
        
        Args:
            metrics: Metric value (for plateau scheduler)
        """
        if self.scheduler is None:
            raise ValueError("Scheduler not initialized")
        
        if self.scheduler_type == "plateau" and metrics is not None:
            self.scheduler.step(metrics)
        else:
            self.scheduler.step()
    
    def get_lr(self) -> float:
        """Get current learning rate."""
        if self.optimizer is None:
            return 0.0
        return self.optimizer.param_groups[0]["lr"]
    
    def set_optimizer(self, optimizer: Any) -> None:
        """Set optimizer and recreate scheduler."""
        self.optimizer = optimizer
        self._create_scheduler()


def create_scheduler(
    optimizer: Any,
    scheduler_config: Dict[str, Any],
) -> LearningRateScheduler:
    """
    Create learning rate scheduler from configuration.
    
    Args:
        optimizer: PyTorch optimizer
        scheduler_config: Scheduler configuration dictionary
        
    Returns:
        LearningRateScheduler instance
    """
    scheduler_type = scheduler_config.get("type", "step")
    kwargs = {k: v for k, v in scheduler_config.items() if k != "type"}
    
    return LearningRateScheduler(
        optimizer=optimizer,
        scheduler_type=scheduler_type,
        **kwargs,
    )

