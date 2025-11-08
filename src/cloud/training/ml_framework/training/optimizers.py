"""
Optimizers - Weight Update Strategies

Implements multiple optimizers: SGD, Adam, RMSProp, and variants.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import structlog
import torch
import torch.optim as optim

logger = structlog.get_logger(__name__)


class OptimizerFactory:
    """Factory for creating optimizers."""
    
    @staticmethod
    def create(
        model: torch.nn.Module,
        optimizer_type: str = "adam",
        learning_rate: float = 0.001,
        **kwargs: Any,
    ) -> torch.optim.Optimizer:
        """
        Create optimizer.
        
        Args:
            model: Neural network model
            optimizer_type: Type of optimizer ("sgd", "adam", "rmsprop", "adagrad", "adamw")
            learning_rate: Learning rate
            **kwargs: Additional optimizer-specific parameters
            
        Returns:
            Optimizer instance
        """
        params = model.parameters()
        
        if optimizer_type.lower() == "sgd":
            momentum = kwargs.get("momentum", 0.9)
            weight_decay = kwargs.get("weight_decay", 0.0)
            nesterov = kwargs.get("nesterov", False)
            return optim.SGD(params, lr=learning_rate, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)
        
        elif optimizer_type.lower() == "adam":
            betas = kwargs.get("betas", (0.9, 0.999))
            eps = kwargs.get("eps", 1e-8)
            weight_decay = kwargs.get("weight_decay", 0.0)
            return optim.Adam(params, lr=learning_rate, betas=betas, eps=eps, weight_decay=weight_decay)
        
        elif optimizer_type.lower() == "adamw":
            betas = kwargs.get("betas", (0.9, 0.999))
            eps = kwargs.get("eps", 1e-8)
            weight_decay = kwargs.get("weight_decay", 0.01)
            return optim.AdamW(params, lr=learning_rate, betas=betas, eps=eps, weight_decay=weight_decay)
        
        elif optimizer_type.lower() == "rmsprop":
            alpha = kwargs.get("alpha", 0.99)
            eps = kwargs.get("eps", 1e-8)
            weight_decay = kwargs.get("weight_decay", 0.0)
            momentum = kwargs.get("momentum", 0.0)
            return optim.RMSprop(params, lr=learning_rate, alpha=alpha, eps=eps, weight_decay=weight_decay, momentum=momentum)
        
        elif optimizer_type.lower() == "adagrad":
            lr_decay = kwargs.get("lr_decay", 0.0)
            weight_decay = kwargs.get("weight_decay", 0.0)
            eps = kwargs.get("eps", 1e-10)
            return optim.Adagrad(params, lr=learning_rate, lr_decay=lr_decay, weight_decay=weight_decay, eps=eps)
        
        elif optimizer_type.lower() == "adadelta":
            rho = kwargs.get("rho", 0.9)
            eps = kwargs.get("eps", 1e-6)
            weight_decay = kwargs.get("weight_decay", 0.0)
            return optim.Adadelta(params, lr=learning_rate, rho=rho, eps=eps, weight_decay=weight_decay)
        
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    @staticmethod
    def get_default_config(optimizer_type: str) -> Dict[str, Any]:
        """Get default configuration for optimizer."""
        configs = {
            "sgd": {
                "learning_rate": 0.01,
                "momentum": 0.9,
                "weight_decay": 0.0,
                "nesterov": False,
            },
            "adam": {
                "learning_rate": 0.001,
                "betas": (0.9, 0.999),
                "eps": 1e-8,
                "weight_decay": 0.0,
            },
            "adamw": {
                "learning_rate": 0.001,
                "betas": (0.9, 0.999),
                "eps": 1e-8,
                "weight_decay": 0.01,
            },
            "rmsprop": {
                "learning_rate": 0.001,
                "alpha": 0.99,
                "eps": 1e-8,
                "weight_decay": 0.0,
                "momentum": 0.0,
            },
            "adagrad": {
                "learning_rate": 0.01,
                "lr_decay": 0.0,
                "weight_decay": 0.0,
                "eps": 1e-10,
            },
            "adadelta": {
                "learning_rate": 1.0,
                "rho": 0.9,
                "eps": 1e-6,
                "weight_decay": 0.0,
            },
        }
        return configs.get(optimizer_type.lower(), configs["adam"])


class OptimizerWrapper:
    """Wrapper for optimizer with additional features."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer_type: str = "adam",
        learning_rate: float = 0.001,
        **kwargs: Any,
    ):
        """
        Initialize optimizer wrapper.
        
        Args:
            model: Neural network model
            optimizer_type: Type of optimizer
            learning_rate: Learning rate
            **kwargs: Additional optimizer parameters
        """
        self.optimizer_type = optimizer_type
        self.learning_rate = learning_rate
        self.optimizer = OptimizerFactory.create(model, optimizer_type, learning_rate, **kwargs)
        
        logger.info(
            "optimizer_wrapper_initialized",
            optimizer_type=optimizer_type,
            learning_rate=learning_rate,
        )
    
    def step(self) -> None:
        """Perform optimization step."""
        self.optimizer.step()
    
    def zero_grad(self) -> None:
        """Zero gradients."""
        self.optimizer.zero_grad()
    
    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]["lr"]
    
    def set_lr(self, learning_rate: float) -> None:
        """Set learning rate."""
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = learning_rate
        logger.info("learning_rate_updated", new_lr=learning_rate)
    
    def state_dict(self) -> Dict[str, Any]:
        """Get optimizer state dictionary."""
        return self.optimizer.state_dict()
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load optimizer state dictionary."""
        self.optimizer.load_state_dict(state_dict)
        logger.info("optimizer_state_loaded")

