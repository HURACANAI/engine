"""
Backpropagation - Gradient Flow and Weight Updates

Implements backpropagation algorithm for training neural networks.
Handles gradient computation and flow through the network.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import structlog
import torch
import torch.nn as nn

logger = structlog.get_logger(__name__)


class Backpropagation:
    """
    Backpropagation engine for neural network training.
    
    Handles:
    - Forward pass
    - Backward pass (gradient computation)
    - Gradient clipping
    - Gradient accumulation
    - Gradient statistics
    """
    
    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        clip_grad_norm: Optional[float] = None,
        accumulation_steps: int = 1,
    ):
        """
        Initialize backpropagation engine.
        
        Args:
            model: Neural network model
            loss_fn: Loss function
            clip_grad_norm: Maximum gradient norm for clipping (None = no clipping)
            accumulation_steps: Number of steps for gradient accumulation
        """
        self.model = model
        self.loss_fn = loss_fn
        self.clip_grad_norm = clip_grad_norm
        self.accumulation_steps = accumulation_steps
        self.accumulation_counter = 0
        
        # Gradient statistics
        self.gradient_stats: List[Dict[str, float]] = []
        
        logger.info(
            "backpropagation_initialized",
            clip_grad_norm=clip_grad_norm,
            accumulation_steps=accumulation_steps,
        )
    
    def forward_backward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> Dict[str, float]:
        """
        Perform forward and backward pass.
        
        Args:
            inputs: Input tensor
            targets: Target tensor
            optimizer: Optimizer for weight updates (optional)
            
        Returns:
            Dictionary with loss and gradient statistics
        """
        # Forward pass
        outputs = self.model(inputs)
        loss = self.loss_fn(outputs, targets)
        
        # Scale loss for gradient accumulation
        if self.accumulation_steps > 1:
            loss = loss / self.accumulation_steps
        
        # Backward pass
        loss.backward()
        
        self.accumulation_counter += 1
        
        # Update weights if accumulation is complete
        if self.accumulation_counter >= self.accumulation_steps:
            # Clip gradients if specified
            if self.clip_grad_norm is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.clip_grad_norm,
                )
            else:
                # Calculate gradient norm without clipping
                grad_norm = self._calculate_gradient_norm()
            
            # Get gradient statistics
            grad_stats = self._get_gradient_statistics()
            
            # Update weights
            if optimizer is not None:
                optimizer.step()
                optimizer.zero_grad()
            
            self.accumulation_counter = 0
            
            return {
                "loss": float(loss.item() * self.accumulation_steps),  # Scale back
                "grad_norm": float(grad_norm),
                **grad_stats,
            }
        else:
            # Just return loss, don't update weights yet
            return {
                "loss": float(loss.item() * self.accumulation_steps),
                "grad_norm": 0.0,
            }
    
    def _calculate_gradient_norm(self) -> float:
        """Calculate total gradient norm."""
        total_norm = 0.0
        for param in self.model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1.0 / 2)
        return total_norm
    
    def _get_gradient_statistics(self) -> Dict[str, float]:
        """Get statistics about gradients."""
        gradients = []
        for param in self.model.parameters():
            if param.grad is not None:
                gradients.append(param.grad.data.flatten())
        
        if not gradients:
            return {
                "grad_mean": 0.0,
                "grad_std": 0.0,
                "grad_max": 0.0,
                "grad_min": 0.0,
            }
        
        all_grads = torch.cat(gradients)
        
        stats = {
            "grad_mean": float(all_grads.mean().item()),
            "grad_std": float(all_grads.std().item()),
            "grad_max": float(all_grads.max().item()),
            "grad_min": float(all_grads.min().item()),
        }
        
        self.gradient_stats.append(stats)
        return stats
    
    def get_gradient_flow(self) -> Dict[str, List[float]]:
        """
        Get gradient flow through layers.
        
        Returns:
            Dictionary mapping layer names to gradient norms
        """
        gradient_flow = {}
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm(2).item()
                gradient_flow[name] = grad_norm
            else:
                gradient_flow[name] = 0.0
        
        return gradient_flow
    
    def check_gradient_health(self) -> Dict[str, Any]:
        """
        Check gradient health (detect vanishing/exploding gradients).
        
        Returns:
            Dictionary with gradient health diagnostics
        """
        gradient_flow = self.get_gradient_flow()
        
        if not gradient_flow:
            return {
                "healthy": False,
                "issue": "no_gradients",
            }
        
        grad_norms = list(gradient_flow.values())
        mean_grad = sum(grad_norms) / len(grad_norms)
        max_grad = max(grad_norms)
        min_grad = min(grad_norms)
        
        # Check for vanishing gradients
        vanishing_threshold = 1e-6
        vanishing_layers = [name for name, norm in gradient_flow.items() if norm < vanishing_threshold]
        
        # Check for exploding gradients
        exploding_threshold = 1e3
        exploding_layers = [name for name, norm in gradient_flow.items() if norm > exploding_threshold]
        
        return {
            "healthy": len(vanishing_layers) == 0 and len(exploding_layers) == 0,
            "mean_gradient": mean_grad,
            "max_gradient": max_grad,
            "min_gradient": min_grad,
            "vanishing_layers": vanishing_layers,
            "exploding_layers": exploding_layers,
        }
    
    def reset_accumulation(self) -> None:
        """Reset gradient accumulation counter."""
        self.accumulation_counter = 0


class GradientTracker:
    """Track gradients over time for analysis."""
    
    def __init__(self):
        """Initialize gradient tracker."""
        self.history: List[Dict[str, float]] = []
    
    def track(self, gradient_flow: Dict[str, List[float]]) -> None:
        """Track gradient flow."""
        self.history.append(gradient_flow)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get gradient statistics over time."""
        if not self.history:
            return {}
        
        # Aggregate statistics
        all_norms = []
        for grad_dict in self.history:
            if isinstance(grad_dict, dict):
                all_norms.extend([v for v in grad_dict.values() if isinstance(v, (int, float))])
        
        if not all_norms:
            return {}
        
        return {
            "mean_gradient": float(sum(all_norms) / len(all_norms)),
            "max_gradient": float(max(all_norms)),
            "min_gradient": float(min(all_norms)),
            "num_tracking_steps": len(self.history),
        }

