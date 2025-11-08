"""
Explainability - Model Interpretability

Visualizes activations and neuron contributions for model interpretation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import structlog
import torch
import torch.nn as nn

logger = structlog.get_logger(__name__)


class ModelExplainability:
    """
    Model explainability and interpretability utilities.
    
    Features:
    - Activation visualization
    - Neuron contribution analysis
    - Feature importance
    - Layer diagnostics
    - Bias detection
    """
    
    def __init__(self, model: nn.Module):
        """
        Initialize explainability analyzer.
        
        Args:
            model: Neural network model
        """
        self.model = model
        self.activations: Dict[str, torch.Tensor] = {}
        self.gradients: Dict[str, torch.Tensor] = {}
        
        # Register hooks
        self._register_hooks()
        
        logger.info("model_explainability_initialized")
    
    def _register_hooks(self) -> None:
        """Register forward and backward hooks."""
        def forward_hook(name: str):
            def hook(module: nn.Module, input: torch.Tensor, output: torch.Tensor) -> None:
                self.activations[name] = output.detach()
            return hook
        
        def backward_hook(name: str):
            def hook(module: nn.Module, grad_input: torch.Tensor, grad_output: torch.Tensor) -> None:
                if grad_output[0] is not None:
                    self.gradients[name] = grad_output[0].detach()
            return hook
        
        # Register hooks for each layer
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # Leaf module
                module.register_forward_hook(forward_hook(name))
                module.register_backward_hook(backward_hook(name))
    
    def get_activations(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get activations for input.
        
        Args:
            x: Input tensor
            
        Returns:
            Dictionary of layer names to activations
        """
        self.activations.clear()
        _ = self.model(x)
        return self.activations.copy()
    
    def get_gradients(self, x: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get gradients for input.
        
        Args:
            x: Input tensor
            target: Target tensor
            
        Returns:
            Dictionary of layer names to gradients
        """
        self.gradients.clear()
        self.model.zero_grad()
        
        output = self.model(x)
        loss = nn.functional.mse_loss(output, target)
        loss.backward()
        
        return self.gradients.copy()
    
    def analyze_layer_activations(self, x: torch.Tensor) -> Dict[str, Dict[str, float]]:
        """
        Analyze layer activations.
        
        Args:
            x: Input tensor
            
        Returns:
            Dictionary of layer statistics
        """
        activations = self.get_activations(x)
        stats = {}
        
        for layer_name, activation in activations.items():
            stats[layer_name] = {
                "mean": float(activation.mean().item()),
                "std": float(activation.std().item()),
                "min": float(activation.min().item()),
                "max": float(activation.max().item()),
                "zeros": float((activation == 0).sum().item() / activation.numel()),
            }
        
        return stats
    
    def detect_dead_neurons(self, x: torch.Tensor, threshold: float = 1e-6) -> Dict[str, List[int]]:
        """
        Detect dead neurons (neurons that never activate).
        
        Args:
            x: Input tensor
            threshold: Activation threshold
            
        Returns:
            Dictionary of layer names to lists of dead neuron indices
        """
        activations = self.get_activations(x)
        dead_neurons = {}
        
        for layer_name, activation in activations.items():
            # Calculate mean activation per neuron
            if len(activation.shape) > 1:
                neuron_activations = activation.mean(dim=0)
                dead_indices = (neuron_activations < threshold).nonzero(as_tuple=True)[0].tolist()
                if dead_indices:
                    dead_neurons[layer_name] = dead_indices
        
        return dead_neurons
    
    def get_feature_importance(self, x: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        """
        Get feature importance using gradients.
        
        Args:
            x: Input tensor
            target: Target tensor
            
        Returns:
            Dictionary of feature names to importance scores
        """
        gradients = self.get_gradients(x, target)
        
        # Calculate importance as gradient magnitude
        importance = {}
        for layer_name, grad in gradients.items():
            if grad is not None:
                importance[layer_name] = float(grad.abs().mean().item())
        
        return importance
    
    def detect_bias(self, x: torch.Tensor) -> Dict[str, Any]:
        """
        Detect bias in model (data imbalance, etc.).
        
        Args:
            x: Input tensor
            
        Returns:
            Dictionary with bias diagnostics
        """
        activations = self.get_activations(x)
        bias_stats = {}
        
        for layer_name, activation in activations.items():
            # Check for activation imbalance
            mean_activation = activation.mean().item()
            std_activation = activation.std().item()
            
            # Detect if activations are consistently high or low (potential bias)
            if mean_activation > 0.9:
                bias_type = "high_activation_bias"
            elif mean_activation < 0.1:
                bias_type = "low_activation_bias"
            else:
                bias_type = "none"
            
            bias_stats[layer_name] = {
                "mean_activation": mean_activation,
                "std_activation": std_activation,
                "bias_type": bias_type,
            }
        
        return bias_stats


class GradientBasedExplainability:
    """Gradient-based explainability methods (Grad-CAM, etc.)."""
    
    @staticmethod
    def grad_cam(model: nn.Module, x: torch.Tensor, target_layer: str) -> torch.Tensor:
        """
        Generate Grad-CAM visualization.
        
        Args:
            model: Neural network model
            x: Input tensor
            target_layer: Name of target layer
            
        Returns:
            Grad-CAM heatmap
        """
        # Implementation would go here
        # For now, return dummy
        return torch.zeros_like(x)

