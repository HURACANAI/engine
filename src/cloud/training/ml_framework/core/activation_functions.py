"""
Activation Functions Library

Comprehensive collection of activation functions for neural networks.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_activation(name: str) -> nn.Module:
    """
    Get activation function by name.
    
    Args:
        name: Activation function name
        
    Returns:
        Activation function module
    """
    activation_map = {
        "relu": nn.ReLU(),
        "leaky_relu": nn.LeakyReLU(),
        "elu": nn.ELU(),
        "gelu": nn.GELU(),
        "sigmoid": nn.Sigmoid(),
        "tanh": nn.Tanh(),
        "softmax": nn.Softmax(dim=-1),
        "softplus": nn.Softplus(),
        "swish": lambda: Swish(),
        "mish": lambda: Mish(),
        "linear": nn.Identity(),
        "none": nn.Identity(),
    }
    
    activation = activation_map.get(name.lower())
    if activation is None:
        raise ValueError(f"Unknown activation function: {name}")
    
    if callable(activation) and not isinstance(activation, nn.Module):
        return activation()
    return activation


class Swish(nn.Module):
    """Swish activation function: x * sigmoid(x)."""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class Mish(nn.Module):
    """Mish activation function: x * tanh(softplus(x))."""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.tanh(F.softplus(x))


class GLU(nn.Module):
    """Gated Linear Unit activation."""
    
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, gate = x.chunk(2, dim=self.dim)
        return out * torch.sigmoid(gate)


class SELU(nn.Module):
    """Scaled Exponential Linear Unit."""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.selu(x)


# Export all activation functions
__all__ = [
    "get_activation",
    "Swish",
    "Mish",
    "GLU",
    "SELU",
]

