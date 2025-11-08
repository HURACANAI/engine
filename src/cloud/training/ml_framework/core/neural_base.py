"""
Neural Network Core - Base Architecture

Implements base neural network components: Layer, Neuron, and Activation functions.
Each layer is treated as a "specialized expert" in the network.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import structlog
import torch
import torch.nn as nn

logger = structlog.get_logger(__name__)

# Check if PyTorch is available
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.warning("pytorch_not_available_for_neural_base")


class ActivationFunction(ABC):
    """Base class for activation functions."""
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        pass
    
    @abstractmethod
    def backward(self, x: torch.Tensor, grad_output: torch.Tensor) -> torch.Tensor:
        """Backward pass (gradient)."""
        pass


class ReLU(ActivationFunction):
    """ReLU activation function."""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x)
    
    def backward(self, x: torch.Tensor, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output * (x > 0).float()


class Sigmoid(ActivationFunction):
    """Sigmoid activation function."""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x)
    
    def backward(self, x: torch.Tensor, grad_output: torch.Tensor) -> torch.Tensor:
        s = torch.sigmoid(x)
        return grad_output * s * (1 - s)


class Tanh(ActivationFunction):
    """Tanh activation function."""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x)
    
    def backward(self, x: torch.Tensor, grad_output: torch.Tensor) -> torch.Tensor:
        t = torch.tanh(x)
        return grad_output * (1 - t ** 2)


class Softmax(ActivationFunction):
    """Softmax activation function."""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(x, dim=-1)
    
    def backward(self, x: torch.Tensor, grad_output: torch.Tensor) -> torch.Tensor:
        s = torch.softmax(x, dim=-1)
        return grad_output * s * (1 - s)


class Neuron(nn.Module):
    """
    Individual neuron (specialized expert) in a neural network.
    
    Each neuron:
    - Receives inputs from previous layer
    - Applies weights and bias
    - Applies activation function
    - Outputs to next layer
    """
    
    def __init__(
        self,
        input_size: int,
        activation: Optional[str] = "relu",
        use_bias: bool = True,
    ):
        """
        Initialize neuron.
        
        Args:
            input_size: Number of input connections
            activation: Activation function name ("relu", "sigmoid", "tanh", "softmax")
            use_bias: Whether to use bias term
        """
        super().__init__()
        
        self.input_size = input_size
        self.use_bias = use_bias
        
        # Initialize weights
        self.weight = nn.Parameter(torch.randn(input_size) * 0.1)
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(1))
        else:
            self.register_parameter("bias", None)
        
        # Set activation function
        activation_map = {
            "relu": nn.ReLU(),
            "sigmoid": nn.Sigmoid(),
            "tanh": nn.Tanh(),
            "softmax": nn.Softmax(dim=-1),
            "linear": nn.Identity(),
        }
        self.activation = activation_map.get(activation, nn.ReLU())
        
        logger.debug(
            "neuron_initialized",
            input_size=input_size,
            activation=activation,
            use_bias=use_bias,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through neuron."""
        # Linear transformation: w^T * x + b
        output = torch.dot(x, self.weight)
        if self.use_bias:
            output = output + self.bias
        
        # Apply activation
        output = self.activation(output)
        
        return output


class Layer(nn.Module):
    """
    Neural network layer (collection of specialized expert neurons).
    
    Each layer:
    - Contains multiple neurons
    - Applies transformation (linear, convolutional, etc.)
    - Applies activation function
    - Can have dropout for regularization
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: str = "relu",
        dropout: float = 0.0,
        use_batch_norm: bool = False,
        layer_type: str = "linear",
        **kwargs: Any,
    ):
        """
        Initialize layer.
        
        Args:
            in_features: Number of input features
            out_features: Number of output features
            activation: Activation function name
            dropout: Dropout probability
            use_batch_norm: Whether to use batch normalization
            layer_type: Type of layer ("linear", "conv1d", "conv2d", "lstm", "attention")
            **kwargs: Additional layer-specific parameters
        """
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.layer_type = layer_type
        self.activation_name = activation
        
        # Create layer based on type
        if layer_type == "linear":
            self.layer = nn.Linear(in_features, out_features)
        elif layer_type == "conv1d":
            kernel_size = kwargs.get("kernel_size", 3)
            self.layer = nn.Conv1d(in_features, out_features, kernel_size, **{k: v for k, v in kwargs.items() if k != "kernel_size"})
        elif layer_type == "conv2d":
            kernel_size = kwargs.get("kernel_size", 3)
            self.layer = nn.Conv2d(in_features, out_features, kernel_size, **{k: v for k, v in kwargs.items() if k != "kernel_size"})
        elif layer_type == "lstm":
            hidden_size = kwargs.get("hidden_size", out_features)
            num_layers = kwargs.get("num_layers", 1)
            self.layer = nn.LSTM(in_features, hidden_size, num_layers, batch_first=True, **{k: v for k, v in kwargs.items() if k not in ["hidden_size", "num_layers"]})
            self.out_features = hidden_size
        else:
            raise ValueError(f"Unknown layer type: {layer_type}")
        
        # Activation function
        activation_map = {
            "relu": nn.ReLU(),
            "sigmoid": nn.Sigmoid(),
            "tanh": nn.Tanh(),
            "softmax": nn.Softmax(dim=-1),
            "linear": nn.Identity(),
            "gelu": nn.GELU(),
        }
        self.activation = activation_map.get(activation, nn.ReLU())
        
        # Batch normalization
        if use_batch_norm:
            if layer_type in ["conv1d", "conv2d"]:
                self.batch_norm = nn.BatchNorm1d(out_features) if layer_type == "conv1d" else nn.BatchNorm2d(out_features)
            else:
                self.batch_norm = nn.BatchNorm1d(out_features)
        else:
            self.register_parameter("batch_norm", None)
        
        # Dropout
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.register_parameter("dropout", None)
        
        logger.debug(
            "layer_initialized",
            layer_type=layer_type,
            in_features=in_features,
            out_features=self.out_features,
            activation=activation,
            dropout=dropout,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through layer."""
        # Apply layer transformation
        output = self.layer(x)
        
        # Apply batch normalization
        if self.batch_norm is not None:
            output = self.batch_norm(output)
        
        # Apply activation
        output = self.activation(output)
        
        # Apply dropout
        if self.dropout is not None:
            output = self.dropout(output)
        
        return output
    
    def get_activation_stats(self, x: torch.Tensor) -> Dict[str, float]:
        """Get statistics about layer activations."""
        with torch.no_grad():
            output = self.forward(x)
            return {
                "mean": float(output.mean().item()),
                "std": float(output.std().item()),
                "min": float(output.min().item()),
                "max": float(output.max().item()),
            }


class BaseNeuralNetwork(nn.Module):
    """
    Base neural network architecture.
    
    Supports dynamic layer building based on configuration.
    Allows deep hierarchies (CNN, Transformer, GAN, etc.).
    """
    
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_layers: List[Dict[str, Any]],
        output_activation: Optional[str] = None,
    ):
        """
        Initialize base neural network.
        
        Args:
            input_size: Size of input features
            output_size: Size of output
            hidden_layers: List of layer configurations
            output_activation: Activation function for output layer
        """
        super().__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.layers = nn.ModuleList()
        
        # Build layers dynamically
        current_size = input_size
        for i, layer_config in enumerate(hidden_layers):
            layer_type = layer_config.get("type", "linear")
            out_features = layer_config.get("out_features", output_size)
            activation = layer_config.get("activation", "relu")
            dropout = layer_config.get("dropout", 0.0)
            use_batch_norm = layer_config.get("use_batch_norm", False)
            
            layer = Layer(
                in_features=current_size,
                out_features=out_features,
                activation=activation,
                dropout=dropout,
                use_batch_norm=use_batch_norm,
                layer_type=layer_type,
                **{k: v for k, v in layer_config.items() if k not in ["type", "out_features", "activation", "dropout", "use_batch_norm"]},
            )
            
            self.layers.append(layer)
            current_size = out_features
        
        # Output layer
        if output_activation:
            activation_map = {
                "relu": nn.ReLU(),
                "sigmoid": nn.Sigmoid(),
                "tanh": nn.Tanh(),
                "softmax": nn.Softmax(dim=-1),
                "linear": nn.Identity(),
            }
            self.output_activation = activation_map.get(output_activation, nn.Identity())
        else:
            self.output_activation = nn.Identity()
        
        logger.info(
            "base_neural_network_initialized",
            input_size=input_size,
            output_size=output_size,
            num_layers=len(self.layers),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through network."""
        # Pass through hidden layers
        for layer in self.layers:
            x = layer(x)
        
        # Apply output activation
        x = self.output_activation(x)
        
        return x
    
    def get_layer_activations(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Get activations from each layer (for interpretability)."""
        activations = []
        current = x
        
        for layer in self.layers:
            current = layer(current)
            activations.append(current.clone())
        
        return activations
    
    def get_layer_stats(self, x: torch.Tensor) -> List[Dict[str, float]]:
        """Get statistics from each layer."""
        stats = []
        current = x
        
        for layer in self.layers:
            layer_stats = layer.get_activation_stats(current)
            stats.append(layer_stats)
            current = layer(current)
        
        return stats

