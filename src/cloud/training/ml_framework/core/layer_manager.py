"""
Layer Manager - Dynamic Layer Building

Manages dynamic layer creation based on configuration.
Supports building complex architectures from YAML configs.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import structlog
import torch
import torch.nn as nn

from .neural_base import Layer

logger = structlog.get_logger(__name__)


class LayerManager:
    """
    Manages dynamic layer creation and architecture building.
    
    Supports:
    - Linear layers
    - Convolutional layers (1D, 2D)
    - Recurrent layers (LSTM, GRU)
    - Attention layers
    - Residual connections
    - Skip connections
    """
    
    def __init__(self):
        """Initialize layer manager."""
        self.layer_registry: Dict[str, Any] = {}
        logger.info("layer_manager_initialized")
    
    def create_layer(
        self,
        layer_config: Dict[str, Any],
        input_size: int,
    ) -> nn.Module:
        """
        Create a layer from configuration.
        
        Args:
            layer_config: Layer configuration dictionary
            input_size: Size of input features
            
        Returns:
            Created layer module
        """
        layer_type = layer_config.get("type", "linear")
        
        if layer_type == "linear":
            return self._create_linear_layer(layer_config, input_size)
        elif layer_type == "conv1d":
            return self._create_conv1d_layer(layer_config, input_size)
        elif layer_type == "conv2d":
            return self._create_conv2d_layer(layer_config, input_size)
        elif layer_type == "lstm":
            return self._create_lstm_layer(layer_config, input_size)
        elif layer_type == "gru":
            return self._create_gru_layer(layer_config, input_size)
        elif layer_type == "attention":
            return self._create_attention_layer(layer_config, input_size)
        elif layer_type == "residual":
            return self._create_residual_layer(layer_config, input_size)
        else:
            raise ValueError(f"Unknown layer type: {layer_type}")
    
    def _create_linear_layer(self, config: Dict[str, Any], input_size: int) -> Layer:
        """Create linear (fully connected) layer."""
        out_features = config.get("out_features", input_size)
        activation = config.get("activation", "relu")
        dropout = config.get("dropout", 0.0)
        use_batch_norm = config.get("use_batch_norm", False)
        
        return Layer(
            in_features=input_size,
            out_features=out_features,
            activation=activation,
            dropout=dropout,
            use_batch_norm=use_batch_norm,
            layer_type="linear",
        )
    
    def _create_conv1d_layer(self, config: Dict[str, Any], input_size: int) -> Layer:
        """Create 1D convolutional layer."""
        out_channels = config.get("out_channels", input_size)
        kernel_size = config.get("kernel_size", 3)
        stride = config.get("stride", 1)
        padding = config.get("padding", 1)
        activation = config.get("activation", "relu")
        dropout = config.get("dropout", 0.0)
        use_batch_norm = config.get("use_batch_norm", False)
        
        return Layer(
            in_features=input_size,
            out_features=out_channels,
            activation=activation,
            dropout=dropout,
            use_batch_norm=use_batch_norm,
            layer_type="conv1d",
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
    
    def _create_conv2d_layer(self, config: Dict[str, Any], input_size: int) -> Layer:
        """Create 2D convolutional layer."""
        out_channels = config.get("out_channels", input_size)
        kernel_size = config.get("kernel_size", 3)
        stride = config.get("stride", 1)
        padding = config.get("padding", 1)
        activation = config.get("activation", "relu")
        dropout = config.get("dropout", 0.0)
        use_batch_norm = config.get("use_batch_norm", False)
        
        return Layer(
            in_features=input_size,
            out_features=out_channels,
            activation=activation,
            dropout=dropout,
            use_batch_norm=use_batch_norm,
            layer_type="conv2d",
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
    
    def _create_lstm_layer(self, config: Dict[str, Any], input_size: int) -> Layer:
        """Create LSTM layer."""
        hidden_size = config.get("hidden_size", input_size)
        num_layers = config.get("num_layers", 1)
        dropout = config.get("dropout", 0.0)
        bidirectional = config.get("bidirectional", False)
        
        return Layer(
            in_features=input_size,
            out_features=hidden_size,
            activation="tanh",
            dropout=dropout,
            use_batch_norm=False,
            layer_type="lstm",
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
        )
    
    def _create_gru_layer(self, config: Dict[str, Any], input_size: int) -> nn.Module:
        """Create GRU layer."""
        hidden_size = config.get("hidden_size", input_size)
        num_layers = config.get("num_layers", 1)
        dropout = config.get("dropout", 0.0)
        bidirectional = config.get("bidirectional", False)
        
        return nn.GRU(
            input_size,
            hidden_size,
            num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True,
        )
    
    def _create_attention_layer(self, config: Dict[str, Any], input_size: int) -> nn.Module:
        """Create attention layer."""
        hidden_size = config.get("hidden_size", input_size)
        num_heads = config.get("num_heads", 8)
        dropout = config.get("dropout", 0.1)
        
        return nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
    
    def _create_residual_layer(self, config: Dict[str, Any], input_size: int) -> nn.Module:
        """Create residual connection layer."""
        # Residual block with two linear layers
        hidden_size = config.get("hidden_size", input_size)
        activation = config.get("activation", "relu")
        dropout = config.get("dropout", 0.0)
        use_batch_norm = config.get("use_batch_norm", True)
        
        class ResidualBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = Layer(
                    in_features=input_size,
                    out_features=hidden_size,
                    activation=activation,
                    dropout=dropout,
                    use_batch_norm=use_batch_norm,
                    layer_type="linear",
                )
                self.layer2 = Layer(
                    in_features=hidden_size,
                    out_features=input_size,
                    activation="linear",
                    dropout=0.0,
                    use_batch_norm=False,
                    layer_type="linear",
                )
            
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                residual = x
                out = self.layer1(x)
                out = self.layer2(out)
                return out + residual  # Residual connection
        
        return ResidualBlock()
    
    def build_architecture(
        self,
        input_size: int,
        layers_config: List[Dict[str, Any]],
    ) -> nn.ModuleList:
        """
        Build complete architecture from configuration.
        
        Args:
            input_size: Size of input features
            layers_config: List of layer configurations
            
        Returns:
            ModuleList of layers
        """
        layers = nn.ModuleList()
        current_size = input_size
        
        for i, layer_config in enumerate(layers_config):
            layer = self.create_layer(layer_config, current_size)
            layers.append(layer)
            
            # Update current size based on layer type
            if hasattr(layer, "out_features"):
                current_size = layer.out_features
            elif hasattr(layer, "hidden_size"):
                current_size = layer.hidden_size
            elif isinstance(layer, nn.GRU) or isinstance(layer, nn.LSTM):
                current_size = layer.hidden_size
            
            logger.debug(
                "layer_added_to_architecture",
                layer_index=i,
                layer_type=layer_config.get("type"),
                output_size=current_size,
            )
        
        logger.info(
            "architecture_built",
            input_size=input_size,
            num_layers=len(layers),
            final_size=current_size,
        )
        
        return layers

