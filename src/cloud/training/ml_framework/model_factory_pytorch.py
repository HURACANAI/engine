"""
PyTorch Model Factory

Integration of PyTorch models with dropout, batch norm, and architecture testing.
Supports feed-forward, LSTM, and hybrid architectures.

Key Features:
- Feed-forward networks with dropout and batch norm
- LSTM networks for sequential data
- Hybrid architectures (CNN+LSTM, etc.)
- Architecture testing and comparison
- Model versioning
- ONNX export support

Author: Huracan Engine Team
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import torch
import torch.nn as nn
import torch.optim as optim
import structlog

logger = structlog.get_logger(__name__)


class ArchitectureType(Enum):
    """Model architecture type"""
    FEED_FORWARD = "feed_forward"
    LSTM = "lstm"
    HYBRID = "hybrid"
    CNN_LSTM = "cnn_lstm"


@dataclass
class ModelConfig:
    """Model configuration"""
    architecture_type: ArchitectureType
    input_size: int
    hidden_sizes: List[int]
    output_size: int = 1
    dropout_rate: float = 0.2
    use_batch_norm: bool = True
    activation: str = "relu"  # "relu", "tanh", "sigmoid", "gelu"
    lstm_hidden_size: Optional[int] = None
    lstm_num_layers: int = 2
    lstm_bidirectional: bool = False
    cnn_channels: Optional[List[int]] = None
    cnn_kernel_sizes: Optional[List[int]] = None


class FeedForwardNetwork(nn.Module):
    """
    Feed-forward neural network with dropout and batch normalization.
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList()
        
        # Input layer
        prev_size = config.input_size
        
        # Hidden layers
        for hidden_size in config.hidden_sizes:
            # Linear layer
            self.layers.append(nn.Linear(prev_size, hidden_size))
            
            # Batch normalization
            if config.use_batch_norm:
                self.layers.append(nn.BatchNorm1d(hidden_size))
            
            # Activation
            activation = self._get_activation(config.activation)
            self.layers.append(activation)
            
            # Dropout
            if config.dropout_rate > 0:
                self.layers.append(nn.Dropout(config.dropout_rate))
            
            prev_size = hidden_size
        
        # Output layer
        self.output_layer = nn.Linear(prev_size, config.output_size)
        
        # Initialize weights
        self._initialize_weights()
    
    def _get_activation(self, activation_name: str) -> nn.Module:
        """Get activation function"""
        activations = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "gelu": nn.GELU(),
        }
        return activations.get(activation_name, nn.ReLU())
    
    def _initialize_weights(self):
        """Initialize weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        for layer in self.layers:
            x = layer(x)
        x = self.output_layer(x)
        return x


class LSTMNetwork(nn.Module):
    """
    LSTM network for sequential data.
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        if config.lstm_hidden_size is None:
            raise ValueError("lstm_hidden_size must be specified for LSTM architecture")
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=config.input_size,
            hidden_size=config.lstm_hidden_size,
            num_layers=config.lstm_num_layers,
            bidirectional=config.lstm_bidirectional,
            dropout=config.dropout_rate if config.lstm_num_layers > 1 else 0,
            batch_first=True
        )
        
        # Calculate LSTM output size
        lstm_output_size = config.lstm_hidden_size * (2 if config.lstm_bidirectional else 1)
        
        # Feed-forward layers after LSTM
        self.fc_layers = nn.ModuleList()
        prev_size = lstm_output_size
        
        for hidden_size in config.hidden_sizes:
            self.fc_layers.append(nn.Linear(prev_size, hidden_size))
            
            if config.use_batch_norm:
                self.fc_layers.append(nn.BatchNorm1d(hidden_size))
            
            activation = self._get_activation(config.activation)
            self.fc_layers.append(activation)
            
            if config.dropout_rate > 0:
                self.fc_layers.append(nn.Dropout(config.dropout_rate))
            
            prev_size = hidden_size
        
        # Output layer
        self.output_layer = nn.Linear(prev_size, config.output_size)
        
        # Initialize weights
        self._initialize_weights()
    
    def _get_activation(self, activation_name: str) -> nn.Module:
        """Get activation function"""
        activations = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "gelu": nn.GELU(),
        }
        return activations.get(activation_name, nn.ReLU())
    
    def _initialize_weights(self):
        """Initialize weights"""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.zeros_(param.data)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        # x shape: (batch_size, seq_len, input_size)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use last output
        x = lstm_out[:, -1, :]
        
        # Feed-forward layers
        for layer in self.fc_layers:
            x = layer(x)
        
        x = self.output_layer(x)
        return x


class HybridNetwork(nn.Module):
    """
    Hybrid network combining CNN and LSTM.
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        if config.cnn_channels is None or config.cnn_kernel_sizes is None:
            raise ValueError("cnn_channels and cnn_kernel_sizes must be specified for hybrid architecture")
        
        # CNN layers
        self.cnn_layers = nn.ModuleList()
        in_channels = 1  # Input channels
        
        for out_channels, kernel_size in zip(config.cnn_channels, config.cnn_kernel_sizes):
            self.cnn_layers.append(nn.Conv1d(in_channels, out_channels, kernel_size))
            self.cnn_layers.append(nn.ReLU())
            if config.use_batch_norm:
                self.cnn_layers.append(nn.BatchNorm1d(out_channels))
            in_channels = out_channels
        
        # Calculate CNN output size (simplified)
        cnn_output_size = config.cnn_channels[-1]
        
        # LSTM layer
        if config.lstm_hidden_size is None:
            raise ValueError("lstm_hidden_size must be specified for hybrid architecture")
        
        self.lstm = nn.LSTM(
            input_size=cnn_output_size,
            hidden_size=config.lstm_hidden_size,
            num_layers=config.lstm_num_layers,
            bidirectional=config.lstm_bidirectional,
            dropout=config.dropout_rate if config.lstm_num_layers > 1 else 0,
            batch_first=True
        )
        
        # Calculate LSTM output size
        lstm_output_size = config.lstm_hidden_size * (2 if config.lstm_bidirectional else 1)
        
        # Feed-forward layers
        self.fc_layers = nn.ModuleList()
        prev_size = lstm_output_size
        
        for hidden_size in config.hidden_sizes:
            self.fc_layers.append(nn.Linear(prev_size, hidden_size))
            
            if config.use_batch_norm:
                self.fc_layers.append(nn.BatchNorm1d(hidden_size))
            
            activation = self._get_activation(config.activation)
            self.fc_layers.append(activation)
            
            if config.dropout_rate > 0:
                self.fc_layers.append(nn.Dropout(config.dropout_rate))
            
            prev_size = hidden_size
        
        # Output layer
        self.output_layer = nn.Linear(prev_size, config.output_size)
        
        # Initialize weights
        self._initialize_weights()
    
    def _get_activation(self, activation_name: str) -> nn.Module:
        """Get activation function"""
        activations = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "gelu": nn.GELU(),
        }
        return activations.get(activation_name, nn.ReLU())
    
    def _initialize_weights(self):
        """Initialize weights"""
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='relu')
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        # x shape: (batch_size, seq_len, input_size)
        batch_size, seq_len, input_size = x.shape
        
        # Reshape for CNN: (batch_size, channels, seq_len)
        x = x.transpose(1, 2).unsqueeze(1)  # Add channel dimension
        
        # CNN layers
        for layer in self.cnn_layers:
            x = layer(x)
        
        # Reshape for LSTM: (batch_size, seq_len, features)
        x = x.transpose(1, 2)
        
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)
        x = lstm_out[:, -1, :]
        
        # Feed-forward layers
        for layer in self.fc_layers:
            x = layer(x)
        
        x = self.output_layer(x)
        return x


class PyTorchModelFactory:
    """
    PyTorch Model Factory.
    
    Creates and manages PyTorch models with various architectures.
    
    Usage:
        factory = PyTorchModelFactory()
        
        # Create model
        config = ModelConfig(
            architecture_type=ArchitectureType.FEED_FORWARD,
            input_size=100,
            hidden_sizes=[64, 32, 16],
            output_size=1,
            dropout_rate=0.2,
            use_batch_norm=True
        )
        
        model = factory.create_model(config)
        
        # Test architecture
        performance = factory.test_architecture(config, train_data, val_data)
    """
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize PyTorch model factory.
        
        Args:
            device: Device to use ("cpu", "cuda", or None for auto)
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        logger.info("pytorch_model_factory_initialized", device=str(self.device))
    
    def create_model(self, config: ModelConfig) -> nn.Module:
        """
        Create a PyTorch model from configuration.
        
        Args:
            config: Model configuration
        
        Returns:
            PyTorch model
        """
        if config.architecture_type == ArchitectureType.FEED_FORWARD:
            model = FeedForwardNetwork(config)
        elif config.architecture_type == ArchitectureType.LSTM:
            model = LSTMNetwork(config)
        elif config.architecture_type == ArchitectureType.HYBRID or config.architecture_type == ArchitectureType.CNN_LSTM:
            model = HybridNetwork(config)
        else:
            raise ValueError(f"Unsupported architecture type: {config.architecture_type}")
        
        model = model.to(self.device)
        
        logger.info(
            "model_created",
            architecture=config.architecture_type.value,
            input_size=config.input_size,
            output_size=config.output_size,
            device=str(self.device)
        )
        
        return model
    
    def test_architecture(
        self,
        config: ModelConfig,
        train_data: Tuple[torch.Tensor, torch.Tensor],
        val_data: Tuple[torch.Tensor, torch.Tensor],
        epochs: int = 10,
        batch_size: int = 32,
        learning_rate: float = 0.001
    ) -> Dict[str, float]:
        """
        Test architecture performance.
        
        Args:
            config: Model configuration
            train_data: Training data (X, y)
            val_data: Validation data (X, y)
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
        
        Returns:
            Performance metrics
        """
        # Create model
        model = self.create_model(config)
        
        # Create optimizer and loss function
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        # Training loop
        train_losses = []
        val_losses = []
        
        X_train, y_train = train_data
        X_val, y_val = val_data
        
        # Move to device
        X_train = X_train.to(self.device)
        y_train = y_train.to(self.device)
        X_val = X_val.to(self.device)
        y_val = y_val.to(self.device)
        
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0.0
            
            # Simple batch training (would use DataLoader in production)
            num_batches = len(X_train) // batch_size
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = start_idx + batch_size
                
                X_batch = X_train[start_idx:end_idx]
                y_batch = y_train[start_idx:end_idx]
                
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= num_batches
            train_losses.append(train_loss)
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val)
                val_loss = criterion(val_outputs, y_val).item()
                val_losses.append(val_loss)
        
        # Calculate final metrics
        final_train_loss = train_losses[-1]
        final_val_loss = val_losses[-1]
        best_val_loss = min(val_losses)
        
        metrics = {
            "final_train_loss": final_train_loss,
            "final_val_loss": final_val_loss,
            "best_val_loss": best_val_loss,
            "train_losses": train_losses,
            "val_losses": val_losses
        }
        
        logger.info(
            "architecture_tested",
            architecture=config.architecture_type.value,
            final_train_loss=final_train_loss,
            final_val_loss=final_val_loss,
            best_val_loss=best_val_loss
        )
        
        return metrics
    
    def export_to_onnx(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        output_path: str
    ) -> None:
        """
        Export model to ONNX format.
        
        Args:
            model: PyTorch model
            input_shape: Input shape (batch_size, ...)
            output_path: Output file path
        """
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(input_shape).to(self.device)
        
        # Export
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
            opset_version=11
        )
        
        logger.info("model_exported_to_onnx", output_path=output_path)
    
    def load_model(self, model_path: str, config: ModelConfig) -> nn.Module:
        """
        Load model from file.
        
        Args:
            model_path: Model file path
            config: Model configuration
        
        Returns:
            Loaded model
        """
        model = self.create_model(config)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        
        logger.info("model_loaded", model_path=model_path)
        
        return model

