"""
Neural Network Models

Implements LSTM and GRU models for time-series forecasting using PyTorch.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd
import structlog
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from .base import BaseModel, ModelConfig, ModelMetrics

logger = structlog.get_logger(__name__)

# Check if PyTorch is available
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.warning("pytorch_not_available", message="PyTorch not installed - LSTM/GRU models will not work")


class LSTMModel(BaseModel):
    """LSTM neural network for time-series forecasting."""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        if not HAS_TORCH:
            raise ImportError("PyTorch is not installed. Install it with: pip install torch")
        
        # Default hyperparameters
        if "sequence_length" not in config.hyperparameters:
            config.hyperparameters["sequence_length"] = 60
        if "hidden_units" not in config.hyperparameters:
            config.hyperparameters["hidden_units"] = 64
        if "num_layers" not in config.hyperparameters:
            config.hyperparameters["num_layers"] = 2
        if "dropout" not in config.hyperparameters:
            config.hyperparameters["dropout"] = 0.2
        if "learning_rate" not in config.hyperparameters:
            config.hyperparameters["learning_rate"] = 0.001
        if "batch_size" not in config.hyperparameters:
            config.hyperparameters["batch_size"] = 32
        if "epochs" not in config.hyperparameters:
            config.hyperparameters["epochs"] = 50
        if "device" not in config.hyperparameters:
            config.hyperparameters["device"] = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.sequence_length = config.hyperparameters["sequence_length"]
        self.hidden_units = config.hyperparameters["hidden_units"]
        self.num_layers = config.hyperparameters["num_layers"]
        self.dropout = config.hyperparameters["dropout"]
        self.learning_rate = config.hyperparameters["learning_rate"]
        self.batch_size = config.hyperparameters["batch_size"]
        self.epochs = config.hyperparameters["epochs"]
        self.device = torch.device(config.hyperparameters["device"])
        
        self.model: Optional[nn.Module] = None
        self.input_size: Optional[int] = None
        self.output_size: int = 1  # For regression, 1 output; for classification, num_classes
    
    def _create_sequences(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM input."""
        X_seq, y_seq = [], []
        for i in range(len(X) - self.sequence_length):
            X_seq.append(X[i : i + self.sequence_length])
            y_seq.append(y[i + self.sequence_length])
        return np.array(X_seq), np.array(y_seq)
    
    def _build_model(self, input_size: int) -> nn.Module:
        """Build LSTM model."""
        class LSTMNetwork(nn.Module):
            def __init__(self, input_size, hidden_units, num_layers, dropout, output_size, model_type):
                super().__init__()
                self.model_type = model_type
                self.lstm = nn.LSTM(
                    input_size,
                    hidden_units,
                    num_layers,
                    dropout=dropout if num_layers > 1 else 0,
                    batch_first=True,
                )
                self.fc = nn.Linear(hidden_units, output_size)
                if model_type == "classification":
                    self.activation = nn.Sigmoid()
                else:
                    self.activation = nn.Identity()
            
            def forward(self, x):
                lstm_out, _ = self.lstm(x)
                # Take the last output
                last_output = lstm_out[:, -1, :]
                output = self.fc(last_output)
                return self.activation(output)
        
        output_size = 1 if self.config.model_type == "regression" else len(np.unique(self._y_train))
        return LSTMNetwork(
            input_size,
            self.hidden_units,
            self.num_layers,
            self.dropout,
            output_size,
            self.config.model_type,
        )
    
    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        X_val: Optional[pd.DataFrame | np.ndarray] = None,
        y_val: Optional[pd.Series | np.ndarray] = None,
    ) -> ModelMetrics:
        """Train LSTM model."""
        logger.info("training_lstm", samples=len(X), sequence_length=self.sequence_length)
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        
        self.input_size = X.shape[1]
        self._y_train = y
        
        # Create sequences
        X_seq, y_seq = self._create_sequences(X, y)
        
        if len(X_seq) == 0:
            raise ValueError(f"Not enough data to create sequences. Need at least {self.sequence_length + 1} samples")
        
        # Build model
        self.model = self._build_model(self.input_size).to(self.device)
        criterion = nn.MSELoss() if self.config.model_type == "regression" else nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Learning rate scheduler (if configured)
        self.scheduler = None
        if "scheduler" in self.config.hyperparameters:
            from .scheduler import create_scheduler
            scheduler_config = self.config.hyperparameters["scheduler"]
            self.scheduler = create_scheduler(optimizer, scheduler_config)
        
        # Create data loader
        dataset = TensorDataset(
            torch.FloatTensor(X_seq),
            torch.FloatTensor(y_seq).reshape(-1, 1),
        )
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            # Step scheduler
            if self.scheduler is not None:
                self.scheduler.step()
            
            if (epoch + 1) % 10 == 0:
                if self.scheduler is not None:
                    try:
                        current_lr = self.scheduler.get_lr()
                    except:
                        current_lr = self.learning_rate
                else:
                    current_lr = self.learning_rate
                logger.info(
                    "lstm_training_epoch",
                    epoch=epoch + 1,
                    loss=total_loss / len(dataloader),
                    learning_rate=current_lr,
                )
        
        self.is_trained = True
        
        # Evaluate
        if X_val is not None and y_val is not None:
            metrics = self.evaluate(X_val, y_val)
        else:
            # Use last portion of training data for evaluation
            split_idx = int(len(X_seq) * 0.8)
            metrics = self.evaluate(X[split_idx:], y[split_idx:])
        
        self.metrics = metrics
        logger.info("lstm_trained", **metrics.to_dict())
        return metrics
    
    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Create sequences
        X_seq, _ = self._create_sequences(X, np.zeros(len(X)))
        
        if len(X_seq) == 0:
            # If not enough data, return zeros
            return np.zeros(len(X))
        
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for i in range(len(X_seq)):
                seq = torch.FloatTensor(X_seq[i : i + 1]).to(self.device)
                pred = self.model(seq)
                predictions.append(pred.cpu().numpy()[0, 0])
        
        # Pad with zeros for the first sequence_length samples
        predictions = np.array([0.0] * self.sequence_length + predictions)
        
        return predictions[: len(X)]
    
    def evaluate(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> ModelMetrics:
        """Evaluate model."""
        y_pred = self.predict(X)
        
        if isinstance(y, pd.Series):
            y = y.values
        
        # Align predictions with targets (predictions are shorter due to sequence requirement)
        if len(y_pred) != len(y):
            min_len = min(len(y_pred), len(y))
            y_pred = y_pred[-min_len:]
            y = y[-min_len:]
        
        if self.config.model_type == "regression":
            return self._calculate_regression_metrics(y, y_pred)
        else:
            # For classification, convert predictions to binary
            y_pred_binary = (y_pred > 0.5).astype(int)
            return self._calculate_classification_metrics(y, y_pred_binary, y_pred)
    
    def save(self, path: Path) -> None:
        """Save model."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "input_size": self.input_size,
                "sequence_length": self.sequence_length,
                "hidden_units": self.hidden_units,
                "num_layers": self.num_layers,
                "dropout": self.dropout,
                "model_type": self.config.model_type,
            },
            path,
        )
        logger.info("lstm_saved", path=str(path))
    
    def load(self, path: Path) -> None:
        """Load model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.input_size = checkpoint["input_size"]
        self.sequence_length = checkpoint["sequence_length"]
        self.hidden_units = checkpoint["hidden_units"]
        self.num_layers = checkpoint["num_layers"]
        self.dropout = checkpoint["dropout"]
        self.model = self._build_model(self.input_size).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.is_trained = True
        logger.info("lstm_loaded", path=str(path))


class GRUModel(LSTMModel):
    """GRU neural network for time-series forecasting."""
    
    def _build_model(self, input_size: int) -> nn.Module:
        """Build GRU model."""
        class GRUNetwork(nn.Module):
            def __init__(self, input_size, hidden_units, num_layers, dropout, output_size, model_type):
                super().__init__()
                self.model_type = model_type
                self.gru = nn.GRU(
                    input_size,
                    hidden_units,
                    num_layers,
                    dropout=dropout if num_layers > 1 else 0,
                    batch_first=True,
                )
                self.fc = nn.Linear(hidden_units, output_size)
                if model_type == "classification":
                    self.activation = nn.Sigmoid()
                else:
                    self.activation = nn.Identity()
            
            def forward(self, x):
                gru_out, _ = self.gru(x)
                last_output = gru_out[:, -1, :]
                output = self.fc(last_output)
                return self.activation(output)
        
        output_size = 1 if self.config.model_type == "regression" else len(np.unique(self._y_train))
        return GRUNetwork(
            input_size,
            self.hidden_units,
            self.num_layers,
            self.dropout,
            output_size,
            self.config.model_type,
        )
    
    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        X_val: Optional[pd.DataFrame | np.ndarray] = None,
        y_val: Optional[pd.Series | np.ndarray] = None,
    ) -> ModelMetrics:
        """Train GRU model."""
        logger.info("training_gru", samples=len(X), sequence_length=self.sequence_length)
        return super().fit(X, y, X_val, y_val)
    
    def save(self, path: Path) -> None:
        """Save model."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "input_size": self.input_size,
                "sequence_length": self.sequence_length,
                "hidden_units": self.hidden_units,
                "num_layers": self.num_layers,
                "dropout": self.dropout,
                "model_type": self.config.model_type,
            },
            path,
        )
        logger.info("gru_saved", path=str(path))
    
    def load(self, path: Path) -> None:
        """Load model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.input_size = checkpoint["input_size"]
        self.sequence_length = checkpoint["sequence_length"]
        self.hidden_units = checkpoint["hidden_units"]
        self.num_layers = checkpoint["num_layers"]
        self.dropout = checkpoint["dropout"]
        self.model = self._build_model(self.input_size).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.is_trained = True
        logger.info("gru_loaded", path=str(path))

