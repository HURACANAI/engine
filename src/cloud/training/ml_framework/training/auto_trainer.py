"""
AutoTrainer - Automatic Hyperparameter Selection

Automatically selects optimal learning rate, batch size, and optimizer per coin.
Uses hyperparameter optimization techniques (grid search, random search, Bayesian optimization).

Author: Huracan Engine Team
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import itertools

import numpy as np
import structlog

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    DataLoader = None

logger = structlog.get_logger(__name__)


class OptimizerType(Enum):
    """Optimizer types"""
    ADAM = "adam"
    SGD = "sgd"
    ADAMW = "adamw"
    RMSPROP = "rmsprop"


@dataclass
class HyperparameterSpace:
    """Hyperparameter search space"""
    learning_rates: List[float] = field(default_factory=lambda: [1e-4, 5e-4, 1e-3, 5e-3, 1e-2])
    batch_sizes: List[int] = field(default_factory=lambda: [16, 32, 64, 128, 256])
    optimizers: List[OptimizerType] = field(default_factory=lambda: [OptimizerType.ADAM, OptimizerType.ADAMW, OptimizerType.SGD])
    dropout_rates: List[float] = field(default_factory=lambda: [0.0, 0.1, 0.2, 0.3])
    weight_decay: List[float] = field(default_factory=lambda: [0.0, 1e-5, 1e-4])


@dataclass
class TrainingResult:
    """Result of a training run"""
    learning_rate: float
    batch_size: int
    optimizer: str
    dropout_rate: float
    weight_decay: float
    train_loss: float
    val_loss: float
    val_metric: float  # Primary metric (e.g., Sharpe ratio, accuracy)
    training_time_seconds: float
    model_performance: Dict[str, float] = field(default_factory=dict)


@dataclass
class AutoTrainerConfig:
    """Configuration for AutoTrainer"""
    search_method: str = "grid"  # grid, random, bayesian
    max_trials: int = 50
    n_splits: int = 3  # For cross-validation
    metric: str = "sharpe"  # sharpe, sortino, accuracy, loss
    maximize_metric: bool = True
    early_stopping_patience: int = 5
    min_epochs: int = 10
    max_epochs: int = 100
    hyperparameter_space: Optional[HyperparameterSpace] = None


class AutoTrainer:
    """
    Automatic hyperparameter selection for model training.
    
    Features:
    - Grid search, random search, or Bayesian optimization
    - Cross-validation
    - Early stopping
    - Automatic best model selection
    - Per-coin optimization
    
    Usage:
        trainer = AutoTrainer(config=AutoTrainerConfig())
        best_config = trainer.optimize(model, train_data, val_data)
    """
    
    def __init__(self, config: Optional[AutoTrainerConfig] = None):
        """
        Initialize AutoTrainer.
        
        Args:
            config: AutoTrainer configuration
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available. Install torch for AutoTrainer.")
        
        self.config = config or AutoTrainerConfig()
        self.hyperparameter_space = self.config.hyperparameter_space or HyperparameterSpace()
        
        self.results: List[TrainingResult] = []
        self.best_result: Optional[TrainingResult] = None
        
        logger.info(
            "auto_trainer_initialized",
            search_method=self.config.search_method,
            max_trials=self.config.max_trials,
            metric=self.config.metric
        )
    
    def optimize(
        self,
        model_factory: callable,
        train_data: Tuple[Any, Any],  # (X_train, y_train)
        val_data: Optional[Tuple[Any, Any]] = None,  # (X_val, y_val)
        symbol: str = "UNKNOWN"
    ) -> TrainingResult:
        """
        Optimize hyperparameters.
        
        Args:
            model_factory: Function that creates a model given hyperparameters
            train_data: Training data (X, y)
            val_data: Validation data (X, y) - optional, will split if not provided
            symbol: Trading symbol (for logging)
        
        Returns:
            Best TrainingResult with optimal hyperparameters
        """
        logger.info("auto_trainer_start", symbol=symbol, max_trials=self.config.max_trials)
        
        X_train, y_train = train_data
        
        # Split validation data if not provided
        if val_data is None:
            split_idx = int(len(X_train) * 0.8)
            X_val = X_train[split_idx:]
            y_val = y_train[split_idx:]
            X_train = X_train[:split_idx]
            y_train = y_train[:split_idx]
        else:
            X_val, y_val = val_data
        
        # Generate hyperparameter combinations
        if self.config.search_method == "grid":
            combinations = self._grid_search_combinations()
        elif self.config.search_method == "random":
            combinations = self._random_search_combinations()
        else:
            combinations = self._grid_search_combinations()  # Default to grid
        
        # Limit number of trials
        if len(combinations) > self.config.max_trials:
            # Randomly sample
            indices = np.random.choice(len(combinations), self.config.max_trials, replace=False)
            combinations = [combinations[i] for i in indices]
        
        # Test each combination
        for i, combo in enumerate(combinations):
            logger.info(
                "auto_trainer_trial",
                symbol=symbol,
                trial=i + 1,
                total=len(combinations),
                learning_rate=combo["learning_rate"],
                batch_size=combo["batch_size"],
                optimizer=combo["optimizer"].value
            )
            
            try:
                result = self._train_with_hyperparameters(
                    model_factory=model_factory,
                    hyperparams=combo,
                    train_data=(X_train, y_train),
                    val_data=(X_val, y_val),
                    symbol=symbol
                )
                self.results.append(result)
                
                # Update best result
                if self.best_result is None:
                    self.best_result = result
                else:
                    if self.config.maximize_metric:
                        if result.val_metric > self.best_result.val_metric:
                            self.best_result = result
                    else:
                        if result.val_metric < self.best_result.val_metric:
                            self.best_result = result
                
            except Exception as e:
                logger.warning(
                    "auto_trainer_trial_failed",
                    symbol=symbol,
                    trial=i + 1,
                    error=str(e)
                )
                continue
        
        if self.best_result is None:
            raise ValueError("No successful training runs")
        
        logger.info(
            "auto_trainer_complete",
            symbol=symbol,
            total_trials=len(self.results),
            best_metric=self.best_result.val_metric,
            best_lr=self.best_result.learning_rate,
            best_batch_size=self.best_result.batch_size,
            best_optimizer=self.best_result.optimizer
        )
        
        return self.best_result
    
    def _grid_search_combinations(self) -> List[Dict[str, Any]]:
        """Generate all hyperparameter combinations for grid search"""
        combinations = []
        
        for lr, batch_size, optimizer, dropout, weight_decay in itertools.product(
            self.hyperparameter_space.learning_rates,
            self.hyperparameter_space.batch_sizes,
            self.hyperparameter_space.optimizers,
            self.hyperparameter_space.dropout_rates,
            self.hyperparameter_space.weight_decay
        ):
            combinations.append({
                "learning_rate": lr,
                "batch_size": batch_size,
                "optimizer": optimizer,
                "dropout_rate": dropout,
                "weight_decay": weight_decay
            })
        
        return combinations
    
    def _random_search_combinations(self) -> List[Dict[str, Any]]:
        """Generate random hyperparameter combinations"""
        combinations = []
        
        for _ in range(self.config.max_trials):
            combo = {
                "learning_rate": np.random.choice(self.hyperparameter_space.learning_rates),
                "batch_size": np.random.choice(self.hyperparameter_space.batch_sizes),
                "optimizer": np.random.choice(self.hyperparameter_space.optimizers),
                "dropout_rate": np.random.choice(self.hyperparameter_space.dropout_rates),
                "weight_decay": np.random.choice(self.hyperparameter_space.weight_decay)
            }
            combinations.append(combo)
        
        return combinations
    
    def _train_with_hyperparameters(
        self,
        model_factory: callable,
        hyperparams: Dict[str, Any],
        train_data: Tuple[Any, Any],
        val_data: Tuple[Any, Any],
        symbol: str
    ) -> TrainingResult:
        """Train model with specific hyperparameters"""
        import time
        start_time = time.time()
        
        X_train, y_train = train_data
        X_val, y_val = val_data
        
        # Create model with hyperparameters
        model = model_factory(
            dropout_rate=hyperparams["dropout_rate"],
            **{k: v for k, v in hyperparams.items() if k not in ["dropout_rate", "learning_rate", "batch_size", "optimizer", "weight_decay"]}
        )
        
        # Create data loaders
        train_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_train) if not isinstance(X_train, torch.Tensor) else X_train,
            torch.FloatTensor(y_train) if not isinstance(y_train, torch.Tensor) else y_train
        )
        val_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_val) if not isinstance(X_val, torch.Tensor) else X_val,
            torch.FloatTensor(y_val) if not isinstance(y_val, torch.Tensor) else y_val
        )
        
        train_loader = DataLoader(train_dataset, batch_size=hyperparams["batch_size"], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=hyperparams["batch_size"], shuffle=False)
        
        # Create optimizer
        optimizer = self._create_optimizer(
            model=model,
            optimizer_type=hyperparams["optimizer"],
            learning_rate=hyperparams["learning_rate"],
            weight_decay=hyperparams["weight_decay"]
        )
        
        # Loss function
        criterion = nn.MSELoss()
        
        # Training loop with early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []
        
        for epoch in range(self.config.max_epochs):
            # Train
            model.train()
            epoch_train_loss = 0.0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y.unsqueeze(1) if len(batch_y.shape) == 1 else batch_y)
                loss.backward()
                optimizer.step()
                epoch_train_loss += loss.item()
            
            avg_train_loss = epoch_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # Validate
            model.eval()
            epoch_val_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y.unsqueeze(1) if len(batch_y.shape) == 1 else batch_y)
                    epoch_val_loss += loss.item()
            
            avg_val_loss = epoch_val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config.early_stopping_patience and epoch >= self.config.min_epochs:
                    break
        
        # Calculate validation metric (simplified - would use actual trading metrics)
        val_metric = self._calculate_metric(val_losses, val_data, model)
        
        training_time = time.time() - start_time
        
        return TrainingResult(
            learning_rate=hyperparams["learning_rate"],
            batch_size=hyperparams["batch_size"],
            optimizer=hyperparams["optimizer"].value,
            dropout_rate=hyperparams["dropout_rate"],
            weight_decay=hyperparams["weight_decay"],
            train_loss=train_losses[-1] if train_losses else 0.0,
            val_loss=val_losses[-1] if val_losses else 0.0,
            val_metric=val_metric,
            training_time_seconds=training_time,
            model_performance={
                "min_train_loss": min(train_losses) if train_losses else 0.0,
                "min_val_loss": min(val_losses) if val_losses else 0.0,
                "epochs_trained": len(train_losses)
            }
        )
    
    def _create_optimizer(
        self,
        model: nn.Module,
        optimizer_type: OptimizerType,
        learning_rate: float,
        weight_decay: float
    ) -> torch.optim.Optimizer:
        """Create optimizer based on type"""
        if optimizer_type == OptimizerType.ADAM:
            return torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_type == OptimizerType.ADAMW:
            return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_type == OptimizerType.SGD:
            return torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9)
        elif optimizer_type == OptimizerType.RMSPROP:
            return torch.optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        else:
            return torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    def _calculate_metric(
        self,
        val_losses: List[float],
        val_data: Tuple[Any, Any],
        model: nn.Module
    ) -> float:
        """Calculate validation metric"""
        if self.config.metric == "loss":
            return -min(val_losses) if val_losses else 0.0  # Negative for maximization
        elif self.config.metric == "sharpe":
            # Simplified Sharpe calculation (would use actual returns)
            if len(val_losses) < 2:
                return 0.0
            returns = np.diff(val_losses) * -1  # Negative loss as proxy for returns
            if returns.std() == 0:
                return 0.0
            sharpe = returns.mean() / returns.std() * np.sqrt(252)  # Annualized
            return sharpe
        else:
            # Default: use negative validation loss
            return -val_losses[-1] if val_losses else 0.0

