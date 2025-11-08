"""
Trainer - Main Training Orchestrator

Coordinates training loop: data → training → inference → feedback
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ..base import ModelMetrics
from .backpropagation import Backpropagation
from .gpu_handler import GPUHandler
from .optimizers import OptimizerFactory, OptimizerWrapper

logger = structlog.get_logger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for training."""
    
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 0.001
    optimizer: str = "adam"
    loss_fn: str = "mse"  # "mse", "ce", "bce"
    device: Optional[str] = None
    use_gpu: bool = True
    clip_grad_norm: Optional[float] = None
    accumulation_steps: int = 1
    save_checkpoints: bool = True
    checkpoint_dir: Path = Path("checkpoints")
    log_interval: int = 10
    eval_interval: int = 1


class Trainer:
    """
    Main training orchestrator.
    
    Handles:
    - Data loading and batching
    - Forward/backward passes
    - Optimizer updates
    - Checkpoint saving
    - Evaluation
    - Logging
    - GPU management
    """
    
    def __init__(self, model: nn.Module, config: TrainingConfig):
        """
        Initialize trainer.
        
        Args:
            model: Neural network model
            config: Training configuration
        """
        self.model = model
        self.config = config
        
        # GPU handler
        self.gpu_handler = GPUHandler(device=config.device, use_multi_gpu=False)
        self.model = self.gpu_handler.wrap_model(model)
        
        # Loss function
        self.loss_fn = self._get_loss_fn(config.loss_fn)
        
        # Optimizer
        self.optimizer = OptimizerWrapper(
            model=self.model,
            optimizer_type=config.optimizer,
            learning_rate=config.learning_rate,
        )
        
        # Backpropagation
        self.backprop = Backpropagation(
            model=self.model,
            loss_fn=self.loss_fn,
            clip_grad_norm=config.clip_grad_norm,
            accumulation_steps=config.accumulation_steps,
        )
        
        # Training state
        self.current_epoch = 0
        self.training_history: List[Dict[str, float]] = []
        
        # Create checkpoint directory
        if config.save_checkpoints:
            config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(
            "trainer_initialized",
            epochs=config.epochs,
            batch_size=config.batch_size,
            device=str(self.gpu_handler.get_device()),
        )
    
    def _get_loss_fn(self, loss_fn_name: str) -> nn.Module:
        """Get loss function by name."""
        loss_map = {
            "mse": nn.MSELoss(),
            "ce": nn.CrossEntropyLoss(),
            "bce": nn.BCELoss(),
            "bce_with_logits": nn.BCEWithLogitsLoss(),
            "mae": nn.L1Loss(),
            "huber": nn.HuberLoss(),
        }
        return loss_map.get(loss_fn_name.lower(), nn.MSELoss())
    
    def train_epoch(
        self,
        dataloader: DataLoader,
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            dataloader: Data loader for training data
            
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            # Move to device
            inputs = self.gpu_handler.to_device(inputs)
            targets = self.gpu_handler.to_device(targets)
            
            # Forward and backward pass
            metrics = self.backprop.forward_backward(inputs, targets, self.optimizer.optimizer)
            
            total_loss += metrics["loss"]
            num_batches += 1
            
            # Log progress
            if batch_idx % self.config.log_interval == 0:
                logger.info(
                    "training_batch",
                    epoch=self.current_epoch,
                    batch=batch_idx,
                    loss=metrics["loss"],
                    grad_norm=metrics.get("grad_norm", 0.0),
                )
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        return {
            "loss": avg_loss,
            "num_batches": num_batches,
        }
    
    def evaluate(
        self,
        dataloader: DataLoader,
    ) -> Dict[str, float]:
        """
        Evaluate model.
        
        Args:
            dataloader: Data loader for evaluation data
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = self.gpu_handler.to_device(inputs)
                targets = self.gpu_handler.to_device(targets)
                
                # Forward pass
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        return {
            "loss": avg_loss,
            "num_batches": num_batches,
        }
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ) -> List[Dict[str, float]]:
        """
        Train model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            
        Returns:
            Training history
        """
        logger.info("training_started", epochs=self.config.epochs)
        
        best_val_loss = float("inf")
        
        for epoch in range(self.config.epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            
            # Evaluate
            val_metrics = None
            if val_loader is not None and (epoch + 1) % self.config.eval_interval == 0:
                val_metrics = self.evaluate(val_loader)
                
                # Save best model
                if val_metrics["loss"] < best_val_loss:
                    best_val_loss = val_metrics["loss"]
                    if self.config.save_checkpoints:
                        self.save_checkpoint(epoch, is_best=True)
            
            # Log epoch
            epoch_log = {
                "epoch": epoch + 1,
                "train_loss": train_metrics["loss"],
            }
            if val_metrics:
                epoch_log["val_loss"] = val_metrics["loss"]
            
            self.training_history.append(epoch_log)
            logger.info("training_epoch_complete", **epoch_log)
            
            # Save checkpoint
            if self.config.save_checkpoints and (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch, is_best=False)
        
        logger.info("training_complete")
        return self.training_history
    
    def save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """
        Save training checkpoint.
        
        Args:
            epoch: Current epoch
            is_best: Whether this is the best model
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "training_history": self.training_history,
        }
        
        # Save regular checkpoint
        checkpoint_path = self.config.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.config.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info("best_model_saved", epoch=epoch, path=str(best_path))
    
    def load_checkpoint(self, checkpoint_path: Path) -> None:
        """
        Load training checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.gpu_handler.get_device())
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.training_history = checkpoint.get("training_history", [])
        
        logger.info("checkpoint_loaded", path=str(checkpoint_path), epoch=self.current_epoch)

