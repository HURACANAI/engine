"""
Distributed Training - Multi-GPU and Multi-Node Training

For large-scale backtesting and live inference across multiple GPUs.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import structlog
import torch
import torch.distributed as dist
import torch.nn as nn

logger = structlog.get_logger(__name__)


class DistributedTrainer:
    """
    Distributed trainer for multi-GPU and multi-node training.
    
    Features:
    - Multi-GPU training
    - Multi-node training
    - Model parallelism
    - Data parallelism
    - Gradient synchronization
    """
    
    def __init__(
        self,
        model: nn.Module,
        backend: str = "nccl",
        world_size: Optional[int] = None,
        rank: Optional[int] = None,
    ):
        """
        Initialize distributed trainer.
        
        Args:
            model: Neural network model
            backend: Distributed backend ("nccl" for GPU, "gloo" for CPU)
            world_size: Number of processes
            rank: Process rank
        """
        self.model = model
        self.backend = backend
        self.world_size = world_size
        self.rank = rank
        
        self.is_distributed = False
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            self.is_distributed = True
        
        logger.info(
            "distributed_trainer_initialized",
            backend=backend,
            is_distributed=self.is_distributed,
        )
    
    def setup_distributed(self, rank: int, world_size: int) -> None:
        """
        Setup distributed training.
        
        Args:
            rank: Process rank
            world_size: Number of processes
        """
        self.rank = rank
        self.world_size = world_size
        
        # Initialize process group
        dist.init_process_group(
            backend=self.backend,
            rank=rank,
            world_size=world_size,
        )
        
        # Set device
        device = torch.device(f"cuda:{rank}")
        self.model = self.model.to(device)
        self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[rank])
        
        logger.info("distributed_training_setup", rank=rank, world_size=world_size)
    
    def cleanup_distributed(self) -> None:
        """Cleanup distributed training."""
        if dist.is_initialized():
            dist.destroy_process_group()
            logger.info("distributed_training_cleanup")
    
    def train_distributed(
        self,
        train_loader: Any,
        optimizer: Any,
        loss_fn: Any,
        epochs: int = 10,
    ) -> List[float]:
        """
        Train model in distributed fashion.
        
        Args:
            train_loader: Distributed data loader
            optimizer: Optimizer
            loss_fn: Loss function
            epochs: Number of epochs
            
        Returns:
            List of losses
        """
        logger.info("starting_distributed_training", epochs=epochs)
        
        losses = []
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                # Forward pass
                outputs = self.model(inputs)
                loss = loss_fn(outputs, targets)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
            losses.append(avg_loss)
            
            if self.rank == 0:  # Only log from rank 0
                logger.info("distributed_training_epoch", epoch=epoch + 1, loss=avg_loss)
        
        logger.info("distributed_training_complete")
        return losses
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information for distributed setup."""
        return {
            "is_distributed": self.is_distributed,
            "world_size": self.world_size,
            "rank": self.rank,
            "backend": self.backend,
        }

