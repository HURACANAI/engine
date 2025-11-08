"""
GPU Handler - Hardware Acceleration

Handles GPU detection, device management, and distributed training.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import structlog
import torch

logger = structlog.get_logger(__name__)

# Check if PyTorch is available
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.warning("pytorch_not_available_for_gpu_handler")


class GPUHandler:
    """
    GPU handler for hardware acceleration.
    
    Features:
    - Automatic GPU detection
    - Device management
    - Multi-GPU support
    - Memory management
    - Fallback to CPU
    """
    
    def __init__(self, device: Optional[str] = None, use_multi_gpu: bool = False):
        """
        Initialize GPU handler.
        
        Args:
            device: Device to use ("cuda", "cpu", or None for auto-detection)
            use_multi_gpu: Whether to use multiple GPUs
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for GPU handling")
        
        self.device = self._get_device(device)
        self.use_multi_gpu = use_multi_gpu and torch.cuda.device_count() > 1
        self.num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        
        logger.info(
            "gpu_handler_initialized",
            device=str(self.device),
            use_multi_gpu=self.use_multi_gpu,
            num_gpus=self.num_gpus,
        )
    
    def _get_device(self, device: Optional[str]) -> torch.device:
        """Get device (auto-detect if not specified)."""
        if device is None:
            # Auto-detect
            if torch.cuda.is_available():
                device = "cuda"
                logger.info("gpu_auto_detected", device="cuda", device_name=torch.cuda.get_device_name(0))
            else:
                device = "cpu"
                logger.info("gpu_not_available_using_cpu")
        elif device == "cuda" and not torch.cuda.is_available():
            logger.warning("cuda_requested_but_not_available_falling_back_to_cpu")
            device = "cpu"
        
        return torch.device(device)
    
    def to_device(self, tensor: Any) -> Any:
        """Move tensor to device."""
        if isinstance(tensor, torch.Tensor):
            return tensor.to(self.device)
        elif isinstance(tensor, (list, tuple)):
            return [self.to_device(t) for t in tensor]
        elif isinstance(tensor, dict):
            return {k: self.to_device(v) for k, v in tensor.items()}
        else:
            return tensor
    
    def get_device(self) -> torch.device:
        """Get current device."""
        return self.device
    
    def is_gpu(self) -> bool:
        """Check if using GPU."""
        return self.device.type == "cuda"
    
    def get_gpu_info(self) -> Dict[str, Any]:
        """Get GPU information."""
        if not self.is_gpu():
            return {"available": False}
        
        info = {
            "available": True,
            "device_count": torch.cuda.device_count(),
            "current_device": torch.cuda.current_device(),
            "device_name": torch.cuda.get_device_name(0),
            "memory_allocated": torch.cuda.memory_allocated(0) / 1024**3,  # GB
            "memory_reserved": torch.cuda.memory_reserved(0) / 1024**3,  # GB
            "max_memory_allocated": torch.cuda.max_memory_allocated(0) / 1024**3,  # GB
        }
        
        return info
    
    def clear_cache(self) -> None:
        """Clear GPU cache."""
        if self.is_gpu():
            torch.cuda.empty_cache()
            logger.debug("gpu_cache_cleared")
    
    def synchronize(self) -> None:
        """Synchronize GPU operations."""
        if self.is_gpu():
            torch.cuda.synchronize()
    
    def wrap_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """
        Wrap model for device and multi-GPU.
        
        Args:
            model: Neural network model
            
        Returns:
            Wrapped model
        """
        # Move to device
        model = model.to(self.device)
        
        # Wrap for multi-GPU if enabled
        if self.use_multi_gpu and self.num_gpus > 1:
            model = torch.nn.DataParallel(model)
            logger.info("model_wrapped_for_multi_gpu", num_gpus=self.num_gpus)
        
        return model


class DistributedTrainer:
    """Distributed training support (for future use)."""
    
    def __init__(self, backend: str = "nccl"):
        """
        Initialize distributed trainer.
        
        Args:
            backend: Distributed backend ("nccl" for GPU, "gloo" for CPU)
        """
        self.backend = backend
        logger.info("distributed_trainer_initialized", backend=backend)
    
    def setup(self, rank: int, world_size: int) -> None:
        """Setup distributed training."""
        torch.distributed.init_process_group(
            backend=self.backend,
            rank=rank,
            world_size=world_size,
        )
        logger.info("distributed_training_setup", rank=rank, world_size=world_size)
    
    def cleanup(self) -> None:
        """Cleanup distributed training."""
        torch.distributed.destroy_process_group()
        logger.info("distributed_training_cleanup")

