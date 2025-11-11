"""
Two-Phase Split - Training vs Inference Separation

Explicit distinction between Phase 1: Training and Phase 2: Inference.
- Use frozen inference weights in live trading; don't update them mid-trade
- Mechanic handles retraining asynchronously on new data windows
- Engine only uses the inference checkpoint tagged "stable"

Author: Huracan Engine Team
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any
import structlog

logger = structlog.get_logger(__name__)


class ModelPhase(Enum):
    """Model phase: Training or Inference."""
    TRAINING = "training"
    INFERENCE = "inference"


@dataclass
class ModelCheckpoint:
    """Model checkpoint with phase and stability information."""
    checkpoint_id: str
    model_id: str
    phase: ModelPhase
    is_stable: bool
    created_at: datetime
    metrics: Dict[str, float]
    weights_path: str
    metadata: Dict[str, Any]


class TrainingInferenceSplit:
    """
    Manages separation between training and inference phases.
    
    Usage:
        split = TrainingInferenceSplit()
        
        # During training
        split.set_training_mode()
        model.train(...)
        checkpoint = split.create_checkpoint(model, metrics)
        
        # Mark as stable after validation
        split.mark_stable(checkpoint.checkpoint_id)
        
        # During inference
        stable_checkpoint = split.get_stable_checkpoint()
        model.load_weights(stable_checkpoint.weights_path)
        split.set_inference_mode()
        predictions = model.predict(...)
    """
    
    def __init__(self, checkpoint_dir: str = "checkpoints/"):
        """
        Initialize training/inference split manager.
        
        Args:
            checkpoint_dir: Directory to store checkpoints
        """
        self.checkpoint_dir = checkpoint_dir
        self.current_phase = ModelPhase.INFERENCE  # Start in inference mode
        self.current_checkpoint: Optional[ModelCheckpoint] = None
        self.checkpoints: Dict[str, ModelCheckpoint] = {}
        
        logger.info(
            "training_inference_split_initialized",
            checkpoint_dir=checkpoint_dir,
            initial_phase=self.current_phase.value
        )
    
    def set_training_mode(self) -> None:
        """Switch to training mode."""
        if self.current_phase == ModelPhase.TRAINING:
            logger.warning("already_in_training_mode")
            return
        
        self.current_phase = ModelPhase.TRAINING
        logger.info("switched_to_training_mode")
    
    def set_inference_mode(self) -> None:
        """Switch to inference mode."""
        if self.current_phase == ModelPhase.INFERENCE:
            logger.warning("already_in_inference_mode")
            return
        
        # Ensure we have a stable checkpoint before inference
        stable = self.get_stable_checkpoint()
        if stable is None:
            logger.error("no_stable_checkpoint_available")
            raise RuntimeError("Cannot switch to inference mode without stable checkpoint")
        
        self.current_phase = ModelPhase.INFERENCE
        self.current_checkpoint = stable
        logger.info(
            "switched_to_inference_mode",
            checkpoint_id=stable.checkpoint_id
        )
    
    def create_checkpoint(
        self,
        model: Any,
        metrics: Dict[str, float],
        model_id: str,
        weights_path: Optional[str] = None
    ) -> ModelCheckpoint:
        """
        Create a checkpoint from current model state.
        
        Only allowed in training mode.
        
        Args:
            model: Model object (must have save_weights method)
            metrics: Training/validation metrics
            model_id: Model identifier
            weights_path: Path to save weights (optional)
        
        Returns:
            ModelCheckpoint object
        """
        if self.current_phase != ModelPhase.TRAINING:
            raise RuntimeError("Checkpoints can only be created in training mode")
        
        checkpoint_id = f"{model_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        if weights_path is None:
            weights_path = f"{self.checkpoint_dir}/{checkpoint_id}_weights.pkl"
        
        # Save model weights
        if hasattr(model, 'save_weights'):
            model.save_weights(weights_path)
        elif hasattr(model, 'save'):
            model.save(weights_path)
        else:
            logger.warning("model_has_no_save_method", model_type=type(model).__name__)
        
        checkpoint = ModelCheckpoint(
            checkpoint_id=checkpoint_id,
            model_id=model_id,
            phase=ModelPhase.TRAINING,
            is_stable=False,  # Not stable until validated
            created_at=datetime.now(),
            metrics=metrics,
            weights_path=weights_path,
            metadata={}
        )
        
        self.checkpoints[checkpoint_id] = checkpoint
        logger.info(
            "checkpoint_created",
            checkpoint_id=checkpoint_id,
            model_id=model_id,
            metrics=metrics
        )
        
        return checkpoint
    
    def mark_stable(
        self,
        checkpoint_id: str,
        validation_metrics: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Mark a checkpoint as stable (ready for inference).
        
        Args:
            checkpoint_id: Checkpoint identifier
            validation_metrics: Optional validation metrics
        """
        if checkpoint_id not in self.checkpoints:
            raise ValueError(f"Checkpoint {checkpoint_id} not found")
        
        checkpoint = self.checkpoints[checkpoint_id]
        checkpoint.is_stable = True
        checkpoint.phase = ModelPhase.INFERENCE
        
        if validation_metrics:
            checkpoint.metrics.update(validation_metrics)
        
        logger.info(
            "checkpoint_marked_stable",
            checkpoint_id=checkpoint_id,
            validation_metrics=validation_metrics
        )
    
    def get_stable_checkpoint(self) -> Optional[ModelCheckpoint]:
        """
        Get the most recent stable checkpoint.
        
        Returns:
            Most recent stable ModelCheckpoint, or None if none available
        """
        stable_checkpoints = [
            ckpt for ckpt in self.checkpoints.values()
            if ckpt.is_stable
        ]
        
        if not stable_checkpoints:
            return None
        
        # Return most recent stable checkpoint
        return max(stable_checkpoints, key=lambda c: c.created_at)
    
    def get_current_checkpoint(self) -> Optional[ModelCheckpoint]:
        """Get current checkpoint (if in inference mode)."""
        if self.current_phase == ModelPhase.INFERENCE:
            return self.current_checkpoint
        return None
    
    def is_inference_mode(self) -> bool:
        """Check if currently in inference mode."""
        return self.current_phase == ModelPhase.INFERENCE
    
    def is_training_mode(self) -> bool:
        """Check if currently in training mode."""
        return self.current_phase == ModelPhase.TRAINING
    
    def freeze_weights(self, model: Any) -> None:
        """
        Freeze model weights for inference.
        
        Args:
            model: Model object (must have trainable attribute or similar)
        """
        if hasattr(model, 'trainable'):
            model.trainable = False
        elif hasattr(model, 'eval'):
            model.eval()  # PyTorch style
        elif hasattr(model, 'set_trainable'):
            model.set_trainable(False)
        
        logger.info("model_weights_frozen")
    
    def unfreeze_weights(self, model: Any) -> None:
        """
        Unfreeze model weights for training.
        
        Args:
            model: Model object
        """
        if hasattr(model, 'trainable'):
            model.trainable = True
        elif hasattr(model, 'train'):
            model.train()  # PyTorch style
        elif hasattr(model, 'set_trainable'):
            model.set_trainable(True)
        
        logger.info("model_weights_unfrozen")

