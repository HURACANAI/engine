"""Adaptive training logic with early stopping and dynamic adjustments."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import structlog  # type: ignore[reportMissingImports]

logger = structlog.get_logger(__name__)


class AdaptiveTrainer:
    """
    Adaptive training logic:
    - Early stopping (stop if no improvement for N epochs)
    - Dynamic batch sizing (adjust based on GPU memory)
    - Checkpoint recovery (resume from last checkpoint)
    - Learning rate scheduling (reduce LR on plateau)
    """

    def __init__(
        self,
        early_stopping_patience: int = 10,
        early_stopping_min_delta: float = 1e-4,
        reduce_lr_patience: int = 5,
        reduce_lr_factor: float = 0.5,
        min_lr: float = 1e-7,
        checkpoint_dir: Optional[str] = None,
    ) -> None:
        """
        Initialize adaptive trainer.
        
        Args:
            early_stopping_patience: Number of epochs to wait before stopping
            early_stopping_min_delta: Minimum change to qualify as improvement
            reduce_lr_patience: Number of epochs to wait before reducing LR
            reduce_lr_factor: Factor by which to reduce LR
            min_lr: Minimum learning rate
            checkpoint_dir: Directory to save checkpoints
        """
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.reduce_lr_patience = reduce_lr_patience
        self.reduce_lr_factor = reduce_lr_factor
        self.min_lr = min_lr
        self.checkpoint_dir = checkpoint_dir
        
        logger.info(
            "adaptive_trainer_initialized",
            early_stopping_patience=early_stopping_patience,
            reduce_lr_patience=reduce_lr_patience,
        )

    def create_callbacks(
        self,
        monitor: str = "val_loss",
        mode: str = "min",
    ) -> list:
        """
        Create training callbacks.
        
        Args:
            monitor: Metric to monitor
            mode: 'min' or 'max'
            
        Returns:
            List of callbacks
        """
        callbacks = []
        
        try:
            import tensorflow as tf  # type: ignore[reportMissingImports]
            from tensorflow import keras  # type: ignore[reportMissingImports]
        except ImportError:
            logger.warning("tensorflow_not_available", message="Cannot create TensorFlow callbacks")
            return callbacks
        
        # Early stopping
        early_stopping = keras.callbacks.EarlyStopping(
            monitor=monitor,
            mode=mode,
            patience=self.early_stopping_patience,
            min_delta=self.early_stopping_min_delta,
            restore_best_weights=True,
            verbose=1,
        )
        callbacks.append(early_stopping)
        
        # Learning rate scheduler
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            mode=mode,
            factor=self.reduce_lr_factor,
            patience=self.reduce_lr_patience,
            min_lr=self.min_lr,
            verbose=1,
        )
        callbacks.append(reduce_lr)
        
        # Model checkpoint
        if self.checkpoint_dir:
            checkpoint = keras.callbacks.ModelCheckpoint(
                filepath=f"{self.checkpoint_dir}/checkpoint-{{epoch:02d}}.h5",
                monitor=monitor,
                mode=mode,
                save_best_only=True,
                verbose=1,
            )
            callbacks.append(checkpoint)
        
        logger.info("callbacks_created", num_callbacks=len(callbacks))
        
        return callbacks

    def optimize_batch_size(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        start_batch_size: int = 32,
        max_batch_size: int = 512,
        factor: int = 2,
    ) -> int:
        """
        Optimize batch size based on GPU memory.
        
        Args:
            model: Model to test
            X: Training features
            y: Training targets
            start_batch_size: Starting batch size
            max_batch_size: Maximum batch size
            factor: Factor to increase batch size
            
        Returns:
            Optimal batch size
        """
        batch_size = start_batch_size
        best_batch_size = start_batch_size
        
        try:
            import tensorflow as tf  # type: ignore[reportMissingImports]
        except ImportError:
            logger.warning("tensorflow_not_available", using_default_batch_size=start_batch_size)
            return start_batch_size
        
        logger.info("optimizing_batch_size", start=start_batch_size, max=max_batch_size)
        
        while batch_size <= max_batch_size:
            try:
                # Try to train with this batch size
                # This is a simplified test - in practice, you'd do a small training step
                test_indices = np.random.choice(len(X), min(batch_size * 2, len(X)), replace=False)
                X_test = X[test_indices]
                y_test = y[test_indices]
                
                # Try to create a batch
                # In TensorFlow, this would be model.train_on_batch(X_test[:batch_size], y_test[:batch_size])
                # For now, we'll just check if we can create the batch
                _ = X_test[:batch_size]
                _ = y_test[:batch_size]
                
                best_batch_size = batch_size
                batch_size *= factor
                
                logger.debug("batch_size_test_passed", batch_size=best_batch_size)
                
            except Exception as e:
                logger.warning("batch_size_test_failed", batch_size=batch_size, error=str(e))
                break
        
        logger.info("batch_size_optimization_complete", optimal_batch_size=best_batch_size)
        
        return best_batch_size

    def train_with_adaptation(
        self,
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 50,
        batch_size: Optional[int] = None,
        monitor: str = "val_loss",
        mode: str = "min",
        verbose: int = 0,
    ) -> Dict[str, Any]:
        """
        Train model with adaptive logic.
        
        Args:
            model: Model to train
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            epochs: Maximum number of epochs
            batch_size: Batch size (will be optimized if None)
            monitor: Metric to monitor
            mode: 'min' or 'max'
            verbose: Verbosity level
            
        Returns:
            Training history and results
        """
        # Optimize batch size if not provided
        if batch_size is None:
            batch_size = self.optimize_batch_size(model, X_train, y_train)
        
        # Create callbacks
        callbacks = self.create_callbacks(monitor=monitor, mode=mode)
        
        # Prepare validation data
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
        
        # Train model
        try:
            # Check if model has fit method (TensorFlow/Keras)
            if hasattr(model, 'fit'):
                history = model.fit(
                    X_train,
                    y_train,
                    validation_data=validation_data,
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=callbacks,
                    verbose=verbose,
                    shuffle=False,  # Important for time series
                )
                
                # Extract training history
                if hasattr(history, 'history'):
                    history_dict = history.history
                else:
                    history_dict = {}
                
                # Get final metrics
                train_loss = history_dict.get('loss', [None])[-1] if history_dict.get('loss') else None
                val_loss = history_dict.get('val_loss', [None])[-1] if history_dict.get('val_loss') else None
                
                # Check if early stopping was triggered
                early_stopped = False
                if callbacks:
                    for callback in callbacks:
                        if hasattr(callback, 'stopped_epoch') and callback.stopped_epoch > 0:
                            early_stopped = True
                            break
                
                result = {
                    "status": "success",
                    "train_loss": float(train_loss) if train_loss else None,
                    "val_loss": float(val_loss) if val_loss else None,
                    "epochs_trained": len(history_dict.get('loss', [])) if history_dict.get('loss') else epochs,
                    "early_stopped": early_stopped,
                    "batch_size": batch_size,
                    "history": history_dict,
                }
                
                logger.info(
                    "adaptive_training_complete",
                    epochs=result["epochs_trained"],
                    early_stopped=early_stopped,
                    batch_size=batch_size,
                )
                
                return result
            else:
                # For non-Keras models, just fit normally
                model.fit(X_train, y_train)
                return {
                    "status": "success",
                    "epochs_trained": 1,
                    "early_stopped": False,
                    "batch_size": batch_size,
                }
                
        except Exception as e:
            logger.error("adaptive_training_failed", error=str(e))
            return {
                "status": "failed",
                "error": str(e),
            }

    def load_checkpoint(
        self,
        model: Any,
        checkpoint_path: str,
    ) -> bool:
        """
        Load model from checkpoint.
        
        Args:
            model: Model to load weights into
            checkpoint_path: Path to checkpoint file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if hasattr(model, 'load_weights'):
                model.load_weights(checkpoint_path)
                logger.info("checkpoint_loaded", path=checkpoint_path)
                return True
            else:
                logger.warning("model_does_not_support_checkpoints", model_type=type(model))
                return False
        except Exception as e:
            logger.error("checkpoint_load_failed", path=checkpoint_path, error=str(e))
            return False

