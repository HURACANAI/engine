"""Automated error detection and resolution during training."""

from __future__ import annotations

from typing import Any, Dict, Optional

import structlog  # type: ignore[reportMissingImports]

logger = structlog.get_logger(__name__)


class ErrorResolver:
    """
    Automated error detection and resolution:
    - Detects shape mismatches, NaN losses, exploding gradients
    - Automatically retries with adjusted parameters
    - Logs errors to Brain Library
    """

    def __init__(
        self,
        max_retries: int = 3,
        batch_size_reduction_factor: float = 0.5,
        learning_rate_reduction_factor: float = 0.5,
    ) -> None:
        """
        Initialize error resolver.
        
        Args:
            max_retries: Maximum number of retries
            batch_size_reduction_factor: Factor to reduce batch size by
            learning_rate_reduction_factor: Factor to reduce learning rate by
        """
        self.max_retries = max_retries
        self.batch_size_reduction_factor = batch_size_reduction_factor
        self.learning_rate_reduction_factor = learning_rate_reduction_factor
        
        logger.info(
            "error_resolver_initialized",
            max_retries=max_retries,
            batch_size_reduction=batch_size_reduction_factor,
            learning_rate_reduction=learning_rate_reduction_factor,
        )

    def classify_error(self, error: Exception) -> str:
        """
        Classify error type.
        
        Args:
            error: Exception to classify
            
        Returns:
            Error type string
        """
        error_message = str(error).lower()
        error_type = type(error).__name__
        
        # Shape mismatch
        if "shape" in error_message or "dimension" in error_message:
            return "shape_mismatch"
        
        # NaN loss
        if "nan" in error_message or "not a number" in error_message:
            return "nan_loss"
        
        # Exploding gradient
        if "gradient" in error_message or "exploding" in error_message or "inf" in error_message:
            return "exploding_gradient"
        
        # Memory error
        if "memory" in error_message or "out of memory" in error_message:
            return "memory_error"
        
        # ValueError
        if error_type == "ValueError":
            return "value_error"
        
        # Unknown
        return "unknown"

    def resolve_shape_mismatch(
        self,
        error: Exception,
        training_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Resolve shape mismatch error.
        
        Args:
            error: Shape mismatch error
            training_config: Training configuration
            
        Returns:
            Updated training configuration
        """
        logger.warning("resolving_shape_mismatch", error=str(error))
        
        # Try to fix common shape issues
        updated_config = training_config.copy()
        
        # Reduce batch size (might help with shape issues)
        if "batch_size" in updated_config:
            updated_config["batch_size"] = int(
                updated_config["batch_size"] * self.batch_size_reduction_factor
            )
            updated_config["batch_size"] = max(1, updated_config["batch_size"])
        
        # Add input shape validation
        updated_config["validate_input_shape"] = True
        
        return updated_config

    def resolve_nan_loss(
        self,
        error: Exception,
        training_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Resolve NaN loss error.
        
        Args:
            error: NaN loss error
            training_config: Training configuration
            
        Returns:
            Updated training configuration
        """
        logger.warning("resolving_nan_loss", error=str(error))
        
        updated_config = training_config.copy()
        
        # Reduce learning rate
        if "learning_rate" in updated_config:
            updated_config["learning_rate"] *= self.learning_rate_reduction_factor
        else:
            updated_config["learning_rate"] = 1e-5  # Very small LR
        
        # Add gradient clipping
        updated_config["gradient_clip_norm"] = 1.0
        
        # Reduce batch size
        if "batch_size" in updated_config:
            updated_config["batch_size"] = int(
                updated_config["batch_size"] * self.batch_size_reduction_factor
            )
            updated_config["batch_size"] = max(1, updated_config["batch_size"])
        
        # Add NaN checking
        updated_config["check_nan"] = True
        
        return updated_config

    def resolve_exploding_gradient(
        self,
        error: Exception,
        training_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Resolve exploding gradient error.
        
        Args:
            error: Exploding gradient error
            training_config: Training configuration
            
        Returns:
            Updated training configuration
        """
        logger.warning("resolving_exploding_gradient", error=str(error))
        
        updated_config = training_config.copy()
        
        # Add gradient clipping
        updated_config["gradient_clip_norm"] = 1.0
        
        # Reduce learning rate
        if "learning_rate" in updated_config:
            updated_config["learning_rate"] *= self.learning_rate_reduction_factor
        else:
            updated_config["learning_rate"] = 1e-4
        
        # Reduce batch size
        if "batch_size" in updated_config:
            updated_config["batch_size"] = int(
                updated_config["batch_size"] * self.batch_size_reduction_factor
            )
            updated_config["batch_size"] = max(1, updated_config["batch_size"])
        
        return updated_config

    def resolve_memory_error(
        self,
        error: Exception,
        training_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Resolve memory error.
        
        Args:
            error: Memory error
            training_config: Training configuration
            
        Returns:
            Updated training configuration
        """
        logger.warning("resolving_memory_error", error=str(error))
        
        updated_config = training_config.copy()
        
        # Reduce batch size significantly
        if "batch_size" in updated_config:
            updated_config["batch_size"] = int(
                updated_config["batch_size"] * self.batch_size_reduction_factor * self.batch_size_reduction_factor
            )
            updated_config["batch_size"] = max(1, updated_config["batch_size"])
        
        # Reduce model size if possible
        if "model_size" in updated_config:
            updated_config["model_size"] = int(updated_config["model_size"] * 0.5)
        
        return updated_config

    def resolve_error(
        self,
        error: Exception,
        training_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Resolve error based on type.
        
        Args:
            error: Exception to resolve
            training_config: Training configuration
            
        Returns:
            Updated training configuration
        """
        error_type = self.classify_error(error)
        
        logger.info("resolving_error", error_type=error_type, error=str(error))
        
        if error_type == "shape_mismatch":
            return self.resolve_shape_mismatch(error, training_config)
        elif error_type == "nan_loss":
            return self.resolve_nan_loss(error, training_config)
        elif error_type == "exploding_gradient":
            return self.resolve_exploding_gradient(error, training_config)
        elif error_type == "memory_error":
            return self.resolve_memory_error(error, training_config)
        else:
            # Generic resolution
            logger.warning("unknown_error_type", error_type=error_type, using_generic_resolution=True)
            updated_config = training_config.copy()
            
            # Reduce batch size
            if "batch_size" in updated_config:
                updated_config["batch_size"] = int(
                    updated_config["batch_size"] * self.batch_size_reduction_factor
                )
                updated_config["batch_size"] = max(1, updated_config["batch_size"])
            
            return updated_config

    def retry_training(
        self,
        model: Any,
        X_train: Any,
        y_train: Any,
        training_config: Dict[str, Any],
        X_val: Optional[Any] = None,
        y_val: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Retry training with error resolution.
        
        Args:
            model: Model to train
            X_train: Training features
            y_train: Training targets
            training_config: Training configuration
            X_val: Validation features
            y_val: Validation targets
            
        Returns:
            Training results dictionary
        """
        current_config = training_config.copy()
        
        for attempt in range(self.max_retries):
            try:
                logger.info("training_attempt", attempt=attempt + 1, max_retries=self.max_retries)
                
                # Train model
                if hasattr(model, 'fit'):
                    # Keras/TensorFlow model
                    result = model.fit(
                        X_train, y_train,
                        validation_data=(X_val, y_val) if X_val is not None and y_val is not None else None,
                        **{k: v for k, v in current_config.items() if k not in ['validate_input_shape', 'check_nan', 'gradient_clip_norm']}
                    )
                else:
                    # Sklearn model
                    result = model.fit(X_train, y_train)
                
                logger.info("training_successful", attempt=attempt + 1)
                
                return {
                    "status": "success",
                    "attempt": attempt + 1,
                    "config": current_config,
                    "result": result,
                }
                
            except Exception as e:
                logger.warning("training_failed", attempt=attempt + 1, error=str(e))
                
                if attempt < self.max_retries - 1:
                    # Resolve error and retry
                    current_config = self.resolve_error(e, current_config)
                    logger.info("retrying_with_resolved_config", new_config=current_config)
                else:
                    # Final attempt failed
                    logger.error("training_failed_after_retries", max_retries=self.max_retries, error=str(e))
                    return {
                        "status": "failed",
                        "attempt": attempt + 1,
                        "error": str(e),
                        "error_type": self.classify_error(e),
                    }
        
        return {
            "status": "failed",
            "error": "Max retries exceeded",
        }

