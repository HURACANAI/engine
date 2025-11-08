"""Class balancing and resampling for imbalanced datasets."""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np
import structlog  # type: ignore[reportMissingImports]

logger = structlog.get_logger(__name__)


class ClassBalancer:
    """
    Handles class imbalance:
    - Oversamples minority class
    - Applies SMOTE for synthetic samples
    - Uses weighted loss functions
    - Dynamically rebalances when data skews
    """

    def __init__(
        self,
        method: str = "smote",
        sampling_strategy: float = 0.5,
        random_state: int = 42,
    ) -> None:
        """
        Initialize class balancer.
        
        Args:
            method: Balancing method ('oversample', 'smote', 'weighted')
            sampling_strategy: Sampling strategy (ratio or 'auto')
            random_state: Random seed
        """
        self.method = method
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        
        logger.info(
            "class_balancer_initialized",
            method=method,
            sampling_strategy=sampling_strategy,
        )

    def is_imbalanced(self, y: np.ndarray, threshold: float = 0.4) -> bool:
        """
        Check if dataset is imbalanced.
        
        Args:
            y: Target array
            threshold: Imbalance threshold (0.0-0.5)
            
        Returns:
            True if imbalanced, False otherwise
        """
        # For binary classification
        if len(np.unique(y)) == 2:
            class_counts = np.bincount(y.astype(int))
            minority_ratio = min(class_counts) / len(y)
            is_imbalanced = minority_ratio < threshold
            
            logger.info(
                "imbalance_check",
                minority_ratio=minority_ratio,
                threshold=threshold,
                is_imbalanced=is_imbalanced,
            )
            
            return is_imbalanced
        
        # For multi-class or regression
        return False

    def balance_classes(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Balance classes using specified method.
        
        Args:
            X: Features
            y: Targets
            
        Returns:
            Tuple of (balanced_X, balanced_y)
        """
        if not self.is_imbalanced(y):
            logger.info("dataset_not_imbalanced", skipping_balancing=True)
            return X, y
        
        if self.method == "oversample":
            return self._oversample(X, y)
        elif self.method == "smote":
            return self._smote(X, y)
        elif self.method == "weighted":
            return X, y  # Weighted loss will be handled during training
        else:
            logger.warning("unknown_balancing_method", method=self.method, using="oversample")
            return self._oversample(X, y)

    def _oversample(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Oversample minority class."""
        logger.info("oversampling_minority_class")
        
        # Get class counts
        unique_classes, class_counts = np.unique(y, return_counts=True)
        majority_class = unique_classes[np.argmax(class_counts)]
        minority_class = unique_classes[np.argmin(class_counts)]
        
        # Get minority samples
        minority_indices = np.where(y == minority_class)[0]
        majority_indices = np.where(y == majority_class)[0]
        
        # Calculate how many samples to generate
        n_majority = len(majority_indices)
        n_minority = len(minority_indices)
        n_to_generate = n_majority - n_minority
        
        # Randomly sample from minority class
        np.random.seed(self.random_state)
        sampled_indices = np.random.choice(minority_indices, size=n_to_generate, replace=True)
        
        # Combine original and sampled
        X_balanced = np.concatenate([X, X[sampled_indices]], axis=0)
        y_balanced = np.concatenate([y, y[sampled_indices]], axis=0)
        
        logger.info(
            "oversampling_complete",
            original_samples=len(X),
            balanced_samples=len(X_balanced),
            minority_samples=n_minority,
            generated_samples=n_to_generate,
        )
        
        return X_balanced, y_balanced

    def _smote(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply SMOTE (Synthetic Minority Oversampling Technique)."""
        try:
            from imblearn.over_sampling import SMOTE  # type: ignore[reportMissingImports]
        except ImportError:
            logger.warning("imblearn_not_available", using_oversample_instead=True)
            return self._oversample(X, y)
        
        logger.info("applying_smote")
        
        # Reshape X if needed (SMOTE expects 2D)
        original_shape = X.shape
        if len(X.shape) > 2:
            X_2d = X.reshape(X.shape[0], -1)
        else:
            X_2d = X
        
        # Apply SMOTE
        smote = SMOTE(
            sampling_strategy=self.sampling_strategy,
            random_state=self.random_state,
        )
        X_balanced_2d, y_balanced = smote.fit_resample(X_2d, y)
        
        # Reshape back if needed
        if len(original_shape) > 2:
            # Try to restore original shape
            new_samples = X_balanced_2d.shape[0]
            X_balanced = X_balanced_2d.reshape(new_samples, *original_shape[1:])
        else:
            X_balanced = X_balanced_2d
        
        logger.info(
            "smote_complete",
            original_samples=len(X),
            balanced_samples=len(X_balanced),
        )
        
        return X_balanced, y_balanced

    def get_class_weights(self, y: np.ndarray) -> dict:
        """
        Calculate class weights for weighted loss.
        
        Args:
            y: Target array
            
        Returns:
            Dictionary of class weights
        """
        unique_classes, class_counts = np.unique(y, return_counts=True)
        total_samples = len(y)
        
        # Calculate weights (inverse frequency)
        weights = {}
        for cls, count in zip(unique_classes, class_counts):
            weights[int(cls)] = total_samples / (len(unique_classes) * count)
        
        logger.info("class_weights_calculated", weights=weights)
        
        return weights

