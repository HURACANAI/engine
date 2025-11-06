"""
Synthetic Data Augmentation with CTGAN

Generates synthetic data for rare regimes (PANIC only ~5% of time):
- CTGAN (Conditional Tabular GAN) for synthetic PANIC regime data
- Validate: Synthetic data must match statistical properties of real data
- 80% real + 20% synthetic for PANIC regime training

Source: "Modeling Tabular Data using Conditional GAN" (Xu et al., 2019)
Expected Impact: +30-50% PANIC regime performance, -20-30% tail drawdown
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import structlog  # type: ignore
import numpy as np
import pandas as pd

try:
    from ctgan import CTGAN
    HAS_CTGAN = True
except ImportError:
    HAS_CTGAN = False

logger = structlog.get_logger(__name__)


@dataclass
class SyntheticDataResult:
    """Result of synthetic data generation."""
    synthetic_data: pd.DataFrame
    n_samples: int
    regime: str
    validation_passed: bool
    statistical_similarity: float  # 0-1, how similar to real data


class SyntheticDataGenerator:
    """
    Synthetic data generator using CTGAN.
    
    Generates synthetic data for rare regimes to improve learning.
    """

    def __init__(
        self,
        epochs: int = 300,  # Training epochs for CTGAN
        batch_size: int = 500,
        synthetic_ratio: float = 0.2,  # 20% synthetic data
    ):
        """
        Initialize synthetic data generator.
        
        Args:
            epochs: Training epochs for CTGAN
            batch_size: Batch size for CTGAN
            synthetic_ratio: Ratio of synthetic to real data
        """
        if not HAS_CTGAN:
            logger.warning("ctgan_not_available", message="Install ctgan: pip install ctgan")
        
        self.epochs = epochs
        self.batch_size = batch_size
        self.synthetic_ratio = synthetic_ratio
        
        # CTGAN models per regime
        self.ctgan_models: Dict[str, Any] = {}
        
        logger.info(
            "synthetic_data_generator_initialized",
            epochs=epochs,
            synthetic_ratio=synthetic_ratio,
        )

    def generate_synthetic_data(
        self,
        real_data: pd.DataFrame,
        regime: str,
        n_samples: Optional[int] = None,
    ) -> SyntheticDataResult:
        """
        Generate synthetic data for a regime.
        
        Args:
            real_data: Real data for the regime
            regime: Market regime
            n_samples: Number of synthetic samples (if None, uses synthetic_ratio)
            
        Returns:
            SyntheticDataResult
        """
        if not HAS_CTGAN:
            # Fallback: Return empty synthetic data
            return SyntheticDataResult(
                synthetic_data=pd.DataFrame(),
                n_samples=0,
                regime=regime,
                validation_passed=False,
                statistical_similarity=0.0,
            )
        
        if len(real_data) < 50:
            logger.warning("insufficient_real_data", regime=regime, n_samples=len(real_data))
            return SyntheticDataResult(
                synthetic_data=pd.DataFrame(),
                n_samples=0,
                regime=regime,
                validation_passed=False,
                statistical_similarity=0.0,
            )
        
        # Calculate number of synthetic samples
        if n_samples is None:
            n_samples = int(len(real_data) * self.synthetic_ratio)
        
        # Train or load CTGAN model
        if regime not in self.ctgan_models:
            logger.info("training_ctgan_model", regime=regime, n_samples=len(real_data))
            ctgan = CTGAN(epochs=self.epochs, batch_size=self.batch_size, verbose=False)
            ctgan.fit(real_data)
            self.ctgan_models[regime] = ctgan
        else:
            ctgan = self.ctgan_models[regime]
        
        # Generate synthetic data
        try:
            synthetic_data = ctgan.sample(n_samples)
            
            # Validate synthetic data
            validation_passed, similarity = self._validate_synthetic_data(real_data, synthetic_data)
            
            logger.info(
                "synthetic_data_generated",
                regime=regime,
                n_samples=n_samples,
                validation_passed=validation_passed,
                similarity=similarity,
            )
            
            return SyntheticDataResult(
                synthetic_data=synthetic_data,
                n_samples=n_samples,
                regime=regime,
                validation_passed=validation_passed,
                statistical_similarity=similarity,
            )
            
        except Exception as e:
            logger.error("synthetic_data_generation_failed", regime=regime, error=str(e))
            return SyntheticDataResult(
                synthetic_data=pd.DataFrame(),
                n_samples=0,
                regime=regime,
                validation_passed=False,
                statistical_similarity=0.0,
            )

    def _validate_synthetic_data(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
    ) -> Tuple[bool, float]:
        """
        Validate synthetic data matches real data statistics.
        
        Returns:
            (validation_passed, statistical_similarity)
        """
        if len(synthetic_data) == 0:
            return False, 0.0
        
        # Compare means and stds
        real_mean = real_data.mean()
        synthetic_mean = synthetic_data.mean()
        
        real_std = real_data.std()
        synthetic_std = synthetic_data.std()
        
        # Calculate similarity (1.0 = identical, 0.0 = very different)
        mean_similarity = 1.0 - np.mean(np.abs(real_mean - synthetic_mean) / (np.abs(real_mean) + 1e-6))
        std_similarity = 1.0 - np.mean(np.abs(real_std - synthetic_std) / (np.abs(real_std) + 1e-6))
        
        overall_similarity = (mean_similarity + std_similarity) / 2.0
        overall_similarity = max(0.0, min(1.0, overall_similarity))
        
        # Validation passes if similarity > 0.7
        validation_passed = overall_similarity > 0.7
        
        return validation_passed, overall_similarity

    def augment_training_data(
        self,
        real_data: pd.DataFrame,
        regime: str,
    ) -> pd.DataFrame:
        """
        Augment training data with synthetic data.
        
        Args:
            real_data: Real training data
            regime: Market regime
            
        Returns:
            Augmented DataFrame (80% real + 20% synthetic)
        """
        # Generate synthetic data
        synthetic_result = self.generate_synthetic_data(real_data, regime)
        
        if not synthetic_result.validation_passed or len(synthetic_result.synthetic_data) == 0:
            logger.warning("synthetic_data_validation_failed", regime=regime)
            return real_data  # Return only real data
        
        # Combine real + synthetic
        augmented = pd.concat([real_data, synthetic_result.synthetic_data], ignore_index=True)
        
        # Shuffle
        augmented = augmented.sample(frac=1.0, random_state=42).reset_index(drop=True)
        
        logger.info(
            "training_data_augmented",
            regime=regime,
            real_samples=len(real_data),
            synthetic_samples=len(synthetic_result.synthetic_data),
            total_samples=len(augmented),
        )
        
        return augmented

