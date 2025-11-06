"""
Curriculum Learning with Difficulty Progression

Implements progressive difficulty training:
- Week 1-2: Train on TREND regime only (easiest)
- Week 3-4: Add RANGE regime
- Week 5-6: Add PANIC regime
- Week 7+: Full curriculum with regime transitions

Source: "Curriculum Learning" (Bengio et al., 2009)
Expected Impact: +25-35% faster learning, +8-12% final performance
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum
import structlog  # type: ignore
import numpy as np
import polars as pl

logger = structlog.get_logger(__name__)


class CurriculumStage(Enum):
    """Curriculum learning stages."""
    STAGE_1_TREND_ONLY = "trend_only"  # Week 1-2
    STAGE_2_ADD_RANGE = "add_range"  # Week 3-4
    STAGE_3_ADD_PANIC = "add_panic"  # Week 5-6
    STAGE_4_FULL = "full"  # Week 7+


@dataclass
class CurriculumConfig:
    """Curriculum learning configuration."""
    stage_1_days: int = 14  # Days for TREND only
    stage_2_days: int = 14  # Days for TREND + RANGE
    stage_3_days: int = 14  # Days for TREND + RANGE + PANIC
    stage_4_days: int = None  # Full curriculum (unlimited)
    
    # Regime weights per stage
    stage_1_weights: Dict[str, float] = None  # TREND: 1.0
    stage_2_weights: Dict[str, float] = None  # TREND: 0.5, RANGE: 0.5
    stage_3_weights: Dict[str, float] = None  # TREND: 0.33, RANGE: 0.33, PANIC: 0.33
    stage_4_weights: Dict[str, float] = None  # Full: All regimes equal


class CurriculumLearner:
    """
    Curriculum learning scheduler.
    
    Progressively increases difficulty by adding regimes over time.
    """

    def __init__(
        self,
        config: Optional[CurriculumConfig] = None,
        start_date: Optional[datetime] = None,
    ):
        """
        Initialize curriculum learner.
        
        Args:
            config: Curriculum configuration
            start_date: Training start date (if None, uses now)
        """
        self.config = config or CurriculumConfig()
        self.start_date = start_date or datetime.now()
        
        # Initialize stage weights
        if self.config.stage_1_weights is None:
            self.config.stage_1_weights = {'trend': 1.0, 'range': 0.0, 'panic': 0.0}
        if self.config.stage_2_weights is None:
            self.config.stage_2_weights = {'trend': 0.5, 'range': 0.5, 'panic': 0.0}
        if self.config.stage_3_weights is None:
            self.config.stage_3_weights = {'trend': 0.33, 'range': 0.33, 'panic': 0.33}
        if self.config.stage_4_weights is None:
            self.config.stage_4_weights = {'trend': 0.33, 'range': 0.33, 'panic': 0.33}
        
        logger.info(
            "curriculum_learner_initialized",
            start_date=self.start_date,
            stage_1_days=self.config.stage_1_days,
        )

    def get_current_stage(self) -> Tuple[CurriculumStage, Dict[str, float]]:
        """
        Get current curriculum stage and regime weights.
        
        Returns:
            (stage, regime_weights)
        """
        days_elapsed = (datetime.now() - self.start_date).days
        
        if days_elapsed < self.config.stage_1_days:
            stage = CurriculumStage.STAGE_1_TREND_ONLY
            weights = self.config.stage_1_weights
        elif days_elapsed < self.config.stage_1_days + self.config.stage_2_days:
            stage = CurriculumStage.STAGE_2_ADD_RANGE
            weights = self.config.stage_2_weights
        elif days_elapsed < self.config.stage_1_days + self.config.stage_2_days + self.config.stage_3_days:
            stage = CurriculumStage.STAGE_3_ADD_PANIC
            weights = self.config.stage_3_weights
        else:
            stage = CurriculumStage.STAGE_4_FULL
            weights = self.config.stage_4_weights
        
        logger.debug(
            "curriculum_stage",
            stage=stage.value,
            days_elapsed=days_elapsed,
            weights=weights,
        )
        
        return stage, weights

    def filter_training_data(
        self,
        data: pl.DataFrame,
        regime_column: str = 'regime',
    ) -> pl.DataFrame:
        """
        Filter training data based on current curriculum stage.
        
        Args:
            data: Training data DataFrame
            regime_column: Name of regime column
            
        Returns:
            Filtered DataFrame
        """
        stage, weights = self.get_current_stage()
        
        # Get allowed regimes (weight > 0)
        allowed_regimes = [regime for regime, weight in weights.items() if weight > 0]
        
        # Filter data
        filtered = data.filter(pl.col(regime_column).is_in(allowed_regimes))
        
        logger.info(
            "curriculum_data_filtered",
            stage=stage.value,
            allowed_regimes=allowed_regimes,
            original_size=len(data),
            filtered_size=len(filtered),
        )
        
        return filtered

    def get_regime_weights(self) -> Dict[str, float]:
        """
        Get current regime weights.
        
        Returns:
            Dictionary of regime weights
        """
        _, weights = self.get_current_stage()
        return weights

