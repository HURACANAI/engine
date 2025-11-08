"""
Curriculum Scheduler

Manages curriculum learning progression.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Optional

import numpy as np
import pandas as pd
import structlog

from .difficulty import (
    calculate_sample_difficulty,
    create_difficulty_bins
)

logger = structlog.get_logger(__name__)


class DifficultyMetric(str, Enum):
    """Difficulty calculation method"""
    VOLATILITY = "volatility"
    LABEL_RARITY = "label_rarity"
    FEATURE_COMPLEXITY = "feature_complexity"
    COMBINED = "combined"


@dataclass
class CurriculumStage:
    """
    Single stage in curriculum

    Represents a difficulty level (e.g., easy, medium, hard).
    """
    stage_num: int
    name: str
    difficulty_range: tuple  # (min, max) difficulty

    data: pd.DataFrame  # Training data for this stage
    num_samples: int

    # Performance requirements to progress
    min_accuracy: float = 0.6
    min_sharpe: float = 0.5

    # Training config for this stage
    max_epochs: int = 100
    early_stopping_patience: int = 10


@dataclass
class Curriculum:
    """
    Complete curriculum

    Contains all stages from easy to hard.
    """
    stages: List[CurriculumStage]
    total_samples: int
    difficulty_metric: DifficultyMetric

    current_stage_idx: int = 0

    def get_current_stage(self) -> CurriculumStage:
        """Get current stage"""
        return self.stages[self.current_stage_idx]

    def advance_stage(self) -> bool:
        """
        Advance to next stage

        Returns:
            True if advanced, False if already at last stage
        """
        if self.current_stage_idx < len(self.stages) - 1:
            self.current_stage_idx += 1
            return True
        return False

    def is_complete(self) -> bool:
        """Check if curriculum is complete"""
        return self.current_stage_idx >= len(self.stages) - 1


class CurriculumScheduler:
    """
    Curriculum Learning Scheduler

    Manages progressive training from easy to hard examples.

    Example:
        scheduler = CurriculumScheduler()

        # Create curriculum
        curriculum = scheduler.create_curriculum(
            training_df=data,
            feature_cols=["close", "volume"],
            label_col="target",
            num_stages=3
        )

        # Train progressively
        for stage in curriculum.stages:
            print(f"Training stage {stage.stage_num}: {stage.name}")

            model.fit(stage.data[feature_cols], stage.data[label_col])

            if scheduler.should_progress(model, stage, val_data):
                print("  ✅ Passed! Advancing...")
                curriculum.advance_stage()
            else:
                print("  ⏳ Need more training")
                break
    """

    def __init__(
        self,
        difficulty_metric: DifficultyMetric = DifficultyMetric.COMBINED,
        num_stages: int = 3
    ):
        """
        Initialize curriculum scheduler

        Args:
            difficulty_metric: How to calculate sample difficulty
            num_stages: Number of curriculum stages (default 3 = easy/medium/hard)
        """
        self.difficulty_metric = difficulty_metric
        self.num_stages = num_stages

    def create_curriculum(
        self,
        training_df: pd.DataFrame,
        feature_cols: List[str],
        label_col: str,
        regimes: Optional[pd.Series] = None
    ) -> Curriculum:
        """
        Create curriculum from training data

        Args:
            training_df: Training DataFrame
            feature_cols: Feature column names
            label_col: Label column name
            regimes: Optional regime labels for regime-specific curriculum

        Returns:
            Curriculum
        """
        logger.info(
            "creating_curriculum",
            num_samples=len(training_df),
            num_stages=self.num_stages,
            difficulty_metric=self.difficulty_metric.value
        )

        # Calculate difficulty for all samples
        features = training_df[feature_cols]
        labels = training_df[label_col]

        difficulty = calculate_sample_difficulty(
            features,
            labels,
            method=self.difficulty_metric.value
        )

        # Create difficulty bins
        bins = create_difficulty_bins(difficulty, num_bins=self.num_stages)

        # Create stages
        stages = []

        stage_names = ["Easy", "Medium", "Hard", "Expert", "Master"]

        for stage_num in range(self.num_stages):
            stage_mask = bins == stage_num

            stage_data = training_df[stage_mask].copy()
            stage_difficulty = difficulty[stage_mask]

            stage = CurriculumStage(
                stage_num=stage_num,
                name=stage_names[stage_num] if stage_num < len(stage_names) else f"Stage {stage_num}",
                difficulty_range=(
                    float(stage_difficulty.min()),
                    float(stage_difficulty.max())
                ),
                data=stage_data,
                num_samples=len(stage_data),
                # Harder stages require better performance
                min_accuracy=0.55 + (stage_num * 0.05),
                min_sharpe=0.5 + (stage_num * 0.2),
                # Harder stages get more training epochs
                max_epochs=50 + (stage_num * 25)
            )

            stages.append(stage)

            logger.info(
                "curriculum_stage_created",
                stage_num=stage_num,
                name=stage.name,
                num_samples=stage.num_samples,
                difficulty_range=stage.difficulty_range
            )

        curriculum = Curriculum(
            stages=stages,
            total_samples=len(training_df),
            difficulty_metric=self.difficulty_metric,
            current_stage_idx=0
        )

        return curriculum

    def should_progress(
        self,
        model: Any,
        stage: CurriculumStage,
        validation_data: pd.DataFrame,
        feature_cols: List[str],
        label_col: str
    ) -> bool:
        """
        Check if model is ready to progress to next stage

        Args:
            model: Trained model
            stage: Current curriculum stage
            validation_data: Validation data
            feature_cols: Feature columns
            label_col: Label column

        Returns:
            True if ready to progress
        """
        # Evaluate model on validation data
        X_val = validation_data[feature_cols]
        y_val = validation_data[label_col]

        # Get predictions
        if hasattr(model, 'predict'):
            y_pred = model.predict(X_val)
        else:
            logger.warning("model_has_no_predict_method")
            return False

        # Calculate accuracy
        accuracy = (y_pred == y_val).mean()

        # Calculate Sharpe (if trading returns)
        if 'close' in validation_data.columns:
            returns = validation_data['close'].pct_change()
            strategy_returns = returns * y_pred

            if strategy_returns.std() > 0:
                sharpe = (
                    (strategy_returns.mean() / strategy_returns.std()) *
                    np.sqrt(252)
                )
            else:
                sharpe = 0.0
        else:
            sharpe = 0.0

        # Check progression criteria
        passed_accuracy = accuracy >= stage.min_accuracy
        passed_sharpe = sharpe >= stage.min_sharpe

        logger.info(
            "curriculum_progression_check",
            stage_num=stage.stage_num,
            stage_name=stage.name,
            accuracy=accuracy,
            min_accuracy=stage.min_accuracy,
            sharpe=sharpe,
            min_sharpe=stage.min_sharpe,
            passed=passed_accuracy and passed_sharpe
        )

        return passed_accuracy and passed_sharpe

    def create_adaptive_curriculum(
        self,
        training_df: pd.DataFrame,
        feature_cols: List[str],
        label_col: str,
        model_performance: float
    ) -> Curriculum:
        """
        Create adaptive curriculum based on current model performance

        If model is struggling, create easier curriculum.
        If model is doing well, create harder curriculum.

        Args:
            training_df: Training DataFrame
            feature_cols: Feature columns
            label_col: Label column
            model_performance: Current model accuracy/Sharpe

        Returns:
            Adapted Curriculum
        """
        # Adjust difficulty based on performance
        if model_performance < 0.5:
            # Struggling - create easier curriculum with more easy stages
            num_stages = 4
            logger.info("creating_easier_curriculum", performance=model_performance)
        elif model_performance > 0.8:
            # Doing well - create harder curriculum
            num_stages = 2
            logger.info("creating_harder_curriculum", performance=model_performance)
        else:
            # Normal curriculum
            num_stages = 3

        # Temporarily override num_stages
        original_num_stages = self.num_stages
        self.num_stages = num_stages

        curriculum = self.create_curriculum(
            training_df,
            feature_cols,
            label_col
        )

        # Restore original
        self.num_stages = original_num_stages

        return curriculum

    def get_stage_weights(self, curriculum: Curriculum) -> np.ndarray:
        """
        Get sample weights for current stage

        Can be used for weighted sampling during training.

        Args:
            curriculum: Curriculum

        Returns:
            Array of sample weights
        """
        current_stage = curriculum.get_current_stage()

        # Give higher weight to current stage samples
        weights = np.zeros(curriculum.total_samples)

        # Placeholder - would need original indices to map back
        # For now, return uniform weights
        logger.warning("get_stage_weights_not_fully_implemented")

        return weights
