"""
Curriculum Learning

Progressive training starting from easy examples to hard examples.

Key Features:
- Difficulty-based sample ordering
- Dynamic difficulty adjustment
- Multi-stage curriculum (easy → medium → hard)
- Performance-based progression
- Regime-aware curriculum

Usage:
    from training.curriculum import CurriculumScheduler

    scheduler = CurriculumScheduler()

    # Create curriculum from training data
    curriculum = scheduler.create_curriculum(
        training_df=historical_data,
        feature_cols=["close", "volume", "volatility"],
        label_col="target"
    )

    # Train progressively
    for stage in curriculum.stages:
        print(f"Stage {stage.stage_num}: {stage.name}")

        # Train on this stage's data
        model.fit(stage.data[feature_cols], stage.data[label_col])

        # Check if ready to progress
        if scheduler.should_progress(model, stage):
            print("  ✅ Passed! Moving to next stage")
        else:
            print("  ⏳ Need more training on this stage")
            break
"""

from .scheduler import (
    CurriculumScheduler,
    Curriculum,
    CurriculumStage,
    DifficultyMetric
)
from .difficulty import (
    calculate_sample_difficulty,
    rank_by_difficulty
)

__all__ = [
    # Scheduler
    "CurriculumScheduler",
    "Curriculum",
    "CurriculumStage",
    "DifficultyMetric",

    # Difficulty
    "calculate_sample_difficulty",
    "rank_by_difficulty",
]
