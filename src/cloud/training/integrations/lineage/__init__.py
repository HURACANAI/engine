"""
Model Lineage & Rollback

Tracks model ancestry and enables safe rollback to previous versions.

Key Features:
- Full model lineage tracking (parent models, hyperparameter changes)
- Safe rollback to previous versions
- Performance comparison across lineage
- Automatic ancestry validation

Usage:
    from src.cloud.training.integrations.lineage import LineageTracker

    # Track new model
    tracker = LineageTracker(registry_dsn="postgresql://...")

    lineage_id = tracker.track_model(
        model_id="btc_trend_v48",
        parent_model_id="btc_trend_v47",
        change_type="hyperparameter_tuning",
        changes={
            "learning_rate": {"old": 0.001, "new": 0.0005},
            "hidden_layers": {"old": 2, "new": 3}
        },
        reason="Improved validation Sharpe from 1.2 to 1.5"
    )

    # Get lineage tree
    tree = tracker.get_lineage_tree("btc_trend_v48")

    # Rollback if new model fails
    if live_performance_bad:
        tracker.rollback_model(
            current_model_id="btc_trend_v48",
            target_model_id="btc_trend_v47",
            reason="Performance degraded in live trading"
        )
"""

from .tracker import (
    LineageTracker,
    LineageNode,
    ChangeType
)
from .rollback import (
    RollbackManager,
    RollbackRecord
)

__all__ = [
    # Lineage tracking
    "LineageTracker",
    "LineageNode",
    "ChangeType",

    # Rollback
    "RollbackManager",
    "RollbackRecord",
]
