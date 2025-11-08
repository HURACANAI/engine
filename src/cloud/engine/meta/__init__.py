"""
Adaptive Meta Engine

Meta-learning engine that learns which base engines perform best in different regimes.

Key Features:
- Tracks engine performance by regime
- Learns optimal engine selection
- Adaptive weighting based on recent performance
- Automatic regime detection integration
- Fallback to best historical engine

Usage:
    from src.cloud.engine.meta import MetaEngine

    meta = MetaEngine(
        base_engines=["trend_following", "mean_reversion", "breakout"]
    )

    # Update with recent performance
    meta.update_performance(
        engine_name="trend_following",
        regime="trending",
        sharpe=1.8,
        pnl_bps=250
    )

    # Select best engine for current regime
    best_engine = meta.select_engine(
        current_regime="trending",
        recent_volatility=0.25
    )

    print(f"Selected: {best_engine}")
    # Returns: "trend_following" (best performer in trending regime)

    # Get engine weights
    weights = meta.get_engine_weights(regime="trending")
    # Returns: {"trend_following": 0.6, "mean_reversion": 0.2, "breakout": 0.2}
"""

from .meta_engine import (
    MetaEngine,
    EnginePerformance,
    EngineSelection
)
from .strategies import (
    select_best_engine,
    blend_engine_signals
)

__all__ = [
    # Meta engine
    "MetaEngine",
    "EnginePerformance",
    "EngineSelection",

    # Strategies
    "select_best_engine",
    "blend_engine_signals",
]
