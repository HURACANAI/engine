"""
Meta Engine Strategies

Different strategies for combining base engine signals.
"""

from typing import Dict, List

import numpy as np


def select_best_engine(
    engine_weights: Dict[str, float],
    threshold: float = 0.6
) -> str:
    """
    Select single best engine if confidence is high enough

    Args:
        engine_weights: Dict of {engine: weight}
        threshold: Minimum weight to select single engine

    Returns:
        Selected engine name

    Example:
        weights = {"trend": 0.7, "mean_reversion": 0.2, "breakout": 0.1}
        engine = select_best_engine(weights, threshold=0.6)
        # Returns: "trend" (weight > threshold)
    """
    best_engine = max(engine_weights, key=engine_weights.get)
    best_weight = engine_weights[best_engine]

    if best_weight >= threshold:
        return best_engine
    else:
        # No clear winner - could blend or return best anyway
        return best_engine


def blend_engine_signals(
    engine_signals: Dict[str, float],
    engine_weights: Dict[str, float]
) -> float:
    """
    Blend signals from multiple engines

    Args:
        engine_signals: Dict of {engine: signal} where signal in [-1, 1]
        engine_weights: Dict of {engine: weight} summing to 1.0

    Returns:
        Blended signal [-1, 1]

    Example:
        signals = {"trend": 0.8, "mean_reversion": -0.3, "breakout": 0.5}
        weights = {"trend": 0.6, "mean_reversion": 0.2, "breakout": 0.2}

        blended = blend_engine_signals(signals, weights)
        # Returns: 0.8*0.6 + (-0.3)*0.2 + 0.5*0.2 = 0.52
    """
    blended_signal = 0.0

    for engine, signal in engine_signals.items():
        weight = engine_weights.get(engine, 0.0)
        blended_signal += signal * weight

    # Clip to [-1, 1]
    blended_signal = max(-1.0, min(1.0, blended_signal))

    return blended_signal


def dynamic_engine_selection(
    engine_weights: Dict[str, float],
    recent_performance: Dict[str, float],
    decay_factor: float = 0.9
) -> str:
    """
    Dynamic engine selection with performance tracking

    Adjusts weights based on recent performance.

    Args:
        engine_weights: Base weights
        recent_performance: Recent Sharpe per engine
        decay_factor: Weight decay for underperformers

    Returns:
        Selected engine name
    """
    adjusted_weights = {}

    for engine, base_weight in engine_weights.items():
        performance = recent_performance.get(engine, 0.0)

        if performance > 0:
            # Good performance - boost weight
            adjusted_weights[engine] = base_weight * (1 + performance / 10)
        else:
            # Poor performance - decay weight
            adjusted_weights[engine] = base_weight * decay_factor

    # Normalize
    total_weight = sum(adjusted_weights.values())
    if total_weight > 0:
        adjusted_weights = {
            e: w / total_weight
            for e, w in adjusted_weights.items()
        }

    # Select best
    best_engine = max(adjusted_weights, key=adjusted_weights.get)

    return best_engine


def regime_aware_blending(
    engine_signals: Dict[str, float],
    regime: str,
    regime_engine_map: Dict[str, str]
) -> float:
    """
    Regime-aware signal blending

    Gives full weight to regime-appropriate engine.

    Args:
        engine_signals: Engine signals
        regime: Current regime
        regime_engine_map: Dict of {regime: best_engine}

    Returns:
        Blended signal

    Example:
        signals = {"trend": 0.8, "mean_reversion": -0.3}
        regime = "trending"
        map = {"trending": "trend", "choppy": "mean_reversion"}

        signal = regime_aware_blending(signals, regime, map)
        # Returns: 0.8 (trend engine signal)
    """
    preferred_engine = regime_engine_map.get(regime)

    if preferred_engine and preferred_engine in engine_signals:
        # Use preferred engine
        return engine_signals[preferred_engine]
    else:
        # Fallback to equal blend
        equal_weight = 1.0 / len(engine_signals)
        weights = {e: equal_weight for e in engine_signals.keys()}
        return blend_engine_signals(engine_signals, weights)
