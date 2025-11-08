"""
Engine Consensus with Diversity

Combines predictions from multiple diverse engines with consensus voting.

Key Features:
- Diversity measurement between engines
- Consensus voting (majority, weighted, unanimous)
- Confidence scoring based on agreement
- Disagreement detection and handling

Usage:
    from src.cloud.engine.consensus import ConsensusEngine

    consensus = ConsensusEngine(
        engines=["trend_v1", "trend_v2", "mean_rev_v1"],
        voting_method="weighted"
    )

    # Get predictions from all engines
    predictions = {
        "trend_v1": {"signal": 1, "confidence": 0.8},
        "trend_v2": {"signal": 1, "confidence": 0.7},
        "mean_rev_v1": {"signal": -1, "confidence": 0.6}
    }

    # Generate consensus
    result = consensus.generate_consensus(predictions)

    print(f"Consensus signal: {result.signal}")
    print(f"Confidence: {result.confidence:.0%}")
    print(f"Agreement: {result.agreement_pct:.0%}")
"""

from .consensus_engine import (
    ConsensusEngine,
    ConsensusResult,
    VotingMethod
)
from .diversity import (
    calculate_diversity,
    measure_prediction_diversity
)

__all__ = [
    # Consensus
    "ConsensusEngine",
    "ConsensusResult",
    "VotingMethod",

    # Diversity
    "calculate_diversity",
    "measure_prediction_diversity",
]
