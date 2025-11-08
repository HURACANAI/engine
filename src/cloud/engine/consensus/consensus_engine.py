"""
Consensus Engine

Combines predictions from multiple engines using consensus voting.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

import numpy as np
import structlog

from .diversity import measure_prediction_diversity

logger = structlog.get_logger(__name__)


class VotingMethod(str, Enum):
    """Consensus voting methods"""
    MAJORITY = "majority"  # Simple majority vote
    WEIGHTED = "weighted"  # Weighted by confidence
    UNANIMOUS = "unanimous"  # Require unanimous agreement
    AVERAGE = "average"  # Average of signals


@dataclass
class ConsensusResult:
    """Consensus prediction result"""
    signal: int  # -1, 0, 1
    confidence: float  # [0-1]
    agreement_pct: float  # [0-100] percentage agreement

    # Individual predictions
    individual_predictions: Dict[str, int]
    individual_confidences: Dict[str, float]

    # Diversity
    diversity_score: float

    # Voting details
    voting_method: str
    num_engines: int
    num_agree: int


class ConsensusEngine:
    """
    Consensus Engine

    Combines predictions from multiple diverse engines.

    Example:
        consensus = ConsensusEngine(
            engines=["trend_v1", "trend_v2", "mean_rev"],
            voting_method="weighted"
        )

        predictions = {
            "trend_v1": {"signal": 1, "confidence": 0.8},
            "trend_v2": {"signal": 1, "confidence": 0.7},
            "mean_rev": {"signal": -1, "confidence": 0.6}
        }

        result = consensus.generate_consensus(predictions)

        if result.confidence > 0.7:
            execute_trade(result.signal)
    """

    def __init__(
        self,
        engines: list[str],
        voting_method: VotingMethod = VotingMethod.WEIGHTED,
        min_agreement_threshold: float = 0.6
    ):
        """
        Initialize consensus engine

        Args:
            engines: List of engine names
            voting_method: Voting method to use
            min_agreement_threshold: Minimum agreement for high confidence
        """
        self.engines = engines
        self.voting_method = voting_method
        self.min_agreement_threshold = min_agreement_threshold

    def generate_consensus(
        self,
        predictions: Dict[str, Dict]
    ) -> ConsensusResult:
        """
        Generate consensus prediction

        Args:
            predictions: Dict of {engine_name: {"signal": int, "confidence": float}}

        Returns:
            ConsensusResult

        Example:
            predictions = {
                "engine1": {"signal": 1, "confidence": 0.8},
                "engine2": {"signal": 1, "confidence": 0.7},
                "engine3": {"signal": -1, "confidence": 0.6}
            }

            result = consensus.generate_consensus(predictions)
        """
        if len(predictions) == 0:
            return self._no_consensus_result()

        # Extract signals and confidences
        signals = {
            engine: pred["signal"]
            for engine, pred in predictions.items()
        }

        confidences = {
            engine: pred["confidence"]
            for engine, pred in predictions.items()
        }

        # Generate consensus based on voting method
        if self.voting_method == VotingMethod.MAJORITY:
            consensus_signal = self._majority_vote(signals)
        elif self.voting_method == VotingMethod.WEIGHTED:
            consensus_signal = self._weighted_vote(signals, confidences)
        elif self.voting_method == VotingMethod.UNANIMOUS:
            consensus_signal = self._unanimous_vote(signals)
        elif self.voting_method == VotingMethod.AVERAGE:
            consensus_signal = self._average_vote(signals, confidences)
        else:
            consensus_signal = self._majority_vote(signals)

        # Calculate agreement
        num_agree = sum(1 for s in signals.values() if s == consensus_signal)
        num_engines = len(signals)
        agreement_pct = (num_agree / num_engines) * 100

        # Calculate confidence
        confidence = self._calculate_consensus_confidence(
            consensus_signal,
            signals,
            confidences,
            agreement_pct
        )

        # Calculate diversity
        diversity_score = measure_prediction_diversity(signals)

        result = ConsensusResult(
            signal=consensus_signal,
            confidence=confidence,
            agreement_pct=agreement_pct,
            individual_predictions=signals,
            individual_confidences=confidences,
            diversity_score=diversity_score,
            voting_method=self.voting_method.value,
            num_engines=num_engines,
            num_agree=num_agree
        )

        logger.info(
            "consensus_generated",
            signal=consensus_signal,
            confidence=confidence,
            agreement_pct=agreement_pct,
            diversity=diversity_score,
            method=self.voting_method.value
        )

        return result

    def _majority_vote(self, signals: Dict[str, int]) -> int:
        """
        Simple majority vote

        Args:
            signals: Dict of {engine: signal}

        Returns:
            Consensus signal
        """
        signal_counts = {-1: 0, 0: 0, 1: 0}

        for signal in signals.values():
            signal_counts[signal] = signal_counts.get(signal, 0) + 1

        # Return signal with most votes
        consensus = max(signal_counts, key=signal_counts.get)

        return consensus

    def _weighted_vote(
        self,
        signals: Dict[str, int],
        confidences: Dict[str, float]
    ) -> int:
        """
        Weighted vote by confidence

        Args:
            signals: Dict of {engine: signal}
            confidences: Dict of {engine: confidence}

        Returns:
            Consensus signal
        """
        weighted_sum = 0.0
        total_weight = 0.0

        for engine, signal in signals.items():
            confidence = confidences.get(engine, 0.5)
            weighted_sum += signal * confidence
            total_weight += confidence

        if total_weight == 0:
            return 0

        # Average weighted signal
        avg_signal = weighted_sum / total_weight

        # Convert to discrete signal
        if avg_signal > 0.33:
            return 1
        elif avg_signal < -0.33:
            return -1
        else:
            return 0

    def _unanimous_vote(self, signals: Dict[str, int]) -> int:
        """
        Unanimous vote (all engines must agree)

        Args:
            signals: Dict of {engine: signal}

        Returns:
            Consensus signal (0 if no unanimity)
        """
        unique_signals = set(signals.values())

        if len(unique_signals) == 1:
            # Unanimous
            return list(unique_signals)[0]
        else:
            # No consensus
            return 0

    def _average_vote(
        self,
        signals: Dict[str, int],
        confidences: Dict[str, float]
    ) -> int:
        """
        Average vote (mean of signals)

        Args:
            signals: Dict of {engine: signal}
            confidences: Dict of {engine: confidence}

        Returns:
            Consensus signal
        """
        signal_values = list(signals.values())
        avg_signal = np.mean(signal_values)

        # Convert to discrete signal
        if avg_signal > 0.33:
            return 1
        elif avg_signal < -0.33:
            return -1
        else:
            return 0

    def _calculate_consensus_confidence(
        self,
        consensus_signal: int,
        signals: Dict[str, int],
        confidences: Dict[str, float],
        agreement_pct: float
    ) -> float:
        """
        Calculate confidence in consensus prediction

        Args:
            consensus_signal: Consensus signal
            signals: Individual signals
            confidences: Individual confidences
            agreement_pct: Agreement percentage

        Returns:
            Confidence [0-1]
        """
        # Base confidence from agreement
        agreement_confidence = agreement_pct / 100.0

        # Average confidence of agreeing engines
        agreeing_confidences = [
            confidences[engine]
            for engine, signal in signals.items()
            if signal == consensus_signal
        ]

        if len(agreeing_confidences) > 0:
            avg_confidence = np.mean(agreeing_confidences)
        else:
            avg_confidence = 0.5

        # Combined confidence (geometric mean)
        confidence = np.sqrt(agreement_confidence * avg_confidence)

        return confidence

    def _no_consensus_result(self) -> ConsensusResult:
        """Return no-consensus result"""
        return ConsensusResult(
            signal=0,
            confidence=0.0,
            agreement_pct=0.0,
            individual_predictions={},
            individual_confidences={},
            diversity_score=0.0,
            voting_method=self.voting_method.value,
            num_engines=0,
            num_agree=0
        )

    def detect_disagreement(
        self,
        predictions: Dict[str, Dict],
        disagreement_threshold: float = 0.5
    ) -> bool:
        """
        Detect significant disagreement between engines

        Args:
            predictions: Engine predictions
            disagreement_threshold: Threshold for flagging disagreement

        Returns:
            True if significant disagreement detected

        Example:
            if consensus.detect_disagreement(predictions):
                print("WARNING: Engines disagree - reduce position size")
        """
        signals = {e: p["signal"] for e, p in predictions.items()}

        diversity = measure_prediction_diversity(signals)

        return diversity > disagreement_threshold
