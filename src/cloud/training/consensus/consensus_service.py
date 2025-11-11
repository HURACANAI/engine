"""
Consensus Service for Training Architecture

Produces a single consensus score S by weighting engine signals
by recent reliability and penalizing correlation.

Author: Huracan Engine Team
Date: 2025-01-27
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
import numpy as np

import structlog

logger = structlog.get_logger(__name__)


class ConsensusLevel(Enum):
    """Consensus level."""
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"
    DIVIDED = "divided"


@dataclass
class EngineVote:
    """Engine vote."""
    engine_type: str
    direction: str  # "long", "short", "hold"
    confidence: float  # 0.0 to 1.0
    edge_bps: float
    reliability_score: float = 1.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class ConsensusResult:
    """Consensus result."""
    consensus_score: float  # Single score S
    consensus_level: ConsensusLevel
    direction: str  # "long", "short", "hold"
    confidence: float
    votes: List[EngineVote]
    weighted_sum: float
    correlation_penalty: float
    reliability_weights: Dict[str, float]
    reasoning: str


class ReliabilityTracker:
    """Tracks engine reliability over time."""
    
    def __init__(self, window_size: int = 100):
        """
        Initialize reliability tracker.
        
        Args:
            window_size: Number of recent predictions to track
        """
        self.window_size = window_size
        self.engine_performance: Dict[str, List[tuple[float, bool]]] = {}  # engine -> [(timestamp, correct), ...]
        logger.info("reliability_tracker_initialized", window_size=window_size)
    
    def record_prediction(
        self,
        engine_type: str,
        correct: bool,
        timestamp: Optional[float] = None,
    ) -> None:
        """
        Record a prediction outcome.
        
        Args:
            engine_type: Engine type
            correct: Whether prediction was correct
            timestamp: Timestamp (default: current time)
        """
        if timestamp is None:
            timestamp = time.time()
        
        if engine_type not in self.engine_performance:
            self.engine_performance[engine_type] = []
        
        self.engine_performance[engine_type].append((timestamp, correct))
        
        # Keep only recent predictions
        if len(self.engine_performance[engine_type]) > self.window_size:
            self.engine_performance[engine_type] = self.engine_performance[engine_type][-self.window_size:]
    
    def get_reliability_score(self, engine_type: str) -> float:
        """
        Get reliability score for an engine.
        
        Returns:
            Reliability score between 0.0 and 1.0
        """
        if engine_type not in self.engine_performance:
            return 1.0  # Default reliability for new engines
        
        performances = self.engine_performance[engine_type]
        if not performances:
            return 1.0
        
        # Calculate win rate
        correct_count = sum(1 for _, correct in performances if correct)
        total_count = len(performances)
        win_rate = correct_count / total_count if total_count > 0 else 0.0
        
        # Reliability = win rate with confidence based on sample size
        confidence = min(1.0, total_count / self.window_size)
        reliability = win_rate * confidence + 0.5 * (1 - confidence)  # Bayesian prior
        
        return reliability
    
    def get_all_reliability_scores(self) -> Dict[str, float]:
        """Get reliability scores for all engines."""
        return {
            engine_type: self.get_reliability_score(engine_type)
            for engine_type in self.engine_performance.keys()
        }


class CorrelationCalculator:
    """Calculates correlation between engine predictions."""
    
    def __init__(self, window_size: int = 100):
        """
        Initialize correlation calculator.
        
        Args:
            window_size: Number of recent predictions to track
        """
        self.window_size = window_size
        self.engine_predictions: Dict[str, List[float]] = {}  # engine -> [direction_signals, ...]
        logger.info("correlation_calculator_initialized", window_size=window_size)
    
    def record_predictions(self, votes: List[EngineVote]) -> None:
        """
        Record predictions for correlation calculation.
        
        Args:
            votes: List of engine votes
        """
        for vote in votes:
            if vote.engine_type not in self.engine_predictions:
                self.engine_predictions[vote.engine_type] = []
            
            # Convert direction to signal: long=1, short=-1, hold=0
            signal = 1.0 if vote.direction == "long" else (-1.0 if vote.direction == "short" else 0.0)
            signal *= vote.confidence
            
            self.engine_predictions[vote.engine_type].append(signal)
            
            # Keep only recent predictions
            if len(self.engine_predictions[vote.engine_type]) > self.window_size:
                self.engine_predictions[vote.engine_type] = self.engine_predictions[vote.engine_type][-self.window_size:]
    
    def calculate_correlation_penalty(self, votes: List[EngineVote]) -> float:
        """
        Calculate correlation penalty for votes.
        
        Returns:
            Penalty between 0.0 and 1.0 (higher = more correlation = more penalty)
        """
        if len(votes) < 2:
            return 0.0
        
        # Get prediction signals for each engine
        signals = []
        for vote in votes:
            if vote.engine_type in self.engine_predictions:
                recent_signals = self.engine_predictions[vote.engine_type]
                if recent_signals:
                    signals.append(np.array(recent_signals))
        
        if len(signals) < 2:
            return 0.0
        
        # Calculate pairwise correlations
        correlations = []
        for i in range(len(signals)):
            for j in range(i + 1, len(signals)):
                if len(signals[i]) == len(signals[j]):
                    corr = np.corrcoef(signals[i], signals[j])[0, 1]
                    if not np.isnan(corr):
                        correlations.append(abs(corr))
        
        if not correlations:
            return 0.0
        
        # Penalty = average correlation (higher correlation = higher penalty)
        avg_correlation = np.mean(correlations)
        return avg_correlation
    
    def get_correlation_matrix(self) -> Dict[str, Dict[str, float]]:
        """Get correlation matrix between engines."""
        matrix = {}
        engines = list(self.engine_predictions.keys())
        
        for i, engine1 in enumerate(engines):
            matrix[engine1] = {}
            for j, engine2 in enumerate(engines):
                if i == j:
                    matrix[engine1][engine2] = 1.0
                else:
                    signals1 = np.array(self.engine_predictions[engine1])
                    signals2 = np.array(self.engine_predictions[engine2])
                    if len(signals1) == len(signals2) and len(signals1) > 0:
                        corr = np.corrcoef(signals1, signals2)[0, 1]
                        matrix[engine1][engine2] = corr if not np.isnan(corr) else 0.0
                    else:
                        matrix[engine1][engine2] = 0.0
        
        return matrix


class ConsensusService:
    """
    Consensus service for combining engine signals.
    
    Features:
    - Reliability-weighted voting
    - Correlation penalty
    - Adaptive threshold
    - Consensus level classification
    """
    
    def __init__(
        self,
        adaptive_threshold: bool = True,
        min_consensus_score: float = 0.5,
        correlation_penalty_weight: float = 0.3,
    ):
        """
        Initialize consensus service.
        
        Args:
            adaptive_threshold: Whether to use adaptive threshold
            min_consensus_score: Minimum consensus score for action
            correlation_penalty_weight: Weight for correlation penalty
        """
        self.adaptive_threshold = adaptive_threshold
        self.min_consensus_score = min_consensus_score
        self.correlation_penalty_weight = correlation_penalty_weight
        
        self.reliability_tracker = ReliabilityTracker()
        self.correlation_calculator = CorrelationCalculator()
        
        logger.info(
            "consensus_service_initialized",
            adaptive_threshold=adaptive_threshold,
            min_consensus_score=min_consensus_score,
        )
    
    def compute_consensus(self, votes: List[EngineVote]) -> ConsensusResult:
        """
        Compute consensus from engine votes.
        
        Args:
            votes: List of engine votes
        
        Returns:
            ConsensusResult with consensus score and reasoning
        """
        if not votes:
            return ConsensusResult(
                consensus_score=0.0,
                consensus_level=ConsensusLevel.DIVIDED,
                direction="hold",
                confidence=0.0,
                votes=[],
                weighted_sum=0.0,
                correlation_penalty=0.0,
                reliability_weights={},
                reasoning="No votes",
            )
        
        # Get reliability weights
        reliability_weights = {}
        for vote in votes:
            reliability = self.reliability_tracker.get_reliability_score(vote.engine_type)
            reliability_weights[vote.engine_type] = reliability
            vote.reliability_score = reliability
        
        # Calculate weighted sum
        weighted_sum = 0.0
        total_weight = 0.0
        
        for vote in votes:
            # Weight = reliability * confidence
            weight = vote.reliability_score * vote.confidence
            total_weight += weight
            
            # Direction signal: long=1, short=-1, hold=0
            signal = 1.0 if vote.direction == "long" else (-1.0 if vote.direction == "short" else 0.0)
            weighted_sum += signal * weight
        
        # Normalize weighted sum
        if total_weight > 0:
            normalized_sum = weighted_sum / total_weight
        else:
            normalized_sum = 0.0
        
        # Calculate correlation penalty
        correlation_penalty = self.correlation_calculator.calculate_correlation_penalty(votes)
        
        # Apply correlation penalty
        penalized_sum = normalized_sum * (1.0 - correlation_penalty * self.correlation_penalty_weight)
        
        # Consensus score = absolute value of penalized sum
        consensus_score = abs(penalized_sum)
        
        # Determine direction
        if penalized_sum > 0.1:
            direction = "long"
        elif penalized_sum < -0.1:
            direction = "short"
        else:
            direction = "hold"
        
        # Determine consensus level
        if consensus_score >= 0.7:
            consensus_level = ConsensusLevel.STRONG
        elif consensus_score >= 0.5:
            consensus_level = ConsensusLevel.MODERATE
        elif consensus_score >= 0.3:
            consensus_level = ConsensusLevel.WEAK
        else:
            consensus_level = ConsensusLevel.DIVIDED
        
        # Calculate confidence (average of vote confidences weighted by reliability)
        if total_weight > 0:
            confidence = sum(
                vote.confidence * vote.reliability_score for vote in votes
            ) / total_weight
        else:
            confidence = 0.0
        
        # Apply adaptive threshold
        if self.adaptive_threshold:
            # Adjust threshold based on recent performance
            recent_performance = self._get_recent_performance()
            if recent_performance < 0.5:
                # Lower threshold if performance is poor
                effective_threshold = self.min_consensus_score * 0.8
            else:
                effective_threshold = self.min_consensus_score
        else:
            effective_threshold = self.min_consensus_score
        
        # Check if consensus meets threshold
        if consensus_score < effective_threshold:
            direction = "hold"
            consensus_level = ConsensusLevel.DIVIDED
        
        # Generate reasoning
        reasoning = self._generate_reasoning(
            votes, consensus_score, consensus_level, direction,
            reliability_weights, correlation_penalty
        )
        
        # Record predictions for correlation calculation
        self.correlation_calculator.record_predictions(votes)
        
        return ConsensusResult(
            consensus_score=consensus_score,
            consensus_level=consensus_level,
            direction=direction,
            confidence=confidence,
            votes=votes,
            weighted_sum=weighted_sum,
            correlation_penalty=correlation_penalty,
            reliability_weights=reliability_weights,
            reasoning=reasoning,
        )
    
    def _get_recent_performance(self) -> float:
        """Get recent performance across all engines."""
        all_scores = self.reliability_tracker.get_all_reliability_scores()
        if not all_scores:
            return 0.5  # Default performance
        
        return sum(all_scores.values()) / len(all_scores)
    
    def _generate_reasoning(
        self,
        votes: List[EngineVote],
        consensus_score: float,
        consensus_level: ConsensusLevel,
        direction: str,
        reliability_weights: Dict[str, float],
        correlation_penalty: float,
    ) -> str:
        """Generate reasoning for consensus result."""
        reasoning_parts = [
            f"Consensus: {consensus_level.value} ({consensus_score:.2f})",
            f"Direction: {direction}",
            f"Votes: {len(votes)}",
            f"Reliability weights: {', '.join(f'{k}={v:.2f}' for k, v in reliability_weights.items())}",
            f"Correlation penalty: {correlation_penalty:.2f}",
        ]
        return "; ".join(reasoning_parts)
    
    def record_outcome(
        self,
        votes: List[EngineVote],
        correct: bool,
        timestamp: Optional[float] = None,
    ) -> None:
        """
        Record outcome for reliability tracking.
        
        Args:
            votes: List of engine votes
            correct: Whether prediction was correct
            timestamp: Timestamp (default: current time)
        """
        for vote in votes:
            self.reliability_tracker.record_prediction(
                vote.engine_type,
                correct,
                timestamp,
            )
    
    def get_reliability_scores(self) -> Dict[str, float]:
        """Get reliability scores for all engines."""
        return self.reliability_tracker.get_all_reliability_scores()
    
    def get_correlation_matrix(self) -> Dict[str, Dict[str, float]]:
        """Get correlation matrix between engines."""
        return self.correlation_calculator.get_correlation_matrix()

