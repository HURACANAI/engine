"""
Enhanced Consensus Engine

Combines 23 engine votes with reliability weights, correlation penalties,
and adaptive thresholds. Implements sophisticated consensus voting with
regime-aware adjustments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple
from datetime import datetime, timezone

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


class MarketRegime(str, Enum):
    """Market regime types."""
    TREND = "TREND"
    RANGE = "RANGE"
    PANIC = "PANIC"
    ILLIQUID = "ILLIQUID"
    UNKNOWN = "UNKNOWN"


@dataclass
class EngineVote:
    """Individual engine vote."""
    engine_id: str
    signal: int  # -1 (sell), 0 (hold), +1 (buy)
    confidence: float  # 0.0 to 1.0
    raw_score: float  # Raw prediction score
    regime_affinity: Dict[MarketRegime, float] = field(default_factory=dict)  # Affinity per regime


@dataclass
class ConsensusResult:
    """Consensus voting result."""
    consensus_signal: int  # -1, 0, +1
    consensus_confidence: float  # 0.0 to 1.0
    weighted_score: float  # Weighted average of signals
    num_engines: int
    num_agree: int
    agreement_ratio: float
    engine_weights: Dict[str, float]  # Final weights used
    correlation_penalties: Dict[str, float]  # Penalties applied
    regime: MarketRegime
    metadata: Dict[str, Any] = field(default_factory=dict)


class EnhancedConsensusEngine:
    """
    Enhanced consensus engine with reliability weights and correlation penalties.
    
    Features:
    - Reliability-weighted voting based on historical performance
    - Correlation penalties for similar engines
    - Adaptive thresholds per regime
    - Confidence calculation based on agreement
    """
    
    def __init__(
        self,
        num_engines: int = 23,
        min_agreement_threshold: float = 0.6,
        reliability_decay_factor: float = 0.95,
        correlation_penalty_threshold: float = 0.8,
        adaptive_thresholds: Optional[Dict[MarketRegime, float]] = None,
    ) -> None:
        """
        Initialize enhanced consensus engine.
        
        Args:
            num_engines: Total number of engines (default: 23)
            min_agreement_threshold: Minimum agreement ratio for high confidence
            reliability_decay_factor: Decay factor for reliability over time
            correlation_penalty_threshold: Correlation threshold for penalty
            adaptive_thresholds: Regime-specific confidence thresholds
        """
        self.num_engines = num_engines
        self.min_agreement_threshold = min_agreement_threshold
        self.reliability_decay_factor = reliability_decay_factor
        self.correlation_penalty_threshold = correlation_penalty_threshold
        
        # Adaptive thresholds per regime
        self.adaptive_thresholds = adaptive_thresholds or {
            MarketRegime.TREND: 0.5,
            MarketRegime.RANGE: 0.55,
            MarketRegime.PANIC: 0.65,
            MarketRegime.ILLIQUID: 0.7,
            MarketRegime.UNKNOWN: 0.6,
        }
        
        # Engine reliability tracking (engine_id -> reliability_score)
        self.engine_reliability: Dict[str, float] = {}
        
        # Engine correlation matrix (engine_id -> {engine_id -> correlation})
        self.engine_correlations: Dict[str, Dict[str, float]] = {}
        
        # Performance history for reliability calculation
        self.performance_history: Dict[str, List[Tuple[float, float]]] = {}  # engine_id -> [(pnl, confidence), ...]
        
        logger.info(
            "enhanced_consensus_initialized",
            num_engines=num_engines,
            min_agreement_threshold=min_agreement_threshold,
            adaptive_thresholds={k.value: v for k, v in self.adaptive_thresholds.items()},
        )
    
    def generate_consensus(
        self,
        votes: List[EngineVote],
        current_regime: MarketRegime,
        update_reliability: bool = True,
    ) -> ConsensusResult:
        """
        Generate consensus from engine votes.
        
        Args:
            votes: List of engine votes
            current_regime: Current market regime
            update_reliability: Whether to update reliability scores after consensus
        
        Returns:
            Consensus result with signal, confidence, and metadata
        """
        if not votes:
            return self._empty_consensus(current_regime)
        
        # Calculate reliability weights
        reliability_weights = self._calculate_reliability_weights(votes)
        
        # Calculate correlation penalties
        correlation_penalties = self._calculate_correlation_penalties(votes)
        
        # Apply penalties to weights
        adjusted_weights = self._apply_correlation_penalties(
            reliability_weights,
            correlation_penalties,
        )
        
        # Normalize weights
        total_weight = sum(adjusted_weights.values())
        if total_weight > 0:
            normalized_weights = {
                engine_id: weight / total_weight
                for engine_id, weight in adjusted_weights.items()
            }
        else:
            # Equal weights if all weights are zero
            normalized_weights = {
                vote.engine_id: 1.0 / len(votes)
                for vote in votes
            }
        
        # Calculate weighted consensus
        weighted_scores = []
        signals = []
        
        for vote in votes:
            weight = normalized_weights.get(vote.engine_id, 0.0)
            weighted_score = vote.signal * vote.confidence * weight
            weighted_scores.append(weighted_score)
            signals.append(vote.signal)
        
        consensus_score = sum(weighted_scores)
        
        # Determine consensus signal
        if consensus_score > 0.1:
            consensus_signal = 1  # Buy
        elif consensus_score < -0.1:
            consensus_signal = -1  # Sell
        else:
            consensus_signal = 0  # Hold
        
        # Calculate agreement
        num_agree = sum(1 for s in signals if s == consensus_signal)
        agreement_ratio = num_agree / len(signals) if signals else 0.0
        
        # Calculate confidence
        consensus_confidence = self._calculate_consensus_confidence(
            consensus_score,
            agreement_ratio,
            votes,
            normalized_weights,
            current_regime,
        )
        
        # Apply adaptive threshold
        regime_threshold = self.adaptive_thresholds.get(current_regime, 0.6)
        if consensus_confidence < regime_threshold:
            # Below threshold, reduce confidence or change to hold
            if consensus_signal != 0:
                consensus_signal = 0
                consensus_confidence = 0.0
        
        result = ConsensusResult(
            consensus_signal=consensus_signal,
            consensus_confidence=consensus_confidence,
            weighted_score=consensus_score,
            num_engines=len(votes),
            num_agree=num_agree,
            agreement_ratio=agreement_ratio,
            engine_weights=normalized_weights,
            correlation_penalties=correlation_penalties,
            regime=current_regime,
            metadata={
                "reliability_weights": reliability_weights,
                "regime_threshold": regime_threshold,
            },
        )
        
        logger.debug(
            "consensus_generated",
            consensus_signal=consensus_signal,
            consensus_confidence=consensus_confidence,
            agreement_ratio=agreement_ratio,
            num_engines=len(votes),
            regime=current_regime.value,
        )
        
        return result
    
    def _calculate_reliability_weights(
        self,
        votes: List[EngineVote],
    ) -> Dict[str, float]:
        """
        Calculate reliability weights for each engine.
        
        Reliability is based on:
        - Historical performance (Sharpe ratio, win rate)
        - Recent performance (exponential decay)
        - Regime-specific performance
        
        Args:
            votes: List of engine votes
        
        Returns:
            Dictionary of engine_id -> reliability_weight
        """
        weights = {}
        
        for vote in votes:
            engine_id = vote.engine_id
            
            # Get base reliability (default to 0.5 if unknown)
            base_reliability = self.engine_reliability.get(engine_id, 0.5)
            
            # Adjust for regime affinity
            regime_affinity = vote.regime_affinity.get(vote.regime_affinity, 1.0) if vote.regime_affinity else 1.0
            
            # Calculate final reliability
            reliability = base_reliability * regime_affinity
            
            # Ensure reliability is in [0, 1]
            reliability = max(0.0, min(1.0, reliability))
            
            weights[engine_id] = reliability
        
        return weights
    
    def _calculate_correlation_penalties(
        self,
        votes: List[EngineVote],
    ) -> Dict[str, float]:
        """
        Calculate correlation penalties for engines.
        
        Engines with high correlation get penalized to avoid over-weighting
        similar strategies.
        
        Args:
            votes: List of engine votes
        
        Returns:
            Dictionary of engine_id -> penalty_factor (0.0 to 1.0)
        """
        penalties = {}
        
        for vote in votes:
            engine_id = vote.engine_id
            penalty = 1.0  # No penalty by default
            
            # Check correlations with other engines
            if engine_id in self.engine_correlations:
                correlations = self.engine_correlations[engine_id]
                
                # Find highly correlated engines in current vote set
                for other_vote in votes:
                    if other_vote.engine_id == engine_id:
                        continue
                    
                    correlation = correlations.get(other_vote.engine_id, 0.0)
                    
                    # Apply penalty if correlation is high
                    if abs(correlation) > self.correlation_penalty_threshold:
                        # Penalty increases with correlation
                        penalty_factor = 1.0 - (abs(correlation) - self.correlation_penalty_threshold) / (1.0 - self.correlation_penalty_threshold)
                        penalty = min(penalty, penalty_factor)
            
            penalties[engine_id] = max(0.0, penalty)
        
        return penalties
    
    def _apply_correlation_penalties(
        self,
        reliability_weights: Dict[str, float],
        correlation_penalties: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Apply correlation penalties to reliability weights.
        
        Args:
            reliability_weights: Base reliability weights
            correlation_penalties: Correlation penalty factors
        
        Returns:
            Adjusted weights
        """
        adjusted_weights = {}
        
        for engine_id, weight in reliability_weights.items():
            penalty = correlation_penalties.get(engine_id, 1.0)
            adjusted_weights[engine_id] = weight * penalty
        
        return adjusted_weights
    
    def _calculate_consensus_confidence(
        self,
        consensus_score: float,
        agreement_ratio: float,
        votes: List[EngineVote],
        weights: Dict[str, float],
        regime: MarketRegime,
    ) -> float:
        """
        Calculate consensus confidence.
        
        Confidence is based on:
        - Agreement ratio (how many engines agree)
        - Weighted confidence of agreeing engines
        - Signal strength (magnitude of consensus_score)
        
        Args:
            consensus_score: Weighted consensus score
            agreement_ratio: Ratio of engines agreeing
            votes: List of engine votes
            weights: Engine weights
            regime: Current market regime
        
        Returns:
            Confidence score (0.0 to 1.0)
        """
        # Base confidence from agreement ratio
        agreement_confidence = agreement_ratio
        
        # Weighted confidence from agreeing engines
        weighted_confidence = 0.0
        total_weight = 0.0
        
        consensus_signal = 1 if consensus_score > 0.1 else (-1 if consensus_score < -0.1 else 0)
        
        for vote in votes:
            if vote.signal == consensus_signal:
                weight = weights.get(vote.engine_id, 0.0)
                weighted_confidence += vote.confidence * weight
                total_weight += weight
        
        if total_weight > 0:
            weighted_confidence = weighted_confidence / total_weight
        else:
            weighted_confidence = 0.0
        
        # Signal strength (magnitude of consensus_score)
        signal_strength = min(1.0, abs(consensus_score))
        
        # Combine factors
        confidence = (
            0.4 * agreement_confidence +
            0.4 * weighted_confidence +
            0.2 * signal_strength
        )
        
        # Regime adjustment (higher threshold in uncertain regimes)
        regime_adjustment = 1.0
        if regime == MarketRegime.PANIC:
            regime_adjustment = 0.9  # Reduce confidence in panic
        elif regime == MarketRegime.ILLIQUID:
            regime_adjustment = 0.8  # Reduce confidence in illiquid
        
        confidence = confidence * regime_adjustment
        
        return max(0.0, min(1.0, confidence))
    
    def update_reliability(
        self,
        engine_id: str,
        pnl: float,
        confidence: float,
        regime: MarketRegime,
    ) -> None:
        """
        Update engine reliability based on performance.
        
        Args:
            engine_id: Engine identifier
            pnl: Profit and loss from the trade
            confidence: Confidence level used
            regime: Market regime during the trade
        """
        # Initialize if needed
        if engine_id not in self.performance_history:
            self.performance_history[engine_id] = []
            self.engine_reliability[engine_id] = 0.5
        
        # Add performance record
        self.performance_history[engine_id].append((pnl, confidence))
        
        # Keep only recent history (last 100 trades)
        if len(self.performance_history[engine_id]) > 100:
            self.performance_history[engine_id] = self.performance_history[engine_id][-100:]
        
        # Calculate reliability from performance
        history = self.performance_history[engine_id]
        
        if len(history) < 5:
            # Not enough data, keep default
            return
        
        # Calculate Sharpe-like metric
        pnls = [h[0] for h in history]
        mean_pnl = np.mean(pnls)
        std_pnl = np.std(pnls) if len(pnls) > 1 else 1.0
        
        if std_pnl > 0:
            sharpe_like = mean_pnl / std_pnl
        else:
            sharpe_like = 0.0
        
        # Convert to reliability score (0 to 1)
        # Positive Sharpe -> high reliability, negative -> low reliability
        reliability = 0.5 + 0.5 * np.tanh(sharpe_like)  # Tanh maps to [0, 1]
        
        # Apply exponential decay to old reliability
        old_reliability = self.engine_reliability[engine_id]
        new_reliability = (
            self.reliability_decay_factor * old_reliability +
            (1 - self.reliability_decay_factor) * reliability
        )
        
        self.engine_reliability[engine_id] = new_reliability
        
        logger.debug(
            "reliability_updated",
            engine_id=engine_id,
            old_reliability=old_reliability,
            new_reliability=new_reliability,
            sharpe_like=sharpe_like,
        )
    
    def update_correlation(
        self,
        engine_id_1: str,
        engine_id_2: str,
        correlation: float,
    ) -> None:
        """
        Update correlation between two engines.
        
        Args:
            engine_id_1: First engine ID
            engine_id_2: Second engine ID
            correlation: Correlation coefficient (-1 to 1)
        """
        if engine_id_1 not in self.engine_correlations:
            self.engine_correlations[engine_id_1] = {}
        
        if engine_id_2 not in self.engine_correlations:
            self.engine_correlations[engine_id_2] = {}
        
        self.engine_correlations[engine_id_1][engine_id_2] = correlation
        self.engine_correlations[engine_id_2][engine_id_1] = correlation
    
    def _empty_consensus(self, regime: MarketRegime) -> ConsensusResult:
        """Return empty consensus result."""
        return ConsensusResult(
            consensus_signal=0,
            consensus_confidence=0.0,
            weighted_score=0.0,
            num_engines=0,
            num_agree=0,
            agreement_ratio=0.0,
            engine_weights={},
            correlation_penalties={},
            regime=regime,
        )
    
    def get_engine_reliability(self, engine_id: str) -> float:
        """Get current reliability for an engine."""
        return self.engine_reliability.get(engine_id, 0.5)
    
    def get_all_reliabilities(self) -> Dict[str, float]:
        """Get all engine reliabilities."""
        return self.engine_reliability.copy()

