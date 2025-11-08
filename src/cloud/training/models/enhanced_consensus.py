"""
Enhanced Consensus System

Upgraded consensus with diversity weighting and minimum diversity score.
Down weights correlated engines and enforces minimum diversity to approve trades.

Key Features:
- Diversity weighting
- Correlation-based down-weighting
- Minimum diversity score requirement
- Engine correlation tracking
- Enhanced voting system

Author: Huracan Engine Team
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from enum import Enum

import numpy as np
import structlog

from .engine_consensus import EngineConsensus, ConsensusResult, ConsensusLevel, EngineOpinion

logger = structlog.get_logger(__name__)


@dataclass
class DiversityMetrics:
    """Diversity metrics for consensus"""
    diversity_score: float  # 0-1, higher is better
    correlation_matrix: Dict[str, Dict[str, float]]  # Engine correlation matrix
    unique_engines: int  # Number of unique engine types
    agreement_ratio: float  # Ratio of agreeing engines
    weighted_consensus: float  # Weighted consensus score


@dataclass
class EnhancedConsensusResult:
    """Enhanced consensus result with diversity metrics"""
    recommendation: str  # "TAKE_TRADE", "REDUCE_SIZE", "SKIP_TRADE"
    adjusted_confidence: float
    consensus_level: ConsensusLevel
    diversity_score: float
    diversity_metrics: DiversityMetrics
    reasoning: str
    warnings: List[str]
    engine_weights: Dict[str, float]  # Weight assigned to each engine
    correlation_penalties: Dict[str, float]  # Penalty for correlation


class EnhancedConsensus(EngineConsensus):
    """
    Enhanced consensus system with diversity weighting.
    
    Features:
    - Down weights correlated engines
    - Enforces minimum diversity score
    - Tracks engine correlations
    - Enhanced voting with diversity weighting
    
    Usage:
        consensus = EnhancedConsensus(
            min_diversity_score=0.5,
            correlation_threshold=0.7
        )
        
        result = consensus.analyze_consensus_with_diversity(
            primary_engine=TradingTechnique.TREND,
            primary_confidence=0.85,
            all_opinions=opinions,
            current_regime='trend'
        )
        
        if result.recommendation == 'TAKE_TRADE' and result.diversity_score >= 0.5:
            execute_trade()
    """
    
    def __init__(
        self,
        min_diversity_score: float = 0.5,  # Minimum diversity to approve trade
        correlation_threshold: float = 0.7,  # Threshold for high correlation
        correlation_penalty: float = 0.3,  # Penalty for correlated engines
        diversity_weight: float = 0.3,  # Weight of diversity in final decision
        **kwargs
    ):
        """
        Initialize enhanced consensus.
        
        Args:
            min_diversity_score: Minimum diversity score to approve trade
            correlation_threshold: Threshold for high correlation
            correlation_penalty: Penalty applied to correlated engines
            diversity_weight: Weight of diversity in final decision
            **kwargs: Additional arguments for base EngineConsensus
        """
        super().__init__(**kwargs)
        
        self.min_diversity_score = min_diversity_score
        self.correlation_threshold = correlation_threshold
        self.correlation_penalty = correlation_penalty
        self.diversity_weight = diversity_weight
        
        # Track engine correlations (simplified - would use historical data)
        self.engine_correlations: Dict[str, Dict[str, float]] = {}
        
        # Track engine performance history for correlation calculation
        self.engine_performance_history: Dict[str, List[float]] = {}
        
        logger.info(
            "enhanced_consensus_initialized",
            min_diversity_score=min_diversity_score,
            correlation_threshold=correlation_threshold,
            correlation_penalty=correlation_penalty
        )
    
    def analyze_consensus_with_diversity(
        self,
        primary_engine: any,  # TradingTechnique
        primary_confidence: float,
        all_opinions: List[EngineOpinion],
        current_regime: str = 'unknown'
    ) -> EnhancedConsensusResult:
        """
        Analyze consensus with diversity weighting.
        
        Args:
            primary_engine: Primary engine technique
            primary_confidence: Primary engine confidence
            all_opinions: All engine opinions
            current_regime: Current market regime
        
        Returns:
            EnhancedConsensusResult with diversity metrics
        """
        # Calculate engine correlations
        engine_correlations = self._calculate_engine_correlations(all_opinions)
        
        # Calculate diversity metrics
        diversity_metrics = self._calculate_diversity_metrics(
            all_opinions=all_opinions,
            engine_correlations=engine_correlations
        )
        
        # Calculate engine weights with diversity penalty
        engine_weights, correlation_penalties = self._calculate_diversity_weights(
            all_opinions=all_opinions,
            engine_correlations=engine_correlations
        )
        
        # Analyze consensus with diversity weighting
        base_result = self.analyze_consensus(
            primary_engine=primary_engine,
            primary_confidence=primary_confidence,
            all_opinions=all_opinions,
            current_regime=current_regime
        )
        
        # Adjust confidence based on diversity
        diversity_adjusted_confidence = self._adjust_confidence_for_diversity(
            base_confidence=base_result.adjusted_confidence,
            diversity_score=diversity_metrics.diversity_score,
            base_result=base_result
        )
        
        # Determine final recommendation
        recommendation = self._determine_recommendation(
            base_result=base_result,
            diversity_score=diversity_metrics.diversity_score,
            adjusted_confidence=diversity_adjusted_confidence
        )
        
        # Create enhanced result
        result = EnhancedConsensusResult(
            recommendation=recommendation,
            adjusted_confidence=diversity_adjusted_confidence,
            consensus_level=base_result.consensus_level,
            diversity_score=diversity_metrics.diversity_score,
            diversity_metrics=diversity_metrics,
            reasoning=self._build_reasoning(
                base_result=base_result,
                diversity_metrics=diversity_metrics,
                recommendation=recommendation
            ),
            warnings=base_result.warnings + self._get_diversity_warnings(diversity_metrics),
            engine_weights=engine_weights,
            correlation_penalties=correlation_penalties
        )
        
        logger.info(
            "enhanced_consensus_analysis",
            recommendation=recommendation,
            diversity_score=diversity_metrics.diversity_score,
            adjusted_confidence=diversity_adjusted_confidence,
            min_diversity_score=self.min_diversity_score
        )
        
        return result
    
    def _calculate_engine_correlations(
        self,
        all_opinions: List[EngineOpinion]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate correlations between engines"""
        # Simplified correlation calculation
        # In production, would use historical performance data
        
        correlations = {}
        engine_names = [op.technique.value for op in all_opinions]
        
        for engine1 in engine_names:
            correlations[engine1] = {}
            for engine2 in engine_names:
                if engine1 == engine2:
                    correlations[engine1][engine2] = 1.0
                elif engine1 in self.engine_correlations and engine2 in self.engine_correlations[engine1]:
                    correlations[engine1][engine2] = self.engine_correlations[engine1][engine2]
                else:
                    # Default correlation (would be calculated from historical data)
                    # Similar engines have higher correlation
                    if self._are_similar_engines(engine1, engine2):
                        correlations[engine1][engine2] = 0.6
                    else:
                        correlations[engine1][engine2] = 0.2
        
        return correlations
    
    def _are_similar_engines(self, engine1: str, engine2: str) -> bool:
        """Check if two engines are similar (simplified)"""
        # Group similar engines
        trend_engines = {"trend", "breakout", "leader"}
        mean_reversion_engines = {"range", "tape"}
        
        engine1_lower = engine1.lower()
        engine2_lower = engine2.lower()
        
        if engine1_lower in trend_engines and engine2_lower in trend_engines:
            return True
        if engine1_lower in mean_reversion_engines and engine2_lower in mean_reversion_engines:
            return True
        
        return False
    
    def _calculate_diversity_metrics(
        self,
        all_opinions: List[EngineOpinion],
        engine_correlations: Dict[str, Dict[str, float]]
    ) -> DiversityMetrics:
        """Calculate diversity metrics"""
        # Count unique engines
        unique_engines = len(set(op.technique.value for op in all_opinions))
        
        # Calculate agreement ratio
        directions = [op.direction for op in all_opinions]
        if directions:
            most_common = max(set(directions), key=directions.count)
            agreement_ratio = directions.count(most_common) / len(directions)
        else:
            agreement_ratio = 0.0
        
        # Calculate average correlation (lower is better for diversity)
        correlations_list = []
        engine_names = [op.technique.value for op in all_opinions]
        for i, engine1 in enumerate(engine_names):
            for j, engine2 in enumerate(engine_names):
                if i < j:  # Avoid duplicates
                    if engine1 in engine_correlations and engine2 in engine_correlations[engine1]:
                        correlations_list.append(engine_correlations[engine1][engine2])
        
        avg_correlation = np.mean(correlations_list) if correlations_list else 0.0
        
        # Diversity score: higher unique engines, lower correlation = higher diversity
        # Normalize to 0-1
        unique_score = min(1.0, unique_engines / 6.0)  # Assuming 6 engines max
        correlation_score = 1.0 - avg_correlation  # Lower correlation = higher score
        
        diversity_score = (unique_score * 0.5 + correlation_score * 0.5)
        
        # Calculate weighted consensus
        weighted_consensus = np.mean([op.confidence for op in all_opinions]) if all_opinions else 0.0
        
        return DiversityMetrics(
            diversity_score=diversity_score,
            correlation_matrix=engine_correlations,
            unique_engines=unique_engines,
            agreement_ratio=agreement_ratio,
            weighted_consensus=weighted_consensus
        )
    
    def _calculate_diversity_weights(
        self,
        all_opinions: List[EngineOpinion],
        engine_correlations: Dict[str, Dict[str, float]]
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Calculate engine weights with diversity penalty"""
        engine_weights = {}
        correlation_penalties = {}
        
        engine_names = [op.technique.value for op in all_opinions]
        
        for engine_name in engine_names:
            # Base weight
            base_weight = 1.0 / len(engine_names)
            
            # Calculate correlation penalty
            correlations = []
            for other_engine in engine_names:
                if engine_name != other_engine:
                    if engine_name in engine_correlations and other_engine in engine_correlations[engine_name]:
                        corr = engine_correlations[engine_name][other_engine]
                        if corr > self.correlation_threshold:
                            correlations.append(corr)
            
            # Penalty increases with number of high correlations
            penalty = min(self.correlation_penalty, len(correlations) * 0.1)
            correlation_penalties[engine_name] = penalty
            
            # Apply penalty
            engine_weights[engine_name] = base_weight * (1.0 - penalty)
        
        # Normalize weights to sum to 1.0
        total_weight = sum(engine_weights.values())
        if total_weight > 0:
            engine_weights = {k: v / total_weight for k, v in engine_weights.items()}
        
        return engine_weights, correlation_penalties
    
    def _adjust_confidence_for_diversity(
        self,
        base_confidence: float,
        diversity_score: float,
        base_result: ConsensusResult
    ) -> float:
        """Adjust confidence based on diversity"""
        # Higher diversity = boost confidence
        # Lower diversity = reduce confidence
        diversity_boost = (diversity_score - 0.5) * 0.2  # Max ±0.1 adjustment
        
        adjusted_confidence = base_confidence + diversity_boost
        
        # Clip to valid range
        adjusted_confidence = max(0.0, min(1.0, adjusted_confidence))
        
        return adjusted_confidence
    
    def _determine_recommendation(
        self,
        base_result: ConsensusResult,
        diversity_score: float,
        adjusted_confidence: float
    ) -> str:
        """Determine final recommendation with diversity check"""
        # Must meet minimum diversity score
        if diversity_score < self.min_diversity_score:
            return "SKIP_TRADE"
        
        # Use base recommendation if diversity is good
        if diversity_score >= self.min_diversity_score:
            return base_result.recommendation
        
        # Default to skip if diversity is too low
        return "SKIP_TRADE"
    
    def _build_reasoning(
        self,
        base_result: ConsensusResult,
        diversity_metrics: DiversityMetrics,
        recommendation: str
    ) -> str:
        """Build reasoning string"""
        reasoning_parts = [base_result.reasoning]
        
        if diversity_metrics.diversity_score < self.min_diversity_score:
            reasoning_parts.append(
                f"Diversity score {diversity_metrics.diversity_score:.2f} below minimum {self.min_diversity_score:.2f}"
            )
        else:
            reasoning_parts.append(
                f"Diversity score {diversity_metrics.diversity_score:.2f} meets requirement"
            )
        
        return "; ".join(reasoning_parts)
    
    def _get_diversity_warnings(self, diversity_metrics: DiversityMetrics) -> List[str]:
        """Get diversity warnings"""
        warnings = []
        
        if diversity_metrics.diversity_score < self.min_diversity_score:
            warnings.append(
                f"Low diversity: {diversity_metrics.diversity_score:.2f} < {self.min_diversity_score:.2f}"
            )
        
        if diversity_metrics.unique_engines < 3:
            warnings.append(f"Too few unique engines: {diversity_metrics.unique_engines}")
        
        # Check for high correlations
        high_correlations = []
        for engine1, correlations in diversity_metrics.correlation_matrix.items():
            for engine2, corr in correlations.items():
                if engine1 != engine2 and corr > self.correlation_threshold:
                    high_correlations.append(f"{engine1}-{engine2}: {corr:.2f}")
        
        if high_correlations:
            warnings.append(f"High correlations: {', '.join(high_correlations[:3])}")
        
        return warnings

