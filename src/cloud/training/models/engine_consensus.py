"""
Engine Consensus System

Gets "second opinion" from all 6 alpha engines before making trades.
Prevents overconfidence by requiring agreement across multiple trading strategies.

Key Philosophy:
- One engine may be overconfident or wrong
- Multiple engines agreeing = stronger signal
- Disagreement = warning sign, reduce position or skip
- Regime-specific consensus requirements

The 6 Alpha Engines:
1. TREND: Follows sustained directional moves
2. RANGE: Mean reversion in sideways markets
3. BREAKOUT: Trades breakouts from consolidation
4. TAPE: Reads order flow and market microstructure
5. LEADER: Follows relative strength leaders
6. SWEEP: Detects liquidity sweeps and stop hunts

Example:
    Engine consensus prevents single-engine mistakes:

    TREND engine: BUY ETH (confidence: 0.85)
    RANGE engine: SELL ETH (confidence: 0.70)
    BREAKOUT engine: NEUTRAL (confidence: 0.45)

    → Disagreement detected!
    → Reduce confidence from 0.85 to 0.60
    → Skip trade (below 0.65 threshold)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


class TradingTechnique(Enum):
    """The 6 alpha engines."""

    TREND = "trend"
    RANGE = "range"
    BREAKOUT = "breakout"
    TAPE = "tape"
    LEADER = "leader"
    SWEEP = "sweep"


class ConsensusLevel(Enum):
    """Consensus strength levels."""

    UNANIMOUS = "unanimous"  # All engines agree
    STRONG = "strong"  # 75%+ agreement
    MODERATE = "moderate"  # 60-75% agreement
    WEAK = "weak"  # 50-60% agreement
    DIVIDED = "divided"  # <50% agreement - dangerous!


@dataclass
class EngineOpinion:
    """Opinion from a single alpha engine."""

    technique: TradingTechnique
    direction: str  # 'buy', 'sell', 'neutral'
    confidence: float  # 0-1
    reasoning: str
    supporting_factors: List[str]


@dataclass
class ConsensusResult:
    """Result of consensus analysis."""

    primary_direction: str  # 'buy', 'sell', 'neutral'
    consensus_level: ConsensusLevel
    agreement_score: float  # 0-1, higher = more agreement
    adjusted_confidence: float  # Original confidence adjusted by consensus
    participating_engines: List[TradingTechnique]
    agreeing_engines: List[TradingTechnique]
    disagreeing_engines: List[TradingTechnique]
    neutral_engines: List[TradingTechnique]
    recommendation: str  # 'TAKE_TRADE', 'REDUCE_SIZE', 'SKIP_TRADE'
    reasoning: str
    warnings: List[str]


class EngineConsensus:
    """
    Manages consensus between multiple alpha engines.

    The consensus system acts as a "council of experts" where each
    alpha engine gets a vote. Strong agreement boosts confidence,
    while disagreement reduces it or blocks the trade entirely.

    Consensus Requirements by Regime:

    TREND Regime:
    - Need TREND engine + 1 other engine agreement
    - RANGE engine disagreement is warning (counter-trend)
    - BREAKOUT engine agreement is strong boost

    RANGE Regime:
    - Need RANGE engine + 1 other engine agreement
    - TREND engine disagreement is warning (trend forming)
    - SWEEP engine agreement is strong boost

    PANIC Regime:
    - Need 3+ engines agreeing (high bar in panic)
    - Any disagreement = skip trade (too risky)

    Usage:
        consensus = EngineConsensus()

        # Collect opinions from all engines
        opinions = [
            EngineOpinion(technique=TradingTechnique.TREND,
                         direction='buy', confidence=0.85, ...),
            EngineOpinion(technique=TradingTechnique.RANGE,
                         direction='sell', confidence=0.70, ...),
            # ... other engines
        ]

        # Analyze consensus
        result = consensus.analyze_consensus(
            primary_engine=TradingTechnique.TREND,
            primary_confidence=0.85,
            all_opinions=opinions,
            current_regime='trend',
        )

        if result.recommendation == 'TAKE_TRADE':
            execute_trade(confidence=result.adjusted_confidence)
        elif result.recommendation == 'REDUCE_SIZE':
            execute_trade(size=size * 0.5, confidence=result.adjusted_confidence)
        else:
            skip_trade()
    """

    def __init__(
        self,
        unanimous_boost: float = 0.10,
        strong_boost: float = 0.05,
        moderate_penalty: float = 0.0,
        weak_penalty: float = -0.05,
        divided_penalty: float = -0.15,
        min_confidence_after_adjustment: float = 0.55,
        min_participating_engines: int = 3,
    ):
        """
        Initialize consensus system.

        Args:
            unanimous_boost: Confidence boost for unanimous agreement
            strong_boost: Confidence boost for strong agreement (75%+)
            moderate_penalty: Adjustment for moderate agreement (60-75%)
            weak_penalty: Confidence penalty for weak agreement (50-60%)
            divided_penalty: Confidence penalty for divided opinion (<50%)
            min_confidence_after_adjustment: Minimum confidence after consensus adjustment
            min_participating_engines: Minimum engines that must have opinions
        """
        self.unanimous_boost = unanimous_boost
        self.strong_boost = strong_boost
        self.moderate_penalty = moderate_penalty
        self.weak_penalty = weak_penalty
        self.divided_penalty = divided_penalty
        self.min_confidence = min_confidence_after_adjustment
        self.min_participating = min_participating_engines

        logger.info(
            "engine_consensus_initialized",
            unanimous_boost=unanimous_boost,
            strong_boost=strong_boost,
            divided_penalty=divided_penalty,
        )

    def analyze_consensus(
        self,
        primary_engine: TradingTechnique,
        primary_direction: str,
        primary_confidence: float,
        all_opinions: List[EngineOpinion],
        current_regime: str,
    ) -> ConsensusResult:
        """
        Analyze consensus across all alpha engines.

        Args:
            primary_engine: Engine that generated the primary signal
            primary_direction: Direction of primary signal ('buy'/'sell')
            primary_confidence: Confidence of primary signal
            all_opinions: Opinions from all engines
            current_regime: Current market regime

        Returns:
            ConsensusResult with agreement analysis and adjusted confidence
        """
        # Filter out neutral opinions and categorize
        participating = [op for op in all_opinions if op.direction != 'neutral']
        neutral = [op for op in all_opinions if op.direction == 'neutral']

        # Check minimum participation
        if len(participating) < self.min_participating:
            logger.warning(
                "insufficient_engine_participation",
                participating=len(participating),
                required=self.min_participating,
            )
            return self._create_skip_result(
                primary_direction=primary_direction,
                primary_confidence=primary_confidence,
                reason=f"Only {len(participating)} engines participated (need {self.min_participating})",
                all_opinions=all_opinions,
            )

        # Categorize engines by agreement
        agreeing = []
        disagreeing = []

        for opinion in participating:
            if opinion.direction == primary_direction:
                agreeing.append(opinion.technique)
            else:
                disagreeing.append(opinion.technique)

        # Calculate agreement score
        agreement_score = len(agreeing) / len(participating)

        # Determine consensus level
        consensus_level = self._classify_consensus(agreement_score)

        # Calculate confidence adjustment
        confidence_adjustment = self._calculate_confidence_adjustment(
            consensus_level=consensus_level,
            primary_engine=primary_engine,
            agreeing_engines=agreeing,
            disagreeing_engines=disagreeing,
            current_regime=current_regime,
        )

        adjusted_confidence = primary_confidence + confidence_adjustment

        # Generate warnings for specific disagreements
        warnings = self._generate_warnings(
            primary_engine=primary_engine,
            primary_direction=primary_direction,
            disagreeing_engines=disagreeing,
            current_regime=current_regime,
        )

        # Determine recommendation
        recommendation, reasoning = self._determine_recommendation(
            adjusted_confidence=adjusted_confidence,
            consensus_level=consensus_level,
            agreement_score=agreement_score,
            warnings=warnings,
            current_regime=current_regime,
        )

        logger.info(
            "consensus_analyzed",
            primary_engine=primary_engine.value,
            consensus_level=consensus_level.value,
            agreement_score=agreement_score,
            adjusted_confidence=adjusted_confidence,
            recommendation=recommendation,
        )

        return ConsensusResult(
            primary_direction=primary_direction,
            consensus_level=consensus_level,
            agreement_score=agreement_score,
            adjusted_confidence=adjusted_confidence,
            participating_engines=[op.technique for op in participating],
            agreeing_engines=agreeing,
            disagreeing_engines=disagreeing,
            neutral_engines=[op.technique for op in neutral],
            recommendation=recommendation,
            reasoning=reasoning,
            warnings=warnings,
        )

    def _classify_consensus(self, agreement_score: float) -> ConsensusLevel:
        """Classify consensus strength based on agreement score."""
        if agreement_score >= 1.0:
            return ConsensusLevel.UNANIMOUS
        elif agreement_score >= 0.75:
            return ConsensusLevel.STRONG
        elif agreement_score >= 0.60:
            return ConsensusLevel.MODERATE
        elif agreement_score >= 0.50:
            return ConsensusLevel.WEAK
        else:
            return ConsensusLevel.DIVIDED

    def _calculate_confidence_adjustment(
        self,
        consensus_level: ConsensusLevel,
        primary_engine: TradingTechnique,
        agreeing_engines: List[TradingTechnique],
        disagreeing_engines: List[TradingTechnique],
        current_regime: str,
    ) -> float:
        """Calculate confidence adjustment based on consensus."""

        # Base adjustment by consensus level
        adjustments = {
            ConsensusLevel.UNANIMOUS: self.unanimous_boost,
            ConsensusLevel.STRONG: self.strong_boost,
            ConsensusLevel.MODERATE: self.moderate_penalty,
            ConsensusLevel.WEAK: self.weak_penalty,
            ConsensusLevel.DIVIDED: self.divided_penalty,
        }

        base_adjustment = adjustments[consensus_level]

        # Additional adjustments for regime-specific relationships
        regime_adjustment = 0.0

        regime_lower = current_regime.lower()

        # TREND regime specifics
        if regime_lower == 'trend':
            # TREND engine should lead in trend regime
            if primary_engine == TradingTechnique.TREND:
                if TradingTechnique.BREAKOUT in agreeing_engines:
                    regime_adjustment += 0.03  # Breakout confirms trend
                if TradingTechnique.RANGE in disagreeing_engines:
                    regime_adjustment -= 0.02  # Range engine sees counter-signal

        # RANGE regime specifics
        elif regime_lower == 'range':
            # RANGE engine should lead in range regime
            if primary_engine == TradingTechnique.RANGE:
                if TradingTechnique.SWEEP in agreeing_engines:
                    regime_adjustment += 0.03  # Sweep confirms range levels
                if TradingTechnique.TREND in disagreeing_engines:
                    regime_adjustment -= 0.02  # Trend engine sees breakout forming

        # PANIC regime specifics
        elif regime_lower == 'panic':
            # In panic, need broader agreement
            if consensus_level == ConsensusLevel.UNANIMOUS:
                regime_adjustment += 0.05  # Rare unanimous in panic = very strong
            elif len(disagreeing_engines) > 0:
                regime_adjustment -= 0.05  # Any disagreement in panic is risky

        return base_adjustment + regime_adjustment

    def _generate_warnings(
        self,
        primary_engine: TradingTechnique,
        primary_direction: str,
        disagreeing_engines: List[TradingTechnique],
        current_regime: str,
    ) -> List[str]:
        """Generate warnings for specific engine disagreements."""
        warnings = []

        # TREND engine disagreement
        if TradingTechnique.TREND in disagreeing_engines:
            if primary_direction == 'buy':
                warnings.append("TREND engine sees downtrend - counter-trend risk")
            else:
                warnings.append("TREND engine sees uptrend - counter-trend risk")

        # RANGE engine disagreement when primary is trend/breakout
        if TradingTechnique.RANGE in disagreeing_engines:
            if primary_engine in [TradingTechnique.TREND, TradingTechnique.BREAKOUT]:
                warnings.append("RANGE engine sees mean reversion - breakout may fail")

        # BREAKOUT engine disagreement
        if TradingTechnique.BREAKOUT in disagreeing_engines:
            if primary_engine == TradingTechnique.TREND:
                warnings.append("BREAKOUT engine sees consolidation - trend may pause")

        # TAPE engine disagreement (order flow)
        if TradingTechnique.TAPE in disagreeing_engines:
            warnings.append("TAPE engine sees opposing order flow - microstructure conflict")

        # Multiple engines disagreeing in PANIC
        if current_regime.lower() == 'panic' and len(disagreeing_engines) >= 2:
            warnings.append("Multiple engines disagree in PANIC regime - extreme risk")

        return warnings

    def _determine_recommendation(
        self,
        adjusted_confidence: float,
        consensus_level: ConsensusLevel,
        agreement_score: float,
        warnings: List[str],
        current_regime: str,
    ) -> Tuple[str, str]:
        """Determine final recommendation based on consensus analysis."""

        # PANIC regime has stricter requirements
        if current_regime.lower() == 'panic':
            if consensus_level == ConsensusLevel.DIVIDED:
                return ('SKIP_TRADE', 'Divided consensus in PANIC regime - too risky')
            elif adjusted_confidence < 0.65:
                return ('SKIP_TRADE', f'Confidence {adjusted_confidence:.2f} too low for PANIC regime')

        # Check minimum confidence
        if adjusted_confidence < self.min_confidence:
            return ('SKIP_TRADE', f'Adjusted confidence {adjusted_confidence:.2f} below minimum {self.min_confidence:.2f}')

        # Unanimous or strong consensus
        if consensus_level in [ConsensusLevel.UNANIMOUS, ConsensusLevel.STRONG]:
            return ('TAKE_TRADE', f'{consensus_level.value.upper()} consensus - high confidence')

        # Moderate consensus
        elif consensus_level == ConsensusLevel.MODERATE:
            if adjusted_confidence >= 0.65:
                return ('TAKE_TRADE', f'Moderate consensus with confidence {adjusted_confidence:.2f}')
            else:
                return ('REDUCE_SIZE', f'Moderate consensus but confidence only {adjusted_confidence:.2f} - reduce size')

        # Weak consensus
        elif consensus_level == ConsensusLevel.WEAK:
            if adjusted_confidence >= 0.70:
                return ('REDUCE_SIZE', f'Weak consensus - reduce size despite confidence {adjusted_confidence:.2f}')
            else:
                return ('SKIP_TRADE', f'Weak consensus with confidence {adjusted_confidence:.2f} - skip')

        # Divided consensus
        else:  # DIVIDED
            return ('SKIP_TRADE', f'DIVIDED consensus (agreement: {agreement_score:.0%}) - skip trade')

    def _create_skip_result(
        self,
        primary_direction: str,
        primary_confidence: float,
        reason: str,
        all_opinions: List[EngineOpinion],
    ) -> ConsensusResult:
        """Create a skip result for insufficient participation."""
        return ConsensusResult(
            primary_direction=primary_direction,
            consensus_level=ConsensusLevel.DIVIDED,
            agreement_score=0.0,
            adjusted_confidence=0.0,
            participating_engines=[],
            agreeing_engines=[],
            disagreeing_engines=[],
            neutral_engines=[op.technique for op in all_opinions],
            recommendation='SKIP_TRADE',
            reasoning=reason,
            warnings=[reason],
        )

    def get_regime_specific_requirements(self, regime: str) -> Dict[str, any]:
        """Get regime-specific consensus requirements."""
        regime_lower = regime.lower()

        if regime_lower == 'trend':
            return {
                'preferred_leaders': [TradingTechnique.TREND, TradingTechnique.BREAKOUT],
                'warning_dissenters': [TradingTechnique.RANGE],
                'min_agreement': 0.60,
                'description': 'TREND regime favors trend-following strategies',
            }
        elif regime_lower == 'range':
            return {
                'preferred_leaders': [TradingTechnique.RANGE, TradingTechnique.SWEEP],
                'warning_dissenters': [TradingTechnique.TREND, TradingTechnique.BREAKOUT],
                'min_agreement': 0.60,
                'description': 'RANGE regime favors mean-reversion strategies',
            }
        elif regime_lower == 'panic':
            return {
                'preferred_leaders': [],  # No preferred leader in panic
                'warning_dissenters': [],  # Any dissent is risky
                'min_agreement': 0.75,  # Higher bar in panic
                'description': 'PANIC regime requires strong consensus across strategies',
            }
        else:
            return {
                'preferred_leaders': [],
                'warning_dissenters': [],
                'min_agreement': 0.60,
                'description': 'Unknown regime - use default requirements',
            }

    def format_consensus_report(self, result: ConsensusResult) -> str:
        """Generate human-readable consensus report."""
        report = []
        report.append(f"=== ENGINE CONSENSUS REPORT ===")
        report.append(f"Direction: {result.primary_direction.upper()}")
        report.append(f"Consensus: {result.consensus_level.value.upper()}")
        report.append(f"Agreement: {result.agreement_score:.0%}")
        report.append(f"Adjusted Confidence: {result.adjusted_confidence:.2f}")
        report.append(f"")
        report.append(f"Agreeing Engines ({len(result.agreeing_engines)}):")
        for engine in result.agreeing_engines:
            report.append(f"  ✓ {engine.value.upper()}")
        report.append(f"")
        report.append(f"Disagreeing Engines ({len(result.disagreeing_engines)}):")
        for engine in result.disagreeing_engines:
            report.append(f"  ✗ {engine.value.upper()}")
        report.append(f"")
        report.append(f"Neutral Engines ({len(result.neutral_engines)}):")
        for engine in result.neutral_engines:
            report.append(f"  - {engine.value.upper()}")
        report.append(f"")
        report.append(f"Recommendation: {result.recommendation}")
        report.append(f"Reasoning: {result.reasoning}")

        if result.warnings:
            report.append(f"")
            report.append(f"Warnings:")
            for warning in result.warnings:
                report.append(f"  ⚠ {warning}")

        return "\n".join(report)
