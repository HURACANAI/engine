"""
Adaptive Position Sizing 2.0

Dynamic position sizing based on edge quality, combining multiple confidence signals.

Key Problems Solved:
1. **Fixed Size Waste**: High-confidence unanimous consensus gets same size as low-confidence trade
2. **Opportunity Mismatch**: Missing big opportunities by undersizing great setups
3. **Risk Mismatch**: Oversizing mediocre trades that shouldn't be full size

Solution: Multi-Factor Position Sizing
- Base size × Confidence multiplier × Consensus multiplier × Regime fit multiplier
- High edge = 2x size, Low edge = 0.3x size

Formula:
    Position Size = Base Size × Confidence Factor × Consensus Factor × Regime Factor × Risk Factor

Example:
    Base Size: $1,000

    High-Edge Trade:
    - Confidence: 0.85 (Factor: 1.4x)
    - Consensus: UNANIMOUS (Factor: 1.3x)
    - Regime Fit: TREND in TREND (Factor: 1.2x)
    - Risk: Low volatility (Factor: 1.0x)
    → Position Size = $1,000 × 1.4 × 1.3 × 1.2 × 1.0 = $2,184 (2.18x base)

    Low-Edge Trade:
    - Confidence: 0.62 (Factor: 0.7x)
    - Consensus: WEAK (Factor: 0.8x)
    - Regime Fit: RANGE in TREND (Factor: 0.6x)
    - Risk: High volatility (Factor: 0.8x)
    → Position Size = $1,000 × 0.7 × 0.8 × 0.6 × 0.8 = $269 (0.27x base)
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class PositionSizingFactors:
    """Individual sizing factors."""

    confidence_factor: float  # Based on entry confidence (0.3-2.0)
    consensus_factor: float  # Based on engine consensus (0.5-1.5)
    regime_factor: float  # Based on regime fit (0.5-1.5)
    risk_factor: float  # Based on volatility/risk (0.5-1.2)
    pattern_factor: float  # Based on pattern quality (0.5-1.3)

    # Final multiplier
    total_multiplier: float

    # Reasoning
    explanation: str


@dataclass
class PositionSizeResult:
    """Result of position sizing calculation."""

    base_size_usd: float
    recommended_size_usd: float
    multiplier: float
    factors: PositionSizingFactors
    risk_assessment: str  # 'LOW', 'MEDIUM', 'HIGH'
    sizing_recommendation: str  # 'FULL_SIZE', 'REDUCED', 'MINIMAL', 'OVERSIZED'


class AdaptivePositionSizer:
    """
    Adaptive position sizing based on edge quality.

    Position size scales with:
    1. Entry Confidence (from Phase 3 calibrator)
    2. Engine Consensus (from Phase 3 consensus system)
    3. Regime Fit (technique affinity for current regime)
    4. Risk/Volatility (higher vol = smaller size)
    5. Pattern Quality (from Phase 4 pattern analyzer)

    Sizing Philosophy:
    - Great setups deserve bigger size (capture opportunity)
    - Mediocre setups deserve smaller size (limit risk)
    - Never go below min multiplier (always participate)
    - Cap at max multiplier (risk management)

    Usage:
        sizer = AdaptivePositionSizer(
            base_size_usd=1000.0,
            min_multiplier=0.25,
            max_multiplier=2.50,
        )

        # Calculate size
        result = sizer.calculate_size(
            confidence=0.85,
            consensus_level='STRONG',
            consensus_agreement=0.78,
            technique='trend',
            regime='trend',
            volatility_bps=120.0,
            pattern_quality=0.75,
        )

        if result.sizing_recommendation == 'OVERSIZED':
            logger.warning("Edge too strong, maxed out at 2.5x")

        position_size = result.recommended_size_usd
    """

    def __init__(
        self,
        base_size_usd: float = 1000.0,
        min_multiplier: float = 0.25,
        max_multiplier: float = 2.50,
        confidence_weight: float = 0.30,
        consensus_weight: float = 0.25,
        regime_weight: float = 0.20,
        risk_weight: float = 0.15,
        pattern_weight: float = 0.10,
    ):
        """
        Initialize adaptive position sizer.

        Args:
            base_size_usd: Base position size in USD
            min_multiplier: Minimum size multiplier (e.g., 0.25 = 25% of base)
            max_multiplier: Maximum size multiplier (e.g., 2.5 = 250% of base)
            confidence_weight: Weight for confidence factor
            consensus_weight: Weight for consensus factor
            regime_weight: Weight for regime fit factor
            risk_weight: Weight for risk/volatility factor
            pattern_weight: Weight for pattern quality factor
        """
        self.base_size = base_size_usd
        self.min_multiplier = min_multiplier
        self.max_multiplier = max_multiplier

        # Factor weights (should sum to 1.0)
        self.weights = {
            'confidence': confidence_weight,
            'consensus': consensus_weight,
            'regime': regime_weight,
            'risk': risk_weight,
            'pattern': pattern_weight,
        }

        # Validate weights
        total_weight = sum(self.weights.values())
        if abs(total_weight - 1.0) > 0.01:
            logger.warning(f"Factor weights sum to {total_weight}, not 1.0. Normalizing...")
            for key in self.weights:
                self.weights[key] /= total_weight

        logger.info(
            "adaptive_position_sizer_initialized",
            base_size=base_size_usd,
            min_multiplier=min_multiplier,
            max_multiplier=max_multiplier,
            weights=self.weights,
        )

    def calculate_size(
        self,
        confidence: float,
        consensus_level: str,
        consensus_agreement: float,
        technique: str,
        regime: str,
        volatility_bps: float,
        pattern_quality: Optional[float] = None,
    ) -> PositionSizeResult:
        """
        Calculate adaptive position size.

        Args:
            confidence: Entry confidence (0-1)
            consensus_level: 'UNANIMOUS', 'STRONG', 'MODERATE', 'WEAK', 'DIVIDED'
            consensus_agreement: Agreement score (0-1)
            technique: Trading technique ('trend', 'range', etc.)
            regime: Market regime ('trend', 'range', 'panic')
            volatility_bps: Current volatility in bps
            pattern_quality: Optional pattern quality score (0-1)

        Returns:
            PositionSizeResult with recommended size
        """
        # Calculate individual factors
        confidence_factor = self._calculate_confidence_factor(confidence)
        consensus_factor = self._calculate_consensus_factor(consensus_level, consensus_agreement)
        regime_factor = self._calculate_regime_factor(technique, regime)
        risk_factor = self._calculate_risk_factor(volatility_bps)
        pattern_factor = self._calculate_pattern_factor(pattern_quality)

        # Calculate weighted multiplier
        total_multiplier = 1.0
        total_multiplier += (confidence_factor - 1.0) * self.weights['confidence']
        total_multiplier += (consensus_factor - 1.0) * self.weights['consensus']
        total_multiplier += (regime_factor - 1.0) * self.weights['regime']
        total_multiplier += (risk_factor - 1.0) * self.weights['risk']
        total_multiplier += (pattern_factor - 1.0) * self.weights['pattern']

        # Clamp to min/max
        unclamped_multiplier = total_multiplier
        total_multiplier = np.clip(total_multiplier, self.min_multiplier, self.max_multiplier)

        # Calculate size
        recommended_size = self.base_size * total_multiplier

        # Create factors object
        factors = PositionSizingFactors(
            confidence_factor=confidence_factor,
            consensus_factor=consensus_factor,
            regime_factor=regime_factor,
            risk_factor=risk_factor,
            pattern_factor=pattern_factor,
            total_multiplier=total_multiplier,
            explanation=self._generate_explanation(
                confidence_factor,
                consensus_factor,
                regime_factor,
                risk_factor,
                pattern_factor,
                total_multiplier,
                unclamped_multiplier,
            ),
        )

        # Risk assessment
        risk_assessment = self._assess_risk(volatility_bps, confidence, consensus_agreement)

        # Sizing recommendation
        if total_multiplier >= self.max_multiplier * 0.95:
            sizing_rec = 'OVERSIZED'
        elif total_multiplier >= 1.5:
            sizing_rec = 'FULL_SIZE'
        elif total_multiplier >= 0.7:
            sizing_rec = 'REDUCED'
        else:
            sizing_rec = 'MINIMAL'

        logger.info(
            "position_size_calculated",
            base_size=self.base_size,
            multiplier=total_multiplier,
            recommended_size=recommended_size,
            confidence=confidence,
            consensus=consensus_level,
            regime=regime,
        )

        return PositionSizeResult(
            base_size_usd=self.base_size,
            recommended_size_usd=recommended_size,
            multiplier=total_multiplier,
            factors=factors,
            risk_assessment=risk_assessment,
            sizing_recommendation=sizing_rec,
        )

    def _calculate_confidence_factor(self, confidence: float) -> float:
        """
        Calculate sizing factor based on confidence.

        Confidence → Factor:
        - 0.90+: 1.8x (very high confidence)
        - 0.80-0.90: 1.4x (high confidence)
        - 0.70-0.80: 1.2x (good confidence)
        - 0.60-0.70: 1.0x (baseline)
        - 0.55-0.60: 0.8x (low confidence)
        - <0.55: 0.5x (very low confidence)
        """
        if confidence >= 0.90:
            return 1.8
        elif confidence >= 0.80:
            return 1.4
        elif confidence >= 0.70:
            return 1.2
        elif confidence >= 0.60:
            return 1.0
        elif confidence >= 0.55:
            return 0.8
        else:
            return 0.5

    def _calculate_consensus_factor(self, level: str, agreement: float) -> float:
        """
        Calculate sizing factor based on consensus.

        Consensus Level → Factor:
        - UNANIMOUS: 1.5x (all engines agree)
        - STRONG: 1.3x (75%+ agree)
        - MODERATE: 1.0x (60-75% agree)
        - WEAK: 0.8x (50-60% agree)
        - DIVIDED: 0.5x (<50% agree)
        """
        level_upper = level.upper()

        if level_upper == 'UNANIMOUS':
            return 1.5
        elif level_upper == 'STRONG':
            return 1.3
        elif level_upper == 'MODERATE':
            return 1.0
        elif level_upper == 'WEAK':
            return 0.8
        else:  # DIVIDED
            return 0.5

    def _calculate_regime_factor(self, technique: str, regime: str) -> float:
        """
        Calculate sizing factor based on technique-regime fit.

        Technique affinity for regime:
        - Perfect fit: 1.4x (e.g., TREND in TREND)
        - Good fit: 1.2x (e.g., BREAKOUT in TREND)
        - Neutral: 1.0x
        - Poor fit: 0.7x (e.g., RANGE in TREND)
        - Very poor fit: 0.5x (e.g., TREND in RANGE)
        """
        technique_lower = technique.lower()
        regime_lower = regime.lower()

        # Regime affinity matrix
        affinity = {
            # TREND regime
            ('trend', 'trend'): 1.4,  # Perfect
            ('breakout', 'trend'): 1.3,  # Excellent
            ('leader', 'trend'): 1.2,  # Good
            ('tape', 'trend'): 1.0,  # Neutral
            ('sweep', 'trend'): 0.8,  # Poor
            ('range', 'trend'): 0.6,  # Very poor

            # RANGE regime
            ('range', 'range'): 1.4,  # Perfect
            ('sweep', 'range'): 1.3,  # Excellent
            ('tape', 'range'): 1.1,  # Good
            ('leader', 'range'): 0.9,  # Neutral
            ('breakout', 'range'): 0.7,  # Poor
            ('trend', 'range'): 0.6,  # Very poor

            # PANIC regime
            ('sweep', 'panic'): 1.3,  # Best in panic
            ('tape', 'panic'): 1.2,  # Good
            ('range', 'panic'): 0.8,  # Risky
            ('trend', 'panic'): 0.7,  # Poor
            ('breakout', 'panic'): 0.6,  # Very poor
            ('leader', 'panic'): 0.6,  # Very poor
        }

        return affinity.get((technique_lower, regime_lower), 1.0)

    def _calculate_risk_factor(self, volatility_bps: float) -> float:
        """
        Calculate sizing factor based on volatility/risk.

        Volatility → Factor:
        - <80 bps: 1.2x (very low vol, can size up)
        - 80-120 bps: 1.0x (normal vol)
        - 120-200 bps: 0.9x (elevated vol)
        - 200-300 bps: 0.7x (high vol, size down)
        - >300 bps: 0.5x (extreme vol, minimal size)
        """
        if volatility_bps < 80:
            return 1.2
        elif volatility_bps < 120:
            return 1.0
        elif volatility_bps < 200:
            return 0.9
        elif volatility_bps < 300:
            return 0.7
        else:
            return 0.5

    def _calculate_pattern_factor(self, pattern_quality: Optional[float]) -> float:
        """
        Calculate sizing factor based on pattern quality.

        Pattern Quality → Factor:
        - None: 1.0x (no pattern info)
        - >0.75: 1.3x (excellent pattern)
        - 0.65-0.75: 1.15x (good pattern)
        - 0.50-0.65: 1.0x (neutral)
        - <0.50: 0.7x (poor pattern)
        """
        if pattern_quality is None:
            return 1.0

        if pattern_quality >= 0.75:
            return 1.3
        elif pattern_quality >= 0.65:
            return 1.15
        elif pattern_quality >= 0.50:
            return 1.0
        else:
            return 0.7

    def _assess_risk(self, volatility_bps: float, confidence: float, consensus: float) -> str:
        """Assess overall risk level."""
        # Calculate risk score (0-100, lower = less risky)
        vol_risk = min(volatility_bps / 3.0, 100)  # Normalize volatility
        confidence_risk = (1 - confidence) * 100
        consensus_risk = (1 - consensus) * 100

        avg_risk = (vol_risk + confidence_risk + consensus_risk) / 3

        if avg_risk < 30:
            return 'LOW'
        elif avg_risk < 60:
            return 'MEDIUM'
        else:
            return 'HIGH'

    def _generate_explanation(
        self,
        conf_factor: float,
        cons_factor: float,
        regime_factor: float,
        risk_factor: float,
        pattern_factor: float,
        final_mult: float,
        unclamped_mult: float,
    ) -> str:
        """Generate human-readable explanation."""
        parts = []

        parts.append(f"Confidence: {conf_factor:.2f}x")
        parts.append(f"Consensus: {cons_factor:.2f}x")
        parts.append(f"Regime Fit: {regime_factor:.2f}x")
        parts.append(f"Risk: {risk_factor:.2f}x")
        parts.append(f"Pattern: {pattern_factor:.2f}x")

        explanation = " | ".join(parts)
        explanation += f" → {final_mult:.2f}x total"

        if unclamped_mult > self.max_multiplier:
            explanation += f" (capped from {unclamped_mult:.2f}x)"
        elif unclamped_mult < self.min_multiplier:
            explanation += f" (floored from {unclamped_mult:.2f}x)"

        return explanation

    def update_base_size(self, new_base_size: float) -> None:
        """Update base position size."""
        old_size = self.base_size
        self.base_size = new_base_size

        logger.info(
            "base_size_updated",
            old_size=old_size,
            new_size=new_base_size,
        )

    def get_size_for_capital_pct(
        self,
        total_capital: float,
        target_pct: float,
        multiplier: float,
    ) -> float:
        """
        Calculate position size as percentage of capital.

        Args:
            total_capital: Total available capital
            target_pct: Target percentage (e.g., 0.10 for 10%)
            multiplier: Adaptive multiplier from calculate_size()

        Returns:
            Position size in USD
        """
        base_allocation = total_capital * target_pct
        return base_allocation * multiplier

    def get_statistics(self) -> Dict[str, any]:
        """Get sizing statistics."""
        return {
            'base_size_usd': self.base_size,
            'min_multiplier': self.min_multiplier,
            'max_multiplier': self.max_multiplier,
            'max_possible_size': self.base_size * self.max_multiplier,
            'min_possible_size': self.base_size * self.min_multiplier,
            'weights': self.weights,
        }
