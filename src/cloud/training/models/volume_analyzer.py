"""
Volume Confirmation System

Validates trading signals based on volume patterns. Different trading techniques
require different volume characteristics to be valid.

Key Principles:
- Breakouts need HIGH volume (1.5x+ average) to confirm
- Trends need CONSISTENT volume (0.8x+ average) to sustain
- Range trades prefer LOW volume (<1.3x average) for mean reversion
- Volume confirms price action or reveals weakness

Example:
    Price breaks resistance at $2000
    WITHOUT volume check: Take trade → Fake breakout → Loss
    WITH volume check: Volume only 0.7x avg → REJECT signal → Saved!
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Tuple

import structlog

logger = structlog.get_logger(__name__)


class TradingTechnique(Enum):
    """Trading technique types (must match alpha_engines.py)."""

    TREND = "trend"
    RANGE = "range"
    BREAKOUT = "breakout"
    TAPE = "tape"
    LEADER = "leader"
    SWEEP = "sweep"


@dataclass
class VolumeRequirement:
    """Volume requirements for a trading technique."""

    technique: TradingTechnique
    min_ratio: float  # Minimum volume/average ratio
    max_ratio: Optional[float]  # Maximum volume/average ratio (None = no max)
    ideal_ratio: float  # Ideal volume/average ratio
    description: str  # Human-readable requirement


@dataclass
class VolumeValidationResult:
    """Result of volume validation."""

    is_valid: bool  # True if volume requirements met
    volume_ratio: float  # Current volume / average volume
    requirement: VolumeRequirement  # Requirements that were checked
    confidence_adjustment: float  # Multiplier for signal confidence (0.5 - 1.5)
    reasoning: str  # Human-readable explanation


class VolumeAnalyzer:
    """
    Validates trading signals based on volume characteristics.

    Each trading technique has different volume requirements:

    1. BREAKOUT: Needs explosive volume
       - Min: 1.5x average (confirmation)
       - Ideal: 2.0x+ average (strong breakout)
       - Reason: Volume confirms breakout is real, not fake

    2. TREND: Needs consistent volume
       - Min: 0.8x average (sustainable)
       - Ideal: 1.0-1.5x average (healthy trend)
       - Reason: Consistent volume = trend has participation

    3. RANGE: Prefers low volume
       - Max: 1.3x average (mean reversion works best in low vol)
       - Ideal: 0.8-1.0x average
       - Reason: Low volume = choppy market = mean reversion opportunity

    4. TAPE: Works in any volume
       - Min: 0.5x average (needs some liquidity)
       - Reason: Microstructure trading less volume-dependent

    5. LEADER: Needs volume confirmation
       - Min: 1.0x average (relative strength needs participation)
       - Ideal: 1.2x+ average

    6. SWEEP: Needs volume spike
       - Min: 1.3x average (liquidity sweep confirmation)
       - Ideal: 1.5x+ average
    """

    def __init__(self):
        """Initialize volume analyzer with technique-specific requirements."""
        # Define requirements for each technique
        self.requirements = {
            TradingTechnique.BREAKOUT: VolumeRequirement(
                technique=TradingTechnique.BREAKOUT,
                min_ratio=1.5,
                max_ratio=None,
                ideal_ratio=2.0,
                description="Breakouts need HIGH volume (1.5x+ avg) to confirm momentum",
            ),
            TradingTechnique.TREND: VolumeRequirement(
                technique=TradingTechnique.TREND,
                min_ratio=0.8,
                max_ratio=None,
                ideal_ratio=1.2,
                description="Trends need CONSISTENT volume (0.8x+ avg) to sustain",
            ),
            TradingTechnique.RANGE: VolumeRequirement(
                technique=TradingTechnique.RANGE,
                min_ratio=0.0,
                max_ratio=1.3,
                ideal_ratio=0.9,
                description="Range trades prefer LOW volume (<1.3x avg) for mean reversion",
            ),
            TradingTechnique.TAPE: VolumeRequirement(
                technique=TradingTechnique.TAPE,
                min_ratio=0.5,
                max_ratio=None,
                ideal_ratio=1.0,
                description="Tape reads work in any volume (min 0.5x avg for liquidity)",
            ),
            TradingTechnique.LEADER: VolumeRequirement(
                technique=TradingTechnique.LEADER,
                min_ratio=1.0,
                max_ratio=None,
                ideal_ratio=1.2,
                description="Leader plays need volume confirmation (1.0x+ avg)",
            ),
            TradingTechnique.SWEEP: VolumeRequirement(
                technique=TradingTechnique.SWEEP,
                min_ratio=1.3,
                max_ratio=None,
                ideal_ratio=1.5,
                description="Sweeps need volume spike (1.3x+ avg) to confirm liquidity grab",
            ),
        }

        logger.info("volume_analyzer_initialized", techniques=len(self.requirements))

    def validate_signal(
        self,
        technique: TradingTechnique,
        current_volume: float,
        average_volume: float,
        direction: str = "buy",
    ) -> VolumeValidationResult:
        """
        Validate if volume supports the trading signal.

        Args:
            technique: Trading technique being used
            current_volume: Current candle volume
            average_volume: Average volume (e.g., 20-period avg)
            direction: Signal direction ('buy', 'sell', 'hold')

        Returns:
            VolumeValidationResult with validation outcome
        """
        # Handle hold direction (always valid, no volume check needed)
        if direction == "hold":
            return VolumeValidationResult(
                is_valid=True,
                volume_ratio=0.0,
                requirement=None,
                confidence_adjustment=1.0,
                reasoning="No volume check for HOLD signals",
            )

        # Calculate volume ratio
        if average_volume <= 0:
            logger.warning(
                "invalid_average_volume",
                average_volume=average_volume,
                technique=technique.value,
            )
            # If no average volume data, assume valid but no confidence boost
            return self._create_neutral_result(technique, 1.0)

        volume_ratio = current_volume / average_volume

        # Get requirements for this technique
        requirement = self.requirements.get(technique)
        if not requirement:
            logger.warning("unknown_technique", technique=technique)
            return self._create_neutral_result(technique, volume_ratio)

        # Check if volume meets requirements
        is_valid, confidence_adjustment, reasoning = self._check_requirements(
            volume_ratio=volume_ratio,
            requirement=requirement,
        )

        result = VolumeValidationResult(
            is_valid=is_valid,
            volume_ratio=volume_ratio,
            requirement=requirement,
            confidence_adjustment=confidence_adjustment,
            reasoning=reasoning,
        )

        logger.debug(
            "volume_validated",
            technique=technique.value,
            volume_ratio=volume_ratio,
            is_valid=is_valid,
            adjustment=confidence_adjustment,
        )

        return result

    def _check_requirements(
        self,
        volume_ratio: float,
        requirement: VolumeRequirement,
    ) -> Tuple[bool, float, str]:
        """
        Check if volume ratio meets requirements.

        Returns:
            (is_valid, confidence_adjustment, reasoning)
        """
        # Check minimum requirement
        if volume_ratio < requirement.min_ratio:
            return (
                False,
                0.5,  # Penalty for insufficient volume
                f"INSUFFICIENT volume: {volume_ratio:.2f}x avg (need {requirement.min_ratio:.2f}x+)",
            )

        # Check maximum requirement (if exists)
        if requirement.max_ratio and volume_ratio > requirement.max_ratio:
            return (
                False,
                0.5,  # Penalty for excessive volume
                f"EXCESSIVE volume: {volume_ratio:.2f}x avg (max {requirement.max_ratio:.2f}x)",
            )

        # Volume meets requirements - calculate confidence adjustment
        confidence_adjustment = self._calculate_confidence_adjustment(
            volume_ratio, requirement
        )

        # Build reasoning
        if volume_ratio >= requirement.ideal_ratio:
            reasoning = f"EXCELLENT volume: {volume_ratio:.2f}x avg (ideal {requirement.ideal_ratio:.2f}x)"
        elif volume_ratio >= requirement.min_ratio * 1.2:
            reasoning = f"GOOD volume: {volume_ratio:.2f}x avg (above {requirement.min_ratio:.2f}x min)"
        else:
            reasoning = f"ADEQUATE volume: {volume_ratio:.2f}x avg (meets {requirement.min_ratio:.2f}x min)"

        return (True, confidence_adjustment, reasoning)

    def _calculate_confidence_adjustment(
        self,
        volume_ratio: float,
        requirement: VolumeRequirement,
    ) -> float:
        """
        Calculate confidence adjustment multiplier based on volume quality.

        Returns:
            Multiplier (0.8 - 1.5):
            - Ideal volume or better → 1.3-1.5x boost
            - Good volume → 1.1-1.3x boost
            - Adequate volume → 0.8-1.1x (minimal boost or slight penalty)
        """
        # Ideal or better → Strong boost
        if volume_ratio >= requirement.ideal_ratio:
            # Cap at 1.5x for very high volume
            boost = min(1.3 + (volume_ratio - requirement.ideal_ratio) * 0.2, 1.5)
            return boost

        # Between min and ideal → Moderate boost
        elif volume_ratio >= requirement.min_ratio * 1.2:
            # Linear interpolation between 1.1x and 1.3x
            progress = (volume_ratio - requirement.min_ratio * 1.2) / (
                requirement.ideal_ratio - requirement.min_ratio * 1.2
            )
            return 1.1 + progress * 0.2

        # Just above minimum → Slight penalty to neutral
        else:
            # Linear interpolation between 0.8x and 1.1x
            progress = (volume_ratio - requirement.min_ratio) / (
                requirement.min_ratio * 0.2
            )
            return 0.8 + progress * 0.3

    def _create_neutral_result(
        self,
        technique: TradingTechnique,
        volume_ratio: float,
    ) -> VolumeValidationResult:
        """Create neutral result when unable to validate."""
        return VolumeValidationResult(
            is_valid=True,  # Default to valid if can't check
            volume_ratio=volume_ratio,
            requirement=None,
            confidence_adjustment=1.0,  # No adjustment
            reasoning="Volume validation skipped (insufficient data)",
        )

    def get_requirement(self, technique: TradingTechnique) -> Optional[VolumeRequirement]:
        """Get volume requirement for a technique."""
        return self.requirements.get(technique)

    def get_all_requirements(self) -> Dict[TradingTechnique, VolumeRequirement]:
        """Get all volume requirements."""
        return self.requirements.copy()
