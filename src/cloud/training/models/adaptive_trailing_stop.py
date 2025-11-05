"""
Adaptive Trailing Stop System

Intelligently trails stop losses to lock in profits while letting winners run.
Unlike static stops, this system adapts to:
- Volatility levels (wider trails in high volatility)
- Profit stages (tightens as profit increases)
- Momentum strength (tightens when momentum weakens)

Key Philosophy:
- Let winners run, but protect profits
- Progressive profit locking at key milestones
- Volatility-adjusted trail distance
- Momentum-aware tightening

Example:
    Entry at $2000
    → +50 bps profit ($2010) → Trail locks at $2005 (+25 bps)
    → +100 bps profit ($2020) → Trail locks at $2010 (+50 bps)
    → +200 bps profit ($2040) → Trail locks at $2030 (+150 bps)
    → Price reverses to $2030 → SOLD (+150 bps instead of giving it all back!)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class TrailStage:
    """Configuration for a profit-locking stage."""

    profit_threshold_bps: float  # When this stage activates
    trail_distance_bps: float  # How far below current price to trail
    description: str  # Human-readable description


@dataclass
class TrailStopResult:
    """Result of trailing stop calculation."""

    stop_price: float  # Updated stop loss price
    trail_distance_bps: float  # Current trail distance
    locked_profit_bps: float  # Minimum profit now locked
    stage_active: str  # Which stage is currently active
    should_tighten: bool  # True if momentum weakening suggests tightening
    reasoning: str  # Human-readable explanation


class AdaptiveTrailingStop:
    """
    Adaptive trailing stop manager that locks in profits progressively.

    Trail Stages (default):
    1. Below +50 bps: Fixed stop (no trail yet)
    2. +50 to +100 bps: Trail 25 bps (lock minimum +25 bps profit)
    3. +100 to +200 bps: Trail 50 bps (lock minimum +50 bps profit)
    4. Above +200 bps: Trail 100 bps (lock minimum +100 bps profit)

    Adjustments:
    - High volatility → Wider trails (×1.5)
    - Low volatility → Tighter trails (×0.7)
    - Weakening momentum → Tighten trail by 20%
    - Strong momentum → Keep trail as-is

    Usage:
        trail_manager = AdaptiveTrailingStop()
        result = trail_manager.calculate_trail_stop(
            entry_price=2000.0,
            current_price=2040.0,
            current_pnl_bps=200.0,
            volatility_bps=150.0,
            momentum_score=0.6,
        )
        # result.stop_price = 2030.0 (lock +150 bps profit)
    """

    def __init__(
        self,
        stages: Optional[List[Dict]] = None,
        volatility_multiplier_high: float = 1.5,
        volatility_multiplier_low: float = 0.7,
        volatility_high_threshold: float = 200.0,
        volatility_low_threshold: float = 80.0,
        momentum_tighten_threshold: float = 0.3,
        momentum_tighten_factor: float = 0.8,
    ):
        """
        Initialize adaptive trailing stop manager.

        Args:
            stages: List of profit stages with trail distances
            volatility_multiplier_high: Multiplier for high volatility (>200 bps)
            volatility_multiplier_low: Multiplier for low volatility (<80 bps)
            volatility_high_threshold: Volatility considered "high" (bps)
            volatility_low_threshold: Volatility considered "low" (bps)
            momentum_tighten_threshold: Momentum below this → tighten trail
            momentum_tighten_factor: How much to tighten (0.8 = 20% tighter)
        """
        # Default progressive profit-locking stages
        if stages is None:
            stages = [
                {'profit_threshold': 50, 'trail_distance': 25, 'description': 'Stage 1: Lock +25 bps'},
                {'profit_threshold': 100, 'trail_distance': 50, 'description': 'Stage 2: Lock +50 bps'},
                {'profit_threshold': 200, 'trail_distance': 100, 'description': 'Stage 3: Lock +100 bps'},
                {'profit_threshold': 400, 'trail_distance': 200, 'description': 'Stage 4: Lock +200 bps'},
            ]

        self.stages = [
            TrailStage(
                profit_threshold_bps=s['profit_threshold'],
                trail_distance_bps=s['trail_distance'],
                description=s['description'],
            )
            for s in stages
        ]

        self.vol_mult_high = volatility_multiplier_high
        self.vol_mult_low = volatility_multiplier_low
        self.vol_high_threshold = volatility_high_threshold
        self.vol_low_threshold = volatility_low_threshold
        self.momentum_tighten_threshold = momentum_tighten_threshold
        self.momentum_tighten_factor = momentum_tighten_factor

        logger.info(
            "adaptive_trailing_stop_initialized",
            num_stages=len(self.stages),
            highest_profit_threshold=self.stages[-1].profit_threshold_bps,
        )

    def calculate_trail_stop(
        self,
        entry_price: float,
        current_price: float,
        current_pnl_bps: float,
        volatility_bps: float,
        momentum_score: float = 0.5,
        direction: str = "buy",
    ) -> TrailStopResult:
        """
        Calculate adaptive trailing stop based on current conditions.

        Args:
            entry_price: Original entry price
            current_price: Current market price
            current_pnl_bps: Current P&L in basis points
            volatility_bps: Current volatility (ATR or similar)
            momentum_score: Momentum strength (0-1, higher = stronger)
            direction: Trade direction ('buy' or 'sell')

        Returns:
            TrailStopResult with updated stop price and reasoning
        """
        # Determine active trail stage
        active_stage = self._get_active_stage(current_pnl_bps)

        if active_stage is None:
            # Below first stage threshold - use fixed stop
            return self._create_fixed_stop_result(
                entry_price=entry_price,
                current_pnl_bps=current_pnl_bps,
                direction=direction,
            )

        # Calculate base trail distance
        base_trail_distance = active_stage.trail_distance_bps

        # Adjust for volatility
        trail_distance = self._adjust_for_volatility(
            base_trail_distance=base_trail_distance,
            volatility_bps=volatility_bps,
        )

        # Adjust for momentum
        should_tighten = False
        if momentum_score < self.momentum_tighten_threshold:
            trail_distance *= self.momentum_tighten_factor
            should_tighten = True
            logger.debug(
                "trail_tightened_momentum_weak",
                momentum=momentum_score,
                tighten_factor=self.momentum_tighten_factor,
            )

        # Calculate stop price
        if direction == "buy":
            # Long position: stop trails below current price
            stop_price = current_price - (trail_distance / 10_000) * current_price
            locked_profit_bps = ((stop_price - entry_price) / entry_price) * 10_000
        else:
            # Short position: stop trails above current price
            stop_price = current_price + (trail_distance / 10_000) * current_price
            locked_profit_bps = ((entry_price - stop_price) / entry_price) * 10_000

        # Build reasoning
        reasoning = self._build_reasoning(
            stage_desc=active_stage.description,
            trail_distance=trail_distance,
            volatility_bps=volatility_bps,
            momentum_score=momentum_score,
            should_tighten=should_tighten,
        )

        result = TrailStopResult(
            stop_price=stop_price,
            trail_distance_bps=trail_distance,
            locked_profit_bps=max(locked_profit_bps, 0.0),
            stage_active=active_stage.description,
            should_tighten=should_tighten,
            reasoning=reasoning,
        )

        logger.debug(
            "trail_stop_calculated",
            entry_price=entry_price,
            current_price=current_price,
            stop_price=stop_price,
            locked_profit_bps=locked_profit_bps,
            stage=active_stage.description,
        )

        return result

    def _get_active_stage(self, current_pnl_bps: float) -> Optional[TrailStage]:
        """
        Get the active trail stage based on current profit.

        Returns the highest stage whose threshold is met.
        """
        active_stage = None

        for stage in self.stages:
            if current_pnl_bps >= stage.profit_threshold_bps:
                active_stage = stage
            else:
                break  # Stages are sorted by threshold

        return active_stage

    def _adjust_for_volatility(
        self,
        base_trail_distance: float,
        volatility_bps: float,
    ) -> float:
        """
        Adjust trail distance based on current volatility.

        High volatility → Wider trail (avoid getting stopped out on noise)
        Low volatility → Tighter trail (maximize profit capture)
        """
        if volatility_bps >= self.vol_high_threshold:
            # High volatility - widen trail
            multiplier = self.vol_mult_high
            logger.debug("trail_widened_high_volatility", vol=volatility_bps, mult=multiplier)
        elif volatility_bps <= self.vol_low_threshold:
            # Low volatility - tighten trail
            multiplier = self.vol_mult_low
            logger.debug("trail_tightened_low_volatility", vol=volatility_bps, mult=multiplier)
        else:
            # Normal volatility - no adjustment
            multiplier = 1.0

        return base_trail_distance * multiplier

    def _create_fixed_stop_result(
        self,
        entry_price: float,
        current_pnl_bps: float,
        direction: str,
        fixed_stop_distance_bps: float = 100.0,
    ) -> TrailStopResult:
        """Create result for fixed stop (before first trail stage)."""
        if direction == "buy":
            stop_price = entry_price * (1 - fixed_stop_distance_bps / 10_000)
        else:
            stop_price = entry_price * (1 + fixed_stop_distance_bps / 10_000)

        return TrailStopResult(
            stop_price=stop_price,
            trail_distance_bps=fixed_stop_distance_bps,
            locked_profit_bps=0.0,
            stage_active="Fixed Stop (No Trail)",
            should_tighten=False,
            reasoning=f"Below profit threshold - using fixed stop at -{fixed_stop_distance_bps:.0f} bps",
        )

    def _build_reasoning(
        self,
        stage_desc: str,
        trail_distance: float,
        volatility_bps: float,
        momentum_score: float,
        should_tighten: bool,
    ) -> str:
        """Build human-readable reasoning."""
        reasons = [stage_desc, f"trail {trail_distance:.0f} bps"]

        # Volatility adjustment note
        if volatility_bps >= self.vol_high_threshold:
            reasons.append(f"widened for high vol ({volatility_bps:.0f} bps)")
        elif volatility_bps <= self.vol_low_threshold:
            reasons.append(f"tightened for low vol ({volatility_bps:.0f} bps)")

        # Momentum note
        if should_tighten:
            reasons.append(f"tightened for weak momentum ({momentum_score:.2f})")

        return "; ".join(reasons)

    def should_update_stop(
        self,
        current_stop_price: float,
        new_stop_price: float,
        direction: str,
    ) -> bool:
        """
        Determine if stop should be updated.

        Trailing stops only move in favorable direction (up for longs, down for shorts).

        Args:
            current_stop_price: Current stop loss price
            new_stop_price: Newly calculated stop loss price
            direction: Trade direction ('buy' or 'sell')

        Returns:
            True if stop should be updated
        """
        if direction == "buy":
            # Long: only raise stop (move up)
            return new_stop_price > current_stop_price
        else:
            # Short: only lower stop (move down)
            return new_stop_price < current_stop_price

    def get_stages(self) -> List[TrailStage]:
        """Get all configured trail stages."""
        return self.stages.copy()

    def get_stage_info(self, current_pnl_bps: float) -> Dict:
        """Get information about current and next stages."""
        active_stage = self._get_active_stage(current_pnl_bps)

        if active_stage is None:
            next_stage = self.stages[0] if self.stages else None
            return {
                'current_stage': None,
                'next_stage': next_stage.description if next_stage else None,
                'next_threshold': next_stage.profit_threshold_bps if next_stage else None,
                'bps_to_next': (next_stage.profit_threshold_bps - current_pnl_bps) if next_stage else None,
            }

        # Find next stage
        current_idx = self.stages.index(active_stage)
        next_stage = self.stages[current_idx + 1] if current_idx < len(self.stages) - 1 else None

        return {
            'current_stage': active_stage.description,
            'current_trail_distance': active_stage.trail_distance_bps,
            'next_stage': next_stage.description if next_stage else "Final Stage",
            'next_threshold': next_stage.profit_threshold_bps if next_stage else None,
            'bps_to_next': (next_stage.profit_threshold_bps - current_pnl_bps) if next_stage else None,
        }
