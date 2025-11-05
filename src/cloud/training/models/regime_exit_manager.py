"""
Regime Exit Manager

Monitors regime transitions during open trades and triggers exits or adjustments
when market conditions fundamentally change. This prevents holding positions
through regime shifts that invalidate the original trade thesis.

Key Philosophy:
- Different regimes = different rules
- Regime shift invalidates trade thesis → exit
- Protect profits during regime deterioration
- Adapt position management to new regime

Regime Transition Handling:
1. TREND → PANIC: Exit immediately if profitable
2. TREND → RANGE: Scale out 50% if profitable
3. RANGE → TREND: Hold if trend aligns, exit if conflicts
4. Any → PANIC: Tighten stops dramatically, exit on profit
5. PANIC → TREND/RANGE: Relax if position aligned

Example:
    Entered long ETH in TREND regime at $2000
    Current: $2030 (+150 bps profit)
    Regime shifts: TREND → PANIC
    → Manager detects regime shift + profit
    → Triggers P1 DANGER exit
    → Exit at $2030, save +150 bps
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Tuple

import structlog

logger = structlog.get_logger(__name__)


class RegimeTransition(Enum):
    """Types of regime transitions."""

    # Deteriorating transitions (high danger)
    TREND_TO_PANIC = "trend_to_panic"
    RANGE_TO_PANIC = "range_to_panic"

    # Stabilizing transitions (moderate adjustment)
    TREND_TO_RANGE = "trend_to_range"
    PANIC_TO_RANGE = "panic_to_range"

    # Momentum transitions (opportunity or risk)
    RANGE_TO_TREND = "range_to_trend"
    PANIC_TO_TREND = "panic_to_trend"

    # No change
    NO_CHANGE = "no_change"


class RegimeAction(Enum):
    """Actions to take on regime transition."""

    EXIT_IMMEDIATELY = "exit_immediately"  # Danger - get out now
    SCALE_OUT_HALF = "scale_out_half"  # Reduce exposure by 50%
    TIGHTEN_STOPS = "tighten_stops"  # Move stops closer
    RELAX_STOPS = "relax_stops"  # Give position more room
    HOLD = "hold"  # No change needed
    REVERSE_POSITION = "reverse_position"  # Rare - position now wrong direction


@dataclass
class RegimeExitSignal:
    """Signal from regime transition analysis."""

    transition: RegimeTransition
    action: RegimeAction
    priority: int  # 1=highest, 3=lowest
    reason: str
    description: str
    confidence: float  # 0-1
    stop_adjustment_bps: Optional[float] = None  # For TIGHTEN/RELAX actions


class RegimeExitManager:
    """
    Manages position exits and adjustments based on regime transitions.

    This system acts as a "regime safety net" that overrides normal
    position management when fundamental market conditions change.

    Regime Transition Matrix:

    FROM TREND:
    - To PANIC with profit → EXIT_IMMEDIATELY (save profits)
    - To PANIC without profit → TIGHTEN_STOPS (limit damage)
    - To RANGE with profit → SCALE_OUT_HALF (lock some profit)
    - To RANGE without profit → TIGHTEN_STOPS (reduce risk)

    FROM RANGE:
    - To PANIC → EXIT_IMMEDIATELY (panic kills range trades)
    - To TREND aligned → RELAX_STOPS (let trend run)
    - To TREND opposed → EXIT_IMMEDIATELY (wrong direction)

    FROM PANIC:
    - To TREND/RANGE with profit → SCALE_OUT_HALF (stabilizing)
    - To TREND/RANGE without profit → RELAX_STOPS (less danger)

    Usage:
        manager = RegimeExitManager()
        signal = manager.check_regime_transition(
            entry_regime='trend',
            current_regime='panic',
            position_direction='buy',
            position_pnl_bps=150.0,
        )

        if signal.action == RegimeAction.EXIT_IMMEDIATELY:
            exit_position()
    """

    def __init__(
        self,
        profit_threshold_bps: float = 50.0,
        panic_stop_tighten_bps: float = 50.0,
        range_stop_tighten_bps: float = 30.0,
        trend_stop_relax_bps: float = 20.0,
    ):
        """
        Initialize regime exit manager.

        Args:
            profit_threshold_bps: Minimum profit to trigger protective exits
            panic_stop_tighten_bps: How much to tighten stops in panic
            range_stop_tighten_bps: How much to tighten stops in range
            trend_stop_relax_bps: How much to relax stops in favorable trend
        """
        self.profit_threshold = profit_threshold_bps
        self.panic_stop_tighten = panic_stop_tighten_bps
        self.range_stop_tighten = range_stop_tighten_bps
        self.trend_stop_relax = trend_stop_relax_bps

        logger.info(
            "regime_exit_manager_initialized",
            profit_threshold=profit_threshold_bps,
            panic_tighten=panic_stop_tighten_bps,
        )

    def check_regime_transition(
        self,
        entry_regime: str,
        current_regime: str,
        position_direction: str,
        position_pnl_bps: float,
        trend_direction: Optional[str] = None,  # 'up' or 'down' for RANGE→TREND
    ) -> Optional[RegimeExitSignal]:
        """
        Check if regime transition requires action.

        Args:
            entry_regime: Regime when position was entered
            current_regime: Current market regime
            position_direction: 'buy' or 'sell'
            position_pnl_bps: Current P&L in basis points
            trend_direction: Direction of trend (for RANGE→TREND transitions)

        Returns:
            RegimeExitSignal if action needed, None otherwise
        """
        # Detect transition type
        transition = self._classify_transition(entry_regime, current_regime)

        if transition == RegimeTransition.NO_CHANGE:
            return None  # No regime change

        # Determine action based on transition type and P&L
        signal = self._determine_action(
            transition=transition,
            position_direction=position_direction,
            position_pnl_bps=position_pnl_bps,
            trend_direction=trend_direction,
        )

        if signal:
            logger.info(
                "regime_transition_detected",
                from_regime=entry_regime,
                to_regime=current_regime,
                action=signal.action.value,
                pnl_bps=position_pnl_bps,
            )

        return signal

    def _classify_transition(
        self,
        from_regime: str,
        to_regime: str,
    ) -> RegimeTransition:
        """Classify the type of regime transition."""
        if from_regime == to_regime:
            return RegimeTransition.NO_CHANGE

        # Normalize regime names
        from_r = from_regime.lower()
        to_r = to_regime.lower()

        # Deteriorating transitions
        if from_r == 'trend' and to_r == 'panic':
            return RegimeTransition.TREND_TO_PANIC
        elif from_r == 'range' and to_r == 'panic':
            return RegimeTransition.RANGE_TO_PANIC

        # Stabilizing transitions
        elif from_r == 'trend' and to_r == 'range':
            return RegimeTransition.TREND_TO_RANGE
        elif from_r == 'panic' and to_r == 'range':
            return RegimeTransition.PANIC_TO_RANGE

        # Momentum transitions
        elif from_r == 'range' and to_r == 'trend':
            return RegimeTransition.RANGE_TO_TREND
        elif from_r == 'panic' and to_r == 'trend':
            return RegimeTransition.PANIC_TO_TREND

        else:
            logger.warning(
                "unknown_regime_transition",
                from_regime=from_regime,
                to_regime=to_regime,
            )
            return RegimeTransition.NO_CHANGE

    def _determine_action(
        self,
        transition: RegimeTransition,
        position_direction: str,
        position_pnl_bps: float,
        trend_direction: Optional[str],
    ) -> Optional[RegimeExitSignal]:
        """Determine what action to take for this transition."""

        has_profit = position_pnl_bps >= self.profit_threshold

        # ===== DETERIORATING TRANSITIONS =====

        if transition == RegimeTransition.TREND_TO_PANIC:
            if has_profit:
                # DANGER: Exit immediately to save profits
                return RegimeExitSignal(
                    transition=transition,
                    action=RegimeAction.EXIT_IMMEDIATELY,
                    priority=1,
                    reason="REGIME_PANIC_WITH_PROFIT",
                    description=f"TREND→PANIC with +{position_pnl_bps:.0f} bps profit - exit immediately",
                    confidence=0.95,
                )
            else:
                # WARNING: Tighten stops to limit damage
                return RegimeExitSignal(
                    transition=transition,
                    action=RegimeAction.TIGHTEN_STOPS,
                    priority=2,
                    reason="REGIME_PANIC_NO_PROFIT",
                    description=f"TREND→PANIC with {position_pnl_bps:.0f} bps - tighten stops",
                    confidence=0.85,
                    stop_adjustment_bps=self.panic_stop_tighten,
                )

        elif transition == RegimeTransition.RANGE_TO_PANIC:
            # Always exit range trades in panic - range logic breaks down
            return RegimeExitSignal(
                transition=transition,
                action=RegimeAction.EXIT_IMMEDIATELY,
                priority=1,
                reason="RANGE_TO_PANIC",
                description=f"RANGE→PANIC - range trading invalid in panic regime",
                confidence=0.90,
            )

        # ===== STABILIZING TRANSITIONS =====

        elif transition == RegimeTransition.TREND_TO_RANGE:
            if has_profit:
                # Take some profit as trend momentum fades
                return RegimeExitSignal(
                    transition=transition,
                    action=RegimeAction.SCALE_OUT_HALF,
                    priority=2,
                    reason="TREND_TO_RANGE_WITH_PROFIT",
                    description=f"TREND→RANGE with +{position_pnl_bps:.0f} bps - scale out 50%",
                    confidence=0.75,
                )
            else:
                # Tighten stops as range doesn't favor trend trades
                return RegimeExitSignal(
                    transition=transition,
                    action=RegimeAction.TIGHTEN_STOPS,
                    priority=3,
                    reason="TREND_TO_RANGE_NO_PROFIT",
                    description=f"TREND→RANGE with {position_pnl_bps:.0f} bps - tighten stops",
                    confidence=0.70,
                    stop_adjustment_bps=self.range_stop_tighten,
                )

        elif transition == RegimeTransition.PANIC_TO_RANGE:
            # Market stabilizing - relax stops slightly
            return RegimeExitSignal(
                transition=transition,
                action=RegimeAction.RELAX_STOPS,
                priority=3,
                reason="PANIC_TO_RANGE",
                description="PANIC→RANGE - market stabilizing, relax stops slightly",
                confidence=0.70,
                stop_adjustment_bps=self.trend_stop_relax / 2,  # Smaller relaxation
            )

        # ===== MOMENTUM TRANSITIONS =====

        elif transition == RegimeTransition.RANGE_TO_TREND:
            # Check if trend aligns with position
            if self._trend_aligns_with_position(position_direction, trend_direction):
                # Favorable - trend in our direction, relax stops
                return RegimeExitSignal(
                    transition=transition,
                    action=RegimeAction.RELAX_STOPS,
                    priority=3,
                    reason="RANGE_TO_TREND_ALIGNED",
                    description=f"RANGE→TREND aligned with {position_direction} - let trend run",
                    confidence=0.80,
                    stop_adjustment_bps=self.trend_stop_relax,
                )
            else:
                # Unfavorable - trend against us, exit
                return RegimeExitSignal(
                    transition=transition,
                    action=RegimeAction.EXIT_IMMEDIATELY,
                    priority=1,
                    reason="RANGE_TO_TREND_OPPOSED",
                    description=f"RANGE→TREND opposed to {position_direction} - exit immediately",
                    confidence=0.85,
                )

        elif transition == RegimeTransition.PANIC_TO_TREND:
            # Check if trend aligns with position
            if self._trend_aligns_with_position(position_direction, trend_direction):
                if has_profit:
                    # Scale out half to lock profit, let rest run
                    return RegimeExitSignal(
                        transition=transition,
                        action=RegimeAction.SCALE_OUT_HALF,
                        priority=2,
                        reason="PANIC_TO_TREND_ALIGNED_PROFIT",
                        description=f"PANIC→TREND aligned with +{position_pnl_bps:.0f} bps - scale out 50%",
                        confidence=0.75,
                    )
                else:
                    # Give position more room as trend may help
                    return RegimeExitSignal(
                        transition=transition,
                        action=RegimeAction.RELAX_STOPS,
                        priority=3,
                        reason="PANIC_TO_TREND_ALIGNED_NO_PROFIT",
                        description=f"PANIC→TREND aligned - relax stops, let trend develop",
                        confidence=0.70,
                        stop_adjustment_bps=self.trend_stop_relax,
                    )
            else:
                # Trend against us - exit immediately
                return RegimeExitSignal(
                    transition=transition,
                    action=RegimeAction.EXIT_IMMEDIATELY,
                    priority=1,
                    reason="PANIC_TO_TREND_OPPOSED",
                    description=f"PANIC→TREND opposed to {position_direction} - exit immediately",
                    confidence=0.90,
                )

        return None

    def _trend_aligns_with_position(
        self,
        position_direction: str,
        trend_direction: Optional[str],
    ) -> bool:
        """Check if trend direction aligns with position direction."""
        if trend_direction is None:
            return False  # Unknown trend direction

        if position_direction == 'buy' and trend_direction == 'up':
            return True
        elif position_direction == 'sell' and trend_direction == 'down':
            return True
        else:
            return False

    def should_override_stops(
        self,
        signal: RegimeExitSignal,
        current_stop_bps: float,
    ) -> Tuple[bool, Optional[float]]:
        """
        Determine if regime signal should override current stop loss.

        Args:
            signal: Regime exit signal
            current_stop_bps: Current stop loss distance in bps

        Returns:
            (should_override, new_stop_bps)
        """
        if signal.action == RegimeAction.EXIT_IMMEDIATELY:
            # Exit signals always override
            return True, 0.0  # Exit at market

        elif signal.action == RegimeAction.TIGHTEN_STOPS:
            # Only tighten if new stop is tighter than current
            if signal.stop_adjustment_bps is not None:
                new_stop = signal.stop_adjustment_bps
                if new_stop < current_stop_bps:
                    return True, new_stop
                else:
                    logger.debug(
                        "regime_stop_not_tighter",
                        current=current_stop_bps,
                        proposed=new_stop,
                    )
                    return False, None

        elif signal.action == RegimeAction.RELAX_STOPS:
            # Only relax if we have profit to protect
            if signal.stop_adjustment_bps is not None:
                new_stop = signal.stop_adjustment_bps
                return True, new_stop

        return False, None

    def get_scale_out_percentage(self, signal: RegimeExitSignal) -> float:
        """Get percentage of position to exit for SCALE_OUT actions."""
        if signal.action == RegimeAction.SCALE_OUT_HALF:
            return 0.5  # 50%
        else:
            return 0.0  # No scale out

    def get_action_description(self, action: RegimeAction) -> str:
        """Get human-readable action description."""
        descriptions = {
            RegimeAction.EXIT_IMMEDIATELY: "EXIT IMMEDIATELY (Market regime invalidates trade)",
            RegimeAction.SCALE_OUT_HALF: "SCALE OUT 50% (Lock partial profit on regime shift)",
            RegimeAction.TIGHTEN_STOPS: "TIGHTEN STOPS (Reduce risk in unfavorable regime)",
            RegimeAction.RELAX_STOPS: "RELAX STOPS (Give position room in favorable regime)",
            RegimeAction.HOLD: "HOLD (Regime change doesn't affect position)",
            RegimeAction.REVERSE_POSITION: "REVERSE POSITION (Rare - complete thesis invalidation)",
        }
        return descriptions.get(action, "UNKNOWN ACTION")
