"""
Dual-Mode Trading Coordinator

Orchestrates dual-mode trading across the Engine:
1. Routes signals to short-hold or long-hold books
2. Manages conflict resolution for assets in both modes
3. Integrates with PPO for per-mode action selection
4. Enforces safety rails
5. Tracks per-mode performance

Flow:
Engines → Alpha Signals → Dual-Mode Coordinator →
    → Short-Hold Gate → book_short
    → Long-Hold Gate → book_long
→ PPO Actions → Position Management → Risk/Costs → Memory
"""

from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import structlog

from .asset_profiles import AssetProfileManager, TradingMode
from .dual_book_manager import DualBookManager, Position
from .mode_policies import LongHoldPolicy, PolicyManager, ShortHoldPolicy, SignalContext
from .safety_rails import SafetyRailsMonitor

logger = structlog.get_logger(__name__)


@dataclass
class DualModeSignal:
    """Signal that has been evaluated for both modes."""

    symbol: str
    context: SignalContext

    # Short-hold evaluation
    short_ok: bool
    short_reason: str

    # Long-hold evaluation
    long_ok: bool
    long_reason: str

    # Routing decision
    route_to: Optional[TradingMode]  # Which mode to route to


@dataclass
class ConflictResolution:
    """Resolution strategy when both modes want to trade same asset."""

    symbol: str
    short_exposure_gbp: float
    long_exposure_gbp: float
    total_exposure_gbp: float
    max_total_exposure_gbp: float
    can_add_short: bool
    can_add_long: bool
    reason: str


class DualModeCoordinator:
    """
    Coordinates dual-mode trading operations.

    Key responsibilities:
    1. Evaluate signals for both short-hold and long-hold modes
    2. Route signals to appropriate books
    3. Resolve conflicts when both modes active on same asset
    4. Monitor safety rails
    5. Track per-mode performance
    """

    def __init__(
        self,
        profile_manager: AssetProfileManager,
        book_manager: DualBookManager,
        policy_manager: PolicyManager,
        safety_monitor: SafetyRailsMonitor,
        total_capital_gbp: float = 10000.0,
    ):
        """
        Initialize dual-mode coordinator.

        Args:
            profile_manager: Asset profile manager
            book_manager: Dual book manager
            policy_manager: Policy manager
            safety_monitor: Safety rails monitor
            total_capital_gbp: Total portfolio capital
        """
        self.profile_manager = profile_manager
        self.book_manager = book_manager
        self.policy_manager = policy_manager
        self.safety_monitor = safety_monitor
        self.total_capital_gbp = total_capital_gbp

        # Track routing decisions (use deque for automatic size limiting)
        self.routing_history = deque(maxlen=1000)

        # Track conflict resolutions (use deque for automatic size limiting)
        self.conflict_history = deque(maxlen=500)

        logger.info(
            "dual_mode_coordinator_initialized",
            total_capital=total_capital_gbp,
        )

    def evaluate_signal(
        self,
        context: SignalContext,
    ) -> DualModeSignal:
        """
        Evaluate signal for both short-hold and long-hold modes.

        Args:
            context: Signal context

        Returns:
            DualModeSignal with evaluation results
        """
        symbol = context.symbol
        profile = self.profile_manager.get_profile(symbol)

        # Evaluate short-hold
        short_ok = False
        short_reason = "Not enabled"

        if self.profile_manager.can_run_short_hold(symbol):
            short_policy = self.policy_manager.get_policy(TradingMode.SHORT_HOLD)
            short_book_state = self.book_manager.get_book_state(TradingMode.SHORT_HOLD)

            short_ok, short_reason = short_policy.should_enter(
                context=context,
                total_capital_gbp=self.total_capital_gbp,
                current_book_exposure_gbp=short_book_state.total_exposure_gbp,
            )

        # Evaluate long-hold
        long_ok = False
        long_reason = "Not enabled"

        if self.profile_manager.can_run_long_hold(symbol):
            long_policy = self.policy_manager.get_policy(TradingMode.LONG_HOLD)
            long_book_state = self.book_manager.get_book_state(TradingMode.LONG_HOLD)

            # Get current asset exposure in long-hold book
            current_asset_exposure = 0.0
            if self.book_manager.has_position(symbol, TradingMode.LONG_HOLD):
                pos = self.book_manager.get_position(symbol, TradingMode.LONG_HOLD)
                if pos:  # Add null check
                    current_asset_exposure = pos.position_size_gbp

            long_ok, long_reason = long_policy.should_enter(
                context=context,
                total_capital_gbp=self.total_capital_gbp,
                current_book_exposure_gbp=long_book_state.total_exposure_gbp,
                current_asset_exposure_gbp=current_asset_exposure,
            )

        # Determine routing
        route_to = None

        if short_ok and long_ok:
            # Both modes want to enter - prioritize based on profile
            if profile.mode == TradingMode.BOTH:
                # Check which mode has more capacity
                short_cap = self._get_mode_capacity(TradingMode.SHORT_HOLD)
                long_cap = self._get_mode_capacity(TradingMode.LONG_HOLD)

                # Route to mode with more capacity and better confidence
                if short_cap > long_cap and context.confidence > 0.60:
                    route_to = TradingMode.SHORT_HOLD
                else:
                    route_to = TradingMode.LONG_HOLD
            elif profile.mode == TradingMode.SHORT_HOLD:
                route_to = TradingMode.SHORT_HOLD
            else:
                route_to = TradingMode.LONG_HOLD

        elif short_ok:
            route_to = TradingMode.SHORT_HOLD
        elif long_ok:
            route_to = TradingMode.LONG_HOLD

        signal = DualModeSignal(
            symbol=symbol,
            context=context,
            short_ok=short_ok,
            short_reason=short_reason,
            long_ok=long_ok,
            long_reason=long_reason,
            route_to=route_to,
        )

        # Track routing decision (deque automatically handles size limit)
        self.routing_history.append(signal)

        logger.debug(
            "signal_evaluated",
            symbol=symbol,
            short_ok=short_ok,
            long_ok=long_ok,
            route_to=route_to.value if route_to else None,
        )

        return signal

    def resolve_conflict(
        self,
        symbol: str,
    ) -> ConflictResolution:
        """
        Resolve conflict when asset has positions in both books.

        Args:
            symbol: Asset symbol

        Returns:
            ConflictResolution with decision
        """
        # Get exposures
        exposure = self.book_manager.get_exposure(symbol)
        short_exp = exposure["short_gbp"]
        long_exp = exposure["long_gbp"]
        total_exp = exposure["total_gbp"]

        # Get per-asset max (from long-hold config, which is usually larger)
        profile = self.profile_manager.get_profile(symbol)
        max_asset_pct = 0.0

        if profile.long_hold:
            max_asset_pct = profile.long_hold.max_book_pct
        elif profile.short_hold:
            max_asset_pct = profile.short_hold.max_book_pct * 2  # Double for conflict case

        max_total_exp = self.total_capital_gbp * max_asset_pct

        # Determine what can be added
        can_add_short = total_exp < max_total_exp and short_exp < max_total_exp * 0.3
        can_add_long = total_exp < max_total_exp and long_exp < max_total_exp * 0.7

        reason = f"Short: {short_exp:.0f}, Long: {long_exp:.0f}, Max: {max_total_exp:.0f}"

        resolution = ConflictResolution(
            symbol=symbol,
            short_exposure_gbp=short_exp,
            long_exposure_gbp=long_exp,
            total_exposure_gbp=total_exp,
            max_total_exposure_gbp=max_total_exp,
            can_add_short=can_add_short,
            can_add_long=can_add_long,
            reason=reason,
        )

        # Track conflict resolution (deque automatically handles size limit)
        self.conflict_history.append(resolution)

        return resolution

    def check_position_safety(
        self,
        symbol: str,
        mode: TradingMode,
        context: SignalContext,
    ) -> Tuple[bool, List[str]]:
        """
        Check position against safety rails.

        Args:
            symbol: Asset symbol
            mode: Trading mode
            context: Signal context

        Returns:
            (is_safe, actions) tuple where actions are recommended
        """
        # Only check long-hold positions
        if mode != TradingMode.LONG_HOLD:
            return (True, [])

        position = self.book_manager.get_position(symbol, mode)
        if not position:
            return (True, [])

        # Check safety rails
        violations = self.safety_monitor.check_position(position, context)

        if not violations:
            return (True, [])

        # Collect recommended actions
        actions = []
        is_safe = True

        for violation in violations:
            logger.warning(
                "safety_rail_violation",
                symbol=symbol,
                rail=violation.rail_type,
                severity=violation.severity,
                message=violation.message,
                action=violation.action,
            )

            if violation.severity == "critical":
                is_safe = False

            actions.append(violation.action)

        return (is_safe, actions)

    def should_add_to_position(
        self,
        symbol: str,
        context: SignalContext,
    ) -> Tuple[bool, str, Optional[float]]:
        """
        Determine if should add to long-hold position.

        Args:
            symbol: Asset symbol
            context: Signal context

        Returns:
            (should_add, reason, add_price) tuple
        """
        # Check if position exists
        position = self.book_manager.get_position(symbol, TradingMode.LONG_HOLD)
        if not position:
            return (False, "No position", None)

        # Check safety rails - clamp adds if needed
        if self.safety_monitor.should_clamp_adds(symbol):
            return (False, "Adds clamped by safety rails", None)

        # Check policy
        long_policy = self.policy_manager.get_policy(TradingMode.LONG_HOLD)
        should_add, reason, add_price = long_policy.should_add(position, context)

        return (should_add, reason, add_price)

    def should_scale_out_position(
        self,
        symbol: str,
        context: SignalContext,
    ) -> Tuple[bool, str, Optional[float]]:
        """
        Determine if should scale out of long-hold position.

        Args:
            symbol: Asset symbol
            context: Signal context

        Returns:
            (should_scale, reason, scale_pct) tuple
        """
        # Check if position exists
        position = self.book_manager.get_position(symbol, TradingMode.LONG_HOLD)
        if not position:
            return (False, "No position", None)

        # Check policy
        long_policy = self.policy_manager.get_policy(TradingMode.LONG_HOLD)
        should_scale, reason, scale_pct = long_policy.should_scale_out(position, context)

        return (should_scale, reason, scale_pct)

    def should_update_trail(
        self,
        symbol: str,
        context: SignalContext,
    ) -> Tuple[bool, str, Optional[float]]:
        """
        Determine if should update trailing stop.

        Args:
            symbol: Asset symbol
            context: Signal context

        Returns:
            (should_update, reason, trail_level_bps) tuple
        """
        # Check if position exists
        position = self.book_manager.get_position(symbol, TradingMode.LONG_HOLD)
        if not position:
            return (False, "No position", None)

        # Only update if in profit
        if position.unrealized_pnl_bps < 50.0:
            return (False, "Not in profit", None)

        # Check if should activate BE lock first
        long_policy = self.policy_manager.get_policy(TradingMode.LONG_HOLD)
        should_be, be_reason = long_policy.should_activate_be_lock(position, context)

        if should_be:
            # Activate BE lock
            self.book_manager.activate_be_lock(symbol, TradingMode.LONG_HOLD)
            logger.info("be_lock_activated", symbol=symbol, reason=be_reason)

        # Calculate trail level
        trail_level = long_policy.calculate_trail_level(position, context)

        # Check if should tighten trail due to safety rails
        if self.safety_monitor.should_tighten_trail(symbol):
            # Tighten by reducing trail distance
            trail_level = max(trail_level, position.unrealized_pnl_bps * 0.7)

        # Only update if new level is higher than current
        if not position.trail_active or trail_level > position.trail_level_bps:
            return (True, "Update trail", trail_level)

        return (False, "Trail already optimal", None)

    def get_mode_stats(self) -> Dict:
        """Get statistics for both modes."""
        combined_stats = self.book_manager.get_combined_stats()

        # Add routing stats
        short_routed = sum(1 for s in self.routing_history if s.route_to == TradingMode.SHORT_HOLD)
        long_routed = sum(1 for s in self.routing_history if s.route_to == TradingMode.LONG_HOLD)

        # Add safety rail stats
        safety_summary = self.safety_monitor.get_violation_summary(hours=24)

        # Add conflict stats
        num_conflicts = len(self.conflict_history)

        combined_stats["routing"] = {
            "total_signals": len(self.routing_history),
            "short_routed": short_routed,
            "long_routed": long_routed,
            "no_route": len(self.routing_history) - short_routed - long_routed,
        }

        combined_stats["safety_rails"] = safety_summary

        # Get recent conflicts (deque doesn't support negative slicing)
        recent_conflicts = list(self.conflict_history)[-100:] if num_conflicts > 0 else []

        combined_stats["conflicts"] = {
            "total_conflicts": num_conflicts,
            "recent_conflicts": len(recent_conflicts),
        }

        return combined_stats

    def _get_mode_capacity(self, mode: TradingMode) -> float:
        """
        Get remaining capacity for a mode.

        Args:
            mode: Trading mode

        Returns:
            Remaining capacity as fraction (0.0-1.0)
        """
        book_state = self.book_manager.get_book_state(mode)
        max_heat = self.total_capital_gbp * (
            self.book_manager.max_short_heat_pct if mode == TradingMode.SHORT_HOLD
            else self.book_manager.max_long_heat_pct
        )

        remaining = max_heat - book_state.total_exposure_gbp
        capacity = remaining / max_heat if max_heat > 0 else 0.0

        return max(0.0, capacity)

    def on_position_closed(self, symbol: str, mode: TradingMode) -> None:
        """
        Clean up tracking when position is closed.

        Args:
            symbol: Asset symbol
            mode: Trading mode
        """
        # Clean up safety monitor
        if mode == TradingMode.LONG_HOLD:
            self.safety_monitor.on_position_closed(symbol)

    def reset_daily(self) -> None:
        """Reset daily statistics."""
        # Reset book daily stats
        short_book = self.book_manager.get_book_state(TradingMode.SHORT_HOLD)
        long_book = self.book_manager.get_book_state(TradingMode.LONG_HOLD)

        short_book.num_trades_today = 0
        short_book.wins_today = 0
        short_book.realized_pnl_gbp = 0.0

        long_book.num_trades_today = 0
        long_book.wins_today = 0
        long_book.realized_pnl_gbp = 0.0

        logger.info("daily_stats_reset")


def create_dual_mode_system(
    total_capital_gbp: float = 10000.0,
) -> Tuple[DualModeCoordinator, AssetProfileManager, DualBookManager]:
    """
    Factory function to create complete dual-mode system.

    Args:
        total_capital_gbp: Total portfolio capital

    Returns:
        (coordinator, profile_manager, book_manager) tuple
    """
    # Create components
    profile_manager = AssetProfileManager()
    book_manager = DualBookManager()
    policy_manager = PolicyManager()
    safety_monitor = SafetyRailsMonitor()

    # Create coordinator
    coordinator = DualModeCoordinator(
        profile_manager=profile_manager,
        book_manager=book_manager,
        policy_manager=policy_manager,
        safety_monitor=safety_monitor,
        total_capital_gbp=total_capital_gbp,
    )

    logger.info(
        "dual_mode_system_created",
        total_capital=total_capital_gbp,
        num_profiles=len(profile_manager.profiles),
    )

    return coordinator, profile_manager, book_manager
