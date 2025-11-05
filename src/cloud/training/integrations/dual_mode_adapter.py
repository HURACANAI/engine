"""
Dual-Mode Integration Adapter

Provides a clean integration layer between the dual-mode trading system
and the existing Engine pipeline.

This adapter:
1. Wraps the dual-mode coordinator for easy plugging
2. Translates existing signals into dual-mode format
3. Routes PPO actions through appropriate books
4. Provides unified interface for monitoring
5. Handles state conversion for RL agent
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import structlog

from ..agents.rl_agent import TradingAction, TradingState
from ..models.alpha_engines import AlphaEngineCoordinator, AlphaSignal
from ..models.asset_profiles import AssetProfileManager, TradingMode
from ..models.dual_book_manager import DualBookManager
from ..models.dual_mode_coordinator import DualModeCoordinator, create_dual_mode_system
from ..models.mode_policies import PolicyManager, SignalContext
from ..models.safety_rails import SafetyRailsMonitor

logger = structlog.get_logger(__name__)


@dataclass
class DualModeConfig:
    """Configuration for dual-mode integration."""

    enabled: bool = True
    total_capital_gbp: float = 10000.0

    # Book limits
    max_short_heat_pct: float = 0.20
    max_long_heat_pct: float = 0.50

    # Safety rails
    enable_safety_rails: bool = True
    log_routing_decisions: bool = True
    log_conflicts: bool = True


class DualModeAdapter:
    """
    Adapter for integrating dual-mode system into existing Engine pipeline.

    This provides a clean interface that doesn't require refactoring
    the entire existing codebase.
    """

    def __init__(
        self,
        config: Optional[DualModeConfig] = None,
        alpha_coordinator: Optional[AlphaEngineCoordinator] = None,
    ):
        """
        Initialize dual-mode adapter.

        Args:
            config: Dual-mode configuration
            alpha_coordinator: Optional alpha engine coordinator
        """
        self.config = config or DualModeConfig()
        self.alpha_coordinator = alpha_coordinator

        # Create dual-mode system
        self.coordinator, self.profile_manager, self.book_manager = create_dual_mode_system(
            total_capital_gbp=self.config.total_capital_gbp,
        )

        # Track current mode context per symbol
        self.current_modes: Dict[str, TradingMode] = {}

        # Performance tracking
        self.total_pnl_short = 0.0
        self.total_pnl_long = 0.0
        self.trades_short = 0
        self.trades_long = 0

        logger.info(
            "dual_mode_adapter_initialized",
            enabled=self.config.enabled,
            total_capital=self.config.total_capital_gbp,
        )

    def evaluate_for_entry(
        self,
        symbol: str,
        features: Dict[str, float],
        regime: str,
        confidence: float,
        current_price: float,
        timestamp: Optional[datetime] = None,
    ) -> Tuple[bool, Optional[TradingMode], str]:
        """
        Evaluate if should enter position and which mode.

        Args:
            symbol: Asset symbol
            features: Feature dictionary
            regime: Market regime
            confidence: Signal confidence
            current_price: Current price
            timestamp: Current timestamp

        Returns:
            (should_enter, mode, reason) tuple
        """
        if not self.config.enabled:
            return (False, None, "Dual-mode disabled")

        # Build signal context
        context = SignalContext(
            symbol=symbol,
            current_price=current_price,
            features=features,
            regime=regime,
            confidence=confidence,
            eps_net=features.get("eps_net", 0.0),
            volatility_bps=features.get("volatility_bps", 80.0),
            spread_bps=features.get("spread_bps", 10.0),
            htf_bias=features.get("htf_bias", 0.5),
            timestamp=timestamp or datetime.now(),
        )

        # Evaluate signal
        signal = self.coordinator.evaluate_signal(context)

        if signal.route_to is not None:
            # Store current mode for this symbol
            self.current_modes[symbol] = signal.route_to

            if self.config.log_routing_decisions:
                logger.info(
                    "signal_routed",
                    symbol=symbol,
                    mode=signal.route_to.value,
                    confidence=confidence,
                    short_ok=signal.short_ok,
                    long_ok=signal.long_ok,
                )

            return (True, signal.route_to, "Signal approved")
        else:
            reason = signal.short_reason if not signal.short_ok else signal.long_reason
            return (False, None, reason)

    def open_position(
        self,
        symbol: str,
        mode: TradingMode,
        entry_price: float,
        size_gbp: float,
        stop_loss_bps: float,
        take_profit_bps: Optional[float] = None,
        entry_regime: str = "unknown",
        entry_technique: str = "unknown",
        entry_confidence: float = 0.0,
    ) -> bool:
        """
        Open position in specified mode.

        Args:
            symbol: Asset symbol
            mode: Trading mode
            entry_price: Entry price
            size_gbp: Position size
            stop_loss_bps: Stop loss
            take_profit_bps: Take profit (optional)
            entry_regime: Market regime
            entry_technique: Trading technique
            entry_confidence: Entry confidence

        Returns:
            True if opened successfully
        """
        if not self.config.enabled:
            return False

        # Check if can open
        can_open, reason = self.book_manager.can_open_position(
            symbol=symbol,
            mode=mode,
            size_gbp=size_gbp,
            total_capital_gbp=self.config.total_capital_gbp,
        )

        if not can_open:
            logger.warning("cannot_open_position", symbol=symbol, mode=mode.value, reason=reason)
            return False

        # Open position
        position = self.book_manager.open_position(
            symbol=symbol,
            mode=mode,
            entry_price=entry_price,
            size_gbp=size_gbp,
            stop_loss_bps=stop_loss_bps,
            take_profit_bps=take_profit_bps,
            entry_regime=entry_regime,
            entry_technique=entry_technique,
            entry_confidence=entry_confidence,
        )

        return position is not None

    def update_positions(
        self,
        symbol: str,
        current_price: float,
        features: Dict[str, float],
        regime: str,
        timestamp: Optional[datetime] = None,
    ) -> List[str]:
        """
        Update positions and check for actions.

        Args:
            symbol: Asset symbol
            current_price: Current price
            features: Feature dictionary
            regime: Market regime
            timestamp: Current timestamp

        Returns:
            List of actions taken
        """
        if not self.config.enabled:
            return []

        actions_taken = []
        timestamp = timestamp or datetime.now()

        # Build context
        context = SignalContext(
            symbol=symbol,
            current_price=current_price,
            features=features,
            regime=regime,
            confidence=features.get("confidence", 0.5),
            eps_net=features.get("eps_net", 0.0),
            volatility_bps=features.get("volatility_bps", 80.0),
            spread_bps=features.get("spread_bps", 10.0),
            htf_bias=features.get("htf_bias", 0.5),
            timestamp=timestamp,
        )

        # Update both books
        for mode in [TradingMode.SHORT_HOLD, TradingMode.LONG_HOLD]:
            position = self.book_manager.update_position_price(
                symbol=symbol,
                mode=mode,
                current_price=current_price,
            )

            if position is None:
                continue

            # Check safety rails for long-hold
            if mode == TradingMode.LONG_HOLD:
                is_safe, safety_actions = self.coordinator.check_position_safety(
                    symbol=symbol,
                    mode=mode,
                    context=context,
                )

                if not is_safe:
                    # Close position due to safety violation
                    _, pnl = self.book_manager.close_position(symbol, mode, current_price)
                    actions_taken.append(f"CLOSED_{mode.value}_SAFETY")
                    self._track_pnl(mode, pnl)
                    continue

                # Check for adds
                should_add, add_reason, add_price = self.coordinator.should_add_to_position(
                    symbol=symbol,
                    context=context,
                )

                if should_add:
                    # Execute add
                    add_size = position.position_size_gbp * 0.5  # 50% add
                    self.book_manager.add_to_position(
                        symbol=symbol,
                        mode=mode,
                        add_price=current_price,
                        add_size_gbp=add_size,
                    )
                    actions_taken.append(f"ADDED_{mode.value}")

                # Check for scale-out
                should_scale, scale_reason, scale_pct = self.coordinator.should_scale_out_position(
                    symbol=symbol,
                    context=context,
                )

                if should_scale and scale_pct:
                    # Execute scale-out
                    _, pnl = self.book_manager.scale_out_position(
                        symbol=symbol,
                        mode=mode,
                        exit_price=current_price,
                        scale_pct=scale_pct,
                    )
                    actions_taken.append(f"SCALED_OUT_{mode.value}")
                    self._track_pnl(mode, pnl)

                # Check for trail update
                should_trail, trail_reason, trail_level = self.coordinator.should_update_trail(
                    symbol=symbol,
                    context=context,
                )

                if should_trail and trail_level is not None:
                    # Update trail
                    self.book_manager.update_trail_level(
                        symbol=symbol,
                        mode=mode,
                        trail_level_bps=trail_level,
                    )
                    actions_taken.append(f"TRAIL_UPDATED_{mode.value}")

            # Check policy for exit
            policy = self.coordinator.policy_manager.get_policy(mode)
            should_exit, exit_reason, exit_type = policy.should_exit(position, context)

            if should_exit:
                # Close position
                _, pnl = self.book_manager.close_position(symbol, mode, current_price)
                actions_taken.append(f"EXITED_{mode.value}_{exit_type}")
                self._track_pnl(mode, pnl)

                # Clean up tracking
                self.coordinator.on_position_closed(symbol, mode)

        return actions_taken

    def get_trading_state_for_rl(
        self,
        symbol: str,
        market_features: np.ndarray,
        base_state: TradingState,
    ) -> TradingState:
        """
        Enhance trading state with dual-mode information.

        Args:
            symbol: Asset symbol
            market_features: Market feature array
            base_state: Base trading state

        Returns:
            Enhanced trading state with dual-mode fields
        """
        # Determine current mode
        current_mode = self.current_modes.get(symbol, "short_hold")

        # Get position info
        short_pos = self.book_manager.get_position(symbol, TradingMode.SHORT_HOLD)
        long_pos = self.book_manager.get_position(symbol, TradingMode.LONG_HOLD)

        # Update state with dual-mode fields
        base_state.trading_mode = current_mode
        base_state.has_short_position = short_pos is not None
        base_state.has_long_position = long_pos is not None

        if short_pos:
            base_state.short_position_pnl_bps = short_pos.unrealized_pnl_bps

        if long_pos:
            base_state.long_position_pnl_bps = long_pos.unrealized_pnl_bps
            base_state.long_position_age_hours = long_pos.age_hours(datetime.now())
            base_state.num_adds = long_pos.add_count
            base_state.be_lock_active = long_pos.be_lock_active
            base_state.trail_active = long_pos.trail_active

        return base_state

    def execute_rl_action(
        self,
        symbol: str,
        action: TradingAction,
        current_price: float,
        features: Dict[str, float],
    ) -> Tuple[bool, str]:
        """
        Execute RL agent action through dual-mode system.

        Args:
            symbol: Asset symbol
            action: Trading action
            current_price: Current price
            features: Feature dictionary

        Returns:
            (success, message) tuple
        """
        mode = self.current_modes.get(symbol, TradingMode.SHORT_HOLD)

        # Route action based on type
        if action == TradingAction.SCRATCH:
            # Fast exit for short-hold
            if self.book_manager.has_position(symbol, TradingMode.SHORT_HOLD):
                _, pnl = self.book_manager.close_position(
                    symbol, TradingMode.SHORT_HOLD, current_price
                )
                self._track_pnl(TradingMode.SHORT_HOLD, pnl)
                return (True, f"Scratched: {pnl:.2f} GBP")
            return (False, "No short position to scratch")

        elif action == TradingAction.ADD_GRID:
            # Add to long-hold
            if self.book_manager.has_position(symbol, TradingMode.LONG_HOLD):
                pos = self.book_manager.get_position(symbol, TradingMode.LONG_HOLD)
                add_size = pos.position_size_gbp * 0.5
                self.book_manager.add_to_position(
                    symbol, TradingMode.LONG_HOLD, current_price, add_size
                )
                return (True, f"Added {add_size:.0f} GBP")
            return (False, "No long position to add to")

        elif action == TradingAction.SCALE_OUT:
            # Scale out of long-hold
            if self.book_manager.has_position(symbol, TradingMode.LONG_HOLD):
                _, pnl = self.book_manager.scale_out_position(
                    symbol, TradingMode.LONG_HOLD, current_price, 0.33
                )
                self._track_pnl(TradingMode.LONG_HOLD, pnl)
                return (True, f"Scaled out: {pnl:.2f} GBP")
            return (False, "No long position to scale out")

        elif action == TradingAction.TRAIL_RUNNER:
            # Activate trail for long-hold
            if self.book_manager.has_position(symbol, TradingMode.LONG_HOLD):
                pos = self.book_manager.get_position(symbol, TradingMode.LONG_HOLD)
                # Calculate trail level (simplified)
                trail_level = pos.unrealized_pnl_bps - 100.0  # Trail 100 bps below current
                self.book_manager.update_trail_level(
                    symbol, TradingMode.LONG_HOLD, trail_level
                )
                return (True, f"Trail activated at {trail_level:.1f} bps")
            return (False, "No long position to trail")

        # Default actions handled by caller
        return (False, "Action not handled by dual-mode")

    def get_statistics(self) -> Dict:
        """Get comprehensive statistics."""
        stats = self.coordinator.get_mode_stats()

        # Add adapter-level stats
        stats["adapter"] = {
            "enabled": self.config.enabled,
            "total_pnl_short": self.total_pnl_short,
            "total_pnl_long": self.total_pnl_long,
            "trades_short": self.trades_short,
            "trades_long": self.trades_long,
            "avg_pnl_short": self.total_pnl_short / max(self.trades_short, 1),
            "avg_pnl_long": self.total_pnl_long / max(self.trades_long, 1),
        }

        return stats

    def reset_daily(self) -> None:
        """Reset daily statistics."""
        self.coordinator.reset_daily()
        logger.info("dual_mode_adapter_daily_reset")

    def _track_pnl(self, mode: TradingMode, pnl: float) -> None:
        """Track P&L by mode."""
        if mode == TradingMode.SHORT_HOLD:
            self.total_pnl_short += pnl
            self.trades_short += 1
        else:
            self.total_pnl_long += pnl
            self.trades_long += 1
