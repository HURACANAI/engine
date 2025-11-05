"""
Per-Mode Trading Policies

Defines entry/exit/sizing logic for each trading mode:
- ShortHoldPolicy: Fast scalping with tight stops
- LongHoldPolicy: Swing trading with adds, scale-outs, and trailing stops

Each policy implements:
- should_enter(): Entry gate logic
- should_exit(): Exit conditions
- should_add(): Add/DCA conditions (long-hold only)
- should_scale_out(): Partial exit conditions (long-hold only)
- calculate_stop_loss(): Stop loss calculation
- calculate_take_profit(): Take profit calculation
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional, Tuple

import structlog

from .asset_profiles import LongHoldConfig, ShortHoldConfig, TrailStyle, TradingMode
from .dual_book_manager import Position

logger = structlog.get_logger(__name__)


@dataclass
class SignalContext:
    """Context for evaluating signals."""

    symbol: str
    current_price: float
    features: Dict[str, float]
    regime: str  # "trend", "range", "panic", etc.
    confidence: float
    eps_net: float  # Edge Per Second (net of costs)
    volatility_bps: float
    spread_bps: float
    htf_bias: float  # Higher timeframe bias (-1 to 1)
    timestamp: datetime


class ShortHoldPolicy:
    """
    Policy for short-hold (scalp) mode.

    Philosophy:
    - Fast in, fast out
    - Bank £1-£2 repeatedly
    - Tight stops
    - High win rate, small winners
    - Maker bias when possible
    """

    def __init__(self, config: ShortHoldConfig):
        """
        Initialize short-hold policy.

        Args:
            config: Short-hold configuration
        """
        self.config = config

        logger.info(
            "short_hold_policy_initialized",
            target_profit_bps=config.target_profit_bps,
            max_hold_minutes=config.max_hold_minutes,
        )

    def should_enter(
        self,
        context: SignalContext,
        total_capital_gbp: float,
        current_book_exposure_gbp: float,
    ) -> Tuple[bool, str]:
        """
        Determine if should enter short-hold position.

        Args:
            context: Signal context
            total_capital_gbp: Total portfolio capital
            current_book_exposure_gbp: Current short-hold book exposure

        Returns:
            (should_enter, reason) tuple
        """
        # Gate 1: EPS filter (edge per second must be positive)
        if context.eps_net <= 0:
            return (False, f"Negative EPS: {context.eps_net:.4f}")

        # Gate 2: Cost gate (spread must be reasonable)
        if context.spread_bps > 15.0:  # Max 15 bps spread for scalps
            return (False, f"Spread too wide: {context.spread_bps:.1f} bps")

        # Gate 3: Confidence threshold
        if context.confidence < 0.55:
            return (False, f"Low confidence: {context.confidence:.2f}")

        # Gate 4: Microstructure check (from features)
        micro_score = context.features.get("micro_score", 50.0)
        if micro_score < 55.0:
            return (False, f"Weak microstructure: {micro_score:.0f}")

        # Gate 5: Book capacity
        max_book_exposure = total_capital_gbp * self.config.max_book_pct
        if current_book_exposure_gbp >= max_book_exposure:
            return (False, f"Short-hold book full: {current_book_exposure_gbp:.0f}/{max_book_exposure:.0f} GBP")

        # Gate 6: Regime check (avoid entering in PANIC for scalps)
        if context.regime == "panic":
            return (False, "PANIC regime: avoid short-hold entries")

        return (True, "All gates passed")

    def should_exit(
        self,
        position: Position,
        context: SignalContext,
    ) -> Tuple[bool, str, Optional[str]]:
        """
        Determine if should exit short-hold position.

        Args:
            position: Current position
            context: Signal context

        Returns:
            (should_exit, reason, exit_type) tuple where exit_type is "tp", "stop", "scratch", or "time"
        """
        # Check 1: Take profit hit
        if position.unrealized_pnl_bps >= self.config.target_profit_bps:
            return (True, f"TP hit: {position.unrealized_pnl_bps:.1f} bps", "tp")

        # Check 2: Stop loss hit
        if position.unrealized_pnl_bps <= position.stop_loss_bps:
            return (True, f"Stop loss: {position.unrealized_pnl_bps:.1f} bps", "stop")

        # Check 3: Scratch if near break-even and micro deteriorates
        if abs(position.unrealized_pnl_bps) < 5.0:  # Near break-even
            micro_score = context.features.get("micro_score", 50.0)
            if micro_score < 45.0:  # Micro turned bad
                return (True, f"Scratch on weak micro: {micro_score:.0f}", "scratch")

        # Check 4: Time stop
        age_minutes = position.age_minutes(context.timestamp)
        if age_minutes > self.config.max_hold_minutes:
            if position.unrealized_pnl_bps < 0:
                # Holding too long and losing, exit
                return (True, f"Time stop underwater: {age_minutes} min", "time")
            elif position.unrealized_pnl_bps < self.config.target_profit_bps * 0.5:
                # Holding too long but not near target, exit
                return (True, f"Time stop stalled: {age_minutes} min", "time")

        # Check 5: Regime flip
        if context.regime == "panic" and position.unrealized_pnl_bps < self.config.target_profit_bps:
            return (True, "Regime flip to PANIC", "scratch")

        return (False, "Hold", None)

    def calculate_stop_loss_bps(self, context: SignalContext) -> float:
        """
        Calculate stop loss for short-hold entry.

        Args:
            context: Signal context

        Returns:
            Stop loss in basis points (negative value)
        """
        # Base stop: tight for scalps
        base_stop = -8.0  # -8 bps base

        # Adjust for volatility
        vol_adjustment = context.volatility_bps / 100.0  # Scale by vol
        adjusted_stop = base_stop - vol_adjustment

        # Floor at -15 bps (max loss)
        return max(adjusted_stop, -15.0)

    def calculate_take_profit_bps(self, context: SignalContext) -> float:
        """
        Calculate take profit for short-hold entry.

        Args:
            context: Signal context

        Returns:
            Take profit in basis points
        """
        return self.config.target_profit_bps


class LongHoldPolicy:
    """
    Policy for long-hold (swing) mode.

    Philosophy:
    - Hold through dips
    - Maximize winners
    - Add on dips (DCA)
    - Scale out at profit levels
    - Trail with structure or ATR
    - Require HTF bias and regime alignment
    """

    def __init__(self, config: LongHoldConfig):
        """
        Initialize long-hold policy.

        Args:
            config: Long-hold configuration
        """
        self.config = config

        logger.info(
            "long_hold_policy_initialized",
            max_book_pct=config.max_book_pct,
            min_hold_hours=config.min_hold_hours,
        )

    def should_enter(
        self,
        context: SignalContext,
        total_capital_gbp: float,
        current_book_exposure_gbp: float,
        current_asset_exposure_gbp: float,
    ) -> Tuple[bool, str]:
        """
        Determine if should enter long-hold position.

        Args:
            context: Signal context
            total_capital_gbp: Total portfolio capital
            current_book_exposure_gbp: Current long-hold book exposure
            current_asset_exposure_gbp: Current exposure to this asset in long-hold

        Returns:
            (should_enter, reason) tuple
        """
        # Gate 1: HTF bias alignment (must have positive HTF bias)
        if context.htf_bias < 0.2:  # Need bullish HTF
            return (False, f"HTF bias too weak: {context.htf_bias:.2f}")

        # Gate 2: Regime check (avoid PANIC entries unless override)
        if context.regime == "panic" and not self.config.panic_override:
            return (False, "PANIC regime: no long-hold entries")

        # Gate 3: Confidence threshold (higher than short-hold)
        if context.confidence < 0.60:
            return (False, f"Low confidence: {context.confidence:.2f}")

        # Gate 4: Setup quality (look for compression→ignition or trend)
        ignition_score = context.features.get("ignition_score", 0.0)
        trend_strength = context.features.get("trend_strength", 0.0)

        has_setup = ignition_score > 60.0 or abs(trend_strength) > 0.6
        if not has_setup:
            return (False, f"No setup: ignition={ignition_score:.0f}, trend={trend_strength:.2f}")

        # Gate 5: Asset-level capacity
        max_asset_exposure = total_capital_gbp * self.config.max_book_pct
        if current_asset_exposure_gbp >= max_asset_exposure:
            return (False, f"Asset limit reached: {current_asset_exposure_gbp:.0f}/{max_asset_exposure:.0f} GBP")

        return (True, "Long-hold gates passed")

    def should_exit(
        self,
        position: Position,
        context: SignalContext,
    ) -> Tuple[bool, str, Optional[str]]:
        """
        Determine if should exit long-hold position.

        Args:
            position: Current position
            context: Signal context

        Returns:
            (should_exit, reason, exit_type) tuple
        """
        age_hours = position.age_hours(context.timestamp)

        # Check 1: Honor min hold period (ignore noise early)
        if age_hours < self.config.min_hold_hours:
            # Only exit if truly catastrophic
            if position.unrealized_pnl_bps < -200.0:
                return (True, f"Catastrophic loss: {position.unrealized_pnl_bps:.1f} bps", "stop")
            return (False, f"Min hold period: {age_hours:.1f}h / {self.config.min_hold_hours}h", None)

        # Check 2: Max floating DD
        if position.unrealized_pnl_bps <= -self.config.max_floating_dd_bps:
            return (True, f"Max DD: {position.unrealized_pnl_bps:.1f} bps", "stop")

        # Check 3: Time stop
        if age_hours > self.config.max_hold_days * 24:
            return (True, f"Max hold time: {age_hours:.1f}h", "time")

        # Check 4: Regime break
        if context.regime == "panic" and self.config.panic_override:
            if position.unrealized_pnl_bps < 50.0:  # Not in good profit
                return (True, "PANIC regime break", "stop")

        # Check 5: HTF bias break
        if context.htf_bias < -0.2 and position.unrealized_pnl_bps < 100.0:
            return (True, f"HTF bias break: {context.htf_bias:.2f}", "stop")

        # Check 6: Trail stop if active
        if position.trail_active:
            current_pnl_bps = position.unrealized_pnl_bps
            trail_level = position.trail_level_bps

            if current_pnl_bps < trail_level:
                return (True, f"Trail stop: {current_pnl_bps:.1f} < {trail_level:.1f}", "trail")

        return (False, "Hold", None)

    def should_add(
        self,
        position: Position,
        context: SignalContext,
    ) -> Tuple[bool, str, Optional[float]]:
        """
        Determine if should add to position (DCA).

        Args:
            position: Current position
            context: Signal context

        Returns:
            (should_add, reason, add_price_target) tuple
        """
        # Check 1: Age check (don't add too early)
        age_hours = position.age_hours(context.timestamp)
        if age_hours < 2.0:
            return (False, "Too early to add", None)

        # Check 2: Max adds
        if position.add_count >= len(self.config.add_grid_bps):
            return (False, f"Max adds reached: {position.add_count}", None)

        # Check 3: Grid levels
        if not self.config.add_grid_bps:
            return (False, "No add grid configured", None)

        # Check which grid level we're at
        next_add_level = self.config.add_grid_bps[position.add_count]
        current_drawdown_bps = position.unrealized_pnl_bps

        if current_drawdown_bps <= next_add_level:
            # We've hit the next add level
            # But only add if:
            # 1. Regime is still valid (not PANIC)
            # 2. HTF bias hasn't broken
            if context.regime == "panic":
                return (False, "PANIC regime: no adds", None)

            if context.htf_bias < 0.0:
                return (False, f"HTF bias broke: {context.htf_bias:.2f}", None)

            # Calculate add price target
            add_price = context.current_price

            return (True, f"Grid add at {next_add_level:.0f} bps", add_price)

        return (False, f"Not at add level yet: {current_drawdown_bps:.1f} > {next_add_level:.1f}", None)

    def should_scale_out(
        self,
        position: Position,
        context: SignalContext,
    ) -> Tuple[bool, str, Optional[float]]:
        """
        Determine if should scale out of position.

        Args:
            position: Current position
            context: Signal context

        Returns:
            (should_scale_out, reason, scale_pct) tuple
        """
        current_pnl_bps = position.unrealized_pnl_bps

        # Check 1: Not in profit yet
        if current_pnl_bps < 50.0:
            return (False, "Not in profit", None)

        # Check 2: Already scaled out too many times
        if position.scale_out_count >= len(self.config.tp_multipliers):
            return (False, "All scale-outs done", None)

        # Determine which TP level we're at
        # Calculate TP levels based on ATR or fixed levels
        base_tp = 100.0  # Base 100 bps
        atr_bps = context.volatility_bps  # Use volatility as proxy for ATR

        # Calculate TP levels
        tp_levels = [base_tp * mult for mult in self.config.tp_multipliers]

        # Check which TP level we've exceeded
        for i, tp_level in enumerate(tp_levels):
            if i == position.scale_out_count:  # Next scale-out level
                scaled_tp = tp_level * (atr_bps / 100.0)  # Scale by ATR
                if current_pnl_bps >= scaled_tp:
                    # Hit this TP level
                    scale_pct = 0.33  # Scale out 33% each time
                    return (True, f"TP{i+1} hit: {current_pnl_bps:.1f} >= {scaled_tp:.1f}", scale_pct)

        return (False, "No TP level hit", None)

    def should_activate_be_lock(
        self,
        position: Position,
        context: SignalContext,
    ) -> Tuple[bool, str]:
        """
        Determine if should activate break-even lock.

        Args:
            position: Current position
            context: Signal context

        Returns:
            (should_activate, reason) tuple
        """
        if position.be_lock_active:
            return (False, "BE lock already active")

        if position.unrealized_pnl_bps >= self.config.be_lock_after_bps:
            return (True, f"BE lock threshold hit: {position.unrealized_pnl_bps:.1f} bps")

        return (False, f"Not at BE lock threshold: {position.unrealized_pnl_bps:.1f} < {self.config.be_lock_after_bps}")

    def calculate_trail_level(
        self,
        position: Position,
        context: SignalContext,
    ) -> float:
        """
        Calculate trailing stop level.

        Args:
            position: Current position
            context: Signal context

        Returns:
            Trail level in basis points
        """
        current_pnl_bps = position.unrealized_pnl_bps

        if self.config.trail_style == TrailStyle.CHANDELIER_ATR_2:
            atr_bps = context.volatility_bps
            trail_distance = 2.0 * atr_bps
            trail_level = current_pnl_bps - trail_distance

        elif self.config.trail_style == TrailStyle.CHANDELIER_ATR_3:
            atr_bps = context.volatility_bps
            trail_distance = 3.0 * atr_bps
            trail_level = current_pnl_bps - trail_distance

        elif self.config.trail_style == TrailStyle.STRUCTURE_SWING:
            # Structure-based: use recent swing low (simplified)
            # In reality, would look at actual swing points
            trail_distance = 100.0  # Default 100 bps
            trail_level = current_pnl_bps - trail_distance

        else:  # FIXED_BPS
            trail_distance = 100.0
            trail_level = current_pnl_bps - trail_distance

        # Never trail below break-even if BE lock is active
        if position.be_lock_active:
            trail_level = max(trail_level, 0.0)

        return trail_level

    def calculate_stop_loss_bps(self, context: SignalContext) -> float:
        """
        Calculate initial stop loss for long-hold entry.

        Args:
            context: Signal context

        Returns:
            Stop loss in basis points (negative value)
        """
        # Long-hold uses wider stops
        atr_bps = context.volatility_bps
        stop_distance = 2.0 * atr_bps  # 2x ATR

        # Floor at -150 bps, ceiling at -500 bps
        stop_loss = -min(max(stop_distance, 150.0), 500.0)

        return stop_loss

    def calculate_initial_size_multiplier(self, context: SignalContext) -> float:
        """
        Calculate position size multiplier for entry.

        Args:
            context: Signal context

        Returns:
            Size multiplier (0.5 - 2.0)
        """
        # Base on confidence
        if context.confidence > 0.75:
            return 1.5  # Large
        elif context.confidence > 0.65:
            return 1.0  # Normal
        else:
            return 0.7  # Smaller


class PolicyManager:
    """Manages both short-hold and long-hold policies."""

    def __init__(
        self,
        short_hold_config: Optional[ShortHoldConfig] = None,
        long_hold_config: Optional[LongHoldConfig] = None,
    ):
        """
        Initialize policy manager.

        Args:
            short_hold_config: Short-hold configuration
            long_hold_config: Long-hold configuration
        """
        self.short_policy = ShortHoldPolicy(short_hold_config or ShortHoldConfig())
        self.long_policy = LongHoldPolicy(long_hold_config or LongHoldConfig())

        logger.info("policy_manager_initialized")

    def get_policy(self, mode: TradingMode):
        """Get policy for specified mode."""
        if mode == TradingMode.SHORT_HOLD:
            return self.short_policy
        else:
            return self.long_policy
