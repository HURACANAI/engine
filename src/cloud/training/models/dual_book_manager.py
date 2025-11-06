"""
Dual-Book Position Manager for Dual-Mode Trading.

This implementation manages two independent books (short-hold and long-hold)
while exposing a high-level API tailored for the dual-mode coordinator,
integration adapter, and demo scripts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import structlog

from .asset_profiles import TradingMode

logger = structlog.get_logger(__name__)


def _now() -> datetime:
    """Return a timestamp helper (kept naive to match tests)."""
    return datetime.now()


@dataclass
class Position:
    """
    Represents an open position in one of the dual books.

    All size and PnL values are represented in GBP notionals so the manager can
    operate without direct custody data. PnL calculations use simple return
    approximations which are sufficient for simulation/demo usage.
    """

    symbol: str
    mode: TradingMode
    entry_price: float
    entry_timestamp: datetime
    position_size_gbp: float
    stop_loss_bps: float
    take_profit_bps: Optional[float] = None
    direction: str = "long"
    entry_regime: str = "unknown"
    entry_confidence: float = 0.0

    current_price: Optional[float] = None
    unrealized_pnl_bps: float = 0.0
    unrealized_pnl_gbp: float = 0.0
    realized_pnl_gbp: float = 0.0
    add_count: int = 0
    scale_out_count: int = 0
    be_lock_active: bool = False
    trail_active: bool = False
    trail_level_bps: float = 0.0
    last_update: datetime = field(default_factory=_now)

    def _pnl_ratio(self, price: float) -> float:
        """Return PnL as a fraction of entry price (positive for gains)."""
        if self.entry_price == 0:
            return 0.0
        if self.direction == "short":
            return (self.entry_price - price) / self.entry_price
        return (price - self.entry_price) / self.entry_price

    def update_price(self, price: float) -> None:
        """Update mark price and recompute unrealized PnL."""
        self.current_price = price
        ratio = self._pnl_ratio(price)
        self.unrealized_pnl_bps = ratio * 10_000.0
        self.unrealized_pnl_gbp = ratio * self.position_size_gbp
        self.last_update = _now()

    def record_add(self, add_price: float, add_size_gbp: float) -> None:
        """Blend entry price when scaling into a position."""
        if add_size_gbp <= 0:
            return

        prior_size = self.position_size_gbp
        new_size = prior_size + add_size_gbp
        weighted_price = (
            (self.entry_price * prior_size) + (add_price * add_size_gbp)
        )
        self.entry_price = weighted_price / max(new_size, 1e-9)
        self.position_size_gbp = new_size
        self.add_count += 1

        # Refresh unrealized PnL using the latest known price
        last_price = self.current_price if self.current_price is not None else add_price
        self.update_price(last_price)

    def record_scale_out(self, exit_price: float, scale_pct: float) -> float:
        """Scale out of the position and return realized PnL for the slice."""
        scale_pct = max(0.0, min(scale_pct, 1.0))
        if scale_pct == 0.0 or self.position_size_gbp <= 0:
            return 0.0

        size_to_close = self.position_size_gbp * scale_pct
        pnl = self._pnl_ratio(exit_price) * size_to_close

        self.position_size_gbp -= size_to_close
        self.realized_pnl_gbp += pnl
        self.scale_out_count += 1
        self.update_price(exit_price)

        return pnl

    def record_close(self, exit_price: float) -> float:
        """Close the full position and return realized PnL."""
        pnl = self._pnl_ratio(exit_price) * self.position_size_gbp
        self.realized_pnl_gbp += pnl
        self.position_size_gbp = 0.0
        self.update_price(exit_price)
        return pnl

    def age_hours(self, current_time: Optional[datetime] = None) -> float:
        """Return holding duration in hours."""
        current_time = current_time or _now()
        delta = current_time - self.entry_timestamp
        return delta.total_seconds() / 3600.0


@dataclass
class BookState:
    """Snapshot of a trading book."""

    mode: TradingMode
    positions: List[Position]
    num_positions: int
    total_exposure_gbp: float
    heat_pct: float
    num_trades_today: int
    wins_today: int
    realized_pnl_gbp: float
    unrealized_pnl_gbp: float


class DualBookManager:
    """
    Manages dual trading books (short-hold and long-hold).

    Provides a high-level API expected by the coordinator, integration adapter,
    and demo scripts. The implementation keeps minimal state so it can operate
    in simulations without exchange connectivity.
    """

    def __init__(
        self,
        total_capital_gbp: float = 10_000.0,
        max_short_heat_pct: float = 0.20,
        max_long_heat_pct: float = 0.50,
        reserve_heat_pct: float = 0.10,
    ):
        self.total_capital_gbp = total_capital_gbp
        self.max_heat_pct = {
            TradingMode.SHORT_HOLD: max_short_heat_pct,
            TradingMode.LONG_HOLD: max_long_heat_pct,
        }
        self.reserve_heat_pct = reserve_heat_pct

        self._positions: Dict[TradingMode, Dict[str, Position]] = {
            TradingMode.SHORT_HOLD: {},
            TradingMode.LONG_HOLD: {},
        }

        # Track daily stats and performance history for win-rate calculations.
        self._daily_trades: Dict[TradingMode, int] = {
            TradingMode.SHORT_HOLD: 0,
            TradingMode.LONG_HOLD: 0,
        }
        self._daily_wins: Dict[TradingMode, int] = {
            TradingMode.SHORT_HOLD: 0,
            TradingMode.LONG_HOLD: 0,
        }
        self._daily_realized: Dict[TradingMode, float] = {
            TradingMode.SHORT_HOLD: 0.0,
            TradingMode.LONG_HOLD: 0.0,
        }
        self._realized_history: Dict[TradingMode, List[float]] = {
            TradingMode.SHORT_HOLD: [],
            TradingMode.LONG_HOLD: [],
        }

        logger.info(
            "dual_book_manager_initialized",
            total_capital=total_capital_gbp,
            max_short_heat=max_short_heat_pct,
            max_long_heat=max_long_heat_pct,
            reserve=reserve_heat_pct,
        )

    # ------------------------------------------------------------------
    # Core book helpers
    # ------------------------------------------------------------------

    def _book_positions(self, mode: TradingMode) -> Dict[str, Position]:
        return self._positions[mode]

    def _book_exposure(self, mode: TradingMode) -> float:
        return sum(pos.position_size_gbp for pos in self._book_positions(mode).values())

    def _total_exposure(self) -> float:
        return sum(self._book_exposure(mode) for mode in self._positions)

    def _available_capacity(self, mode: TradingMode) -> float:
        """Return remaining GBP capacity for a mode."""
        max_heat_gbp = self.max_heat_pct[mode] * self.total_capital_gbp
        return max_heat_gbp - self._book_exposure(mode)

    def _within_capacity(self, mode: TradingMode, additional_size_gbp: float) -> Tuple[bool, str]:
        """Capacity checks shared by open/add operations."""
        if additional_size_gbp <= 0:
            return False, "Size must be positive"

        # Per-book heat
        remaining = self._available_capacity(mode)
        if additional_size_gbp > remaining + 1e-9:
            max_heat_pct = self.max_heat_pct[mode] * 100
            return (
                False,
                f"Exceeds {mode.value} heat limit ({remaining:.2f} available, "
                f"max {max_heat_pct:.0f}% of capital)",
            )

        # Portfolio-level reserve
        total_after = self._total_exposure() + additional_size_gbp
        max_total = (1.0 - self.reserve_heat_pct) * self.total_capital_gbp
        if total_after > max_total + 1e-9:
            return (
                False,
                "Insufficient reserve heat remaining",
            )

        return True, "OK"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def can_open_position(
        self,
        symbol: str,
        mode: TradingMode,
        size_gbp: float,
        total_capital_gbp: Optional[float] = None,
    ) -> Tuple[bool, str]:
        """
        Check if a new position of the requested size can be opened.

        Args:
            symbol: Asset symbol (unused but reserved for per-asset policy).
            mode: Trading mode (short/long hold).
            size_gbp: Requested position size in GBP notionals.
            total_capital_gbp: Optional override for total capital.
        """
        if symbol in self._book_positions(mode):
            return False, f"{symbol} already has an open {mode.value} position"

        if total_capital_gbp is not None:
            self.total_capital_gbp = total_capital_gbp

        return self._within_capacity(mode, size_gbp)

    def open_position(
        self,
        symbol: str,
        mode: TradingMode,
        entry_price: float,
        size_gbp: float,
        stop_loss_bps: float,
        take_profit_bps: Optional[float] = None,
        entry_regime: str = "unknown",
        entry_confidence: float = 0.0,
        direction: str = "long",
    ) -> Optional[Position]:
        """Open a new position in the specified mode."""
        can_open, reason = self.can_open_position(
            symbol=symbol,
            mode=mode,
            size_gbp=size_gbp,
        )
        if not can_open:
            logger.warning(
                "open_position_blocked",
                symbol=symbol,
                mode=mode.value,
                reason=reason,
            )
            return None

        position = Position(
            symbol=symbol,
            mode=mode,
            entry_price=entry_price,
            entry_timestamp=_now(),
            position_size_gbp=size_gbp,
            stop_loss_bps=stop_loss_bps,
            take_profit_bps=take_profit_bps,
            direction=direction,
            entry_regime=entry_regime,
            entry_confidence=entry_confidence,
        )
        position.update_price(entry_price)

        self._book_positions(mode)[symbol] = position
        self._daily_trades[mode] += 1

        logger.info(
            "position_opened",
            symbol=symbol,
            mode=mode.value,
            size_gbp=size_gbp,
            entry_price=entry_price,
        )

        return position

    def has_position(self, symbol: str, mode: TradingMode) -> bool:
        """Return True if the book has an open position for symbol+mode."""
        return symbol in self._book_positions(mode)

    def get_position(self, symbol: str, mode: TradingMode) -> Optional[Position]:
        """Return the open position for symbol+mode if present."""
        return self._book_positions(mode).get(symbol)

    def update_position_price(
        self,
        symbol: str,
        mode: TradingMode,
        current_price: float,
    ) -> Optional[Position]:
        """
        Update mark price for a position.

        Returns the updated position or None if not present.
        """
        position = self.get_position(symbol, mode)
        if position is None:
            return None

        position.update_price(current_price)
        return position

    def close_position(
        self,
        symbol: str,
        mode: TradingMode,
        exit_price: float,
        reason: str = "manual",
    ) -> Tuple[Optional[Position], float]:
        """
        Close an entire position and return (position, realized_pnl_gbp).
        """
        position = self.get_position(symbol, mode)
        if position is None:
            logger.warning("close_position_not_found", symbol=symbol, mode=mode.value)
            return None, 0.0

        pnl = position.record_close(exit_price)
        del self._book_positions(mode)[symbol]

        self._daily_realized[mode] += pnl
        self._realized_history[mode].append(pnl)
        if pnl > 0:
            self._daily_wins[mode] += 1

        logger.info(
            "position_closed",
            symbol=symbol,
            mode=mode.value,
            exit_price=exit_price,
            pnl_gbp=pnl,
            reason=reason,
        )

        return position, pnl

    def add_to_position(
        self,
        symbol: str,
        mode: TradingMode,
        add_price: float,
        add_size_gbp: float,
    ) -> Optional[Position]:
        """
        Increase a position size (DCA). Returns the updated position.
        """
        position = self.get_position(symbol, mode)
        if position is None:
            logger.warning("add_position_missing", symbol=symbol, mode=mode.value)
            return None

        can_add, reason = self._within_capacity(mode, add_size_gbp)
        if not can_add:
            logger.warning("add_position_blocked", symbol=symbol, mode=mode.value, reason=reason)
            return None

        position.record_add(add_price, add_size_gbp)

        logger.info(
            "position_added",
            symbol=symbol,
            mode=mode.value,
            add_size_gbp=add_size_gbp,
            new_size_gbp=position.position_size_gbp,
        )

        return position

    def scale_out_position(
        self,
        symbol: str,
        mode: TradingMode,
        exit_price: float,
        scale_pct: float,
    ) -> Tuple[Optional[Position], float]:
        """
        Scale out of a position by percentage. Returns (position, realized_pnl).
        """
        position = self.get_position(symbol, mode)
        if position is None:
            logger.warning("scale_out_missing", symbol=symbol, mode=mode.value)
            return None, 0.0

        pnl = position.record_scale_out(exit_price, scale_pct)
        self._daily_realized[mode] += pnl
        self._realized_history[mode].append(pnl)
        if pnl > 0:
            self._daily_wins[mode] += 1

        # Auto-remove if fully scaled out
        if position.position_size_gbp <= 1e-6:
            del self._book_positions(mode)[symbol]

        logger.info(
            "position_scaled_out",
            symbol=symbol,
            mode=mode.value,
            scale_pct=scale_pct,
            pnl_gbp=pnl,
        )

        return position, pnl

    def activate_be_lock(self, symbol: str, mode: TradingMode) -> None:
        """Activate break-even protection on a position."""
        position = self.get_position(symbol, mode)
        if position:
            position.be_lock_active = True
            logger.info("be_lock_activated", symbol=symbol, mode=mode.value)

    def update_trail_level(
        self,
        symbol: str,
        mode: TradingMode,
        trail_level_bps: float,
    ) -> None:
        """Update trailing stop information for a position."""
        position = self.get_position(symbol, mode)
        if position:
            position.trail_active = True
            position.trail_level_bps = trail_level_bps
            logger.info(
                "trail_updated",
                symbol=symbol,
                mode=mode.value,
                trail_level_bps=trail_level_bps,
            )

    def get_exposure(self, symbol: str) -> Dict[str, float]:
        """Return exposure for a symbol across both books."""
        short_pos = self.get_position(symbol, TradingMode.SHORT_HOLD)
        long_pos = self.get_position(symbol, TradingMode.LONG_HOLD)
        short_gbp = short_pos.position_size_gbp if short_pos else 0.0
        long_gbp = long_pos.position_size_gbp if long_pos else 0.0
        return {
            "short_gbp": short_gbp,
            "long_gbp": long_gbp,
            "total_gbp": short_gbp + long_gbp,
        }

    def get_book_state(self, mode: TradingMode) -> BookState:
        """Return a summary of the requested book."""
        positions = list(self._book_positions(mode).values())
        total_exposure = sum(p.position_size_gbp for p in positions)
        unrealized = sum(p.unrealized_pnl_gbp for p in positions)
        heat_pct = (
            total_exposure / self.total_capital_gbp
            if self.total_capital_gbp > 0
            else 0.0
        )

        return BookState(
            mode=mode,
            positions=positions,
            num_positions=len(positions),
            total_exposure_gbp=total_exposure,
            heat_pct=heat_pct,
            num_trades_today=self._daily_trades[mode],
            wins_today=self._daily_wins[mode],
            realized_pnl_gbp=self._daily_realized[mode],
            unrealized_pnl_gbp=unrealized,
        )

    def get_book_heat(self, mode: TradingMode) -> float:
        """Return current book heat as a fraction of capital."""
        state = self.get_book_state(mode)
        return state.heat_pct

    def get_combined_stats(self) -> Dict[str, Dict[str, float]]:
        """Return aggregated statistics for both books."""
        short_state = self.get_book_state(TradingMode.SHORT_HOLD)
        long_state = self.get_book_state(TradingMode.LONG_HOLD)

        def _win_rate(mode: TradingMode) -> float:
            history = self._realized_history[mode]
            if not history:
                return 0.5
            wins = sum(1 for pnl in history if pnl > 0)
            return wins / len(history) if history else 0.5

        stats = {
            "short_hold": {
                "num_positions": short_state.num_positions,
                "total_exposure_gbp": short_state.total_exposure_gbp,
                "realized_pnl_gbp": self._daily_realized[TradingMode.SHORT_HOLD],
                "unrealized_pnl_gbp": short_state.unrealized_pnl_gbp,
                "win_rate": _win_rate(TradingMode.SHORT_HOLD),
            },
            "long_hold": {
                "num_positions": long_state.num_positions,
                "total_exposure_gbp": long_state.total_exposure_gbp,
                "realized_pnl_gbp": self._daily_realized[TradingMode.LONG_HOLD],
                "unrealized_pnl_gbp": long_state.unrealized_pnl_gbp,
                "win_rate": _win_rate(TradingMode.LONG_HOLD),
            },
        }

        stats["total"] = {
            "num_positions": short_state.num_positions + long_state.num_positions,
            "exposure_gbp": short_state.total_exposure_gbp + long_state.total_exposure_gbp,
            "realized_pnl_gbp": (
                self._daily_realized[TradingMode.SHORT_HOLD]
                + self._daily_realized[TradingMode.LONG_HOLD]
            ),
            "unrealized_pnl_gbp": (
                short_state.unrealized_pnl_gbp + long_state.unrealized_pnl_gbp
            ),
        }

        return stats

    def clear(self) -> None:
        """Reset all positions and stats (mainly for testing)."""
        for mode in self._positions:
            self._positions[mode].clear()
            self._daily_trades[mode] = 0
            self._daily_wins[mode] = 0
            self._daily_realized[mode] = 0.0
            self._realized_history[mode].clear()

        logger.info("dual_book_manager_cleared")


# Backwards compatibility alias for legacy imports
BookType = TradingMode
