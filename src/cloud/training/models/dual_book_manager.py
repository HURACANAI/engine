"""
Dual Position Book Manager

Manages two independent position books:
1. Short-Hold Book: Fast scalps (£1-£2 net targets)
2. Long-Hold Book: Swing positions (maximize gains, tolerate dips)

Each book:
- Has independent positions
- Tracks separate P&L
- Enforces separate risk caps
- Can hold same asset concurrently (with conflict resolution)
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import structlog

from .asset_profiles import TradingMode, TrailStyle

logger = structlog.get_logger(__name__)


@dataclass
class Position:
    """Represents a single position in a book."""

    symbol: str
    mode: TradingMode
    entry_price: float
    entry_timestamp: datetime
    position_size_gbp: float
    stop_loss_bps: float
    take_profit_bps: Optional[float] = None

    # Tracking
    unrealized_pnl_gbp: float = 0.0
    unrealized_pnl_bps: float = 0.0
    current_price: float = 0.0
    last_updated: Optional[datetime] = None

    # Long-hold specific
    add_count: int = 0  # Number of adds (DCA)
    scale_out_count: int = 0  # Number of partial exits
    be_lock_active: bool = False  # Break-even lock engaged
    trail_active: bool = False  # Trailing stop active
    trail_level_bps: float = 0.0  # Current trail level

    # Metadata
    entry_regime: str = "unknown"
    entry_technique: str = "unknown"
    entry_confidence: float = 0.0

    def age_minutes(self, current_time: datetime) -> int:
        """Get position age in minutes."""
        return int((current_time - self.entry_timestamp).total_seconds() / 60)

    def age_hours(self, current_time: datetime) -> float:
        """Get position age in hours."""
        return (current_time - self.entry_timestamp).total_seconds() / 3600


@dataclass
class BookState:
    """State of a position book."""

    mode: TradingMode
    positions: Dict[str, Position]  # symbol → position
    total_exposure_gbp: float = 0.0
    total_unrealized_pnl_gbp: float = 0.0
    num_positions: int = 0
    realized_pnl_gbp: float = 0.0  # Session P&L
    num_trades_today: int = 0
    wins_today: int = 0


class DualBookManager:
    """
    Manages dual position books for short-hold and long-hold modes.

    Responsibilities:
    1. Track positions in separate books
    2. Enforce per-book risk caps
    3. Handle adds (DCA) for long-hold
    4. Handle scale-outs for long-hold
    5. Resolve conflicts when both modes trade same asset
    6. Track per-mode P&L
    """

    def __init__(
        self,
        max_short_heat_pct: float = 0.20,  # Max 20% of capital in short-hold
        max_long_heat_pct: float = 0.50,   # Max 50% of capital in long-hold
        max_positions_per_book: int = 10,
    ):
        """
        Initialize dual book manager.

        Args:
            max_short_heat_pct: Max portfolio % for short-hold book
            max_long_heat_pct: Max portfolio % for long-hold book
            max_positions_per_book: Max concurrent positions per book
        """
        self.max_short_heat_pct = max_short_heat_pct
        self.max_long_heat_pct = max_long_heat_pct
        self.max_positions_per_book = max_positions_per_book

        # Initialize books
        self.book_short = BookState(mode=TradingMode.SHORT_HOLD, positions={})
        self.book_long = BookState(mode=TradingMode.LONG_HOLD, positions={})

        logger.info(
            "dual_book_manager_initialized",
            max_short_heat=max_short_heat_pct,
            max_long_heat=max_long_heat_pct,
            max_positions=max_positions_per_book,
        )

    def can_open_position(
        self,
        symbol: str,
        mode: TradingMode,
        size_gbp: float,
        total_capital_gbp: float,
    ) -> Tuple[bool, str]:
        """
        Check if we can open a position.

        Args:
            symbol: Asset symbol
            mode: Trading mode
            size_gbp: Proposed position size
            total_capital_gbp: Total portfolio capital

        Returns:
            (can_open, reason) tuple
        """
        book = self._get_book(mode)

        # Check if already have position in this book
        if symbol in book.positions:
            return (False, f"Already have {mode.value} position in {symbol}")

        # Check book capacity
        if len(book.positions) >= self.max_positions_per_book:
            return (False, f"{mode.value} book full ({self.max_positions_per_book} positions)")

        # Check heat limits
        new_exposure = book.total_exposure_gbp + size_gbp
        max_heat = total_capital_gbp * (
            self.max_short_heat_pct if mode == TradingMode.SHORT_HOLD
            else self.max_long_heat_pct
        )

        if new_exposure > max_heat:
            heat_pct = new_exposure / total_capital_gbp
            max_pct = self.max_short_heat_pct if mode == TradingMode.SHORT_HOLD else self.max_long_heat_pct
            return (
                False,
                f"{mode.value} heat {heat_pct:.1%} exceeds max {max_pct:.1%}"
            )

        return (True, "OK")

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
    ) -> Position:
        """
        Open a new position.

        Args:
            symbol: Asset symbol
            mode: Trading mode
            entry_price: Entry price
            size_gbp: Position size
            stop_loss_bps: Stop loss in bps
            take_profit_bps: Take profit in bps (optional)
            entry_regime: Market regime at entry
            entry_technique: Trading technique used
            entry_confidence: Entry confidence score

        Returns:
            Position object
        """
        book = self._get_book(mode)
        now = datetime.now()

        position = Position(
            symbol=symbol,
            mode=mode,
            entry_price=entry_price,
            entry_timestamp=now,
            position_size_gbp=size_gbp,
            stop_loss_bps=stop_loss_bps,
            take_profit_bps=take_profit_bps,
            current_price=entry_price,
            last_updated=now,
            entry_regime=entry_regime,
            entry_technique=entry_technique,
            entry_confidence=entry_confidence,
        )

        book.positions[symbol] = position
        book.total_exposure_gbp += size_gbp
        book.num_positions = len(book.positions)

        logger.info(
            "position_opened",
            symbol=symbol,
            mode=mode.value,
            entry_price=entry_price,
            size_gbp=size_gbp,
            stop_bps=stop_loss_bps,
            total_positions=book.num_positions,
        )

        return position

    def update_position_price(
        self,
        symbol: str,
        mode: TradingMode,
        current_price: float,
    ) -> Optional[Position]:
        """
        Update position with current price.

        Args:
            symbol: Asset symbol
            mode: Trading mode
            current_price: Current market price

        Returns:
            Updated position or None if not found
        """
        book = self._get_book(mode)
        position = book.positions.get(symbol)

        if not position:
            return None

        position.current_price = current_price
        position.last_updated = datetime.now()

        # Calculate P&L
        price_change_pct = (current_price - position.entry_price) / position.entry_price
        position.unrealized_pnl_gbp = price_change_pct * position.position_size_gbp
        position.unrealized_pnl_bps = price_change_pct * 10_000

        # Update book total
        book.total_unrealized_pnl_gbp = sum(
            pos.unrealized_pnl_gbp for pos in book.positions.values()
        )

        return position

    def add_to_position(
        self,
        symbol: str,
        mode: TradingMode,
        add_price: float,
        add_size_gbp: float,
    ) -> Optional[Position]:
        """
        Add to existing position (DCA for long-hold).

        Args:
            symbol: Asset symbol
            mode: Trading mode (must be LONG_HOLD)
            add_price: Add price
            add_size_gbp: Additional size

        Returns:
            Updated position or None if not found
        """
        if mode != TradingMode.LONG_HOLD:
            logger.warning("add_to_position_invalid_mode", symbol=symbol, mode=mode.value)
            return None

        book = self._get_book(mode)
        position = book.positions.get(symbol)

        if not position:
            logger.warning("add_to_position_not_found", symbol=symbol)
            return None

        # Calculate new average entry price
        total_size = position.position_size_gbp + add_size_gbp
        new_avg_price = (
            (position.entry_price * position.position_size_gbp) +
            (add_price * add_size_gbp)
        ) / total_size

        # Update position
        position.entry_price = new_avg_price
        position.position_size_gbp = total_size
        position.add_count += 1

        # Update book exposure
        book.total_exposure_gbp += add_size_gbp

        logger.info(
            "position_add_executed",
            symbol=symbol,
            add_price=add_price,
            add_size=add_size_gbp,
            new_avg_price=new_avg_price,
            total_size=total_size,
            add_count=position.add_count,
        )

        return position

    def scale_out_position(
        self,
        symbol: str,
        mode: TradingMode,
        exit_price: float,
        scale_pct: float,  # 0.0-1.0
    ) -> Tuple[Optional[Position], float]:
        """
        Scale out of position (partial exit for long-hold).

        Args:
            symbol: Asset symbol
            mode: Trading mode
            exit_price: Exit price
            scale_pct: Percentage to exit (0.0-1.0)

        Returns:
            (updated_position, realized_pnl_gbp) tuple
        """
        book = self._get_book(mode)
        position = book.positions.get(symbol)

        if not position:
            logger.warning("scale_out_position_not_found", symbol=symbol)
            return (None, 0.0)

        scale_pct = max(0.0, min(scale_pct, 1.0))
        exit_size = position.position_size_gbp * scale_pct

        # Calculate P&L for scaled portion
        price_change_pct = (exit_price - position.entry_price) / position.entry_price
        realized_pnl_gbp = price_change_pct * exit_size

        # Update position
        position.position_size_gbp -= exit_size
        position.scale_out_count += 1

        # Update book
        book.total_exposure_gbp -= exit_size
        book.realized_pnl_gbp += realized_pnl_gbp
        book.wins_today += 1 if realized_pnl_gbp > 0 else 0

        logger.info(
            "position_scaled_out",
            symbol=symbol,
            mode=mode.value,
            exit_price=exit_price,
            scale_pct=scale_pct,
            realized_pnl=realized_pnl_gbp,
            remaining_size=position.position_size_gbp,
        )

        return (position, realized_pnl_gbp)

    def close_position(
        self,
        symbol: str,
        mode: TradingMode,
        exit_price: float,
    ) -> Tuple[Optional[Position], float]:
        """
        Close position completely.

        Args:
            symbol: Asset symbol
            mode: Trading mode
            exit_price: Exit price

        Returns:
            (closed_position, realized_pnl_gbp) tuple
        """
        book = self._get_book(mode)
        position = book.positions.get(symbol)

        if not position:
            logger.warning("close_position_not_found", symbol=symbol, mode=mode.value)
            return (None, 0.0)

        # Calculate final P&L
        price_change_pct = (exit_price - position.entry_price) / position.entry_price
        realized_pnl_gbp = price_change_pct * position.position_size_gbp
        realized_pnl_bps = price_change_pct * 10_000

        # Update book
        book.total_exposure_gbp -= position.position_size_gbp
        book.realized_pnl_gbp += realized_pnl_gbp
        book.num_trades_today += 1
        book.wins_today += 1 if realized_pnl_gbp > 0 else 0

        # Remove position
        del book.positions[symbol]
        book.num_positions = len(book.positions)

        logger.info(
            "position_closed",
            symbol=symbol,
            mode=mode.value,
            exit_price=exit_price,
            entry_price=position.entry_price,
            size_gbp=position.position_size_gbp,
            pnl_gbp=realized_pnl_gbp,
            pnl_bps=realized_pnl_bps,
            hold_minutes=position.age_minutes(datetime.now()),
        )

        return (position, realized_pnl_gbp)

    def get_position(self, symbol: str, mode: TradingMode) -> Optional[Position]:
        """Get position from specified book."""
        book = self._get_book(mode)
        return book.positions.get(symbol)

    def has_position(self, symbol: str, mode: TradingMode) -> bool:
        """Check if position exists in specified book."""
        book = self._get_book(mode)
        return symbol in book.positions

    def get_all_positions(self, mode: Optional[TradingMode] = None) -> List[Position]:
        """
        Get all positions (optionally filtered by mode).

        Args:
            mode: Filter by mode (None = all positions)

        Returns:
            List of positions
        """
        if mode is None:
            return list(self.book_short.positions.values()) + \
                   list(self.book_long.positions.values())
        elif mode == TradingMode.SHORT_HOLD:
            return list(self.book_short.positions.values())
        else:
            return list(self.book_long.positions.values())

    def get_book_state(self, mode: TradingMode) -> BookState:
        """Get state of specified book."""
        return self._get_book(mode)

    def get_exposure(self, symbol: str) -> Dict[str, float]:
        """
        Get total exposure for a symbol across both books.

        Args:
            symbol: Asset symbol

        Returns:
            Dict with short_gbp, long_gbp, total_gbp
        """
        short_pos = self.book_short.positions.get(symbol)
        long_pos = self.book_long.positions.get(symbol)

        short_gbp = short_pos.position_size_gbp if short_pos else 0.0
        long_gbp = long_pos.position_size_gbp if long_pos else 0.0

        return {
            "short_gbp": short_gbp,
            "long_gbp": long_gbp,
            "total_gbp": short_gbp + long_gbp,
        }

    def update_trail_level(
        self,
        symbol: str,
        mode: TradingMode,
        trail_level_bps: float,
    ) -> Optional[Position]:
        """
        Update trailing stop level for position.

        Args:
            symbol: Asset symbol
            mode: Trading mode
            trail_level_bps: New trail level in bps

        Returns:
            Updated position or None
        """
        position = self.get_position(symbol, mode)
        if not position:
            return None

        position.trail_active = True
        position.trail_level_bps = trail_level_bps

        logger.debug(
            "trail_level_updated",
            symbol=symbol,
            mode=mode.value,
            trail_bps=trail_level_bps,
        )

        return position

    def activate_be_lock(
        self,
        symbol: str,
        mode: TradingMode,
    ) -> Optional[Position]:
        """
        Activate break-even lock for position.

        Args:
            symbol: Asset symbol
            mode: Trading mode

        Returns:
            Updated position or None
        """
        position = self.get_position(symbol, mode)
        if not position:
            return None

        position.be_lock_active = True

        logger.info(
            "be_lock_activated",
            symbol=symbol,
            mode=mode.value,
            unrealized_pnl_bps=position.unrealized_pnl_bps,
        )

        return position

    def get_combined_stats(self) -> Dict:
        """Get combined statistics across both books."""
        short_win_rate = (
            self.book_short.wins_today / self.book_short.num_trades_today
            if self.book_short.num_trades_today > 0 else 0.0
        )

        long_win_rate = (
            self.book_long.wins_today / self.book_long.num_trades_today
            if self.book_long.num_trades_today > 0 else 0.0
        )

        return {
            "short_hold": {
                "num_positions": self.book_short.num_positions,
                "exposure_gbp": self.book_short.total_exposure_gbp,
                "unrealized_pnl_gbp": self.book_short.total_unrealized_pnl_gbp,
                "realized_pnl_gbp": self.book_short.realized_pnl_gbp,
                "num_trades": self.book_short.num_trades_today,
                "win_rate": short_win_rate,
            },
            "long_hold": {
                "num_positions": self.book_long.num_positions,
                "exposure_gbp": self.book_long.total_exposure_gbp,
                "unrealized_pnl_gbp": self.book_long.total_unrealized_pnl_gbp,
                "realized_pnl_gbp": self.book_long.realized_pnl_gbp,
                "num_trades": self.book_long.num_trades_today,
                "win_rate": long_win_rate,
            },
            "total": {
                "num_positions": self.book_short.num_positions + self.book_long.num_positions,
                "exposure_gbp": self.book_short.total_exposure_gbp + self.book_long.total_exposure_gbp,
                "unrealized_pnl_gbp": self.book_short.total_unrealized_pnl_gbp + self.book_long.total_unrealized_pnl_gbp,
                "realized_pnl_gbp": self.book_short.realized_pnl_gbp + self.book_long.realized_pnl_gbp,
            },
        }

    def _get_book(self, mode: TradingMode) -> BookState:
        """Get book by mode."""
        if mode == TradingMode.SHORT_HOLD:
            return self.book_short
        else:
            return self.book_long
