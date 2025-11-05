"""
Dual-Book Position Manager - Separates scalps from runners.

Key Architecture:
- SHORT_HOLD book: Fast scalps (£1-£2 target, 5-15 sec hold, 70-75% WR)
- LONG_HOLD book: Runners (£5-£20 target, 5-60 min hold, 95%+ WR)

Each book has:
- Independent position tracking
- Separate heat/risk caps
- Different exit ladders
- Per-asset profiles

Why This Matters:
Scalps and runners have different:
- Target sizes (scalp wants volume, runner wants precision)
- Hold times (seconds vs minutes)
- Risk profiles (many small vs few large)
- Gate requirements (loose vs strict)

Mixing them in one book creates conflicts. Separation allows optimization of each.

Usage:
    manager = DualBookManager(
        total_capital=10000.0,
        max_short_heat=0.40,  # 40% in scalps
        max_long_heat=0.50,   # 50% in runners
    )

    # Define asset profiles
    manager.set_asset_profile('ETH-USD', AssetProfile(
        allowed_books=[BookType.SHORT_HOLD, BookType.LONG_HOLD],
        scalp_target_bps=100,  # £1 on £100 = 100 bps
        runner_target_bps=800,  # £8 on £100 = 800 bps
    ))

    # Add position
    manager.add_position(
        symbol='ETH-USD',
        book=BookType.SHORT_HOLD,
        entry_price=2000.0,
        size=0.05,  # £100
        direction='long',
    )

    # Check heat
    scalp_heat = manager.get_book_heat(BookType.SHORT_HOLD)  # 0.10 (10%)
    can_add = manager.can_add_position(BookType.SHORT_HOLD, size_usd=200.0)  # True

    # Exit
    manager.close_position(position_id, exit_price=2020.0)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set
import time
import uuid

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


class BookType(Enum):
    """Position book types."""

    SHORT_HOLD = "short_hold"  # Scalps: seconds to minutes
    LONG_HOLD = "long_hold"  # Runners: minutes to hours


@dataclass
class AssetProfile:
    """Per-asset trading profile."""

    # Which books can trade this asset
    allowed_books: List[BookType]

    # Target profit levels (bps)
    scalp_target_bps: float = 100.0  # £1 on £100
    runner_target_bps: float = 800.0  # £8 on £100

    # Hold time expectations (seconds)
    scalp_hold_sec: float = 10.0  # 5-15 sec
    runner_hold_sec: float = 600.0  # 5-60 min

    # Max position size per trade (USD)
    scalp_max_size: float = 200.0
    runner_max_size: float = 1000.0

    # Liquidity requirements
    min_liquidity_score: float = 0.60

    # Regime preferences
    scalp_regimes: Set[str] = field(default_factory=lambda: {'TREND', 'RANGE', 'PANIC'})
    runner_regimes: Set[str] = field(default_factory=lambda: {'TREND'})


@dataclass
class Position:
    """A single position."""

    position_id: str
    symbol: str
    book: BookType
    direction: str  # 'long' or 'short'

    # Entry
    entry_price: float
    size: float  # In asset units (e.g., 0.05 ETH)
    size_usd: float  # In USD
    entry_time: float

    # Current state
    current_price: float
    unrealized_pnl: float = 0.0

    # Exit tracking
    exit_price: Optional[float] = None
    exit_time: Optional[float] = None
    realized_pnl: Optional[float] = None
    closed: bool = False

    # Metadata
    technique: Optional[str] = None  # Which engine opened this
    regime_at_entry: Optional[str] = None
    confidence_at_entry: Optional[float] = None


@dataclass
class BookMetrics:
    """Metrics for a single book."""

    total_positions: int
    open_positions: int
    closed_positions: int

    # Heat (capital allocation)
    current_heat: float  # % of total capital
    peak_heat: float  # Max heat reached

    # P&L
    total_realized_pnl: float
    total_unrealized_pnl: float
    largest_win: float
    largest_loss: float

    # Performance
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float


class DualBookManager:
    """
    Dual-book position manager.

    Manages two independent books:
    - SHORT_HOLD: High-frequency scalps (£1-£2, 70-75% WR)
    - LONG_HOLD: Precision runners (£5-£20, 95%+ WR)

    Key Features:
    1. Independent heat caps per book
    2. Per-asset profiles (which books allowed)
    3. Separate P&L tracking
    4. Book-specific exit ladders

    Architecture:
        Total Capital: £10,000
        ├── SHORT_HOLD Book (max 40% = £4,000)
        │   ├── ETH-USD: £100 (scalp)
        │   ├── SOL-USD: £150 (scalp)
        │   └── BTC-USD: £200 (scalp)
        └── LONG_HOLD Book (max 50% = £5,000)
            ├── ETH-USD: £800 (runner)
            └── BTC-USD: £1,200 (runner)
    """

    def __init__(
        self,
        total_capital: float = 10000.0,
        max_short_heat: float = 0.40,  # Max 40% in scalps
        max_long_heat: float = 0.50,  # Max 50% in runners
        reserve_heat: float = 0.10,  # Keep 10% reserve
    ):
        """
        Initialize dual-book manager.

        Args:
            total_capital: Total trading capital in USD
            max_short_heat: Max % of capital in SHORT_HOLD book
            max_long_heat: Max % of capital in LONG_HOLD book
            reserve_heat: Reserve % (no trading)
        """
        self.total_capital = total_capital
        self.max_short_heat = max_short_heat
        self.max_long_heat = max_long_heat
        self.reserve_heat = reserve_heat

        # Position books
        self.book_short: Dict[str, Position] = {}
        self.book_long: Dict[str, Position] = {}

        # Asset profiles
        self.asset_profiles: Dict[str, AssetProfile] = {}

        # Statistics
        self.total_trades = 0
        self.trades_by_book: Dict[BookType, int] = {
            BookType.SHORT_HOLD: 0,
            BookType.LONG_HOLD: 0,
        }

        logger.info(
            "dual_book_manager_initialized",
            total_capital=total_capital,
            max_short_heat=max_short_heat,
            max_long_heat=max_long_heat,
        )

    def set_asset_profile(
        self,
        symbol: str,
        profile: AssetProfile,
    ) -> None:
        """
        Set trading profile for an asset.

        Args:
            symbol: Asset symbol (e.g., 'ETH-USD')
            profile: Asset profile configuration
        """
        self.asset_profiles[symbol] = profile

        logger.info(
            "asset_profile_set",
            symbol=symbol,
            allowed_books=[b.value for b in profile.allowed_books],
        )

    def can_add_position(
        self,
        book: BookType,
        size_usd: float,
        symbol: Optional[str] = None,
    ) -> tuple[bool, str]:
        """
        Check if we can add a position to a book.

        Args:
            book: Which book to check
            size_usd: Position size in USD
            symbol: Asset symbol (optional, for profile check)

        Returns:
            (can_add, reason)
        """
        # Check book-specific heat limit
        current_heat = self.get_book_heat(book)
        max_heat = self.max_short_heat if book == BookType.SHORT_HOLD else self.max_long_heat

        new_heat = current_heat + (size_usd / self.total_capital)

        if new_heat > max_heat:
            return False, f"Would exceed {book.value} heat limit: {new_heat:.2%} > {max_heat:.2%}"

        # Check asset profile (if symbol provided)
        if symbol and symbol in self.asset_profiles:
            profile = self.asset_profiles[symbol]

            if book not in profile.allowed_books:
                return False, f"{symbol} not allowed in {book.value} book"

            max_size = (
                profile.scalp_max_size
                if book == BookType.SHORT_HOLD
                else profile.runner_max_size
            )

            if size_usd > max_size:
                return False, f"Size ${size_usd:.0f} exceeds {book.value} max ${max_size:.0f}"

        return True, "OK"

    def add_position(
        self,
        symbol: str,
        book: BookType,
        entry_price: float,
        size: float,
        direction: str,
        technique: Optional[str] = None,
        regime: Optional[str] = None,
        confidence: Optional[float] = None,
    ) -> Optional[Position]:
        """
        Add a new position to a book.

        Args:
            symbol: Asset symbol
            book: Which book (SHORT_HOLD or LONG_HOLD)
            entry_price: Entry price
            size: Position size in asset units
            direction: 'long' or 'short'
            technique: Trading technique used
            regime: Market regime at entry
            confidence: Engine confidence at entry

        Returns:
            Position if added, None if blocked
        """
        size_usd = size * entry_price

        # Check if can add
        can_add, reason = self.can_add_position(book, size_usd, symbol)
        if not can_add:
            logger.warning(
                "position_blocked",
                symbol=symbol,
                book=book.value,
                reason=reason,
            )
            return None

        # Create position
        position = Position(
            position_id=str(uuid.uuid4()),
            symbol=symbol,
            book=book,
            direction=direction,
            entry_price=entry_price,
            size=size,
            size_usd=size_usd,
            entry_time=time.time(),
            current_price=entry_price,
            technique=technique,
            regime_at_entry=regime,
            confidence_at_entry=confidence,
        )

        # Add to appropriate book
        if book == BookType.SHORT_HOLD:
            self.book_short[position.position_id] = position
        else:
            self.book_long[position.position_id] = position

        self.total_trades += 1
        self.trades_by_book[book] += 1

        logger.info(
            "position_opened",
            position_id=position.position_id[:8],
            symbol=symbol,
            book=book.value,
            direction=direction,
            size_usd=size_usd,
            technique=technique,
        )

        return position

    def update_position_price(
        self,
        position_id: str,
        current_price: float,
    ) -> Optional[Position]:
        """
        Update position with current price and recalculate P&L.

        Args:
            position_id: Position ID
            current_price: Current market price

        Returns:
            Updated position, or None if not found
        """
        position = self._get_position(position_id)
        if not position:
            return None

        position.current_price = current_price

        # Calculate unrealized P&L
        if position.direction == 'long':
            pnl = (current_price - position.entry_price) * position.size
        else:  # short
            pnl = (position.entry_price - current_price) * position.size

        position.unrealized_pnl = pnl

        return position

    def close_position(
        self,
        position_id: str,
        exit_price: float,
        reason: str = "manual",
    ) -> Optional[float]:
        """
        Close a position and realize P&L.

        Args:
            position_id: Position ID
            exit_price: Exit price
            reason: Reason for exit

        Returns:
            Realized P&L in USD, or None if position not found
        """
        position = self._get_position(position_id)
        if not position:
            logger.warning("position_not_found", position_id=position_id)
            return None

        if position.closed:
            logger.warning("position_already_closed", position_id=position_id)
            return position.realized_pnl

        # Calculate realized P&L
        if position.direction == 'long':
            pnl = (exit_price - position.entry_price) * position.size
        else:  # short
            pnl = (position.entry_price - exit_price) * position.size

        position.exit_price = exit_price
        position.exit_time = time.time()
        position.realized_pnl = pnl
        position.closed = True

        hold_time_sec = position.exit_time - position.entry_time

        logger.info(
            "position_closed",
            position_id=position.position_id[:8],
            symbol=position.symbol,
            book=position.book.value,
            pnl=pnl,
            hold_time_sec=hold_time_sec,
            reason=reason,
        )

        return pnl

    def get_book_heat(self, book: BookType) -> float:
        """
        Get current heat (capital allocation) for a book.

        Args:
            book: Which book

        Returns:
            Heat as % of total capital (0.0 to 1.0)
        """
        positions = self._get_book_positions(book, open_only=True)

        total_allocated = sum(p.size_usd for p in positions)

        heat = total_allocated / self.total_capital

        return heat

    def get_book_metrics(self, book: BookType) -> BookMetrics:
        """
        Get metrics for a book.

        Args:
            book: Which book

        Returns:
            BookMetrics
        """
        all_positions = self._get_book_positions(book, open_only=False)
        open_positions = [p for p in all_positions if not p.closed]
        closed_positions = [p for p in all_positions if p.closed]

        # P&L
        total_realized = sum(p.realized_pnl for p in closed_positions if p.realized_pnl)
        total_unrealized = sum(p.unrealized_pnl for p in open_positions)

        # Win/loss tracking
        winners = [p for p in closed_positions if p.realized_pnl and p.realized_pnl > 0]
        losers = [p for p in closed_positions if p.realized_pnl and p.realized_pnl < 0]

        win_rate = len(winners) / len(closed_positions) if closed_positions else 0.0
        avg_win = np.mean([p.realized_pnl for p in winners]) if winners else 0.0
        avg_loss = np.mean([abs(p.realized_pnl) for p in losers]) if losers else 0.0

        total_win_amount = sum(p.realized_pnl for p in winners)
        total_loss_amount = sum(abs(p.realized_pnl) for p in losers)

        profit_factor = (
            total_win_amount / total_loss_amount if total_loss_amount > 0 else 0.0
        )

        largest_win = max((p.realized_pnl for p in winners), default=0.0)
        largest_loss = min((p.realized_pnl for p in losers), default=0.0)

        # Heat tracking
        current_heat = self.get_book_heat(book)
        # TODO: Track peak heat over time
        peak_heat = current_heat  # Simplified

        return BookMetrics(
            total_positions=len(all_positions),
            open_positions=len(open_positions),
            closed_positions=len(closed_positions),
            current_heat=current_heat,
            peak_heat=peak_heat,
            total_realized_pnl=total_realized,
            total_unrealized_pnl=total_unrealized,
            largest_win=largest_win,
            largest_loss=largest_loss,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
        )

    def get_combined_metrics(self) -> Dict[str, BookMetrics]:
        """
        Get metrics for both books.

        Returns:
            Dict mapping book type to metrics
        """
        return {
            BookType.SHORT_HOLD.value: self.get_book_metrics(BookType.SHORT_HOLD),
            BookType.LONG_HOLD.value: self.get_book_metrics(BookType.LONG_HOLD),
        }

    def _get_position(self, position_id: str) -> Optional[Position]:
        """Get position by ID from either book."""
        if position_id in self.book_short:
            return self.book_short[position_id]
        elif position_id in self.book_long:
            return self.book_long[position_id]
        else:
            return None

    def _get_book_positions(
        self,
        book: BookType,
        open_only: bool = False,
    ) -> List[Position]:
        """Get all positions in a book."""
        if book == BookType.SHORT_HOLD:
            positions = list(self.book_short.values())
        else:
            positions = list(self.book_long.values())

        if open_only:
            positions = [p for p in positions if not p.closed]

        return positions

    def get_summary(self) -> Dict:
        """Get overall summary across both books."""
        short_metrics = self.get_book_metrics(BookType.SHORT_HOLD)
        long_metrics = self.get_book_metrics(BookType.LONG_HOLD)

        total_realized = short_metrics.total_realized_pnl + long_metrics.total_realized_pnl
        total_unrealized = (
            short_metrics.total_unrealized_pnl + long_metrics.total_unrealized_pnl
        )

        total_heat = short_metrics.current_heat + long_metrics.current_heat

        return {
            'total_capital': self.total_capital,
            'total_heat': total_heat,
            'available_heat': 1.0 - total_heat - self.reserve_heat,
            'short_book': {
                'positions': short_metrics.open_positions,
                'heat': short_metrics.current_heat,
                'pnl': short_metrics.total_realized_pnl,
                'win_rate': short_metrics.win_rate,
            },
            'long_book': {
                'positions': long_metrics.open_positions,
                'heat': long_metrics.current_heat,
                'pnl': long_metrics.total_realized_pnl,
                'win_rate': long_metrics.win_rate,
            },
            'combined': {
                'total_positions': short_metrics.open_positions
                + long_metrics.open_positions,
                'total_realized_pnl': total_realized,
                'total_unrealized_pnl': total_unrealized,
                'total_trades': self.total_trades,
            },
        }
