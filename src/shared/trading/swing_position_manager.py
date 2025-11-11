"""
Swing Trading Position Manager

Manages swing trading positions with stop-loss, take-profit curves, and holding logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import structlog

from ..engines.enhanced_engine_interface import TradingHorizon, Direction

logger = structlog.get_logger(__name__)


class ExitReason(Enum):
    """Exit reason."""
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    TRAILING_STOP = "trailing_stop"
    TIME_LIMIT = "time_limit"
    MANUAL = "manual"
    REGIME_CHANGE = "regime_change"
    FUNDING_COST = "funding_cost"


@dataclass
class StopLossLevel:
    """Stop loss level."""
    price: float
    bps: float  # Stop loss in basis points from entry
    is_trailing: bool = False
    trailing_distance_bps: Optional[float] = None
    activated_at: Optional[datetime] = None


@dataclass
class TakeProfitLevel:
    """Take profit level."""
    price: float
    bps: float  # Take profit in basis points from entry
    exit_percentage: float  # Percentage of position to exit (0.0 to 1.0)
    is_activated: bool = False
    activated_at: Optional[datetime] = None


@dataclass
class SwingPosition:
    """Swing trading position."""
    symbol: str
    direction: Direction  # BUY or SELL
    entry_price: float
    entry_size: float  # Size in base currency
    entry_timestamp: datetime
    current_price: float
    current_size: float  # Remaining size after partial exits
    horizon_type: TradingHorizon
    # Stop loss and take profit
    stop_loss: Optional[StopLossLevel] = None
    take_profit_levels: List[TakeProfitLevel] = field(default_factory=list)
    trailing_stop: Optional[StopLossLevel] = None
    # Risk management
    max_holding_hours: Optional[float] = None
    max_funding_cost_bps: Optional[float] = None
    # State
    unrealized_pnl_bps: float = 0.0
    unrealized_pnl_usd: float = 0.0
    realized_pnl_bps: float = 0.0
    realized_pnl_usd: float = 0.0
    funding_cost_accumulated_bps: float = 0.0
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def update_price(self, current_price: float, funding_cost_bps: float = 0.0) -> None:
        """Update position with current price.
        
        Args:
            current_price: Current market price
            funding_cost_bps: Funding cost since last update in basis points
        """
        self.current_price = current_price
        self.last_updated = datetime.now(timezone.utc)
        
        # Calculate unrealized P&L
        if self.direction == Direction.BUY:
            price_change_bps = ((current_price - self.entry_price) / self.entry_price) * 10000.0
        else:  # SELL (short)
            price_change_bps = ((self.entry_price - current_price) / self.entry_price) * 10000.0
        
        self.unrealized_pnl_bps = price_change_bps
        self.unrealized_pnl_usd = (price_change_bps / 10000.0) * self.entry_size * self.entry_price
        
        # Accumulate funding cost
        self.funding_cost_accumulated_bps += funding_cost_bps
        
        # Update trailing stop if active
        if self.trailing_stop and self.trailing_stop.is_trailing:
            self._update_trailing_stop(current_price)
    
    def _update_trailing_stop(self, current_price: float) -> None:
        """Update trailing stop level.
        
        Args:
            current_price: Current market price
        """
        if not self.trailing_stop or not self.trailing_stop.is_trailing:
            return
        
        trailing_distance_bps = self.trailing_stop.trailing_distance_bps or 100.0  # Default 1%
        
        if self.direction == Direction.BUY:
            # For long positions, trail stop upward
            new_stop_price = current_price * (1 - trailing_distance_bps / 10000.0)
            if new_stop_price > self.trailing_stop.price:
                self.trailing_stop.price = new_stop_price
                self.trailing_stop.bps = ((self.entry_price - new_stop_price) / self.entry_price) * 10000.0
        else:  # SELL (short)
            # For short positions, trail stop downward
            new_stop_price = current_price * (1 + trailing_distance_bps / 10000.0)
            if new_stop_price < self.trailing_stop.price or self.trailing_stop.price == 0.0:
                self.trailing_stop.price = new_stop_price
                self.trailing_stop.bps = ((new_stop_price - self.entry_price) / self.entry_price) * 10000.0
    
    def get_holding_duration_hours(self) -> float:
        """Get holding duration in hours.
        
        Returns:
            Holding duration in hours
        """
        duration = self.last_updated - self.entry_timestamp
        return duration.total_seconds() / 3600.0
    
    def should_exit(
        self,
        current_price: float,
        current_regime: str,
        funding_cost_bps: float = 0.0,
    ) -> Tuple[bool, Optional[ExitReason], Optional[float]]:
        """Check if position should be exited.
        
        Args:
            current_price: Current market price
            current_regime: Current market regime
            funding_cost_bps: Funding cost since last update in basis points
            
        Returns:
            Tuple of (should_exit, exit_reason, exit_size_percentage)
        """
        # Update position
        self.update_price(current_price, funding_cost_bps)
        
        # Check stop loss
        if self.stop_loss:
            if self.direction == Direction.BUY and current_price <= self.stop_loss.price:
                return True, ExitReason.STOP_LOSS, 1.0  # Exit 100%
            elif self.direction == Direction.SELL and current_price >= self.stop_loss.price:
                return True, ExitReason.STOP_LOSS, 1.0  # Exit 100%
        
        # Check trailing stop
        if self.trailing_stop and self.trailing_stop.is_trailing:
            if self.direction == Direction.BUY and current_price <= self.trailing_stop.price:
                return True, ExitReason.TRAILING_STOP, 1.0  # Exit 100%
            elif self.direction == Direction.SELL and current_price >= self.trailing_stop.price:
                return True, ExitReason.TRAILING_STOP, 1.0  # Exit 100%
        
        # Check take profit levels
        for tp_level in self.take_profit_levels:
            if tp_level.is_activated:
                continue  # Already activated
            
            if self.direction == Direction.BUY and current_price >= tp_level.price:
                tp_level.is_activated = True
                tp_level.activated_at = datetime.now(timezone.utc)
                return True, ExitReason.TAKE_PROFIT, tp_level.exit_percentage
            elif self.direction == Direction.SELL and current_price <= tp_level.price:
                tp_level.is_activated = True
                tp_level.activated_at = datetime.now(timezone.utc)
                return True, ExitReason.TAKE_PROFIT, tp_level.exit_percentage
        
        # Check time limit
        if self.max_holding_hours:
            holding_hours = self.get_holding_duration_hours()
            if holding_hours >= self.max_holding_hours:
                return True, ExitReason.TIME_LIMIT, 1.0  # Exit 100%
        
        # Check funding cost limit
        if self.max_funding_cost_bps:
            if self.funding_cost_accumulated_bps >= self.max_funding_cost_bps:
                return True, ExitReason.FUNDING_COST, 1.0  # Exit 100%
        
        # Check regime change (panic regime forces exit for swing trades)
        if current_regime == "PANIC" and self.horizon_type in [TradingHorizon.SWING, TradingHorizon.POSITION]:
            return True, ExitReason.REGIME_CHANGE, 1.0  # Exit 100%
        
        return False, None, None
    
    def partial_exit(self, exit_percentage: float, exit_price: float) -> Dict[str, Any]:
        """Execute partial exit.
        
        Args:
            exit_percentage: Percentage of position to exit (0.0 to 1.0)
            exit_price: Exit price
            
        Returns:
            Exit details dictionary
        """
        exit_size = self.current_size * exit_percentage
        exit_value = exit_size * exit_price
        
        # Calculate realized P&L
        if self.direction == Direction.BUY:
            price_change_bps = ((exit_price - self.entry_price) / self.entry_price) * 10000.0
        else:  # SELL (short)
            price_change_bps = ((self.entry_price - exit_price) / self.entry_price) * 10000.0
        
        realized_pnl_bps = price_change_bps * exit_percentage
        realized_pnl_usd = (realized_pnl_bps / 10000.0) * self.entry_size * self.entry_price
        
        # Update position
        self.current_size -= exit_size
        self.realized_pnl_bps += realized_pnl_bps
        self.realized_pnl_usd += realized_pnl_usd
        
        # Update entry price for remaining position (weighted average)
        if self.current_size > 0:
            remaining_value = self.current_size * exit_price
            self.entry_price = (self.entry_price * self.current_size + exit_price * exit_size) / (self.current_size + exit_size)
        
        exit_details = {
            "exit_size": exit_size,
            "exit_price": exit_price,
            "exit_value": exit_value,
            "realized_pnl_bps": realized_pnl_bps,
            "realized_pnl_usd": realized_pnl_usd,
            "remaining_size": self.current_size,
        }
        
        logger.info(
            "partial_exit_executed",
            symbol=self.symbol,
            exit_percentage=exit_percentage,
            exit_price=exit_price,
            realized_pnl_bps=realized_pnl_bps,
            remaining_size=self.current_size,
        )
        
        return exit_details


@dataclass
class SwingPositionConfig:
    """Configuration for swing trading positions."""
    # Stop loss
    default_stop_loss_bps: float = 200.0  # Default 2% stop loss
    use_trailing_stop: bool = True
    trailing_stop_distance_bps: float = 100.0  # 1% trailing distance
    # Take profit
    take_profit_levels: List[Tuple[float, float]] = field(default_factory=lambda: [
        (200.0, 0.30),  # 30% at 2% profit
        (400.0, 0.40),  # 40% at 4% profit
        (600.0, 0.20),  # 20% at 6% profit
        # Remaining 10% trails
    ])
    # Risk management
    max_holding_hours: Optional[float] = None  # None = no time limit
    max_funding_cost_bps: Optional[float] = 500.0  # Max 5% funding cost
    # Regime gating
    exit_on_panic: bool = True  # Exit on panic regime
    exit_on_illiquid: bool = False  # Exit on illiquid regime


class SwingPositionManager:
    """Manages swing trading positions."""
    
    def __init__(self, config: SwingPositionConfig):
        """Initialize swing position manager.
        
        Args:
            config: Swing position configuration
        """
        self.config = config
        self.positions: Dict[str, SwingPosition] = {}  # symbol -> position
        logger.info("swing_position_manager_initialized")
    
    def open_position(
        self,
        symbol: str,
        direction: Direction,
        entry_price: float,
        entry_size: float,
        horizon_type: TradingHorizon,
        stop_loss_bps: Optional[float] = None,
        take_profit_levels: Optional[List[Tuple[float, float]]] = None,
        trailing_stop_bps: Optional[float] = None,
        max_holding_hours: Optional[float] = None,
        max_funding_cost_bps: Optional[float] = None,
    ) -> SwingPosition:
        """Open a swing trading position.
        
        Args:
            symbol: Trading symbol
            direction: Trade direction (BUY or SELL)
            entry_price: Entry price
            entry_size: Entry size in base currency
            horizon_type: Trading horizon type
            stop_loss_bps: Stop loss in basis points (uses default if None)
            take_profit_levels: Take profit levels [(profit_bps, exit_pct), ...]
            trailing_stop_bps: Trailing stop distance in basis points
            max_holding_hours: Maximum holding time in hours
            max_funding_cost_bps: Maximum funding cost in basis points
            
        Returns:
            Swing position instance
        """
        if symbol in self.positions:
            logger.warning("position_already_exists", symbol=symbol, message="Closing existing position first")
            self.close_position(symbol, "replaced")
        
        # Set defaults
        stop_loss_bps = stop_loss_bps or self.config.default_stop_loss_bps
        take_profit_levels = take_profit_levels or self.config.take_profit_levels
        trailing_stop_bps = trailing_stop_bps or (self.config.trailing_stop_distance_bps if self.config.use_trailing_stop else None)
        max_holding_hours = max_holding_hours or self.config.max_holding_hours
        max_funding_cost_bps = max_funding_cost_bps or self.config.max_funding_cost_bps
        
        # Calculate stop loss price
        if direction == Direction.BUY:
            stop_loss_price = entry_price * (1 - stop_loss_bps / 10000.0)
        else:  # SELL (short)
            stop_loss_price = entry_price * (1 + stop_loss_bps / 10000.0)
        
        stop_loss = StopLossLevel(
            price=stop_loss_price,
            bps=stop_loss_bps,
            is_trailing=False,
        )
        
        # Calculate take profit levels
        take_profit_levels_list = []
        for profit_bps, exit_pct in take_profit_levels:
            if direction == Direction.BUY:
                tp_price = entry_price * (1 + profit_bps / 10000.0)
            else:  # SELL (short)
                tp_price = entry_price * (1 - profit_bps / 10000.0)
            
            take_profit_levels_list.append(
                TakeProfitLevel(
                    price=tp_price,
                    bps=profit_bps,
                    exit_percentage=exit_pct,
                    is_activated=False,
                )
            )
        
        # Create trailing stop
        trailing_stop = None
        if trailing_stop_bps:
            trailing_stop = StopLossLevel(
                price=stop_loss_price,  # Start at stop loss price
                bps=stop_loss_bps,
                is_trailing=True,
                trailing_distance_bps=trailing_stop_bps,
            )
        
        # Create position
        position = SwingPosition(
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            entry_size=entry_size,
            entry_timestamp=datetime.now(timezone.utc),
            current_price=entry_price,
            current_size=entry_size,
            horizon_type=horizon_type,
            stop_loss=stop_loss,
            take_profit_levels=take_profit_levels_list,
            trailing_stop=trailing_stop,
            max_holding_hours=max_holding_hours,
            max_funding_cost_bps=max_funding_cost_bps,
        )
        
        self.positions[symbol] = position
        
        logger.info(
            "swing_position_opened",
            symbol=symbol,
            direction=direction.value,
            entry_price=entry_price,
            entry_size=entry_size,
            horizon_type=horizon_type.value,
            stop_loss_bps=stop_loss_bps,
        )
        
        return position
    
    def update_position(
        self,
        symbol: str,
        current_price: float,
        current_regime: str,
        funding_cost_bps: float = 0.0,
    ) -> Optional[Dict[str, Any]]:
        """Update position and check for exits.
        
        Args:
            symbol: Trading symbol
            current_price: Current market price
            current_regime: Current market regime
            funding_cost_bps: Funding cost since last update in basis points
            
        Returns:
            Exit action dictionary if exit is triggered, None otherwise
        """
        if symbol not in self.positions:
            return None
        
        position = self.positions[symbol]
        
        # Check if should exit
        should_exit, exit_reason, exit_percentage = position.should_exit(
            current_price=current_price,
            current_regime=current_regime,
            funding_cost_bps=funding_cost_bps,
        )
        
        if should_exit and exit_percentage:
            # Execute exit
            if exit_percentage >= 1.0:
                # Full exit
                return self.close_position(symbol, exit_reason.value if exit_reason else "unknown")
            else:
                # Partial exit
                exit_details = position.partial_exit(exit_percentage, current_price)
                exit_details["exit_reason"] = exit_reason.value if exit_reason else "unknown"
                exit_details["symbol"] = symbol
                return exit_details
        
        return None
    
    def close_position(self, symbol: str, reason: str = "manual") -> Dict[str, Any]:
        """Close a position.
        
        Args:
            symbol: Trading symbol
            reason: Exit reason
            
        Returns:
            Exit details dictionary
        """
        if symbol not in self.positions:
            logger.warning("position_not_found", symbol=symbol)
            return {}
        
        position = self.positions[symbol]
        
        # Calculate final P&L
        exit_details = {
            "symbol": symbol,
            "exit_reason": reason,
            "entry_price": position.entry_price,
            "exit_price": position.current_price,
            "entry_size": position.entry_size,
            "exit_size": position.current_size,
            "realized_pnl_bps": position.realized_pnl_bps,
            "realized_pnl_usd": position.realized_pnl_usd,
            "unrealized_pnl_bps": position.unrealized_pnl_bps,
            "unrealized_pnl_usd": position.unrealized_pnl_usd,
            "funding_cost_bps": position.funding_cost_accumulated_bps,
            "holding_hours": position.get_holding_duration_hours(),
        }
        
        # Remove position
        del self.positions[symbol]
        
        logger.info(
            "swing_position_closed",
            symbol=symbol,
            reason=reason,
            realized_pnl_bps=position.realized_pnl_bps,
            holding_hours=position.get_holding_duration_hours(),
        )
        
        return exit_details
    
    def get_position(self, symbol: str) -> Optional[SwingPosition]:
        """Get position for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Swing position, or None if not found
        """
        return self.positions.get(symbol)
    
    def get_all_positions(self) -> List[SwingPosition]:
        """Get all positions.
        
        Returns:
            List of all positions
        """
        return list(self.positions.values())
    
    def has_position(self, symbol: str) -> bool:
        """Check if position exists for symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            True if position exists
        """
        return symbol in self.positions

