"""
Spread Threshold + Auto-Cancel Logic

Manages orders with spread thresholds and automatic cancellation logic.
Monitors order status and cancels if spread moves beyond threshold.

Key Features:
- Spread threshold monitoring
- Automatic order cancellation
- Order status tracking
- Spread-based order management
- Time-based cancellation

Author: Huracan Engine Team
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Callable
import structlog

logger = structlog.get_logger(__name__)


class OrderStatus(Enum):
    """Order status."""
    PENDING = "pending"
    ACTIVE = "active"
    FILLED = "filled"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


@dataclass
class Order:
    """Order with spread threshold."""
    order_id: str
    symbol: str
    side: str  # "buy" or "sell"
    order_type: str  # "limit", "market", etc.
    price: float
    size: float
    spread_threshold_bps: float  # Max spread before cancel
    created_at: datetime
    expires_at: Optional[datetime]
    status: OrderStatus
    exchange: str
    metadata: Dict[str, any]


@dataclass
class SpreadSnapshot:
    """Current spread snapshot."""
    symbol: str
    timestamp: datetime
    best_bid: float
    best_ask: float
    spread_bps: float
    mid_price: float


class SpreadThresholdManager:
    """
    Manages orders with spread thresholds and auto-cancel logic.
    
    Usage:
        manager = SpreadThresholdManager(
            max_spread_bps=10.0,
            auto_cancel=True
        )
        
        # Place order with spread threshold
        order = manager.place_order(
            order_id="order_123",
            symbol="BTC/USDT",
            side="buy",
            price=50000.0,
            size=1.0,
            spread_threshold_bps=5.0
        )
        
        # Monitor and auto-cancel if spread exceeds threshold
        manager.monitor_orders()
        
        # Check if order should be cancelled
        should_cancel = manager.should_cancel_order(order, current_spread_bps=12.0)
    """
    
    def __init__(
        self,
        max_spread_bps: float = 10.0,
        auto_cancel: bool = True,
        check_interval_seconds: int = 1,
        max_order_age_seconds: Optional[int] = None
    ):
        """
        Initialize spread threshold manager.
        
        Args:
            max_spread_bps: Default maximum spread in basis points
            auto_cancel: Whether to auto-cancel orders
            check_interval_seconds: How often to check orders
            max_order_age_seconds: Maximum order age before auto-cancel (None = no limit)
        """
        self.max_spread_bps = max_spread_bps
        self.auto_cancel = auto_cancel
        self.check_interval_seconds = check_interval_seconds
        self.max_order_age_seconds = max_order_age_seconds
        
        # Order storage (with max size limit to prevent unbounded growth)
        self.orders: Dict[str, Order] = {}
        self.max_orders: int = 10000  # Maximum orders to keep in memory
        
        # Spread snapshots
        self.spread_snapshots: Dict[str, SpreadSnapshot] = {}
        
        # Cancel callback (called when order is cancelled)
        self.cancel_callback: Optional[Callable[[Order, str], None]] = None
        
        logger.info(
            "spread_threshold_manager_initialized",
            max_spread_bps=max_spread_bps,
            auto_cancel=auto_cancel
        )
    
    def place_order(
        self,
        order_id: str,
        symbol: str,
        side: str,
        price: float,
        size: float,
        order_type: str = "limit",
        spread_threshold_bps: Optional[float] = None,
        expires_at: Optional[datetime] = None,
        exchange: str = "unknown",
        metadata: Optional[Dict[str, any]] = None
    ) -> Order:
        """
        Place an order with spread threshold.
        
        Args:
            order_id: Unique order identifier
            symbol: Trading symbol
            side: "buy" or "sell"
            price: Order price
            size: Order size
            order_type: Order type
            spread_threshold_bps: Spread threshold (uses default if None)
            expires_at: Order expiration time
            exchange: Exchange name
            metadata: Additional order metadata
        
        Returns:
            Order object
        """
        if spread_threshold_bps is None:
            spread_threshold_bps = self.max_spread_bps
        
        order = Order(
            order_id=order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            price=price,
            size=size,
            spread_threshold_bps=spread_threshold_bps,
            created_at=datetime.now(),
            expires_at=expires_at,
            status=OrderStatus.PENDING,
            exchange=exchange,
            metadata=metadata or {}
        )
        
        self.orders[order_id] = order
        
        # Cleanup old orders if we exceed max size
        if len(self.orders) > self.max_orders:
            self._cleanup_old_orders()
        
        logger.info(
            "order_placed",
            order_id=order_id,
            symbol=symbol,
            side=side,
            price=price,
            spread_threshold_bps=spread_threshold_bps
        )
        
        return order
    
    def update_spread(
        self,
        symbol: str,
        best_bid: float,
        best_ask: float
    ) -> None:
        """
        Update spread snapshot for a symbol.
        
        Args:
            symbol: Trading symbol
            best_bid: Best bid price
            best_ask: Best ask price
        """
        spread_bps = ((best_ask - best_bid) / best_bid * 10000) if best_bid > 0 else 0.0
        mid_price = (best_bid + best_ask) / 2.0
        
        snapshot = SpreadSnapshot(
            symbol=symbol,
            timestamp=datetime.now(),
            best_bid=best_bid,
            best_ask=best_ask,
            spread_bps=spread_bps,
            mid_price=mid_price
        )
        
        self.spread_snapshots[symbol] = snapshot
        
        logger.debug(
            "spread_updated",
            symbol=symbol,
            spread_bps=spread_bps,
            best_bid=best_bid,
            best_ask=best_ask
        )
    
    def should_cancel_order(
        self,
        order: Order,
        current_spread_bps: Optional[float] = None
    ) -> Tuple[bool, str]:
        """
        Check if order should be cancelled.
        
        Args:
            order: Order to check
            current_spread_bps: Current spread (uses snapshot if None)
        
        Returns:
            Tuple of (should_cancel, reason)
        """
        # Check if already filled or cancelled
        if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.EXPIRED]:
            return False, "order_already_terminated"
        
        # Check expiration
        if order.expires_at and datetime.now() > order.expires_at:
            return True, "order_expired"
        
        # Check max age
        if self.max_order_age_seconds:
            age_seconds = (datetime.now() - order.created_at).total_seconds()
            if age_seconds > self.max_order_age_seconds:
                return True, "order_too_old"
        
        # Check spread threshold
        if current_spread_bps is None:
            snapshot = self.spread_snapshots.get(order.symbol)
            if snapshot:
                current_spread_bps = snapshot.spread_bps
            else:
                return False, "no_spread_data"
        
        if current_spread_bps > order.spread_threshold_bps:
            return True, f"spread_exceeded_threshold_{current_spread_bps:.1f}_bps"
        
        return False, "ok"
    
    def monitor_orders(self) -> List[Order]:
        """
        Monitor all active orders and cancel if needed.
        
        Returns:
            List of cancelled orders
        """
        cancelled_orders = []
        
        for order_id, order in list(self.orders.items()):
            should_cancel, reason = self.should_cancel_order(order)
            
            if should_cancel and self.auto_cancel:
                self.cancel_order(order_id, reason)
                cancelled_orders.append(order)
        
        if cancelled_orders:
            logger.info(
                "orders_cancelled",
                count=len(cancelled_orders),
                reasons=[o.metadata.get("cancel_reason", "unknown") for o in cancelled_orders]
            )
        
        return cancelled_orders
    
    def cancel_order(self, order_id: str, reason: str = "manual") -> bool:
        """
        Cancel an order.
        
        Args:
            order_id: Order identifier
            reason: Cancellation reason
        
        Returns:
            True if cancelled, False if not found
        """
        if order_id not in self.orders:
            return False
        
        order = self.orders[order_id]
        
        if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED]:
            return False
        
        order.status = OrderStatus.CANCELLED
        order.metadata["cancel_reason"] = reason
        order.metadata["cancelled_at"] = datetime.now()
        
        # Call cancel callback if set
        if self.cancel_callback:
            try:
                self.cancel_callback(order, reason)
            except Exception as e:
                logger.error("cancel_callback_failed", error=str(e))
        
        logger.info(
            "order_cancelled",
            order_id=order_id,
            symbol=order.symbol,
            reason=reason
        )
        
        return True
    
    def fill_order(self, order_id: str) -> bool:
        """
        Mark order as filled.
        
        Args:
            order_id: Order identifier
        
        Returns:
            True if filled, False if not found
        """
        if order_id not in self.orders:
            return False
        
        order = self.orders[order_id]
        order.status = OrderStatus.FILLED
        order.metadata["filled_at"] = datetime.now()
        
        logger.info("order_filled", order_id=order_id, symbol=order.symbol)
        
        return True
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID."""
        return self.orders.get(order_id)
    
    def get_active_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get all active orders."""
        active = [
            order for order in self.orders.values()
            if order.status == OrderStatus.ACTIVE or order.status == OrderStatus.PENDING
        ]
        
        if symbol:
            active = [o for o in active if o.symbol == symbol]
        
        return active
    
    def set_cancel_callback(self, callback: Callable[[Order, str], None]) -> None:
        """Set callback function for order cancellation."""
        self.cancel_callback = callback
        logger.info("cancel_callback_set")
    
    def get_spread_snapshot(self, symbol: str) -> Optional[SpreadSnapshot]:
        """Get current spread snapshot for a symbol."""
        return self.spread_snapshots.get(symbol)
    
    def _cleanup_old_orders(self) -> None:
        """Internal method to cleanup old orders when max size exceeded."""
        # Remove oldest filled/cancelled orders first
        cutoff = datetime.now() - timedelta(hours=24)
        
        to_remove = [
            order_id for order_id, order in self.orders.items()
            if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED]
            and order.created_at < cutoff
        ]
        
        # If still too many, remove oldest regardless of status
        if len(self.orders) - len(to_remove) > self.max_orders:
            remaining = [
                (order_id, order.created_at)
                for order_id, order in self.orders.items()
                if order_id not in to_remove
            ]
            remaining.sort(key=lambda x: x[1])  # Sort by creation time
            
            excess = len(self.orders) - self.max_orders
            to_remove.extend([order_id for order_id, _ in remaining[:excess]])
        
        for order_id in to_remove:
            del self.orders[order_id]
        
        logger.debug("orders_cleaned_up", removed=len(to_remove), remaining=len(self.orders))
    
    def clear_old_orders(self, max_age_hours: int = 24) -> int:
        """Clear old filled/cancelled orders."""
        cutoff = datetime.now() - timedelta(hours=max_age_hours)
        
        to_remove = [
            order_id for order_id, order in self.orders.items()
            if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED]
            and order.created_at < cutoff
        ]
        
        for order_id in to_remove:
            del self.orders[order_id]
        
        logger.info("old_orders_cleared", count=len(to_remove))
        
        return len(to_remove)

