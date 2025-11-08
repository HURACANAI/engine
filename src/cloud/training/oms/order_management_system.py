"""
Order Management System (OMS)

Tracks every order's lifecycle (sent, filled, rejected).
Integrates with latency monitoring and provides real-time dashboards.

Key Features:
- Order lifecycle tracking
- Status management
- Latency tracking
- Fill tracking
- Rejection tracking
- Real-time dashboards
- Integration with monitoring

Author: Huracan Engine Team
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from enum import Enum
from datetime import datetime, timezone
import uuid

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


class OrderStatus(Enum):
    """Order status"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    ACKNOWLEDGED = "acknowledged"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    REJECTED = "rejected"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    FAILED = "failed"


class OrderType(Enum):
    """Order type"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


@dataclass
class Order:
    """Order representation"""
    order_id: str
    symbol: str
    direction: str  # "buy" or "sell"
    order_type: OrderType
    size: float
    price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    exchange_id: Optional[str] = None
    exchange_order_id: Optional[str] = None
    
    # Timestamps (nanoseconds)
    created_at_ns: int = 0
    submitted_at_ns: Optional[int] = None
    acknowledged_at_ns: Optional[int] = None
    filled_at_ns: Optional[int] = None
    rejected_at_ns: Optional[int] = None
    
    # Fill information
    filled_size: float = 0.0
    filled_price: Optional[float] = None
    average_fill_price: Optional[float] = None
    
    # Rejection information
    rejection_reason: Optional[str] = None
    rejection_code: Optional[str] = None
    
    # Metadata
    metadata: Dict[str, any] = field(default_factory=dict)


@dataclass
class OrderLifecycle:
    """Order lifecycle tracking"""
    order_id: str
    stages: List[Dict[str, any]] = field(default_factory=list)
    total_duration_ns: int = 0
    tick_to_trade_ns: Optional[int] = None
    submit_latency_ns: Optional[int] = None
    fill_latency_ns: Optional[int] = None


@dataclass
class OMSMetrics:
    """OMS metrics"""
    total_orders: int
    pending_orders: int
    filled_orders: int
    rejected_orders: int
    cancelled_orders: int
    fill_rate: float
    rejection_rate: float
    avg_fill_latency_ms: float
    avg_submit_latency_ms: float
    tick_to_trade_avg_ms: float


class OrderManagementSystem:
    """
    Order Management System.
    
    Tracks every order's lifecycle and provides real-time monitoring.
    
    Usage:
        oms = OrderManagementSystem()
        
        # Create order
        order = oms.create_order(
            symbol="BTCUSDT",
            direction="buy",
            order_type=OrderType.MARKET,
            size=0.1
        )
        
        # Update status
        oms.update_order_status(order.order_id, OrderStatus.SUBMITTED)
        
        # Track fill
        oms.record_fill(order.order_id, filled_size=0.1, filled_price=50000.0)
        
        # Get metrics
        metrics = oms.get_metrics()
    """
    
    def __init__(self):
        """Initialize OMS"""
        self.orders: Dict[str, Order] = {}
        self.order_lifecycles: Dict[str, OrderLifecycle] = {}
        
        logger.info("oms_initialized")
    
    def create_order(
        self,
        symbol: str,
        direction: str,
        order_type: OrderType,
        size: float,
        price: Optional[float] = None,
        exchange_id: Optional[str] = None,
        metadata: Optional[Dict[str, any]] = None,
        tick_timestamp_ns: Optional[int] = None
    ) -> Order:
        """
        Create a new order.
        
        Args:
            symbol: Trading symbol
            direction: Order direction
            order_type: Order type
            size: Order size
            price: Order price (optional)
            exchange_id: Exchange ID (optional)
            metadata: Optional metadata
            tick_timestamp_ns: Tick timestamp in nanoseconds (for latency tracking)
        
        Returns:
            Order
        """
        import time
        
        order_id = str(uuid.uuid4())
        created_at_ns = time.perf_counter_ns()
        
        order = Order(
            order_id=order_id,
            symbol=symbol,
            direction=direction,
            order_type=order_type,
            size=size,
            price=price,
            status=OrderStatus.PENDING,
            exchange_id=exchange_id,
            created_at_ns=created_at_ns,
            metadata=metadata or {}
        )
        
        # Create lifecycle tracking
        lifecycle = OrderLifecycle(
            order_id=order_id,
            tick_to_trade_ns=tick_timestamp_ns
        )
        lifecycle.stages.append({
            "stage": "created",
            "timestamp_ns": created_at_ns,
            "status": OrderStatus.PENDING.value
        })
        
        self.orders[order_id] = order
        self.order_lifecycles[order_id] = lifecycle
        
        logger.info(
            "order_created",
            order_id=order_id,
            symbol=symbol,
            direction=direction,
            order_type=order_type.value,
            size=size
        )
        
        return order
    
    def update_order_status(
        self,
        order_id: str,
        status: OrderStatus,
        exchange_order_id: Optional[str] = None,
        rejection_reason: Optional[str] = None,
        rejection_code: Optional[str] = None
    ) -> None:
        """
        Update order status.
        
        Args:
            order_id: Order ID
            status: New status
            exchange_order_id: Exchange order ID (optional)
            rejection_reason: Rejection reason (optional)
            rejection_code: Rejection code (optional)
        """
        import time
        
        if order_id not in self.orders:
            logger.warning("order_not_found", order_id=order_id)
            return
        
        order = self.orders[order_id]
        order.status = status
        
        timestamp_ns = time.perf_counter_ns()
        
        # Update timestamps based on status
        if status == OrderStatus.SUBMITTED:
            order.submitted_at_ns = timestamp_ns
            if order.created_at_ns > 0:
                submit_latency = timestamp_ns - order.created_at_ns
                if order_id in self.order_lifecycles:
                    self.order_lifecycles[order_id].submit_latency_ns = submit_latency
        elif status == OrderStatus.ACKNOWLEDGED:
            order.acknowledged_at_ns = timestamp_ns
        elif status == OrderStatus.REJECTED:
            order.rejected_at_ns = timestamp_ns
            order.rejection_reason = rejection_reason
            order.rejection_code = rejection_code
            if order.created_at_ns > 0:
                lifecycle = self.order_lifecycles.get(order_id)
                if lifecycle:
                    lifecycle.total_duration_ns = timestamp_ns - order.created_at_ns
        elif status == OrderStatus.FILLED:
            order.filled_at_ns = timestamp_ns
            if order.submitted_at_ns:
                fill_latency = timestamp_ns - order.submitted_at_ns
                if order_id in self.order_lifecycles:
                    self.order_lifecycles[order_id].fill_latency_ns = fill_latency
            if order.created_at_ns > 0:
                lifecycle = self.order_lifecycles.get(order_id)
                if lifecycle:
                    lifecycle.total_duration_ns = timestamp_ns - order.created_at_ns
                    if lifecycle.tick_to_trade_ns:
                        lifecycle.tick_to_trade_ns = timestamp_ns - lifecycle.tick_to_trade_ns
        
        if exchange_order_id:
            order.exchange_order_id = exchange_order_id
        
        # Update lifecycle
        if order_id in self.order_lifecycles:
            self.order_lifecycles[order_id].stages.append({
                "stage": status.value,
                "timestamp_ns": timestamp_ns,
                "status": status.value
            })
        
        logger.debug(
            "order_status_updated",
            order_id=order_id,
            status=status.value,
            rejection_reason=rejection_reason
        )
    
    def record_fill(
        self,
        order_id: str,
        filled_size: float,
        filled_price: float,
        partial: bool = False
    ) -> None:
        """
        Record order fill.
        
        Args:
            order_id: Order ID
            filled_size: Filled size
            filled_price: Fill price
            partial: Whether this is a partial fill
        """
        if order_id not in self.orders:
            logger.warning("order_not_found", order_id=order_id)
            return
        
        order = self.orders[order_id]
        
        # Update fill information
        if order.filled_size == 0:
            order.filled_size = filled_size
            order.filled_price = filled_price
            order.average_fill_price = filled_price
        else:
            # Partial fill: update average price
            total_filled = order.filled_size + filled_size
            order.average_fill_price = (
                (order.average_fill_price * order.filled_size + filled_price * filled_size) /
                total_filled
            )
            order.filled_size = total_filled
        
        # Update status
        if order.filled_size >= order.size:
            self.update_order_status(order_id, OrderStatus.FILLED)
        elif partial:
            self.update_order_status(order_id, OrderStatus.PARTIALLY_FILLED)
        
        logger.info(
            "order_fill_recorded",
            order_id=order_id,
            filled_size=filled_size,
            filled_price=filled_price,
            total_filled=order.filled_size,
            average_price=order.average_fill_price
        )
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID"""
        return self.orders.get(order_id)
    
    def get_orders_by_status(self, status: OrderStatus) -> List[Order]:
        """Get all orders with given status"""
        return [order for order in self.orders.values() if order.status == status]
    
    def get_orders_by_symbol(self, symbol: str) -> List[Order]:
        """Get all orders for a symbol"""
        return [order for order in self.orders.values() if order.symbol == symbol]
    
    def get_metrics(self) -> OMSMetrics:
        """Get OMS metrics"""
        total_orders = len(self.orders)
        
        pending_orders = len(self.get_orders_by_status(OrderStatus.PENDING))
        filled_orders = len(self.get_orders_by_status(OrderStatus.FILLED))
        rejected_orders = len(self.get_orders_by_status(OrderStatus.REJECTED))
        cancelled_orders = len(self.get_orders_by_status(OrderStatus.CANCELLED))
        
        fill_rate = filled_orders / total_orders if total_orders > 0 else 0.0
        rejection_rate = rejected_orders / total_orders if total_orders > 0 else 0.0
        
        # Calculate average latencies
        fill_latencies = []
        submit_latencies = []
        tick_to_trade_latencies = []
        
        for lifecycle in self.order_lifecycles.values():
            if lifecycle.fill_latency_ns:
                fill_latencies.append(lifecycle.fill_latency_ns / 1_000_000)  # Convert to ms
            if lifecycle.submit_latency_ns:
                submit_latencies.append(lifecycle.submit_latency_ns / 1_000_000)
            if lifecycle.tick_to_trade_ns:
                tick_to_trade_latencies.append(lifecycle.tick_to_trade_ns / 1_000_000)
        
        avg_fill_latency_ms = sum(fill_latencies) / len(fill_latencies) if fill_latencies else 0.0
        avg_submit_latency_ms = sum(submit_latencies) / len(submit_latencies) if submit_latencies else 0.0
        tick_to_trade_avg_ms = sum(tick_to_trade_latencies) / len(tick_to_trade_latencies) if tick_to_trade_latencies else 0.0
        
        return OMSMetrics(
            total_orders=total_orders,
            pending_orders=pending_orders,
            filled_orders=filled_orders,
            rejected_orders=rejected_orders,
            cancelled_orders=cancelled_orders,
            fill_rate=fill_rate,
            rejection_rate=rejection_rate,
            avg_fill_latency_ms=avg_fill_latency_ms,
            avg_submit_latency_ms=avg_submit_latency_ms,
            tick_to_trade_avg_ms=tick_to_trade_avg_ms
        )
    
    def get_dashboard_data(self) -> Dict[str, any]:
        """Get data for OMS dashboard"""
        metrics = self.get_metrics()
        
        # Get recent orders
        recent_orders = list(self.orders.values())[-100:]  # Last 100 orders
        
        # Get orders by status
        orders_by_status = {
            status.value: len(self.get_orders_by_status(status))
            for status in OrderStatus
        }
        
        # Get latency statistics
        fill_latencies = [
            l.fill_latency_ns / 1_000_000
            for l in self.order_lifecycles.values()
            if l.fill_latency_ns
        ]
        
        return {
            "metrics": {
                "total_orders": metrics.total_orders,
                "fill_rate": metrics.fill_rate,
                "rejection_rate": metrics.rejection_rate,
                "avg_fill_latency_ms": metrics.avg_fill_latency_ms,
                "avg_submit_latency_ms": metrics.avg_submit_latency_ms,
                "tick_to_trade_avg_ms": metrics.tick_to_trade_avg_ms
            },
            "orders_by_status": orders_by_status,
            "recent_orders": [
                {
                    "order_id": o.order_id,
                    "symbol": o.symbol,
                    "status": o.status.value,
                    "size": o.size,
                    "filled_size": o.filled_size
                }
                for o in recent_orders
            ],
            "latency_stats": {
                "fill_latency_p50_ms": float(np.percentile(fill_latencies, 50)) if fill_latencies else 0.0,
                "fill_latency_p95_ms": float(np.percentile(fill_latencies, 95)) if fill_latencies else 0.0,
                "fill_latency_p99_ms": float(np.percentile(fill_latencies, 99)) if fill_latencies else 0.0
            }
        }

