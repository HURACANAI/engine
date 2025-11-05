"""
Fill-Time SLA Tracker - Execution Quality Monitoring

Key Problem:
Maker orders may not fill, wasting opportunities.
- Post limit order at best bid/ask
- Wait for fill... wait... wait... STILL NO FILL
- Price moves away → Missed opportunity
- No tracking = No awareness of problem

Solution: Fill-Time SLA Tracking
- Monitor time-to-fill for all orders
- Track fill rates by order type
- Detect systematic fill failures
- Auto-switch to taker if maker repeatedly fails

Metrics Tracked:
1. **Fill Rate**: % of orders that fill
2. **Time-to-Fill**: How long until order fills
3. **Timeout Rate**: % of orders that timeout
4. **Slippage on Timeout**: Cost of falling back to taker

Example:
    Maker orders on ETH-USD:
    - Fill rate: 60% (40% timeout!)
    - Avg time-to-fill: 8 seconds
    - Timeout slippage: avg 12 bps

    → Problem: Too many timeouts, expensive fallback
    → Solution: Switch to taker orders OR widen limit prices

Benefits:
- Quantifies execution quality
- Detects fill rate issues
- Automatic fallback strategies
- Per-asset and per-book tracking

Usage:
    sla_tracker = FillTimeSLATracker(
        target_fill_rate=0.80,  # Want 80%+ fills
        max_wait_time_sec=10.0,  # 10 sec timeout
    )

    # Submit order
    order_id = sla_tracker.submit_order(
        symbol='ETH-USD',
        order_type='maker',
        size_usd=200.0,
        book='scalp',
    )

    # ... wait for fill ...

    # Record fill
    sla_tracker.record_fill(
        order_id=order_id,
        filled=True,
        fill_time_sec=3.5,
        slippage_bps=1.2,
    )

    # Check if should switch to taker
    if sla_tracker.should_use_taker('ETH-USD', 'scalp'):
        order_type = 'taker'
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum
import time
import uuid

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


class OrderStatus(Enum):
    """Order status."""

    PENDING = "pending"
    FILLED = "filled"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class OrderRecord:
    """Record of a single order."""

    order_id: str
    symbol: str
    order_type: str  # 'maker' or 'taker'
    book: str  # 'scalp' or 'runner'
    size_usd: float

    submit_time: float
    status: OrderStatus = OrderStatus.PENDING

    # Fill details (if filled)
    filled: bool = False
    fill_time: Optional[float] = None
    time_to_fill_sec: Optional[float] = None
    slippage_bps: Optional[float] = None


@dataclass
class FillRateMetrics:
    """Fill rate metrics for a symbol/book combination."""

    symbol: str
    book: str
    order_type: str

    total_orders: int
    filled_orders: int
    timeout_orders: int
    cancelled_orders: int

    fill_rate: float  # % filled
    timeout_rate: float  # % timeout

    avg_time_to_fill_sec: float
    median_time_to_fill_sec: float
    p95_time_to_fill_sec: float

    avg_slippage_bps: float
    avg_timeout_slippage_bps: float  # Slippage when timing out


class FillTimeSLATracker:
    """
    Track fill-time SLA for execution quality monitoring.

    Architecture:
        Submit Order → Track Status → Record Fill/Timeout
            ↓
        Calculate Metrics (fill rate, time-to-fill, slippage)
            ↓
        Recommend Adjustments (switch to taker, widen limits)
    """

    def __init__(
        self,
        target_fill_rate: float = 0.80,  # Want 80%+ fill rate
        max_wait_time_sec: float = 10.0,  # Timeout after 10 sec
        min_orders_for_recommendation: int = 20,
    ):
        """
        Initialize fill-time SLA tracker.

        Args:
            target_fill_rate: Target fill rate (e.g., 0.80 = 80%)
            max_wait_time_sec: Max time to wait before timeout
            min_orders_for_recommendation: Min orders before recommendations
        """
        self.target_fill_rate = target_fill_rate
        self.max_wait_time = max_wait_time_sec
        self.min_orders = min_orders_for_recommendation

        # Order tracking
        self.orders: Dict[str, OrderRecord] = {}

        # Statistics by (symbol, book, order_type)
        self.metrics_cache: Dict[tuple, FillRateMetrics] = {}

        logger.info(
            "fill_time_sla_initialized",
            target_fill_rate=target_fill_rate,
            max_wait_time=max_wait_time_sec,
        )

    def submit_order(
        self,
        symbol: str,
        order_type: str,
        size_usd: float,
        book: str,
    ) -> str:
        """
        Submit order and start tracking.

        Args:
            symbol: Asset symbol
            order_type: 'maker' or 'taker'
            size_usd: Order size in USD
            book: 'scalp' or 'runner'

        Returns:
            order_id for reference
        """
        order_id = str(uuid.uuid4())

        order = OrderRecord(
            order_id=order_id,
            symbol=symbol,
            order_type=order_type,
            book=book,
            size_usd=size_usd,
            submit_time=time.time(),
        )

        self.orders[order_id] = order

        logger.debug(
            "order_submitted",
            order_id=order_id[:8],
            symbol=symbol,
            order_type=order_type,
            book=book,
        )

        return order_id

    def record_fill(
        self,
        order_id: str,
        filled: bool,
        fill_time_sec: Optional[float] = None,
        slippage_bps: Optional[float] = None,
    ) -> None:
        """
        Record order fill or timeout.

        Args:
            order_id: Order ID
            filled: True if filled, False if timeout
            fill_time_sec: Time to fill (if filled)
            slippage_bps: Slippage in bps
        """
        if order_id not in self.orders:
            logger.warning("order_not_found", order_id=order_id)
            return

        order = self.orders[order_id]

        if filled:
            order.status = OrderStatus.FILLED
            order.filled = True
            order.fill_time = time.time()
            order.time_to_fill_sec = fill_time_sec or (order.fill_time - order.submit_time)
            order.slippage_bps = slippage_bps or 0.0

            logger.debug(
                "order_filled",
                order_id=order_id[:8],
                symbol=order.symbol,
                time_to_fill=order.time_to_fill_sec,
                slippage=order.slippage_bps,
            )
        else:
            # Timeout
            order.status = OrderStatus.TIMEOUT
            order.filled = False
            order.time_to_fill_sec = time.time() - order.submit_time
            order.slippage_bps = slippage_bps or 0.0  # Slippage from fallback order

            logger.warning(
                "order_timeout",
                order_id=order_id[:8],
                symbol=order.symbol,
                waited_sec=order.time_to_fill_sec,
                fallback_slippage=order.slippage_bps,
            )

        # Invalidate metrics cache
        key = (order.symbol, order.book, order.order_type)
        if key in self.metrics_cache:
            del self.metrics_cache[key]

    def check_timeouts(self) -> List[str]:
        """
        Check for orders that have exceeded max wait time.

        Returns:
            List of order IDs that have timed out
        """
        current_time = time.time()
        timed_out = []

        for order_id, order in self.orders.items():
            if order.status == OrderStatus.PENDING:
                elapsed = current_time - order.submit_time

                if elapsed > self.max_wait_time:
                    order.status = OrderStatus.TIMEOUT
                    order.time_to_fill_sec = elapsed
                    timed_out.append(order_id)

                    logger.warning(
                        "order_auto_timeout",
                        order_id=order_id[:8],
                        symbol=order.symbol,
                        elapsed_sec=elapsed,
                    )

        return timed_out

    def get_metrics(
        self,
        symbol: str,
        book: str,
        order_type: str = 'maker',
    ) -> Optional[FillRateMetrics]:
        """
        Get fill rate metrics for symbol/book/order_type.

        Args:
            symbol: Asset symbol
            book: 'scalp' or 'runner'
            order_type: 'maker' or 'taker'

        Returns:
            FillRateMetrics or None if insufficient data
        """
        key = (symbol, book, order_type)

        # Check cache
        if key in self.metrics_cache:
            return self.metrics_cache[key]

        # Filter orders
        relevant_orders = [
            o
            for o in self.orders.values()
            if o.symbol == symbol and o.book == book and o.order_type == order_type
            and o.status in [OrderStatus.FILLED, OrderStatus.TIMEOUT]
        ]

        if len(relevant_orders) < self.min_orders:
            return None

        # Calculate metrics
        total_orders = len(relevant_orders)
        filled_orders = sum(1 for o in relevant_orders if o.status == OrderStatus.FILLED)
        timeout_orders = sum(1 for o in relevant_orders if o.status == OrderStatus.TIMEOUT)
        cancelled_orders = sum(1 for o in relevant_orders if o.status == OrderStatus.CANCELLED)

        fill_rate = filled_orders / total_orders
        timeout_rate = timeout_orders / total_orders

        # Time-to-fill stats (only for filled orders)
        filled_times = [o.time_to_fill_sec for o in relevant_orders if o.filled and o.time_to_fill_sec]

        if filled_times:
            avg_time_to_fill = np.mean(filled_times)
            median_time_to_fill = np.median(filled_times)
            p95_time_to_fill = np.percentile(filled_times, 95)
        else:
            avg_time_to_fill = 0.0
            median_time_to_fill = 0.0
            p95_time_to_fill = 0.0

        # Slippage stats
        filled_slippages = [o.slippage_bps for o in relevant_orders if o.filled and o.slippage_bps is not None]
        timeout_slippages = [o.slippage_bps for o in relevant_orders if o.status == OrderStatus.TIMEOUT and o.slippage_bps is not None]

        avg_slippage = np.mean(filled_slippages) if filled_slippages else 0.0
        avg_timeout_slippage = np.mean(timeout_slippages) if timeout_slippages else 0.0

        metrics = FillRateMetrics(
            symbol=symbol,
            book=book,
            order_type=order_type,
            total_orders=total_orders,
            filled_orders=filled_orders,
            timeout_orders=timeout_orders,
            cancelled_orders=cancelled_orders,
            fill_rate=fill_rate,
            timeout_rate=timeout_rate,
            avg_time_to_fill_sec=avg_time_to_fill,
            median_time_to_fill_sec=median_time_to_fill,
            p95_time_to_fill_sec=p95_time_to_fill,
            avg_slippage_bps=avg_slippage,
            avg_timeout_slippage_bps=avg_timeout_slippage,
        )

        # Cache result
        self.metrics_cache[key] = metrics

        return metrics

    def should_use_taker(
        self,
        symbol: str,
        book: str,
    ) -> tuple[bool, str]:
        """
        Determine if should switch to taker orders based on fill rate.

        Args:
            symbol: Asset symbol
            book: 'scalp' or 'runner'

        Returns:
            (should_switch, reason)
        """
        metrics = self.get_metrics(symbol, book, order_type='maker')

        if metrics is None:
            return False, "Insufficient data"

        # Check fill rate
        if metrics.fill_rate < self.target_fill_rate:
            return (
                True,
                f"Low fill rate: {metrics.fill_rate:.0%} < target {self.target_fill_rate:.0%}, "
                f"timeout rate: {metrics.timeout_rate:.0%}, "
                f"timeout slippage: {metrics.avg_timeout_slippage_bps:.1f} bps",
            )

        # Check timeout cost
        if metrics.timeout_rate > 0.20 and metrics.avg_timeout_slippage_bps > 5.0:
            return (
                True,
                f"High timeout cost: {metrics.timeout_rate:.0%} timeout with "
                f"{metrics.avg_timeout_slippage_bps:.1f} bps slippage",
            )

        return False, f"Fill rate OK: {metrics.fill_rate:.0%}"

    def get_summary(self) -> Dict:
        """Get summary of all tracked orders."""
        # Group by (symbol, book, order_type)
        summary = {}

        symbols_books_types = set(
            (o.symbol, o.book, o.order_type)
            for o in self.orders.values()
            if o.status in [OrderStatus.FILLED, OrderStatus.TIMEOUT]
        )

        for symbol, book, order_type in symbols_books_types:
            metrics = self.get_metrics(symbol, book, order_type)

            if metrics:
                key = f"{symbol}_{book}_{order_type}"
                summary[key] = {
                    'fill_rate': metrics.fill_rate,
                    'timeout_rate': metrics.timeout_rate,
                    'avg_time_to_fill_sec': metrics.avg_time_to_fill_sec,
                    'avg_slippage_bps': metrics.avg_slippage_bps,
                    'total_orders': metrics.total_orders,
                }

        return summary


def run_fill_time_example():
    """Example usage of fill-time SLA tracker."""
    np.random.seed(42)

    tracker = FillTimeSLATracker(
        target_fill_rate=0.80,
        max_wait_time_sec=10.0,
        min_orders_for_recommendation=20,
    )

    print("=" * 70)
    print("FILL-TIME SLA TRACKING SIMULATION")
    print("=" * 70)

    # Simulate 50 maker orders with varying fill rates
    for i in range(50):
        order_id = tracker.submit_order(
            symbol='ETH-USD',
            order_type='maker',
            size_usd=200.0,
            book='scalp',
        )

        # Simulate fill outcome (70% fill rate)
        filled = np.random.random() < 0.70

        if filled:
            # Filled - quick fill time
            fill_time = np.random.uniform(1.0, 5.0)
            slippage = np.random.uniform(0.5, 2.0)
        else:
            # Timeout - long wait, higher slippage from fallback
            fill_time = 10.0
            slippage = np.random.uniform(8.0, 15.0)

        tracker.record_fill(
            order_id=order_id,
            filled=filled,
            fill_time_sec=fill_time,
            slippage_bps=slippage,
        )

    # Get metrics
    metrics = tracker.get_metrics('ETH-USD', 'scalp', 'maker')

    if metrics:
        print(f"\nMETRICS FOR ETH-USD SCALP MAKER:")
        print(f"  Total Orders: {metrics.total_orders}")
        print(f"  Fill Rate: {metrics.fill_rate:.0%}")
        print(f"  Timeout Rate: {metrics.timeout_rate:.0%}")
        print(f"  Avg Time-to-Fill: {metrics.avg_time_to_fill_sec:.2f} sec")
        print(f"  Median Time-to-Fill: {metrics.median_time_to_fill_sec:.2f} sec")
        print(f"  P95 Time-to-Fill: {metrics.p95_time_to_fill_sec:.2f} sec")
        print(f"  Avg Slippage: {metrics.avg_slippage_bps:.2f} bps")
        print(f"  Avg Timeout Slippage: {metrics.avg_timeout_slippage_bps:.2f} bps")

    # Check recommendation
    should_switch, reason = tracker.should_use_taker('ETH-USD', 'scalp')

    print(f"\nRECOMMENDATION:")
    print(f"  Should use taker: {should_switch}")
    print(f"  Reason: {reason}")


if __name__ == "__main__":
    run_fill_time_example()
