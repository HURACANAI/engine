"""
Execution Simulator

Simulates realistic order execution.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


class OrderType(str, Enum):
    """Order types"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_MARKET = "STOP_MARKET"


@dataclass
class ExecutionResult:
    """Execution simulation result"""
    order_type: str
    side: str  # BUY or SELL

    requested_quantity: float
    filled_quantity: float
    fill_rate: float  # [0-1]

    market_price: float
    avg_fill_price: float
    slippage_bps: float

    total_fees: float
    net_cost: float  # Total cost including fees

    latency_ms: float
    partial_fill: bool


class ExecutionSimulator:
    """
    Execution Simulator

    Simulates realistic order execution with:
    - Slippage (market impact + volatility)
    - Partial fills
    - Latency
    - Fees

    Example:
        sim = ExecutionSimulator()

        result = sim.simulate_execution(
            order_type="MARKET",
            side="BUY",
            quantity=2.0,
            market_price=50000,
            volatility=0.02
        )

        print(f"Slippage: {result.slippage_bps:.1f} bps")
    """

    def __init__(
        self,
        base_fee_bps: float = 4.0,  # 4 bps = 0.04%
        base_slippage_bps: float = 2.0,
        base_latency_ms: float = 50.0,
        partial_fill_threshold: float = 0.001  # 0.1% of book depth
    ):
        """
        Initialize execution simulator

        Args:
            base_fee_bps: Base trading fee in bps
            base_slippage_bps: Base slippage in bps
            base_latency_ms: Base latency in milliseconds
            partial_fill_threshold: Threshold for partial fills
        """
        self.base_fee_bps = base_fee_bps
        self.base_slippage_bps = base_slippage_bps
        self.base_latency_ms = base_latency_ms
        self.partial_fill_threshold = partial_fill_threshold

    def simulate_execution(
        self,
        order_type: OrderType,
        side: str,
        quantity: float,
        market_price: float,
        book_liquidity: Optional[Dict] = None,
        volatility: float = 0.02,
        urgent: bool = False
    ) -> ExecutionResult:
        """
        Simulate order execution

        Args:
            order_type: Order type
            side: BUY or SELL
            quantity: Order quantity
            market_price: Current market price
            book_liquidity: Optional book depth info
            volatility: Market volatility [0-1]
            urgent: Whether order is urgent (affects slippage)

        Returns:
            ExecutionResult
        """
        # Calculate slippage
        slippage_bps = self._calculate_slippage(
            quantity, market_price, volatility, urgent, book_liquidity
        )

        # Determine fill rate (partial fills)
        fill_rate = self._calculate_fill_rate(
            quantity, book_liquidity, order_type
        )

        filled_quantity = quantity * fill_rate
        partial_fill = fill_rate < 1.0

        # Calculate fill price
        if side == "BUY":
            avg_fill_price = market_price * (1 + slippage_bps / 10000)
        else:  # SELL
            avg_fill_price = market_price * (1 - slippage_bps / 10000)

        # Calculate fees
        total_fees = self._calculate_fees(filled_quantity, avg_fill_price)

        # Net cost (includes fees)
        if side == "BUY":
            net_cost = filled_quantity * avg_fill_price + total_fees
        else:
            net_cost = filled_quantity * avg_fill_price - total_fees

        # Latency
        latency_ms = self._calculate_latency(urgent)

        result = ExecutionResult(
            order_type=order_type.value,
            side=side,
            requested_quantity=quantity,
            filled_quantity=filled_quantity,
            fill_rate=fill_rate,
            market_price=market_price,
            avg_fill_price=avg_fill_price,
            slippage_bps=slippage_bps,
            total_fees=total_fees,
            net_cost=net_cost,
            latency_ms=latency_ms,
            partial_fill=partial_fill
        )

        logger.debug(
            "execution_simulated",
            side=side,
            quantity=quantity,
            fill_rate=fill_rate,
            slippage_bps=slippage_bps
        )

        return result

    def _calculate_slippage(
        self,
        quantity: float,
        market_price: float,
        volatility: float,
        urgent: bool,
        book_liquidity: Optional[Dict]
    ) -> float:
        """
        Calculate slippage in bps

        Slippage = base + market_impact + volatility_component

        Args:
            quantity: Order size
            market_price: Market price
            volatility: Volatility
            urgent: Urgency flag
            book_liquidity: Book depth

        Returns:
            Slippage in bps
        """
        # Base slippage
        slippage = self.base_slippage_bps

        # Market impact (larger orders = more impact)
        if book_liquidity:
            total_depth = book_liquidity.get('bid_depth', 10) + book_liquidity.get('ask_depth', 10)
            order_value = quantity * market_price
            depth_value = total_depth * market_price

            if depth_value > 0:
                market_impact = (order_value / depth_value) * 50  # Scale factor
                slippage += market_impact

        # Volatility component
        volatility_component = volatility * 100  # Convert to bps
        slippage += volatility_component

        # Urgency multiplier
        if urgent:
            slippage *= 1.5

        return slippage

    def _calculate_fill_rate(
        self,
        quantity: float,
        book_liquidity: Optional[Dict],
        order_type: OrderType
    ) -> float:
        """
        Calculate fill rate (1.0 = complete fill)

        Args:
            quantity: Order quantity
            book_liquidity: Book depth
            order_type: Order type

        Returns:
            Fill rate [0-1]
        """
        # Market orders usually fill completely (unless huge)
        if order_type == OrderType.MARKET:
            if book_liquidity:
                depth = book_liquidity.get('bid_depth', 10) + book_liquidity.get('ask_depth', 10)

                if quantity > depth * self.partial_fill_threshold:
                    # Large order - partial fill
                    fill_rate = min(1.0, depth / quantity * 0.8)
                    return fill_rate

            return 1.0  # Complete fill

        # Limit orders may not fill
        elif order_type == OrderType.LIMIT:
            # Simulate 70% fill rate for limit orders
            return 0.7 if np.random.random() < 0.7 else 0.0

        return 1.0

    def _calculate_fees(self, quantity: float, price: float) -> float:
        """
        Calculate trading fees

        Args:
            quantity: Filled quantity
            price: Fill price

        Returns:
            Total fees
        """
        order_value = quantity * price
        fees = order_value * (self.base_fee_bps / 10000)

        return fees

    def _calculate_latency(self, urgent: bool) -> float:
        """
        Calculate execution latency

        Args:
            urgent: Urgency flag

        Returns:
            Latency in milliseconds
        """
        # Base latency with some randomness
        latency = self.base_latency_ms * (0.8 + np.random.random() * 0.4)

        # Urgent orders get priority (lower latency)
        if urgent:
            latency *= 0.6

        return latency

    def simulate_portfolio_execution(
        self,
        orders: list[Dict]
    ) -> list[ExecutionResult]:
        """
        Simulate execution of multiple orders

        Args:
            orders: List of order dicts

        Returns:
            List of ExecutionResult
        """
        results = []

        for order in orders:
            result = self.simulate_execution(
                order_type=OrderType(order['order_type']),
                side=order['side'],
                quantity=order['quantity'],
                market_price=order['market_price'],
                book_liquidity=order.get('book_liquidity'),
                volatility=order.get('volatility', 0.02),
                urgent=order.get('urgent', False)
            )
            results.append(result)

        return results

    def get_execution_statistics(
        self,
        results: list[ExecutionResult]
    ) -> Dict:
        """
        Get aggregate execution statistics

        Args:
            results: List of execution results

        Returns:
            Statistics dict
        """
        if len(results) == 0:
            return {}

        total_slippage = [r.slippage_bps for r in results]
        total_fees = sum(r.total_fees for r in results)
        fill_rates = [r.fill_rate for r in results]
        latencies = [r.latency_ms for r in results]

        return {
            "num_executions": len(results),
            "avg_slippage_bps": np.mean(total_slippage),
            "total_fees": total_fees,
            "avg_fill_rate": np.mean(fill_rates),
            "avg_latency_ms": np.mean(latencies),
            "partial_fills": sum(1 for r in results if r.partial_fill)
        }
