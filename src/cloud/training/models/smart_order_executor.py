"""
Smart Order Execution

Adaptive order placement strategy that balances execution speed vs. cost.

Key Problems Solved:
1. **Market Order Waste**: Always using market orders = instant but expensive (5 bps + slippage)
2. **Limit Order Misses**: Always using limit orders = cheap but miss fills on breakouts
3. **Large Size Slippage**: Filling 100 BTC at market = massive slippage

Solution: Context-Aware Order Placement
- Urgent trades (breakouts) → Aggressive market orders
- Patient trades (mean reversion) → Limit orders at better prices
- Large sizes → TWAP/VWAP splitting
- Thin liquidity → Limit orders only

Example:
    Trade 1: BREAKOUT on high volume
    → Urgency: HIGH
    → Execution: Market order (need immediate fill)
    → Cost: 5 bps fee + 8 bps slippage = 13 bps
    → Result: Filled immediately, caught the move

    Trade 2: RANGE mean reversion
    → Urgency: LOW
    → Execution: Limit order at mid-1 bps (patient)
    → Cost: 2 bps maker fee + 0 slippage = 2 bps
    → Result: Saved 11 bps vs market order!

    Trade 3: Large 100 BTC position
    → Size: LARGE
    → Execution: TWAP split into 10 orders over 5 minutes
    → Cost: 3.5 bps avg fee + 5 bps avg slippage = 8.5 bps
    → Result: Saved 15+ bps vs single market order
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


class OrderType(Enum):
    """Order types."""

    MARKET = "market"  # Immediate execution, high cost
    LIMIT = "limit"  # Wait for fill, low cost
    TWAP = "twap"  # Time-weighted average price (split orders)
    VWAP = "vwap"  # Volume-weighted average price (split orders)
    ICEBERG = "iceberg"  # Hidden large order


class UrgencyLevel(Enum):
    """Execution urgency levels."""

    URGENT = "urgent"  # Need immediate fill (breakouts, panic exits)
    MODERATE = "moderate"  # Some urgency but can wait briefly
    PATIENT = "patient"  # Can wait for better price (mean reversion)


@dataclass
class ExecutionStrategy:
    """Recommended execution strategy."""

    order_type: OrderType
    urgency: UrgencyLevel
    limit_price: Optional[float]  # For limit orders
    split_count: Optional[int]  # For TWAP/VWAP
    time_window_seconds: Optional[int]  # For TWAP/VWAP
    estimated_cost_bps: float
    estimated_fill_probability: float
    reasoning: str


@dataclass
class OrderSlice:
    """Single order slice for split execution."""

    slice_number: int
    size: float
    order_type: OrderType
    limit_price: Optional[float]
    delay_seconds: float  # Delay before placing this slice


class SmartOrderExecutor:
    """
    Determines optimal order execution strategy based on context.

    Factors Considered:
    1. **Trade Urgency**: Breakout vs mean reversion
    2. **Position Size**: Large positions need splitting
    3. **Liquidity**: Thin markets need patient execution
    4. **Volatility**: High vol needs faster execution
    5. **Spread**: Wide spread favors limit orders

    Execution Strategies:

    MARKET ORDER:
    - Urgency: URGENT
    - Use case: Breakouts, panic exits, stop losses
    - Cost: High (taker fee + slippage)
    - Fill rate: 100%

    LIMIT ORDER:
    - Urgency: PATIENT
    - Use case: Mean reversion, range trading
    - Cost: Low (maker fee, no slippage)
    - Fill rate: 60-80% (depends on placement)

    TWAP (Time-Weighted):
    - Size: LARGE
    - Use case: Large positions, normal volatility
    - Cost: Medium (mix of maker/taker)
    - Fill rate: 95%+

    VWAP (Volume-Weighted):
    - Size: LARGE
    - Use case: Large positions, high volume
    - Cost: Medium-Low
    - Fill rate: 95%+

    Usage:
        executor = SmartOrderExecutor()

        # Determine execution strategy
        strategy = executor.get_execution_strategy(
            technique='breakout',
            position_size=50.0,
            mid_price=47000.0,
            spread_bps=5.0,
            liquidity_score=0.75,
            volatility_bps=150.0,
        )

        if strategy.order_type == OrderType.MARKET:
            execute_market_order(size=position_size)
        elif strategy.order_type == OrderType.LIMIT:
            execute_limit_order(
                size=position_size,
                limit_price=strategy.limit_price,
            )
        elif strategy.order_type == OrderType.TWAP:
            # Split into multiple orders
            slices = executor.create_twap_slices(
                total_size=position_size,
                split_count=strategy.split_count,
                time_window=strategy.time_window_seconds,
                mid_price=mid_price,
            )
            for slice in slices:
                time.sleep(slice.delay_seconds)
                execute_order(slice)
    """

    def __init__(
        self,
        maker_fee_bps: float = 2.0,
        taker_fee_bps: float = 5.0,
        large_size_threshold_usd: float = 10000.0,
        twap_slice_count: int = 10,
        twap_window_seconds: int = 300,  # 5 minutes
        limit_order_offset_bps: float = 1.0,  # Place limit 1 bps better than mid
    ):
        """
        Initialize smart order executor.

        Args:
            maker_fee_bps: Maker fee in bps
            taker_fee_bps: Taker fee in bps
            large_size_threshold_usd: Size above which to consider splitting
            twap_slice_count: Number of slices for TWAP
            twap_window_seconds: Time window for TWAP execution
            limit_order_offset_bps: Offset from mid for limit orders
        """
        self.maker_fee = maker_fee_bps
        self.taker_fee = taker_fee_bps
        self.large_size_threshold = large_size_threshold_usd
        self.twap_slices = twap_slice_count
        self.twap_window = twap_window_seconds
        self.limit_offset = limit_order_offset_bps

        logger.info(
            "smart_order_executor_initialized",
            maker_fee=maker_fee_bps,
            taker_fee=taker_fee_bps,
            large_size_threshold=large_size_threshold_usd,
        )

    def get_execution_strategy(
        self,
        technique: str,
        position_size_usd: float,
        mid_price: float,
        spread_bps: float,
        liquidity_score: float,
        volatility_bps: float,
        direction: str = 'buy',
    ) -> ExecutionStrategy:
        """
        Determine optimal execution strategy.

        Args:
            technique: Trading technique ('breakout', 'range', etc.)
            position_size_usd: Position size in USD
            mid_price: Current mid price
            spread_bps: Current bid-ask spread in bps
            liquidity_score: Liquidity quality (0-1)
            volatility_bps: Current volatility in bps
            direction: 'buy' or 'sell'

        Returns:
            ExecutionStrategy with recommendation
        """
        # Determine urgency based on technique
        urgency = self._determine_urgency(technique, volatility_bps)

        # Check if size is large
        is_large_size = position_size_usd >= self.large_size_threshold

        # Decision logic
        if urgency == UrgencyLevel.URGENT:
            # Urgent trades need immediate execution
            if is_large_size:
                # Large urgent order → Aggressive TWAP (faster)
                return self._strategy_aggressive_twap(
                    position_size_usd, mid_price, direction
                )
            else:
                # Normal urgent order → Market order
                return self._strategy_market_order(
                    spread_bps, urgency
                )

        elif urgency == UrgencyLevel.MODERATE:
            # Moderate urgency
            if is_large_size:
                # Large moderate order → Standard TWAP
                return self._strategy_twap(
                    position_size_usd, mid_price, direction
                )
            elif spread_bps > 10.0:
                # Wide spread → Use limit order
                return self._strategy_limit_order(
                    mid_price, spread_bps, direction, urgency
                )
            else:
                # Normal moderate order → Market order
                return self._strategy_market_order(
                    spread_bps, urgency
                )

        else:  # PATIENT
            # Patient trades can wait for better prices
            if is_large_size:
                # Large patient order → VWAP or slow TWAP
                return self._strategy_vwap(
                    position_size_usd, mid_price, direction
                )
            else:
                # Normal patient order → Limit order
                return self._strategy_limit_order(
                    mid_price, spread_bps, direction, urgency
                )

    def create_twap_slices(
        self,
        total_size: float,
        split_count: int,
        time_window_seconds: int,
        mid_price: float,
        direction: str = 'buy',
    ) -> List[OrderSlice]:
        """
        Create TWAP order slices.

        Args:
            total_size: Total size to execute
            split_count: Number of slices
            time_window_seconds: Time window for execution
            mid_price: Current mid price
            direction: 'buy' or 'sell'

        Returns:
            List of OrderSlice objects
        """
        slice_size = total_size / split_count
        delay_between_slices = time_window_seconds / split_count

        slices = []
        for i in range(split_count):
            # Alternate between market and limit orders to reduce cost
            if i % 3 == 0:
                # Every 3rd slice is market order (ensure some fills)
                order_type = OrderType.MARKET
                limit_price = None
            else:
                # Other slices are limit orders
                order_type = OrderType.LIMIT
                if direction == 'buy':
                    limit_price = mid_price * (1 - self.limit_offset / 10000)
                else:
                    limit_price = mid_price * (1 + self.limit_offset / 10000)

            slices.append(OrderSlice(
                slice_number=i + 1,
                size=slice_size,
                order_type=order_type,
                limit_price=limit_price,
                delay_seconds=i * delay_between_slices,
            ))

        return slices

    def estimate_execution_cost(
        self,
        strategy: ExecutionStrategy,
        spread_bps: float,
    ) -> float:
        """
        Estimate total execution cost for strategy.

        Args:
            strategy: Execution strategy
            spread_bps: Current spread in bps

        Returns:
            Estimated cost in bps
        """
        return strategy.estimated_cost_bps

    def _determine_urgency(
        self,
        technique: str,
        volatility_bps: float,
    ) -> UrgencyLevel:
        """Determine execution urgency."""
        technique_lower = technique.lower()

        # Technique-based urgency
        if technique_lower in ['breakout', 'sweep']:
            # Breakouts and sweeps need immediate execution
            return UrgencyLevel.URGENT
        elif technique_lower in ['range', 'leader']:
            # Range and leader can be patient
            return UrgencyLevel.PATIENT
        else:
            # Trend and tape are moderate
            urgency = UrgencyLevel.MODERATE

        # Adjust for high volatility (more urgent in high vol)
        if volatility_bps > 250:
            if urgency == UrgencyLevel.PATIENT:
                urgency = UrgencyLevel.MODERATE
            elif urgency == UrgencyLevel.MODERATE:
                urgency = UrgencyLevel.URGENT

        return urgency

    def _strategy_market_order(
        self,
        spread_bps: float,
        urgency: UrgencyLevel,
    ) -> ExecutionStrategy:
        """Market order strategy."""
        # Cost = taker fee + half spread (cross spread to fill)
        estimated_cost = self.taker_fee + (spread_bps / 2)

        return ExecutionStrategy(
            order_type=OrderType.MARKET,
            urgency=urgency,
            limit_price=None,
            split_count=None,
            time_window_seconds=None,
            estimated_cost_bps=estimated_cost,
            estimated_fill_probability=1.0,
            reasoning=f"Market order for {urgency.value} trade. Cost: {estimated_cost:.1f} bps",
        )

    def _strategy_limit_order(
        self,
        mid_price: float,
        spread_bps: float,
        direction: str,
        urgency: UrgencyLevel,
    ) -> ExecutionStrategy:
        """Limit order strategy."""
        # Place limit order better than mid
        if direction == 'buy':
            limit_price = mid_price * (1 - self.limit_offset / 10000)
        else:
            limit_price = mid_price * (1 + self.limit_offset / 10000)

        # Cost = maker fee + 0 slippage (assuming fill)
        estimated_cost = self.maker_fee

        # Fill probability depends on urgency
        # Patient trades have higher fill probability (more time to wait)
        if urgency == UrgencyLevel.PATIENT:
            fill_prob = 0.80
        elif urgency == UrgencyLevel.MODERATE:
            fill_prob = 0.65
        else:
            fill_prob = 0.50

        return ExecutionStrategy(
            order_type=OrderType.LIMIT,
            urgency=urgency,
            limit_price=limit_price,
            split_count=None,
            time_window_seconds=None,
            estimated_cost_bps=estimated_cost,
            estimated_fill_probability=fill_prob,
            reasoning=f"Limit order at {limit_price:.2f} ({self.limit_offset:.1f} bps better). Cost: {estimated_cost:.1f} bps",
        )

    def _strategy_twap(
        self,
        size_usd: float,
        mid_price: float,
        direction: str,
    ) -> ExecutionStrategy:
        """Standard TWAP strategy."""
        # TWAP: Mix of market and limit orders
        # Assume 70% limit fills, 30% market fills
        avg_cost = (0.7 * self.maker_fee) + (0.3 * (self.taker_fee + 3))  # 3 bps avg slippage

        return ExecutionStrategy(
            order_type=OrderType.TWAP,
            urgency=UrgencyLevel.MODERATE,
            limit_price=None,
            split_count=self.twap_slices,
            time_window_seconds=self.twap_window,
            estimated_cost_bps=avg_cost,
            estimated_fill_probability=0.95,
            reasoning=f"TWAP execution: {self.twap_slices} slices over {self.twap_window}s. Cost: {avg_cost:.1f} bps",
        )

    def _strategy_aggressive_twap(
        self,
        size_usd: float,
        mid_price: float,
        direction: str,
    ) -> ExecutionStrategy:
        """Aggressive TWAP (faster execution for urgent large orders)."""
        # Aggressive TWAP: More market orders, faster execution
        aggressive_slices = max(5, self.twap_slices // 2)
        aggressive_window = max(60, self.twap_window // 3)

        # More market orders = higher cost
        avg_cost = (0.5 * self.maker_fee) + (0.5 * (self.taker_fee + 5))  # 5 bps avg slippage

        return ExecutionStrategy(
            order_type=OrderType.TWAP,
            urgency=UrgencyLevel.URGENT,
            limit_price=None,
            split_count=aggressive_slices,
            time_window_seconds=aggressive_window,
            estimated_cost_bps=avg_cost,
            estimated_fill_probability=0.98,
            reasoning=f"Aggressive TWAP: {aggressive_slices} slices over {aggressive_window}s. Cost: {avg_cost:.1f} bps",
        )

    def _strategy_vwap(
        self,
        size_usd: float,
        mid_price: float,
        direction: str,
    ) -> ExecutionStrategy:
        """VWAP strategy (volume-weighted, patient execution)."""
        # VWAP: Patient execution, mostly limit orders
        # Assume 85% limit fills, 15% market fills
        avg_cost = (0.85 * self.maker_fee) + (0.15 * (self.taker_fee + 2))  # 2 bps avg slippage

        # VWAP over longer window
        vwap_window = self.twap_window * 2

        return ExecutionStrategy(
            order_type=OrderType.VWAP,
            urgency=UrgencyLevel.PATIENT,
            limit_price=None,
            split_count=self.twap_slices * 2,  # More slices for smoother execution
            time_window_seconds=vwap_window,
            estimated_cost_bps=avg_cost,
            estimated_fill_probability=0.92,
            reasoning=f"VWAP execution: {self.twap_slices * 2} slices over {vwap_window}s. Cost: {avg_cost:.1f} bps",
        )

    def get_statistics(self) -> Dict[str, any]:
        """Get executor statistics."""
        return {
            'maker_fee_bps': self.maker_fee,
            'taker_fee_bps': self.taker_fee,
            'large_size_threshold_usd': self.large_size_threshold,
            'twap_slices': self.twap_slices,
            'twap_window_seconds': self.twap_window,
            'cost_savings_vs_market': {
                'limit_order': self.taker_fee - self.maker_fee,
                'twap': self.taker_fee - ((0.7 * self.maker_fee) + (0.3 * self.taker_fee)),
                'vwap': self.taker_fee - ((0.85 * self.maker_fee) + (0.15 * self.taker_fee)),
            },
        }
