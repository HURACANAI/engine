"""
Liquidity Depth Analyzer

Pre-trade liquidity validation to ensure sufficient order book depth for execution.

Key Problems Solved:
1. **Blind Execution**: Entering 50 BTC trade when book only has 10 BTC liquidity
2. **Slippage Surprise**: Expected fill at $47k, actual fill at $46.8k due to thin book
3. **Exit Risk**: Can't exit large position without massive slippage

Solution: Pre-Trade Liquidity Checks
- Validate order book depth before entry
- Estimate execution cost (slippage) for given size
- Scale down position if insufficient liquidity
- Alert on liquidity crises

Example:
    Proposed Trade: Buy 50 BTC at market

    Order Book Analysis:
    Bid Depth within 20 bps: 12 BTC
    Ask Depth within 20 bps: 45 BTC

    Validation:
    - Entry: 45 BTC available (GOOD - can fill 50 BTC with 11% slippage)
    - Exit: Only 12 BTC bid liquidity (BAD - can't exit cleanly!)

    Recommendation: REDUCE SIZE to 12 BTC or use limit orders
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class LiquidityLevel:
    """Liquidity at a specific distance from mid."""

    distance_bps: float  # Distance from mid price in bps
    cumulative_size: float  # Cumulative size available
    avg_price: float  # Volume-weighted average price
    num_levels: int  # Number of order book levels


@dataclass
class ExecutionCostEstimate:
    """Estimated execution cost for given size."""

    requested_size: float
    fillable_size: float  # How much can actually be filled
    fill_percentage: float  # % of requested size fillable
    avg_fill_price: float  # Expected average fill price
    mid_price: float  # Current mid price
    slippage_bps: float  # Expected slippage in bps
    total_cost_bps: float  # Total cost including fees

    is_acceptable: bool  # Whether execution is acceptable
    recommendation: str  # 'EXECUTE', 'REDUCE_SIZE', 'USE_LIMIT_ORDER', 'ABORT'


@dataclass
class LiquidityValidationResult:
    """Result of liquidity validation."""

    symbol: str
    direction: str  # 'buy' or 'sell'
    requested_size: float

    # Entry liquidity
    entry_estimate: ExecutionCostEstimate

    # Exit liquidity (opposite side)
    exit_estimate: ExecutionCostEstimate

    # Overall assessment
    liquidity_score: float  # 0-1, 1 = excellent liquidity
    is_valid: bool  # Whether trade should proceed
    recommended_size: float  # Size adjusted for liquidity
    warnings: List[str]


class LiquidityDepthAnalyzer:
    """
    Analyzes order book depth for pre-trade liquidity validation.

    Key Functions:
    1. Depth Analysis: How much size available within X bps of mid
    2. Execution Cost: Estimate slippage for given size
    3. Entry/Exit Validation: Check both entry AND exit liquidity
    4. Size Recommendation: Reduce size if insufficient liquidity

    Usage:
        analyzer = LiquidityDepthAnalyzer(
            max_slippage_bps=20.0,
            min_liquidity_ratio=0.80,
        )

        # Validate trade
        result = analyzer.validate_trade(
            symbol='BTC',
            direction='buy',
            requested_size=50.0,
            order_book=current_orderbook,
            mid_price=47000.0,
        )

        if not result.is_valid:
            logger.warning(f"Insufficient liquidity: {result.warnings}")
            if result.recommended_size < requested_size * 0.5:
                # Less than 50% fillable - skip trade
                skip_trade()
            else:
                # Reduce size to recommended
                execute_trade(size=result.recommended_size)
    """

    def __init__(
        self,
        max_slippage_bps: float = 20.0,
        max_total_cost_bps: float = 30.0,
        min_liquidity_ratio: float = 0.80,
        maker_fee_bps: float = 2.0,
        taker_fee_bps: float = 5.0,
        liquidity_check_distances: List[float] = None,
    ):
        """
        Initialize liquidity depth analyzer.

        Args:
            max_slippage_bps: Maximum acceptable slippage
            max_total_cost_bps: Maximum total cost (slippage + fees)
            min_liquidity_ratio: Minimum fillable ratio (0.80 = need 80% fillable)
            maker_fee_bps: Maker fee in bps
            taker_fee_bps: Taker fee in bps
            liquidity_check_distances: Distances to check (default: [5, 10, 20, 50])
        """
        self.max_slippage = max_slippage_bps
        self.max_total_cost = max_total_cost_bps
        self.min_liquidity_ratio = min_liquidity_ratio
        self.maker_fee = maker_fee_bps
        self.taker_fee = taker_fee_bps

        if liquidity_check_distances is None:
            self.check_distances = [5.0, 10.0, 20.0, 50.0]
        else:
            self.check_distances = sorted(liquidity_check_distances)

        logger.info(
            "liquidity_depth_analyzer_initialized",
            max_slippage=max_slippage_bps,
            min_liquidity_ratio=min_liquidity_ratio,
        )

    def validate_trade(
        self,
        symbol: str,
        direction: str,
        requested_size: float,
        order_book: Dict[str, List[Tuple[float, float]]],
        mid_price: float,
    ) -> LiquidityValidationResult:
        """
        Validate trade liquidity.

        Args:
            symbol: Trading symbol
            direction: 'buy' or 'sell'
            requested_size: Requested position size
            order_book: {'bids': [(price, size), ...], 'asks': [(price, size), ...]}
            mid_price: Current mid price

        Returns:
            LiquidityValidationResult with validation outcome
        """
        # Estimate entry execution cost
        if direction == 'buy':
            entry_side = 'asks'  # Buy from asks
            exit_side = 'bids'  # Exit to bids
        else:
            entry_side = 'bids'  # Sell to bids
            exit_side = 'asks'  # Exit from asks

        entry_estimate = self._estimate_execution_cost(
            requested_size=requested_size,
            order_book_side=order_book[entry_side],
            mid_price=mid_price,
            direction=direction,
        )

        # Estimate exit execution cost (on opposite side)
        exit_direction = 'sell' if direction == 'buy' else 'buy'
        exit_estimate = self._estimate_execution_cost(
            requested_size=requested_size,
            order_book_side=order_book[exit_side],
            mid_price=mid_price,
            direction=exit_direction,
        )

        # Calculate liquidity score (0-1)
        # Factors: entry fillability, exit fillability, slippage
        entry_fill_score = entry_estimate.fill_percentage
        exit_fill_score = exit_estimate.fill_percentage
        entry_slippage_score = max(0, 1 - (entry_estimate.slippage_bps / 50.0))
        exit_slippage_score = max(0, 1 - (exit_estimate.slippage_bps / 50.0))

        liquidity_score = (
            entry_fill_score * 0.35 +
            exit_fill_score * 0.35 +
            entry_slippage_score * 0.15 +
            exit_slippage_score * 0.15
        )

        # Determine if valid
        is_valid = True
        warnings = []

        if entry_estimate.fill_percentage < self.min_liquidity_ratio:
            is_valid = False
            warnings.append(
                f"Insufficient entry liquidity: Only {entry_estimate.fill_percentage:.0%} "
                f"of size fillable ({entry_estimate.fillable_size:.2f}/{requested_size:.2f})"
            )

        if exit_estimate.fill_percentage < self.min_liquidity_ratio:
            is_valid = False
            warnings.append(
                f"Insufficient exit liquidity: Only {exit_estimate.fill_percentage:.0%} "
                f"fillable on exit"
            )

        if entry_estimate.slippage_bps > self.max_slippage:
            is_valid = False
            warnings.append(
                f"Entry slippage too high: {entry_estimate.slippage_bps:.1f} bps "
                f"(max: {self.max_slippage:.1f} bps)"
            )

        if entry_estimate.total_cost_bps > self.max_total_cost:
            is_valid = False
            warnings.append(
                f"Total entry cost too high: {entry_estimate.total_cost_bps:.1f} bps "
                f"(max: {self.max_total_cost:.1f} bps)"
            )

        # Recommend size adjustment
        # Use minimum of entry and exit fillable size
        recommended_size = min(
            entry_estimate.fillable_size,
            exit_estimate.fillable_size,
        )

        # If recommended size is very small, warn
        if recommended_size < requested_size * 0.3:
            warnings.append(
                f"Severe liquidity constraint: Can only fill {recommended_size:.2f} "
                f"of requested {requested_size:.2f}"
            )

        logger.info(
            "liquidity_validated",
            symbol=symbol,
            direction=direction,
            requested=requested_size,
            recommended=recommended_size,
            liquidity_score=liquidity_score,
            is_valid=is_valid,
        )

        return LiquidityValidationResult(
            symbol=symbol,
            direction=direction,
            requested_size=requested_size,
            entry_estimate=entry_estimate,
            exit_estimate=exit_estimate,
            liquidity_score=liquidity_score,
            is_valid=is_valid,
            recommended_size=recommended_size,
            warnings=warnings,
        )

    def get_liquidity_depth(
        self,
        order_book: Dict[str, List[Tuple[float, float]]],
        mid_price: float,
        side: str = 'both',
    ) -> Dict[str, Dict[float, LiquidityLevel]]:
        """
        Get liquidity depth at various distances from mid.

        Args:
            order_book: Order book data
            mid_price: Current mid price
            side: 'bids', 'asks', or 'both'

        Returns:
            Dict of {side: {distance_bps: LiquidityLevel}}
        """
        result = {}

        sides_to_check = []
        if side in ['bids', 'both']:
            sides_to_check.append('bids')
        if side in ['asks', 'both']:
            sides_to_check.append('asks')

        for book_side in sides_to_check:
            side_liquidity = {}

            for distance in self.check_distances:
                liquidity_level = self._calculate_liquidity_at_distance(
                    order_book_side=order_book[book_side],
                    mid_price=mid_price,
                    distance_bps=distance,
                    is_bid=(book_side == 'bids'),
                )
                side_liquidity[distance] = liquidity_level

            result[book_side] = side_liquidity

        return result

    def _estimate_execution_cost(
        self,
        requested_size: float,
        order_book_side: List[Tuple[float, float]],
        mid_price: float,
        direction: str,
    ) -> ExecutionCostEstimate:
        """Estimate execution cost for given size."""
        if not order_book_side:
            # Empty book - can't fill
            return ExecutionCostEstimate(
                requested_size=requested_size,
                fillable_size=0.0,
                fill_percentage=0.0,
                avg_fill_price=mid_price,
                mid_price=mid_price,
                slippage_bps=999.9,
                total_cost_bps=999.9,
                is_acceptable=False,
                recommendation='ABORT',
            )

        # Simulate filling through order book
        remaining_size = requested_size
        total_cost = 0.0
        filled_size = 0.0
        levels_used = 0

        for price, size in order_book_side:
            if remaining_size <= 0:
                break

            fill_at_level = min(remaining_size, size)
            total_cost += fill_at_level * price
            filled_size += fill_at_level
            remaining_size -= fill_at_level
            levels_used += 1

            # Stop if we've gone too far from mid (e.g., 100 bps)
            distance_bps = abs((price - mid_price) / mid_price) * 10000
            if distance_bps > 100:
                break

        if filled_size == 0:
            # Couldn't fill anything
            return ExecutionCostEstimate(
                requested_size=requested_size,
                fillable_size=0.0,
                fill_percentage=0.0,
                avg_fill_price=mid_price,
                mid_price=mid_price,
                slippage_bps=999.9,
                total_cost_bps=999.9,
                is_acceptable=False,
                recommendation='ABORT',
            )

        # Calculate metrics
        avg_fill_price = total_cost / filled_size
        fill_percentage = filled_size / requested_size

        # Calculate slippage
        if direction == 'buy':
            slippage_bps = ((avg_fill_price - mid_price) / mid_price) * 10000
        else:  # sell
            slippage_bps = ((mid_price - avg_fill_price) / mid_price) * 10000

        # Total cost including fees (assume taker fee for market orders)
        total_cost_bps = slippage_bps + self.taker_fee

        # Determine if acceptable
        is_acceptable = (
            fill_percentage >= self.min_liquidity_ratio and
            slippage_bps <= self.max_slippage and
            total_cost_bps <= self.max_total_cost
        )

        # Recommendation
        if is_acceptable:
            recommendation = 'EXECUTE'
        elif fill_percentage >= 0.5:
            recommendation = 'REDUCE_SIZE'
        elif slippage_bps > self.max_slippage * 0.5:
            recommendation = 'USE_LIMIT_ORDER'
        else:
            recommendation = 'ABORT'

        return ExecutionCostEstimate(
            requested_size=requested_size,
            fillable_size=filled_size,
            fill_percentage=fill_percentage,
            avg_fill_price=avg_fill_price,
            mid_price=mid_price,
            slippage_bps=slippage_bps,
            total_cost_bps=total_cost_bps,
            is_acceptable=is_acceptable,
            recommendation=recommendation,
        )

    def _calculate_liquidity_at_distance(
        self,
        order_book_side: List[Tuple[float, float]],
        mid_price: float,
        distance_bps: float,
        is_bid: bool,
    ) -> LiquidityLevel:
        """Calculate cumulative liquidity within distance from mid."""
        cumulative_size = 0.0
        total_value = 0.0
        num_levels = 0

        # Calculate price threshold
        if is_bid:
            # Bids: price threshold = mid * (1 - distance/10000)
            price_threshold = mid_price * (1 - distance_bps / 10000)
            # Count bids >= threshold
            for price, size in order_book_side:
                if price >= price_threshold:
                    cumulative_size += size
                    total_value += price * size
                    num_levels += 1
        else:
            # Asks: price threshold = mid * (1 + distance/10000)
            price_threshold = mid_price * (1 + distance_bps / 10000)
            # Count asks <= threshold
            for price, size in order_book_side:
                if price <= price_threshold:
                    cumulative_size += size
                    total_value += price * size
                    num_levels += 1

        # Calculate volume-weighted average price
        avg_price = total_value / cumulative_size if cumulative_size > 0 else mid_price

        return LiquidityLevel(
            distance_bps=distance_bps,
            cumulative_size=cumulative_size,
            avg_price=avg_price,
            num_levels=num_levels,
        )

    def detect_liquidity_crisis(
        self,
        order_book: Dict[str, List[Tuple[float, float]]],
        mid_price: float,
        normal_liquidity_baseline: float,
    ) -> Optional[str]:
        """
        Detect liquidity crisis (abnormally low liquidity).

        Args:
            order_book: Current order book
            mid_price: Current mid price
            normal_liquidity_baseline: Normal liquidity size (e.g., avg of last 100 candles)

        Returns:
            Warning message if crisis detected, None otherwise
        """
        # Check liquidity within 20 bps on both sides
        liquidity_20bps = self.get_liquidity_depth(order_book, mid_price, side='both')

        bid_liquidity = liquidity_20bps['bids'][20.0].cumulative_size
        ask_liquidity = liquidity_20bps['asks'][20.0].cumulative_size

        total_liquidity = bid_liquidity + ask_liquidity

        # Check if below 50% of normal
        if total_liquidity < normal_liquidity_baseline * 0.5:
            return (
                f"LIQUIDITY CRISIS: Only {total_liquidity:.2f} within 20 bps "
                f"(normal: {normal_liquidity_baseline:.2f}). "
                f"Bid: {bid_liquidity:.2f}, Ask: {ask_liquidity:.2f}"
            )

        # Check for severe imbalance (>80% one-sided)
        if bid_liquidity > 0 and ask_liquidity > 0:
            imbalance_ratio = max(bid_liquidity, ask_liquidity) / (bid_liquidity + ask_liquidity)
            if imbalance_ratio > 0.80:
                dominant_side = 'BID' if bid_liquidity > ask_liquidity else 'ASK'
                return (
                    f"LIQUIDITY IMBALANCE: {imbalance_ratio:.0%} {dominant_side}-sided. "
                    f"Potential manipulation or one-way market."
                )

        return None
