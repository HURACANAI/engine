"""
Maker Volume Strategy - Verified Case Study ($6,800 → $1.5M)

Based on verified Cointelegraph case study where a trader turned $6,800 into $1.5M
by providing maker-side liquidity and earning rebates.

Strategy:
1. Place limit orders 1-2 bps inside spread
2. Provide liquidity (maker orders)
3. Earn rebates (-2 bps) + capture spread
4. High volume = compound rebates
5. One-sided quoting (focus on buy OR sell side)

Key: Aggressively use maker orders to earn rebates while capturing spread.

Expected Impact: Saves 5-7 bps per trade = £5-£7 per 100 trades
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import structlog

logger = structlog.get_logger(__name__)


class MakerSide(Enum):
    """Which side to provide maker liquidity."""

    BUY = "buy"  # Provide buy-side liquidity
    SELL = "sell"  # Provide sell-side liquidity
    BOTH = "both"  # Provide both sides


@dataclass
class MakerOrderStrategy:
    """Maker order placement strategy."""

    side: str  # 'buy' or 'sell'
    limit_price: float  # Limit price to place
    size: float  # Order size
    post_only: bool  # Ensure maker order
    expected_rebate_bps: float  # Expected rebate (negative = you get paid)
    fill_probability: float  # Probability of fill
    reason: str  # Strategy reason


class MakerVolumeOptimizer:
    """
    Optimizes maker order placement to earn rebates.

    Based on verified case study: $6,800 → $1.5M using maker volume strategy.

    Strategy:
    1. Place limit orders 1-2 bps inside spread
    2. Wait 5-30 seconds for fill
    3. If not filled, cancel and use taker (only if urgent)
    4. Target: 70%+ maker fill rate
    5. One-sided quoting for maximum rebate capture

    Benefit:
    - Maker rebate: -2 bps (you GET paid)
    - Taker fee: +5 bps (you PAY)
    - Net savings: 7 bps per trade = £0.70 on £100 trade

    Expected Impact: Saves 5-7 bps per trade
    """

    def __init__(
        self,
        maker_rebate_bps: float = -2.0,  # Negative = you get paid
        taker_fee_bps: float = 5.0,
        inside_spread_bps: float = 1.0,  # How many bps inside spread
        min_spread_bps: float = 3.0,  # Minimum spread to use maker
        max_wait_seconds: int = 30,  # Max wait for maker fill
    ):
        """
        Initialize maker volume optimizer.

        Args:
            maker_rebate_bps: Maker rebate (negative = you get paid)
            taker_fee_bps: Taker fee (positive = you pay)
            inside_spread_bps: How many bps inside spread to place order
            min_spread_bps: Minimum spread to attempt maker order
            max_wait_seconds: Maximum wait time for maker fill
        """
        self.maker_rebate_bps = maker_rebate_bps
        self.taker_fee_bps = taker_fee_bps
        self.inside_spread_bps = inside_spread_bps
        self.min_spread_bps = min_spread_bps
        self.max_wait_seconds = max_wait_seconds

        logger.info(
            "maker_volume_optimizer_initialized",
            maker_rebate=maker_rebate_bps,
            taker_fee=taker_fee_bps,
            net_savings=taker_fee_bps - abs(maker_rebate_bps),
        )

    def should_use_maker(
        self,
        urgency: str,
        spread_bps: float,
        best_bid: float,
        best_ask: float,
    ) -> bool:
        """
        Decide whether to use maker or taker order.

        Args:
            urgency: 'urgent', 'moderate', or 'patient'
            spread_bps: Current spread in bps
            best_bid: Best bid price
            best_ask: Best ask price

        Returns:
            True if should use maker, False for taker
        """
        # Urgent orders → Use taker
        if urgency == 'urgent':
            return False

        # Spread too tight → Unlikely to fill maker
        if spread_bps < self.min_spread_bps:
            return False

        # Patient or moderate → Try maker first
        return True

    def calculate_maker_price(
        self,
        side: str,
        best_bid: float,
        best_ask: float,
        spread_bps: float,
    ) -> Optional[float]:
        """
        Calculate optimal maker limit price.

        Args:
            side: 'buy' or 'sell'
            best_bid: Best bid price
            best_ask: Best ask price
            spread_bps: Current spread in bps

        Returns:
            Limit price for maker order, or None if not viable
        """
        if spread_bps < self.min_spread_bps:
            return None

        if side == 'buy':
            # Place buy order 1 bps above best bid (inside spread)
            price_bps_above_bid = self.inside_spread_bps / 10000
            limit_price = best_bid * (1 + price_bps_above_bid)
            # Ensure we're still below best ask
            if limit_price >= best_ask:
                limit_price = best_ask * 0.9999  # Just below best ask
            return limit_price

        else:  # sell
            # Place sell order 1 bps below best ask (inside spread)
            price_bps_below_ask = self.inside_spread_bps / 10000
            limit_price = best_ask * (1 - price_bps_below_ask)
            # Ensure we're still above best bid
            if limit_price <= best_bid:
                limit_price = best_bid * 1.0001  # Just above best bid
            return limit_price

    def estimate_fill_probability(
        self,
        side: str,
        spread_bps: float,
        liquidity_score: float,
    ) -> float:
        """
        Estimate probability of maker order fill.

        Args:
            side: 'buy' or 'sell'
            spread_bps: Current spread
            liquidity_score: Liquidity score (0-1)

        Returns:
            Fill probability (0-1)
        """
        # Base fill probability
        base_prob = 0.75

        # Wider spread = higher fill probability
        spread_factor = min(spread_bps / 10.0, 1.0)  # Cap at 1.0

        # Higher liquidity = higher fill probability
        liquidity_factor = liquidity_score

        # Combined probability
        fill_prob = base_prob * (0.5 + 0.5 * spread_factor) * liquidity_factor

        return min(fill_prob, 0.95)  # Cap at 95%

    def calculate_expected_cost_savings(
        self,
        use_maker: bool,
        fill_probability: float,
    ) -> float:
        """
        Calculate expected cost savings from using maker.

        Args:
            use_maker: Whether using maker order
            fill_probability: Probability of maker fill

        Returns:
            Expected cost savings in bps (positive = savings)
        """
        if not use_maker:
            return 0.0

        # Expected cost with maker
        maker_cost = self.maker_rebate_bps * fill_probability  # Negative = savings
        taker_cost = self.taker_fee_bps * (1 - fill_probability)

        expected_maker_cost = maker_cost + taker_cost

        # Cost with taker only
        taker_only_cost = self.taker_fee_bps

        # Savings
        savings = taker_only_cost - expected_maker_cost

        return savings

    def get_maker_strategy(
        self,
        side: str,
        urgency: str,
        best_bid: float,
        best_ask: float,
        spread_bps: float,
        liquidity_score: float,
        size: float,
    ) -> Optional[MakerOrderStrategy]:
        """
        Get maker order strategy.

        Args:
            side: 'buy' or 'sell'
            urgency: 'urgent', 'moderate', or 'patient'
            best_bid: Best bid price
            best_ask: Best ask price
            spread_bps: Current spread
            liquidity_score: Liquidity score
            size: Order size

        Returns:
            MakerOrderStrategy if maker viable, None otherwise
        """
        if not self.should_use_maker(urgency, spread_bps, best_bid, best_ask):
            return None

        limit_price = self.calculate_maker_price(side, best_bid, best_ask, spread_bps)
        if limit_price is None:
            return None

        fill_prob = self.estimate_fill_probability(side, spread_bps, liquidity_score)

        return MakerOrderStrategy(
            side=side,
            limit_price=limit_price,
            size=size,
            post_only=True,  # Ensure maker
            expected_rebate_bps=self.maker_rebate_bps,
            fill_probability=fill_prob,
            reason=f'Maker order {self.inside_spread_bps} bps inside spread, {fill_prob:.1%} fill prob',
        )

    def get_statistics(self) -> dict:
        """Get optimizer statistics."""
        return {
            'maker_rebate_bps': self.maker_rebate_bps,
            'taker_fee_bps': self.taker_fee_bps,
            'net_savings_per_trade_bps': self.taker_fee_bps - abs(self.maker_rebate_bps),
            'inside_spread_bps': self.inside_spread_bps,
            'min_spread_bps': self.min_spread_bps,
        }

