"""
Order Book Depth Analysis

Analyzes order book structure to detect:
- Liquidity imbalances (bid vs ask)
- Support/resistance levels (large orders)
- Spoofing/manipulation patterns
- Optimal execution levels

Traditional: Trade at market price
Microstructure: Understand book depth, find best execution price

Example:
Price | Bid Size | Ask Size | Imbalance
45050 |        0 |     10.5 | -10.5 (heavy selling)
45049 |        0 |      8.2 | -8.2
45048 |      2.1 |        0 | +2.1
45047 |     15.8 |        0 | +15.8 (support level!)
→ Strong support at 45047, wait for pullback to that level
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class OrderBookLevel:
    """Single price level in order book."""

    price: float
    bid_size: float
    ask_size: float
    bid_orders: int  # Number of bid orders at this level
    ask_orders: int  # Number of ask orders at this level


@dataclass
class OrderBookSnapshot:
    """Complete order book snapshot."""

    timestamp: float
    symbol: str
    bids: List[OrderBookLevel]  # Sorted descending by price
    asks: List[OrderBookLevel]  # Sorted ascending by price
    mid_price: float
    spread_bps: float


@dataclass
class BookImbalance:
    """Order book imbalance metrics."""

    imbalance_ratio: float  # Bid size / Total size (-1 to +1)
    weighted_imbalance: float  # Volume-weighted imbalance
    depth_imbalance: float  # Deep book imbalance (5+ levels)
    momentum_score: float  # Directional momentum from imbalance
    support_level: Optional[float]  # Strong support price
    resistance_level: Optional[float]  # Strong resistance price
    predicted_direction: str  # "UP", "DOWN", "NEUTRAL"


@dataclass
class LiquidityMetrics:
    """Order book liquidity metrics."""

    total_bid_liquidity: float  # Total bid size
    total_ask_liquidity: float  # Total ask size
    effective_spread_bps: float  # Effective spread for given size
    market_depth_score: float  # 0-1, how deep is the book
    liquidity_score: float  # 0-1, overall liquidity quality
    is_liquid: bool  # Above minimum threshold


class OrderBookAnalyzer:
    """
    Analyze order book structure for trading signals.

    Detects:
    - Liquidity imbalances (buying vs selling pressure)
    - Support/resistance levels (large orders)
    - Market depth and execution quality
    - Optimal entry/exit prices
    """

    def __init__(
        self,
        min_liquidity_score: float = 0.5,
        imbalance_threshold: float = 0.3,
        max_levels: int = 20,
    ):
        """
        Initialize order book analyzer.

        Args:
            min_liquidity_score: Minimum liquidity to trade (0-1)
            imbalance_threshold: Imbalance threshold for signals
            max_levels: Maximum levels to analyze
        """
        self.min_liquidity_score = min_liquidity_score
        self.imbalance_threshold = imbalance_threshold
        self.max_levels = max_levels

        # Track historical imbalances
        self.imbalance_history: List[float] = []

        logger.info(
            "orderbook_analyzer_initialized",
            min_liquidity=min_liquidity_score,
            imbalance_threshold=imbalance_threshold,
        )

    def analyze_book(self, snapshot: OrderBookSnapshot) -> Tuple[BookImbalance, LiquidityMetrics]:
        """
        Analyze order book snapshot.

        Args:
            snapshot: Order book snapshot

        Returns:
            (imbalance_metrics, liquidity_metrics)
        """
        # Calculate imbalances
        imbalance = self._calculate_imbalance(snapshot)

        # Calculate liquidity
        liquidity = self._calculate_liquidity(snapshot)

        # Update history
        self.imbalance_history.append(imbalance.imbalance_ratio)
        if len(self.imbalance_history) > 100:
            self.imbalance_history.pop(0)

        logger.debug(
            "orderbook_analyzed",
            symbol=snapshot.symbol,
            imbalance=imbalance.imbalance_ratio,
            liquidity_score=liquidity.liquidity_score,
            predicted_direction=imbalance.predicted_direction,
        )

        return imbalance, liquidity

    def _calculate_imbalance(self, snapshot: OrderBookSnapshot) -> BookImbalance:
        """Calculate order book imbalance metrics."""
        # Get top N levels
        bids = snapshot.bids[: self.max_levels]
        asks = snapshot.asks[: self.max_levels]

        if len(bids) == 0 or len(asks) == 0:
            return self._neutral_imbalance()

        # Simple imbalance: bid volume / total volume
        total_bid = sum(level.bid_size for level in bids)
        total_ask = sum(level.ask_size for level in asks)
        total = total_bid + total_ask

        if total == 0:
            return self._neutral_imbalance()

        # Imbalance ratio: -1 (all asks) to +1 (all bids)
        imbalance_ratio = (total_bid - total_ask) / total

        # Weighted imbalance (closer levels weighted higher)
        weighted_imbalance = self._weighted_imbalance(bids, asks, snapshot.mid_price)

        # Deep book imbalance (levels 5-20)
        depth_imbalance = self._depth_imbalance(bids, asks)

        # Find support/resistance levels
        support_level = self._find_support_level(bids)
        resistance_level = self._find_resistance_level(asks)

        # Momentum score (from imbalance trend)
        momentum_score = self._calculate_momentum()

        # Predicted direction
        combined_score = imbalance_ratio * 0.5 + weighted_imbalance * 0.3 + momentum_score * 0.2

        if combined_score > self.imbalance_threshold:
            predicted_direction = "UP"
        elif combined_score < -self.imbalance_threshold:
            predicted_direction = "DOWN"
        else:
            predicted_direction = "NEUTRAL"

        return BookImbalance(
            imbalance_ratio=imbalance_ratio,
            weighted_imbalance=weighted_imbalance,
            depth_imbalance=depth_imbalance,
            momentum_score=momentum_score,
            support_level=support_level,
            resistance_level=resistance_level,
            predicted_direction=predicted_direction,
        )

    def _weighted_imbalance(
        self,
        bids: List[OrderBookLevel],
        asks: List[OrderBookLevel],
        mid_price: float,
    ) -> float:
        """Calculate volume-weighted imbalance (closer levels weighted more)."""
        weighted_bid = 0.0
        weighted_ask = 0.0

        for i, level in enumerate(bids):
            # Weight decays with distance from mid
            distance_bps = abs(level.price - mid_price) / mid_price * 10000
            weight = 1.0 / (1.0 + distance_bps / 100.0)  # Decay by distance
            weighted_bid += level.bid_size * weight

        for i, level in enumerate(asks):
            distance_bps = abs(level.price - mid_price) / mid_price * 10000
            weight = 1.0 / (1.0 + distance_bps / 100.0)
            weighted_ask += level.ask_size * weight

        total = weighted_bid + weighted_ask
        if total == 0:
            return 0.0

        return (weighted_bid - weighted_ask) / total

    def _depth_imbalance(
        self,
        bids: List[OrderBookLevel],
        asks: List[OrderBookLevel],
    ) -> float:
        """Calculate imbalance in deep book (levels beyond immediate top)."""
        # Look at levels 5-20 (if available)
        deep_bids = bids[5:20] if len(bids) > 5 else []
        deep_asks = asks[5:20] if len(asks) > 5 else []

        total_deep_bid = sum(level.bid_size for level in deep_bids)
        total_deep_ask = sum(level.ask_size for level in deep_asks)
        total = total_deep_bid + total_deep_ask

        if total == 0:
            return 0.0

        return (total_deep_bid - total_deep_ask) / total

    def _find_support_level(self, bids: List[OrderBookLevel]) -> Optional[float]:
        """Find strong support level (large bid concentration)."""
        if len(bids) == 0:
            return None

        # Find level with largest bid size
        max_bid_level = max(bids, key=lambda x: x.bid_size)

        # Check if it's significantly larger than average
        avg_bid_size = sum(level.bid_size for level in bids) / len(bids)

        if max_bid_level.bid_size > avg_bid_size * 2.0:  # 2x larger than average
            return max_bid_level.price

        return None

    def _find_resistance_level(self, asks: List[OrderBookLevel]) -> Optional[float]:
        """Find strong resistance level (large ask concentration)."""
        if len(asks) == 0:
            return None

        # Find level with largest ask size
        max_ask_level = max(asks, key=lambda x: x.ask_size)

        # Check if it's significantly larger than average
        avg_ask_size = sum(level.ask_size for level in asks) / len(asks)

        if max_ask_level.ask_size > avg_ask_size * 2.0:
            return max_ask_level.price

        return None

    def _calculate_momentum(self) -> float:
        """Calculate momentum from imbalance history."""
        if len(self.imbalance_history) < 5:
            return 0.0

        # Recent imbalance trend
        recent = self.imbalance_history[-5:]
        trend = (recent[-1] - recent[0]) / 5.0  # Average change per period

        return np.clip(trend, -1.0, 1.0)

    def _calculate_liquidity(self, snapshot: OrderBookSnapshot) -> LiquidityMetrics:
        """Calculate liquidity metrics."""
        bids = snapshot.bids[: self.max_levels]
        asks = snapshot.asks[: self.max_levels]

        # Total liquidity
        total_bid = sum(level.bid_size for level in bids)
        total_ask = sum(level.ask_size for level in asks)

        # Effective spread (for a typical trade size)
        typical_size = 1.0  # 1 BTC or equivalent
        effective_spread = self._calculate_effective_spread(bids, asks, typical_size)

        # Market depth score (how much liquidity at top levels)
        top_5_bid = sum(level.bid_size for level in bids[:5]) if len(bids) >= 5 else total_bid
        top_5_ask = sum(level.ask_size for level in asks[:5]) if len(asks) >= 5 else total_ask
        depth_score = (top_5_bid + top_5_ask) / (total_bid + total_ask + 1e-9)

        # Liquidity score (composite)
        liquidity_score = self._calculate_liquidity_score(
            total_bid + total_ask,
            effective_spread,
            depth_score,
        )

        return LiquidityMetrics(
            total_bid_liquidity=total_bid,
            total_ask_liquidity=total_ask,
            effective_spread_bps=effective_spread,
            market_depth_score=depth_score,
            liquidity_score=liquidity_score,
            is_liquid=liquidity_score >= self.min_liquidity_score,
        )

    def _calculate_effective_spread(
        self,
        bids: List[OrderBookLevel],
        asks: List[OrderBookLevel],
        size: float,
    ) -> float:
        """Calculate effective spread for a given trade size."""
        if len(bids) == 0 or len(asks) == 0:
            return 1000.0  # Very wide spread

        # Average execution price for buying `size`
        remaining_size = size
        total_cost = 0.0

        for level in asks:
            if remaining_size <= 0:
                break

            take_size = min(remaining_size, level.ask_size)
            total_cost += take_size * level.price
            remaining_size -= take_size

        if remaining_size > 0:
            # Not enough liquidity
            return 1000.0

        avg_buy_price = total_cost / size

        # Average execution price for selling `size`
        remaining_size = size
        total_revenue = 0.0

        for level in bids:
            if remaining_size <= 0:
                break

            take_size = min(remaining_size, level.bid_size)
            total_revenue += take_size * level.price
            remaining_size -= take_size

        if remaining_size > 0:
            return 1000.0

        avg_sell_price = total_revenue / size

        # Effective spread in bps
        mid = (avg_buy_price + avg_sell_price) / 2.0
        effective_spread_bps = ((avg_buy_price - avg_sell_price) / mid) * 10000

        return effective_spread_bps

    def _calculate_liquidity_score(
        self,
        total_liquidity: float,
        effective_spread: float,
        depth_score: float,
    ) -> float:
        """Calculate composite liquidity score (0-1)."""
        # Liquidity component (normalize to [0, 1])
        # Assume 100 BTC equivalent is "excellent" liquidity
        liquidity_component = min(total_liquidity / 100.0, 1.0)

        # Spread component (tighter spread = better)
        # Assume 10 bps is "excellent", 100 bps is "poor"
        spread_component = max(0.0, 1.0 - (effective_spread - 10.0) / 90.0)

        # Depth component (already 0-1)
        depth_component = depth_score

        # Weighted combination
        liquidity_score = (
            liquidity_component * 0.4 + spread_component * 0.4 + depth_component * 0.2
        )

        return np.clip(liquidity_score, 0.0, 1.0)

    def _neutral_imbalance(self) -> BookImbalance:
        """Return neutral imbalance when book is empty."""
        return BookImbalance(
            imbalance_ratio=0.0,
            weighted_imbalance=0.0,
            depth_imbalance=0.0,
            momentum_score=0.0,
            support_level=None,
            resistance_level=None,
            predicted_direction="NEUTRAL",
        )

    def get_optimal_entry_price(
        self,
        snapshot: OrderBookSnapshot,
        direction: str,
        size: float,
    ) -> Optional[float]:
        """
        Get optimal entry price based on order book.

        Args:
            snapshot: Order book snapshot
            direction: "BUY" or "SELL"
            size: Trade size

        Returns:
            Optimal limit price (or None if insufficient liquidity)
        """
        imbalance, liquidity = self.analyze_book(snapshot)

        if not liquidity.is_liquid:
            logger.warning("insufficient_liquidity", symbol=snapshot.symbol)
            return None

        if direction == "BUY":
            # Look for support level or aggressive bid
            if imbalance.support_level is not None:
                # Wait for pullback to support
                return imbalance.support_level
            else:
                # Place limit slightly below best ask
                best_ask = snapshot.asks[0].price if snapshot.asks else snapshot.mid_price
                return best_ask * 0.9995  # 5 bps below

        else:  # SELL
            # Look for resistance level or aggressive ask
            if imbalance.resistance_level is not None:
                # Wait for rally to resistance
                return imbalance.resistance_level
            else:
                # Place limit slightly above best bid
                best_bid = snapshot.bids[0].price if snapshot.bids else snapshot.mid_price
                return best_bid * 1.0005  # 5 bps above

    def detect_spoofing(self, snapshot: OrderBookSnapshot) -> bool:
        """
        Detect potential spoofing (fake orders to manipulate).

        Spoofing signs:
        - Large order far from mid (unlikely to execute)
        - Order appears/disappears frequently
        - Size much larger than typical

        Returns:
            True if spoofing detected
        """
        # Simple heuristic: very large order far from mid price
        for level in snapshot.bids:
            distance_bps = (snapshot.mid_price - level.price) / snapshot.mid_price * 10000

            # Large order (10x median) far from mid (>50 bps)
            median_bid = np.median([l.bid_size for l in snapshot.bids])
            if level.bid_size > median_bid * 10 and distance_bps > 50:
                logger.warning(
                    "potential_spoofing_detected",
                    side="BID",
                    price=level.price,
                    size=level.bid_size,
                    distance_bps=distance_bps,
                )
                return True

        for level in snapshot.asks:
            distance_bps = (level.price - snapshot.mid_price) / snapshot.mid_price * 10000

            median_ask = np.median([l.ask_size for l in snapshot.asks])
            if level.ask_size > median_ask * 10 and distance_bps > 50:
                logger.warning(
                    "potential_spoofing_detected",
                    side="ASK",
                    price=level.price,
                    size=level.ask_size,
                    distance_bps=distance_bps,
                )
                return True

        return False
