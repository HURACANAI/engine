"""
Tape Reading (Order Flow Analysis)

Reads the "tape" (stream of executed trades) to detect:
- Institutional flow (large trades)
- Aggression (market orders vs limit orders)
- Absorption (large orders getting filled without price moving)
- Momentum shifts (acceleration in buying/selling)

Traditional: Look at price action only
Tape Reading: Understand WHO is trading and HOW

Example:
Time    | Price | Size | Side | Aggressor
10:00:01| 45000 | 0.5  | BUY  | TAKER (aggressive buy)
10:00:02| 45001 | 2.1  | BUY  | TAKER (large aggressive buy!)
10:00:03| 45002 | 1.8  | BUY  | TAKER (continuation)
10:00:04| 45002 | 3.5  | SELL | MAKER (absorption - price stops rising)
→ Strong buying pressure absorbed at 45002 → resistance level
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class Trade:
    """Single executed trade."""

    timestamp: float
    price: float
    size: float
    side: str  # "BUY" or "SELL"
    is_aggressive: bool  # True if market order (taker), False if limit (maker)


@dataclass
class OrderFlowMetrics:
    """Order flow analysis metrics."""

    buy_volume: float  # Total buy volume
    sell_volume: float  # Total sell volume
    aggressive_buy_volume: float  # Market buy orders
    aggressive_sell_volume: float  # Market sell orders
    large_trade_count: int  # Number of large trades
    avg_trade_size: float  # Average trade size
    flow_direction: str  # "BUYING", "SELLING", "NEUTRAL"
    aggression_score: float  # -1 to +1 (selling to buying)
    institutional_signal: bool  # Large player detected
    absorption_detected: bool  # Large order absorbed


class TapeReader:
    """
    Read and analyze order flow from executed trades.

    Detects patterns that indicate:
    - Institutional activity (large trades)
    - Market momentum (aggressive buying/selling)
    - Absorption (supply/demand exhaustion)
    - Trend continuation or reversal
    """

    def __init__(
        self,
        large_trade_threshold: float = 5.0,  # BTC equivalent
        lookback_trades: int = 100,
        aggression_threshold: float = 0.3,
    ):
        """
        Initialize tape reader.

        Args:
            large_trade_threshold: Size threshold for "large" trade
            lookback_trades: Number of recent trades to analyze
            aggression_threshold: Threshold for directional signal
        """
        self.large_trade_threshold = large_trade_threshold
        self.lookback_trades = lookback_trades
        self.aggression_threshold = aggression_threshold

        # Recent trade history
        self.trade_history: List[Trade] = []

        logger.info(
            "tape_reader_initialized",
            large_threshold=large_trade_threshold,
            lookback=lookback_trades,
        )

    def add_trade(self, trade: Trade) -> None:
        """Add new trade to tape."""
        self.trade_history.append(trade)

        # Keep only recent trades
        if len(self.trade_history) > self.lookback_trades * 2:
            self.trade_history = self.trade_history[-self.lookback_trades :]

    def analyze_flow(self, recent_trades: Optional[List[Trade]] = None) -> OrderFlowMetrics:
        """
        Analyze recent order flow.

        Args:
            recent_trades: Trades to analyze (or None to use history)

        Returns:
            Order flow metrics
        """
        if recent_trades is None:
            recent_trades = self.trade_history[-self.lookback_trades :]

        if len(recent_trades) == 0:
            return self._neutral_flow()

        # Calculate volumes
        buy_volume = sum(t.size for t in recent_trades if t.side == "BUY")
        sell_volume = sum(t.size for t in recent_trades if t.side == "SELL")

        aggressive_buy = sum(t.size for t in recent_trades if t.side == "BUY" and t.is_aggressive)
        aggressive_sell = sum(t.size for t in recent_trades if t.side == "SELL" and t.is_aggressive)

        # Large trades
        large_trades = [t for t in recent_trades if t.size >= self.large_trade_threshold]
        large_trade_count = len(large_trades)

        # Average size
        avg_size = sum(t.size for t in recent_trades) / len(recent_trades)

        # Flow direction
        total_volume = buy_volume + sell_volume
        if total_volume == 0:
            flow_direction = "NEUTRAL"
            net_flow_ratio = 0.0
        else:
            net_flow_ratio = (buy_volume - sell_volume) / total_volume

            if net_flow_ratio > self.aggression_threshold:
                flow_direction = "BUYING"
            elif net_flow_ratio < -self.aggression_threshold:
                flow_direction = "SELLING"
            else:
                flow_direction = "NEUTRAL"

        # Aggression score (aggressive buying/selling vs passive)
        total_aggressive = aggressive_buy + aggressive_sell
        if total_aggressive == 0:
            aggression_score = 0.0
        else:
            aggression_score = (aggressive_buy - aggressive_sell) / total_aggressive

        # Institutional signal (multiple large trades in same direction)
        institutional_signal = self._detect_institutional_flow(large_trades)

        # Absorption (large volume without price moving)
        absorption_detected = self._detect_absorption(recent_trades)

        return OrderFlowMetrics(
            buy_volume=buy_volume,
            sell_volume=sell_volume,
            aggressive_buy_volume=aggressive_buy,
            aggressive_sell_volume=aggressive_sell,
            large_trade_count=large_trade_count,
            avg_trade_size=avg_size,
            flow_direction=flow_direction,
            aggression_score=aggression_score,
            institutional_signal=institutional_signal,
            absorption_detected=absorption_detected,
        )

    def _detect_institutional_flow(self, large_trades: List[Trade]) -> bool:
        """Detect institutional activity from large trades."""
        if len(large_trades) < 3:
            return False

        # Check if large trades are in same direction
        buy_count = sum(1 for t in large_trades if t.side == "BUY")
        sell_count = len(large_trades) - buy_count

        # 70%+ in same direction = institutional
        if buy_count / len(large_trades) > 0.7 or sell_count / len(large_trades) > 0.7:
            logger.info(
                "institutional_flow_detected",
                large_trades=len(large_trades),
                buy_count=buy_count,
                sell_count=sell_count,
            )
            return True

        return False

    def _detect_absorption(self, trades: List[Trade]) -> bool:
        """
        Detect absorption (large volume without price movement).

        Absorption happens when large order gets filled but price doesn't move.
        This indicates strong support/resistance.
        """
        if len(trades) < 20:
            return False

        # Split into chunks of 10 trades
        chunk_size = 10
        chunks = [trades[i : i + chunk_size] for i in range(0, len(trades), chunk_size)]

        for chunk in chunks:
            if len(chunk) < chunk_size:
                continue

            # Calculate volume and price movement
            total_volume = sum(t.size for t in chunk)
            prices = [t.price for t in chunk]
            price_change_pct = abs(prices[-1] - prices[0]) / prices[0]

            # Large volume (>5 BTC) with small price movement (<0.1%)
            if total_volume > 5.0 and price_change_pct < 0.001:
                logger.info(
                    "absorption_detected",
                    volume=total_volume,
                    price_change_pct=price_change_pct * 100,
                )
                return True

        return False

    def get_momentum_signal(self) -> str:
        """
        Get momentum signal from order flow.

        Returns:
            "BULLISH", "BEARISH", or "NEUTRAL"
        """
        flow = self.analyze_flow()

        # Strong bullish: buying flow + aggressive buying + institutional
        if (
            flow.flow_direction == "BUYING"
            and flow.aggression_score > 0.3
            and flow.institutional_signal
        ):
            return "BULLISH"

        # Strong bearish: selling flow + aggressive selling + institutional
        if (
            flow.flow_direction == "SELLING"
            and flow.aggression_score < -0.3
            and flow.institutional_signal
        ):
            return "BEARISH"

        # Moderate bullish: buying flow + aggressive
        if flow.flow_direction == "BUYING" and flow.aggression_score > 0.2:
            return "BULLISH"

        # Moderate bearish: selling flow + aggressive
        if flow.flow_direction == "SELLING" and flow.aggression_score < -0.2:
            return "BEARISH"

        return "NEUTRAL"

    def detect_iceberg_order(self, trades: List[Trade], price_threshold: float = 0.001) -> bool:
        """
        Detect iceberg order (large hidden order being filled in chunks).

        Iceberg signs:
        - Many trades at same price
        - Similar sizes
        - Same side
        - No price movement
        """
        if len(trades) < 10:
            return False

        # Group trades by price (within threshold)
        price_groups: Dict[float, List[Trade]] = {}

        for trade in trades:
            # Find matching price group
            found_group = False
            for price, group_trades in price_groups.items():
                if abs(trade.price - price) / price < price_threshold:
                    group_trades.append(trade)
                    found_group = True
                    break

            if not found_group:
                price_groups[trade.price] = [trade]

        # Check each group for iceberg pattern
        for price, group in price_groups.items():
            if len(group) < 5:  # Need at least 5 trades
                continue

            # Check if same side
            buy_count = sum(1 for t in group if t.side == "BUY")
            if buy_count != len(group) and buy_count != 0:
                continue  # Mixed sides

            # Check if similar sizes (low variance)
            sizes = [t.size for t in group]
            avg_size = np.mean(sizes)
            std_size = np.std(sizes)

            if std_size / avg_size < 0.3:  # Low variance
                logger.info(
                    "iceberg_order_detected",
                    price=price,
                    trades=len(group),
                    avg_size=avg_size,
                )
                return True

        return False

    def _neutral_flow(self) -> OrderFlowMetrics:
        """Return neutral flow metrics."""
        return OrderFlowMetrics(
            buy_volume=0.0,
            sell_volume=0.0,
            aggressive_buy_volume=0.0,
            aggressive_sell_volume=0.0,
            large_trade_count=0,
            avg_trade_size=0.0,
            flow_direction="NEUTRAL",
            aggression_score=0.0,
            institutional_signal=False,
            absorption_detected=False,
        )

    def get_flow_strength(self) -> float:
        """
        Get flow strength score (0-1).

        Combines volume, aggression, and institutional signals.
        """
        flow = self.analyze_flow()

        # Volume component
        total_volume = flow.buy_volume + flow.sell_volume
        volume_score = min(total_volume / 100.0, 1.0)  # 100 BTC = max

        # Aggression component
        aggression_score = abs(flow.aggression_score)

        # Institutional component
        institutional_score = 1.0 if flow.institutional_signal else 0.0

        # Weighted combination
        strength = volume_score * 0.4 + aggression_score * 0.4 + institutional_score * 0.2

        return strength
