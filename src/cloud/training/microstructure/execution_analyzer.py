"""
Execution Quality & Slippage Prediction

Predicts execution quality for trading decisions:
- Slippage estimation (difference between expected and actual fill)
- Optimal execution strategy (market vs limit vs TWAP)
- Impact cost (price impact of trade size)
- Fill probability (likelihood of limit order filling)

Traditional: Submit market order, accept slippage
Smart Execution: Predict slippage, choose optimal strategy

Example:
Trade: Buy 5 BTC
Market order → -25 bps slippage (immediate but expensive)
Limit order @ best bid → 90% fill probability, -5 bps slippage
TWAP over 10 min → -10 bps slippage (split across time)
→ Choose limit order (best risk/reward)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import structlog

from .orderbook_analyzer import OrderBookSnapshot, LiquidityMetrics
from .tape_reader import OrderFlowMetrics

logger = structlog.get_logger(__name__)


@dataclass
class SlippageEstimate:
    """Slippage prediction for a trade."""

    expected_slippage_bps: float  # Expected slippage in bps
    worst_case_slippage_bps: float  # 95th percentile slippage
    confidence: float  # 0-1, confidence in estimate
    impact_cost_bps: float  # Market impact component
    spread_cost_bps: float  # Spread component
    factors: Dict[str, float]  # Contributing factors


@dataclass
class ExecutionStrategy:
    """Recommended execution strategy."""

    strategy: str  # "MARKET", "LIMIT", "TWAP", "ICEBERG"
    urgency: str  # "HIGH", "MEDIUM", "LOW"
    recommended_price: Optional[float]  # For limit orders
    time_horizon_seconds: Optional[int]  # For TWAP
    chunk_size: Optional[float]  # For TWAP/ICEBERG
    expected_cost_bps: float  # Total expected execution cost
    fill_probability: float  # Probability of complete fill


@dataclass
class LiquidityScore:
    """Comprehensive liquidity scoring."""

    overall_score: float  # 0-1, overall liquidity quality
    depth_score: float  # Order book depth
    spread_score: float  # Bid-ask spread tightness
    resilience_score: float  # How quickly book recovers
    volatility_score: float  # Price stability
    is_tradeable: bool  # Above minimum threshold
    risk_level: str  # "LOW", "MEDIUM", "HIGH"


class ExecutionAnalyzer:
    """
    Analyze execution quality and predict slippage.

    Uses order book, order flow, and historical data to:
    - Predict slippage for different trade sizes
    - Recommend optimal execution strategy
    - Score market liquidity
    - Estimate fill probability
    """

    def __init__(
        self,
        min_liquidity_score: float = 0.5,
        max_acceptable_slippage_bps: float = 25.0,
    ):
        """
        Initialize execution analyzer.

        Args:
            min_liquidity_score: Minimum liquidity to trade
            max_acceptable_slippage_bps: Maximum acceptable slippage
        """
        self.min_liquidity_score = min_liquidity_score
        self.max_acceptable_slippage_bps = max_acceptable_slippage_bps

        # Historical slippage data
        self.slippage_history: List[float] = []

        logger.info(
            "execution_analyzer_initialized",
            min_liquidity=min_liquidity_score,
            max_slippage=max_acceptable_slippage_bps,
        )

    def predict_slippage(
        self,
        orderbook: OrderBookSnapshot,
        flow_metrics: OrderFlowMetrics,
        direction: str,
        size: float,
        volatility: float,
    ) -> SlippageEstimate:
        """
        Predict slippage for a trade.

        Args:
            orderbook: Order book snapshot
            flow_metrics: Recent order flow
            direction: "BUY" or "SELL"
            size: Trade size
            volatility: Current volatility

        Returns:
            Slippage estimate
        """
        # Factor 1: Spread cost
        spread_cost = orderbook.spread_bps / 2  # Cross half spread

        # Factor 2: Market impact (price moves against you)
        impact_cost = self._estimate_market_impact(orderbook, direction, size)

        # Factor 3: Volatility cost (price might move during execution)
        volatility_cost = volatility * 10000 * 0.1  # 10% of volatility in bps

        # Factor 4: Flow pressure (order flow pushing price)
        flow_cost = self._estimate_flow_cost(flow_metrics, direction)

        # Total expected slippage
        expected_slippage = spread_cost + impact_cost + volatility_cost + flow_cost

        # Worst case (95th percentile) - add 2x std dev
        historical_std = np.std(self.slippage_history) if len(self.slippage_history) > 10 else 10.0
        worst_case = expected_slippage + 2.0 * historical_std

        # Confidence based on data quality
        confidence = min(len(self.slippage_history) / 100.0, 1.0)

        factors = {
            "spread_cost": spread_cost,
            "impact_cost": impact_cost,
            "volatility_cost": volatility_cost,
            "flow_cost": flow_cost,
        }

        return SlippageEstimate(
            expected_slippage_bps=expected_slippage,
            worst_case_slippage_bps=worst_case,
            confidence=confidence,
            impact_cost_bps=impact_cost,
            spread_cost_bps=spread_cost,
            factors=factors,
        )

    def _estimate_market_impact(
        self,
        orderbook: OrderBookSnapshot,
        direction: str,
        size: float,
    ) -> float:
        """
        Estimate market impact of trade.

        Uses square-root model: Impact ∝ √(size / liquidity)
        """
        levels = orderbook.asks if direction == "BUY" else orderbook.bids

        if len(levels) == 0:
            return 50.0  # High impact for empty book

        # Available liquidity in top 5 levels
        available_liquidity = sum(
            level.ask_size if direction == "BUY" else level.bid_size for level in levels[:5]
        )

        if available_liquidity == 0:
            return 50.0

        # Square-root impact model
        impact_ratio = size / available_liquidity
        impact_bps = 20.0 * np.sqrt(impact_ratio)  # Calibrated constant

        return min(impact_bps, 100.0)  # Cap at 100 bps

    def _estimate_flow_cost(self, flow_metrics: OrderFlowMetrics, direction: str) -> float:
        """Estimate cost from order flow pressure."""
        # If we're buying into strong buying flow → pay premium
        # If we're selling into strong selling flow → pay premium

        if direction == "BUY":
            if flow_metrics.flow_direction == "BUYING":
                # Buying into buying pressure → premium
                return abs(flow_metrics.aggression_score) * 10.0
            else:
                # Buying into selling pressure → discount
                return -abs(flow_metrics.aggression_score) * 5.0

        else:  # SELL
            if flow_metrics.flow_direction == "SELLING":
                # Selling into selling pressure → premium
                return abs(flow_metrics.aggression_score) * 10.0
            else:
                # Selling into buying pressure → discount
                return -abs(flow_metrics.aggression_score) * 5.0

    def recommend_execution_strategy(
        self,
        slippage_estimate: SlippageEstimate,
        orderbook: OrderBookSnapshot,
        urgency: str,
        size: float,
    ) -> ExecutionStrategy:
        """
        Recommend optimal execution strategy.

        Args:
            slippage_estimate: Predicted slippage
            orderbook: Order book snapshot
            urgency: "HIGH", "MEDIUM", "LOW"
            size: Trade size

        Returns:
            Execution strategy recommendation
        """
        if urgency == "HIGH":
            # High urgency → market order
            return ExecutionStrategy(
                strategy="MARKET",
                urgency=urgency,
                recommended_price=None,
                time_horizon_seconds=None,
                chunk_size=None,
                expected_cost_bps=slippage_estimate.expected_slippage_bps,
                fill_probability=1.0,
            )

        elif slippage_estimate.expected_slippage_bps > self.max_acceptable_slippage_bps:
            # High slippage → use TWAP to reduce impact
            num_chunks = max(3, int(slippage_estimate.expected_slippage_bps / 10))
            chunk_size = size / num_chunks
            time_horizon = num_chunks * 30  # 30 seconds per chunk

            # TWAP reduces impact by ~40%
            twap_cost = slippage_estimate.expected_slippage_bps * 0.6

            return ExecutionStrategy(
                strategy="TWAP",
                urgency=urgency,
                recommended_price=None,
                time_horizon_seconds=time_horizon,
                chunk_size=chunk_size,
                expected_cost_bps=twap_cost,
                fill_probability=0.95,
            )

        else:
            # Normal slippage → limit order
            # Place at mid-price or better
            fill_prob = self._estimate_fill_probability(orderbook, orderbook.mid_price)

            return ExecutionStrategy(
                strategy="LIMIT",
                urgency=urgency,
                recommended_price=orderbook.mid_price,
                time_horizon_seconds=60,  # 1 minute patience
                chunk_size=None,
                expected_cost_bps=slippage_estimate.spread_cost_bps / 2,  # Cross quarter spread
                fill_probability=fill_prob,
            )

    def _estimate_fill_probability(self, orderbook: OrderBookSnapshot, limit_price: float) -> float:
        """Estimate probability of limit order filling."""
        # Simple model: distance from mid → lower probability
        distance_bps = abs(limit_price - orderbook.mid_price) / orderbook.mid_price * 10000

        # Exponential decay
        fill_prob = np.exp(-distance_bps / 10.0)  # 50% prob at 7 bps

        return np.clip(fill_prob, 0.1, 1.0)

    def score_liquidity(
        self,
        orderbook: OrderBookSnapshot,
        liquidity_metrics: LiquidityMetrics,
        flow_metrics: OrderFlowMetrics,
        volatility: float,
    ) -> LiquidityScore:
        """
        Comprehensive liquidity scoring.

        Args:
            orderbook: Order book snapshot
            liquidity_metrics: Book liquidity metrics
            flow_metrics: Order flow metrics
            volatility: Current volatility

        Returns:
            Comprehensive liquidity score
        """
        # Depth score (from liquidity metrics)
        depth_score = liquidity_metrics.market_depth_score

        # Spread score (tighter = better)
        spread_score = max(0.0, 1.0 - orderbook.spread_bps / 50.0)  # 50 bps = 0 score

        # Resilience score (how quickly book recovers after trades)
        # Proxy: balance of buy/sell volume
        total_volume = flow_metrics.buy_volume + flow_metrics.sell_volume
        if total_volume > 0:
            balance = 1.0 - abs(flow_metrics.buy_volume - flow_metrics.sell_volume) / total_volume
            resilience_score = balance
        else:
            resilience_score = 0.5

        # Volatility score (lower vol = better for execution)
        volatility_score = max(0.0, 1.0 - volatility / 1.0)  # 100% vol = 0 score

        # Overall score (weighted combination)
        overall_score = (
            depth_score * 0.3 + spread_score * 0.3 + resilience_score * 0.2 + volatility_score * 0.2
        )

        # Risk level
        if overall_score >= 0.7:
            risk_level = "LOW"
        elif overall_score >= 0.4:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"

        return LiquidityScore(
            overall_score=overall_score,
            depth_score=depth_score,
            spread_score=spread_score,
            resilience_score=resilience_score,
            volatility_score=volatility_score,
            is_tradeable=overall_score >= self.min_liquidity_score,
            risk_level=risk_level,
        )

    def update_slippage_history(self, actual_slippage_bps: float) -> None:
        """Update historical slippage data."""
        self.slippage_history.append(actual_slippage_bps)

        # Keep last 1000 trades
        if len(self.slippage_history) > 1000:
            self.slippage_history.pop(0)

    def get_execution_dashboard(self) -> Dict:
        """Get execution analytics dashboard."""
        if len(self.slippage_history) == 0:
            return {
                "avg_slippage_bps": 0.0,
                "median_slippage_bps": 0.0,
                "p95_slippage_bps": 0.0,
                "sample_size": 0,
            }

        return {
            "avg_slippage_bps": np.mean(self.slippage_history),
            "median_slippage_bps": np.median(self.slippage_history),
            "p95_slippage_bps": np.percentile(self.slippage_history, 95),
            "std_slippage_bps": np.std(self.slippage_history),
            "min_slippage_bps": np.min(self.slippage_history),
            "max_slippage_bps": np.max(self.slippage_history),
            "sample_size": len(self.slippage_history),
        }
