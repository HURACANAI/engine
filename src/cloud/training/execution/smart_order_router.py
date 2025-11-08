"""
Smart Order Router

Liquidity-based routing with pre-trade risk checks.
Decides where to send orders based on liquidity, fees, and latency.

Key Features:
- Liquidity-based routing
- Fee optimization
- Latency optimization
- Pre-trade risk integration
- Exchange selection
- Order type selection

Author: Huracan Engine Team
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum

import structlog

logger = structlog.get_logger(__name__)


class RouteDecision(Enum):
    """Route decision"""
    ROUTE = "route"
    REJECT = "reject"
    DELAY = "delay"
    REDUCE_SIZE = "reduce_size"


@dataclass
class ExchangeLiquidity:
    """Exchange liquidity information"""
    exchange_id: str
    symbol: str
    bid_depth_usd: float
    ask_depth_usd: float
    spread_bps: float
    liquidity_score: float
    fee_taker_bps: float
    fee_maker_bps: float
    latency_ms: float
    reliability_score: float


@dataclass
class RoutingDecision:
    """Routing decision"""
    route_decision: RouteDecision
    selected_exchange: Optional[str]
    order_type: str  # "market", "limit", "maker", "taker"
    recommended_size: float
    reasoning: str
    alternatives: List[str] = field(default_factory=list)
    metadata: Dict[str, any] = field(default_factory=dict)


class SmartOrderRouter:
    """
    Smart Order Router.
    
    Decides where to send orders based on liquidity, fees, and latency.
    Integrates with pre-trade risk checks.
    
    Usage:
        router = SmartOrderRouter()
        
        decision = router.route_order(
            symbol="BTCUSDT",
            direction="buy",
            size_usd=1000.0,
            exchanges_liquidity={
                "binance": ExchangeLiquidity(...),
                "coinbase": ExchangeLiquidity(...),
                ...
            }
        )
        
        if decision.route_decision == RouteDecision.ROUTE:
            send_order(decision.selected_exchange, decision.order_type, ...)
    """
    
    def __init__(
        self,
        liquidity_weight: float = 0.4,
        fee_weight: float = 0.3,
        latency_weight: float = 0.2,
        reliability_weight: float = 0.1,
        min_liquidity_score: float = 0.5,
        max_latency_ms: float = 100.0,
        prefer_maker: bool = True  # Prefer maker orders for fee rebates
    ):
        """
        Initialize smart order router.
        
        Args:
            liquidity_weight: Weight for liquidity in routing decision
            fee_weight: Weight for fees in routing decision
            latency_weight: Weight for latency in routing decision
            reliability_weight: Weight for reliability in routing decision
            min_liquidity_score: Minimum liquidity score to route
            max_latency_ms: Maximum acceptable latency
            prefer_maker: Prefer maker orders for fee rebates
        """
        self.liquidity_weight = liquidity_weight
        self.fee_weight = fee_weight
        self.latency_weight = latency_weight
        self.reliability_weight = reliability_weight
        self.min_liquidity_score = min_liquidity_score
        self.max_latency_ms = max_latency_ms
        self.prefer_maker = prefer_maker
        
        # Validate weights sum to 1.0
        total_weight = liquidity_weight + fee_weight + latency_weight + reliability_weight
        if abs(total_weight - 1.0) > 0.01:
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}")
        
        logger.info(
            "smart_order_router_initialized",
            liquidity_weight=liquidity_weight,
            fee_weight=fee_weight,
            latency_weight=latency_weight,
            min_liquidity_score=min_liquidity_score
        )
    
    def route_order(
        self,
        symbol: str,
        direction: str,
        size_usd: float,
        exchanges_liquidity: Dict[str, ExchangeLiquidity],
        pre_trade_risk_result: Optional[any] = None  # PreTradeRiskResult
    ) -> RoutingDecision:
        """
        Route order to best exchange.
        
        Args:
            symbol: Trading symbol
            direction: Order direction ("buy" or "sell")
            size_usd: Order size in USD
            exchanges_liquidity: Dictionary of exchange -> liquidity info
            pre_trade_risk_result: Pre-trade risk check result (optional)
        
        Returns:
            RoutingDecision
        """
        # Check pre-trade risk if provided
        if pre_trade_risk_result and not pre_trade_risk_result.approved:
            return RoutingDecision(
                route_decision=RouteDecision.REJECT,
                selected_exchange=None,
                order_type="market",
                recommended_size=0.0,
                reasoning=f"Pre-trade risk check failed: {pre_trade_risk_result.rejection_reason}",
                metadata={"risk_check": "failed"}
            )
        
        # Use risk-adjusted size if available
        if pre_trade_risk_result and pre_trade_risk_result.recommended_size < size_usd:
            size_usd = pre_trade_risk_result.recommended_size
            if size_usd == 0:
                return RoutingDecision(
                    route_decision=RouteDecision.REJECT,
                    selected_exchange=None,
                    order_type="market",
                    recommended_size=0.0,
                    reasoning="Risk check reduced size to zero",
                    metadata={"risk_check": "size_reduced_to_zero"}
                )
        
        if not exchanges_liquidity:
            return RoutingDecision(
                route_decision=RouteDecision.REJECT,
                selected_exchange=None,
                order_type="market",
                recommended_size=size_usd,
                reasoning="No exchanges available",
                metadata={"error": "no_exchanges"}
            )
        
        # Score each exchange
        exchange_scores = {}
        for exchange_id, liquidity in exchanges_liquidity.items():
            score = self._score_exchange(
                exchange_id=exchange_id,
                liquidity=liquidity,
                direction=direction,
                size_usd=size_usd
            )
            exchange_scores[exchange_id] = score
        
        # Filter exchanges by minimum requirements
        viable_exchanges = {
            ex_id: score for ex_id, score in exchange_scores.items()
            if exchanges_liquidity[ex_id].liquidity_score >= self.min_liquidity_score
            and exchanges_liquidity[ex_id].latency_ms <= self.max_latency_ms
        }
        
        if not viable_exchanges:
            return RoutingDecision(
                route_decision=RouteDecision.REJECT,
                selected_exchange=None,
                order_type="market",
                recommended_size=size_usd,
                reasoning="No viable exchanges (liquidity or latency constraints)",
                metadata={"liquidity_scores": {ex: liq.liquidity_score for ex, liq in exchanges_liquidity.items()}}
            )
        
        # Select best exchange
        best_exchange = max(viable_exchanges.items(), key=lambda x: x[1])[0]
        best_liquidity = exchanges_liquidity[best_exchange]
        
        # Determine order type
        order_type = self._determine_order_type(
            liquidity=best_liquidity,
            direction=direction,
            size_usd=size_usd,
            prefer_maker=self.prefer_maker
        )
        
        # Get alternatives (top 3)
        sorted_exchanges = sorted(viable_exchanges.items(), key=lambda x: x[1], reverse=True)
        alternatives = [ex_id for ex_id, _ in sorted_exchanges[1:4]]
        
        decision = RoutingDecision(
            route_decision=RouteDecision.ROUTE,
            selected_exchange=best_exchange,
            order_type=order_type,
            recommended_size=size_usd,
            reasoning=f"Best exchange: {best_exchange} (score: {exchange_scores[best_exchange]:.3f})",
            alternatives=alternatives,
            metadata={
                "exchange_scores": exchange_scores,
                "liquidity_score": best_liquidity.liquidity_score,
                "spread_bps": best_liquidity.spread_bps,
                "fee_bps": best_liquidity.fee_taker_bps if order_type == "taker" else best_liquidity.fee_maker_bps
            }
        )
        
        logger.info(
            "order_routed",
            symbol=symbol,
            direction=direction,
            size_usd=size_usd,
            exchange=best_exchange,
            order_type=order_type,
            score=exchange_scores[best_exchange]
        )
        
        return decision
    
    def _score_exchange(
        self,
        exchange_id: str,
        liquidity: ExchangeLiquidity,
        direction: str,
        size_usd: float
    ) -> float:
        """Score exchange for routing"""
        # Liquidity score (higher is better)
        liquidity_score = liquidity.liquidity_score
        
        # Fee score (lower fees = higher score, negative for maker rebates)
        if self.prefer_maker and liquidity.fee_maker_bps < 0:
            # Maker rebate is good
            fee_score = 1.0 + abs(liquidity.fee_maker_bps) / 100.0
        else:
            # Lower taker fees = higher score
            fee_score = 1.0 - (liquidity.fee_taker_bps / 100.0)
        fee_score = max(0.0, min(1.0, fee_score))
        
        # Latency score (lower latency = higher score)
        latency_score = max(0.0, 1.0 - (liquidity.latency_ms / self.max_latency_ms))
        
        # Reliability score (higher is better)
        reliability_score = liquidity.reliability_score
        
        # Depth score (check if enough liquidity for order)
        required_depth = size_usd
        if direction == "buy":
            available_depth = liquidity.ask_depth_usd
        else:
            available_depth = liquidity.bid_depth_usd
        
        depth_score = min(1.0, available_depth / required_depth) if required_depth > 0 else 1.0
        
        # Combined score
        score = (
            liquidity_score * self.liquidity_weight +
            fee_score * self.fee_weight +
            latency_score * self.latency_weight +
            reliability_score * self.reliability_weight
        ) * depth_score  # Penalize if insufficient depth
        
        return float(score)
    
    def _determine_order_type(
        self,
        liquidity: ExchangeLiquidity,
        direction: str,
        size_usd: float,
        prefer_maker: bool
    ) -> str:
        """Determine optimal order type"""
        # Check if enough liquidity for maker order
        if direction == "buy":
            available_depth = liquidity.ask_depth_usd
        else:
            available_depth = liquidity.bid_depth_usd
        
        # If prefer_maker and enough liquidity, use maker
        if prefer_maker and available_depth >= size_usd * 1.2:  # 20% buffer
            return "maker"
        elif prefer_maker and available_depth >= size_usd * 0.8:  # 80% of size
            return "limit"  # Limit order that may become maker
        else:
            return "taker"  # Market order (taker)

