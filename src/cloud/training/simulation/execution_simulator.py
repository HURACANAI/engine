"""
Execution Simulator for Slippage Learning

Learns slippage from order book snapshots.
Feeds expected slippage into backtests and sizing.

Key Features:
- Slippage learning from order book data
- Market impact modeling
- Fill probability estimation
- Order book depth analysis
- Historical slippage tracking
- Integration with backtests

Author: Huracan Engine Team
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
from datetime import datetime, timezone

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


class OrderType(Enum):
    """Order type"""
    MARKET = "market"
    LIMIT = "limit"
    MAKER = "maker"
    TAKER = "taker"


@dataclass
class OrderBookSnapshot:
    """Order book snapshot"""
    symbol: str
    timestamp: datetime
    bid_prices: List[float]
    bid_sizes: List[float]
    ask_prices: List[float]
    ask_sizes: List[float]
    mid_price: float
    spread_bps: float


@dataclass
class SlippageEstimate:
    """Slippage estimate"""
    expected_slippage_bps: float
    confidence: float
    market_impact_bps: float
    spread_cost_bps: float
    fill_probability: float
    metadata: Dict[str, any] = field(default_factory=dict)


@dataclass
class ExecutionResult:
    """Execution result"""
    symbol: str
    order_type: OrderType
    direction: str  # "buy" or "sell"
    size_usd: float
    requested_price: float
    executed_price: float
    slippage_bps: float
    fill_probability: float
    execution_time_ms: float
    metadata: Dict[str, any] = field(default_factory=dict)


class ExecutionSimulator:
    """
    Execution Simulator for Slippage Learning.
    
    Learns slippage from order book snapshots and historical fills.
    Feeds expected slippage into backtests and sizing.
    
    Usage:
        simulator = ExecutionSimulator()
        
        # Learn from historical data
        simulator.learn_slippage(
            order_book_snapshots=[...],
            actual_fills=[...]
        )
        
        # Estimate slippage
        estimate = simulator.estimate_slippage(
            symbol="BTCUSDT",
            direction="buy",
            size_usd=1000.0,
            order_book=order_book_snapshot
        )
        
        # Simulate execution
        result = simulator.simulate_execution(
            symbol="BTCUSDT",
            direction="buy",
            size_usd=1000.0,
            order_book=order_book_snapshot
        )
    """
    
    def __init__(
        self,
        taker_fee_bps: float = 5.0,  # Taker fee in bps
        maker_fee_bps: float = -2.0,  # Maker fee (rebate) in bps
        market_impact_factor: float = 0.1,  # Market impact factor
        min_fill_probability: float = 0.5  # Minimum fill probability
    ):
        """
        Initialize execution simulator.
        
        Args:
            taker_fee_bps: Taker fee in basis points
            maker_fee_bps: Maker fee (rebate) in basis points
            market_impact_factor: Market impact factor
            min_fill_probability: Minimum fill probability
        """
        self.taker_fee_bps = taker_fee_bps
        self.maker_fee_bps = maker_fee_bps
        self.market_impact_factor = market_impact_factor
        self.min_fill_probability = min_fill_probability
        
        # Slippage learning data
        self.slippage_history: Dict[str, List[Tuple[float, float]]] = {}  # symbol -> [(size_usd, slippage_bps)]
        self.order_book_history: Dict[str, List[OrderBookSnapshot]] = {}  # symbol -> snapshots
        
        # Learned models
        self.slippage_models: Dict[str, Dict[str, float]] = {}  # symbol -> model params
        
        logger.info(
            "execution_simulator_initialized",
            taker_fee_bps=taker_fee_bps,
            maker_fee_bps=maker_fee_bps,
            market_impact_factor=market_impact_factor
        )
    
    def learn_slippage(
        self,
        symbol: str,
        order_book_snapshots: List[OrderBookSnapshot],
        actual_fills: List[Tuple[float, float, float]]  # [(size_usd, requested_price, filled_price)]
    ) -> None:
        """
        Learn slippage from historical data.
        
        Args:
            symbol: Trading symbol
            order_book_snapshots: Order book snapshots
            actual_fills: Actual fill data
        """
        if symbol not in self.slippage_history:
            self.slippage_history[symbol] = []
        
        # Calculate slippage for each fill
        for size_usd, requested_price, filled_price in actual_fills:
            slippage_bps = abs((filled_price - requested_price) / requested_price) * 10000
            self.slippage_history[symbol].append((size_usd, slippage_bps))
        
        # Store order book snapshots
        if symbol not in self.order_book_history:
            self.order_book_history[symbol] = []
        
        self.order_book_history[symbol].extend(order_book_snapshots)
        
        # Learn model (simplified linear model)
        if len(self.slippage_history[symbol]) > 10:
            self._learn_slippage_model(symbol)
        
        logger.info(
            "slippage_learned",
            symbol=symbol,
            num_samples=len(self.slippage_history[symbol])
        )
    
    def estimate_slippage(
        self,
        symbol: str,
        direction: str,
        size_usd: float,
        order_book: Optional[OrderBookSnapshot] = None,
        order_type: OrderType = OrderType.MARKET
    ) -> SlippageEstimate:
        """
        Estimate slippage for an order.
        
        Args:
            symbol: Trading symbol
            direction: Order direction ("buy" or "sell")
            size_usd: Order size in USD
            order_book: Order book snapshot (optional)
            order_type: Order type
        
        Returns:
            SlippageEstimate
        """
        # Calculate spread cost
        if order_book:
            spread_bps = order_book.spread_bps
            spread_cost_bps = spread_bps / 2  # Half spread for market orders
        else:
            spread_bps = 10.0  # Default spread
            spread_cost_bps = 5.0
        
        # Calculate market impact
        market_impact_bps = self._calculate_market_impact(
            symbol,
            size_usd,
            order_book,
            direction
        )
        
        # Estimate slippage from learned model
        if symbol in self.slippage_models:
            model_slippage = self._predict_slippage_from_model(symbol, size_usd)
        else:
            # Fallback to simple model
            model_slippage = market_impact_bps + spread_cost_bps
        
        # Combine estimates
        expected_slippage_bps = model_slippage
        
        # Calculate fill probability
        fill_probability = self._calculate_fill_probability(
            symbol,
            size_usd,
            order_book,
            order_type,
            direction
        )
        
        # Calculate confidence
        confidence = self._calculate_confidence(symbol, size_usd)
        
        estimate = SlippageEstimate(
            expected_slippage_bps=expected_slippage_bps,
            confidence=confidence,
            market_impact_bps=market_impact_bps,
            spread_cost_bps=spread_cost_bps,
            fill_probability=fill_probability,
            metadata={
                "order_type": order_type.value,
                "direction": direction,
                "size_usd": size_usd
            }
        )
        
        logger.debug(
            "slippage_estimated",
            symbol=symbol,
            direction=direction,
            size_usd=size_usd,
            expected_slippage_bps=expected_slippage_bps,
            fill_probability=fill_probability
        )
        
        return estimate
    
    def simulate_execution(
        self,
        symbol: str,
        direction: str,
        size_usd: float,
        requested_price: float,
        order_book: Optional[OrderBookSnapshot] = None,
        order_type: OrderType = OrderType.MARKET
    ) -> ExecutionResult:
        """
        Simulate order execution.
        
        Args:
            symbol: Trading symbol
            direction: Order direction
            size_usd: Order size in USD
            requested_price: Requested price
            order_book: Order book snapshot
            order_type: Order type
        
        Returns:
            ExecutionResult
        """
        # Estimate slippage
        estimate = self.estimate_slippage(
            symbol=symbol,
            direction=direction,
            size_usd=size_usd,
            order_book=order_book,
            order_type=order_type
        )
        
        # Check fill probability
        if estimate.fill_probability < self.min_fill_probability:
            # Order may not fill
            executed_price = requested_price
            slippage_bps = 0.0
        else:
            # Calculate executed price with slippage
            if direction == "buy":
                slippage_multiplier = 1 + (estimate.expected_slippage_bps / 10000)
                executed_price = requested_price * slippage_multiplier
            else:
                slippage_multiplier = 1 - (estimate.expected_slippage_bps / 10000)
                executed_price = requested_price * slippage_multiplier
            
            slippage_bps = estimate.expected_slippage_bps
        
        # Add fees
        if order_type == OrderType.TAKER or order_type == OrderType.MARKET:
            fee_bps = self.taker_fee_bps
        else:
            fee_bps = self.maker_fee_bps
        
        # Total cost
        total_cost_bps = slippage_bps + fee_bps
        
        # Execution time (simplified)
        execution_time_ms = 10.0 if order_type == OrderType.MARKET else 100.0
        
        result = ExecutionResult(
            symbol=symbol,
            order_type=order_type,
            direction=direction,
            size_usd=size_usd,
            requested_price=requested_price,
            executed_price=executed_price,
            slippage_bps=slippage_bps,
            fill_probability=estimate.fill_probability,
            execution_time_ms=execution_time_ms,
            metadata={
                "total_cost_bps": total_cost_bps,
                "fee_bps": fee_bps,
                "market_impact_bps": estimate.market_impact_bps
            }
        )
        
        return result
    
    def _calculate_market_impact(
        self,
        symbol: str,
        size_usd: float,
        order_book: Optional[OrderBookSnapshot]
    ) -> float:
        """Calculate market impact"""
        if order_book is None:
            # Fallback: use size-based impact
            return size_usd / 10000.0 * self.market_impact_factor
        
        # Calculate available liquidity
        # For buy orders, need ask liquidity; for sell orders, need bid liquidity
        # Note: direction parameter determines which side to use
        available_liquidity = sum(order_book.ask_sizes[:5]) if len(order_book.ask_sizes) >= 5 else sum(order_book.ask_sizes)
        
        # Market impact based on size vs liquidity
        if available_liquidity > 0:
            impact_ratio = size_usd / (available_liquidity * order_book.mid_price)
            market_impact_bps = impact_ratio * 100 * self.market_impact_factor
        else:
            market_impact_bps = 50.0  # High impact if no liquidity
        
        return float(market_impact_bps)
    
    def _calculate_fill_probability(
        self,
        symbol: str,
        size_usd: float,
        order_book: Optional[OrderBookSnapshot],
        order_type: OrderType,
        direction: Optional[str] = None
    ) -> float:
        """Calculate fill probability"""
        if order_type == OrderType.MARKET:
            return 1.0  # Market orders always fill (in simulation)
        
        if order_book is None:
            return 0.8  # Default fill probability
        
        # Calculate available liquidity
        # For buy orders, need ask liquidity; for sell orders, need bid liquidity
        if len(order_book.ask_sizes) >= 5:
            ask_liquidity = sum(order_book.ask_sizes[:5])
        else:
            ask_liquidity = sum(order_book.ask_sizes) if order_book.ask_sizes else 0.0
        
        if len(order_book.bid_sizes) >= 5:
            bid_liquidity = sum(order_book.bid_sizes[:5])
        else:
            bid_liquidity = sum(order_book.bid_sizes) if order_book.bid_sizes else 0.0
        
        # Use appropriate side based on direction
        if direction == "buy":
            available_liquidity = ask_liquidity  # Buy orders consume ask liquidity
        elif direction == "sell":
            available_liquidity = bid_liquidity  # Sell orders consume bid liquidity
        else:
            # Default: use average
            available_liquidity = (ask_liquidity + bid_liquidity) / 2.0 if (ask_liquidity > 0 or bid_liquidity > 0) else 0.0
        
        # Fill probability based on size vs liquidity
        if available_liquidity > 0:
            fill_ratio = min(1.0, (available_liquidity * order_book.mid_price) / size_usd)
            fill_probability = fill_ratio * 0.9  # 90% of available liquidity
        else:
            fill_probability = 0.1
        
        return float(fill_probability)
    
    def _calculate_confidence(self, symbol: str, size_usd: float) -> float:
        """Calculate confidence in slippage estimate"""
        if symbol not in self.slippage_history:
            return 0.5  # Low confidence if no history
        
        # More samples = higher confidence
        num_samples = len(self.slippage_history[symbol])
        confidence = min(1.0, num_samples / 100.0)
        
        # Higher confidence for similar sizes
        similar_sizes = sum(1 for size, _ in self.slippage_history[symbol] 
                           if abs(size - size_usd) / size_usd < 0.2)
        size_confidence = min(1.0, similar_sizes / 20.0)
        
        return (confidence + size_confidence) / 2
    
    def _learn_slippage_model(self, symbol: str) -> None:
        """Learn slippage model from history"""
        if symbol not in self.slippage_history:
            return
        
        data = self.slippage_history[symbol]
        if len(data) < 10:
            return
        
        sizes = np.array([d[0] for d in data])
        slippages = np.array([d[1] for d in data])
        
        # Simple linear model: slippage = a * size + b
        # In production, would use more sophisticated models
        if len(sizes) > 1:
            coeffs = np.polyfit(sizes, slippages, 1)
            self.slippage_models[symbol] = {
                "slope": float(coeffs[0]),
                "intercept": float(coeffs[1])
            }
        else:
            self.slippage_models[symbol] = {
                "slope": 0.0,
                "intercept": float(np.mean(slippages))
            }
    
    def _predict_slippage_from_model(self, symbol: str, size_usd: float) -> float:
        """Predict slippage from learned model"""
        if symbol not in self.slippage_models:
            return 5.0  # Default slippage
        
        model = self.slippage_models[symbol]
        slippage = model["slope"] * size_usd + model["intercept"]
        
        return float(max(0.0, slippage))

