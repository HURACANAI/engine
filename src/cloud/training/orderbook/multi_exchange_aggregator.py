"""
Multi-Exchange Orderbook Aggregator

Aggregates orderbooks from multiple exchanges to find best prices and liquidity.
Provides unified view across exchanges for optimal execution.

Key Features:
- Aggregate bids/asks from multiple exchanges
- Calculate best bid/ask across all exchanges
- Depth aggregation (total liquidity at each price level)
- Exchange selection based on best price
- Latency-weighted aggregation

Author: Huracan Engine Team
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import numpy as np
import structlog

from .in_memory_orderbook import OrderBookSnapshot, OrderBookLevel, OrderBookSide

logger = structlog.get_logger(__name__)


@dataclass
class ExchangeOrderBook:
    """Orderbook from a single exchange."""
    exchange: str
    symbol: str
    snapshot: OrderBookSnapshot
    latency_ms: float  # Exchange latency in milliseconds
    timestamp: datetime
    reliability_score: float  # 0-1, exchange reliability


@dataclass
class AggregatedLevel:
    """Aggregated orderbook level across exchanges."""
    price: float
    total_size: float
    exchanges: List[str]  # Exchanges contributing to this level
    weighted_price: float  # Latency-weighted price


@dataclass
class AggregatedOrderBook:
    """Aggregated orderbook across multiple exchanges."""
    symbol: str
    timestamp: datetime
    bids: List[AggregatedLevel]
    asks: List[AggregatedLevel]
    best_bid: float
    best_ask: float
    spread_bps: float
    total_bid_depth: float
    total_ask_depth: float
    exchange_count: int


class MultiExchangeOrderbookAggregator:
    """
    Aggregates orderbooks from multiple exchanges.
    
    Usage:
        aggregator = MultiExchangeOrderbookAggregator()
        
        # Add exchange orderbooks
        aggregator.add_exchange_orderbook(
            exchange="binance",
            snapshot=binance_snapshot,
            latency_ms=50.0
        )
        aggregator.add_exchange_orderbook(
            exchange="kraken",
            snapshot=kraken_snapshot,
            latency_ms=80.0
        )
        
        # Get aggregated orderbook
        aggregated = aggregator.aggregate(symbol="BTC/USDT")
        
        # Get best price across exchanges
        best_bid, best_ask = aggregated.best_bid, aggregated.best_ask
    """
    
    def __init__(
        self,
        latency_weight: float = 0.3,  # Weight for latency in aggregation
        min_reliability: float = 0.5,  # Minimum exchange reliability
        max_price_diff_pct: float = 0.01  # Max price difference to aggregate (1%)
    ):
        """
        Initialize multi-exchange aggregator.
        
        Args:
            latency_weight: Weight for latency in price weighting (0-1)
            min_reliability: Minimum exchange reliability to include
            max_price_diff_pct: Maximum price difference to aggregate levels
        """
        self.latency_weight = latency_weight
        self.min_reliability = min_reliability
        self.max_price_diff_pct = max_price_diff_pct
        
        # Store orderbooks by symbol -> exchange -> ExchangeOrderBook
        self.orderbooks: Dict[str, Dict[str, ExchangeOrderBook]] = defaultdict(dict)
        
        logger.info(
            "multi_exchange_aggregator_initialized",
            latency_weight=latency_weight,
            min_reliability=min_reliability
        )
    
    def add_exchange_orderbook(
        self,
        exchange: str,
        snapshot: OrderBookSnapshot,
        latency_ms: float = 50.0,
        reliability_score: float = 1.0
    ) -> None:
        """
        Add orderbook from an exchange.
        
        Args:
            exchange: Exchange name (e.g., "binance", "kraken")
            snapshot: Orderbook snapshot
            latency_ms: Exchange latency in milliseconds
            reliability_score: Exchange reliability (0-1)
        """
        if reliability_score < self.min_reliability:
            logger.warning(
                "exchange_below_reliability_threshold",
                exchange=exchange,
                reliability=reliability_score
            )
            return
        
        exchange_ob = ExchangeOrderBook(
            exchange=exchange,
            symbol=snapshot.symbol,
            snapshot=snapshot,
            latency_ms=latency_ms,
            timestamp=datetime.now(),
            reliability_score=reliability_score
        )
        
        self.orderbooks[snapshot.symbol][exchange] = exchange_ob
        
        logger.debug(
            "exchange_orderbook_added",
            exchange=exchange,
            symbol=snapshot.symbol,
            latency_ms=latency_ms
        )
    
    def aggregate(
        self,
        symbol: str,
        max_levels: int = 20,
        use_latency_weighting: bool = True
    ) -> AggregatedOrderBook:
        """
        Aggregate orderbooks for a symbol.
        
        Args:
            symbol: Trading symbol
            max_levels: Maximum price levels to return
            use_latency_weighting: Whether to weight by latency
        
        Returns:
            AggregatedOrderBook with best prices across exchanges
        """
        if symbol not in self.orderbooks:
            raise ValueError(f"No orderbooks found for symbol {symbol}")
        
        exchange_obs = self.orderbooks[symbol]
        
        if not exchange_obs:
            raise ValueError(f"No valid orderbooks for symbol {symbol}")
        
        # Aggregate bids
        aggregated_bids = self._aggregate_side(
            exchange_obs,
            side=OrderBookSide.BID,
            max_levels=max_levels,
            use_latency_weighting=use_latency_weighting
        )
        
        # Aggregate asks
        aggregated_asks = self._aggregate_side(
            exchange_obs,
            side=OrderBookSide.ASK,
            max_levels=max_levels,
            use_latency_weighting=use_latency_weighting
        )
        
        # Calculate best bid/ask
        best_bid = aggregated_bids[0].price if aggregated_bids else 0.0
        best_ask = aggregated_asks[0].price if aggregated_asks else 0.0
        
        # Calculate spread
        spread_bps = ((best_ask - best_bid) / best_bid * 10000) if best_bid > 0 else 0.0
        
        # Calculate total depth
        total_bid_depth = sum(level.total_size for level in aggregated_bids)
        total_ask_depth = sum(level.total_size for level in aggregated_asks)
        
        aggregated = AggregatedOrderBook(
            symbol=symbol,
            timestamp=datetime.now(),
            bids=aggregated_bids,
            asks=aggregated_asks,
            best_bid=best_bid,
            best_ask=best_ask,
            spread_bps=spread_bps,
            total_bid_depth=total_bid_depth,
            total_ask_depth=total_ask_depth,
            exchange_count=len(exchange_obs)
        )
        
        logger.info(
            "orderbook_aggregated",
            symbol=symbol,
            exchanges=len(exchange_obs),
            best_bid=best_bid,
            best_ask=best_ask,
            spread_bps=spread_bps
        )
        
        return aggregated
    
    def _aggregate_side(
        self,
        exchange_obs: Dict[str, ExchangeOrderBook],
        side: OrderBookSide,
        max_levels: int,
        use_latency_weighting: bool
    ) -> List[AggregatedLevel]:
        """Aggregate one side (bids or asks) of the orderbook."""
        # Collect all price levels from all exchanges
        price_levels: Dict[float, Dict[str, float]] = defaultdict(dict)  # price -> exchange -> size
        
        for exchange, exchange_ob in exchange_obs.items():
            snapshot = exchange_ob.snapshot
            
            if side == OrderBookSide.BID:
                levels = snapshot.bids
            else:
                levels = snapshot.asks
            
            for level in levels:
                price = level.price
                size = level.size
                
                # Find matching price level (within max_price_diff_pct)
                matching_price = self._find_matching_price(
                    price,
                    list(price_levels.keys()),
                    side
                )
                
                if matching_price is not None:
                    price_levels[matching_price][exchange] = size
                else:
                    price_levels[price][exchange] = size
        
        # Aggregate levels
        aggregated_levels = []
        for price, exchange_sizes in price_levels.items():
            total_size = sum(exchange_sizes.values())
            exchanges = list(exchange_sizes.keys())
            
            # Calculate latency-weighted price
            if use_latency_weighting and exchanges:
                weights = []
                prices = []
                for exchange in exchanges:
                    exchange_ob = exchange_obs[exchange]
                    # Lower latency = higher weight
                    weight = 1.0 / (exchange_ob.latency_ms + 1.0)
                    weights.append(weight)
                    prices.append(price)
                
                weights = np.array(weights)
                weights = weights / weights.sum()  # Normalize
                weighted_price = np.average(prices, weights=weights)
            else:
                weighted_price = price
            
            aggregated_levels.append(AggregatedLevel(
                price=price,
                total_size=total_size,
                exchanges=exchanges,
                weighted_price=weighted_price
            ))
        
        # Sort by price (descending for bids, ascending for asks)
        if side == OrderBookSide.BID:
            aggregated_levels.sort(key=lambda x: x.price, reverse=True)
        else:
            aggregated_levels.sort(key=lambda x: x.price)
        
        # Return top N levels
        return aggregated_levels[:max_levels]
    
    def _find_matching_price(
        self,
        price: float,
        existing_prices: List[float],
        side: OrderBookSide
    ) -> Optional[float]:
        """Find matching price level within max_price_diff_pct."""
        for existing_price in existing_prices:
            price_diff_pct = abs(price - existing_price) / existing_price
            if price_diff_pct <= self.max_price_diff_pct:
                return existing_price
        return None
    
    def get_best_exchange(
        self,
        symbol: str,
        side: str,  # "buy" or "sell"
        size_usd: float
    ) -> Tuple[str, float]:
        """
        Get best exchange for a trade.
        
        Args:
            symbol: Trading symbol
            side: "buy" or "sell"
            size_usd: Trade size in USD
        
        Returns:
            Tuple of (best_exchange, best_price)
        """
        aggregated = self.aggregate(symbol)
        
        if side.lower() == "buy":
            # For buying, we want the lowest ask
            best_price = aggregated.best_ask
            levels = aggregated.asks
        else:
            # For selling, we want the highest bid
            best_price = aggregated.best_bid
            levels = aggregated.bids
        
        # Find exchange with best price and sufficient liquidity
        best_exchange = None
        for level in levels:
            if level.total_size * best_price >= size_usd:
                # Use first exchange at this level (could be improved with routing)
                best_exchange = level.exchanges[0] if level.exchanges else None
                break
        
        if not best_exchange:
            # Fallback to exchange with best price
            if levels:
                best_exchange = levels[0].exchanges[0] if levels[0].exchanges else None
        
        logger.info(
            "best_exchange_selected",
            symbol=symbol,
            side=side,
            exchange=best_exchange,
            price=best_price
        )
        
        return best_exchange or "unknown", best_price
    
    def get_available_exchanges(self, symbol: str) -> List[str]:
        """Get list of available exchanges for a symbol."""
        if symbol not in self.orderbooks:
            return []
        return list(self.orderbooks[symbol].keys())
    
    def remove_exchange(self, symbol: str, exchange: str) -> None:
        """Remove exchange orderbook."""
        if symbol in self.orderbooks and exchange in self.orderbooks[symbol]:
            del self.orderbooks[symbol][exchange]
            logger.info("exchange_removed", symbol=symbol, exchange=exchange)

