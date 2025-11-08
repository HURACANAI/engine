"""
In-Memory Order Book

Replicated order books for fault tolerance and speed.
Sync via in-memory streams (Redis/Kafka interface).

Key Features:
- In-memory order book storage
- Replicated order books for fault tolerance
- Fast lookups and updates
- Redis/Kafka sync interface
- Order book snapshots
- Depth analysis
- Imbalance calculation

Author: Huracan Engine Team
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Deque
from datetime import datetime, timezone
from collections import deque, defaultdict
from enum import Enum
import threading
import structlog

logger = structlog.get_logger(__name__)


class OrderBookSide(Enum):
    """Order book side"""
    BID = "bid"
    ASK = "ask"


@dataclass
class OrderBookLevel:
    """Order book level"""
    price: float
    size: float
    orders: int = 1  # Number of orders at this price level


@dataclass
class OrderBookSnapshot:
    """Order book snapshot"""
    symbol: str
    timestamp: datetime
    bids: List[OrderBookLevel] = field(default_factory=list)
    asks: List[OrderBookLevel] = field(default_factory=list)
    mid_price: float = 0.0
    spread_bps: float = 0.0
    best_bid: float = 0.0
    best_ask: float = 0.0


class InMemoryOrderBook:
    """
    In-Memory Order Book.
    
    Fast, in-memory order book with replication support.
    
    Usage:
        orderbook = InMemoryOrderBook(symbol="BTC-USD")
        
        # Update order book
        orderbook.update_bid(price=50000.0, size=1.0)
        orderbook.update_ask(price=50001.0, size=1.0)
        
        # Get snapshot
        snapshot = orderbook.get_snapshot()
        
        # Get depth
        depth = orderbook.get_depth(levels=10)
    """
    
    def __init__(
        self,
        symbol: str,
        max_levels: int = 100
    ):
        """
        Initialize in-memory order book.
        
        Args:
            symbol: Trading symbol
            max_levels: Maximum price levels to keep
        """
        self.symbol = symbol
        self.max_levels = max_levels
        
        # Order book data (price -> size)
        self.bids: Dict[float, float] = {}  # price -> size
        self.asks: Dict[float, float] = {}  # price -> size
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Snapshot history
        self.snapshot_history: Deque[OrderBookSnapshot] = deque(maxlen=1000)
        
        logger.info(
            "in_memory_orderbook_initialized",
            symbol=symbol,
            max_levels=max_levels
        )
    
    def update_bid(self, price: float, size: float) -> None:
        """Update bid level"""
        with self.lock:
            if size == 0:
                self.bids.pop(price, None)
            else:
                self.bids[price] = size
            
            # Maintain max levels
            if len(self.bids) > self.max_levels:
                # Remove lowest prices
                sorted_prices = sorted(self.bids.keys(), reverse=True)
                for price_to_remove in sorted_prices[self.max_levels:]:
                    del self.bids[price_to_remove]
    
    def update_ask(self, price: float, size: float) -> None:
        """Update ask level"""
        with self.lock:
            if size == 0:
                self.asks.pop(price, None)
            else:
                self.asks[price] = size
            
            # Maintain max levels
            if len(self.asks) > self.max_levels:
                # Remove highest prices
                sorted_prices = sorted(self.asks.keys())
                for price_to_remove in sorted_prices[self.max_levels:]:
                    del self.asks[price_to_remove]
    
    def get_best_bid(self) -> Optional[float]:
        """Get best bid price"""
        with self.lock:
            if not self.bids:
                return None
            return max(self.bids.keys())
    
    def get_best_ask(self) -> Optional[float]:
        """Get best ask price"""
        with self.lock:
            if not self.asks:
                return None
            return min(self.asks.keys())
    
    def get_mid_price(self) -> Optional[float]:
        """Get mid price"""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        
        if best_bid is None or best_ask is None:
            return None
        
        return (best_bid + best_ask) / 2.0
    
    def get_spread_bps(self) -> Optional[float]:
        """Get spread in basis points"""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        
        if best_bid is None or best_ask is None:
            return None
        
        mid_price = (best_bid + best_ask) / 2.0
        spread = best_ask - best_bid
        spread_bps = (spread / mid_price) * 10000
        
        return spread_bps
    
    def get_depth(self, levels: int = 10) -> Dict[str, List[Tuple[float, float]]]:
        """Get order book depth"""
        with self.lock:
            # Get top N bids (highest prices)
            sorted_bids = sorted(self.bids.items(), key=lambda x: x[0], reverse=True)
            top_bids = sorted_bids[:levels]
            
            # Get top N asks (lowest prices)
            sorted_asks = sorted(self.asks.items(), key=lambda x: x[0])
            top_asks = sorted_asks[:levels]
            
            return {
                "bids": [(price, size) for price, size in top_bids],
                "asks": [(price, size) for price, size in top_asks]
            }
    
    def get_depth_usd(self, levels: int = 10) -> Dict[str, float]:
        """Get order book depth in USD"""
        depth = self.get_depth(levels=levels)
        
        bid_depth = sum(price * size for price, size in depth["bids"])
        ask_depth = sum(price * size for price, size in depth["asks"])
        
        return {
            "bid_depth_usd": bid_depth,
            "ask_depth_usd": ask_depth,
            "total_depth_usd": bid_depth + ask_depth
        }
    
    def get_imbalance(self, levels: int = 10) -> float:
        """Get order book imbalance (-1 to 1)"""
        depth = self.get_depth_usd(levels=levels)
        
        bid_depth = depth["bid_depth_usd"]
        ask_depth = depth["ask_depth_usd"]
        total_depth = depth["total_depth_usd"]
        
        if total_depth == 0:
            return 0.0
        
        imbalance = (bid_depth - ask_depth) / total_depth
        return imbalance
    
    def get_snapshot(self) -> OrderBookSnapshot:
        """Get order book snapshot"""
        with self.lock:
            best_bid = self.get_best_bid()
            best_ask = self.get_best_ask()
            mid_price = self.get_mid_price()
            spread_bps = self.get_spread_bps()
            
            # Convert to levels
            bids = [
                OrderBookLevel(price=price, size=size)
                for price, size in sorted(self.bids.items(), reverse=True)
            ]
            asks = [
                OrderBookLevel(price=price, size=size)
                for price, size in sorted(self.asks.items())
            ]
            
            snapshot = OrderBookSnapshot(
                symbol=self.symbol,
                timestamp=datetime.now(timezone.utc),
                bids=bids,
                asks=asks,
                mid_price=mid_price or 0.0,
                spread_bps=spread_bps or 0.0,
                best_bid=best_bid or 0.0,
                best_ask=best_ask or 0.0
            )
            
            # Store in history
            self.snapshot_history.append(snapshot)
            
            return snapshot
    
    def clear(self) -> None:
        """Clear order book"""
        with self.lock:
            self.bids.clear()
            self.asks.clear()


class OrderBookManager:
    """
    Order Book Manager.
    
    Manages multiple order books and replication.
    
    Usage:
        manager = OrderBookManager()
        
        # Add order book
        manager.add_orderbook("BTC-USD")
        
        # Update order book
        manager.update_orderbook("BTC-USD", side="bid", price=50000.0, size=1.0)
        
        # Get snapshot
        snapshot = manager.get_snapshot("BTC-USD")
    """
    
    def __init__(self):
        """Initialize order book manager"""
        self.orderbooks: Dict[str, InMemoryOrderBook] = {}
        self.lock = threading.Lock()
        
        logger.info("orderbook_manager_initialized")
    
    def add_orderbook(self, symbol: str, max_levels: int = 100) -> None:
        """Add order book for symbol"""
        with self.lock:
            if symbol not in self.orderbooks:
                self.orderbooks[symbol] = InMemoryOrderBook(symbol=symbol, max_levels=max_levels)
                logger.info("orderbook_added", symbol=symbol)
    
    def update_orderbook(
        self,
        symbol: str,
        side: OrderBookSide,
        price: float,
        size: float
    ) -> None:
        """Update order book"""
        with self.lock:
            if symbol not in self.orderbooks:
                self.add_orderbook(symbol)
            
            orderbook = self.orderbooks[symbol]
            
            if side == OrderBookSide.BID:
                orderbook.update_bid(price, size)
            else:
                orderbook.update_ask(price, size)
    
    def get_snapshot(self, symbol: str) -> Optional[OrderBookSnapshot]:
        """Get order book snapshot"""
        with self.lock:
            if symbol not in self.orderbooks:
                return None
            
            return self.orderbooks[symbol].get_snapshot()
    
    def get_all_snapshots(self) -> Dict[str, OrderBookSnapshot]:
        """Get snapshots for all order books"""
        with self.lock:
            return {
                symbol: orderbook.get_snapshot()
                for symbol, orderbook in self.orderbooks.items()
            }


class OrderBookReplicator:
    """
    Order Book Replicator.
    
    Interface for replicating order books via Redis/Kafka.
    
    Usage:
        replicator = OrderBookReplicator(backend="redis")
        
        # Replicate order book
        replicator.replicate(snapshot)
        
        # Sync from replica
        snapshot = replicator.sync_from_replica(symbol)
    """
    
    def __init__(self, backend: str = "redis"):
        """
        Initialize order book replicator.
        
        Args:
            backend: Backend type ("redis", "kafka", "memory")
        """
        self.backend = backend
        self.replica_storage: Dict[str, OrderBookSnapshot] = {}  # For memory backend
        
        logger.info("orderbook_replicator_initialized", backend=backend)
    
    def replicate(self, snapshot: OrderBookSnapshot) -> None:
        """Replicate order book snapshot"""
        if self.backend == "memory":
            self.replica_storage[snapshot.symbol] = snapshot
        elif self.backend == "redis":
            self._replicate_redis(snapshot)
        elif self.backend == "kafka":
            self._replicate_kafka(snapshot)
        else:
            logger.warning("unknown_backend", backend=self.backend)
    
    def sync_from_replica(self, symbol: str) -> Optional[OrderBookSnapshot]:
        """Sync order book from replica"""
        if self.backend == "memory":
            return self.replica_storage.get(symbol)
        elif self.backend == "redis":
            return self._sync_from_redis(symbol)
        elif self.backend == "kafka":
            return self._sync_from_kafka(symbol)
        else:
            return None
    
    def _replicate_redis(self, snapshot: OrderBookSnapshot) -> None:
        """Replicate to Redis (placeholder)"""
        # In production, would use redis-py
        logger.debug("redis_replication_not_implemented", symbol=snapshot.symbol)
    
    def _replicate_kafka(self, snapshot: OrderBookSnapshot) -> None:
        """Replicate to Kafka (placeholder)"""
        # In production, would use kafka-python
        logger.debug("kafka_replication_not_implemented", symbol=snapshot.symbol)
    
    def _sync_from_redis(self, symbol: str) -> Optional[OrderBookSnapshot]:
        """Sync from Redis (placeholder)"""
        # In production, would use redis-py
        logger.debug("redis_sync_not_implemented", symbol=symbol)
        return None
    
    def _sync_from_kafka(self, symbol: str) -> Optional[OrderBookSnapshot]:
        """Sync from Kafka (placeholder)"""
        # In production, would use kafka-python
        logger.debug("kafka_sync_not_implemented", symbol=symbol)
        return None

