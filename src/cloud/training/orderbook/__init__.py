"""
Order Book Module

In-memory order books with replication support and multi-exchange aggregation.
"""

from .in_memory_orderbook import (
    InMemoryOrderBook,
    OrderBookManager,
    OrderBookReplicator,
    OrderBookSnapshot,
    OrderBookLevel,
    OrderBookSide,
)
from .multi_exchange_aggregator import (
    MultiExchangeOrderbookAggregator,
    AggregatedOrderBook,
    AggregatedLevel,
    ExchangeOrderBook,
)

__all__ = [
    "InMemoryOrderBook",
    "OrderBookManager",
    "OrderBookReplicator",
    "OrderBookSnapshot",
    "OrderBookLevel",
    "OrderBookSide",
    "MultiExchangeOrderbookAggregator",
    "AggregatedOrderBook",
    "AggregatedLevel",
    "ExchangeOrderBook",
]

