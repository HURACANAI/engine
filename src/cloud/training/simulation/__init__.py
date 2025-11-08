"""
Simulation Module

Execution simulation and live trading simulation.
"""

from .execution_simulator import (
    ExecutionSimulator,
    OrderType,
    OrderBookSnapshot,
    SlippageEstimate,
    ExecutionResult,
)

from .live_simulator import (
    LiveSimulator,
    TransactionCosts,
    LiveTradeResult,
    FeeType,
)

__all__ = [
    "ExecutionSimulator",
    "OrderType",
    "OrderBookSnapshot",
    "SlippageEstimate",
    "ExecutionResult",
    "LiveSimulator",
    "TransactionCosts",
    "LiveTradeResult",
    "FeeType",
]
