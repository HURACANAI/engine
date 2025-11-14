"""
Execution Simulator

Simulates realistic order execution with slippage, partial fills, and latency.

Key Features:
- Realistic slippage modeling by order size
- Partial fill simulation
- Latency simulation
- Fee calculation
- Market impact estimation

Usage:
    from src.cloud.training.portfolio.execution_sim import ExecutionSimulator

    sim = ExecutionSimulator()

    # Simulate market order
    result = sim.simulate_execution(
        order_type="MARKET",
        side="BUY",
        quantity=1.5,  # BTC
        market_price=50000,
        book_liquidity={"bid_depth": 10, "ask_depth": 15},
        volatility=0.02
    )

    print(f"Fill price: ${result.avg_fill_price:.2f}")
    print(f"Slippage: {result.slippage_bps:.1f} bps")
    print(f"Filled: {result.filled_quantity}/{result.requested_quantity}")
"""

from .simulator import (
    ExecutionSimulator,
    ExecutionResult,
    OrderType
)

__all__ = [
    "ExecutionSimulator",
    "ExecutionResult",
    "OrderType",
]
