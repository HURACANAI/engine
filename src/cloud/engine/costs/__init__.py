"""
Transaction Cost Analysis (TCA) for Huracan V2

Realistic cost modeling is CRITICAL for profitable trading. This module
provides accurate cost estimation including:
- Exchange fees (maker/taker, historical schedules)
- Spread paid (bid-ask spread)
- Slippage (market impact, volatility-based)
- Partial fill delays

Without realistic costs, your backtest will show profits that evaporate in live trading.
"""

from .realistic_tca import CostEstimator, TCAReport, print_tca_report

__all__ = ['CostEstimator', 'TCAReport', 'print_tca_report']
