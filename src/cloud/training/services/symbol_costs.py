"""
Symbol Costs Service

Fetches per-symbol costs (fees, spread, slippage) before training.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, Optional

import structlog

logger = structlog.get_logger(__name__)


def get_symbol_costs(
    symbol: str,
    config: Optional[Dict] = None,
) -> Dict[str, float]:
    """Get costs for a symbol.
    
    Args:
        symbol: Trading symbol
        config: Configuration dictionary (optional)
        
    Returns:
        Dictionary with cost components
    """
    if config is None:
        from src.shared.config_loader import load_config
        config = load_config()
    
    # Get default costs from config
    costs_config = config.get("costs", {})
    
    # Check for per-symbol overrides
    per_symbol_costs = config.get("engine", {}).get("per_coin", {}).get("per_symbol_costs", {})
    
    if symbol in per_symbol_costs:
        # Use per-symbol override
        symbol_costs = per_symbol_costs[symbol]
        costs = {
            "taker_fee_bps": symbol_costs.get("taker_fee_bps", costs_config.get("taker_fee_bps", 4.0)),
            "maker_fee_bps": symbol_costs.get("maker_fee_bps", costs_config.get("maker_fee_bps", 2.0)),
            "median_spread_bps": symbol_costs.get("median_spread_bps", costs_config.get("median_spread_bps", 5.0)),
            "slippage_bps_per_sigma": symbol_costs.get("slippage_bps_per_sigma", costs_config.get("slippage_bps_per_sigma", 2.0)),
        }
    else:
        # Use defaults
        costs = {
            "taker_fee_bps": costs_config.get("taker_fee_bps", 4.0),
            "maker_fee_bps": costs_config.get("maker_fee_bps", 2.0),
            "median_spread_bps": costs_config.get("median_spread_bps", 5.0),
            "slippage_bps_per_sigma": costs_config.get("slippage_bps_per_sigma", 2.0),
        }
    
    # TODO: Fetch real-time costs from exchange if available
    # For now, use configured defaults
    
    logger.debug("symbol_costs_retrieved", symbol=symbol, costs=costs)
    return costs

