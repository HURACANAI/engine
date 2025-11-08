"""
Portfolio Allocator

Creates equal-weight or value-weight portfolios using matrix math.
Uses NumPy vectorization to allocate capital efficiently.

Key Features:
- Equal-weight allocation
- Value-weight allocation
- Risk-parity allocation
- Capital efficiency
- Diversification limits
- Correlation constraints

Author: Huracan Engine Team
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


class AllocationMethod(Enum):
    """Allocation method"""
    EQUAL_WEIGHT = "equal_weight"
    VALUE_WEIGHT = "value_weight"
    RISK_PARITY = "risk_parity"
    MARKET_CAP_WEIGHT = "market_cap_weight"
    CUSTOM = "custom"


@dataclass
class PortfolioAllocation:
    """Portfolio allocation result"""
    symbols: List[str]
    weights: Dict[str, float]
    allocation_method: AllocationMethod
    total_capital: float
    allocations_usd: Dict[str, float]
    diversification_score: float
    metadata: Dict[str, any] = field(default_factory=dict)


@dataclass
class AllocationConstraints:
    """Allocation constraints"""
    max_weight_per_symbol: float = 0.20  # 20% max per symbol
    min_weight_per_symbol: float = 0.01  # 1% min per symbol
    max_correlation: float = 0.7  # Max correlation between positions
    min_diversification: float = 0.5  # Minimum diversification score
    max_total_exposure: float = 1.0  # Max total exposure (for leverage)
    sector_limits: Dict[str, float] = field(default_factory=dict)  # Sector limits


class PortfolioAllocator:
    """
    Portfolio Allocator - Efficient capital allocation.
    
    Creates portfolios using vectorized operations for performance.
    Supports multiple allocation methods and constraints.
    
    Usage:
        allocator = PortfolioAllocator(constraints=AllocationConstraints())
        
        allocation = allocator.allocate(
            symbols=["BTCUSDT", "ETHUSDT", ...],
            total_capital=100000.0,
            method=AllocationMethod.EQUAL_WEIGHT
        )
        
        # Get allocations
        allocations_usd = allocation.allocations_usd
    """
    
    def __init__(
        self,
        constraints: Optional[AllocationConstraints] = None
    ):
        """
        Initialize portfolio allocator.
        
        Args:
            constraints: Allocation constraints
        """
        self.constraints = constraints or AllocationConstraints()
        
        logger.info(
            "portfolio_allocator_initialized",
            max_weight=self.constraints.max_weight_per_symbol,
            min_weight=self.constraints.min_weight_per_symbol
        )
    
    def allocate(
        self,
        symbols: List[str],
        total_capital: float,
        method: AllocationMethod = AllocationMethod.EQUAL_WEIGHT,
        market_caps: Optional[Dict[str, float]] = None,
        volatilities: Optional[Dict[str, float]] = None,
        correlations: Optional[Dict[str, Dict[str, float]]] = None,
        custom_weights: Optional[Dict[str, float]] = None
    ) -> PortfolioAllocation:
        """
        Allocate capital across symbols.
        
        Args:
            symbols: List of symbols to allocate
            total_capital: Total capital to allocate
            method: Allocation method
            market_caps: Market caps for value-weighting (optional)
            volatilities: Volatilities for risk-parity (optional)
            correlations: Correlation matrix (optional)
            custom_weights: Custom weights (optional)
        
        Returns:
            PortfolioAllocation
        """
        if not symbols:
            return PortfolioAllocation(
                symbols=[],
                weights={},
                allocation_method=method,
                total_capital=total_capital,
                allocations_usd={},
                diversification_score=0.0
            )
        
        # Calculate base weights
        if method == AllocationMethod.EQUAL_WEIGHT:
            weights = self._equal_weight_allocation(symbols)
        elif method == AllocationMethod.VALUE_WEIGHT:
            weights = self._value_weight_allocation(symbols, market_caps)
        elif method == AllocationMethod.RISK_PARITY:
            weights = self._risk_parity_allocation(symbols, volatilities)
        elif method == AllocationMethod.MARKET_CAP_WEIGHT:
            weights = self._market_cap_weight_allocation(symbols, market_caps)
        elif method == AllocationMethod.CUSTOM:
            weights = custom_weights or {}
        else:
            weights = self._equal_weight_allocation(symbols)
        
        # Apply constraints
        weights = self._apply_constraints(weights, correlations)
        
        # Calculate allocations in USD
        allocations_usd = {
            symbol: weights.get(symbol, 0.0) * total_capital
            for symbol in symbols
        }
        
        # Calculate diversification score
        diversification_score = self._calculate_diversification_score(weights, correlations)
        
        allocation = PortfolioAllocation(
            symbols=symbols,
            weights=weights,
            allocation_method=method,
            total_capital=total_capital,
            allocations_usd=allocations_usd,
            diversification_score=diversification_score,
            metadata={
                "num_symbols": len(symbols),
                "total_weight": sum(weights.values())
            }
        )
        
        logger.info(
            "portfolio_allocation_computed",
            method=method.value,
            num_symbols=len(symbols),
            total_capital=total_capital,
            diversification_score=diversification_score
        )
        
        return allocation
    
    def _equal_weight_allocation(self, symbols: List[str]) -> Dict[str, float]:
        """Equal-weight allocation"""
        num_symbols = len(symbols)
        weight = 1.0 / num_symbols
        return {symbol: weight for symbol in symbols}
    
    def _value_weight_allocation(
        self,
        symbols: List[str],
        market_caps: Optional[Dict[str, float]]
    ) -> Dict[str, float]:
        """Value-weight allocation"""
        if not market_caps:
            # Fallback to equal weight
            return self._equal_weight_allocation(symbols)
        
        # Calculate total market cap
        total_market_cap = sum(market_caps.get(symbol, 0.0) for symbol in symbols)
        
        if total_market_cap == 0:
            return self._equal_weight_allocation(symbols)
        
        # Weight by market cap
        weights = {}
        for symbol in symbols:
            market_cap = market_caps.get(symbol, 0.0)
            weights[symbol] = market_cap / total_market_cap
        
        return weights
    
    def _risk_parity_allocation(
        self,
        symbols: List[str],
        volatilities: Optional[Dict[str, float]]
    ) -> Dict[str, float]:
        """Risk-parity allocation (inverse volatility weighting)"""
        if not volatilities:
            # Fallback to equal weight
            return self._equal_weight_allocation(symbols)
        
        # Calculate inverse volatilities
        inverse_vols = {}
        for symbol in symbols:
            vol = volatilities.get(symbol, 0.01)
            if vol > 0:
                inverse_vols[symbol] = 1.0 / vol
            else:
                inverse_vols[symbol] = 0.0
        
        # Normalize to sum to 1.0
        total_inverse_vol = sum(inverse_vols.values())
        if total_inverse_vol == 0:
            return self._equal_weight_allocation(symbols)
        
        weights = {}
        for symbol in symbols:
            weights[symbol] = inverse_vols.get(symbol, 0.0) / total_inverse_vol
        
        return weights
    
    def _market_cap_weight_allocation(
        self,
        symbols: List[str],
        market_caps: Optional[Dict[str, float]]
    ) -> Dict[str, float]:
        """Market cap weight allocation (same as value weight)"""
        return self._value_weight_allocation(symbols, market_caps)
    
    def _apply_constraints(
        self,
        weights: Dict[str, float],
        correlations: Optional[Dict[str, Dict[str, float]]]
    ) -> Dict[str, float]:
        """Apply allocation constraints"""
        constrained_weights = weights.copy()
        
        # Apply max weight constraint
        for symbol in constrained_weights:
            if constrained_weights[symbol] > self.constraints.max_weight_per_symbol:
                constrained_weights[symbol] = self.constraints.max_weight_per_symbol
        
        # Apply min weight constraint (remove symbols below threshold)
        symbols_to_remove = []
        for symbol in constrained_weights:
            if constrained_weights[symbol] < self.constraints.min_weight_per_symbol:
                symbols_to_remove.append(symbol)
        
        for symbol in symbols_to_remove:
            del constrained_weights[symbol]
        
        # Normalize weights to sum to 1.0
        total_weight = sum(constrained_weights.values())
        if total_weight > 0:
            constrained_weights = {
                symbol: weight / total_weight
                for symbol, weight in constrained_weights.items()
            }
        else:
            # Fallback to equal weight if all removed
            if constrained_weights:
                num_symbols = len(constrained_weights)
                constrained_weights = {
                    symbol: 1.0 / num_symbols
                    for symbol in constrained_weights.keys()
                }
        
        # Apply correlation constraints (simplified - would use optimization)
        if correlations:
            constrained_weights = self._apply_correlation_constraints(
                constrained_weights,
                correlations
            )
        
        return constrained_weights
    
    def _apply_correlation_constraints(
        self,
        weights: Dict[str, float],
        correlations: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """Apply correlation constraints (simplified)"""
        # Reduce weights for highly correlated symbols
        adjusted_weights = weights.copy()
        
        for symbol1 in weights:
            for symbol2 in weights:
                if symbol1 != symbol2:
                    corr = correlations.get(symbol1, {}).get(symbol2, 0.0)
                    if abs(corr) > self.constraints.max_correlation:
                        # Reduce weight for more correlated symbol
                        if weights[symbol1] > weights[symbol2]:
                            adjusted_weights[symbol1] *= 0.8
                        else:
                            adjusted_weights[symbol2] *= 0.8
        
        # Renormalize
        total_weight = sum(adjusted_weights.values())
        if total_weight > 0:
            adjusted_weights = {
                symbol: weight / total_weight
                for symbol, weight in adjusted_weights.items()
            }
        
        return adjusted_weights
    
    def _calculate_diversification_score(
        self,
        weights: Dict[str, float],
        correlations: Optional[Dict[str, Dict[str, float]]]
    ) -> float:
        """Calculate diversification score (0-1, higher is better)"""
        if not weights:
            return 0.0
        
        # Number of symbols component
        num_symbols = len(weights)
        symbol_score = min(1.0, num_symbols / 20.0)  # Normalize to 20 symbols
        
        # Weight concentration component (lower concentration = better)
        weight_array = np.array(list(weights.values()))
        concentration = np.sum(weight_array ** 2)  # Herfindahl index
        concentration_score = 1.0 - concentration  # Lower concentration = higher score
        
        # Correlation component (lower average correlation = better)
        correlation_score = 1.0
        if correlations:
            correlations_list = []
            symbols = list(weights.keys())
            for i, symbol1 in enumerate(symbols):
                for j, symbol2 in enumerate(symbols):
                    if i < j:
                        corr = correlations.get(symbol1, {}).get(symbol2, 0.0)
                        correlations_list.append(abs(corr))
            
            if correlations_list:
                avg_correlation = np.mean(correlations_list)
                correlation_score = 1.0 - avg_correlation
        
        # Combined score
        diversification_score = (
            symbol_score * 0.3 +
            concentration_score * 0.4 +
            correlation_score * 0.3
        )
        
        return float(diversification_score)

