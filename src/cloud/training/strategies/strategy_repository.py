"""
Strategy Repository

Library of quantitative strategies including momentum, value, and equal-weight.
Forms the Engine's Strategy Repository that the Mechanic can tune and test.

Key Strategies:
- Equal-Weight Index (systematic allocation)
- Momentum Strategy (rank and select top performers)
- Value Strategy (compare valuation metrics)
- Factor-based strategies

Author: Huracan Engine Team
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


class StrategyType(Enum):
    """Strategy type"""
    EQUAL_WEIGHT = "equal_weight"
    MOMENTUM = "momentum"
    VALUE = "value"
    MOMENTUM_VALUE_COMBO = "momentum_value_combo"
    CROSS_SECTIONAL_MOMENTUM = "cross_sectional_momentum"


@dataclass
class StrategyResult:
    """Strategy result"""
    strategy_type: StrategyType
    selected_symbols: List[str]
    weights: Dict[str, float]
    scores: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, any] = field(default_factory=dict)


class StrategyRepository:
    """
    Strategy Repository - Library of quantitative strategies.
    
    Implements factor-based strategies:
    - Equal-Weight Index
    - Momentum Strategy
    - Value Strategy
    - Combined strategies
    
    Usage:
        repo = StrategyRepository()
        
        # Momentum strategy
        result = repo.momentum_strategy(
            symbols_data={
                "BTCUSDT": {"1yr_return": 0.5, "3m_return": 0.2, ...},
                "ETHUSDT": {"1yr_return": 0.3, "3m_return": 0.15, ...},
                ...
            },
            top_n=50
        )
        
        # Portfolio weights
        weights = result.weights
    """
    
    def __init__(
        self,
        min_symbols: int = 10,  # Minimum symbols for strategy
        max_symbols: int = 100  # Maximum symbols for strategy
    ):
        """
        Initialize strategy repository.
        
        Args:
            min_symbols: Minimum number of symbols required
            max_symbols: Maximum number of symbols to select
        """
        self.min_symbols = min_symbols
        self.max_symbols = max_symbols
        
        logger.info(
            "strategy_repository_initialized",
            min_symbols=min_symbols,
            max_symbols=max_symbols
        )
    
    def equal_weight_strategy(
        self,
        symbols_data: Dict[str, Dict[str, float]],
        filter_criteria: Optional[Dict[str, any]] = None
    ) -> StrategyResult:
        """
        Equal-weight index strategy.
        
        Args:
            symbols_data: Dictionary of symbol -> data
            filter_criteria: Optional filter criteria (e.g., min_volume, min_market_cap)
        
        Returns:
            StrategyResult with equal weights
        """
        # Filter symbols if criteria provided
        filtered_symbols = self._filter_symbols(symbols_data, filter_criteria)
        
        if len(filtered_symbols) < self.min_symbols:
            logger.warning(
                "insufficient_symbols_for_equal_weight",
                num_symbols=len(filtered_symbols),
                min_required=self.min_symbols
            )
            return StrategyResult(
                strategy_type=StrategyType.EQUAL_WEIGHT,
                selected_symbols=[],
                weights={}
            )
        
        # Equal weight allocation
        num_symbols = len(filtered_symbols)
        weight = 1.0 / num_symbols
        
        weights = {symbol: weight for symbol in filtered_symbols}
        
        logger.info(
            "equal_weight_strategy_computed",
            num_symbols=num_symbols,
            weight_per_symbol=weight
        )
        
        return StrategyResult(
            strategy_type=StrategyType.EQUAL_WEIGHT,
            selected_symbols=filtered_symbols,
            weights=weights,
            metadata={"num_symbols": num_symbols}
        )
    
    def momentum_strategy(
        self,
        symbols_data: Dict[str, Dict[str, float]],
        return_column: str = "1yr_return",
        top_n: Optional[int] = None,
        min_return: Optional[float] = None,
        filter_criteria: Optional[Dict[str, any]] = None
    ) -> StrategyResult:
        """
        Momentum strategy - rank and select top performers.
        
        Args:
            symbols_data: Dictionary of symbol -> data
            return_column: Column name for returns (e.g., "1yr_return", "3m_return")
            top_n: Number of top symbols to select (default: max_symbols)
            min_return: Minimum return threshold
            filter_criteria: Optional filter criteria
        
        Returns:
            StrategyResult with momentum-based weights
        """
        # Filter symbols
        filtered_data = self._filter_symbols_data(symbols_data, filter_criteria)
        
        if len(filtered_data) < self.min_symbols:
            return StrategyResult(
                strategy_type=StrategyType.MOMENTUM,
                selected_symbols=[],
                weights={}
            )
        
        # Extract returns
        returns = {}
        for symbol, data in filtered_data.items():
            return_value = data.get(return_column, 0.0)
            if min_return is None or return_value >= min_return:
                returns[symbol] = return_value
        
        if not returns:
            return StrategyResult(
                strategy_type=StrategyType.MOMENTUM,
                selected_symbols=[],
                weights={}
            )
        
        # Rank by returns (descending)
        sorted_symbols = sorted(returns.items(), key=lambda x: x[1], reverse=True)
        
        # Select top N
        top_n = top_n or self.max_symbols
        top_symbols = [symbol for symbol, _ in sorted_symbols[:top_n]]
        
        # Calculate momentum scores (rank-based)
        momentum_scores = {}
        for rank, (symbol, return_value) in enumerate(sorted_symbols, 1):
            momentum_scores[symbol] = len(sorted_symbols) - rank + 1  # Higher rank = higher score
        
        # Weight by momentum score (normalized)
        total_score = sum(momentum_scores.get(s, 0) for s in top_symbols)
        weights = {}
        for symbol in top_symbols:
            if total_score > 0:
                weights[symbol] = momentum_scores.get(symbol, 0) / total_score
            else:
                weights[symbol] = 1.0 / len(top_symbols)  # Equal weight fallback
        
        logger.info(
            "momentum_strategy_computed",
            num_selected=len(top_symbols),
            top_symbol=top_symbols[0] if top_symbols else None,
            top_return=returns.get(top_symbols[0], 0.0) if top_symbols else 0.0
        )
        
        return StrategyResult(
            strategy_type=StrategyType.MOMENTUM,
            selected_symbols=top_symbols,
            weights=weights,
            scores=momentum_scores,
            metadata={
                "return_column": return_column,
                "top_n": len(top_symbols),
                "min_return": min_return
            }
        )
    
    def value_strategy(
        self,
        symbols_data: Dict[str, Dict[str, float]],
        value_metric: str = "pe_ratio",  # Lower is better for value
        top_n: Optional[int] = None,
        max_value: Optional[float] = None,
        filter_criteria: Optional[Dict[str, any]] = None
    ) -> StrategyResult:
        """
        Value strategy - select undervalued assets.
        
        Args:
            symbols_data: Dictionary of symbol -> data
            value_metric: Value metric (e.g., "pe_ratio", "pb_ratio")
            top_n: Number of top symbols to select
            max_value: Maximum value metric threshold
            filter_criteria: Optional filter criteria
        
        Returns:
            StrategyResult with value-based weights
        """
        # Filter symbols
        filtered_data = self._filter_symbols_data(symbols_data, filter_criteria)
        
        if len(filtered_data) < self.min_symbols:
            return StrategyResult(
                strategy_type=StrategyType.VALUE,
                selected_symbols=[],
                weights={}
            )
        
        # Extract value metrics
        value_metrics = {}
        for symbol, data in filtered_data.items():
            value = data.get(value_metric, float('inf'))
            if max_value is None or value <= max_value:
                value_metrics[symbol] = value
        
        if not value_metrics:
            return StrategyResult(
                strategy_type=StrategyType.VALUE,
                selected_symbols=[],
                weights={}
            )
        
        # Rank by value metric (ascending - lower is better)
        sorted_symbols = sorted(value_metrics.items(), key=lambda x: x[1])
        
        # Select top N (lowest value metrics)
        top_n = top_n or self.max_symbols
        top_symbols = [symbol for symbol, _ in sorted_symbols[:top_n]]
        
        # Calculate value scores (inverse of rank - lower value metric = higher score)
        value_scores = {}
        for rank, (symbol, value) in enumerate(sorted_symbols, 1):
            value_scores[symbol] = len(sorted_symbols) - rank + 1
        
        # Weight by value score (normalized)
        total_score = sum(value_scores.get(s, 0) for s in top_symbols)
        weights = {}
        for symbol in top_symbols:
            if total_score > 0:
                weights[symbol] = value_scores.get(symbol, 0) / total_score
            else:
                weights[symbol] = 1.0 / len(top_symbols)
        
        logger.info(
            "value_strategy_computed",
            num_selected=len(top_symbols),
            top_symbol=top_symbols[0] if top_symbols else None,
            top_value=value_metrics.get(top_symbols[0], 0.0) if top_symbols else 0.0
        )
        
        return StrategyResult(
            strategy_type=StrategyType.VALUE,
            selected_symbols=top_symbols,
            weights=weights,
            scores=value_scores,
            metadata={
                "value_metric": value_metric,
                "top_n": len(top_symbols),
                "max_value": max_value
            }
        )
    
    def momentum_value_combo_strategy(
        self,
        symbols_data: Dict[str, Dict[str, float]],
        momentum_weight: float = 0.6,
        value_weight: float = 0.4,
        top_n: Optional[int] = None,
        filter_criteria: Optional[Dict[str, any]] = None
    ) -> StrategyResult:
        """
        Combined momentum and value strategy.
        
        Args:
            symbols_data: Dictionary of symbol -> data
            momentum_weight: Weight for momentum component
            value_weight: Weight for value component
            top_n: Number of top symbols to select
            filter_criteria: Optional filter criteria
        
        Returns:
            StrategyResult with combined weights
        """
        # Get momentum and value strategies
        momentum_result = self.momentum_strategy(
            symbols_data,
            top_n=top_n,
            filter_criteria=filter_criteria
        )
        
        value_result = self.value_strategy(
            symbols_data,
            top_n=top_n,
            filter_criteria=filter_criteria
        )
        
        # Combine scores
        combined_scores = {}
        all_symbols = set(momentum_result.selected_symbols + value_result.selected_symbols)
        
        for symbol in all_symbols:
            momentum_score = momentum_result.scores.get(symbol, 0.0)
            value_score = value_result.scores.get(symbol, 0.0)
            
            # Normalize scores to 0-1
            max_momentum = max(momentum_result.scores.values()) if momentum_result.scores else 1.0
            max_value = max(value_result.scores.values()) if value_result.scores else 1.0
            
            normalized_momentum = momentum_score / max_momentum if max_momentum > 0 else 0.0
            normalized_value = value_score / max_value if max_value > 0 else 0.0
            
            # Combined score
            combined_scores[symbol] = (
                normalized_momentum * momentum_weight +
                normalized_value * value_weight
            )
        
        # Select top N by combined score
        sorted_symbols = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        top_n = top_n or self.max_symbols
        top_symbols = [symbol for symbol, _ in sorted_symbols[:top_n]]
        
        # Calculate weights
        total_score = sum(combined_scores.get(s, 0) for s in top_symbols)
        weights = {}
        for symbol in top_symbols:
            if total_score > 0:
                weights[symbol] = combined_scores.get(symbol, 0) / total_score
            else:
                weights[symbol] = 1.0 / len(top_symbols)
        
        logger.info(
            "momentum_value_combo_strategy_computed",
            num_selected=len(top_symbols),
            momentum_weight=momentum_weight,
            value_weight=value_weight
        )
        
        return StrategyResult(
            strategy_type=StrategyType.MOMENTUM_VALUE_COMBO,
            selected_symbols=top_symbols,
            weights=weights,
            scores=combined_scores,
            metadata={
                "momentum_weight": momentum_weight,
                "value_weight": value_weight,
                "top_n": len(top_symbols)
            }
        )
    
    def _filter_symbols(
        self,
        symbols_data: Dict[str, Dict[str, float]],
        filter_criteria: Optional[Dict[str, any]]
    ) -> List[str]:
        """Filter symbols based on criteria"""
        if not filter_criteria:
            return list(symbols_data.keys())
        
        filtered = []
        for symbol, data in symbols_data.items():
            include = True
            
            # Check minimum volume
            if "min_volume" in filter_criteria:
                if data.get("volume", 0) < filter_criteria["min_volume"]:
                    include = False
            
            # Check minimum market cap
            if "min_market_cap" in filter_criteria:
                if data.get("market_cap", 0) < filter_criteria["min_market_cap"]:
                    include = False
            
            # Check minimum price
            if "min_price" in filter_criteria:
                if data.get("price", 0) < filter_criteria["min_price"]:
                    include = False
            
            if include:
                filtered.append(symbol)
        
        return filtered
    
    def _filter_symbols_data(
        self,
        symbols_data: Dict[str, Dict[str, float]],
        filter_criteria: Optional[Dict[str, any]]
    ) -> Dict[str, Dict[str, float]]:
        """Filter symbols data based on criteria"""
        filtered_symbols = self._filter_symbols(symbols_data, filter_criteria)
        return {symbol: symbols_data[symbol] for symbol in filtered_symbols if symbol in symbols_data}

