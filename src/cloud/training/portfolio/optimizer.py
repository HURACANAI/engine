"""
Portfolio Optimization Layer.

Optimizes weights across active signals with risk budget.
Penalizes turnover and concentration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
from scipy.optimize import minimize
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class Signal:
    """Trading signal."""
    symbol: str
    direction: int  # -1, 0, +1
    confidence: float
    expected_edge_bps: float
    volatility: float


@dataclass
class PortfolioAllocation:
    """Portfolio allocation result."""
    allocations: Dict[str, float]  # symbol -> size_usd
    total_exposure: float
    hhi: float  # Herfindahl-Hirschman Index (concentration)
    turnover_penalty: float
    risk_budget_used: float


class PortfolioOptimizer:
    """
    Portfolio optimizer with risk budget.
    
    Features:
    - Optimize weights across active signals
    - Risk budget constraint
    - Turnover penalty
    - Concentration penalty (HHI-based cap)
    - Cash buffer preservation
    """
    
    def __init__(
        self,
        equity_usd: float = 100000.0,
        risk_budget_pct: float = 70.0,  # 70% of equity at risk
        max_hhi: float = 0.25,  # Max concentration (HHI)
        turnover_penalty: float = 0.1,  # Penalty for high turnover
        cash_buffer_pct: float = 10.0,  # 10% cash buffer
    ) -> None:
        """
        Initialize portfolio optimizer.
        
        Args:
            equity_usd: Total equity
            risk_budget_pct: Risk budget as % of equity
            max_hhi: Maximum HHI (concentration)
            turnover_penalty: Turnover penalty factor
            cash_buffer_pct: Cash buffer as % of equity
        """
        self.equity_usd = equity_usd
        self.risk_budget_pct = risk_budget_pct
        self.max_hhi = max_hhi
        self.turnover_penalty = turnover_penalty
        self.cash_buffer_pct = cash_buffer_pct
        
        # Track previous allocations for turnover calculation
        self.previous_allocations: Dict[str, float] = {}
        
        logger.info(
            "portfolio_optimizer_initialized",
            equity_usd=equity_usd,
            risk_budget_pct=risk_budget_pct,
            max_hhi=max_hhi
        )
    
    def optimize(
        self,
        signals: List[Signal],
        current_positions: Optional[Dict[str, float]] = None
    ) -> PortfolioAllocation:
        """
        Optimize portfolio allocation across signals.
        
        Args:
            signals: List of active signals
            current_positions: Current positions (for turnover calculation)
        
        Returns:
            PortfolioAllocation with optimized weights
        """
        if not signals:
            return PortfolioAllocation(
                allocations={},
                total_exposure=0.0,
                hhi=0.0,
                turnover_penalty=0.0,
                risk_budget_used=0.0
            )
        
        if current_positions is None:
            current_positions = {}
        
        # Calculate risk budget
        risk_budget_usd = self.equity_usd * (self.risk_budget_pct / 100.0)
        cash_buffer_usd = self.equity_usd * (self.cash_buffer_pct / 100.0)
        max_exposure = self.equity_usd - cash_buffer_usd
        
        # Optimize weights
        symbols = [s.symbol for s in signals]
        n = len(signals)
        
        # Objective: maximize expected return - penalties
        def objective(weights):
            # Expected return
            expected_return = sum(
                w * s.expected_edge_bps * s.confidence
                for w, s in zip(weights, signals)
            )
            
            # Turnover penalty
            turnover = self._calculate_turnover(weights, symbols, current_positions)
            turnover_penalty = self.turnover_penalty * turnover
            
            # Concentration penalty (HHI)
            hhi = self._calculate_hhi(weights)
            concentration_penalty = max(0, hhi - self.max_hhi) * 100.0
            
            # Minimize negative return + penalties
            return -(expected_return - turnover_penalty - concentration_penalty)
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},  # Sum to 1
        ]
        
        # Bounds: 0 <= w <= 1
        bounds = [(0.0, 1.0) for _ in range(n)]
        
        # Initial guess: equal weights
        initial_weights = np.ones(n) / n
        
        # Optimize
        try:
            result = minimize(
                objective,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            if result.success:
                optimal_weights = result.x
            else:
                logger.warning("portfolio_optimization_failed", message=result.message)
                optimal_weights = initial_weights
        except Exception as e:
            logger.error("portfolio_optimization_error", error=str(e))
            optimal_weights = initial_weights
        
        # Convert weights to USD allocations
        allocations = {}
        total_exposure = 0.0
        
        for weight, signal in zip(optimal_weights, signals):
            if weight > 0.01:  # Only allocate if weight > 1%
                size_usd = weight * max_exposure
                allocations[signal.symbol] = size_usd
                total_exposure += size_usd
        
        # Calculate metrics
        hhi = self._calculate_hhi(optimal_weights)
        turnover = self._calculate_turnover(optimal_weights, symbols, current_positions)
        risk_budget_used = (total_exposure / self.equity_usd) * 100.0
        
        # Update previous allocations
        self.previous_allocations = allocations.copy()
        
        return PortfolioAllocation(
            allocations=allocations,
            total_exposure=total_exposure,
            hhi=hhi,
            turnover_penalty=turnover * self.turnover_penalty,
            risk_budget_used=risk_budget_used
        )
    
    def _calculate_hhi(self, weights: np.ndarray) -> float:
        """
        Calculate Herfindahl-Hirschman Index (concentration measure).
        
        HHI = sum(w_i^2)
        Range: [0, 1] where 1 = maximum concentration
        
        Args:
            weights: Portfolio weights
        
        Returns:
            HHI value
        """
        return float(np.sum(weights ** 2))
    
    def _calculate_turnover(
        self,
        new_weights: np.ndarray,
        symbols: List[str],
        current_positions: Dict[str, float]
    ) -> float:
        """
        Calculate portfolio turnover.
        
        Turnover = sum(|w_new - w_old|) / 2
        
        Args:
            new_weights: New portfolio weights
            symbols: List of symbols
            current_positions: Current positions in USD
        
        Returns:
            Turnover value (0 to 1)
        """
        if not current_positions:
            return 0.0
        
        total_current = sum(abs(v) for v in current_positions.values())
        if total_current == 0:
            return 0.0
        
        # Calculate old weights
        old_weights = np.zeros(len(symbols))
        for i, symbol in enumerate(symbols):
            if symbol in current_positions:
                old_weights[i] = abs(current_positions[symbol]) / total_current
        
        # Calculate turnover
        turnover = np.sum(np.abs(new_weights - old_weights)) / 2.0
        
        return float(turnover)
