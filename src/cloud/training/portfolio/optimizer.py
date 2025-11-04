"""
Portfolio-Level Optimization

Optimizes across multiple assets simultaneously, not just individual trades.

Traditional approach: Optimize each asset independently
Portfolio approach: Optimize entire portfolio considering correlations, diversification, risk budget

Key features:
1. Multi-asset optimization: Allocate capital across BTC, ETH, SOL, etc.
2. Correlation-aware: Reduce portfolio volatility through diversification
3. Risk budgeting: Allocate risk, not just capital
4. Portfolio constraints: Max positions, sector limits, concentration limits

Example:
- 3 assets all giving +10% expected return
- But BTC-ETH correlation = 0.9 (highly correlated)
- BTC-SOL correlation = 0.6 (moderately correlated)
- Optimal: Underweight ETH, overweight SOL (better diversification)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl
import structlog
from scipy.optimize import minimize

logger = structlog.get_logger(__name__)


@dataclass
class PortfolioConstraints:
    """Portfolio optimization constraints."""

    max_total_positions: int = 5  # Max simultaneous positions
    max_position_weight: float = 0.4  # Max 40% in any single asset
    min_position_weight: float = 0.05  # Min 5% per position (avoid dust)
    max_sector_weight: float = 0.6  # Max 60% in any sector
    max_correlation_exposure: float = 0.8  # Max correlation-weighted exposure
    target_volatility: float = 0.15  # Target portfolio volatility (15% annualized)
    min_diversification_ratio: float = 1.5  # Min diversification benefit


@dataclass
class AssetSignal:
    """Trading signal for a single asset."""

    symbol: str
    expected_return: float  # Expected return (bps or %)
    confidence: float  # Agent confidence (0-1)
    volatility: float  # Asset volatility
    beta: float  # Beta to market (BTC)
    correlation_to_btc: float  # Correlation with BTC
    current_regime: str  # Market regime
    sector: str  # Sector (L1, L2, DeFi, etc.)


@dataclass
class PortfolioAllocation:
    """Optimal portfolio allocation."""

    weights: Dict[str, float]  # Symbol → weight (sums to 1.0)
    expected_return: float  # Portfolio expected return
    expected_volatility: float  # Portfolio volatility
    sharpe_ratio: float  # Expected Sharpe ratio
    diversification_ratio: float  # Diversification benefit
    risk_contributions: Dict[str, float]  # Symbol → risk contribution
    metadata: Dict  # Additional info


class PortfolioOptimizer:
    """
    Multi-asset portfolio optimizer.

    Optimizes allocation across multiple trading opportunities considering:
    - Expected returns (from RL agents)
    - Volatilities and correlations
    - Portfolio constraints
    - Risk budgeting
    """

    def __init__(
        self,
        constraints: PortfolioConstraints,
        risk_free_rate: float = 0.0,
    ):
        """
        Initialize portfolio optimizer.

        Args:
            constraints: Portfolio constraints
            risk_free_rate: Risk-free rate for Sharpe calculation
        """
        self.constraints = constraints
        self.risk_free_rate = risk_free_rate

        # Correlation matrix cache
        self.correlation_matrix: Optional[np.ndarray] = None
        self.symbols_order: List[str] = []

        logger.info(
            "portfolio_optimizer_initialized",
            max_positions=constraints.max_total_positions,
            target_volatility=constraints.target_volatility,
        )

    def optimize(
        self,
        signals: List[AssetSignal],
        correlation_matrix: Optional[np.ndarray] = None,
        objective: str = "max_sharpe",
    ) -> PortfolioAllocation:
        """
        Optimize portfolio allocation.

        Args:
            signals: List of asset signals
            correlation_matrix: Asset correlation matrix (if None, estimate from data)
            objective: Optimization objective ("max_sharpe", "min_variance", "risk_parity")

        Returns:
            Optimal portfolio allocation
        """
        if len(signals) == 0:
            return self._empty_allocation()

        # Filter signals by confidence
        signals = [s for s in signals if s.confidence >= 0.5]

        if len(signals) == 0:
            return self._empty_allocation()

        # Limit to max positions
        signals = sorted(signals, key=lambda s: s.expected_return * s.confidence, reverse=True)
        signals = signals[: self.constraints.max_total_positions]

        # Update correlation matrix
        self.symbols_order = [s.symbol for s in signals]
        if correlation_matrix is not None:
            self.correlation_matrix = correlation_matrix
        else:
            self.correlation_matrix = self._estimate_correlation_matrix(signals)

        # Optimize based on objective
        if objective == "max_sharpe":
            weights = self._maximize_sharpe(signals)
        elif objective == "min_variance":
            weights = self._minimize_variance(signals)
        elif objective == "risk_parity":
            weights = self._risk_parity(signals)
        else:
            raise ValueError(f"Unknown objective: {objective}")

        # Create allocation
        allocation = self._create_allocation(signals, weights)

        logger.info(
            "portfolio_optimized",
            num_assets=len(signals),
            objective=objective,
            expected_return=allocation.expected_return,
            expected_volatility=allocation.expected_volatility,
            sharpe_ratio=allocation.sharpe_ratio,
        )

        return allocation

    def _maximize_sharpe(self, signals: List[AssetSignal]) -> np.ndarray:
        """Maximize portfolio Sharpe ratio."""
        n = len(signals)

        # Expected returns vector
        returns = np.array([s.expected_return * s.confidence for s in signals])

        # Volatilities
        vols = np.array([s.volatility for s in signals])

        # Covariance matrix
        cov_matrix = self._build_covariance_matrix(vols)

        # Objective: Minimize negative Sharpe ratio
        def negative_sharpe(weights):
            portfolio_return = np.dot(weights, returns)
            portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
            if portfolio_vol == 0:
                return 1e10
            sharpe = (portfolio_return - self.risk_free_rate) / portfolio_vol
            return -sharpe  # Minimize negative

        # Constraints
        constraints = self._build_constraints(signals, n)

        # Bounds
        bounds = [(self.constraints.min_position_weight, self.constraints.max_position_weight) for _ in range(n)]

        # Initial guess: equal weight
        x0 = np.ones(n) / n

        # Optimize
        result = minimize(
            negative_sharpe,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        if not result.success:
            logger.warning("sharpe_optimization_failed", message=result.message)
            return np.ones(n) / n  # Fallback to equal weight

        return result.x

    def _minimize_variance(self, signals: List[AssetSignal]) -> np.ndarray:
        """Minimize portfolio variance."""
        n = len(signals)

        # Volatilities
        vols = np.array([s.volatility for s in signals])

        # Covariance matrix
        cov_matrix = self._build_covariance_matrix(vols)

        # Objective: Minimize variance
        def portfolio_variance(weights):
            return np.dot(weights, np.dot(cov_matrix, weights))

        # Constraints
        constraints = self._build_constraints(signals, n)

        # Bounds
        bounds = [(self.constraints.min_position_weight, self.constraints.max_position_weight) for _ in range(n)]

        # Initial guess
        x0 = np.ones(n) / n

        # Optimize
        result = minimize(
            portfolio_variance,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        if not result.success:
            logger.warning("variance_optimization_failed", message=result.message)
            return np.ones(n) / n

        return result.x

    def _risk_parity(self, signals: List[AssetSignal]) -> np.ndarray:
        """Equal risk contribution from each asset."""
        n = len(signals)

        # Volatilities
        vols = np.array([s.volatility for s in signals])

        # Covariance matrix
        cov_matrix = self._build_covariance_matrix(vols)

        # Objective: Minimize sum of squared differences in risk contributions
        def risk_parity_objective(weights):
            portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
            if portfolio_vol == 0:
                return 1e10

            # Marginal risk contributions
            mrc = np.dot(cov_matrix, weights) / portfolio_vol

            # Risk contributions
            rc = weights * mrc

            # Target: equal risk contribution
            target_rc = portfolio_vol / n

            # Minimize squared deviations
            return np.sum((rc - target_rc) ** 2)

        # Constraints
        constraints = self._build_constraints(signals, n)

        # Bounds
        bounds = [(self.constraints.min_position_weight, self.constraints.max_position_weight) for _ in range(n)]

        # Initial guess
        x0 = np.ones(n) / n

        # Optimize
        result = minimize(
            risk_parity_objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        if not result.success:
            logger.warning("risk_parity_optimization_failed", message=result.message)
            # Fallback: inverse volatility weighting
            inv_vol = 1.0 / vols
            return inv_vol / inv_vol.sum()

        return result.x

    def _build_covariance_matrix(self, vols: np.ndarray) -> np.ndarray:
        """Build covariance matrix from volatilities and correlations."""
        # Cov(i,j) = vol(i) * vol(j) * corr(i,j)
        n = len(vols)
        cov_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i == j:
                    cov_matrix[i, j] = vols[i] ** 2
                else:
                    corr = self.correlation_matrix[i, j] if self.correlation_matrix is not None else 0.5
                    cov_matrix[i, j] = vols[i] * vols[j] * corr

        return cov_matrix

    def _build_constraints(self, signals: List[AssetSignal], n: int) -> List[Dict]:
        """Build optimization constraints."""
        constraints = []

        # Weights sum to 1
        constraints.append({"type": "eq", "fun": lambda w: np.sum(w) - 1.0})

        # Sector constraints
        sectors = {}
        for idx, signal in enumerate(signals):
            if signal.sector not in sectors:
                sectors[signal.sector] = []
            sectors[signal.sector].append(idx)

        for sector, indices in sectors.items():
            if len(indices) > 0:

                def sector_constraint(w, idxs=indices):
                    return self.constraints.max_sector_weight - np.sum([w[i] for i in idxs])

                constraints.append({"type": "ineq", "fun": sector_constraint})

        return constraints

    def _estimate_correlation_matrix(self, signals: List[AssetSignal]) -> np.ndarray:
        """Estimate correlation matrix from signals."""
        n = len(signals)
        corr_matrix = np.eye(n)  # Start with identity

        # Use correlation to BTC as proxy
        for i in range(n):
            for j in range(i + 1, n):
                # Estimate correlation from individual BTC correlations
                corr_i = signals[i].correlation_to_btc
                corr_j = signals[j].correlation_to_btc

                # Simple heuristic: corr(i,j) ≈ corr(i,btc) * corr(j,btc)
                estimated_corr = corr_i * corr_j

                corr_matrix[i, j] = estimated_corr
                corr_matrix[j, i] = estimated_corr

        return corr_matrix

    def _create_allocation(
        self,
        signals: List[AssetSignal],
        weights: np.ndarray,
    ) -> PortfolioAllocation:
        """Create portfolio allocation from optimized weights."""
        # Expected returns
        returns = np.array([s.expected_return * s.confidence for s in signals])
        portfolio_return = np.dot(weights, returns)

        # Volatilities
        vols = np.array([s.volatility for s in signals])
        cov_matrix = self._build_covariance_matrix(vols)
        portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))

        # Sharpe ratio
        sharpe = (portfolio_return - self.risk_free_rate) / portfolio_vol if portfolio_vol > 0 else 0.0

        # Diversification ratio
        weighted_vol = np.dot(weights, vols)  # Sum of weighted volatilities
        diversification_ratio = weighted_vol / portfolio_vol if portfolio_vol > 0 else 1.0

        # Risk contributions
        mrc = np.dot(cov_matrix, weights) / portfolio_vol if portfolio_vol > 0 else np.zeros(len(weights))
        rc = weights * mrc
        risk_contributions = {signals[i].symbol: rc[i] for i in range(len(signals))}

        # Weights dict
        weights_dict = {signals[i].symbol: weights[i] for i in range(len(signals))}

        return PortfolioAllocation(
            weights=weights_dict,
            expected_return=portfolio_return,
            expected_volatility=portfolio_vol,
            sharpe_ratio=sharpe,
            diversification_ratio=diversification_ratio,
            risk_contributions=risk_contributions,
            metadata={
                "num_assets": len(signals),
                "max_weight": max(weights),
                "min_weight": min(weights),
            },
        )

    def _empty_allocation(self) -> PortfolioAllocation:
        """Return empty allocation when no signals."""
        return PortfolioAllocation(
            weights={},
            expected_return=0.0,
            expected_volatility=0.0,
            sharpe_ratio=0.0,
            diversification_ratio=0.0,
            risk_contributions={},
            metadata={"num_assets": 0},
        )

    def rebalance_portfolio(
        self,
        current_allocation: PortfolioAllocation,
        new_allocation: PortfolioAllocation,
        rebalance_threshold: float = 0.05,
    ) -> Dict[str, float]:
        """
        Calculate rebalancing trades.

        Args:
            current_allocation: Current portfolio weights
            new_allocation: Target portfolio weights
            rebalance_threshold: Min weight change to trigger rebalance (5%)

        Returns:
            Dict of symbol → weight_change (positive = buy, negative = sell)
        """
        trades = {}

        # All symbols
        all_symbols = set(current_allocation.weights.keys()) | set(new_allocation.weights.keys())

        for symbol in all_symbols:
            current_weight = current_allocation.weights.get(symbol, 0.0)
            new_weight = new_allocation.weights.get(symbol, 0.0)
            change = new_weight - current_weight

            # Only rebalance if change exceeds threshold
            if abs(change) >= rebalance_threshold:
                trades[symbol] = change

        logger.info("rebalance_calculated", num_trades=len(trades), threshold=rebalance_threshold)

        return trades
