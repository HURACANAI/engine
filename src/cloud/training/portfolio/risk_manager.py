"""
Comprehensive Risk Management

Combines all Phase 3 components:
1. Portfolio optimization (correlation-aware)
2. Dynamic position sizing (Kelly + volatility)
3. Risk budgeting and diversification
4. Real-time risk monitoring

This is the brain that decides:
- Which assets to trade (portfolio optimizer)
- How much to trade (position sizer)
- When to reduce risk (risk monitor)
- How to diversify (correlation manager)
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import structlog

from .optimizer import (
    AssetSignal,
    PortfolioAllocation,
    PortfolioConstraints,
    PortfolioOptimizer,
)
from .position_sizer import (
    DynamicPositionSizer,
    PositionSizeRecommendation,
    PositionSizingConfig,
    calculate_optimal_leverage,
)

logger = structlog.get_logger(__name__)


@dataclass
class RiskLimits:
    """Global risk limits."""

    max_portfolio_volatility: float = 0.20  # Max 20% annualized volatility
    max_drawdown: float = 0.15  # Max 15% drawdown before shutdown
    max_var_95: float = 0.05  # Max 5% VaR at 95% confidence
    max_correlation_concentration: float = 0.70  # Max correlation-weighted exposure
    min_sharpe_ratio: float = 0.5  # Minimum acceptable Sharpe
    stress_test_scenarios: int = 100  # Number of stress scenarios


@dataclass
class PortfolioRisk:
    """Current portfolio risk metrics."""

    total_exposure_gbp: float
    portfolio_volatility: float
    estimated_var_95: float
    current_drawdown: float
    sharpe_ratio: float
    correlation_score: float  # How diversified (1.0 = perfect, 0.0 = all correlated)
    heat_utilization: float  # % of risk budget used
    num_positions: int
    largest_position_weight: float
    warnings: List[str]  # Risk warnings


class ComprehensiveRiskManager:
    """
    Complete risk management system.

    Orchestrates portfolio optimization, position sizing, and risk monitoring.
    """

    def __init__(
        self,
        portfolio_constraints: PortfolioConstraints,
        position_sizing_config: PositionSizingConfig,
        risk_limits: RiskLimits,
    ):
        """
        Initialize risk manager.

        Args:
            portfolio_constraints: Portfolio optimization constraints
            position_sizing_config: Position sizing configuration
            risk_limits: Global risk limits
        """
        self.constraints = portfolio_constraints
        self.limits = risk_limits

        # Initialize components
        self.portfolio_optimizer = PortfolioOptimizer(portfolio_constraints)
        self.position_sizer = DynamicPositionSizer(position_sizing_config)

        # Track current state
        self.current_allocation: Optional[PortfolioAllocation] = None
        self.peak_portfolio_value = 0.0
        self.portfolio_value_history: List[float] = []

        logger.info(
            "comprehensive_risk_manager_initialized",
            max_portfolio_vol=risk_limits.max_portfolio_volatility,
            max_drawdown=risk_limits.max_drawdown,
        )

    def get_trading_recommendations(
        self,
        signals: List[AssetSignal],
        current_portfolio_value: float,
        correlation_matrix: Optional[np.ndarray] = None,
    ) -> Tuple[PortfolioAllocation, Dict[str, PositionSizeRecommendation], PortfolioRisk]:
        """
        Get complete trading recommendations.

        Args:
            signals: Asset signals from RL agents
            current_portfolio_value: Current total portfolio value
            correlation_matrix: Asset correlation matrix

        Returns:
            (optimal_allocation, position_sizes, risk_metrics)
        """
        # Step 1: Optimize portfolio allocation
        optimal_allocation = self.portfolio_optimizer.optimize(
            signals=signals,
            correlation_matrix=correlation_matrix,
            objective="max_sharpe",
        )

        self.current_allocation = optimal_allocation

        # Step 2: Calculate position sizes
        position_sizes = {}
        for signal in signals:
            if signal.symbol in optimal_allocation.weights:
                # Calculate position size
                size_rec = self.position_sizer.calculate_position_size(
                    symbol=signal.symbol,
                    confidence=signal.confidence,
                    volatility=signal.volatility,
                    stop_loss_bps=100.0,  # Default stop loss
                    expected_return_bps=signal.expected_return,
                    current_price=100.0,  # Would get from market data
                )

                # Scale by portfolio weight
                allocation_weight = optimal_allocation.weights[signal.symbol]
                adjusted_size = size_rec.size_gbp * allocation_weight * 5  # Scale up from base

                size_rec.size_gbp = adjusted_size
                position_sizes[signal.symbol] = size_rec

        # Step 3: Calculate portfolio risk
        portfolio_risk = self._calculate_portfolio_risk(
            allocation=optimal_allocation,
            position_sizes=position_sizes,
            current_value=current_portfolio_value,
        )

        # Step 4: Check risk limits and adjust if needed
        if not self._check_risk_limits(portfolio_risk):
            # Reduce positions if limits exceeded
            optimal_allocation, position_sizes = self._reduce_risk(
                allocation=optimal_allocation,
                position_sizes=position_sizes,
                risk_metrics=portfolio_risk,
            )

            # Recalculate risk
            portfolio_risk = self._calculate_portfolio_risk(
                allocation=optimal_allocation,
                position_sizes=position_sizes,
                current_value=current_portfolio_value,
            )

        logger.info(
            "trading_recommendations_generated",
            num_assets=len(optimal_allocation.weights),
            portfolio_sharpe=optimal_allocation.sharpe_ratio,
            portfolio_vol=optimal_allocation.expected_volatility,
            heat_utilization=portfolio_risk.heat_utilization,
        )

        return optimal_allocation, position_sizes, portfolio_risk

    def _calculate_portfolio_risk(
        self,
        allocation: PortfolioAllocation,
        position_sizes: Dict[str, PositionSizeRecommendation],
        current_value: float,
    ) -> PortfolioRisk:
        """Calculate comprehensive portfolio risk metrics."""
        # Total exposure
        total_exposure = sum(size.size_gbp for size in position_sizes.values())

        # Portfolio volatility (from allocation)
        portfolio_vol = allocation.expected_volatility

        # VaR (95% confidence, normal distribution assumption)
        # VaR = μ - 1.65 * σ (one-tailed)
        var_95 = allocation.expected_return - 1.65 * portfolio_vol

        # Current drawdown
        self.portfolio_value_history.append(current_value)
        if current_value > self.peak_portfolio_value:
            self.peak_portfolio_value = current_value

        current_dd = (self.peak_portfolio_value - current_value) / self.peak_portfolio_value if self.peak_portfolio_value > 0 else 0.0

        # Correlation score (diversification benefit)
        correlation_score = allocation.diversification_ratio / 2.0  # Normalize to [0, 1]

        # Heat utilization
        heat_stats = self.position_sizer.get_portfolio_heat()
        heat_utilization = heat_stats["heat_utilization"]

        # Largest position
        largest_weight = max(allocation.weights.values()) if allocation.weights else 0.0

        # Risk warnings
        warnings = []
        if portfolio_vol > self.limits.max_portfolio_volatility:
            warnings.append(f"Portfolio volatility ({portfolio_vol:.2%}) exceeds limit ({self.limits.max_portfolio_volatility:.2%})")

        if current_dd > self.limits.max_drawdown:
            warnings.append(f"Drawdown ({current_dd:.2%}) exceeds limit ({self.limits.max_drawdown:.2%})")

        if var_95 < -self.limits.max_var_95:
            warnings.append(f"VaR ({var_95:.2%}) exceeds limit ({self.limits.max_var_95:.2%})")

        if correlation_score < 0.3:
            warnings.append("Low diversification - positions highly correlated")

        return PortfolioRisk(
            total_exposure_gbp=total_exposure,
            portfolio_volatility=portfolio_vol,
            estimated_var_95=var_95,
            current_drawdown=current_dd,
            sharpe_ratio=allocation.sharpe_ratio,
            correlation_score=correlation_score,
            heat_utilization=heat_utilization,
            num_positions=len(allocation.weights),
            largest_position_weight=largest_weight,
            warnings=warnings,
        )

    def _check_risk_limits(self, risk: PortfolioRisk) -> bool:
        """Check if portfolio is within risk limits."""
        if risk.portfolio_volatility > self.limits.max_portfolio_volatility:
            logger.warning("portfolio_volatility_exceeded", vol=risk.portfolio_volatility)
            return False

        if risk.current_drawdown > self.limits.max_drawdown:
            logger.warning("drawdown_limit_exceeded", dd=risk.current_drawdown)
            return False

        if risk.estimated_var_95 < -self.limits.max_var_95:
            logger.warning("var_limit_exceeded", var=risk.estimated_var_95)
            return False

        return True

    def _reduce_risk(
        self,
        allocation: PortfolioAllocation,
        position_sizes: Dict[str, PositionSizeRecommendation],
        risk_metrics: PortfolioRisk,
    ) -> Tuple[PortfolioAllocation, Dict[str, PositionSizeRecommendation]]:
        """Reduce risk by scaling down positions."""
        # Calculate scale factor
        vol_ratio = risk_metrics.portfolio_volatility / self.limits.max_portfolio_volatility
        dd_ratio = risk_metrics.current_drawdown / self.limits.max_drawdown if risk_metrics.current_drawdown > 0 else 1.0

        # Use most conservative scaling
        scale_factor = 1.0 / max(vol_ratio, dd_ratio, 1.0)
        scale_factor = max(scale_factor, 0.5)  # Minimum 50% reduction

        logger.warning(
            "reducing_portfolio_risk",
            scale_factor=scale_factor,
            vol_ratio=vol_ratio,
            dd_ratio=dd_ratio,
        )

        # Scale down position sizes
        for symbol, size_rec in position_sizes.items():
            size_rec.size_gbp *= scale_factor
            size_rec.risk_gbp *= scale_factor

        # Update allocation expected values
        allocation.expected_volatility *= scale_factor
        allocation.expected_return *= scale_factor

        return allocation, position_sizes

    def calculate_correlation_matrix(
        self,
        price_histories: Dict[str, List[float]],
    ) -> np.ndarray:
        """
        Calculate correlation matrix from price histories.

        Args:
            price_histories: Dict of symbol → price history

        Returns:
            Correlation matrix
        """
        symbols = list(price_histories.keys())
        n = len(symbols)

        if n == 0:
            return np.array([])

        # Calculate returns
        returns_matrix = []
        for symbol in symbols:
            prices = np.array(price_histories[symbol])
            if len(prices) < 2:
                returns_matrix.append(np.zeros(len(prices)))
            else:
                returns = np.diff(prices) / prices[:-1]
                returns_matrix.append(returns)

        # Ensure all return series have same length
        min_length = min(len(r) for r in returns_matrix)
        returns_matrix = [r[:min_length] for r in returns_matrix]

        # Calculate correlation matrix
        returns_array = np.array(returns_matrix)
        if returns_array.shape[1] < 2:
            return np.eye(n)  # Not enough data, return identity

        correlation_matrix = np.corrcoef(returns_array)

        return correlation_matrix

    def monte_carlo_stress_test(
        self,
        allocation: PortfolioAllocation,
        num_scenarios: int = 1000,
        shock_magnitude: float = 0.3,
    ) -> Dict[str, float]:
        """
        Run Monte Carlo stress test on portfolio.

        Args:
            allocation: Current portfolio allocation
            num_scenarios: Number of scenarios to simulate
            shock_magnitude: Size of market shock (30% default)

        Returns:
            Stress test results
        """
        if not allocation.weights:
            return {"worst_case_loss": 0.0, "var_99": 0.0, "expected_shortfall": 0.0}

        # Simulate price shocks
        portfolio_returns = []

        for _ in range(num_scenarios):
            # Random shock to each asset
            asset_shocks = np.random.normal(-shock_magnitude / 2, shock_magnitude, len(allocation.weights))

            # Portfolio return
            weights = np.array(list(allocation.weights.values()))
            portfolio_return = np.dot(weights, asset_shocks)
            portfolio_returns.append(portfolio_return)

        portfolio_returns = np.array(portfolio_returns)

        # Calculate risk metrics
        worst_case = np.min(portfolio_returns)
        var_99 = np.percentile(portfolio_returns, 1)  # 99% VaR
        expected_shortfall = np.mean(portfolio_returns[portfolio_returns < var_99])  # CVaR

        logger.info(
            "stress_test_complete",
            scenarios=num_scenarios,
            worst_case=worst_case,
            var_99=var_99,
            expected_shortfall=expected_shortfall,
        )

        return {
            "worst_case_loss": worst_case,
            "var_99": var_99,
            "expected_shortfall": expected_shortfall,
            "scenarios_tested": num_scenarios,
        }

    def get_risk_dashboard(self) -> Dict:
        """Get comprehensive risk dashboard."""
        heat_stats = self.position_sizer.get_portfolio_heat()

        dashboard = {
            "portfolio": {
                "current_allocation": self.current_allocation.weights if self.current_allocation else {},
                "expected_return": self.current_allocation.expected_return if self.current_allocation else 0.0,
                "expected_volatility": self.current_allocation.expected_volatility if self.current_allocation else 0.0,
                "sharpe_ratio": self.current_allocation.sharpe_ratio if self.current_allocation else 0.0,
            },
            "risk_budget": {
                "current_heat_gbp": heat_stats["current_heat_gbp"],
                "max_heat_gbp": heat_stats["max_heat_gbp"],
                "utilization": heat_stats["heat_utilization"],
                "available_gbp": heat_stats["available_heat_gbp"],
            },
            "performance": {
                "win_rate": heat_stats["win_rate"],
                "avg_win_bps": heat_stats["avg_win_bps"],
                "avg_loss_bps": heat_stats["avg_loss_bps"],
                "total_trades": heat_stats["total_trades"],
            },
            "limits": {
                "max_volatility": self.limits.max_portfolio_volatility,
                "max_drawdown": self.limits.max_drawdown,
                "max_var_95": self.limits.max_var_95,
            },
        }

        return dashboard
