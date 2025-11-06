"""
Value at Risk (VaR) and Expected Shortfall (ES) Risk Measurement

Advanced risk measurement techniques used by hedge funds and investment banks:
- Value at Risk (VaR) - Maximum expected loss at confidence level
- Expected Shortfall (ES) - Average loss beyond VaR threshold
- Conditional VaR (CVaR) - Expected loss in worst-case scenarios

Source: Verified academic research on risk measurement
Expected Impact: Better risk quantification, improved risk management
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import structlog  # type: ignore
from scipy import stats  # type: ignore

logger = structlog.get_logger(__name__)


@dataclass
class VaRResult:
    """Value at Risk calculation result."""
    var_95: float  # VaR at 95% confidence
    var_99: float  # VaR at 99% confidence
    expected_shortfall_95: float  # ES at 95% confidence
    expected_shortfall_99: float  # ES at 99% confidence
    method: str  # Calculation method
    confidence_levels: Dict[float, float]  # confidence -> VaR


@dataclass
class PortfolioVaR:
    """Portfolio-level VaR metrics."""
    portfolio_var_95: float
    portfolio_var_99: float
    portfolio_es_95: float
    portfolio_es_99: float
    component_vars: Dict[str, float]  # Asset -> VaR contribution
    diversification_benefit: float  # Reduction in VaR from diversification


class VaRCalculator:
    """
    Calculates Value at Risk and Expected Shortfall.
    
    Methods:
    1. Historical Simulation - Uses historical returns
    2. Parametric (Variance-Covariance) - Assumes normal distribution
    3. Monte Carlo Simulation - Simulates future scenarios
    """

    def __init__(
        self,
        method: str = 'historical',  # 'historical', 'parametric', 'monte_carlo'
        confidence_levels: List[float] = None,
    ):
        """
        Initialize VaR calculator.
        
        Args:
            method: Calculation method
            confidence_levels: List of confidence levels (default: [0.95, 0.99])
        """
        self.method = method
        self.confidence_levels = confidence_levels or [0.95, 0.99]
        
        logger.info("var_calculator_initialized", method=method)

    def calculate_var(
        self,
        returns: np.ndarray,  # Historical returns
        portfolio_value: float = 1.0,
        position_weights: Optional[np.ndarray] = None,
    ) -> VaRResult:
        """
        Calculate Value at Risk.
        
        Args:
            returns: Historical returns (can be single asset or portfolio)
            portfolio_value: Portfolio value in GBP
            position_weights: Optional weights for portfolio (if None, assumes single asset)
            
        Returns:
            VaRResult with VaR and ES at different confidence levels
        """
        if self.method == 'historical':
            return self._calculate_historical_var(returns, portfolio_value, position_weights)
        elif self.method == 'parametric':
            return self._calculate_parametric_var(returns, portfolio_value, position_weights)
        elif self.method == 'monte_carlo':
            return self._calculate_monte_carlo_var(returns, portfolio_value, position_weights)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _calculate_historical_var(
        self,
        returns: np.ndarray,
        portfolio_value: float,
        position_weights: Optional[np.ndarray],
    ) -> VaRResult:
        """Calculate VaR using historical simulation."""
        # Calculate portfolio returns if weights provided
        if position_weights is not None and returns.ndim > 1:
            portfolio_returns = np.dot(returns, position_weights)
        else:
            portfolio_returns = returns.flatten() if returns.ndim > 1 else returns
        
        # Sort returns
        sorted_returns = np.sort(portfolio_returns)
        
        # Calculate VaR at each confidence level
        var_dict = {}
        es_dict = {}
        
        for confidence in self.confidence_levels:
            # VaR is the (1-confidence) percentile
            percentile = (1 - confidence) * 100
            var = np.percentile(sorted_returns, percentile)
            var_dict[confidence] = float(var * portfolio_value)
            
            # Expected Shortfall is average of losses beyond VaR
            tail_returns = sorted_returns[sorted_returns <= var]
            if len(tail_returns) > 0:
                es = np.mean(tail_returns)
            else:
                es = var
            es_dict[confidence] = float(es * portfolio_value)
        
        return VaRResult(
            var_95=var_dict.get(0.95, 0.0),
            var_99=var_dict.get(0.99, 0.0),
            expected_shortfall_95=es_dict.get(0.95, 0.0),
            expected_shortfall_99=es_dict.get(0.99, 0.0),
            method='historical',
            confidence_levels=var_dict,
        )

    def _calculate_parametric_var(
        self,
        returns: np.ndarray,
        portfolio_value: float,
        position_weights: Optional[np.ndarray],
    ) -> VaRResult:
        """Calculate VaR using parametric method (assumes normal distribution)."""
        # Calculate portfolio returns
        if position_weights is not None and returns.ndim > 1:
            portfolio_returns = np.dot(returns, position_weights)
        else:
            portfolio_returns = returns.flatten() if returns.ndim > 1 else returns
        
        # Calculate mean and std
        mean_return = np.mean(portfolio_returns)
        std_return = np.std(portfolio_returns)
        
        # Calculate VaR at each confidence level
        var_dict = {}
        es_dict = {}
        
        for confidence in self.confidence_levels:
            # Z-score for confidence level
            z_score = stats.norm.ppf(1 - confidence)
            
            # VaR = mean - z_score * std
            var = mean_return - z_score * std_return
            var_dict[confidence] = float(var * portfolio_value)
            
            # Expected Shortfall for normal distribution
            # ES = mean - std * (pdf(z) / (1 - confidence))
            z_pdf = stats.norm.pdf(z_score)
            es = mean_return - std_return * (z_pdf / (1 - confidence))
            es_dict[confidence] = float(es * portfolio_value)
        
        return VaRResult(
            var_95=var_dict.get(0.95, 0.0),
            var_99=var_dict.get(0.99, 0.0),
            expected_shortfall_95=es_dict.get(0.95, 0.0),
            expected_shortfall_99=es_dict.get(0.99, 0.0),
            method='parametric',
            confidence_levels=var_dict,
        )

    def _calculate_monte_carlo_var(
        self,
        returns: np.ndarray,
        portfolio_value: float,
        position_weights: Optional[np.ndarray],
        n_simulations: int = 10000,
    ) -> VaRResult:
        """Calculate VaR using Monte Carlo simulation."""
        # Calculate portfolio returns
        if position_weights is not None and returns.ndim > 1:
            portfolio_returns = np.dot(returns, position_weights)
        else:
            portfolio_returns = returns.flatten() if returns.ndim > 1 else returns
        
        # Fit distribution to returns
        mean_return = np.mean(portfolio_returns)
        std_return = np.std(portfolio_returns)
        
        # Simulate future returns
        simulated_returns = np.random.normal(mean_return, std_return, n_simulations)
        
        # Calculate VaR from simulated returns
        sorted_simulated = np.sort(simulated_returns)
        
        var_dict = {}
        es_dict = {}
        
        for confidence in self.confidence_levels:
            percentile = (1 - confidence) * 100
            var = np.percentile(sorted_simulated, percentile)
            var_dict[confidence] = float(var * portfolio_value)
            
            # Expected Shortfall
            tail_returns = sorted_simulated[sorted_simulated <= var]
            if len(tail_returns) > 0:
                es = np.mean(tail_returns)
            else:
                es = var
            es_dict[confidence] = float(es * portfolio_value)
        
        return VaRResult(
            var_95=var_dict.get(0.95, 0.0),
            var_99=var_dict.get(0.99, 0.0),
            expected_shortfall_95=es_dict.get(0.95, 0.0),
            expected_shortfall_99=es_dict.get(0.99, 0.0),
            method='monte_carlo',
            confidence_levels=var_dict,
        )

    def calculate_portfolio_var(
        self,
        returns: np.ndarray,  # Shape: (n_samples, n_assets)
        weights: np.ndarray,  # Portfolio weights
        portfolio_value: float,
        correlation_matrix: Optional[np.ndarray] = None,
    ) -> PortfolioVaR:
        """
        Calculate portfolio-level VaR considering correlations.
        
        Args:
            returns: Historical returns for each asset
            weights: Portfolio weights
            portfolio_value: Total portfolio value
            correlation_matrix: Optional correlation matrix
            
        Returns:
            PortfolioVaR with portfolio and component VaRs
        """
        # Calculate individual asset VaRs
        component_vars = {}
        individual_vars = []
        
        for i in range(returns.shape[1]):
            asset_returns = returns[:, i]
            asset_var = self.calculate_var(asset_returns, portfolio_value * weights[i])
            component_vars[f"asset_{i}"] = asset_var.var_95
            individual_vars.append(asset_var.var_95)
        
        # Calculate portfolio VaR
        portfolio_returns = np.dot(returns, weights)
        portfolio_var_result = self.calculate_var(portfolio_returns, portfolio_value)
        
        # Calculate diversification benefit
        # Sum of individual VaRs vs portfolio VaR
        sum_individual_var = sum(individual_vars)
        diversification_benefit = (sum_individual_var - portfolio_var_result.var_95) / sum_individual_var if sum_individual_var > 0 else 0.0
        
        return PortfolioVaR(
            portfolio_var_95=portfolio_var_result.var_95,
            portfolio_var_99=portfolio_var_result.var_99,
            portfolio_es_95=portfolio_var_result.expected_shortfall_95,
            portfolio_es_99=portfolio_var_result.expected_shortfall_99,
            component_vars=component_vars,
            diversification_benefit=max(0.0, diversification_benefit),
        )

