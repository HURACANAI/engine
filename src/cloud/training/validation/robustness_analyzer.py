"""
Robustness Analyzer

Monte Carlo and permutation visualization for strategy robustness testing.
Analyzes strategy performance under various market conditions and randomizations.

Key Features:
- Monte Carlo simulation
- Permutation testing visualization
- Sensitivity analysis
- Performance distribution analysis
- Robustness metrics
- Visualization support

Author: Huracan Engine Team
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable, Any
from enum import Enum
import numpy as np
import structlog

logger = structlog.get_logger(__name__)


class RandomizationType(Enum):
    """Randomization type for robustness testing"""
    PERMUTATION = "permutation"  # Shuffle trade sequence
    BOOTSTRAP = "bootstrap"  # Bootstrap resampling
    NOISE = "noise"  # Add noise to returns
    REGIME_SHUFFLE = "regime_shuffle"  # Shuffle regime labels


@dataclass
class RobustnessMetrics:
    """Robustness metrics"""
    original_sharpe: float
    mean_random_sharpe: float
    std_random_sharpe: float
    p_value: float
    z_score: float
    percentile_rank: float
    is_robust: bool
    robustness_score: float  # 0-1, higher is more robust


@dataclass
class MonteCarloResult:
    """Monte Carlo simulation result"""
    iteration: int
    sharpe_ratio: float
    total_return: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    metrics: Dict[str, float] = field(default_factory=dict)


class RobustnessAnalyzer:
    """
    Robustness Analyzer.
    
    Analyzes strategy robustness using Monte Carlo simulation and permutation testing.
    
    Usage:
        analyzer = RobustnessAnalyzer(num_simulations=1000)
        
        # Run Monte Carlo simulation
        results = analyzer.run_monte_carlo(
            returns=returns,
            strategy_func=strategy_function
        )
        
        # Analyze robustness
        metrics = analyzer.analyze_robustness(
            original_sharpe=2.5,
            random_sharpes=[r.sharpe_ratio for r in results]
        )
    """
    
    def __init__(
        self,
        num_simulations: int = 1000,
        random_seed: Optional[int] = None
    ):
        """
        Initialize robustness analyzer.
        
        Args:
            num_simulations: Number of Monte Carlo simulations
            random_seed: Random seed for reproducibility
        """
        self.num_simulations = num_simulations
        self.random_seed = random_seed
        
        if random_seed is not None:
            np.random.seed(random_seed)
        
        logger.info(
            "robustness_analyzer_initialized",
            num_simulations=num_simulations,
            random_seed=random_seed
        )
    
    def run_monte_carlo(
        self,
        returns: np.ndarray,
        strategy_func: Optional[Callable[[np.ndarray], Dict[str, float]]] = None,
        randomization_type: RandomizationType = RandomizationType.PERMUTATION
    ) -> List[MonteCarloResult]:
        """
        Run Monte Carlo simulation.
        
        Args:
            returns: Original returns array
            strategy_func: Strategy function (optional, uses default if not provided)
            randomization_type: Type of randomization
        
        Returns:
            List of MonteCarloResult
        """
        results = []
        
        for i in range(self.num_simulations):
            # Generate randomized returns
            randomized_returns = self._randomize_returns(returns, randomization_type)
            
            # Calculate metrics
            if strategy_func:
                metrics = strategy_func(randomized_returns)
            else:
                metrics = self._calculate_default_metrics(randomized_returns)
            
            # Create result
            result = MonteCarloResult(
                iteration=i,
                sharpe_ratio=metrics.get("sharpe_ratio", 0.0),
                total_return=metrics.get("total_return", 0.0),
                max_drawdown=metrics.get("max_drawdown", 0.0),
                win_rate=metrics.get("win_rate", 0.0),
                profit_factor=metrics.get("profit_factor", 0.0),
                metrics=metrics
            )
            
            results.append(result)
            
            if (i + 1) % 100 == 0:
                logger.info("monte_carlo_progress", iteration=i + 1, total=self.num_simulations)
        
        logger.info("monte_carlo_complete", total_simulations=len(results))
        
        return results
    
    def _randomize_returns(
        self,
        returns: np.ndarray,
        randomization_type: RandomizationType
    ) -> np.ndarray:
        """Randomize returns based on type"""
        if randomization_type == RandomizationType.PERMUTATION:
            # Shuffle returns
            return np.random.permutation(returns)
        
        elif randomization_type == RandomizationType.BOOTSTRAP:
            # Bootstrap resampling
            n = len(returns)
            indices = np.random.choice(n, size=n, replace=True)
            return returns[indices]
        
        elif randomization_type == RandomizationType.NOISE:
            # Add noise to returns
            noise = np.random.normal(0, returns.std() * 0.1, size=len(returns))
            return returns + noise
        
        elif randomization_type == RandomizationType.REGIME_SHUFFLE:
            # Shuffle returns (same as permutation for now)
            return np.random.permutation(returns)
        
        else:
            return returns
    
    def _calculate_default_metrics(self, returns: np.ndarray) -> Dict[str, float]:
        """Calculate default metrics from returns"""
        if len(returns) == 0:
            return {
                "sharpe_ratio": 0.0,
                "total_return": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0,
                "profit_factor": 0.0
            }
        
        # Sharpe ratio
        sharpe = returns.mean() / (returns.std() + 1e-9) * np.sqrt(252) if returns.std() > 0 else 0.0
        
        # Total return
        total_return = returns.sum()
        
        # Max drawdown
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = cumulative - running_max
        max_drawdown = abs(np.min(drawdown)) if len(drawdown) > 0 else 0.0
        
        # Win rate
        wins = np.sum(returns > 0)
        win_rate = wins / len(returns) if len(returns) > 0 else 0.0
        
        # Profit factor
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        avg_winner = positive_returns.mean() if len(positive_returns) > 0 else 0.0
        avg_loser = abs(negative_returns.mean()) if len(negative_returns) > 0 else 1.0
        profit_factor = avg_winner / avg_loser if avg_loser > 0 else 0.0
        
        return {
            "sharpe_ratio": sharpe,
            "total_return": total_return,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "profit_factor": profit_factor
        }
    
    def analyze_robustness(
        self,
        original_sharpe: float,
        random_sharpes: List[float],
        confidence_level: float = 0.95
    ) -> RobustnessMetrics:
        """
        Analyze robustness from Monte Carlo results.
        
        Args:
            original_sharpe: Original strategy Sharpe ratio
            random_sharpes: List of Sharpe ratios from randomizations
            confidence_level: Confidence level (default: 0.95)
        
        Returns:
            RobustnessMetrics
        """
        if not random_sharpes:
            return RobustnessMetrics(
                original_sharpe=original_sharpe,
                mean_random_sharpe=0.0,
                std_random_sharpe=0.0,
                p_value=1.0,
                z_score=0.0,
                percentile_rank=0.0,
                is_robust=False,
                robustness_score=0.0
            )
        
        random_sharpes_array = np.array(random_sharpes)
        mean_random = np.mean(random_sharpes_array)
        std_random = np.std(random_sharpes_array)
        
        # Calculate z-score
        if std_random > 0:
            z_score = (original_sharpe - mean_random) / std_random
        else:
            z_score = 0.0
        
        # Calculate p-value (one-sided test)
        p_value = 1.0 - self._normal_cdf(z_score)
        
        # Calculate percentile rank
        percentile_rank = np.mean(original_sharpe > random_sharpes_array)
        
        # Determine if robust
        is_robust = p_value < (1 - confidence_level) and percentile_rank > confidence_level
        
        # Calculate robustness score (0-1)
        # Higher score = more robust
        robustness_score = min(percentile_rank, 1.0 - p_value)
        
        return RobustnessMetrics(
            original_sharpe=original_sharpe,
            mean_random_sharpe=mean_random,
            std_random_sharpe=std_random,
            p_value=p_value,
            z_score=z_score,
            percentile_rank=percentile_rank,
            is_robust=is_robust,
            robustness_score=robustness_score
        )
    
    def _normal_cdf(self, z: float) -> float:
        """Cumulative distribution function of standard normal"""
        # Approximation using error function
        return 0.5 * (1 + np.sign(z) * (1 - np.exp(-2 * z**2 / np.pi)))
    
    def get_performance_distribution(
        self,
        results: List[MonteCarloResult],
        metric: str = "sharpe_ratio"
    ) -> Dict[str, float]:
        """
        Get performance distribution statistics.
        
        Args:
            results: Monte Carlo results
            metric: Metric to analyze
        
        Returns:
            Distribution statistics
        """
        values = [getattr(r, metric) for r in results]
        
        if not values:
            return {}
        
        values_array = np.array(values)
        
        return {
            "mean": float(np.mean(values_array)),
            "median": float(np.median(values_array)),
            "std": float(np.std(values_array)),
            "min": float(np.min(values_array)),
            "max": float(np.max(values_array)),
            "p5": float(np.percentile(values_array, 5)),
            "p25": float(np.percentile(values_array, 25)),
            "p75": float(np.percentile(values_array, 75)),
            "p95": float(np.percentile(values_array, 95)),
            "p99": float(np.percentile(values_array, 99))
        }
    
    def sensitivity_analysis(
        self,
        returns: np.ndarray,
        parameter_values: List[float],
        parameter_name: str,
        strategy_func: Callable[[np.ndarray, float], Dict[str, float]]
    ) -> Dict[float, Dict[str, float]]:
        """
        Perform sensitivity analysis on a parameter.
        
        Args:
            returns: Returns array
            parameter_values: List of parameter values to test
            parameter_name: Parameter name
            strategy_func: Strategy function that takes returns and parameter
        
        Returns:
            Dictionary mapping parameter values to metrics
        """
        results = {}
        
        for param_value in parameter_values:
            metrics = strategy_func(returns, param_value)
            results[param_value] = metrics
        
        logger.info(
            "sensitivity_analysis_complete",
            parameter=parameter_name,
            values_tested=len(parameter_values)
        )
        
        return results
    
    def visualize_results(
        self,
        original_sharpe: float,
        random_sharpes: List[float],
        output_path: Optional[str] = None
    ) -> None:
        """
        Visualize robustness analysis results.
        
        Args:
            original_sharpe: Original Sharpe ratio
            random_sharpes: List of random Sharpe ratios
            output_path: Output path for visualization (optional)
        """
        try:
            import matplotlib.pyplot as plt
            
            # Create histogram
            plt.figure(figsize=(10, 6))
            plt.hist(random_sharpes, bins=50, alpha=0.7, label="Randomized Strategies")
            plt.axvline(original_sharpe, color='r', linestyle='--', linewidth=2, label=f"Original Strategy (Sharpe={original_sharpe:.2f})")
            plt.xlabel("Sharpe Ratio")
            plt.ylabel("Frequency")
            plt.title("Robustness Analysis: Sharpe Ratio Distribution")
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            if output_path:
                plt.savefig(output_path)
                logger.info("visualization_saved", path=output_path)
            else:
                plt.show()
            
            plt.close()
            
        except ImportError:
            logger.warning("matplotlib_not_available", message="Install matplotlib for visualization")
        except Exception as e:
            logger.error("visualization_failed", error=str(e))

