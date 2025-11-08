"""
Permutation Testing Module

Validates whether a strategy's backtest performance is statistically significant
or random using permutation tests.

Key Features:
- Randomly shuffle trade sequences 1,000+ times
- Compare actual Sharpe/return distribution to random permutations
- Calculate p-values for statistical significance
- Pass/fail models based on 99th percentile threshold
- Integrate with Council voting system

Author: Huracan Engine Team
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


class PermutationResult(Enum):
    """Permutation test result"""
    SIGNIFICANT = "significant"  # p-value < 0.01
    MARGINAL = "marginal"  # 0.01 <= p-value < 0.05
    INSIGNIFICANT = "insignificant"  # p-value >= 0.05


@dataclass
class PermutationTestResult:
    """Result of permutation test"""
    actual_sharpe: float
    actual_return: float
    permutation_sharpes: List[float]
    permutation_returns: List[float]
    p_value_sharpe: float
    p_value_return: float
    percentile_rank_sharpe: float  # Where actual Sharpe ranks (0-100)
    percentile_rank_return: float
    is_significant: bool
    result: PermutationResult
    num_permutations: int
    test_duration_seconds: float


class PermutationTester:
    """
    Permutation testing for strategy validation.
    
    Tests whether a strategy's performance is statistically significant
    or could have occurred by random chance.
    
    Usage:
        tester = PermutationTester(num_permutations=1000)
        result = tester.test_strategy(
            trades=[...],  # List of trade returns
            confidence_level=0.99
        )
        
        if result.is_significant:
            print("Strategy passed permutation test!")
    """
    
    def __init__(
        self,
        num_permutations: int = 1000,
        random_seed: Optional[int] = None
    ):
        """
        Initialize permutation tester.
        
        Args:
            num_permutations: Number of random permutations to generate
            random_seed: Random seed for reproducibility
        """
        self.num_permutations = num_permutations
        self.random_seed = random_seed
        
        if random_seed is not None:
            np.random.seed(random_seed)
        
        logger.info(
            "permutation_tester_initialized",
            num_permutations=num_permutations,
            random_seed=random_seed
        )
    
    def test_strategy(
        self,
        trades: List[float],  # Trade returns or PnL values
        confidence_level: float = 0.99,
        metric: str = "sharpe"  # "sharpe", "return", "both"
    ) -> PermutationTestResult:
        """
        Test strategy using permutation testing.
        
        Args:
            trades: List of trade returns (or PnL values)
            confidence_level: Confidence level (0.99 = 99th percentile)
            metric: Metric to test ("sharpe", "return", "both")
        
        Returns:
            PermutationTestResult with p-values and significance
        """
        start_time = time.time()
        
        if len(trades) < 10:
            logger.warning(
                "insufficient_trades_for_permutation",
                num_trades=len(trades),
                message="Need at least 10 trades for permutation test"
            )
            return PermutationTestResult(
                actual_sharpe=0.0,
                actual_return=0.0,
                permutation_sharpes=[],
                permutation_returns=[],
                p_value_sharpe=1.0,
                p_value_return=1.0,
                percentile_rank_sharpe=0.0,
                percentile_rank_return=0.0,
                is_significant=False,
                result=PermutationResult.INSIGNIFICANT,
                num_permutations=0,
                test_duration_seconds=0.0
            )
        
        # Calculate actual metrics
        actual_sharpe = self._calculate_sharpe(trades)
        actual_return = np.mean(trades)
        
        # Generate permutations
        permutation_sharpes = []
        permutation_returns = []
        
        for _ in range(self.num_permutations):
            # Shuffle trades randomly
            shuffled_trades = np.random.permutation(trades).tolist()
            
            # Calculate metrics for shuffled sequence
            perm_sharpe = self._calculate_sharpe(shuffled_trades)
            perm_return = np.mean(shuffled_trades)
            
            permutation_sharpes.append(perm_sharpe)
            permutation_returns.append(perm_return)
        
        # Calculate p-values
        p_value_sharpe = self._calculate_p_value(actual_sharpe, permutation_sharpes)
        p_value_return = self._calculate_p_value(actual_return, permutation_returns)
        
        # Calculate percentile ranks
        percentile_rank_sharpe = self._calculate_percentile_rank(actual_sharpe, permutation_sharpes)
        percentile_rank_return = self._calculate_percentile_rank(actual_return, permutation_returns)
        
        # Determine significance
        threshold = 1.0 - confidence_level  # 0.01 for 99% confidence
        is_significant = p_value_sharpe < threshold
        
        # Determine result
        if p_value_sharpe < 0.01:
            result = PermutationResult.SIGNIFICANT
        elif p_value_sharpe < 0.05:
            result = PermutationResult.MARGINAL
        else:
            result = PermutationResult.INSIGNIFICANT
        
        test_duration = time.time() - start_time
        
        logger.info(
            "permutation_test_complete",
            actual_sharpe=actual_sharpe,
            p_value_sharpe=p_value_sharpe,
            percentile_rank_sharpe=percentile_rank_sharpe,
            is_significant=is_significant,
            result=result.value,
            test_duration=test_duration
        )
        
        return PermutationTestResult(
            actual_sharpe=actual_sharpe,
            actual_return=actual_return,
            permutation_sharpes=permutation_sharpes,
            permutation_returns=permutation_returns,
            p_value_sharpe=p_value_sharpe,
            p_value_return=p_value_return,
            percentile_rank_sharpe=percentile_rank_sharpe,
            percentile_rank_return=percentile_rank_return,
            is_significant=is_significant,
            result=result,
            num_permutations=self.num_permutations,
            test_duration_seconds=test_duration
        )
    
    def _calculate_sharpe(self, returns: List[float]) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) < 2:
            return 0.0
        
        returns_array = np.array(returns)
        mean_return = np.mean(returns_array)
        std_return = np.std(returns_array)
        
        if std_return == 0:
            return 0.0
        
        # Annualized Sharpe (assuming 252 trading days)
        sharpe = (mean_return / std_return) * np.sqrt(252)
        return float(sharpe)
    
    def _calculate_p_value(self, actual_value: float, permutation_values: List[float]) -> float:
        """Calculate p-value (proportion of permutations >= actual value)"""
        if not permutation_values:
            return 1.0
        
        # Count how many permutations have value >= actual
        count_above = sum(1 for v in permutation_values if v >= actual_value)
        p_value = count_above / len(permutation_values)
        
        return p_value
    
    def _calculate_percentile_rank(self, actual_value: float, permutation_values: List[float]) -> float:
        """Calculate percentile rank (0-100)"""
        if not permutation_values:
            return 0.0
        
        # Count how many permutations are below actual value
        count_below = sum(1 for v in permutation_values if v < actual_value)
        percentile_rank = (count_below / len(permutation_values)) * 100.0
        
        return percentile_rank
    
    def test_model_performance(
        self,
        model_id: str,
        trade_results: List[Dict[str, float]],  # [{"return": 0.01, "sharpe": 1.5}, ...]
        confidence_level: float = 0.99
    ) -> Tuple[PermutationTestResult, Dict[str, float]]:
        """
        Test model performance with permutation testing.
        
        Args:
            model_id: Model identifier
            trade_results: List of trade results with returns and Sharpe
            confidence_level: Confidence level
        
        Returns:
            (PermutationTestResult, model_metrics)
        """
        # Extract returns
        returns = [t.get("return", 0.0) for t in trade_results]
        
        # Run permutation test
        result = self.test_strategy(
            trades=returns,
            confidence_level=confidence_level,
            metric="both"
        )
        
        # Calculate model metrics
        model_metrics = {
            "model_id": model_id,
            "num_trades": len(trade_results),
            "actual_sharpe": result.actual_sharpe,
            "actual_return": result.actual_return,
            "p_value_sharpe": result.p_value_sharpe,
            "p_value_return": result.p_value_return,
            "percentile_rank_sharpe": result.percentile_rank_sharpe,
            "is_significant": result.is_significant,
            "permutation_result": result.result.value
        }
        
        return result, model_metrics


class RobustnessAnalyzer:
    """
    Analyzes model robustness using Monte Carlo and permutation testing.
    
    Features:
    - Sensitivity to randomization
    - Noise injection testing
    - Visualize robustness metrics
    - Integration with Council voting
    """
    
    def __init__(self, permutation_tester: Optional[PermutationTester] = None):
        """
        Initialize robustness analyzer.
        
        Args:
            permutation_tester: Optional permutation tester instance
        """
        self.permutation_tester = permutation_tester or PermutationTester()
        
        logger.info("robustness_analyzer_initialized")
    
    def analyze_model_robustness(
        self,
        model_id: str,
        trades: List[float],
        noise_levels: List[float] = [0.01, 0.05, 0.10],  # Noise as percentage
        num_monte_carlo_runs: int = 100
    ) -> Dict[str, any]:
        """
        Analyze model robustness to noise and randomization.
        
        Args:
            model_id: Model identifier
            trades: List of trade returns
            noise_levels: List of noise levels to test
            num_monte_carlo_runs: Number of Monte Carlo runs
        
        Returns:
            Robustness analysis results
        """
        logger.info(
            "robustness_analysis_start",
            model_id=model_id,
            num_trades=len(trades),
            noise_levels=noise_levels
        )
        
        # Base permutation test
        base_result = self.permutation_tester.test_strategy(trades)
        
        # Test with noise injection
        noise_results = {}
        for noise_level in noise_levels:
            noisy_sharpes = []
            for _ in range(num_monte_carlo_runs):
                # Add noise to trades
                noise = np.random.normal(0, noise_level, len(trades))
                noisy_trades = np.array(trades) + noise
                
                # Calculate Sharpe
                sharpe = self.permutation_tester._calculate_sharpe(noisy_trades.tolist())
                noisy_sharpes.append(sharpe)
            
            noise_results[noise_level] = {
                "mean_sharpe": np.mean(noisy_sharpes),
                "std_sharpe": np.std(noisy_sharpes),
                "min_sharpe": np.min(noisy_sharpes),
                "max_sharpe": np.max(noisy_sharpes)
            }
        
        # Calculate robustness score (stability across noise levels)
        robustness_score = self._calculate_robustness_score(base_result, noise_results)
        
        results = {
            "model_id": model_id,
            "base_sharpe": base_result.actual_sharpe,
            "base_p_value": base_result.p_value_sharpe,
            "base_percentile_rank": base_result.percentile_rank_sharpe,
            "is_significant": base_result.is_significant,
            "noise_results": noise_results,
            "robustness_score": robustness_score,
            "recommendation": self._get_recommendation(robustness_score, base_result)
        }
        
        logger.info(
            "robustness_analysis_complete",
            model_id=model_id,
            robustness_score=robustness_score,
            recommendation=results["recommendation"]
        )
        
        return results
    
    def _calculate_robustness_score(
        self,
        base_result: PermutationTestResult,
        noise_results: Dict[float, Dict[str, float]]
    ) -> float:
        """Calculate robustness score (0-1, higher is better)"""
        # Base score from permutation test
        base_score = base_result.percentile_rank_sharpe / 100.0
        
        # Stability score (how much Sharpe varies with noise)
        sharpe_stability = []
        for noise_level, result in noise_results.items():
            # Lower std = more stable = better
            stability = 1.0 / (1.0 + result["std_sharpe"])
            sharpe_stability.append(stability)
        
        stability_score = np.mean(sharpe_stability) if sharpe_stability else 0.5
        
        # Combined robustness score
        robustness_score = (base_score * 0.6) + (stability_score * 0.4)
        
        return float(robustness_score)
    
    def _get_recommendation(
        self,
        robustness_score: float,
        base_result: PermutationTestResult
    ) -> str:
        """Get recommendation based on robustness analysis"""
        if robustness_score >= 0.8 and base_result.is_significant:
            return "APPROVE"
        elif robustness_score >= 0.6 and base_result.result == PermutationResult.MARGINAL:
            return "CONDITIONAL_APPROVE"
        elif robustness_score >= 0.4:
            return "MONITOR"
        else:
            return "REJECT"

