"""
A/B Testing Framework - Statistical Testing Layer

For performance validation and model comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import structlog
from scipy import stats

logger = structlog.get_logger(__name__)


@dataclass
class ABTestResult:
    """A/B test result."""
    
    test_name: str
    metric_name: str
    group_a_mean: float
    group_b_mean: float
    difference: float
    p_value: float
    is_significant: bool
    confidence_interval: tuple[float, float]
    effect_size: float
    sample_size_a: int
    sample_size_b: int


class ABTestingFramework:
    """
    A/B testing framework for model performance validation.
    
    Features:
    - Statistical hypothesis testing
    - Confidence intervals
    - Effect size calculation
    - Multiple comparison correction
    """
    
    def __init__(self, alpha: float = 0.05):
        """
        Initialize A/B testing framework.
        
        Args:
            alpha: Significance level
        """
        self.alpha = alpha
        logger.info("ab_testing_framework_initialized", alpha=alpha)
    
    def t_test(
        self,
        group_a: np.ndarray,
        group_b: np.ndarray,
        test_name: str = "t_test",
        metric_name: str = "metric",
        alternative: str = "two-sided",
    ) -> ABTestResult:
        """
        Perform t-test for comparing two groups.
        
        Args:
            group_a: Group A data
            group_b: Group B data
            test_name: Name of the test
            metric_name: Name of the metric being tested
            alternative: Alternative hypothesis ("two-sided", "less", "greater")
            
        Returns:
            ABTestResult with test results
        """
        logger.info("performing_t_test", test_name=test_name)
        
        # Calculate statistics
        mean_a = np.mean(group_a)
        mean_b = np.mean(group_b)
        difference = mean_b - mean_a
        
        # Perform t-test
        t_stat, p_value = stats.ttest_ind(group_a, group_b, alternative=alternative)
        
        # Calculate confidence interval
        se = np.sqrt(np.var(group_a) / len(group_a) + np.var(group_b) / len(group_b))
        t_critical = stats.t.ppf(1 - self.alpha / 2, len(group_a) + len(group_b) - 2)
        ci_lower = difference - t_critical * se
        ci_upper = difference + t_critical * se
        
        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt((np.var(group_a) + np.var(group_b)) / 2)
        effect_size = difference / pooled_std if pooled_std > 0 else 0.0
        
        # Check significance
        is_significant = p_value < self.alpha
        
        result = ABTestResult(
            test_name=test_name,
            metric_name=metric_name,
            group_a_mean=float(mean_a),
            group_b_mean=float(mean_b),
            difference=float(difference),
            p_value=float(p_value),
            is_significant=is_significant,
            confidence_interval=(float(ci_lower), float(ci_upper)),
            effect_size=float(effect_size),
            sample_size_a=len(group_a),
            sample_size_b=len(group_b),
        )
        
        logger.info(
            "t_test_complete",
            test_name=test_name,
            is_significant=is_significant,
            p_value=p_value,
            effect_size=effect_size,
        )
        
        return result
    
    def mann_whitney_test(
        self,
        group_a: np.ndarray,
        group_b: np.ndarray,
        test_name: str = "mann_whitney",
        metric_name: str = "metric",
        alternative: str = "two-sided",
    ) -> ABTestResult:
        """
        Perform Mann-Whitney U test (non-parametric).
        
        Args:
            group_a: Group A data
            group_b: Group B data
            test_name: Name of the test
            metric_name: Name of the metric being tested
            alternative: Alternative hypothesis
            
        Returns:
            ABTestResult with test results
        """
        logger.info("performing_mann_whitney_test", test_name=test_name)
        
        # Calculate statistics
        mean_a = np.mean(group_a)
        mean_b = np.mean(group_b)
        difference = mean_b - mean_a
        
        # Perform Mann-Whitney U test
        u_stat, p_value = stats.mannwhitneyu(group_a, group_b, alternative=alternative)
        
        # Approximate confidence interval (simplified)
        se = np.sqrt(np.var(group_a) / len(group_a) + np.var(group_b) / len(group_b))
        z_critical = stats.norm.ppf(1 - self.alpha / 2)
        ci_lower = difference - z_critical * se
        ci_upper = difference + z_critical * se
        
        # Effect size
        pooled_std = np.sqrt((np.var(group_a) + np.var(group_b)) / 2)
        effect_size = difference / pooled_std if pooled_std > 0 else 0.0
        
        is_significant = p_value < self.alpha
        
        result = ABTestResult(
            test_name=test_name,
            metric_name=metric_name,
            group_a_mean=float(mean_a),
            group_b_mean=float(mean_b),
            difference=float(difference),
            p_value=float(p_value),
            is_significant=is_significant,
            confidence_interval=(float(ci_lower), float(ci_upper)),
            effect_size=float(effect_size),
            sample_size_a=len(group_a),
            sample_size_b=len(group_b),
        )
        
        logger.info("mann_whitney_test_complete", is_significant=is_significant, p_value=p_value)
        return result
    
    def compare_models(
        self,
        model_a_results: Dict[str, np.ndarray],
        model_b_results: Dict[str, np.ndarray],
        test_type: str = "t_test",
    ) -> Dict[str, ABTestResult]:
        """
        Compare two models across multiple metrics.
        
        Args:
            model_a_results: Dictionary of metric names to values for model A
            model_b_results: Dictionary of metric names to values for model B
            test_type: Type of test ("t_test", "mann_whitney")
            
        Returns:
            Dictionary of metric names to test results
        """
        logger.info("comparing_models", test_type=test_type)
        
        results = {}
        
        # Get common metrics
        common_metrics = set(model_a_results.keys()) & set(model_b_results.keys())
        
        for metric in common_metrics:
            group_a = model_a_results[metric]
            group_b = model_b_results[metric]
            
            if test_type == "t_test":
                result = self.t_test(group_a, group_b, test_name=f"model_comparison_{metric}", metric_name=metric)
            elif test_type == "mann_whitney":
                result = self.mann_whitney_test(group_a, group_b, test_name=f"model_comparison_{metric}", metric_name=metric)
            else:
                raise ValueError(f"Unknown test type: {test_type}")
            
            results[metric] = result
        
        logger.info("model_comparison_complete", num_metrics=len(results))
        return results
    
    def multiple_comparison_correction(
        self,
        results: List[ABTestResult],
        method: str = "bonferroni",
    ) -> List[ABTestResult]:
        """
        Apply multiple comparison correction.
        
        Args:
            results: List of test results
            method: Correction method ("bonferroni", "fdr_bh")
            
        Returns:
            List of corrected test results
        """
        p_values = [r.p_value for r in results]
        
        try:
            from statsmodels.stats.multitest import multipletests
            HAS_STATSMODELS = True
        except ImportError:
            HAS_STATSMODELS = False
            logger.warning("statsmodels_not_available_using_simple_bonferroni")
        
        if HAS_STATSMODELS:
            if method == "bonferroni":
                corrected_p_values = multipletests(p_values, method="bonferroni")[1]
            elif method == "fdr_bh":
                corrected_p_values = multipletests(p_values, method="fdr_bh")[1]
            else:
                corrected_p_values = p_values
        else:
            # Simple Bonferroni correction (no statsmodels)
            corrected_p_values = [min(p * len(p_values), 1.0) for p in p_values]
        
        # Update results with corrected p-values
        corrected_results = []
        for result, corrected_p in zip(results, corrected_p_values):
            result.p_value = float(corrected_p)
            result.is_significant = corrected_p < self.alpha
            corrected_results.append(result)
        
        logger.info("multiple_comparison_correction_complete", method=method)
        return corrected_results

