"""Evaluation modules for counterfactual analysis and performance evaluation."""

from .counterfactual_evaluator import (
    CounterfactualEvaluator,
    CounterfactualTrade,
    ExitRuleRecommendation,
    CounterfactualReport,
    RegretType,
)

__all__ = [
    "CounterfactualEvaluator",
    "CounterfactualTrade",
    "ExitRuleRecommendation",
    "CounterfactualReport",
    "RegretType",
]

