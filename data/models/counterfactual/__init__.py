"""
Counterfactual Evaluator

Evaluates "what if" scenarios for decision validation.

Key Features:
- Counterfactual outcome estimation
- Alternative strategy comparison
- Opportunity cost calculation
- Decision quality scoring

Usage:
    from models.counterfactual import CounterfactualEvaluator

    evaluator = CounterfactualEvaluator()

    # Evaluate a trade decision
    result = evaluator.evaluate_decision(
        decision="BUY",
        actual_pnl_bps=50,
        alternative_actions={"SELL": -30, "HOLD": 0},
        market_context={"regime": "trending", "volatility": 0.2}
    )

    print(f"Decision quality: {result.quality_score:.2f}")
    print(f"Opportunity cost: {result.opportunity_cost_bps:.1f} bps")

    if result.should_have_acted_differently:
        print(f"Better action: {result.best_alternative}")
"""

from .evaluator import (
    CounterfactualEvaluator,
    CounterfactualResult,
    DecisionQuality
)

__all__ = [
    "CounterfactualEvaluator",
    "CounterfactualResult",
    "DecisionQuality",
]
