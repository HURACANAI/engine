"""
Counterfactual Evaluator

Evaluates counterfactual "what if" scenarios.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


class DecisionQuality(str, Enum):
    """Decision quality classification"""
    EXCELLENT = "excellent"  # Top 20% of decisions
    GOOD = "good"  # Top 40%
    AVERAGE = "average"  # Middle 40%
    POOR = "poor"  # Bottom 40%
    VERY_POOR = "very_poor"  # Bottom 20%


@dataclass
class CounterfactualResult:
    """
    Counterfactual evaluation result

    Compares actual decision to alternatives.
    """
    decision_taken: str
    actual_outcome: float  # PnL in bps

    # Counterfactuals
    alternative_outcomes: Dict[str, float]
    best_alternative: str
    best_alternative_outcome: float

    # Analysis
    opportunity_cost_bps: float  # Lost profit vs best alternative
    decision_quality: DecisionQuality
    quality_score: float  # [0-1]

    should_have_acted_differently: bool

    # Context
    market_context: Optional[Dict] = None


class CounterfactualEvaluator:
    """
    Counterfactual Evaluator

    Evaluates "what if" scenarios for trading decisions.

    Helps understand:
    - Was the decision optimal?
    - What was the opportunity cost?
    - How to improve future decisions?

    Example:
        evaluator = CounterfactualEvaluator()

        # Evaluate a trade
        result = evaluator.evaluate_decision(
            decision="BUY",
            actual_pnl_bps=50,
            alternative_actions={
                "SELL": -30,
                "HOLD": 0,
                "BUY_DOUBLE": 100
            }
        )

        if result.opportunity_cost_bps > 20:
            print(f"Missed {result.opportunity_cost_bps:.0f} bps")
            print(f"Should have: {result.best_alternative}")
    """

    def __init__(self):
        """Initialize counterfactual evaluator"""
        self.evaluation_history: List[CounterfactualResult] = []

    def evaluate_decision(
        self,
        decision: str,
        actual_pnl_bps: float,
        alternative_actions: Dict[str, float],
        market_context: Optional[Dict] = None
    ) -> CounterfactualResult:
        """
        Evaluate a trading decision

        Args:
            decision: Decision that was taken
            actual_pnl_bps: Actual PnL in basis points
            alternative_actions: Dict of {action: estimated_pnl_bps}
            market_context: Optional market context

        Returns:
            CounterfactualResult

        Example:
            result = evaluator.evaluate_decision(
                decision="BUY",
                actual_pnl_bps=50,
                alternative_actions={"SELL": -30, "HOLD": 0}
            )
        """
        # Add actual decision to alternatives
        all_outcomes = {decision: actual_pnl_bps}
        all_outcomes.update(alternative_actions)

        # Find best alternative
        best_alternative = max(all_outcomes, key=all_outcomes.get)
        best_outcome = all_outcomes[best_alternative]

        # Calculate opportunity cost
        opportunity_cost = best_outcome - actual_pnl_bps

        # Should have acted differently?
        should_have_acted_differently = best_alternative != decision

        # Calculate decision quality
        quality_score = self._calculate_quality_score(
            actual_pnl_bps,
            best_outcome,
            all_outcomes
        )

        decision_quality = self._classify_quality(quality_score)

        result = CounterfactualResult(
            decision_taken=decision,
            actual_outcome=actual_pnl_bps,
            alternative_outcomes=alternative_actions,
            best_alternative=best_alternative,
            best_alternative_outcome=best_outcome,
            opportunity_cost_bps=opportunity_cost,
            decision_quality=decision_quality,
            quality_score=quality_score,
            should_have_acted_differently=should_have_acted_differently,
            market_context=market_context
        )

        self.evaluation_history.append(result)

        logger.info(
            "decision_evaluated",
            decision=decision,
            actual_pnl=actual_pnl_bps,
            best_alternative=best_alternative,
            opportunity_cost=opportunity_cost,
            quality=decision_quality.value
        )

        return result

    def evaluate_portfolio_decisions(
        self,
        trades_df: pd.DataFrame,
        decision_col: str = "decision",
        pnl_col: str = "pnl_bps"
    ) -> pd.DataFrame:
        """
        Evaluate a portfolio of decisions

        Args:
            trades_df: DataFrame with trades
            decision_col: Column with decision taken
            pnl_col: Column with actual PnL

        Returns:
            DataFrame with counterfactual analysis

        Example:
            analysis = evaluator.evaluate_portfolio_decisions(trades_df)
            print(f"Total opportunity cost: {analysis['opportunity_cost'].sum()}")
        """
        results = []

        for idx, trade in trades_df.iterrows():
            # For simplicity, assume alternative was to not trade
            result = self.evaluate_decision(
                decision=trade[decision_col],
                actual_pnl_bps=trade[pnl_col],
                alternative_actions={"NO_TRADE": 0.0}
            )

            results.append({
                "trade_idx": idx,
                "decision": result.decision_taken,
                "actual_pnl": result.actual_outcome,
                "opportunity_cost": result.opportunity_cost_bps,
                "quality_score": result.quality_score,
                "decision_quality": result.decision_quality.value
            })

        return pd.DataFrame(results)

    def _calculate_quality_score(
        self,
        actual_outcome: float,
        best_outcome: float,
        all_outcomes: Dict[str, float]
    ) -> float:
        """
        Calculate decision quality score

        Args:
            actual_outcome: Actual PnL
            best_outcome: Best possible PnL
            all_outcomes: All possible outcomes

        Returns:
            Quality score [0-1]
        """
        outcomes_list = list(all_outcomes.values())
        worst_outcome = min(outcomes_list)
        outcome_range = best_outcome - worst_outcome

        if outcome_range == 0:
            return 0.5  # All outcomes same

        # Normalize actual outcome to [0, 1]
        normalized = (actual_outcome - worst_outcome) / outcome_range

        return normalized

    def _classify_quality(self, quality_score: float) -> DecisionQuality:
        """
        Classify decision quality from score

        Args:
            quality_score: Quality score [0-1]

        Returns:
            DecisionQuality
        """
        if quality_score >= 0.8:
            return DecisionQuality.EXCELLENT
        elif quality_score >= 0.6:
            return DecisionQuality.GOOD
        elif quality_score >= 0.4:
            return DecisionQuality.AVERAGE
        elif quality_score >= 0.2:
            return DecisionQuality.POOR
        else:
            return DecisionQuality.VERY_POOR

    def get_decision_statistics(self) -> Dict:
        """
        Get statistics on historical decisions

        Returns:
            Dict with decision statistics

        Example:
            stats = evaluator.get_decision_statistics()
            print(f"Average opportunity cost: {stats['avg_opportunity_cost']:.1f} bps")
        """
        if len(self.evaluation_history) == 0:
            return {}

        opportunity_costs = [r.opportunity_cost_bps for r in self.evaluation_history]
        quality_scores = [r.quality_score for r in self.evaluation_history]

        num_should_have_changed = sum(
            1 for r in self.evaluation_history
            if r.should_have_acted_differently
        )

        return {
            "num_decisions": len(self.evaluation_history),
            "avg_opportunity_cost": np.mean(opportunity_costs),
            "total_opportunity_cost": np.sum(opportunity_costs),
            "avg_quality_score": np.mean(quality_scores),
            "pct_suboptimal": (num_should_have_changed / len(self.evaluation_history)) * 100,
            "quality_distribution": self._get_quality_distribution()
        }

    def _get_quality_distribution(self) -> Dict[str, int]:
        """Get distribution of decision qualities"""
        distribution = {
            quality.value: 0
            for quality in DecisionQuality
        }

        for result in self.evaluation_history:
            distribution[result.decision_quality.value] += 1

        return distribution

    def generate_report(self) -> str:
        """
        Generate counterfactual analysis report

        Returns:
            Report string
        """
        stats = self.get_decision_statistics()

        if len(stats) == 0:
            return "No decisions evaluated yet"

        lines = [
            "=" * 80,
            "COUNTERFACTUAL ANALYSIS REPORT",
            "=" * 80,
            "",
            f"Total Decisions Evaluated: {stats['num_decisions']}",
            f"Average Opportunity Cost: {stats['avg_opportunity_cost']:.1f} bps",
            f"Total Opportunity Cost: {stats['total_opportunity_cost']:.1f} bps",
            f"Average Quality Score: {stats['avg_quality_score']:.2%}",
            f"Suboptimal Decisions: {stats['pct_suboptimal']:.1f}%",
            "",
            "QUALITY DISTRIBUTION:",
            "-" * 80
        ]

        for quality, count in stats['quality_distribution'].items():
            pct = (count / stats['num_decisions']) * 100
            lines.append(f"  {quality:15s}: {count:4d} ({pct:5.1f}%)")

        lines.append("")
        lines.append("=" * 80)

        return "\n".join(lines)
