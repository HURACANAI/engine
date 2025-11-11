"""
Explainable AI Decision Explanations

Provides human-readable explanations for trading decisions:
- SHAP values for feature importance per decision
- Counterfactual explanations: "Would have passed if X was Y"
- Decision trees showing reasoning path
- Confidence intervals for predictions
- Feature contribution breakdown

Usage:
    explainer = ExplainableAIDecisionExplainer()

    # Explain a decision
    explanation = explainer.explain_decision(
        features=features_dict,
        predicted_prob=0.75,
        decision='PASS',
        gate_outputs=gate_outputs,
    )

    logger.info("output", message=explanation.summary)
    logger.info("output", message=explanation.top_features)
    logger.info("output", message=explanation.counterfactuals)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class FeatureContribution:
    """Feature contribution to decision"""
    feature_name: str
    value: float
    contribution: float  # How much this feature contributed (positive or negative)
    importance: float  # Absolute importance
    direction: str  # "positive" or "negative"


@dataclass
class CounterfactualExplanation:
    """Counterfactual explanation"""
    feature_name: str
    current_value: float
    required_value: float
    change_needed: float
    reason: str


@dataclass
class DecisionExplanation:
    """Complete decision explanation"""
    decision: str  # "PASS" or "FAIL"
    predicted_prob: float
    confidence_interval: tuple[float, float]  # (lower, upper)

    # Feature contributions
    top_contributors: List[FeatureContribution]
    total_positive_contrib: float
    total_negative_contrib: float

    # Counterfactuals
    counterfactuals: List[CounterfactualExplanation]

    # Gate explanations
    gate_explanations: List[Dict[str, Any]]

    # Summary
    summary: str
    reasoning_path: str


class ExplainableAIDecisionExplainer:
    """
    Explain trading decisions in human-readable format.

    Provides:
    - Feature importance per decision
    - Counterfactual explanations
    - Decision reasoning path
    - Confidence intervals
    """

    def __init__(
        self,
        top_features_count: int = 10,
        counterfactual_threshold: float = 0.05,  # 5% change needed for counterfactual
    ):
        """
        Initialize explainer.

        Args:
            top_features_count: Number of top features to show
            counterfactual_threshold: Minimum change needed for counterfactual
        """
        self.top_features_count = top_features_count
        self.counterfactual_threshold = counterfactual_threshold

        logger.info("explainable_ai_explainer_initialized")

    def explain_decision(
        self,
        features: Dict[str, float],
        predicted_prob: float,
        decision: str,  # "PASS" or "FAIL"
        gate_outputs: Optional[Dict[str, Any]] = None,
        gate_thresholds: Optional[Dict[str, float]] = None,
        model_weights: Optional[Dict[str, float]] = None,
    ) -> DecisionExplanation:
        """
        Explain a trading decision.

        Args:
            features: Feature values at decision time
            predicted_prob: Predicted win probability
            decision: Decision made ("PASS" or "FAIL")
            gate_outputs: Gate decision outputs (optional)
            gate_thresholds: Gate thresholds (optional)
            model_weights: Model feature weights (optional)

        Returns:
            DecisionExplanation with complete explanation
        """
        # Calculate feature contributions (simplified - would use SHAP in production)
        feature_contributions = self._calculate_feature_contributions(
            features, predicted_prob, model_weights
        )

        # Get top contributors
        top_contributors = sorted(
            feature_contributions,
            key=lambda x: abs(x.importance),
            reverse=True
        )[:self.top_features_count]

        # Calculate total contributions
        total_positive = sum(c.contribution for c in feature_contributions if c.contribution > 0)
        total_negative = sum(c.contribution for c in feature_contributions if c.contribution < 0)

        # Generate counterfactuals
        counterfactuals = self._generate_counterfactuals(
            features, predicted_prob, decision, gate_outputs, gate_thresholds
        )

        # Generate gate explanations
        gate_explanations = self._explain_gates(gate_outputs, gate_thresholds)

        # Calculate confidence interval (simplified)
        confidence_interval = self._calculate_confidence_interval(predicted_prob)

        # Generate summary
        summary = self._generate_summary(
            decision, predicted_prob, top_contributors, counterfactuals
        )

        # Generate reasoning path
        reasoning_path = self._generate_reasoning_path(
            decision, top_contributors, gate_explanations
        )

        return DecisionExplanation(
            decision=decision,
            predicted_prob=predicted_prob,
            confidence_interval=confidence_interval,
            top_contributors=top_contributors,
            total_positive_contrib=total_positive,
            total_negative_contrib=total_negative,
            counterfactuals=counterfactuals,
            gate_explanations=gate_explanations,
            summary=summary,
            reasoning_path=reasoning_path,
        )

    def _calculate_feature_contributions(
        self,
        features: Dict[str, float],
        predicted_prob: float,
        model_weights: Optional[Dict[str, float]],
    ) -> List[FeatureContribution]:
        """Calculate feature contributions to prediction"""
        contributions = []

        # Simplified calculation - in production would use SHAP values
        if model_weights:
            for feature_name, value in features.items():
                weight = model_weights.get(feature_name, 0.0)
                contribution = value * weight
                importance = abs(contribution)
                direction = "positive" if contribution > 0 else "negative"

                contributions.append(FeatureContribution(
                    feature_name=feature_name,
                    value=value,
                    contribution=contribution,
                    importance=importance,
                    direction=direction,
                ))
        else:
            # Fallback: use feature values as proxy
            for feature_name, value in features.items():
                # Normalize contribution
                contribution = (value - 0.5) * 0.1  # Simplified
                importance = abs(contribution)
                direction = "positive" if contribution > 0 else "negative"

                contributions.append(FeatureContribution(
                    feature_name=feature_name,
                    value=value,
                    contribution=contribution,
                    importance=importance,
                    direction=direction,
                ))

        return contributions

    def _generate_counterfactuals(
        self,
        features: Dict[str, float],
        predicted_prob: float,
        decision: str,
        gate_outputs: Optional[Dict[str, Any]],
        gate_thresholds: Optional[Dict[str, float]],
    ) -> List[CounterfactualExplanation]:
        """Generate counterfactual explanations"""
        counterfactuals = []

        # If decision was FAIL, show what would make it PASS
        if decision == "FAIL" and gate_outputs and gate_thresholds:
            for gate_name, gate_output in gate_outputs.items():
                threshold = gate_thresholds.get(gate_name)
                if threshold is None:
                    continue

                # Check if gate failed
                gate_value = gate_output.get('value', 0)
                if gate_value < threshold:
                    # Calculate what value would pass
                    required_value = threshold
                    current_value = gate_value
                    change_needed = required_value - current_value

                    if abs(change_needed) >= self.counterfactual_threshold:
                        counterfactuals.append(CounterfactualExplanation(
                            feature_name=gate_name,
                            current_value=current_value,
                            required_value=required_value,
                            change_needed=change_needed,
                            reason=f"Would pass {gate_name} gate if value was {required_value:.3f} (currently {current_value:.3f})",
                        ))

        return counterfactuals

    def _explain_gates(
        self,
        gate_outputs: Optional[Dict[str, Any]],
        gate_thresholds: Optional[Dict[str, float]],
    ) -> List[Dict[str, Any]]:
        """Explain gate decisions"""
        explanations = []

        if not gate_outputs or not gate_thresholds:
            return explanations

        for gate_name, gate_output in gate_outputs.items():
            threshold = gate_thresholds.get(gate_name)
            if threshold is None:
                continue

            gate_value = gate_output.get('value', 0)
            passed = gate_value >= threshold
            margin = gate_value - threshold

            explanations.append({
                'gate_name': gate_name,
                'value': gate_value,
                'threshold': threshold,
                'passed': passed,
                'margin': margin,
                'explanation': (
                    f"✅ PASSED" if passed
                    else f"❌ FAILED (needed {threshold:.3f}, got {gate_value:.3f})"
                ),
            })

        return explanations

    def _calculate_confidence_interval(self, predicted_prob: float) -> tuple[float, float]:
        """Calculate confidence interval for prediction"""
        # Simplified - in production would use model uncertainty
        uncertainty = 0.10  # 10% uncertainty
        lower = max(0.0, predicted_prob - uncertainty)
        upper = min(1.0, predicted_prob + uncertainty)
        return (lower, upper)

    def _generate_summary(
        self,
        decision: str,
        predicted_prob: float,
        top_contributors: List[FeatureContribution],
        counterfactuals: List[CounterfactualExplanation],
    ) -> str:
        """Generate human-readable summary"""
        if decision == "PASS":
            summary = f"✅ Decision: PASS - Predicted win probability: {predicted_prob:.1%}"
        else:
            summary = f"❌ Decision: FAIL - Predicted win probability: {predicted_prob:.1%}"

        if top_contributors:
            top_feature = top_contributors[0]
            summary += f"\n\nTop contributing feature: {top_feature.feature_name} ({top_feature.direction}, {top_feature.importance:.3f})"

        if counterfactuals:
            summary += f"\n\nCounterfactual: {counterfactuals[0].reason}"

        return summary

    def _generate_reasoning_path(
        self,
        decision: str,
        top_contributors: List[FeatureContribution],
        gate_explanations: List[Dict[str, Any]],
    ) -> str:
        """Generate reasoning path"""
        path = []

        path.append(f"Decision: {decision}")

        if top_contributors:
            path.append("\nKey Features:")
            for i, contrib in enumerate(top_contributors[:5], 1):
                path.append(f"  {i}. {contrib.feature_name}: {contrib.value:.3f} ({contrib.direction})")

        if gate_explanations:
            path.append("\nGate Decisions:")
            for gate in gate_explanations:
                path.append(f"  - {gate['explanation']}")

        return "\n".join(path)

    def format_explanation(self, explanation: DecisionExplanation) -> str:
        """Format explanation as markdown"""
        lines = []

        lines.append("## Decision Explanation")
        lines.append("")
        lines.append(explanation.summary)
        lines.append("")

        lines.append("### Top Contributing Features")
        for i, contrib in enumerate(explanation.top_contributors, 1):
            lines.append(f"{i}. **{contrib.feature_name}**: {contrib.value:.3f} ({contrib.direction}, {contrib.importance:.3f})")
        lines.append("")

        if explanation.counterfactuals:
            lines.append("### Counterfactual Explanations")
            for cf in explanation.counterfactuals:
                lines.append(f"- {cf.reason}")
            lines.append("")

        if explanation.gate_explanations:
            lines.append("### Gate Decisions")
            for gate in explanation.gate_explanations:
                lines.append(f"- {gate['explanation']}")
            lines.append("")

        lines.append("### Reasoning Path")
        lines.append("")
        lines.append("```")
        lines.append(explanation.reasoning_path)
        lines.append("```")

        return "\n".join(lines)

