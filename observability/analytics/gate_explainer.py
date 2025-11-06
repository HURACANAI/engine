"""
Gate Explainer

Explains why gates rejected signals with detailed reasoning.

Provides:
- Human-readable rejection reasons
- Margin analysis (how close to passing)
- Feature contributions
- Counterfactual analysis
- Actionable suggestions

This answers: "Why was this signal blocked? What would need to change to pass?"

Usage:
    explainer = GateExplainer()

    explanation = explainer.explain_rejection(
        gate_name="meta_label",
        decision="FAIL",
        inputs={"probability": 0.42},
        context={"threshold": 0.45, "mode": "scalp"}
    )

    print(explanation.summary)
    print(explanation.what_to_change)
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import structlog
import yaml
from pathlib import Path

logger = structlog.get_logger(__name__)


@dataclass
class GateExplanation:
    """Structured gate rejection explanation"""
    gate_name: str
    decision: str  # "PASS", "FAIL"
    reason: str
    margin: Optional[float]  # How far from threshold
    margin_pct: Optional[float]  # Margin as percentage
    inputs: Dict[str, float]
    thresholds: Dict[str, float]

    # Counterfactual
    would_pass_if: List[str]
    what_to_change: str

    # Context
    mode: str
    is_good_block: Optional[bool]  # Was blocking this signal correct?
    counterfactual_pnl: Optional[float]  # What would have happened

    # Formatting
    summary: str
    detailed: str


class GateExplainer:
    """
    Explain gate decisions with rich context.

    Uses gates.yaml for thresholds and explanations.
    """

    def __init__(self, config_path: str = "observability/configs/gates.yaml"):
        """
        Initialize gate explainer.

        Args:
            config_path: Path to gates.yaml
        """
        self.config_path = Path(config_path)

        if not self.config_path.exists():
            logger.warning("gates_yaml_not_found", path=str(self.config_path))
            self.config = {"gates": {}}
        else:
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)

        logger.info("gate_explainer_initialized", config_path=str(self.config_path))

    def explain_rejection(
        self,
        gate_name: str,
        decision: str,
        inputs: Dict[str, float],
        context: Dict[str, Any],
        counterfactual_pnl: Optional[float] = None
    ) -> GateExplanation:
        """
        Explain a gate decision.

        Args:
            gate_name: Name of gate
            decision: "PASS" or "FAIL"
            inputs: Gate input values
            context: Context (mode, thresholds, etc.)
            counterfactual_pnl: What happened if signal was taken

        Returns:
            GateExplanation with detailed reasoning
        """
        mode = context.get('mode', 'unknown')

        # Get gate config
        gate_config = self.config.get('gates', {}).get(gate_name, {})

        # Generate explanation based on gate type
        if gate_name == "meta_label":
            return self._explain_meta_label(decision, inputs, context, counterfactual_pnl)
        elif gate_name == "cost_gate":
            return self._explain_cost_gate(decision, inputs, context, counterfactual_pnl)
        elif gate_name == "confidence_gate":
            return self._explain_confidence_gate(decision, inputs, context, counterfactual_pnl)
        elif gate_name == "regime_gate":
            return self._explain_regime_gate(decision, inputs, context, counterfactual_pnl)
        elif gate_name == "spread_gate":
            return self._explain_spread_gate(decision, inputs, context, counterfactual_pnl)
        elif gate_name == "volume_gate":
            return self._explain_volume_gate(decision, inputs, context, counterfactual_pnl)
        else:
            return self._explain_generic(gate_name, decision, inputs, context, counterfactual_pnl)

    def _explain_meta_label(
        self,
        decision: str,
        inputs: Dict[str, float],
        context: Dict[str, Any],
        counterfactual_pnl: Optional[float]
    ) -> GateExplanation:
        """Explain meta-label gate decision"""
        probability = inputs.get('probability', 0)
        threshold = context.get('threshold', 0.5)
        mode = context.get('mode', 'unknown')

        margin = probability - threshold
        margin_pct = (margin / threshold) * 100 if threshold > 0 else 0

        if decision == "FAIL":
            reason = f"Predicted win probability {probability:.1%} is below threshold {threshold:.1%}"

            # What would need to change
            required_increase = abs(margin)
            would_pass_if = [
                f"Win probability increases by {required_increase:.1%} (from {probability:.1%} to {threshold:.1%})",
                "Model confidence improves through retraining",
                "Market conditions become more favorable"
            ]

            what_to_change = f"Need {required_increase:.1%} higher predicted win rate"

        else:  # PASS
            reason = f"Predicted win probability {probability:.1%} exceeds threshold {threshold:.1%}"
            would_pass_if = ["Already passing"]
            what_to_change = "N/A"

        # Assess if this was a good block
        is_good_block = None
        if counterfactual_pnl is not None and decision == "FAIL":
            is_good_block = counterfactual_pnl < 0  # Good block if would have lost money

        # Summary
        if decision == "FAIL":
            summary = f"❌ Meta-label REJECTED: {reason}"
            if is_good_block:
                summary += f" ✅ GOOD BLOCK (would have lost {abs(counterfactual_pnl):.1f} bps)"
            elif is_good_block is False:
                summary += f" ⚠️ BAD BLOCK (missed {counterfactual_pnl:.1f} bps)"
        else:
            summary = f"✅ Meta-label PASSED: {reason}"

        detailed = f"""
Gate: meta_label ({mode} mode)
Decision: {decision}
Reason: {reason}
Margin: {margin:.3f} ({margin_pct:+.1f}% vs threshold)

Input Values:
  - Predicted probability: {probability:.1%}
  - Threshold ({mode}): {threshold:.1%}

To Pass:
  {chr(10).join('  • ' + item for item in would_pass_if)}
"""

        if counterfactual_pnl is not None:
            detailed += f"\nCounterfactual: If taken, P&L would be {counterfactual_pnl:+.1f} bps"

        return GateExplanation(
            gate_name="meta_label",
            decision=decision,
            reason=reason,
            margin=margin,
            margin_pct=margin_pct,
            inputs=inputs,
            thresholds={"threshold": threshold},
            would_pass_if=would_pass_if,
            what_to_change=what_to_change,
            mode=mode,
            is_good_block=is_good_block,
            counterfactual_pnl=counterfactual_pnl,
            summary=summary,
            detailed=detailed
        )

    def _explain_cost_gate(
        self,
        decision: str,
        inputs: Dict[str, float],
        context: Dict[str, Any],
        counterfactual_pnl: Optional[float]
    ) -> GateExplanation:
        """Explain cost gate decision"""
        expected_return = inputs.get('expected_return_bps', 0)
        total_cost = inputs.get('total_cost_bps', 0)
        buffer = context.get('buffer_bps', 3.0)
        mode = context.get('mode', 'unknown')

        required_return = total_cost + buffer
        margin = expected_return - required_return
        margin_pct = (margin / required_return) * 100 if required_return > 0 else 0

        if decision == "FAIL":
            reason = f"Expected return {expected_return:.1f} bps < costs {total_cost:.1f} bps + buffer {buffer:.1f} bps"

            shortfall = abs(margin)
            would_pass_if = [
                f"Expected return increases by {shortfall:.1f} bps (to {required_return:.1f} bps)",
                f"Trading costs decrease by {shortfall:.1f} bps",
                "Spreads tighten",
                "Signal strength improves"
            ]

            what_to_change = f"Need {shortfall:.1f} bps higher expected return or {shortfall:.1f} bps lower costs"

        else:  # PASS
            reason = f"Expected return {expected_return:.1f} bps > costs {total_cost:.1f} bps + buffer {buffer:.1f} bps"
            would_pass_if = ["Already passing"]
            what_to_change = "N/A"

        is_good_block = None
        if counterfactual_pnl is not None and decision == "FAIL":
            is_good_block = counterfactual_pnl < 0

        if decision == "FAIL":
            summary = f"❌ Cost gate REJECTED: {reason}"
            if is_good_block:
                summary += f" ✅ GOOD BLOCK"
            elif is_good_block is False:
                summary += f" ⚠️ BAD BLOCK"
        else:
            summary = f"✅ Cost gate PASSED: {reason}"

        detailed = f"""
Gate: cost_gate ({mode} mode)
Decision: {decision}
Reason: {reason}
Margin: {margin:.1f} bps ({margin_pct:+.1f}%)

Cost Breakdown:
  - Expected return: {expected_return:.1f} bps
  - Total costs: {total_cost:.1f} bps
  - Buffer required: {buffer:.1f} bps
  - Minimum needed: {required_return:.1f} bps

To Pass:
  {chr(10).join('  • ' + item for item in would_pass_if)}
"""

        return GateExplanation(
            gate_name="cost_gate",
            decision=decision,
            reason=reason,
            margin=margin,
            margin_pct=margin_pct,
            inputs=inputs,
            thresholds={"buffer_bps": buffer, "required_return": required_return},
            would_pass_if=would_pass_if,
            what_to_change=what_to_change,
            mode=mode,
            is_good_block=is_good_block,
            counterfactual_pnl=counterfactual_pnl,
            summary=summary,
            detailed=detailed
        )

    def _explain_confidence_gate(
        self,
        decision: str,
        inputs: Dict[str, float],
        context: Dict[str, Any],
        counterfactual_pnl: Optional[float]
    ) -> GateExplanation:
        """Explain confidence gate decision"""
        confidence = inputs.get('confidence', 0)
        threshold = context.get('min_confidence', 0.6)
        mode = context.get('mode', 'unknown')

        margin = confidence - threshold
        margin_pct = (margin / threshold) * 100 if threshold > 0 else 0

        if decision == "FAIL":
            reason = f"Model confidence {confidence:.1%} below minimum {threshold:.1%}"
            would_pass_if = [f"Confidence increases by {abs(margin):.1%}"]
            what_to_change = f"Need {abs(margin):.1%} higher confidence"
        else:
            reason = f"Model confidence {confidence:.1%} exceeds minimum {threshold:.1%}"
            would_pass_if = ["Already passing"]
            what_to_change = "N/A"

        is_good_block = None
        if counterfactual_pnl is not None and decision == "FAIL":
            is_good_block = counterfactual_pnl < 0

        summary = f"{'❌' if decision == 'FAIL' else '✅'} Confidence gate {decision}: {reason}"

        detailed = f"""
Gate: confidence_gate ({mode} mode)
Decision: {decision}
Reason: {reason}
Margin: {margin:.3f} ({margin_pct:+.1f}%)

Values:
  - Model confidence: {confidence:.1%}
  - Minimum required: {threshold:.1%}
"""

        return GateExplanation(
            gate_name="confidence_gate",
            decision=decision,
            reason=reason,
            margin=margin,
            margin_pct=margin_pct,
            inputs=inputs,
            thresholds={"min_confidence": threshold},
            would_pass_if=would_pass_if,
            what_to_change=what_to_change,
            mode=mode,
            is_good_block=is_good_block,
            counterfactual_pnl=counterfactual_pnl,
            summary=summary,
            detailed=detailed
        )

    def _explain_regime_gate(
        self,
        decision: str,
        inputs: Dict[str, float],
        context: Dict[str, Any],
        counterfactual_pnl: Optional[float]
    ) -> GateExplanation:
        """Explain regime gate decision"""
        current_regime = context.get('current_regime', 'UNKNOWN')
        allowed_regimes = context.get('allowed_regimes', [])
        mode = context.get('mode', 'unknown')

        if decision == "FAIL":
            reason = f"Current regime {current_regime} not in allowed list {allowed_regimes}"
            would_pass_if = [f"Market regime changes to {regime}" for regime in allowed_regimes]
            what_to_change = f"Wait for regime to become: {', '.join(allowed_regimes)}"
        else:
            reason = f"Current regime {current_regime} is allowed"
            would_pass_if = ["Already passing"]
            what_to_change = "N/A"

        is_good_block = None
        if counterfactual_pnl is not None and decision == "FAIL":
            is_good_block = counterfactual_pnl < 0

        summary = f"{'❌' if decision == 'FAIL' else '✅'} Regime gate {decision}: {reason}"

        detailed = f"""
Gate: regime_gate ({mode} mode)
Decision: {decision}
Reason: {reason}

Regime Info:
  - Current regime: {current_regime}
  - Allowed regimes: {', '.join(allowed_regimes)}
"""

        return GateExplanation(
            gate_name="regime_gate",
            decision=decision,
            reason=reason,
            margin=None,
            margin_pct=None,
            inputs=inputs,
            thresholds={"allowed_regimes": allowed_regimes},
            would_pass_if=would_pass_if,
            what_to_change=what_to_change,
            mode=mode,
            is_good_block=is_good_block,
            counterfactual_pnl=counterfactual_pnl,
            summary=summary,
            detailed=detailed
        )

    def _explain_spread_gate(
        self,
        decision: str,
        inputs: Dict[str, float],
        context: Dict[str, Any],
        counterfactual_pnl: Optional[float]
    ) -> GateExplanation:
        """Explain spread gate decision"""
        spread_bps = inputs.get('spread_bps', 0)
        max_spread = context.get('max_spread_bps', 10.0)
        mode = context.get('mode', 'unknown')

        margin = max_spread - spread_bps
        margin_pct = (margin / max_spread) * 100 if max_spread > 0 else 0

        if decision == "FAIL":
            reason = f"Spread {spread_bps:.1f} bps exceeds maximum {max_spread:.1f} bps"
            would_pass_if = [f"Spread tightens by {abs(margin):.1f} bps"]
            what_to_change = f"Wait for spread < {max_spread:.1f} bps"
        else:
            reason = f"Spread {spread_bps:.1f} bps within limit {max_spread:.1f} bps"
            would_pass_if = ["Already passing"]
            what_to_change = "N/A"

        is_good_block = None
        if counterfactual_pnl is not None and decision == "FAIL":
            is_good_block = counterfactual_pnl < 0

        summary = f"{'❌' if decision == 'FAIL' else '✅'} Spread gate {decision}: {reason}"

        detailed = f"""
Gate: spread_gate ({mode} mode)
Decision: {decision}
Reason: {reason}
Margin: {margin:.1f} bps

Spread Info:
  - Current spread: {spread_bps:.1f} bps
  - Maximum allowed: {max_spread:.1f} bps
"""

        return GateExplanation(
            gate_name="spread_gate",
            decision=decision,
            reason=reason,
            margin=margin,
            margin_pct=margin_pct,
            inputs=inputs,
            thresholds={"max_spread_bps": max_spread},
            would_pass_if=would_pass_if,
            what_to_change=what_to_change,
            mode=mode,
            is_good_block=is_good_block,
            counterfactual_pnl=counterfactual_pnl,
            summary=summary,
            detailed=detailed
        )

    def _explain_volume_gate(
        self,
        decision: str,
        inputs: Dict[str, float],
        context: Dict[str, Any],
        counterfactual_pnl: Optional[float]
    ) -> GateExplanation:
        """Explain volume gate decision"""
        volume_ratio = inputs.get('volume_ratio', 0)
        min_ratio = context.get('min_volume_ratio', 0.5)
        mode = context.get('mode', 'unknown')

        margin = volume_ratio - min_ratio
        margin_pct = (margin / min_ratio) * 100 if min_ratio > 0 else 0

        if decision == "FAIL":
            reason = f"Volume ratio {volume_ratio:.2f} below minimum {min_ratio:.2f}"
            would_pass_if = [f"Volume increases by {abs(margin):.2f}x"]
            what_to_change = f"Wait for volume > {min_ratio:.2f}x average"
        else:
            reason = f"Volume ratio {volume_ratio:.2f} exceeds minimum {min_ratio:.2f}"
            would_pass_if = ["Already passing"]
            what_to_change = "N/A"

        is_good_block = None
        if counterfactual_pnl is not None and decision == "FAIL":
            is_good_block = counterfactual_pnl < 0

        summary = f"{'❌' if decision == 'FAIL' else '✅'} Volume gate {decision}: {reason}"

        detailed = f"""
Gate: volume_gate ({mode} mode)
Decision: {decision}
Reason: {reason}
Margin: {margin:.2f}x ({margin_pct:+.1f}%)

Volume Info:
  - Current volume ratio: {volume_ratio:.2f}x average
  - Minimum required: {min_ratio:.2f}x average
"""

        return GateExplanation(
            gate_name="volume_gate",
            decision=decision,
            reason=reason,
            margin=margin,
            margin_pct=margin_pct,
            inputs=inputs,
            thresholds={"min_volume_ratio": min_ratio},
            would_pass_if=would_pass_if,
            what_to_change=what_to_change,
            mode=mode,
            is_good_block=is_good_block,
            counterfactual_pnl=counterfactual_pnl,
            summary=summary,
            detailed=detailed
        )

    def _explain_generic(
        self,
        gate_name: str,
        decision: str,
        inputs: Dict[str, float],
        context: Dict[str, Any],
        counterfactual_pnl: Optional[float]
    ) -> GateExplanation:
        """Generic explanation for unknown gate types"""
        reason = context.get('reason', f"Gate {gate_name} returned {decision}")
        mode = context.get('mode', 'unknown')

        summary = f"{'❌' if decision == 'FAIL' else '✅'} {gate_name} {decision}"

        detailed = f"""
Gate: {gate_name} ({mode} mode)
Decision: {decision}
Reason: {reason}

Inputs: {inputs}
Context: {context}
"""

        return GateExplanation(
            gate_name=gate_name,
            decision=decision,
            reason=reason,
            margin=None,
            margin_pct=None,
            inputs=inputs,
            thresholds={},
            would_pass_if=[],
            what_to_change="Unknown",
            mode=mode,
            is_good_block=None,
            counterfactual_pnl=counterfactual_pnl,
            summary=summary,
            detailed=detailed
        )


if __name__ == '__main__':
    # Example usage
    print("Gate Explainer Example")
    print("=" * 80)

    explainer = GateExplainer()

    # Example 1: Meta-label rejection
    print("\n📊 Example 1: Meta-label rejection")
    print("-" * 80)

    explanation = explainer.explain_rejection(
        gate_name="meta_label",
        decision="FAIL",
        inputs={"probability": 0.42},
        context={"threshold": 0.45, "mode": "scalp"},
        counterfactual_pnl=-8.5  # Would have lost money
    )

    print(explanation.summary)
    print(explanation.detailed)

    # Example 2: Cost gate rejection
    print("\n📊 Example 2: Cost gate rejection")
    print("-" * 80)

    explanation2 = explainer.explain_rejection(
        gate_name="cost_gate",
        decision="FAIL",
        inputs={"expected_return_bps": 10.0, "total_cost_bps": 8.5},
        context={"buffer_bps": 3.0, "mode": "scalp"},
        counterfactual_pnl=12.3  # Would have made money (bad block)
    )

    print(explanation2.summary)
    print(explanation2.detailed)

    # Example 3: Passing gate
    print("\n📊 Example 3: Confidence gate passing")
    print("-" * 80)

    explanation3 = explainer.explain_rejection(
        gate_name="confidence_gate",
        decision="PASS",
        inputs={"confidence": 0.82},
        context={"min_confidence": 0.55, "mode": "scalp"}
    )

    print(explanation3.summary)

    print("\n✓ Gate explainer ready!")
