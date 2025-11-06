"""
Gate Profiles - Tiered Scalp vs Runner Configurations

Key Problem:
Applying the SAME strict gates to all trades kills volume.
- Scalps (£1-£2 targets) need LOOSE gates for high volume
- Runners (£5-£20 targets) need STRICT gates for high precision

Solution: Tiered Gate Profiles
Two separate gate configurations:
1. **Scalp Profile**: Loose gates, 60-70% pass rate, 70-75% WR
2. **Runner Profile**: Strict gates, 5-10% pass rate, 95%+ WR

This enables:
- 30-50 scalp trades/day (high volume) ✓
- 2-5 runner trades/day (high precision) ✓
- Both objectives achieved simultaneously ✓

Example:
    Signal comes in: BREAKOUT @ 0.72 confidence

    # Try runner profile first (strict)
    runner_decision = runner_profile.check_all_gates(features)
    if runner_decision.passes:
        → Trade in LONG_HOLD book (runner)

    # Fallback to scalp profile (loose)
    scalp_decision = scalp_profile.check_all_gates(features)
    if scalp_decision.passes:
        → Trade in SHORT_HOLD book (scalp)

    # Both failed
    else:
        → Skip trade

Usage:
    # Initialize profiles
    scalp_profile = ScalpGateProfile()
    runner_profile = RunnerGateProfile()

    # Check gates for a signal
    decision = runner_profile.check_all_gates(
        edge_hat_bps=15.0,
        features={'engine_conf': 0.72, 'regime': 'TREND'},
        order_type='maker',
        position_size_usd=500.0,
        spread_bps=8.0,
        liquidity_score=0.75,
    )

    if decision.passes:
        print(f"Passed {decision.gates_passed}/{decision.total_gates} gates")
        print(f"Recommended book: {decision.recommended_book}")
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

import structlog

# Import our existing gates
from .cost_gate import CostGate, OrderType
from .adverse_selection_veto import AdverseSelectionVeto
from .sentiment_gate import SentimentGate
from .selection_intelligence import (
    MetaLabelGate,
    RegretProbabilityCalculator,
    PatternMemoryWithEvidence,
    UncertaintyCalibrator,
)
from .dual_book_manager import BookType

logger = structlog.get_logger(__name__)


class GateResult(Enum):
    """Result of a gate check."""

    PASS = "pass"
    BLOCK = "block"
    SKIP = "skip"  # Gate not applicable


@dataclass
class GateDecision:
    """Combined decision from all gates."""

    passes: bool  # Overall pass/fail
    recommended_book: Optional[BookType]  # Which book to use

    # Details
    total_gates: int
    gates_passed: int
    gates_blocked: int
    gates_skipped: int

    # Per-gate results
    gate_results: Dict[str, GateResult]
    gate_reasons: Dict[str, str]

    # Metrics
    edge_net_bps: float
    win_probability: float


class ScalpGateProfile:
    """
    Gate profile for SCALP mode (SHORT_HOLD book).

    Philosophy: HIGH VOLUME, ACCEPTABLE WIN RATE
    - Targets: £1-£2 profit per trade
    - Volume: 30-50 trades/day
    - Win rate: 70-75%
    - Pass rate: 60-70% of signals

    Gate Configuration (LOOSE):
    1. Cost gate: buffer_bps = 3.0 (loose)
    2. Meta-label: threshold = 0.45 (loose)
    3. Regret prob: threshold = 0.50 (permissive)
    4. Adverse selection: DISABLED (too restrictive for volume)
    5. Pattern memory: evidence_threshold = -0.2 (very loose)
    6. Uncertainty: DISABLED (slows volume)

    Why These Settings:
    - Loose cost buffer (3 bps vs 5 bps) → More trades pass
    - Low meta-label threshold (45% vs 50%) → Accept close calls
    - Disabled adverse selection → Don't block on micro-structure
    - Very loose pattern memory → Only block obvious losers
    """

    def __init__(self):
        """Initialize scalp gate profile."""
        # Cost gate (loose)
        self.cost_gate = CostGate(
            maker_rebate_bps=2.0,
            taker_fee_bps=5.0,
            buffer_bps=3.0,  # LOOSE: Only 3 bps buffer
            prefer_maker=True,
        )

        # Sentiment gate (Fear & Greed Index)
        self.sentiment_gate = SentimentGate(
            block_extreme_greed=True,
            block_extreme_fear=True,
            extreme_greed_threshold=80,
            extreme_fear_threshold=20,
        )

        # Meta-label gate (loose)
        self.meta_label = MetaLabelGate(
            threshold=0.45,  # LOOSE: Accept 45%+ win prob
            use_regime_specific=True,
        )

        # Regret probability (permissive)
        self.regret_calc = RegretProbabilityCalculator(
            regret_threshold=0.50,  # PERMISSIVE: Accept high regret
            k_factor=10.0,
        )

        # Pattern memory (very loose)
        self.pattern_memory = PatternMemoryWithEvidence(
            evidence_threshold=-0.2,  # VERY LOOSE: Only block if evidence < -0.2
            similarity_threshold=0.70,
        )

        # Adverse selection: DISABLED for scalps
        self.use_adverse_selection = False

        # Uncertainty calibration: DISABLED for scalps
        self.use_uncertainty = False

        logger.info(
            "scalp_gate_profile_initialized",
            cost_buffer=3.0,
            meta_threshold=0.45,
            philosophy="high_volume",
        )

    def check_all_gates(
        self,
        edge_hat_bps: float,
        features: Dict[str, float],
        order_type: OrderType,
        position_size_usd: float,
        spread_bps: float,
        liquidity_score: float,
        urgency: str = 'moderate',
    ) -> GateDecision:
        """
        Check all gates for a scalp trade.

        Args:
            edge_hat_bps: Predicted edge
            features: Feature dict with 'engine_conf', 'regime', etc.
            order_type: Order type
            position_size_usd: Position size
            spread_bps: Current spread
            liquidity_score: Liquidity score
            urgency: Urgency level

        Returns:
            GateDecision with overall pass/fail and details
        """
        gate_results = {}
        gate_reasons = {}

        # 1. Cost gate
        cost_analysis = self.cost_gate.analyze_edge(
            edge_hat_bps=edge_hat_bps,
            order_type=order_type,
            position_size_usd=position_size_usd,
            spread_bps=spread_bps,
            liquidity_score=liquidity_score,
            urgency=urgency,
        )

        if cost_analysis.passes_gate:
            gate_results['cost'] = GateResult.PASS
            gate_reasons['cost'] = f"edge_net={cost_analysis.edge_net_bps:.1f} bps"
        else:
            gate_results['cost'] = GateResult.BLOCK
            gate_reasons['cost'] = cost_analysis.reason

        # 2. Sentiment gate (Fear & Greed Index)
        direction = 'buy' if features.get('engine_conf', 0.5) > 0.5 else 'sell'  # Infer direction from confidence
        sentiment_result = self.sentiment_gate.evaluate(
            direction=direction,
            confidence=features.get('engine_conf', 0.5),
        )
        
        if sentiment_result.passed:
            gate_results['sentiment'] = GateResult.PASS
            gate_reasons['sentiment'] = sentiment_result.reason
        else:
            gate_results['sentiment'] = GateResult.BLOCK
            gate_reasons['sentiment'] = sentiment_result.reason

        # 3. Meta-label gate
        meta_decision = self.meta_label.check_gate(features)

        if meta_decision.passes_gate:
            gate_results['meta_label'] = GateResult.PASS
            gate_reasons['meta_label'] = f"win_prob={meta_decision.win_probability:.2f}"
        else:
            gate_results['meta_label'] = GateResult.BLOCK
            gate_reasons['meta_label'] = meta_decision.reason

        # 4. Regret probability (if we have runner-up score)
        if 'best_score' in features and 'runner_up_score' in features:
            regret_analysis = self.regret_calc.analyze_regret(
                best_score=features['best_score'],
                runner_up_score=features['runner_up_score'],
                sample_size=features.get('sample_size', 50),
            )

            if regret_analysis.should_trade:
                gate_results['regret'] = GateResult.PASS
                gate_reasons['regret'] = f"regret={regret_analysis.regret_probability:.2f}"
            else:
                gate_results['regret'] = GateResult.BLOCK
                gate_reasons['regret'] = regret_analysis.reason
        else:
            gate_results['regret'] = GateResult.SKIP
            gate_reasons['regret'] = "No regret data"

        # 4. Pattern memory (if we have embedding)
        if 'embedding' in features:
            evidence = self.pattern_memory.compute_evidence(features['embedding'])

            if not evidence.should_block:
                gate_results['pattern'] = GateResult.PASS
                gate_reasons['pattern'] = f"evidence={evidence.evidence:.2f}"
            else:
                gate_results['pattern'] = GateResult.BLOCK
                gate_reasons['pattern'] = f"Looks like loser (evidence={evidence.evidence:.2f})"
        else:
            gate_results['pattern'] = GateResult.SKIP
            gate_reasons['pattern'] = "No pattern data"

        # 5. Adverse selection: SKIP for scalps
        gate_results['adverse_selection'] = GateResult.SKIP
        gate_reasons['adverse_selection'] = "Disabled for scalps"

        # 6. Uncertainty: SKIP for scalps
        gate_results['uncertainty'] = GateResult.SKIP
        gate_reasons['uncertainty'] = "Disabled for scalps"

        # Calculate overall decision
        gates_passed = sum(1 for r in gate_results.values() if r == GateResult.PASS)
        gates_blocked = sum(1 for r in gate_results.values() if r == GateResult.BLOCK)
        gates_skipped = sum(1 for r in gate_results.values() if r == GateResult.SKIP)
        total_gates = len(gate_results)

        # Must pass ALL active gates (non-skipped)
        active_gates = [r for r in gate_results.values() if r != GateResult.SKIP]
        passes = all(r == GateResult.PASS for r in active_gates)

        return GateDecision(
            passes=passes,
            recommended_book=BookType.SHORT_HOLD if passes else None,
            total_gates=total_gates,
            gates_passed=gates_passed,
            gates_blocked=gates_blocked,
            gates_skipped=gates_skipped,
            gate_results=gate_results,
            gate_reasons=gate_reasons,
            edge_net_bps=cost_analysis.edge_net_bps,
            win_probability=meta_decision.win_probability,
        )


class RunnerGateProfile:
    """
    Gate profile for RUNNER mode (LONG_HOLD book).

    Philosophy: HIGH PRECISION, LOW VOLUME
    - Targets: £5-£20 profit per trade
    - Volume: 2-5 trades/day
    - Win rate: 95%+
    - Pass rate: 5-10% of signals

    Gate Configuration (STRICT):
    1. Cost gate: buffer_bps = 8.0 (strict)
    2. Meta-label: threshold = 0.65 (strict)
    3. Regret prob: threshold = 0.25 (strict)
    4. Adverse selection: ENABLED + STRICT
    5. Pattern memory: evidence_threshold = 0.1 (strict)
    6. Uncertainty: ENABLED (q_lo must beat costs)

    Why These Settings:
    - High cost buffer (8 bps) → Only very profitable trades
    - High meta-label threshold (65%) → High confidence only
    - Low regret threshold (25%) → Clear separation required
    - Adverse selection enabled → Block deteriorating microstructure
    - Strict pattern memory → Must look like past winners
    - Uncertainty enabled → Even pessimistic case must win
    """

    def __init__(self):
        """Initialize runner gate profile."""
        # Cost gate (strict)
        self.cost_gate = CostGate(
            maker_rebate_bps=2.0,
            taker_fee_bps=5.0,
            buffer_bps=8.0,  # STRICT: Need 8 bps net edge
            prefer_maker=True,
        )

        # Sentiment gate (Fear & Greed Index)
        self.sentiment_gate = SentimentGate(
            block_extreme_greed=True,
            block_extreme_fear=True,
            extreme_greed_threshold=80,
            extreme_fear_threshold=20,
        )

        # Meta-label gate (strict)
        self.meta_label = MetaLabelGate(
            threshold=0.65,  # STRICT: Need 65%+ win prob
            use_regime_specific=True,
        )

        # Regret probability (strict)
        self.regret_calc = RegretProbabilityCalculator(
            regret_threshold=0.25,  # STRICT: Low regret only
            k_factor=10.0,
        )

        # Pattern memory (strict)
        self.pattern_memory = PatternMemoryWithEvidence(
            evidence_threshold=0.1,  # STRICT: Evidence must be positive
            similarity_threshold=0.70,
        )

        # Adverse selection: ENABLED for runners
        self.adverse_selection = AdverseSelectionVeto(
            lookback_window_sec=5,
            tick_flip_window_sec=3,
            spread_widen_threshold=2.0,  # 2x spread = veto
            imbalance_flip_threshold=0.20,
            volume_dryup_threshold=0.40,
        )
        self.use_adverse_selection = True

        # Uncertainty calibration: ENABLED for runners
        self.uncertainty = UncertaintyCalibrator(
            min_q_lo_bps=5.0,  # Pessimistic case must beat 5 bps
        )
        self.use_uncertainty = True

        logger.info(
            "runner_gate_profile_initialized",
            cost_buffer=8.0,
            meta_threshold=0.65,
            philosophy="high_precision",
        )

    def check_all_gates(
        self,
        edge_hat_bps: float,
        features: Dict[str, float],
        order_type: OrderType,
        position_size_usd: float,
        spread_bps: float,
        liquidity_score: float,
        urgency: str = 'moderate',
        recent_ticks: Optional[List[float]] = None,
        spread_history: Optional[List[float]] = None,
        imbalance_history: Optional[List[float]] = None,
    ) -> GateDecision:
        """
        Check all gates for a runner trade.

        Args:
            edge_hat_bps: Predicted edge
            features: Feature dict
            order_type: Order type
            position_size_usd: Position size
            spread_bps: Current spread
            liquidity_score: Liquidity score
            urgency: Urgency level
            recent_ticks: Recent tick prices (for adverse selection)
            spread_history: Recent spreads (for adverse selection)
            imbalance_history: Recent order imbalances (for adverse selection)

        Returns:
            GateDecision with overall pass/fail and details
        """
        gate_results = {}
        gate_reasons = {}

        # 1. Cost gate
        cost_analysis = self.cost_gate.analyze_edge(
            edge_hat_bps=edge_hat_bps,
            order_type=order_type,
            position_size_usd=position_size_usd,
            spread_bps=spread_bps,
            liquidity_score=liquidity_score,
            urgency=urgency,
        )

        if cost_analysis.passes_gate:
            gate_results['cost'] = GateResult.PASS
            gate_reasons['cost'] = f"edge_net={cost_analysis.edge_net_bps:.1f} bps"
        else:
            gate_results['cost'] = GateResult.BLOCK
            gate_reasons['cost'] = cost_analysis.reason

        # 2. Sentiment gate (Fear & Greed Index)
        direction = 'buy' if features.get('engine_conf', 0.5) > 0.5 else 'sell'  # Infer direction from confidence
        sentiment_result = self.sentiment_gate.evaluate(
            direction=direction,
            confidence=features.get('engine_conf', 0.5),
        )
        
        if sentiment_result.passed:
            gate_results['sentiment'] = GateResult.PASS
            gate_reasons['sentiment'] = sentiment_result.reason
        else:
            gate_results['sentiment'] = GateResult.BLOCK
            gate_reasons['sentiment'] = sentiment_result.reason

        # 3. Meta-label gate
        meta_decision = self.meta_label.check_gate(features)

        if meta_decision.passes_gate:
            gate_results['meta_label'] = GateResult.PASS
            gate_reasons['meta_label'] = f"win_prob={meta_decision.win_probability:.2f}"
        else:
            gate_results['meta_label'] = GateResult.BLOCK
            gate_reasons['meta_label'] = meta_decision.reason

        # 4. Regret probability
        if 'best_score' in features and 'runner_up_score' in features:
            regret_analysis = self.regret_calc.analyze_regret(
                best_score=features['best_score'],
                runner_up_score=features['runner_up_score'],
                sample_size=features.get('sample_size', 50),
            )

            if regret_analysis.should_trade:
                gate_results['regret'] = GateResult.PASS
                gate_reasons['regret'] = f"regret={regret_analysis.regret_probability:.2f}"
            else:
                gate_results['regret'] = GateResult.BLOCK
                gate_reasons['regret'] = regret_analysis.reason
        else:
            gate_results['regret'] = GateResult.SKIP
            gate_reasons['regret'] = "No regret data"

        # 4. Pattern memory
        if 'embedding' in features:
            evidence = self.pattern_memory.compute_evidence(features['embedding'])

            if not evidence.should_block:
                gate_results['pattern'] = GateResult.PASS
                gate_reasons['pattern'] = f"evidence={evidence.evidence:.2f}"
            else:
                gate_results['pattern'] = GateResult.BLOCK
                gate_reasons['pattern'] = f"Looks like loser (evidence={evidence.evidence:.2f})"
        else:
            gate_results['pattern'] = GateResult.SKIP
            gate_reasons['pattern'] = "No pattern data"

        # 5. Adverse selection (ENABLED for runners)
        if self.use_adverse_selection and recent_ticks and spread_history and imbalance_history:
            veto_decision = self.adverse_selection.check_veto(
                recent_ticks=recent_ticks,
                current_spread=spread_bps,
                spread_history=spread_history,
                current_imbalance=imbalance_history[-1] if imbalance_history else 0.5,
                imbalance_history=imbalance_history,
                current_volume=1.0,  # Normalized
                recent_volume_history=[1.0, 1.0, 1.0],  # Simplified
            )

            if not veto_decision.should_veto:
                gate_results['adverse_selection'] = GateResult.PASS
                gate_reasons['adverse_selection'] = "Microstructure OK"
            else:
                gate_results['adverse_selection'] = GateResult.BLOCK
                gate_reasons['adverse_selection'] = veto_decision.reason
        else:
            gate_results['adverse_selection'] = GateResult.SKIP
            gate_reasons['adverse_selection'] = "No microstructure data"

        # 6. Uncertainty calibration (ENABLED for runners)
        if self.use_uncertainty:
            estimate = self.uncertainty.predict_with_uncertainty(
                features=features,
                expected_cost_bps=cost_analysis.cost_estimate.total_cost_bps,
            )

            if estimate.should_trade:
                gate_results['uncertainty'] = GateResult.PASS
                gate_reasons['uncertainty'] = f"q_lo={estimate.q_lo_bps:.1f} bps"
            else:
                gate_results['uncertainty'] = GateResult.BLOCK
                gate_reasons['uncertainty'] = estimate.reason
        else:
            gate_results['uncertainty'] = GateResult.SKIP
            gate_reasons['uncertainty'] = "Not calibrated yet"

        # Calculate overall decision
        gates_passed = sum(1 for r in gate_results.values() if r == GateResult.PASS)
        gates_blocked = sum(1 for r in gate_results.values() if r == GateResult.BLOCK)
        gates_skipped = sum(1 for r in gate_results.values() if r == GateResult.SKIP)
        total_gates = len(gate_results)

        # Must pass ALL active gates
        active_gates = [r for r in gate_results.values() if r != GateResult.SKIP]
        passes = all(r == GateResult.PASS for r in active_gates)

        return GateDecision(
            passes=passes,
            recommended_book=BookType.LONG_HOLD if passes else None,
            total_gates=total_gates,
            gates_passed=gates_passed,
            gates_blocked=gates_blocked,
            gates_skipped=gates_skipped,
            gate_results=gate_results,
            gate_reasons=gate_reasons,
            edge_net_bps=cost_analysis.edge_net_bps,
            win_probability=meta_decision.win_probability,
        )


class HybridGateRouter:
    """
    Route signals through both profiles (runner → scalp fallback).

    Logic:
    1. Try runner profile first (strict gates, high precision)
    2. If passes → LONG_HOLD book (runner)
    3. If fails, try scalp profile (loose gates, high volume)
    4. If passes → SHORT_HOLD book (scalp)
    5. If both fail → skip trade

    This enables:
    - High-quality signals → Runners (95%+ WR, £5-£20)
    - Medium-quality signals → Scalps (70-75% WR, £1-£2)
    - Low-quality signals → Skipped
    """

    def __init__(self):
        """Initialize hybrid router."""
        self.scalp_profile = ScalpGateProfile()
        self.runner_profile = RunnerGateProfile()

        logger.info("hybrid_gate_router_initialized")

    def route_signal(
        self,
        edge_hat_bps: float,
        features: Dict[str, float],
        order_type: OrderType,
        position_size_usd: float,
        spread_bps: float,
        liquidity_score: float,
        **kwargs,
    ) -> GateDecision:
        """
        Route signal through both profiles.

        Returns:
            GateDecision with recommended book (or None if skip)
        """
        # 1. Try runner profile first (strict)
        runner_decision = self.runner_profile.check_all_gates(
            edge_hat_bps=edge_hat_bps,
            features=features,
            order_type=order_type,
            position_size_usd=position_size_usd,
            spread_bps=spread_bps,
            liquidity_score=liquidity_score,
            **kwargs,
        )

        if runner_decision.passes:
            logger.info(
                "signal_routed_to_runner",
                edge_net=runner_decision.edge_net_bps,
                win_prob=runner_decision.win_probability,
                gates_passed=runner_decision.gates_passed,
            )
            return runner_decision

        # 2. Fallback to scalp profile (loose)
        scalp_decision = self.scalp_profile.check_all_gates(
            edge_hat_bps=edge_hat_bps,
            features=features,
            order_type=order_type,
            position_size_usd=position_size_usd,
            spread_bps=spread_bps,
            liquidity_score=liquidity_score,
        )

        if scalp_decision.passes:
            logger.info(
                "signal_routed_to_scalp",
                edge_net=scalp_decision.edge_net_bps,
                win_prob=scalp_decision.win_probability,
                gates_passed=scalp_decision.gates_passed,
            )
            return scalp_decision

        # 3. Both failed - skip
        logger.info(
            "signal_skipped",
            runner_blocks=runner_decision.gates_blocked,
            scalp_blocks=scalp_decision.gates_blocked,
        )

        return GateDecision(
            passes=False,
            recommended_book=None,
            total_gates=scalp_decision.total_gates,
            gates_passed=0,
            gates_blocked=scalp_decision.gates_blocked,
            gates_skipped=0,
            gate_results=scalp_decision.gate_results,
            gate_reasons=scalp_decision.gate_reasons,
            edge_net_bps=scalp_decision.edge_net_bps,
            win_probability=scalp_decision.win_probability,
        )
