"""
Execution Intelligence Systems

Bundled execution-focused improvements:
1. Fill Probability & Time-to-Fill
2. Setup-Trigger Gate
3. Scratch Policy
4. Scalp EPS Ranking
5. Scalp-to-Runner Unlock

These systems optimize trade execution, entry timing, and position management.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple
import time

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


# ============================================================================
# 1. FILL PROBABILITY & TIME-TO-FILL
# ============================================================================


@dataclass
class FillProbabilityEstimate:
    """Fill probability estimation."""

    fill_prob_1s: float  # Probability of fill within 1 second
    fill_prob_3s: float  # Probability of fill within 3 seconds
    fill_prob_10s: float  # Probability of fill within 10 seconds
    expected_time_to_fill_sec: float  # Expected time to fill
    confidence: float  # Confidence in estimate (0-1)

    # Supporting data
    queue_position: int  # Position in order queue
    queue_depth: float  # Total depth at price level
    depletion_rate: float  # Queue depletion rate (orders/sec)


class FillProbabilityCalculator:
    """
    Estimate fill probability and time-to-fill for limit orders.

    Uses order book depth and queue depletion rates.

    Key Insight:
    - Maker orders don't always fill (60-80% typically)
    - Time-to-fill varies based on queue position
    - Fast scalps need high fill_prob_1s
    - Patient orders tolerate longer waits

    Usage:
        calc = FillProbabilityCalculator()

        estimate = calc.estimate_fill_probability(
            our_price=47000.0,
            our_size=1.0,
            order_book_depth_at_price=50.0,
            recent_fill_rate=10.0,  # 10 BTC/sec
            spread_bps=5.0,
        )

        # For scalp (need quick fill)
        if estimate.fill_prob_1s < 0.70:
            logger.warning("Low 1s fill prob, use market order instead")
            use_market_order()

        # Check expected wait
        if estimate.expected_time_to_fill_sec > 10.0:
            logger.warning("Expected wait > 10s, may be stale")
            skip_trade()
    """

    def __init__(
        self,
        base_fill_rate: float = 0.80,  # 80% of limit orders fill eventually
        fast_fill_threshold_sec: float = 3.0,
    ):
        self.base_fill_rate = base_fill_rate
        self.fast_threshold = fast_fill_threshold_sec

    def estimate_fill_probability(
        self,
        our_price: float,
        our_size: float,
        order_book_depth_at_price: float,
        recent_fill_rate: float,  # Orders per second
        spread_bps: float,
    ) -> FillProbabilityEstimate:
        """Estimate fill probability and time."""

        # Queue position (assume we're at back)
        queue_position = order_book_depth_at_price

        # Time to clear queue ahead of us
        if recent_fill_rate > 0:
            time_to_clear = queue_position / recent_fill_rate
        else:
            time_to_clear = 999.0  # Unknown

        # Fill probabilities at different horizons
        # Exponential decay based on time
        lambda_param = 1.0 / max(time_to_clear, 0.1)

        fill_prob_1s = min(1.0 - np.exp(-lambda_param * 1.0), self.base_fill_rate)
        fill_prob_3s = min(1.0 - np.exp(-lambda_param * 3.0), self.base_fill_rate)
        fill_prob_10s = min(1.0 - np.exp(-lambda_param * 10.0), self.base_fill_rate)

        # Expected time to fill
        expected_time = time_to_clear if time_to_clear < 999 else 30.0

        # Confidence based on spread (tight spread = more confidence)
        if spread_bps < 5:
            confidence = 0.90
        elif spread_bps < 10:
            confidence = 0.75
        else:
            confidence = 0.60

        return FillProbabilityEstimate(
            fill_prob_1s=fill_prob_1s,
            fill_prob_3s=fill_prob_3s,
            fill_prob_10s=fill_prob_10s,
            expected_time_to_fill_sec=expected_time,
            confidence=confidence,
            queue_position=int(queue_position),
            queue_depth=order_book_depth_at_price,
            depletion_rate=recent_fill_rate,
        )

    def should_use_market_order(
        self,
        fill_estimate: FillProbabilityEstimate,
        trade_type: str,  # 'scalp', 'swing', 'runner'
    ) -> bool:
        """Determine if should use market order instead of limit."""

        if trade_type == 'scalp':
            # Scalps need fast fills
            return fill_estimate.fill_prob_1s < 0.70
        elif trade_type == 'swing':
            # Swings can wait longer
            return fill_estimate.fill_prob_3s < 0.60
        else:
            # Runners can be patient
            return fill_estimate.fill_prob_10s < 0.50


# ============================================================================
# 2. SETUP-TRIGGER GATE
# ============================================================================


@dataclass
class SetupTriggerState:
    """Setup-trigger tracking state."""

    has_setup: bool
    setup_timestamp: Optional[float]
    setup_features: Dict[str, float]

    has_trigger: bool
    trigger_timestamp: Optional[float]
    trigger_features: Dict[str, float]

    is_valid: bool  # Setup + Trigger within window
    time_since_setup: Optional[float]


class SetupTriggerGate:
    """
    Require both "setup" (compression) and "trigger" (ignition) within N seconds.

    Example:
        BREAKOUT setup: Bollinger Bands squeeze, ADX rising
        BREAKOUT trigger: Price breaks band, volume surge

        Required: Both setup AND trigger within 10 seconds
        Expired: Setup 20 seconds ago, trigger now → REJECT

    Usage:
        gate = SetupTriggerGate(window_sec=10)

        # Detect setup (compression)
        if bb_squeeze and adx_rising:
            gate.mark_setup({'bb_width': 0.02, 'adx': 35})

        # Later, detect trigger (ignition)
        if price_breaks_band and volume_surge:
            gate.mark_trigger({'volume_ratio': 2.5, 'breakout_bps': 15})

        # Check validity
        state = gate.get_state()
        if not state.is_valid:
            logger.info("Setup expired, skipping trigger")
            return None
    """

    def __init__(
        self,
        window_sec: float = 10.0,
        require_setup_first: bool = True,
    ):
        self.window = window_sec
        self.require_setup_first = require_setup_first

        self.setup_time: Optional[float] = None
        self.setup_features: Dict[str, float] = {}

        self.trigger_time: Optional[float] = None
        self.trigger_features: Dict[str, float] = {}

    def mark_setup(self, features: Dict[str, float]) -> None:
        """Mark setup detected."""
        self.setup_time = time.time()
        self.setup_features = features.copy()
        logger.debug("setup_detected", features=features)

    def mark_trigger(self, features: Dict[str, float]) -> None:
        """Mark trigger detected."""
        self.trigger_time = time.time()
        self.trigger_features = features.copy()
        logger.debug("trigger_detected", features=features)

    def get_state(self) -> SetupTriggerState:
        """Get current setup-trigger state."""

        now = time.time()

        has_setup = self.setup_time is not None
        has_trigger = self.trigger_time is not None

        if has_setup:
            time_since_setup = now - self.setup_time
        else:
            time_since_setup = None

        # Check validity
        is_valid = False
        if has_setup and has_trigger:
            # Both exist - check timing
            if self.require_setup_first:
                # Setup must come before trigger
                if self.setup_time < self.trigger_time:
                    time_between = self.trigger_time - self.setup_time
                    if time_between <= self.window:
                        is_valid = True
            else:
                # Just check both within window
                if time_since_setup <= self.window:
                    is_valid = True

        return SetupTriggerState(
            has_setup=has_setup,
            setup_timestamp=self.setup_time,
            setup_features=self.setup_features,
            has_trigger=has_trigger,
            trigger_timestamp=self.trigger_time,
            trigger_features=self.trigger_features,
            is_valid=is_valid,
            time_since_setup=time_since_setup,
        )

    def reset(self) -> None:
        """Reset state."""
        self.setup_time = None
        self.setup_features.clear()
        self.trigger_time = None
        self.trigger_features.clear()


# ============================================================================
# 3. SCRATCH POLICY
# ============================================================================


@dataclass
class ScratchDecision:
    """Scratch (immediate exit) decision."""

    should_scratch: bool
    reason: str
    expected_loss_bps: float


class ScratchPolicy:
    """
    Immediately exit if entry goes wrong (protects scalp WR).

    Scratch triggers:
    1. Entry slippage > model estimate
    2. Micro flips immediately after entry
    3. Adverse price move within 3 seconds

    Example:
        Expected entry: 47000
        Actual fill: 47015 (15 bps slippage, expected 5)
        → SCRATCH: Exit immediately, protect capital

    Usage:
        policy = ScratchPolicy(
            slippage_tolerance_bps=5.0,
            micro_flip_window_sec=3.0,
            adverse_move_threshold_bps=10.0,
        )

        # After entry
        if actual_fill_price != expected_price:
            decision = policy.check_scratch(
                expected_price=47000.0,
                actual_price=47015.0,
                entry_timestamp=entry_time,
                micro_flipped=False,
            )

            if decision.should_scratch:
                logger.warning("Scratching trade", reason=decision.reason)
                exit_immediately()
    """

    def __init__(
        self,
        slippage_tolerance_bps: float = 5.0,
        micro_flip_window_sec: float = 3.0,
        adverse_move_threshold_bps: float = 10.0,
    ):
        self.slippage_tolerance = slippage_tolerance_bps
        self.micro_window = micro_flip_window_sec
        self.adverse_threshold = adverse_move_threshold_bps

    def check_scratch(
        self,
        expected_price: float,
        actual_price: float,
        entry_timestamp: float,
        current_price: float,
        micro_flipped: bool,
        direction: str,  # 'long' or 'short'
    ) -> ScratchDecision:
        """Check if should scratch position."""

        # 1. Check entry slippage
        slippage_bps = abs(actual_price - expected_price) / expected_price * 10000

        if slippage_bps > self.slippage_tolerance:
            return ScratchDecision(
                should_scratch=True,
                reason=f"Excessive entry slippage: {slippage_bps:.1f} bps",
                expected_loss_bps=slippage_bps,
            )

        # 2. Check micro flip
        time_since_entry = time.time() - entry_timestamp
        if time_since_entry < self.micro_window and micro_flipped:
            return ScratchDecision(
                should_scratch=True,
                reason="Microstructure flipped immediately after entry",
                expected_loss_bps=5.0,
            )

        # 3. Check adverse move
        if direction == 'long':
            pnl_bps = (current_price - actual_price) / actual_price * 10000
        else:
            pnl_bps = (actual_price - current_price) / actual_price * 10000

        if pnl_bps < -self.adverse_threshold:
            return ScratchDecision(
                should_scratch=True,
                reason=f"Adverse move: {pnl_bps:.1f} bps",
                expected_loss_bps=abs(pnl_bps),
            )

        return ScratchDecision(
            should_scratch=False,
            reason="Entry clean",
            expected_loss_bps=0.0,
        )


# ============================================================================
# 4. SCALP EPS RANKING
# ============================================================================


@dataclass
class ScalpRanking:
    """Scalp ranking by edge-per-second."""

    edge_net_bps: float
    expected_time_to_exit_sec: float
    eps: float  # Edge per second
    rank_score: float  # Normalized score (0-1)


class ScalpEPSRanker:
    """
    Rank scalps by edge-per-second (EPS).

    Formula: EPS = edge_net_bps / expected_time_to_exit_sec

    Prioritize quick, profitable scalps over slow ones.

    Example:
        Scalp A: +10 bps in 5 seconds → EPS = 2.0
        Scalp B: +15 bps in 15 seconds → EPS = 1.0
        → Pick Scalp A (higher EPS)

    Usage:
        ranker = ScalpEPSRanker()

        candidates = [
            {'edge': 10, 'time': 5, 'id': 'A'},
            {'edge': 15, 'time': 15, 'id': 'B'},
            {'edge': 8, 'time': 3, 'id': 'C'},
        ]

        ranked = ranker.rank_scalps(candidates)
        best = ranked[0]  # Highest EPS
    """

    def rank_scalps(
        self,
        scalp_candidates: List[Dict],
    ) -> List[Tuple[Dict, ScalpRanking]]:
        """
        Rank scalp candidates by EPS.

        Args:
            scalp_candidates: List of dicts with 'edge_net_bps' and 'expected_time_sec'

        Returns:
            Sorted list of (candidate, ranking) tuples
        """
        rankings = []

        for candidate in scalp_candidates:
            edge = candidate.get('edge_net_bps', 0)
            time_sec = candidate.get('expected_time_to_exit_sec', 10)

            # Calculate EPS
            eps = edge / max(time_sec, 0.1)

            ranking = ScalpRanking(
                edge_net_bps=edge,
                expected_time_to_exit_sec=time_sec,
                eps=eps,
                rank_score=0.0,  # Will be normalized
            )

            rankings.append((candidate, ranking))

        # Sort by EPS (descending)
        rankings.sort(key=lambda x: x[1].eps, reverse=True)

        # Normalize scores
        if rankings:
            max_eps = rankings[0][1].eps
            for _, ranking in rankings:
                ranking.rank_score = ranking.eps / max_eps if max_eps > 0 else 0.0

        return rankings


# ============================================================================
# 5. SCALP-TO-RUNNER UNLOCK
# ============================================================================


@dataclass
class RunnerUnlockDecision:
    """Decision on unlocking runner."""

    should_unlock: bool
    confidence: float
    supporting_evidence: List[str]
    reason: str


class ScalpToRunnerUnlock:
    """
    Keep runner only when post-entry evidence strengthens.

    Checks:
    - ADX increasing (trend strengthening)
    - Ignition/momentum increasing
    - Microstructure improving
    - Continuation pattern memory > 0

    Usage:
        unlocker = ScalpToRunnerUnlock()

        # After scalp TP hit, check if should keep runner
        decision = unlocker.should_unlock_runner(
            entry_adx=25,
            current_adx=32,  # Strengthening!
            entry_momentum=0.5,
            current_momentum=0.7,
            micro_improving=True,
            continuation_memory_score=0.65,
        )

        if decision.should_unlock:
            logger.info("Unlocking runner", evidence=decision.supporting_evidence)
            keep_runner()
        else:
            logger.info("Exiting runner", reason=decision.reason)
            exit_all()
    """

    def __init__(
        self,
        adx_increase_threshold: float = 5.0,
        momentum_increase_threshold: float = 0.10,
        continuation_memory_threshold: float = 0.50,
        min_evidence_count: int = 2,
    ):
        self.adx_threshold = adx_increase_threshold
        self.momentum_threshold = momentum_increase_threshold
        self.memory_threshold = continuation_memory_threshold
        self.min_evidence = min_evidence_count

    def should_unlock_runner(
        self,
        entry_adx: float,
        current_adx: float,
        entry_momentum: float,
        current_momentum: float,
        micro_improving: bool,
        continuation_memory_score: float,
    ) -> RunnerUnlockDecision:
        """Determine if should keep runner."""

        evidence = []

        # 1. ADX strengthening
        adx_delta = current_adx - entry_adx
        if adx_delta >= self.adx_threshold:
            evidence.append(f"ADX +{adx_delta:.1f} (trend strengthening)")

        # 2. Momentum increasing
        momentum_delta = current_momentum - entry_momentum
        if momentum_delta >= self.momentum_threshold:
            evidence.append(f"Momentum +{momentum_delta:.2f}")

        # 3. Microstructure improving
        if micro_improving:
            evidence.append("Microstructure improving")

        # 4. Continuation memory positive
        if continuation_memory_score >= self.memory_threshold:
            evidence.append(f"Continuation memory {continuation_memory_score:.2f}")

        # Decision
        evidence_count = len(evidence)
        should_unlock = evidence_count >= self.min_evidence
        confidence = evidence_count / 4.0  # 4 possible evidence types

        if should_unlock:
            reason = f"Strong continuation evidence ({evidence_count}/4 signals)"
        else:
            reason = f"Insufficient continuation evidence ({evidence_count}/{self.min_evidence} required)"

        return RunnerUnlockDecision(
            should_unlock=should_unlock,
            confidence=confidence,
            supporting_evidence=evidence,
            reason=reason,
        )
