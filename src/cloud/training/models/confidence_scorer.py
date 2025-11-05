"""
Confidence Scoring System for Trading Decisions.

Calculates confidence scores based on:
- Sample size (sigmoid-based: more data = more confidence)
- Score separation (best vs runner-up actions)
- Pattern similarity (match quality with historical patterns)
- Regime alignment (current regime matches historical success)

This allows the RL agent to know when to trade vs when to sit out.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class ConfidenceFactors:
    """Individual factors contributing to confidence score."""

    # Data availability
    sample_count: int  # Number of historical examples
    sample_confidence: float  # 0-1, sigmoid of sample count

    # Decision quality
    best_score: float  # Score of best action (0-1)
    runner_up_score: float  # Score of second-best action
    score_separation: float  # Difference between best and runner-up

    # Pattern matching
    pattern_similarity: float  # 0-1, how well current state matches patterns
    pattern_reliability: float  # 0-1, reliability of matched pattern

    # Regime alignment
    regime_match: bool  # Does current regime match best historical regime?
    regime_confidence: float  # 0-1, regime detection confidence
    meta_signal: float  # Additional meta-model signal (0-1)
    orderbook_bias: float  # -1 to 1, order-book imbalance signal


@dataclass
class ConfidenceResult:
    """Result of confidence calculation."""

    confidence: float  # Final confidence score (0-1)
    factors: ConfidenceFactors  # Individual factors
    decision: str  # "trade" or "skip"
    reason: str  # Human-readable explanation


class ConfidenceScorer:
    """
    Calculates confidence scores for trading decisions.

    Based on Revuelto's proven formula with enhancements:
    - Base confidence from sample size (sigmoid)
    - Score separation (clear winner)
    - Pattern similarity (good matches)
    - Regime alignment (right conditions)
    """

    def __init__(
        self,
        min_confidence_threshold: float = 0.52,
        sample_threshold: int = 20,
        strong_alignment_threshold: float = 0.7,
        use_regime_thresholds: bool = True,
        regime_thresholds: Optional[Dict[str, float]] = None,
    ):
        """
        Args:
            min_confidence_threshold: Default minimum confidence to trade
            sample_threshold: Sample count for 63% confidence (sigmoid inflection)
            strong_alignment_threshold: Threshold for "strong alignment" bonus
            use_regime_thresholds: Enable regime-specific confidence thresholds
            regime_thresholds: Custom thresholds per regime (overrides defaults)
        """
        self.min_confidence_threshold = min_confidence_threshold
        self.sample_threshold = sample_threshold
        self.strong_alignment_threshold = strong_alignment_threshold
        self.use_regime_thresholds = use_regime_thresholds

        # Regime-specific thresholds (context-aware decision making!)
        # TREND: Lower threshold (more aggressive) - strong directional moves
        # RANGE: Moderate threshold - mean reversion opportunities
        # PANIC: Higher threshold (conservative) - chaotic conditions, only high-conviction
        self.regime_thresholds = regime_thresholds or {
            "trend": 0.50,   # Most aggressive - trending markets are profitable
            "range": 0.55,   # Moderate - range-bound trading requires precision
            "panic": 0.65,   # Most conservative - high volatility = high risk
            "unknown": 0.60, # Conservative default when regime unclear
        }

        logger.info(
            "confidence_scorer_initialized",
            min_threshold=min_confidence_threshold,
            sample_threshold=sample_threshold,
            strong_alignment_threshold=strong_alignment_threshold,
            use_regime_thresholds=use_regime_thresholds,
            regime_thresholds=self.regime_thresholds,
        )

    def calculate_confidence(
        self,
        sample_count: int,
        best_score: float,
        runner_up_score: float,
        pattern_similarity: float = 0.5,
        pattern_reliability: float = 0.5,
        regime_match: bool = False,
        regime_confidence: float = 0.5,
        meta_features: Optional[Dict[str, float]] = None,
        current_regime: Optional[str] = None,
    ) -> ConfidenceResult:
        """
        Calculate confidence score for a trading decision with regime-aware thresholding.

        Args:
            sample_count: Number of historical examples
            best_score: Score of best action (0-1)
            runner_up_score: Score of second-best action (0-1)
            pattern_similarity: How well current state matches patterns (0-1)
            pattern_reliability: Reliability of matched pattern (0-1)
            regime_match: Does current regime match best historical regime?
            regime_confidence: Regime detection confidence (0-1)
            meta_features: Additional meta-model features
            current_regime: Current market regime ("trend", "range", "panic", "unknown")

        Returns:
            ConfidenceResult with confidence score and context-aware decision
        """
        # 1. Sample-based confidence (sigmoid function)
        sample_confidence = self._sigmoid_confidence(sample_count)

        # 2. Score separation (how clear is the winner?)
        score_separation = abs(best_score - runner_up_score)

        # 3. Base confidence (weighted combination)
        base_confidence = (
            0.6 * sample_confidence +
            0.4 * (0.5 + score_separation)  # Score separation centered at 0.5
        )

        # 4. Pattern matching bonus
        pattern_bonus = 0.0
        if pattern_similarity > 0.7:
            pattern_bonus += 0.05 * pattern_reliability

        # 5. Strong alignment bonus
        alignment_bonus = 0.0
        if best_score > self.strong_alignment_threshold:
            alignment_bonus += 0.1

        # 6. Regime alignment bonus
        regime_bonus = 0.0
        if regime_match and regime_confidence > 0.6:
            regime_bonus += 0.05

        meta_signal = 0.5
        orderbook_bias = 0.0
        if meta_features:
            meta_signal = float(meta_features.get("meta_signal", 0.5))
            orderbook_bias = float(meta_features.get("orderbook_bias", 0.0))

        # Meta-model contributes soft gating (70% confidence -> +0.05 boost)
        meta_bonus = (meta_signal - 0.5) * 0.1
        orderbook_bonus = orderbook_bias * 0.05

        # 7. Final confidence
        confidence = np.clip(
            base_confidence + pattern_bonus + alignment_bonus + regime_bonus + meta_bonus + orderbook_bonus,
            0.0,
            1.0,
        )

        # 8. Build factors
        factors = ConfidenceFactors(
            sample_count=sample_count,
            sample_confidence=sample_confidence,
            best_score=best_score,
            runner_up_score=runner_up_score,
            score_separation=score_separation,
            pattern_similarity=pattern_similarity,
            pattern_reliability=pattern_reliability,
            regime_match=regime_match,
            regime_confidence=regime_confidence,
            meta_signal=meta_signal,
            orderbook_bias=orderbook_bias,
        )

        # 9. Make context-aware decision using regime-specific threshold
        effective_threshold = self._get_effective_threshold(current_regime)

        if confidence >= effective_threshold:
            decision = "trade"
            reason = self._build_trade_reason(confidence, factors, current_regime, effective_threshold)
        else:
            decision = "skip"
            reason = self._build_skip_reason(confidence, factors, current_regime, effective_threshold)

        logger.debug(
            "confidence_decision",
            confidence=confidence,
            effective_threshold=effective_threshold,
            regime=current_regime,
            decision=decision,
        )

        return ConfidenceResult(
            confidence=confidence,
            factors=factors,
            decision=decision,
            reason=reason,
        )

    def _get_effective_threshold(self, current_regime: Optional[str]) -> float:
        """
        Get regime-specific confidence threshold.

        Args:
            current_regime: Current market regime

        Returns:
            Effective confidence threshold for decision making
        """
        if not self.use_regime_thresholds or current_regime is None:
            return self.min_confidence_threshold

        # Normalize regime name (handle case variations)
        regime_key = current_regime.lower() if current_regime else "unknown"

        # Return regime-specific threshold, fallback to default
        return self.regime_thresholds.get(regime_key, self.min_confidence_threshold)

    def _sigmoid_confidence(self, sample_count: int) -> float:
        """
        Calculate sigmoid-based confidence from sample count.

        Formula: 1 - exp(-sample_count / threshold)

        At sample_threshold, confidence = 1 - 1/e ≈ 0.63
        As samples → ∞, confidence → 1.0
        """
        return 1.0 - np.exp(-sample_count / self.sample_threshold)

    def _build_trade_reason(
        self,
        confidence: float,
        factors: ConfidenceFactors,
        current_regime: Optional[str] = None,
        effective_threshold: Optional[float] = None,
    ) -> str:
        """Build human-readable explanation for trading."""
        reasons = []

        # Include regime context if available
        if current_regime:
            reasons.append(f"Regime: {current_regime.upper()}")
            if effective_threshold:
                reasons.append(f"threshold: {effective_threshold:.2f}")

        if confidence > 0.8:
            reasons.append("Very high confidence")
        elif confidence > 0.7:
            reasons.append("High confidence")
        else:
            reasons.append("Sufficient confidence")

        if factors.sample_count >= self.sample_threshold * 2:
            reasons.append(f"strong sample size ({factors.sample_count})")
        elif factors.sample_count >= self.sample_threshold:
            reasons.append(f"good sample size ({factors.sample_count})")

        if factors.score_separation > 0.3:
            reasons.append("clear winner")

        if factors.pattern_similarity > 0.7:
            reasons.append("strong pattern match")

        if factors.regime_match:
            reasons.append("favorable regime")

        return "; ".join(reasons)

    def _build_skip_reason(
        self,
        confidence: float,
        factors: ConfidenceFactors,
        current_regime: Optional[str] = None,
        effective_threshold: Optional[float] = None,
    ) -> str:
        """Build human-readable explanation for skipping."""
        reasons = []

        # Include regime context
        if current_regime:
            reasons.append(f"Regime: {current_regime.upper()}")
            if effective_threshold:
                reasons.append(f"needs {effective_threshold:.2f}")

        if factors.sample_count < 10:
            reasons.append(f"insufficient data ({factors.sample_count} samples)")
        elif factors.sample_confidence < 0.5:
            reasons.append("low sample confidence")

        if factors.score_separation < 0.1:
            reasons.append("unclear winner (scores too close)")

        if factors.pattern_similarity < 0.3:
            reasons.append("poor pattern match")

        if not factors.regime_match and factors.regime_confidence > 0.7:
            reasons.append("unfavorable regime")

        if not reasons:
            threshold = effective_threshold if effective_threshold else self.min_confidence_threshold
            reasons.append(f"below threshold ({confidence:.2f} < {threshold:.2f})")

        return "; ".join(reasons)

    def adjust_threshold_by_regime(
        self,
        base_threshold: float,
        regime: str,
        regime_performance: Optional[Dict[str, float]] = None,
    ) -> float:
        """
        Adjust confidence threshold based on regime performance.

        If a regime historically performs better, we can be more aggressive (lower threshold).
        If a regime performs worse, be more conservative (higher threshold).

        Args:
            base_threshold: Base confidence threshold
            regime: Current regime (trend/range/panic)
            regime_performance: Historical win rates per regime

        Returns:
            Adjusted threshold
        """
        if not regime_performance or regime not in regime_performance:
            return base_threshold

        regime_win_rate = regime_performance[regime]
        overall_win_rate = np.mean(list(regime_performance.values()))

        # Adjust threshold based on relative performance
        if regime_win_rate > overall_win_rate * 1.1:
            # 10% better than average: lower threshold by up to 0.05
            adjustment = -0.05 * min(1.0, (regime_win_rate - overall_win_rate) / overall_win_rate)
        elif regime_win_rate < overall_win_rate * 0.9:
            # 10% worse than average: raise threshold by up to 0.1
            adjustment = 0.1 * min(1.0, (overall_win_rate - regime_win_rate) / overall_win_rate)
        else:
            adjustment = 0.0

        adjusted = np.clip(base_threshold + adjustment, 0.4, 0.8)

        logger.debug(
            "threshold_adjusted",
            regime=regime,
            base=base_threshold,
            adjusted=adjusted,
            regime_win_rate=regime_win_rate,
            overall_win_rate=overall_win_rate,
        )

        return adjusted


class ConfidenceTracker:
    """
    Tracks confidence scores and outcomes over time.

    This allows us to calibrate confidence scores:
    - If confidence=0.8 trades win 80% of the time → well calibrated
    - If confidence=0.8 trades win 60% of the time → overconfident
    - If confidence=0.8 trades win 90% of the time → underconfident
    """

    def __init__(self):
        # Binned confidence -> outcomes
        self.confidence_bins: Dict[str, list] = {
            "0.5-0.6": [],
            "0.6-0.7": [],
            "0.7-0.8": [],
            "0.8-0.9": [],
            "0.9-1.0": [],
        }

    def record_outcome(self, confidence: float, won: bool) -> None:
        """Record trade outcome for confidence calibration."""
        bin_key = self._get_bin(confidence)
        if bin_key in self.confidence_bins:
            self.confidence_bins[bin_key].append(1 if won else 0)

    def _get_bin(self, confidence: float) -> str:
        """Get confidence bin key."""
        if confidence < 0.6:
            return "0.5-0.6"
        elif confidence < 0.7:
            return "0.6-0.7"
        elif confidence < 0.8:
            return "0.7-0.8"
        elif confidence < 0.9:
            return "0.8-0.9"
        else:
            return "0.9-1.0"

    def get_calibration_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Get calibration statistics.

        Returns:
            Dict of bin -> {expected, actual, count, calibration_error}
        """
        stats = {}

        bin_centers = {
            "0.5-0.6": 0.55,
            "0.6-0.7": 0.65,
            "0.7-0.8": 0.75,
            "0.8-0.9": 0.85,
            "0.9-1.0": 0.95,
        }

        for bin_key, outcomes in self.confidence_bins.items():
            if not outcomes:
                continue

            expected = bin_centers[bin_key]
            actual = np.mean(outcomes)
            count = len(outcomes)
            error = abs(expected - actual)

            stats[bin_key] = {
                "expected_win_rate": expected,
                "actual_win_rate": actual,
                "count": count,
                "calibration_error": error,
            }

        return stats

    def get_overall_calibration_error(self) -> float:
        """
        Get overall calibration error (mean absolute error).

        Perfect calibration = 0.0
        Poor calibration > 0.1
        """
        stats = self.get_calibration_stats()
        if not stats:
            return 0.0

        errors = [s["calibration_error"] for s in stats.values()]
        return float(np.mean(errors))


def calculate_simple_confidence(
    sample_count: int,
    best_score: float,
    runner_up_score: float,
) -> float:
    """
    Quick confidence calculation without all the bells and whistles.

    Args:
        sample_count: Number of samples
        best_score: Best action score (0-1)
        runner_up_score: Runner-up score (0-1)

    Returns:
        Confidence score (0-1)
    """
    scorer = ConfidenceScorer()
    result = scorer.calculate_confidence(
        sample_count=sample_count,
        best_score=best_score,
        runner_up_score=runner_up_score,
    )
    return result.confidence
