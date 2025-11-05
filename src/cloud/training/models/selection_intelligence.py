"""
Selection Intelligence Systems

Bundled selection-focused improvements:
1. Meta-Label Gate (false-positive killer)
2. Separation → Regret Probability
3. Pattern Memory with Evidence Scoring
4. Uncertainty Calibration

These systems improve trade selection quality.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple
import time

import numpy as np
from scipy.special import expit  # sigmoid
import structlog

logger = structlog.get_logger(__name__)


# ============================================================================
# 1. META-LABEL GATE (False-Positive Killer)
# ============================================================================


@dataclass
class MetaGateDecision:
    """Meta-label gate decision."""

    win_probability: float  # P(win | features, regime, engine)
    passes_gate: bool
    threshold: float
    confidence: float
    reason: str


class MetaLabelGate:
    """
    Tiny classifier that predicts P(win | features, regime, engine).

    Blocks trades with low win probability before weights are applied.

    Key Insight:
    - Engine says BREAKOUT with 0.72 confidence
    - But historically, BREAKOUT in RANGE regime with this pattern = 35% WR
    - Meta-gate blocks it: P(win) < 0.50

    Training:
    - Features: [engine_confidence, regime, technique, market_features...]
    - Label: 1 if trade won, 0 if lost
    - Model: Logistic regression or small neural net

    Usage:
        gate = MetaLabelGate(
            threshold=0.50,
        )

        # Train on historical trades
        gate.fit(features_history, outcomes_history)

        # Before trading
        decision = gate.check_gate(
            features={'engine_conf': 0.72, 'regime': 'RANGE', 'technique': 'BREAKOUT'},
        )

        if not decision.passes_gate:
            logger.warning("meta_gate_blocked", win_prob=decision.win_probability)
            return None

    Implementation Note:
    This is a simplified version. Full implementation would use sklearn/torch.
    """

    def __init__(
        self,
        threshold: float = 0.50,
        use_regime_specific: bool = True,
    ):
        self.threshold = threshold
        self.use_regime_specific = use_regime_specific

        # Simple learned weights (in production, use proper ML model)
        self.weights: Dict[str, Dict[str, float]] = {
            'TREND': {},
            'RANGE': {},
            'PANIC': {},
        }

        # Historical data for calibration
        self.history: List[Tuple[Dict, bool]] = []

    def fit(
        self,
        features_list: List[Dict[str, float]],
        outcomes: List[bool],
    ) -> None:
        """
        Fit meta-label model.

        Args:
            features_list: List of feature dicts
            outcomes: List of win/loss outcomes
        """
        # Store history
        for features, outcome in zip(features_list, outcomes):
            self.history.append((features, outcome))

        # Simplified fitting - group by (regime, technique)
        # Calculate historical win rate for each group
        groups: Dict[Tuple[str, str], List[bool]] = {}

        for features, outcome in zip(features_list, outcomes):
            regime = features.get('regime', 'TREND')
            technique = features.get('technique', 'unknown')
            key = (regime, technique)

            if key not in groups:
                groups[key] = []
            groups[key].append(outcome)

        # Calculate win rates
        for key, outcomes_list in groups.items():
            regime, technique = key
            win_rate = np.mean(outcomes_list)

            if regime not in self.weights:
                self.weights[regime] = {}

            self.weights[regime][technique] = win_rate

        logger.info("meta_label_fitted", groups_count=len(groups))

    def predict_win_probability(
        self,
        features: Dict[str, float],
    ) -> float:
        """
        Predict P(win | features).

        Returns:
            Win probability (0-1)
        """
        regime = features.get('regime', 'TREND')
        technique = features.get('technique', 'unknown')
        engine_conf = features.get('engine_confidence', 0.60)

        # Get base win rate from historical data
        if self.use_regime_specific and regime in self.weights:
            base_wr = self.weights[regime].get(technique, 0.55)
        else:
            # Global average
            all_wrs = []
            for regime_weights in self.weights.values():
                all_wrs.extend(regime_weights.values())
            base_wr = np.mean(all_wrs) if all_wrs else 0.55

        # Adjust by engine confidence
        # Higher confidence → higher win prob
        adjusted_prob = base_wr + (engine_conf - 0.60) * 0.5

        # Clamp to [0.1, 0.9]
        adjusted_prob = np.clip(adjusted_prob, 0.1, 0.9)

        return adjusted_prob

    def check_gate(
        self,
        features: Dict[str, float],
    ) -> MetaGateDecision:
        """
        Check if trade passes meta-label gate.

        Args:
            features: Feature dict with 'regime', 'technique', 'engine_confidence', etc.

        Returns:
            MetaGateDecision
        """
        win_prob = self.predict_win_probability(features)

        passes = win_prob >= self.threshold

        # Confidence based on sample size
        regime = features.get('regime', 'TREND')
        technique = features.get('technique', 'unknown')

        # Count historical samples for this (regime, technique)
        sample_count = sum(
            1 for feat, _ in self.history
            if feat.get('regime') == regime and feat.get('technique') == technique
        )

        # More samples = higher confidence
        confidence = min(sample_count / 50.0, 1.0)

        if passes:
            reason = f"Win probability {win_prob:.2f} ≥ threshold {self.threshold:.2f}"
        else:
            reason = f"Win probability {win_prob:.2f} < threshold {self.threshold:.2f}"

        return MetaGateDecision(
            win_probability=win_prob,
            passes_gate=passes,
            threshold=self.threshold,
            confidence=confidence,
            reason=reason,
        )


# ============================================================================
# 2. SEPARATION → REGRET PROBABILITY
# ============================================================================


@dataclass
class RegretAnalysis:
    """Regret probability analysis."""

    best_score: float
    runner_up_score: float
    separation: float  # best - runner_up
    regret_probability: float  # 0-1, P(regret choosing best)
    sample_size: int
    should_trade: bool
    reason: str


class RegretProbabilityCalculator:
    """
    Convert separation (best - runner_up) into regret probability.

    Key Insight:
    - Large separation (0.8 vs 0.4) → Low regret, high confidence
    - Small separation (0.62 vs 0.58) → High regret, coin flip

    Uses sigmoid with sample size adjustment:
    regret_prob = sigmoid(-k * separation * sqrt(sample_size))

    Usage:
        calc = RegretProbabilityCalculator(regret_threshold=0.40)

        analysis = calc.analyze_regret(
            best_score=0.72,
            runner_up_score=0.68,
            sample_size=50,
        )

        if analysis.regret_probability > 0.40:
            logger.warning("High regret risk - close call")
            position_size *= 0.5  # Size down
"""

    def __init__(
        self,
        regret_threshold: float = 0.40,
        k_factor: float = 10.0,  # Steepness of sigmoid
    ):
        self.regret_threshold = regret_threshold
        self.k = k_factor

    def analyze_regret(
        self,
        best_score: float,
        runner_up_score: float,
        sample_size: int,
    ) -> RegretAnalysis:
        """
        Analyze regret probability.

        Args:
            best_score: Best engine score
            runner_up_score: Runner-up score
            sample_size: Sample size for confidence

        Returns:
            RegretAnalysis
        """
        separation = best_score - runner_up_score

        # Adjust separation by sample size (more samples = more confidence in separation)
        confidence_factor = np.sqrt(max(sample_size, 1))

        # Calculate regret probability using sigmoid
        # Large positive separation → low regret
        # Small separation → high regret
        regret_prob = expit(-self.k * separation * confidence_factor / 10.0)

        should_trade = regret_prob <= self.regret_threshold

        if should_trade:
            reason = f"Low regret risk: {regret_prob:.2f} ≤ {self.regret_threshold:.2f}"
        else:
            reason = f"High regret risk: {regret_prob:.2f} > {self.regret_threshold:.2f} (close call)"

        return RegretAnalysis(
            best_score=best_score,
            runner_up_score=runner_up_score,
            separation=separation,
            regret_probability=regret_prob,
            sample_size=sample_size,
            should_trade=should_trade,
            reason=reason,
        )

    def get_size_multiplier(
        self,
        regret_prob: float,
    ) -> float:
        """Get position size multiplier based on regret."""
        if regret_prob < 0.20:
            return 1.2  # High confidence, size up
        elif regret_prob < 0.30:
            return 1.0  # Normal size
        elif regret_prob < 0.40:
            return 0.7  # Some doubt, size down
        else:
            return 0.0  # High regret, skip


# ============================================================================
# 3. PATTERN MEMORY WITH EVIDENCE SCORING
# ============================================================================


@dataclass
class PatternEvidence:
    """Pattern evidence from winner/loser similarity."""

    winner_similarity: float  # 0-1, similarity to winning patterns
    loser_similarity: float  # 0-1, similarity to losing patterns
    evidence: float  # winner_sim - loser_sim (-1 to +1)
    should_block: bool
    confidence: float


class PatternMemoryWithEvidence:
    """
    Store both winner and loser embeddings.

    Evidence = winner_similarity - loser_similarity

    Block if evidence ≤ 0 (looks more like losers than winners).

    Usage:
        memory = PatternMemoryWithEvidence()

        # Store historical patterns
        memory.store_winner(embedding=[0.5, 0.3, ...])
        memory.store_loser(embedding=[0.2, 0.8, ...])

        # Check new pattern
        evidence = memory.compute_evidence(new_embedding)

        if evidence.should_block:
            logger.warning("Pattern looks like historical losers")
            return None
    """

    def __init__(
        self,
        evidence_threshold: float = 0.0,
        similarity_threshold: float = 0.70,
        max_patterns: int = 1000,
    ):
        self.evidence_threshold = evidence_threshold
        self.similarity_threshold = similarity_threshold
        self.max_patterns = max_patterns

        self.winner_embeddings: List[np.ndarray] = []
        self.loser_embeddings: List[np.ndarray] = []

    def store_winner(self, embedding: np.ndarray) -> None:
        """Store winning pattern."""
        self.winner_embeddings.append(embedding)
        if len(self.winner_embeddings) > self.max_patterns:
            self.winner_embeddings.pop(0)

    def store_loser(self, embedding: np.ndarray) -> None:
        """Store losing pattern."""
        self.loser_embeddings.append(embedding)
        if len(self.loser_embeddings) > self.max_patterns:
            self.loser_embeddings.pop(0)

    def compute_evidence(
        self,
        embedding: np.ndarray,
    ) -> PatternEvidence:
        """
        Compute pattern evidence.

        Args:
            embedding: Current pattern embedding

        Returns:
            PatternEvidence
        """
        # Calculate similarity to winners
        if self.winner_embeddings:
            winner_sims = [
                self._cosine_similarity(embedding, w)
                for w in self.winner_embeddings
            ]
            winner_sim = np.max(winner_sims)
        else:
            winner_sim = 0.0

        # Calculate similarity to losers
        if self.loser_embeddings:
            loser_sims = [
                self._cosine_similarity(embedding, l)
                for l in self.loser_embeddings
            ]
            loser_sim = np.max(loser_sims)
        else:
            loser_sim = 0.0

        # Evidence = winner_sim - loser_sim
        evidence = winner_sim - loser_sim

        # Block if evidence ≤ threshold
        should_block = evidence <= self.evidence_threshold

        # Confidence based on number of stored patterns
        confidence = min(
            (len(self.winner_embeddings) + len(self.loser_embeddings)) / 100.0,
            1.0
        )

        return PatternEvidence(
            winner_similarity=winner_sim,
            loser_similarity=loser_sim,
            evidence=evidence,
            should_block=should_block,
            confidence=confidence,
        )

    def _cosine_similarity(
        self,
        a: np.ndarray,
        b: np.ndarray,
    ) -> float:
        """Compute cosine similarity."""
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot / (norm_a * norm_b)


# ============================================================================
# 4. UNCERTAINTY CALIBRATION
# ============================================================================


@dataclass
class UncertaintyEstimate:
    """Uncertainty-calibrated edge estimate."""

    edge_hat_bps: float  # Point estimate
    q_lo_bps: float  # 10th percentile (pessimistic)
    q_hi_bps: float  # 90th percentile (optimistic)
    uncertainty: float  # q_hi - q_lo (spread)
    expected_shortfall: float  # Expected loss if in left tail

    should_trade: bool
    reason: str


class UncertaintyCalibrator:
    """
    Provide uncertainty-calibrated predictions using quantile regression.

    Instead of just edge_hat, provide:
    - edge_hat (mean)
    - q_lo (10th percentile)
    - q_hi (90th percentile)

    Trade only if q_lo > expected_cost (even pessimistic case wins).
    Size by expected shortfall, not mean.

    Usage:
        calibrator = UncertaintyCalibrator()

        # Train on historical (features, outcomes)
        calibrator.fit(features_history, outcomes_history)

        # Predict with uncertainty
        estimate = calibrator.predict_with_uncertainty(
            features={'momentum': 0.7, 'volume': 1.5},
            expected_cost_bps=5.0,
        )

        if not estimate.should_trade:
            logger.warning("Pessimistic case loses", q_lo=estimate.q_lo_bps)
            return None

        # Size by expected shortfall (conservative)
        position_size = base_size * (estimate.expected_shortfall / 100.0)
    """

    def __init__(
        self,
        min_q_lo_bps: float = 5.0,  # Require q_lo > 5 bps
    ):
        self.min_q_lo = min_q_lo_bps

        # Historical outcomes for quantile estimation
        self.outcomes_history: List[float] = []

        # Simple quantile estimates (in production, use proper quantile regression)
        self.q10: float = 0.0
        self.q50: float = 0.0
        self.q90: float = 0.0

    def fit(
        self,
        features_list: List[Dict[str, float]],
        outcomes_bps: List[float],
    ) -> None:
        """
        Fit quantile model.

        Args:
            features_list: List of feature dicts
            outcomes_bps: List of outcome bps
        """
        self.outcomes_history = outcomes_bps.copy()

        # Calculate quantiles
        if len(outcomes_bps) > 10:
            self.q10 = np.percentile(outcomes_bps, 10)
            self.q50 = np.percentile(outcomes_bps, 50)
            self.q90 = np.percentile(outcomes_bps, 90)

            logger.info(
                "uncertainty_calibrator_fitted",
                q10=self.q10,
                q50=self.q50,
                q90=self.q90,
            )

    def predict_with_uncertainty(
        self,
        features: Dict[str, float],
        expected_cost_bps: float,
    ) -> UncertaintyEstimate:
        """
        Predict with uncertainty quantiles.

        Args:
            features: Feature dict
            expected_cost_bps: Expected trading cost

        Returns:
            UncertaintyEstimate
        """
        # In simple version, use global quantiles
        # In production, use quantile regression model conditioned on features
        edge_hat = self.q50
        q_lo = self.q10
        q_hi = self.q90

        uncertainty = q_hi - q_lo

        # Expected shortfall (conditional expectation in left tail)
        left_tail = [x for x in self.outcomes_history if x < q_lo]
        if left_tail:
            expected_shortfall = abs(np.mean(left_tail))
        else:
            expected_shortfall = 20.0  # Default

        # Check if should trade
        # Require pessimistic case (q_lo) to beat costs
        should_trade = q_lo > expected_cost_bps

        if should_trade:
            reason = f"Pessimistic case ({q_lo:.1f} bps) > cost ({expected_cost_bps:.1f} bps)"
        else:
            reason = f"Pessimistic case ({q_lo:.1f} bps) ≤ cost ({expected_cost_bps:.1f} bps)"

        return UncertaintyEstimate(
            edge_hat_bps=edge_hat,
            q_lo_bps=q_lo,
            q_hi_bps=q_hi,
            uncertainty=uncertainty,
            expected_shortfall=expected_shortfall,
            should_trade=should_trade,
            reason=reason,
        )

    def get_size_multiplier(
        self,
        estimate: UncertaintyEstimate,
    ) -> float:
        """Get position size multiplier based on uncertainty."""
        # High uncertainty → smaller size
        # Size by expected shortfall
        if estimate.uncertainty < 20:
            return 1.2  # Low uncertainty
        elif estimate.uncertainty < 40:
            return 1.0
        elif estimate.uncertainty < 60:
            return 0.7
        else:
            return 0.5  # High uncertainty
