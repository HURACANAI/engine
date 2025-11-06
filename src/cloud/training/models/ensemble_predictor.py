"""
Ensemble Predictor - Phase 3 + Revuelto Integration

Combines multiple prediction sources for more robust trading decisions:
1. RL Agent predictions
2. Pattern-based predictions
3. Regime-based predictions
4. Historical similarity predictions
5. Alpha Engines (6 specialized techniques from Revuelto)

Uses weighted voting and confidence aggregation.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import structlog

from .alpha_engines import AlphaEngineCoordinator, AlphaSignal, TradingTechnique

logger = structlog.get_logger()


@dataclass
class PredictionSource:
    """A single prediction from one source."""

    source_name: str  # "rl_agent", "pattern", "regime", "similarity"
    prediction: str  # "buy", "sell", "hold"
    confidence: float  # 0-1
    reasoning: str
    weight: float = 1.0  # Can adjust based on source reliability


@dataclass
class EnsemblePrediction:
    """Ensemble prediction combining multiple sources."""

    final_prediction: str  # "buy", "sell", "hold"
    ensemble_confidence: float  # 0-1
    source_predictions: List[PredictionSource]
    agreement_score: float  # 0-1, how much sources agree
    strongest_source: str  # Which source had highest confidence
    reasoning: str


class EnsemblePredictor:
    """
    Combines predictions from multiple sources using ensemble methods.

    Key features:
    1. Weighted voting based on source reliability
    2. Confidence aggregation
    3. Source weight learning based on performance
    4. Agreement scoring
    """

    def __init__(
        self,
        ema_alpha: float = 0.05,
        min_agreement_threshold: float = 0.6,
    ):
        """
        Initialize ensemble predictor.

        Args:
            ema_alpha: Learning rate for source weight updates
            min_agreement_threshold: Minimum agreement to trade confidently
        """
        self.ema_alpha = ema_alpha
        self.min_agreement = min_agreement_threshold

        # Initialize Alpha Engine Coordinator (Revuelto integration)
        self.alpha_engines = AlphaEngineCoordinator()

        # Source weights (learned from performance)
        self.source_weights: Dict[str, float] = {
            "rl_agent": 1.0,
            "pattern": 0.8,
            "regime": 0.9,
            "similarity": 0.85,
            "alpha_engines": 0.95,  # High weight for battle-tested Revuelto engines
        }

        # Source accuracy tracking
        self.source_accuracy: Dict[str, List[float]] = {
            "rl_agent": [],
            "pattern": [],
            "regime": [],
            "similarity": [],
            "alpha_engines": [],
        }

        logger.info(
            "ensemble_predictor_initialized",
            ema_alpha=ema_alpha,
            min_agreement=min_agreement_threshold,
            alpha_engines_enabled=True,
        )

    def predict(
        self,
        rl_prediction: Optional[PredictionSource] = None,
        pattern_prediction: Optional[PredictionSource] = None,
        regime_prediction: Optional[PredictionSource] = None,
        similarity_prediction: Optional[PredictionSource] = None,
        features: Optional[Dict[str, float]] = None,
        current_regime: str = "unknown",
    ) -> EnsemblePrediction:
        """
        Make ensemble prediction from multiple sources.

        Args:
            rl_prediction: Prediction from RL agent
            pattern_prediction: Prediction from pattern recognition
            regime_prediction: Prediction from regime analysis
            similarity_prediction: Prediction from historical similarity
            features: Feature dictionary for Alpha Engines
            current_regime: Current market regime

        Returns:
            EnsemblePrediction combining all sources
        """
        # Collect all predictions
        predictions = []
        if rl_prediction:
            predictions.append(rl_prediction)
        if pattern_prediction:
            predictions.append(pattern_prediction)
        if regime_prediction:
            predictions.append(regime_prediction)
        if similarity_prediction:
            predictions.append(similarity_prediction)

        # Run Alpha Engines (Revuelto) if features available
        if features:
            alpha_signals = self.alpha_engines.generate_all_signals(features, current_regime)
            # Use combine_signals for weighted voting (better than select_best_technique)
            combined_alpha = self.alpha_engines.combine_signals(alpha_signals, current_regime)

            # Convert AlphaSignal to PredictionSource
            alpha_prediction = PredictionSource(
                source_name="alpha_engines",
                prediction=combined_alpha.direction,
                confidence=combined_alpha.confidence,
                reasoning=f"{combined_alpha.technique.value}: {combined_alpha.reasoning}",
                weight=self.source_weights.get("alpha_engines", 0.95),
            )

            if combined_alpha.direction != "hold":
                predictions.append(alpha_prediction)

            logger.debug(
                "alpha_engines_signal",
                technique=combined_alpha.technique.value,
                direction=combined_alpha.direction,
                confidence=combined_alpha.confidence,
            )

        if not predictions:
            # No predictions available
            return EnsemblePrediction(
                final_prediction="hold",
                ensemble_confidence=0.0,
                source_predictions=[],
                agreement_score=0.0,
                strongest_source="none",
                reasoning="No prediction sources available",
            )

        # Apply source weights
        for pred in predictions:
            pred.weight = self.source_weights.get(pred.source_name, 1.0)

        # Calculate weighted votes
        vote_scores = {"buy": 0.0, "sell": 0.0, "hold": 0.0}

        for pred in predictions:
            weighted_confidence = pred.confidence * pred.weight
            vote_scores[pred.prediction] += weighted_confidence

        # Determine winner
        final_prediction = max(vote_scores.items(), key=lambda x: x[1])[0]
        final_score = vote_scores[final_prediction]

        # Calculate agreement score
        agreement = self._calculate_agreement(predictions)

        # Calculate ensemble confidence
        # Higher if:
        # - Strong agreement between sources
        # - High individual confidences
        # - High source weights
        avg_confidence = np.mean([p.confidence for p in predictions])
        ensemble_confidence = (avg_confidence * 0.6 + agreement * 0.4)

        # Find strongest source
        strongest = max(predictions, key=lambda x: x.confidence * x.weight)

        # Generate reasoning
        reasoning = self._generate_reasoning(
            predictions, final_prediction, agreement, ensemble_confidence
        )

        return EnsemblePrediction(
            final_prediction=final_prediction,
            ensemble_confidence=ensemble_confidence,
            source_predictions=predictions,
            agreement_score=agreement,
            strongest_source=strongest.source_name,
            reasoning=reasoning,
        )

    def _calculate_agreement(self, predictions: List[PredictionSource]) -> float:
        """
        Calculate how much the sources agree.

        Args:
            predictions: List of predictions

        Returns:
            Agreement score 0-1
        """
        if len(predictions) <= 1:
            return 1.0  # Only one source, perfect agreement

        # Count how many agree with most common prediction
        pred_counts = {}
        for pred in predictions:
            pred_counts[pred.prediction] = pred_counts.get(pred.prediction, 0) + 1

        most_common_count = max(pred_counts.values())
        agreement = most_common_count / len(predictions)

        return float(agreement)

    def _generate_reasoning(
        self,
        predictions: List[PredictionSource],
        final_prediction: str,
        agreement: float,
        confidence: float,
    ) -> str:
        """Generate human-readable reasoning."""
        parts = []

        # Agreement
        if agreement >= 0.8:
            parts.append(f"Strong agreement ({agreement:.0%})")
        elif agreement >= 0.6:
            parts.append(f"Moderate agreement ({agreement:.0%})")
        else:
            parts.append(f"Weak agreement ({agreement:.0%})")

        # Confidence
        if confidence >= 0.7:
            parts.append(f"high confidence ({confidence:.0%})")
        elif confidence >= 0.5:
            parts.append(f"moderate confidence ({confidence:.0%})")
        else:
            parts.append(f"low confidence ({confidence:.0%})")

        # Source breakdown
        source_summary = ", ".join([
            f"{p.source_name}={p.prediction}" for p in predictions
        ])
        parts.append(f"sources: {source_summary}")

        return " | ".join(parts)

    def update_source_performance(
        self, source_name: str, was_correct: bool
    ) -> None:
        """
        Update source weight based on performance.

        Args:
            source_name: Name of prediction source
            was_correct: Whether prediction was correct
        """
        if source_name not in self.source_weights:
            return

        # Track accuracy
        accuracy_signal = 1.0 if was_correct else 0.0
        self.source_accuracy[source_name].append(accuracy_signal)

        # Keep only recent history
        if len(self.source_accuracy[source_name]) > 100:
            self.source_accuracy[source_name] = self.source_accuracy[source_name][-100:]

        # Update weight using EMA
        current_weight = self.source_weights[source_name]

        # Calculate recent accuracy
        recent_accuracy = np.mean(self.source_accuracy[source_name][-20:])

        # Target weight: 0.5 to 1.5 based on accuracy
        # 50% accuracy → 0.5 weight, 75% accuracy → 1.0 weight, 100% accuracy → 1.5 weight
        target_weight = 0.5 + (recent_accuracy - 0.5) * 2.0
        target_weight = np.clip(target_weight, 0.3, 1.5)

        # Update with EMA
        new_weight = (1 - self.ema_alpha) * current_weight + self.ema_alpha * target_weight

        self.source_weights[source_name] = float(new_weight)

        logger.debug(
            "source_weight_updated",
            source=source_name,
            was_correct=was_correct,
            new_weight=new_weight,
            recent_accuracy=recent_accuracy,
        )

    def get_source_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all sources."""
        stats = {}

        for source_name, weight in self.source_weights.items():
            accuracy_history = self.source_accuracy[source_name]

            if accuracy_history:
                recent_accuracy = np.mean(accuracy_history[-20:])
                overall_accuracy = np.mean(accuracy_history)
                sample_count = len(accuracy_history)
            else:
                recent_accuracy = 0.5
                overall_accuracy = 0.5
                sample_count = 0

            stats[source_name] = {
                "weight": weight,
                "recent_accuracy": recent_accuracy,
                "overall_accuracy": overall_accuracy,
                "sample_count": sample_count,
            }

        return stats

    def get_state(self) -> Dict:
        """Get state for persistence."""
        return {
            "source_weights": self.source_weights.copy(),
            "source_accuracy": {
                k: list(v) for k, v in self.source_accuracy.items()
            },
            "alpha_engines": self.alpha_engines.get_state(),
        }

    def load_state(self, state: Dict) -> None:
        """Load state from persistence."""
        self.source_weights = state.get("source_weights", self.source_weights)
        self.source_accuracy = {
            k: list(v) for k, v in state.get("source_accuracy", {}).items()
        }

        # Load Alpha Engines state
        if "alpha_engines" in state:
            self.alpha_engines.load_state(state["alpha_engines"])

        logger.info(
            "ensemble_predictor_state_loaded",
            num_sources=len(self.source_weights),
            alpha_engines_loaded=True,
        )
