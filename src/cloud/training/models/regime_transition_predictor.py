"""
Regime Transition Prediction

Predicts upcoming regime changes BEFORE they happen, enabling proactive positioning.

Current regime detection tells you what's happening NOW.
Regime transition prediction tells you what will happen NEXT.

Example:
- Detects early signs of RISK_ON → RISK_OFF transition
- Predicts transition will occur in ~4 hours
- System reduces high-beta positions NOW, before the crash

Uses leading indicators:
- Volatility acceleration
- Correlation breakdown
- Volume surges
- Leader divergence
- Cross-asset spread widening
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import structlog

from .market_structure import MarketRegime

logger = structlog.get_logger(__name__)


@dataclass
class TransitionFeatures:
    """Leading indicators of regime transition."""

    # Volatility indicators
    volatility_acceleration: float  # Rate of change of volatility
    volatility_zscore: float  # Current vol vs historical

    # Correlation indicators
    correlation_breakdown: float  # Assets decoupling
    correlation_trend: float  # Rising or falling correlation

    # Volume indicators
    volume_surge: float  # Institutional moves
    volume_trend: float  # Increasing/decreasing volume

    # Leader indicators
    leader_divergence: float  # BTC vs alts
    leader_momentum_change: float  # Leader momentum shift

    # Spread indicators
    spread_widening: float  # Bid-ask spreading
    cross_asset_spread: float  # Price dislocations

    # Sentiment proxy
    fear_gauge: float  # Volatility + correlation proxy


@dataclass
class RegimeTransitionPrediction:
    """Prediction of regime transition."""

    current_regime: MarketRegime
    next_regime: MarketRegime
    probability: float  # 0-1, confidence in prediction
    expected_time_hours: float  # Estimated time until transition
    confidence: float  # 0-1, reliability of prediction
    leading_indicators: Dict[str, float]  # What's signaling change
    timestamp: datetime


class RegimeTransitionPredictor:
    """
    Predicts regime transitions using leading indicators.

    Uses a combination of:
    - Statistical thresholds (rules-based)
    - Historical patterns (pattern matching)
    - Optional: ML classifier (XGBoost/LSTM)
    """

    def __init__(
        self,
        lookback_periods: int = 50,
        transition_threshold: float = 0.65,
        min_confidence: float = 0.55,
    ):
        """
        Initialize regime transition predictor.

        Args:
            lookback_periods: Historical window for analysis
            transition_threshold: Probability threshold for transition
            min_confidence: Minimum confidence to issue prediction
        """
        self.lookback_periods = lookback_periods
        self.transition_threshold = transition_threshold
        self.min_confidence = min_confidence

        # Historical regime sequence
        self.regime_history: List[Tuple[datetime, MarketRegime]] = []

        # Transition statistics (from history)
        self.transition_matrix: Dict[Tuple[MarketRegime, MarketRegime], float] = {}

        logger.info(
            "regime_transition_predictor_initialized",
            lookback_periods=lookback_periods,
            threshold=transition_threshold,
        )

    def predict_transition(
        self,
        current_regime: MarketRegime,
        features: TransitionFeatures,
        current_time: datetime,
    ) -> Optional[RegimeTransitionPrediction]:
        """
        Predict if regime will transition soon.

        Args:
            current_regime: Current market regime
            features: Leading indicator features
            current_time: Current timestamp

        Returns:
            RegimeTransitionPrediction or None
        """
        # Calculate transition scores for each potential next regime
        scores = {}

        for next_regime in MarketRegime:
            if next_regime == current_regime:
                continue  # Don't predict staying in same regime

            score = self._calculate_transition_score(
                current_regime, next_regime, features
            )
            scores[next_regime] = score

        # Find most likely transition
        if not scores:
            return None

        next_regime = max(scores, key=scores.get)
        probability = scores[next_regime]

        # Only return if above threshold
        if probability < self.transition_threshold:
            return None

        # Estimate time until transition
        expected_hours = self._estimate_transition_time(features, probability)

        # Calculate confidence
        confidence = self._calculate_confidence(probability, features)

        if confidence < self.min_confidence:
            return None

        # Extract leading indicators
        leading_indicators = self._get_top_indicators(features)

        return RegimeTransitionPrediction(
            current_regime=current_regime,
            next_regime=next_regime,
            probability=probability,
            expected_time_hours=expected_hours,
            confidence=confidence,
            leading_indicators=leading_indicators,
            timestamp=current_time,
        )

    def _calculate_transition_score(
        self,
        from_regime: MarketRegime,
        to_regime: MarketRegime,
        features: TransitionFeatures,
    ) -> float:
        """
        Calculate probability of transition from one regime to another.

        Args:
            from_regime: Current regime
            to_regime: Target regime
            features: Leading indicators

        Returns:
            Transition probability (0-1)
        """
        # Base probability from historical transition matrix
        base_prob = self.transition_matrix.get((from_regime, to_regime), 0.1)

        # Feature-based adjustments
        feature_score = 0.0

        # RISK_ON → RISK_OFF transitions
        if from_regime == MarketRegime.RISK_ON and to_regime == MarketRegime.RISK_OFF:
            # Look for fear signals
            feature_score += features.volatility_acceleration * 0.25
            feature_score += features.correlation_breakdown * 0.20
            feature_score += features.fear_gauge * 0.20
            feature_score += features.spread_widening * 0.15
            feature_score += max(features.volume_surge - 0.5, 0) * 0.20

        # RISK_OFF → RISK_ON transitions
        elif from_regime == MarketRegime.RISK_OFF and to_regime == MarketRegime.RISK_ON:
            # Look for stabilization signals
            feature_score += (1.0 - features.volatility_acceleration) * 0.25
            feature_score += (1.0 - features.fear_gauge) * 0.20
            feature_score += max(features.correlation_trend, 0) * 0.20  # Correlation increasing
            feature_score += (1.0 - features.spread_widening) * 0.15
            feature_score += features.leader_momentum_change * 0.20

        # RISK_ON → ROTATION
        elif from_regime == MarketRegime.RISK_ON and to_regime == MarketRegime.ROTATION:
            feature_score += features.leader_divergence * 0.35
            feature_score += abs(features.correlation_trend) * 0.25
            feature_score += features.volume_surge * 0.20
            feature_score += features.cross_asset_spread * 0.20

        # ROTATION → RISK_OFF
        elif from_regime == MarketRegime.ROTATION and to_regime == MarketRegime.RISK_OFF:
            feature_score += features.volatility_acceleration * 0.30
            feature_score += features.correlation_breakdown * 0.25
            feature_score += features.fear_gauge * 0.25
            feature_score += features.spread_widening * 0.20

        # Default: mix of base prob and features
        else:
            feature_score = 0.5  # Neutral

        # Combine base probability with feature score
        # 70% features, 30% historical
        combined = 0.7 * feature_score + 0.3 * base_prob

        return np.clip(combined, 0.0, 1.0)

    def _estimate_transition_time(
        self,
        features: TransitionFeatures,
        probability: float,
    ) -> float:
        """
        Estimate time until transition.

        Higher probability + stronger signals = sooner transition.

        Args:
            features: Leading indicators
            probability: Transition probability

        Returns:
            Estimated hours until transition
        """
        # Base time: 4-12 hours depending on signal strength
        signal_strength = (
            features.volatility_acceleration * 0.3 +
            features.fear_gauge * 0.3 +
            features.volume_surge * 0.2 +
            features.correlation_breakdown * 0.2
        )

        # High signal strength = sooner
        base_hours = 12.0 - (signal_strength * 8.0)  # 4-12 hours

        # High probability = sooner
        time_factor = 2.0 - probability  # 1.0-2.0x multiplier

        estimated_hours = base_hours * time_factor

        return np.clip(estimated_hours, 1.0, 24.0)

    def _calculate_confidence(
        self,
        probability: float,
        features: TransitionFeatures,
    ) -> float:
        """
        Calculate confidence in prediction.

        Args:
            probability: Transition probability
            features: Leading indicators

        Returns:
            Confidence (0-1)
        """
        # Base confidence from probability
        base_conf = probability

        # Boost if multiple strong signals
        num_strong_signals = sum([
            features.volatility_acceleration > 0.7,
            features.fear_gauge > 0.7,
            features.correlation_breakdown > 0.6,
            features.volume_surge > 0.6,
            features.spread_widening > 0.5,
        ])

        # More signals = higher confidence
        signal_bonus = num_strong_signals * 0.08

        confidence = base_conf + signal_bonus

        return np.clip(confidence, 0.0, 1.0)

    def _get_top_indicators(
        self,
        features: TransitionFeatures,
    ) -> Dict[str, float]:
        """
        Get top leading indicators.

        Args:
            features: All features

        Returns:
            Dictionary of strongest signals
        """
        all_indicators = {
            "volatility_acceleration": features.volatility_acceleration,
            "volatility_zscore": features.volatility_zscore,
            "correlation_breakdown": features.correlation_breakdown,
            "volume_surge": features.volume_surge,
            "leader_divergence": features.leader_divergence,
            "spread_widening": features.spread_widening,
            "fear_gauge": features.fear_gauge,
        }

        # Sort by strength, take top 3
        sorted_indicators = sorted(
            all_indicators.items(),
            key=lambda x: abs(x[1]),
            reverse=True,
        )

        return dict(sorted_indicators[:3])

    def update_regime_history(
        self,
        regime: MarketRegime,
        timestamp: datetime,
    ) -> None:
        """
        Update regime history and transition matrix.

        Args:
            regime: Current regime
            timestamp: Timestamp
        """
        self.regime_history.append((timestamp, regime))

        # Keep only recent history
        if len(self.regime_history) > self.lookback_periods * 2:
            self.regime_history = self.regime_history[-self.lookback_periods * 2 :]

        # Update transition matrix
        self._update_transition_matrix()

    def _update_transition_matrix(self) -> None:
        """Update transition probabilities from history."""
        if len(self.regime_history) < 10:
            return

        # Count transitions
        transitions: Dict[Tuple[MarketRegime, MarketRegime], int] = {}
        regime_counts: Dict[MarketRegime, int] = {}

        for i in range(len(self.regime_history) - 1):
            from_regime = self.regime_history[i][1]
            to_regime = self.regime_history[i + 1][1]

            key = (from_regime, to_regime)
            transitions[key] = transitions.get(key, 0) + 1
            regime_counts[from_regime] = regime_counts.get(from_regime, 0) + 1

        # Calculate probabilities
        for (from_regime, to_regime), count in transitions.items():
            total_from = regime_counts.get(from_regime, 1)
            prob = count / total_from
            self.transition_matrix[(from_regime, to_regime)] = prob

    def get_pre_positioning_strategy(
        self,
        prediction: RegimeTransitionPrediction,
    ) -> Dict[str, str]:
        """
        Get recommended strategy for predicted transition.

        Args:
            prediction: Regime transition prediction

        Returns:
            Dictionary of strategy recommendations
        """
        strategies = {}

        # RISK_ON → RISK_OFF
        if (
            prediction.current_regime == MarketRegime.RISK_ON
            and prediction.next_regime == MarketRegime.RISK_OFF
        ):
            strategies = {
                "action": "reduce_risk",
                "high_beta_positions": "reduce_or_exit",
                "stops": "tighten",
                "new_entries": "pause_or_conservative",
                "cash_allocation": "increase",
                "hedges": "consider_adding",
            }

        # RISK_OFF → RISK_ON
        elif (
            prediction.current_regime == MarketRegime.RISK_OFF
            and prediction.next_regime == MarketRegime.RISK_ON
        ):
            strategies = {
                "action": "accumulate",
                "high_quality_longs": "accumulate",
                "stops": "loosen_slightly",
                "new_entries": "prepare_aggressive",
                "cash_allocation": "deploy",
                "position_sizes": "increase_gradually",
            }

        # ROTATION transitions
        elif prediction.next_regime == MarketRegime.ROTATION:
            strategies = {
                "action": "rebalance",
                "laggards": "consider_adding",
                "leaders": "consider_taking_profits",
                "diversification": "increase",
                "watch_list": "monitor_new_leaders",
            }

        # Default
        else:
            strategies = {
                "action": "monitor",
                "positions": "maintain",
                "risk": "neutral",
            }

        return strategies

    def get_stats(self) -> Dict:
        """
        Get predictor statistics.

        Returns:
            Dictionary of stats
        """
        return {
            "regime_history_length": len(self.regime_history),
            "num_transitions_tracked": len(self.transition_matrix),
            "lookback_periods": self.lookback_periods,
            "threshold": self.transition_threshold,
        }


def calculate_transition_features(
    volatility: float,
    volatility_history: List[float],
    correlation: float,
    correlation_history: List[float],
    volume: float,
    volume_history: List[float],
    leader_momentum: float,
    spread: float,
) -> TransitionFeatures:
    """
    Calculate transition features from market data.

    Args:
        volatility: Current volatility
        volatility_history: Recent volatility values
        correlation: Current correlation
        correlation_history: Recent correlation values
        volume: Current volume
        volume_history: Recent volume values
        leader_momentum: Leader momentum
        spread: Current spread

    Returns:
        TransitionFeatures
    """
    # Volatility indicators
    if len(volatility_history) >= 10:
        vol_mean = np.mean(volatility_history)
        vol_std = np.std(volatility_history)
        vol_acceleration = (volatility - volatility_history[-5]) / (volatility_history[-5] + 1e-9)
        vol_zscore = (volatility - vol_mean) / (vol_std + 1e-9)
    else:
        vol_acceleration = 0.0
        vol_zscore = 0.0

    # Correlation indicators
    if len(correlation_history) >= 10:
        corr_breakdown = max(0, correlation_history[-5] - correlation)  # Falling correlation
        corr_trend = correlation - np.mean(correlation_history[-10:])
    else:
        corr_breakdown = 0.0
        corr_trend = 0.0

    # Volume indicators
    if len(volume_history) >= 10:
        vol_mean = np.mean(volume_history)
        volume_surge = max(0, (volume - vol_mean) / (vol_mean + 1e-9))
        volume_trend = volume - np.mean(volume_history[-5:])
    else:
        volume_surge = 0.0
        volume_trend = 0.0

    # Leader indicators
    leader_divergence = abs(leader_momentum) if abs(leader_momentum) > 0.5 else 0.0
    leader_momentum_change = abs(leader_momentum)

    # Spread indicators
    spread_widening = min(spread / 0.001, 1.0) if spread > 0 else 0.0  # Normalize
    cross_asset_spread = spread_widening  # Simplified

    # Fear gauge (combined indicator)
    fear_gauge = np.mean([vol_acceleration, corr_breakdown, spread_widening])
    fear_gauge = np.clip(fear_gauge, 0.0, 1.0)

    return TransitionFeatures(
        volatility_acceleration=float(np.clip(vol_acceleration, 0.0, 1.0)),
        volatility_zscore=float(np.clip(vol_zscore / 3.0, -1.0, 1.0)),
        correlation_breakdown=float(np.clip(corr_breakdown, 0.0, 1.0)),
        correlation_trend=float(np.clip(corr_trend, -1.0, 1.0)),
        volume_surge=float(np.clip(volume_surge, 0.0, 2.0)),
        volume_trend=float(np.clip(volume_trend / vol_mean if vol_mean > 0 else 0, -1.0, 1.0)),
        leader_divergence=float(leader_divergence),
        leader_momentum_change=float(leader_momentum_change),
        spread_widening=float(spread_widening),
        cross_asset_spread=float(cross_asset_spread),
        fear_gauge=float(fear_gauge),
    )
