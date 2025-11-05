"""
Regime Transition Anticipation

Predicts regime shifts BEFORE they happen, allowing proactive position management.

Key Problems Solved:
1. **Reactive Exits**: Bot reacts AFTER regime shifts (TREND → PANIC), losing profits
2. **Missed Opportunities**: Bot enters AFTER regime shift (RANGE → TREND), missing the move
3. **Whipsaw**: Bot trades regime change, but it's false signal, market reverts

Solution: Lead Indicators for Regime Changes
- Pre-PANIC signals: Volatility spiking + volume surging + ADX collapsing
- Pre-TREND signals: Compression releasing + volume confirming + momentum building
- Pre-RANGE signals: ADX declining + volatility compressing + failed breakouts

Example:
    Current: TREND regime (ADX: 35, Vol: 120 bps)

    Pre-PANIC Signals Detected:
    - Volatility: 120 → 280 bps (spike!)
    - Volume: 1.5x → 3.2x average (surge!)
    - ADX: 35 → 22 (collapsing!)
    - Compression: Increasing rapidly

    Anticipation: PANIC incoming in 5-15 minutes (confidence: 0.78)

    Action: Exit 50% of positions NOW before PANIC confirmed
            (Instead of waiting for official regime change)

    Result: Exit at +150 bps before PANIC drives price to +50 bps
            Saved: 100 bps by anticipating!
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


class RegimeType(Enum):
    """Regime types."""

    TREND = "trend"
    RANGE = "range"
    PANIC = "panic"
    UNKNOWN = "unknown"


class TransitionType(Enum):
    """Types of regime transitions."""

    TREND_TO_PANIC = "trend_to_panic"
    TREND_TO_RANGE = "trend_to_range"
    RANGE_TO_TREND = "range_to_trend"
    RANGE_TO_PANIC = "range_to_panic"
    PANIC_TO_TREND = "panic_to_trend"
    PANIC_TO_RANGE = "panic_to_range"


@dataclass
class TransitionSignal:
    """Signal for potential regime transition."""

    from_regime: RegimeType
    to_regime: RegimeType
    transition_type: TransitionType
    confidence: float  # 0-1
    time_horizon_minutes: Tuple[float, float]  # (min, max) estimated time until transition
    lead_indicators: Dict[str, float]  # Indicator values that triggered signal
    reasoning: str
    recommended_action: str  # 'EXIT_50PCT', 'EXIT_ALL', 'ENTER_EARLY', 'TIGHTEN_STOPS', 'MONITOR'


@dataclass
class RegimeStability:
    """Assessment of current regime stability."""

    current_regime: RegimeType
    stability_score: float  # 0-1, 1 = very stable
    transition_probability: float  # 0-1, probability of transition soon
    most_likely_next_regime: Optional[RegimeType]
    warning_level: str  # 'STABLE', 'WATCH', 'WARNING', 'CRITICAL'


class RegimeTransitionAnticipator:
    """
    Anticipates regime transitions before they occur.

    Three main functions:
    1. **Pre-PANIC Detection**: Volatility spike + volume surge + ADX collapse
    2. **Pre-TREND Detection**: Compression release + volume confirm + momentum build
    3. **Pre-RANGE Detection**: ADX decline + vol compress + failed breakouts

    Approach:
    - Monitor lead indicators (volatility, volume, compression, ADX, momentum)
    - Detect rapid changes in indicators (spikes, collapses)
    - Combine signals for regime transition prediction
    - Provide early warning (5-15 minutes before official change)

    Usage:
        anticipator = RegimeTransitionAnticipator()

        # Update with latest features
        anticipator.update_features(
            current_regime='trend',
            features={
                'volatility_bps': 280.0,  # Spiking!
                'volume_ratio': 3.2,  # Surging!
                'adx': 22.0,  # Collapsing!
                'compression': 0.25,
                # ... other features
            },
        )

        # Check for transition signals
        signal = anticipator.check_transition()

        if signal and signal.confidence > 0.70:
            logger.warning(f"Regime transition anticipated: {signal.reasoning}")

            if signal.recommended_action == 'EXIT_50PCT':
                # PANIC incoming - exit half now
                exit_positions(percentage=0.50)
    """

    def __init__(
        self,
        volatility_spike_threshold: float = 2.0,  # 2x increase
        volume_surge_threshold: float = 2.5,  # 2.5x increase
        adx_collapse_threshold: float = 0.6,  # 40% decrease
        compression_release_threshold: float = 0.7,  # 0.7+ compression releasing
        min_confidence: float = 0.65,
        lookback_periods: int = 20,
    ):
        """
        Initialize regime transition anticipator.

        Args:
            volatility_spike_threshold: Multiplier for vol spike
            volume_surge_threshold: Multiplier for volume surge
            adx_collapse_threshold: Multiplier for ADX collapse (0.6 = 40% drop)
            compression_release_threshold: Compression score for release
            min_confidence: Minimum confidence to report signal
            lookback_periods: Periods to track for baseline
        """
        self.vol_spike_threshold = volatility_spike_threshold
        self.vol_surge_threshold = volume_surge_threshold
        self.adx_collapse_threshold = adx_collapse_threshold
        self.compression_threshold = compression_release_threshold
        self.min_confidence = min_confidence
        self.lookback = lookback_periods

        # Feature history for baseline calculation
        self.feature_history: List[Dict[str, float]] = []
        self.current_regime: Optional[RegimeType] = None

        logger.info(
            "regime_transition_anticipator_initialized",
            vol_spike_threshold=volatility_spike_threshold,
            volume_surge_threshold=volume_surge_threshold,
        )

    def update_features(
        self,
        current_regime: str,
        features: Dict[str, float],
    ) -> None:
        """
        Update with latest features.

        Args:
            current_regime: Current regime ('trend', 'range', 'panic')
            features: Dict of feature values
        """
        self.current_regime = RegimeType(current_regime.lower())

        # Add to history
        self.feature_history.append(features)

        # Keep only lookback periods
        if len(self.feature_history) > self.lookback:
            self.feature_history = self.feature_history[-self.lookback:]

    def check_transition(self) -> Optional[TransitionSignal]:
        """
        Check for regime transition signals.

        Returns:
            TransitionSignal if transition anticipated, None otherwise
        """
        if len(self.feature_history) < self.lookback // 2:
            # Not enough history
            return None

        if self.current_regime is None:
            return None

        # Check for each transition type
        if self.current_regime == RegimeType.TREND:
            # Check TREND → PANIC
            signal = self._check_trend_to_panic()
            if signal and signal.confidence >= self.min_confidence:
                return signal

            # Check TREND → RANGE
            signal = self._check_trend_to_range()
            if signal and signal.confidence >= self.min_confidence:
                return signal

        elif self.current_regime == RegimeType.RANGE:
            # Check RANGE → PANIC
            signal = self._check_range_to_panic()
            if signal and signal.confidence >= self.min_confidence:
                return signal

            # Check RANGE → TREND
            signal = self._check_range_to_trend()
            if signal and signal.confidence >= self.min_confidence:
                return signal

        elif self.current_regime == RegimeType.PANIC:
            # Check PANIC → TREND
            signal = self._check_panic_to_trend()
            if signal and signal.confidence >= self.min_confidence:
                return signal

            # Check PANIC → RANGE
            signal = self._check_panic_to_range()
            if signal and signal.confidence >= self.min_confidence:
                return signal

        return None

    def assess_stability(self) -> RegimeStability:
        """
        Assess current regime stability.

        Returns:
            RegimeStability with stability assessment
        """
        if len(self.feature_history) < self.lookback // 2 or self.current_regime is None:
            return RegimeStability(
                current_regime=self.current_regime or RegimeType.UNKNOWN,
                stability_score=0.5,
                transition_probability=0.0,
                most_likely_next_regime=None,
                warning_level='STABLE',
            )

        # Calculate stability based on feature volatility
        recent_features = self.feature_history[-10:]

        # Check volatility of key features
        vol_changes = [f.get('volatility_bps', 100) for f in recent_features]
        adx_changes = [f.get('adx', 25) for f in recent_features]

        vol_std = np.std(vol_changes) / np.mean(vol_changes) if np.mean(vol_changes) > 0 else 0
        adx_std = np.std(adx_changes) / np.mean(adx_changes) if np.mean(adx_changes) > 0 else 0

        # Stability score (lower std = more stable)
        stability_score = 1.0 - min((vol_std + adx_std) / 2, 1.0)

        # Transition probability (inverse of stability)
        transition_prob = 1.0 - stability_score

        # Most likely next regime
        if self.current_regime == RegimeType.TREND:
            if transition_prob > 0.7:
                most_likely_next = RegimeType.PANIC
            elif transition_prob > 0.4:
                most_likely_next = RegimeType.RANGE
            else:
                most_likely_next = None
        elif self.current_regime == RegimeType.RANGE:
            if transition_prob > 0.6:
                most_likely_next = RegimeType.TREND
            else:
                most_likely_next = None
        else:  # PANIC
            if transition_prob > 0.5:
                most_likely_next = RegimeType.RANGE
            else:
                most_likely_next = None

        # Warning level
        if transition_prob > 0.75:
            warning_level = 'CRITICAL'
        elif transition_prob > 0.60:
            warning_level = 'WARNING'
        elif transition_prob > 0.40:
            warning_level = 'WATCH'
        else:
            warning_level = 'STABLE'

        return RegimeStability(
            current_regime=self.current_regime,
            stability_score=stability_score,
            transition_probability=transition_prob,
            most_likely_next_regime=most_likely_next,
            warning_level=warning_level,
        )

    def _check_trend_to_panic(self) -> Optional[TransitionSignal]:
        """Check for TREND → PANIC transition."""
        current = self.feature_history[-1]
        baseline = self._get_baseline_features()

        # Lead indicators for PANIC:
        # 1. Volatility spiking (2x+)
        # 2. Volume surging (2.5x+)
        # 3. ADX collapsing (40%+ drop)

        vol_current = current.get('volatility_bps', 100)
        vol_baseline = baseline.get('volatility_bps', 100)
        vol_ratio = vol_current / vol_baseline if vol_baseline > 0 else 1.0

        volume_current = current.get('volume_ratio', 1.0)
        volume_baseline = baseline.get('volume_ratio', 1.0)
        volume_ratio = volume_current / volume_baseline if volume_baseline > 0 else 1.0

        adx_current = current.get('adx', 25)
        adx_baseline = baseline.get('adx', 25)
        adx_ratio = adx_current / adx_baseline if adx_baseline > 0 else 1.0

        # Check thresholds
        vol_spiking = vol_ratio >= self.vol_spike_threshold
        volume_surging = volume_ratio >= self.vol_surge_threshold
        adx_collapsing = adx_ratio <= self.adx_collapse_threshold

        # Count signals
        signals = sum([vol_spiking, volume_surging, adx_collapsing])

        if signals >= 2:  # Need 2/3 signals
            # Calculate confidence
            confidence = 0.0
            confidence += 0.35 if vol_spiking else 0.0
            confidence += 0.35 if volume_surging else 0.0
            confidence += 0.30 if adx_collapsing else 0.0

            lead_indicators = {
                'volatility_ratio': vol_ratio,
                'volume_ratio': volume_ratio,
                'adx_ratio': adx_ratio,
            }

            reasoning = (
                f"Pre-PANIC signals: Vol {vol_ratio:.1f}x "
                f"({'SPIKE' if vol_spiking else 'normal'}), "
                f"Volume {volume_ratio:.1f}x "
                f"({'SURGE' if volume_surging else 'normal'}), "
                f"ADX {adx_ratio:.1f}x "
                f"({'COLLAPSE' if adx_collapsing else 'normal'})"
            )

            return TransitionSignal(
                from_regime=RegimeType.TREND,
                to_regime=RegimeType.PANIC,
                transition_type=TransitionType.TREND_TO_PANIC,
                confidence=confidence,
                time_horizon_minutes=(5, 15),
                lead_indicators=lead_indicators,
                reasoning=reasoning,
                recommended_action='EXIT_50PCT',  # Exit half before panic confirmed
            )

        return None

    def _check_trend_to_range(self) -> Optional[TransitionSignal]:
        """Check for TREND → RANGE transition."""
        current = self.feature_history[-1]
        baseline = self._get_baseline_features()

        # Lead indicators for RANGE:
        # 1. ADX declining (trend weakening)
        # 2. Compression increasing (range forming)
        # 3. Momentum flattening

        adx_current = current.get('adx', 25)
        adx_baseline = baseline.get('adx', 25)
        adx_declining = adx_current < adx_baseline * 0.8

        compression_current = current.get('compression', 0.5)
        compression_increasing = compression_current > 0.60

        momentum_current = abs(current.get('momentum_slope', 0.5))
        momentum_flattening = momentum_current < 0.3

        signals = sum([adx_declining, compression_increasing, momentum_flattening])

        if signals >= 2:
            confidence = 0.0
            confidence += 0.35 if adx_declining else 0.0
            confidence += 0.40 if compression_increasing else 0.0
            confidence += 0.25 if momentum_flattening else 0.0

            return TransitionSignal(
                from_regime=RegimeType.TREND,
                to_regime=RegimeType.RANGE,
                transition_type=TransitionType.TREND_TO_RANGE,
                confidence=confidence,
                time_horizon_minutes=(10, 30),
                lead_indicators={
                    'adx': adx_current,
                    'compression': compression_current,
                    'momentum': momentum_current,
                },
                reasoning=f"Pre-RANGE: ADX {adx_current:.0f}, Compression {compression_current:.2f}",
                recommended_action='TIGHTEN_STOPS',
            )

        return None

    def _check_range_to_trend(self) -> Optional[TransitionSignal]:
        """Check for RANGE → TREND transition."""
        current = self.feature_history[-1]

        # Lead indicators for TREND:
        # 1. Compression releasing (>0.7)
        # 2. Volume confirming (>1.5x)
        # 3. Momentum building

        compression = current.get('compression', 0.5)
        volume_ratio = current.get('volume_ratio', 1.0)
        momentum = abs(current.get('momentum_slope', 0.0))

        compression_releasing = compression > self.compression_threshold
        volume_confirming = volume_ratio > 1.5
        momentum_building = momentum > 0.4

        signals = sum([compression_releasing, volume_confirming, momentum_building])

        if signals >= 2:
            confidence = 0.0
            confidence += 0.40 if compression_releasing else 0.0
            confidence += 0.35 if volume_confirming else 0.0
            confidence += 0.25 if momentum_building else 0.0

            return TransitionSignal(
                from_regime=RegimeType.RANGE,
                to_regime=RegimeType.TREND,
                transition_type=TransitionType.RANGE_TO_TREND,
                confidence=confidence,
                time_horizon_minutes=(5, 20),
                lead_indicators={
                    'compression': compression,
                    'volume_ratio': volume_ratio,
                    'momentum': momentum,
                },
                reasoning=f"Pre-TREND: Compression {compression:.2f}, Volume {volume_ratio:.1f}x",
                recommended_action='ENTER_EARLY',  # Enter before breakout confirmed
            )

        return None

    def _check_range_to_panic(self) -> Optional[TransitionSignal]:
        """Check for RANGE → PANIC transition."""
        # Similar to TREND → PANIC but from RANGE
        return self._check_trend_to_panic()  # Same indicators

    def _check_panic_to_trend(self) -> Optional[TransitionSignal]:
        """Check for PANIC → TREND transition."""
        current = self.feature_history[-1]
        baseline = self._get_baseline_features()

        # Lead indicators for stabilization:
        # 1. Volatility declining
        # 2. Volume normalizing
        # 3. Momentum building directionally

        vol_current = current.get('volatility_bps', 200)
        vol_baseline = baseline.get('volatility_bps', 200)
        vol_declining = vol_current < vol_baseline * 0.7

        volume_current = current.get('volume_ratio', 2.0)
        volume_normalizing = 0.8 <= volume_current <= 1.5

        momentum = abs(current.get('momentum_slope', 0.0))
        momentum_building = momentum > 0.5

        signals = sum([vol_declining, volume_normalizing, momentum_building])

        if signals >= 2:
            confidence = 0.0
            confidence += 0.35 if vol_declining else 0.0
            confidence += 0.30 if volume_normalizing else 0.0
            confidence += 0.35 if momentum_building else 0.0

            return TransitionSignal(
                from_regime=RegimeType.PANIC,
                to_regime=RegimeType.TREND,
                transition_type=TransitionType.PANIC_TO_TREND,
                confidence=confidence,
                time_horizon_minutes=(10, 30),
                lead_indicators={
                    'volatility_bps': vol_current,
                    'volume_ratio': volume_current,
                    'momentum': momentum,
                },
                reasoning="Market stabilizing from PANIC, TREND forming",
                recommended_action='ENTER_EARLY',
            )

        return None

    def _check_panic_to_range(self) -> Optional[TransitionSignal]:
        """Check for PANIC → RANGE transition."""
        current = self.feature_history[-1]

        vol_current = current.get('volatility_bps', 200)
        compression = current.get('compression', 0.3)

        vol_normalizing = vol_current < 150
        compression_forming = compression > 0.55

        if vol_normalizing and compression_forming:
            confidence = 0.70

            return TransitionSignal(
                from_regime=RegimeType.PANIC,
                to_regime=RegimeType.RANGE,
                transition_type=TransitionType.PANIC_TO_RANGE,
                confidence=confidence,
                time_horizon_minutes=(15, 45),
                lead_indicators={
                    'volatility_bps': vol_current,
                    'compression': compression,
                },
                reasoning="Market stabilizing into RANGE",
                recommended_action='MONITOR',
            )

        return None

    def _get_baseline_features(self) -> Dict[str, float]:
        """Calculate baseline (average) features from recent history."""
        if len(self.feature_history) < 5:
            return self.feature_history[-1] if self.feature_history else {}

        # Average last 10 periods (excluding current)
        recent = self.feature_history[-11:-1] if len(self.feature_history) > 10 else self.feature_history[:-1]

        baseline = {}
        if recent:
            all_keys = set()
            for features in recent:
                all_keys.update(features.keys())

            for key in all_keys:
                values = [f.get(key, 0) for f in recent if key in f]
                if values:
                    baseline[key] = np.mean(values)

        return baseline
