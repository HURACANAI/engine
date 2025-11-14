"""
Regime Warning System

Main detection system for early regime shift warnings.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import List, Optional

import numpy as np
import pandas as pd
import structlog

from .indicators import (
    detect_volatility_shift,
    detect_volume_anomaly,
    detect_correlation_breakdown,
    calculate_cusum,
    calculate_pages_test,
    detect_return_distribution_shift
)

logger = structlog.get_logger(__name__)


class WarningSignal(str, Enum):
    """Types of warning signals"""
    VOLATILITY_SPIKE = "volatility_spike"
    VOLATILITY_DROP = "volatility_drop"
    VOLUME_SPIKE = "volume_spike"
    VOLUME_DROP = "volume_drop"
    CORRELATION_BREAKDOWN = "correlation_breakdown"
    MEAN_SHIFT = "mean_shift"
    DISTRIBUTION_SHIFT = "distribution_shift"


@dataclass
class RegimeWarning:
    """
    Regime shift warning

    Contains information about detected early warning signals.
    """
    timestamp: datetime
    current_regime: str

    # Warning status
    shift_likely: bool
    predicted_regime: Optional[str]
    confidence: float  # [0-1]

    # Signal details
    signals: List[WarningSignal]
    signal_strengths: dict  # {signal: strength}

    # Lead time estimate
    estimated_lead_minutes: int

    # Context
    volatility_zscore: Optional[float] = None
    volume_zscore: Optional[float] = None


class RegimeWarningSystem:
    """
    Regime Early Warning System

    Detects regime shifts 5-30 minutes before they fully materialize.

    Uses multiple statistical indicators to predict regime changes:
    - Volatility shifts
    - Volume anomalies
    - Correlation breakdowns
    - Mean/distribution shifts

    Example:
        system = RegimeWarningSystem()

        # Check for warnings
        warning = system.check_for_warnings(
            candles_df=recent_candles,
            current_regime="trending"
        )

        if warning.shift_likely and warning.confidence > 0.7:
            print(f"HIGH CONFIDENCE WARNING: {warning.predicted_regime}")
            print(f"Signals: {warning.signals}")

            # Reduce position size, tighten stops, etc.
            reduce_exposure()
    """

    def __init__(
        self,
        volatility_window: int = 20,
        volume_window: int = 20,
        volatility_threshold: float = 2.0,
        volume_threshold: float = 3.0,
        confidence_threshold: float = 0.6
    ):
        """
        Initialize regime warning system

        Args:
            volatility_window: Rolling window for volatility
            volume_window: Rolling window for volume
            volatility_threshold: Std threshold for volatility shift
            volume_threshold: Std threshold for volume anomaly
            confidence_threshold: Minimum confidence to issue warning
        """
        self.volatility_window = volatility_window
        self.volume_window = volume_window
        self.volatility_threshold = volatility_threshold
        self.volume_threshold = volume_threshold
        self.confidence_threshold = confidence_threshold

        # Regime transition probabilities (learned from history)
        self._regime_transitions = {
            "trending": {
                "choppy": 0.4,
                "volatile": 0.3,
                "trending": 0.3
            },
            "choppy": {
                "trending": 0.35,
                "volatile": 0.35,
                "choppy": 0.3
            },
            "volatile": {
                "choppy": 0.4,
                "trending": 0.3,
                "volatile": 0.3
            }
        }

    def check_for_warnings(
        self,
        candles_df: pd.DataFrame,
        current_regime: str,
        correlated_asset_returns: Optional[np.ndarray] = None
    ) -> RegimeWarning:
        """
        Check for regime shift warning signals

        Args:
            candles_df: Recent candles (OHLCV)
            current_regime: Current market regime
            correlated_asset_returns: Optional returns for correlation check (e.g., BTC for ETH)

        Returns:
            RegimeWarning
        """
        if len(candles_df) < 50:
            # Not enough data
            return self._no_warning(current_regime)

        # Calculate returns
        returns = candles_df['close'].pct_change().dropna().values
        volumes = candles_df['volume'].values

        signals = []
        signal_strengths = {}

        # 1. Volatility shift detection
        vol_shift, vol_zscore = detect_volatility_shift(
            returns,
            window=self.volatility_window,
            threshold_std=self.volatility_threshold
        )

        if vol_shift:
            if vol_zscore > 0:
                signals.append(WarningSignal.VOLATILITY_SPIKE)
            else:
                signals.append(WarningSignal.VOLATILITY_DROP)
            signal_strengths[WarningSignal.VOLATILITY_SPIKE] = abs(vol_zscore)

        # 2. Volume anomaly detection
        vol_anomaly, vol_zscore_val = detect_volume_anomaly(
            volumes,
            window=self.volume_window,
            threshold_std=self.volume_threshold
        )

        if vol_anomaly:
            if vol_zscore_val > 0:
                signals.append(WarningSignal.VOLUME_SPIKE)
            else:
                signals.append(WarningSignal.VOLUME_DROP)
            signal_strengths[WarningSignal.VOLUME_SPIKE] = abs(vol_zscore_val)

        # 3. Correlation breakdown (if correlated asset provided)
        if correlated_asset_returns is not None:
            corr_breakdown, corr_change = detect_correlation_breakdown(
                returns,
                correlated_asset_returns
            )

            if corr_breakdown:
                signals.append(WarningSignal.CORRELATION_BREAKDOWN)
                signal_strengths[WarningSignal.CORRELATION_BREAKDOWN] = corr_change

        # 4. Mean shift detection (CUSUM)
        mean_shift, cusum_val = calculate_cusum(
            returns,
            target_mean=returns[:-20].mean() if len(returns) > 20 else 0
        )

        if mean_shift:
            signals.append(WarningSignal.MEAN_SHIFT)
            signal_strengths[WarningSignal.MEAN_SHIFT] = cusum_val

        # 5. Distribution shift detection
        dist_shift, ks_stat = detect_return_distribution_shift(returns)

        if dist_shift:
            signals.append(WarningSignal.DISTRIBUTION_SHIFT)
            signal_strengths[WarningSignal.DISTRIBUTION_SHIFT] = ks_stat

        # Calculate confidence and predict regime
        shift_likely = len(signals) >= 2  # At least 2 signals
        confidence = min(len(signals) / 3.0, 1.0)  # More signals = higher confidence

        predicted_regime = None
        if shift_likely:
            predicted_regime = self._predict_regime(
                current_regime,
                signals,
                vol_zscore,
                vol_zscore_val
            )

        # Estimate lead time based on signal strength
        if shift_likely:
            # Stronger signals = shorter lead time
            avg_strength = np.mean(list(signal_strengths.values()))
            estimated_lead_minutes = int(30 / (1 + avg_strength))  # 5-30 minutes
        else:
            estimated_lead_minutes = 0

        warning = RegimeWarning(
            timestamp=datetime.utcnow(),
            current_regime=current_regime,
            shift_likely=shift_likely,
            predicted_regime=predicted_regime,
            confidence=confidence,
            signals=signals,
            signal_strengths=signal_strengths,
            estimated_lead_minutes=estimated_lead_minutes,
            volatility_zscore=vol_zscore if vol_shift else None,
            volume_zscore=vol_zscore_val if vol_anomaly else None
        )

        if shift_likely:
            logger.warning(
                "regime_shift_warning",
                current_regime=current_regime,
                predicted_regime=predicted_regime,
                confidence=confidence,
                signals=[s.value for s in signals],
                lead_minutes=estimated_lead_minutes
            )

        return warning

    def _predict_regime(
        self,
        current_regime: str,
        signals: List[WarningSignal],
        vol_zscore: float,
        volume_zscore: float
    ) -> str:
        """
        Predict next regime based on signals

        Args:
            current_regime: Current regime
            signals: Detected signals
            vol_zscore: Volatility z-score
            volume_zscore: Volume z-score

        Returns:
            Predicted regime
        """
        # Heuristic regime prediction
        # (In production, this could be a trained classifier)

        # High volatility spike → volatile regime
        if (WarningSignal.VOLATILITY_SPIKE in signals and
                abs(vol_zscore) > 3.0):
            return "volatile"

        # Volume spike + volatility → trending
        if (WarningSignal.VOLUME_SPIKE in signals and
                WarningSignal.VOLATILITY_SPIKE in signals):
            return "trending"

        # Volatility drop → choppy
        if WarningSignal.VOLATILITY_DROP in signals:
            return "choppy"

        # Distribution shift → use transition probs
        if WarningSignal.DISTRIBUTION_SHIFT in signals:
            if current_regime in self._regime_transitions:
                transitions = self._regime_transitions[current_regime]
                # Return most likely transition
                return max(transitions, key=transitions.get)

        # Default: stay in current regime
        return current_regime

    def _no_warning(self, current_regime: str) -> RegimeWarning:
        """Return no-warning result"""
        return RegimeWarning(
            timestamp=datetime.utcnow(),
            current_regime=current_regime,
            shift_likely=False,
            predicted_regime=None,
            confidence=0.0,
            signals=[],
            signal_strengths={},
            estimated_lead_minutes=0
        )

    def update_transition_probs(
        self,
        historical_regimes: pd.DataFrame
    ) -> None:
        """
        Update regime transition probabilities from historical data

        Args:
            historical_regimes: DataFrame with ['timestamp', 'regime']

        Example:
            system.update_transition_probs(historical_regimes_df)
        """
        if len(historical_regimes) < 100:
            logger.warning("insufficient_data_for_transition_probs")
            return

        # Calculate empirical transition probabilities
        regimes = historical_regimes['regime'].values

        transitions = {}

        for i in range(len(regimes) - 1):
            from_regime = regimes[i]
            to_regime = regimes[i + 1]

            if from_regime not in transitions:
                transitions[from_regime] = {}

            if to_regime not in transitions[from_regime]:
                transitions[from_regime][to_regime] = 0

            transitions[from_regime][to_regime] += 1

        # Normalize to probabilities
        for from_regime in transitions:
            total = sum(transitions[from_regime].values())
            for to_regime in transitions[from_regime]:
                transitions[from_regime][to_regime] /= total

        self._regime_transitions = transitions

        logger.info(
            "transition_probs_updated",
            num_samples=len(regimes),
            transitions=transitions
        )
