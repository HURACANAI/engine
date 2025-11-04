"""
Market Regime Detection System.

Classifies market conditions into three regimes:
- TREND: Strong directional movement (ADX > 25, consistent momentum)
- RANGE: Consolidation/sideways action (low ATR%, high compression)
- PANIC: High volatility/stress conditions (ATR% spike, high kurtosis)

This allows the RL agent to learn regime-specific strategies.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

import numpy as np
import polars as pl
import structlog

logger = structlog.get_logger(__name__)


class MarketRegime(Enum):
    """Market regime classifications."""
    TREND = "trend"
    RANGE = "range"
    PANIC = "panic"
    UNKNOWN = "unknown"


@dataclass
class RegimeFeatures:
    """Features used for regime detection."""

    # Volatility metrics
    atr_pct: float  # ATR as % of close
    volatility_ratio: float  # Short-term / long-term volatility
    volatility_percentile: float  # Percentile of recent volatility

    # Trend metrics
    adx: float  # Average Directional Index
    trend_strength: float  # -1 to +1
    ema_slope: float  # EMA slope
    momentum_slope: float  # Price momentum slope

    # Distribution metrics
    kurtosis: float  # Tail risk (>3 = fat tails)
    skewness: float  # Asymmetry

    # Compression metrics
    compression_score: float  # Range tightness (0-1)
    bb_width_pct: float  # Bollinger Bands width %


@dataclass
class RegimeDetectionResult:
    """Result of regime detection."""
    regime: MarketRegime
    confidence: float  # 0-1, how confident we are
    regime_scores: Dict[str, float]  # Score for each regime
    features: RegimeFeatures


class RegimeDetector:
    """
    Detects market regime using rule-based scoring.

    This is intentionally simple and interpretable - no HMM or complex ML needed.
    The RL agent will learn what to do in each regime.
    """

    def __init__(
        self,
        trend_threshold: float = 0.6,
        range_threshold: float = 0.6,
        panic_threshold: float = 0.7,
    ):
        """
        Args:
            trend_threshold: Score threshold to classify as TREND
            range_threshold: Score threshold to classify as RANGE
            panic_threshold: Score threshold to classify as PANIC
        """
        self.trend_threshold = trend_threshold
        self.range_threshold = range_threshold
        self.panic_threshold = panic_threshold

        logger.info(
            "regime_detector_initialized",
            trend_threshold=trend_threshold,
            range_threshold=range_threshold,
            panic_threshold=panic_threshold,
        )

    def detect_regime(
        self,
        features_df: pl.DataFrame,
        current_idx: Optional[int] = None,
    ) -> RegimeDetectionResult:
        """
        Detect market regime from features DataFrame.

        Args:
            features_df: DataFrame with calculated features
            current_idx: Index to use (if None, uses last row)

        Returns:
            RegimeDetectionResult with regime and confidence
        """
        if current_idx is None:
            current_idx = features_df.height - 1

        # Extract features for current timepoint
        row = features_df.row(current_idx, named=True)

        # Build RegimeFeatures from row
        regime_features = self._extract_regime_features(features_df, current_idx)

        # Calculate regime scores
        trend_score = self._calculate_trend_score(regime_features)
        range_score = self._calculate_range_score(regime_features)
        panic_score = self._calculate_panic_score(regime_features)

        regime_scores = {
            "trend": trend_score,
            "range": range_score,
            "panic": panic_score,
        }

        # Determine regime (panic takes priority)
        if panic_score >= self.panic_threshold:
            regime = MarketRegime.PANIC
            confidence = panic_score
        elif trend_score >= self.trend_threshold and trend_score > range_score:
            regime = MarketRegime.TREND
            confidence = trend_score
        elif range_score >= self.range_threshold:
            regime = MarketRegime.RANGE
            confidence = range_score
        else:
            # Mixed/unclear regime
            regime = MarketRegime.UNKNOWN
            confidence = max(trend_score, range_score, panic_score)

        return RegimeDetectionResult(
            regime=regime,
            confidence=confidence,
            regime_scores=regime_scores,
            features=regime_features,
        )

    def _extract_regime_features(
        self,
        features_df: pl.DataFrame,
        current_idx: int,
    ) -> RegimeFeatures:
        """Extract features needed for regime detection."""
        row = features_df.row(current_idx, named=True)

        # Get volatility metrics
        atr = row.get("atr", 0.0)
        close = row.get("close", 1.0)
        atr_pct = (atr / close) * 100 if close > 0 else 0.0

        # Calculate volatility ratio (short-term / long-term)
        vol_30 = row.get("realized_sigma_30", 0.01)
        vol_60 = row.get("realized_sigma_60", 0.01)
        vol_ratio = vol_30 / vol_60 if vol_60 > 0 else 1.0

        # Get volatility percentile (estimate from recent data)
        if current_idx >= 60:
            recent_atr = features_df["atr"][max(0, current_idx - 60):current_idx + 1]
            vol_percentile = self._calculate_percentile(atr, recent_atr.to_list())
        else:
            vol_percentile = 0.5

        # Trend metrics
        adx = row.get("adx", 0.0) if "adx" in row else 20.0

        # Calculate trend strength from EMA difference
        ema_5 = row.get("ema_5", close)
        ema_21 = row.get("ema_21", close)
        trend_strength = (ema_5 - ema_21) / close if close > 0 else 0.0
        trend_strength = np.clip(trend_strength * 100, -1.0, 1.0)  # Normalize to [-1, 1]

        # EMA slope (estimate from recent data)
        if current_idx >= 5:
            recent_ema_21 = features_df["ema_21"][max(0, current_idx - 5):current_idx + 1]
            ema_slope = (recent_ema_21[-1] - recent_ema_21[0]) / (close * 5) if close > 0 else 0.0
        else:
            ema_slope = 0.0

        # Momentum slope
        if current_idx >= 5:
            recent_close = features_df["close"][max(0, current_idx - 5):current_idx + 1]
            momentum_slope = (recent_close[-1] - recent_close[0]) / (close * 5) if close > 0 else 0.0
        else:
            momentum_slope = 0.0

        # Distribution metrics (kurtosis, skewness)
        if current_idx >= 20:
            recent_returns = features_df["ret_1"][max(0, current_idx - 20):current_idx + 1]
            returns_array = recent_returns.to_numpy()
            returns_array = returns_array[~np.isnan(returns_array)]

            if len(returns_array) > 4:
                kurtosis = float(self._calculate_kurtosis(returns_array))
                skewness = float(self._calculate_skewness(returns_array))
            else:
                kurtosis = 3.0
                skewness = 0.0
        else:
            kurtosis = 3.0
            skewness = 0.0

        # Compression score (from BB width or range)
        bb_width_pct = row.get("bb_width", 0.04) * 100 if "bb_width" in row else 4.0
        compression_score = 1.0 / (1.0 + bb_width_pct / 2.0)  # High when BB tight

        return RegimeFeatures(
            atr_pct=atr_pct,
            volatility_ratio=vol_ratio,
            volatility_percentile=vol_percentile,
            adx=adx,
            trend_strength=trend_strength,
            ema_slope=ema_slope,
            momentum_slope=momentum_slope,
            kurtosis=kurtosis,
            skewness=skewness,
            compression_score=compression_score,
            bb_width_pct=bb_width_pct,
        )

    def _calculate_trend_score(self, features: RegimeFeatures) -> float:
        """
        Calculate trend regime score (0-1).

        High score when:
        - ADX > 25 (strong trend)
        - Trend strength significant
        - Momentum consistent
        - Low compression (trending, not range-bound)
        """
        score = 0.0

        # ADX component (30% weight)
        if features.adx > 25:
            score += 0.3 * min(1.0, (features.adx - 25) / 25)

        # Trend strength component (30% weight)
        score += 0.3 * abs(features.trend_strength)

        # Momentum alignment (20% weight)
        if abs(features.momentum_slope) > 0.01:
            score += 0.2

        # EMA slope (10% weight)
        score += 0.1 * min(1.0, abs(features.ema_slope) * 10)

        # Anti-compression (10% weight) - trends have wider ranges
        score += 0.1 * (1.0 - features.compression_score)

        return np.clip(score, 0.0, 1.0)

    def _calculate_range_score(self, features: RegimeFeatures) -> float:
        """
        Calculate range regime score (0-1).

        High score when:
        - Low ADX (< 25, weak trend)
        - High compression (tight BB)
        - Low volatility
        - Symmetrical distribution
        """
        score = 0.0

        # Low ADX component (30% weight)
        if features.adx < 25:
            score += 0.3 * (1.0 - features.adx / 25)

        # Compression component (40% weight)
        score += 0.4 * features.compression_score

        # Low trend strength (15% weight)
        score += 0.15 * (1.0 - abs(features.trend_strength))

        # Low volatility ratio (15% weight)
        if features.volatility_ratio < 1.2:
            score += 0.15

        return np.clip(score, 0.0, 1.0)

    def _calculate_panic_score(self, features: RegimeFeatures) -> float:
        """
        Calculate panic regime score (0-1).

        High score when:
        - Very high volatility (ATR% spike)
        - High kurtosis (fat tails)
        - Volatility ratio > 1.5 (recent vol surge)
        - Extreme ATR percentile
        """
        score = 0.0

        # ATR spike component (30% weight)
        if features.atr_pct > 3.0:  # > 3% ATR is high for crypto
            score += 0.3 * min(1.0, (features.atr_pct - 3.0) / 3.0)

        # Volatility ratio component (25% weight)
        if features.volatility_ratio > 1.5:
            score += 0.25 * min(1.0, (features.volatility_ratio - 1.5) / 1.0)

        # Kurtosis component (25% weight) - fat tails indicate panic
        if features.kurtosis > 4.0:
            score += 0.25 * min(1.0, (features.kurtosis - 4.0) / 6.0)

        # Volatility percentile (20% weight)
        if features.volatility_percentile > 0.8:
            score += 0.2 * (features.volatility_percentile - 0.8) / 0.2

        return np.clip(score, 0.0, 1.0)

    @staticmethod
    def _calculate_percentile(value: float, data: list) -> float:
        """Calculate percentile of value in data."""
        if not data:
            return 0.5
        data_array = np.array(data)
        return float(np.sum(data_array <= value) / len(data_array))

    @staticmethod
    def _calculate_kurtosis(data: np.ndarray) -> float:
        """Calculate excess kurtosis (normal distribution = 0)."""
        if len(data) < 4:
            return 0.0
        mean = np.mean(data)
        std = np.std(data)
        if std < 1e-9:
            return 0.0
        normalized = (data - mean) / std
        kurtosis = np.mean(normalized ** 4) - 3.0  # Excess kurtosis
        return kurtosis

    @staticmethod
    def _calculate_skewness(data: np.ndarray) -> float:
        """Calculate skewness (symmetry measure)."""
        if len(data) < 3:
            return 0.0
        mean = np.mean(data)
        std = np.std(data)
        if std < 1e-9:
            return 0.0
        normalized = (data - mean) / std
        skewness = np.mean(normalized ** 3)
        return skewness


def detect_regime_from_features(
    features_df: pl.DataFrame,
    current_idx: Optional[int] = None,
) -> MarketRegime:
    """
    Convenience function to detect regime from features DataFrame.

    Args:
        features_df: DataFrame with calculated features
        current_idx: Index to use (if None, uses last row)

    Returns:
        MarketRegime enum
    """
    detector = RegimeDetector()
    result = detector.detect_regime(features_df, current_idx)

    logger.debug(
        "regime_detected",
        regime=result.regime.value,
        confidence=result.confidence,
        scores=result.regime_scores,
    )

    return result.regime
