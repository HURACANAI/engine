"""
Crypto-Optimized Regime Detector

This is a calibrated version of RegimeDetector specifically for cryptocurrency markets.
Uses higher volatility thresholds and balanced scoring appropriate for crypto's natural volatility.
"""

from .regime_detector import RegimeDetector


class CryptoRegimeDetector(RegimeDetector):
    """
    Regime detector optimized for cryptocurrency markets.

    Key differences from base RegimeDetector:
    1. Higher panic threshold (0.80 vs 0.70) - crypto is naturally volatile
    2. Lower range threshold (0.50 vs 0.60) - crypto ranges are wider
    3. Adjusted panic scoring - 5% ATR instead of 3%
    4. Equal priority scoring - no automatic panic priority
    """

    def __init__(self):
        """Initialize with crypto-optimized thresholds."""
        super().__init__(
            trend_threshold=0.55,   # Slightly easier for crypto trends
            range_threshold=0.50,   # Much lower - crypto ranges are wide
            panic_threshold=0.80,   # Higher - only real panic, not normal vol
        )

        # Override panic weights for crypto
        self._meta_weights["panic"] = {
            "intercept": -1.2,              # Harder to trigger
            "volatility_ratio": 1.2,        # Less weight
            "volatility_percentile": 1.4,   # Less weight
            "kurtosis": 0.04,               # Less weight
            "skewness": -0.02,              # Less weight
        }

    def _calculate_panic_score(self, features) -> float:
        """
        Calculate panic score for crypto (stricter than base).

        Crypto-specific adjustments:
        - ATR threshold raised from 3.0% to 5.0%
        - Volatility ratio threshold raised from 1.5 to 2.0
        - Kurtosis threshold raised from 4.0 to 5.0
        """
        import numpy as np

        score = 0.0

        # ATR spike component (30% weight) - CRYPTO: 5% threshold
        if features.atr_pct > 5.0:  # Raised from 3.0 for crypto
            score += 0.3 * min(1.0, (features.atr_pct - 5.0) / 5.0)

        # Volatility ratio component (25% weight) - CRYPTO: 2.0 threshold
        if features.volatility_ratio > 2.0:  # Raised from 1.5 for crypto
            score += 0.25 * min(1.0, (features.volatility_ratio - 2.0) / 1.0)

        # Kurtosis component (25% weight) - CRYPTO: 5.0 threshold
        if features.kurtosis > 5.0:  # Raised from 4.0 for crypto
            score += 0.25 * min(1.0, (features.kurtosis - 5.0) / 8.0)

        # Volatility percentile (20% weight) - unchanged
        if features.volatility_percentile > 0.85:  # Raised from 0.8
            score += 0.2 * (features.volatility_percentile - 0.85) / 0.15

        return np.clip(score, 0.0, 1.0)

    def detect_regime(self, features_df, current_idx=None):
        """
        Detect regime using crypto-optimized logic.

        Key difference: Uses HIGHEST SCORE instead of panic priority.
        This prevents over-classification of normal volatility as panic.
        """
        # Get base detection result
        result = super().detect_regime(features_df, current_idx)

        # CRYPTO FIX: Use highest score instead of panic priority
        blended_scores = {
            "trend": result.regime_scores.get("trend_blended", 0.0),
            "range": result.regime_scores.get("range_blended", 0.0),
            "panic": result.regime_scores.get("panic_blended", 0.0),
        }

        # Find highest scoring regime
        max_regime = max(blended_scores.items(), key=lambda x: x[1])
        regime_name, max_score = max_regime

        # Only classify if confidence is sufficient
        from .regime_detector import MarketRegime

        if max_score >= 0.5:  # Lower threshold for crypto
            if regime_name == "panic" and max_score >= self.panic_threshold:
                regime = MarketRegime.PANIC
                confidence = max_score
            elif regime_name == "trend" and max_score >= self.trend_threshold:
                regime = MarketRegime.TREND
                confidence = max_score
            elif regime_name == "range" and max_score >= self.range_threshold:
                regime = MarketRegime.RANGE
                confidence = max_score
            else:
                # Score present but below threshold
                regime = MarketRegime.UNKNOWN
                confidence = max_score
        else:
            # No strong signal
            regime = MarketRegime.UNKNOWN
            confidence = max_score

        # Update result with crypto-optimized regime
        from dataclasses import replace
        return replace(result, regime=regime, confidence=confidence)


def detect_crypto_regime(features_df, current_idx=None):
    """
    Convenience function for crypto regime detection.

    Usage:
        from src.cloud.training.models.crypto_regime_detector import detect_crypto_regime

        regime_result = detect_crypto_regime(features_df, current_idx)
        print(f"Regime: {regime_result.regime.value}, Confidence: {regime_result.confidence:.2f}")
    """
    detector = CryptoRegimeDetector()
    return detector.detect_regime(features_df, current_idx)
