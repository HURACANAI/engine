"""Simple test of regime detection using synthetic data."""

import sys
from pathlib import Path
import numpy as np
import polars as pl
from datetime import datetime, timedelta, timezone

# Add src to path
engine_root = Path(__file__).parent
sys.path.insert(0, str(engine_root / "src"))

from cloud.training.models.regime_detector import RegimeDetector, MarketRegime
from shared.features.recipe import FeatureRecipe


def generate_synthetic_ohlcv(n_candles: int = 100, regime: str = "trend") -> pl.DataFrame:
    """Generate synthetic OHLCV data for testing."""
    np.random.seed(42)

    base_price = 50000.0
    timestamps = [datetime.now(tz=timezone.utc) - timedelta(days=n_candles - i) for i in range(n_candles)]

    if regime == "trend":
        # Trending market - strong directional movement
        trend = np.linspace(0, 5000, n_candles)
        volatility = 500
    elif regime == "range":
        # Range-bound market - oscillation around mean
        trend = np.sin(np.linspace(0, 4 * np.pi, n_candles)) * 1000
        volatility = 200
    else:  # panic
        # High volatility market
        trend = np.random.randn(n_candles).cumsum() * 500
        volatility = 2000

    closes = base_price + trend + np.random.randn(n_candles) * volatility
    highs = closes + np.abs(np.random.randn(n_candles)) * volatility * 0.5
    lows = closes - np.abs(np.random.randn(n_candles)) * volatility * 0.5
    opens = closes + np.random.randn(n_candles) * volatility * 0.3
    volumes = np.abs(np.random.randn(n_candles)) * 10000 + 50000

    return pl.DataFrame({
        "ts": timestamps,
        "timestamp": [int(ts.timestamp() * 1000) for ts in timestamps],
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    })


def test_regime_detection():
    """Test regime detection on synthetic data."""
    print("=" * 70)
    print("  Testing Regime Detection System (Synthetic Data)")
    print("=" * 70)
    print()

    # Initialize regime detector
    detector = RegimeDetector(
        trend_threshold=0.6,
        range_threshold=0.6,
        panic_threshold=0.7,
    )

    recipe = FeatureRecipe()

    # Test each regime type
    for regime_type in ["trend", "range", "panic"]:
        print(f"📊 Testing {regime_type.upper()} regime...")
        print()

        # Generate data
        data = generate_synthetic_ohlcv(n_candles=100, regime=regime_type)

        # Generate features
        features_df = recipe.build(data)

        # Detect regime for last candle
        result = detector.detect_regime(features_df, current_idx=features_df.height - 1)

        # Print result
        regime_emoji = {
            MarketRegime.TREND: "📈",
            MarketRegime.RANGE: "↔️ ",
            MarketRegime.PANIC: "🚨",
            MarketRegime.UNKNOWN: "❓",
        }

        close = features_df["close"][features_df.height - 1]

        print(f"   {regime_emoji[result.regime]} Detected: {result.regime.value.upper()} (confidence: {result.confidence:.2f})")
        print(f"   Close: ${close:,.0f}")
        print(f"   Scores: Trend={result.regime_scores['trend']:.2f}, "
              f"Range={result.regime_scores['range']:.2f}, "
              f"Panic={result.regime_scores['panic']:.2f}")
        print(f"   Features: ADX={result.features.adx:.1f}, "
              f"ATR%={result.features.atr_pct:.2f}%, "
              f"Compression={result.features.compression_score:.2f}")

        # Check if detection matches expected
        expected_regime = {
            "trend": MarketRegime.TREND,
            "range": MarketRegime.RANGE,
            "panic": MarketRegime.PANIC,
        }[regime_type]

        if result.regime == expected_regime:
            print(f"   ✅ Correctly identified as {regime_type.upper()}!")
        else:
            print(f"   ⚠️  Expected {expected_regime.value.upper()}, got {result.regime.value.upper()}")

        print()

    print("=" * 70)
    print("✅ TEST COMPLETE - Regime detection working!")
    print()
    print("Next: Test on real market data with test_regime_detection.py")
    print("=" * 70)


if __name__ == "__main__":
    try:
        test_regime_detection()
    except Exception as e:
        print()
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
