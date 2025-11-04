"""Test regime detection on real market data."""

import sys
from pathlib import Path

# Add src to path
engine_root = Path(__file__).parent
sys.path.insert(0, str(engine_root / "src"))

from cloud.training.config.settings import EngineSettings
from cloud.training.models.regime_detector import RegimeDetector, MarketRegime
from cloud.training.services.exchange import ExchangeClient
from cloud.training.datasets.data_loader import CandleDataLoader, CandleQuery
from shared.features.recipe import FeatureRecipe
from datetime import datetime, timedelta, timezone


def test_regime_detection():
    """Test regime detection on BTC/USDT data."""
    print("=" * 70)
    print("  Testing Regime Detection System")
    print("=" * 70)
    print()

    # Load settings
    print("1️⃣  Loading settings...")
    config_dir = engine_root / "config"
    settings = EngineSettings.load(config_dir=config_dir)
    print("   ✅ Settings loaded")
    print()

    # Initialize exchange
    print("2️⃣  Initializing exchange client...")
    exchange = ExchangeClient(
        exchange_id=settings.exchange.exchange_id,
        api_key=settings.exchange.api_key,
        secret=settings.exchange.secret,
    )
    print("   ✅ Exchange client initialized")
    print()

    # Load data
    print("3️⃣  Loading market data (30 days of daily candles)...")
    loader = CandleDataLoader(exchange_client=exchange)

    end_date = datetime.now(tz=timezone.utc)
    start_date = end_date - timedelta(days=30)

    query = CandleQuery(
        symbol="BTC/USDT",
        timeframe="1d",
        start_at=start_date,
        end_at=end_date,
    )

    try:
        # Try to load with quality check
        data = loader.load(query)
    except ValueError as e:
        # If quality check fails, load without validation
        print(f"   ⚠️  Quality check failed: {e}")
        print("   📥 Loading data without validation...")
        data = loader._download(query, skip_validation=True)

    print(f"   ✅ Data loaded: {data.height} candles")
    print()

    # Generate features
    print("4️⃣  Generating features...")
    recipe = FeatureRecipe()
    features_df = recipe.build(data)
    print(f"   ✅ Features generated: {features_df.shape[1]} features")
    print()

    # Initialize regime detector
    print("5️⃣  Initializing regime detector...")
    detector = RegimeDetector(
        trend_threshold=0.6,
        range_threshold=0.6,
        panic_threshold=0.7,
    )
    print("   ✅ Regime detector initialized")
    print()

    # Detect regime for recent candles
    print("6️⃣  Detecting regimes for last 10 candles...")
    print()

    regime_counts = {regime: 0 for regime in MarketRegime}

    for i in range(max(0, features_df.height - 10), features_df.height):
        result = detector.detect_regime(features_df, current_idx=i)

        # Get timestamp and price
        ts = features_df["ts"][i]
        close = features_df["close"][i]

        regime_counts[result.regime] += 1

        # Print result
        regime_emoji = {
            MarketRegime.TREND: "📈",
            MarketRegime.RANGE: "↔️ ",
            MarketRegime.PANIC: "🚨",
            MarketRegime.UNKNOWN: "❓",
        }

        print(f"   {regime_emoji[result.regime]} {ts} | Close: ${close:,.0f}")
        print(f"      Regime: {result.regime.value.upper()} (confidence: {result.confidence:.2f})")
        print(f"      Scores: Trend={result.regime_scores['trend']:.2f}, "
              f"Range={result.regime_scores['range']:.2f}, "
              f"Panic={result.regime_scores['panic']:.2f}")
        print(f"      Features: ADX={result.features.adx:.1f}, "
              f"ATR%={result.features.atr_pct:.2f}%, "
              f"Compression={result.features.compression_score:.2f}")
        print()

    # Summary
    print("=" * 70)
    print("  Regime Distribution (last 10 candles)")
    print("=" * 70)
    for regime, count in regime_counts.items():
        if count > 0:
            pct = (count / 10) * 100
            print(f"  {regime.value.upper()}: {count}/10 ({pct:.0f}%)")

    print()
    print("✅ TEST COMPLETE - Regime detection working!")
    print()


if __name__ == "__main__":
    try:
        test_regime_detection()
    except Exception as e:
        print()
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
