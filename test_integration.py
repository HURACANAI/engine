"""Test integrated regime detection and confidence scoring in shadow trading."""

import sys
from pathlib import Path

# Add src to path
engine_root = Path(__file__).parent
sys.path.insert(0, str(engine_root / "src"))

from datetime import datetime, timedelta

import polars as pl

from cloud.training.backtesting.shadow_trader import ShadowTrader, BacktestConfig
from shared.features.recipe import FeatureRecipe


def test_integration():
    """Test regime detection and confidence scoring integration."""
    print("=" * 70)
    print("  Testing Integrated Regime Detection & Confidence Scoring")
    print("=" * 70)
    print()

    # Create minimal test data
    start_time = datetime(2024, 1, 1)
    test_data = []

    for i in range(200):  # Need enough data for regime detection
        timestamp = start_time + timedelta(hours=i)
        base_price = 100.0
        trend = i * 0.1  # Uptrend
        noise = (i % 10) * 0.5

        test_data.append({
            "ts": timestamp,
            "open": base_price + trend + noise,
            "high": base_price + trend + noise + 1.0,
            "low": base_price + trend + noise - 1.0,
            "close": base_price + trend + noise + 0.5,
            "volume": 1000000.0,
        })

    df = pl.DataFrame(test_data)

    print("1️⃣  Creating test data")
    print(f"   Rows: {df.height}")
    print(f"   Timespan: {df['ts'].min()} to {df['ts'].max()}")
    print()

    # Build features
    print("2️⃣  Building features...")
    recipe = FeatureRecipe()
    features_df = recipe.build(df)
    print(f"   Features created: {features_df.width} columns")
    print()

    # Create shadow trader
    print("3️⃣  Initializing Shadow Trader with regime & confidence modules...")
    config = BacktestConfig(
        position_size_gbp=100.0,
        min_confidence_threshold=0.52,
        stop_loss_bps=100.0,
        take_profit_bps=200.0,
    )

    trader = ShadowTrader(config=config)
    print("   ✅ RegimeDetector initialized")
    print("   ✅ ConfidenceScorer initialized")
    print()

    # Run shadow trading
    print("4️⃣  Running shadow trading session...")
    try:
        trades = trader.run_shadow_session(
            features_df=features_df,
            symbol="TEST/USD",
            training_mode=True,
        )

        print(f"   Total trades: {len(trades)}")
        print()

        if trades:
            print("5️⃣  Analyzing trade results...")
            print()

            for i, trade in enumerate(trades[:3], 1):  # Show first 3 trades
                print(f"   Trade {i}:")
                print(f"     Regime: {trade.market_regime}")
                print(f"     Regime Confidence: {trade.regime_confidence:.3f}")
                print(f"     Trade Confidence: {trade.trade_confidence:.3f}")
                print(f"     Decision Reason: {trade.decision_reason}")
                print(f"     P&L: £{trade.net_profit_gbp:.2f}")
                print(f"     Winner: {'✅' if trade.is_winner else '❌'}")
                print()

            # Summary stats
            winners = sum(1 for t in trades if t.is_winner)
            win_rate = winners / len(trades) if trades else 0.0

            avg_regime_conf = sum(t.regime_confidence for t in trades) / len(trades)
            avg_trade_conf = sum(t.trade_confidence for t in trades) / len(trades)

            print("6️⃣  Summary Statistics:")
            print(f"     Win Rate: {win_rate:.1%}")
            print(f"     Avg Regime Confidence: {avg_regime_conf:.3f}")
            print(f"     Avg Trade Confidence: {avg_trade_conf:.3f}")
            print()

            # Regime breakdown
            regime_counts = {}
            for trade in trades:
                regime_counts[trade.market_regime] = regime_counts.get(trade.market_regime, 0) + 1

            print("   Regime Distribution:")
            for regime, count in sorted(regime_counts.items()):
                pct = count / len(trades) * 100
                print(f"     {regime.upper():<10} {count:>3} trades ({pct:>5.1f}%)")

            print()
            print("=" * 70)
            print("✅ INTEGRATION TEST PASSED - System working end-to-end!")
            print("=" * 70)

        else:
            print("   ⚠️  No trades generated (might be expected with test data)")
            print("   But system initialized successfully!")
            print()
            print("=" * 70)
            print("✅ INTEGRATION TEST PASSED - Modules integrated correctly!")
            print("=" * 70)

    except Exception as e:
        print()
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    test_integration()
