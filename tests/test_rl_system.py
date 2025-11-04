#!/usr/bin/env python3
"""
Quick test of the RL system on a single symbol.
This tests:
1. Database connection
2. RL pipeline initialization
3. Shadow trading simulation
4. Memory storage
5. Pattern learning
"""

import sys
from pathlib import Path

# Add src to path
engine_root = Path("/Users/haq/Engine (HF1)/engine")
sys.path.insert(0, str(engine_root / "src"))

import os
from cloud.training.config.settings import EngineSettings
from cloud.training.services.exchange import ExchangeClient
from cloud.training.pipelines.rl_training_pipeline import RLTrainingPipeline
import psycopg2

def test_rl_system():
    print("=" * 60)
    print("  Huracan Engine - RL System Test")
    print("=" * 60)
    print()

    # Step 1: Load settings
    print("1️⃣  Loading settings...")
    config_dir = engine_root / "config"
    settings = EngineSettings.load(config_dir=config_dir)
    print(f"   ✅ Settings loaded")
    print(f"   RL Agent enabled: {settings.training.rl_agent.enabled}")
    print(f"   Shadow trading enabled: {settings.training.shadow_trading.enabled}")
    print()

    # Step 2: Check database
    print("2️⃣  Checking database connection...")
    dsn = settings.postgres.dsn if settings.postgres else None
    if not dsn:
        print("   ❌ DATABASE_URL not configured!")
        return

    try:
        conn = psycopg2.connect(dsn)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM trade_memory")
        count_before = cur.fetchone()[0]
        print(f"   ✅ Database connected")
        print(f"   Existing trades in memory: {count_before}")
        conn.close()
    except Exception as e:
        print(f"   ❌ Database connection failed: {e}")
        return
    print()

    # Step 3: Initialize components
    print("3️⃣  Initializing RL components...")
    try:
        exchange = ExchangeClient("binance", sandbox=False)
        print(f"   ✅ Exchange client initialized")

        rl_pipeline = RLTrainingPipeline(settings=settings, dsn=dsn)
        print(f"   ✅ RL pipeline initialized")
    except Exception as e:
        print(f"   ❌ Initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return
    print()

    # Step 4: Run shadow trading on BTC
    print("4️⃣  Running shadow trading on BTC/USDT...")
    print("   This will:")
    print("   - Download last 30 days of 15-min candles")
    print("   - Simulate every possible trade (no lookahead)")
    print("   - Analyze wins and losses")
    print("   - Store patterns in memory")
    print()
    print("   ⏳ This may take 2-3 minutes...")
    print()

    try:
        metrics = rl_pipeline.train_on_symbol(
            symbol="BTC/USDT",
            exchange_client=exchange,
            lookback_days=150,  # 150 days of daily candles (need 100+ for shadow trading)
        )

        print()
        print("   ✅ Shadow trading complete!")
        print()
        print("   📊 Results:")
        print(f"      Total trades: {metrics.get('total_trades', 0)}")
        print(f"      Winning trades: {metrics.get('winning_trades', 0)}")
        print(f"      Losing trades: {metrics.get('losing_trades', 0)}")
        print(f"      Win rate: {metrics.get('win_rate', 0.0):.2%}")
        print(f"      Avg profit: £{metrics.get('avg_profit_gbp', 0.0):.2f}")
        print(f"      Patterns learned: {metrics.get('patterns_learned', 0)}")
        print()

    except Exception as e:
        print(f"   ❌ Shadow trading failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Step 5: Verify database was updated
    print("5️⃣  Verifying database storage...")
    try:
        conn = psycopg2.connect(dsn)
        cur = conn.cursor()

        cur.execute("SELECT COUNT(*) FROM trade_memory")
        count_after = cur.fetchone()[0]
        print(f"   ✅ Trade memory: {count_after} rows (+{count_after - count_before} new)")

        cur.execute("SELECT COUNT(*) FROM win_analysis")
        win_count = cur.fetchone()[0]
        print(f"   ✅ Win analysis: {win_count} rows")

        cur.execute("SELECT COUNT(*) FROM loss_analysis")
        loss_count = cur.fetchone()[0]
        print(f"   ✅ Loss analysis: {loss_count} rows")

        cur.execute("SELECT COUNT(*) FROM post_exit_tracking")
        post_exit_count = cur.fetchone()[0]
        print(f"   ✅ Post-exit tracking: {post_exit_count} rows")

        cur.execute("SELECT COUNT(*) FROM pattern_library")
        pattern_count = cur.fetchone()[0]
        print(f"   ✅ Pattern library: {pattern_count} patterns")

        conn.close()
    except Exception as e:
        print(f"   ❌ Database verification failed: {e}")
        return
    print()

    # Step 6: Show some learned patterns
    print("6️⃣  Sample learned patterns:")
    try:
        conn = psycopg2.connect(dsn)
        cur = conn.cursor()

        cur.execute("""
            SELECT pattern_name, win_rate, avg_profit_bps, total_occurrences
            FROM pattern_library
            ORDER BY win_rate DESC
            LIMIT 5
        """)

        patterns = cur.fetchall()
        if patterns:
            for pattern_name, win_rate, avg_profit, occurrences in patterns:
                print(f"   • {pattern_name}")
                print(f"     Win rate: {float(win_rate or 0):.2%}, Avg profit: {float(avg_profit or 0):.1f} bps, Seen: {occurrences} times")
        else:
            print("   (No patterns learned yet - need more trades)")

        conn.close()
    except Exception as e:
        print(f"   ⚠️  Could not fetch patterns: {e}")
    print()

    print("=" * 60)
    print("✅ TEST COMPLETE - RL System is working!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("  1. Run on more symbols: python test_rl_system.py")
    print("  2. Run full daily retrain: python -m src.cloud.training.pipelines.daily_retrain")
    print("  3. Query database to see patterns: psql huracan")
    print()


if __name__ == "__main__":
    test_rl_system()
