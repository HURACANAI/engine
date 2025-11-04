#!/usr/bin/env python3
"""
Verify that the Huracan Engine is fully operational.
Run this to check all systems before training.
"""

import sys
from pathlib import Path

engine_root = Path("/Users/haq/Engine (HF1)/engine")
sys.path.insert(0, str(engine_root / "src"))

def verify_system():
    print("=" * 70)
    print("  🔍 HURACAN ENGINE - SYSTEM VERIFICATION")
    print("=" * 70)
    print()

    checks_passed = 0
    checks_total = 0

    # Check 1: Dependencies
    print("1️⃣  Checking dependencies...")
    checks_total += 1
    try:
        import torch
        import psycopg2
        import psutil
        print(f"   ✅ torch {torch.__version__}")
        print(f"   ✅ psycopg2 {psycopg2.__version__}")
        print(f"   ✅ psutil {psutil.__version__}")
        checks_passed += 1
    except Exception as e:
        print(f"   ❌ Missing dependencies: {e}")
    print()

    # Check 2: Configuration
    print("2️⃣  Checking configuration...")
    checks_total += 1
    try:
        from cloud.training.config.settings import EngineSettings
        config_dir = engine_root / "config"
        settings = EngineSettings.load(config_dir=config_dir)
        print(f"   ✅ Settings loaded")
        print(f"   RL Agent enabled: {settings.training.rl_agent.enabled}")
        print(f"   Shadow trading enabled: {settings.training.shadow_trading.enabled}")
        print(f"   Monitoring enabled: {settings.training.monitoring.enabled}")
        checks_passed += 1
    except Exception as e:
        print(f"   ❌ Configuration error: {e}")
    print()

    # Check 3: Database
    print("3️⃣  Checking database...")
    checks_total += 1
    try:
        import psycopg2
        dsn = settings.postgres.dsn if settings.postgres else None
        if not dsn:
            print("   ❌ DATABASE_URL not configured")
        else:
            conn = psycopg2.connect(dsn)
            cur = conn.cursor()

            # Check each table
            tables = ['trade_memory', 'post_exit_tracking', 'win_analysis',
                     'loss_analysis', 'pattern_library', 'model_performance']

            all_tables_exist = True
            for table in tables:
                cur.execute(f"SELECT COUNT(*) FROM {table}")
                count = cur.fetchone()[0]
                print(f"   ✅ {table}: {count} rows")

            conn.close()
            checks_passed += 1
    except Exception as e:
        print(f"   ❌ Database error: {e}")
    print()

    # Check 4: RL Components
    print("4️⃣  Checking RL components...")
    checks_total += 1
    try:
        from cloud.training.memory.store import MemoryStore
        from cloud.training.agents.rl_agent import RLTradingAgent
        from cloud.training.pipelines.rl_training_pipeline import RLTrainingPipeline
        from cloud.training.monitoring.health_monitor import HealthMonitorOrchestrator
        print("   ✅ MemoryStore")
        print("   ✅ RLTradingAgent")
        print("   ✅ RLTrainingPipeline")
        print("   ✅ HealthMonitorOrchestrator")
        checks_passed += 1
    except Exception as e:
        print(f"   ❌ Import error: {e}")
        import traceback
        traceback.print_exc()
    print()

    # Check 5: PostgreSQL Service
    print("5️⃣  Checking PostgreSQL service...")
    checks_total += 1
    try:
        import subprocess
        result = subprocess.run(['pg_isready'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"   ✅ PostgreSQL is running")
            checks_passed += 1
        else:
            print(f"   ❌ PostgreSQL not responding")
    except Exception as e:
        print(f"   ⚠️  Could not check PostgreSQL: {e}")
    print()

    # Summary
    print("=" * 70)
    if checks_passed == checks_total:
        print("✅ ALL CHECKS PASSED - SYSTEM READY FOR PRODUCTION!")
    else:
        print(f"⚠️  {checks_passed}/{checks_total} checks passed - Review issues above")
    print("=" * 70)
    print()

    if checks_passed == checks_total:
        print("🚀 Next steps:")
        print("   1. Run test: python test_rl_system.py")
        print("   2. Run full training: python -m src.cloud.training.pipelines.daily_retrain")
        print("   3. Query database: psql postgresql://haq@localhost:5432/huracan")
    else:
        print("🔧 Fix the issues above before running the system.")
    print()

    return checks_passed == checks_total

if __name__ == "__main__":
    success = verify_system()
    sys.exit(0 if success else 1)
