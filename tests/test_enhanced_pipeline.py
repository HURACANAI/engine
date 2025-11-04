"""
Integration Test for Enhanced RL Pipeline

Tests the complete integration of Phase 1 components into the training pipeline.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Dict

import polars as pl
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.cloud.training.pipelines.enhanced_rl_pipeline import EnhancedRLPipeline
from src.cloud.config.settings import EngineSettings


def create_synthetic_market_data(symbol: str, days: int = 100) -> pl.DataFrame:
    """Create synthetic market data for testing."""
    np.random.seed(42)

    # Generate timestamps
    end_time = datetime.now(tz=timezone.utc)
    start_time = end_time - timedelta(days=days)
    timestamps = [start_time + timedelta(days=i) for i in range(days)]

    # Generate price data with realistic movements
    initial_price = 45000.0 if symbol == "BTC/USD" else 100.0
    returns = np.random.normal(0.001, 0.02, days)  # 0.1% daily drift, 2% volatility
    prices = initial_price * np.exp(np.cumsum(returns))

    # Generate OHLC
    data = {
        "ts": timestamps,
        "open": prices * (1 + np.random.uniform(-0.01, 0.01, days)),
        "high": prices * (1 + np.random.uniform(0, 0.02, days)),
        "low": prices * (1 + np.random.uniform(-0.02, 0, days)),
        "close": prices,
        "volume": np.random.uniform(1000, 10000, days),
    }

    return pl.DataFrame(data)


def test_enhanced_pipeline_initialization():
    """Test that enhanced pipeline initializes correctly with all Phase 1 components."""
    print("\n" + "="*80)
    print("TEST 1: Enhanced Pipeline Initialization")
    print("="*80)

    # Create minimal settings
    settings = EngineSettings()

    # Initialize enhanced pipeline
    pipeline = EnhancedRLPipeline(
        settings=settings,
        dsn="postgresql://localhost/test_db",
        enable_advanced_rewards=True,
        enable_higher_order_features=True,
        enable_granger_causality=True,
        enable_regime_prediction=True,
    )

    print("✓ Pipeline initialized successfully")

    # Verify Phase 1 components exist
    assert hasattr(pipeline, "reward_calculator"), "Missing reward calculator"
    assert hasattr(pipeline, "higher_order_builder"), "Missing higher-order builder"
    assert hasattr(pipeline, "granger_detector"), "Missing Granger detector"
    assert hasattr(pipeline, "regime_predictor"), "Missing regime predictor"

    print("✓ All Phase 1 components present")
    print("✓ Advanced rewards enabled")
    print("✓ Higher-order features enabled")
    print("✓ Granger causality enabled")
    print("✓ Regime prediction enabled")

    print("✅ PASSED: Enhanced pipeline initialization")
    return True


def test_enhanced_feature_building():
    """Test enhanced feature building with higher-order features."""
    print("\n" + "="*80)
    print("TEST 2: Enhanced Feature Building")
    print("="*80)

    settings = EngineSettings()

    # Initialize pipeline with only higher-order features
    pipeline = EnhancedRLPipeline(
        settings=settings,
        dsn="postgresql://localhost/test_db",
        enable_advanced_rewards=False,
        enable_higher_order_features=True,
        enable_granger_causality=False,
        enable_regime_prediction=False,
    )

    # Create synthetic data
    data = create_synthetic_market_data("SOL/USD", days=100)
    market_context = {
        "BTC/USD": create_synthetic_market_data("BTC/USD", days=100),
        "ETH/USD": create_synthetic_market_data("ETH/USD", days=100),
        "SOL/USD": create_synthetic_market_data("SOL/USD", days=100),
    }

    print(f"✓ Created synthetic data: {data.height} rows")

    # Build enhanced features
    try:
        enhanced = pipeline._build_enhanced_features(
            data=data,
            symbol="SOL/USD",
            market_context=market_context,
        )

        original_cols = len(data.columns)
        enhanced_cols = len(enhanced.columns)
        added_cols = enhanced_cols - original_cols

        print(f"✓ Original columns: {original_cols}")
        print(f"✓ Enhanced columns: {enhanced_cols}")
        print(f"✓ Added features: {added_cols}")

        assert enhanced_cols > original_cols, "Should add features"
        assert added_cols >= 50, f"Should add at least 50 features, got {added_cols}"

        print("✅ PASSED: Enhanced feature building")
        return True

    except Exception as e:
        print(f"❌ FAILED: Enhanced feature building - {e}")
        import traceback
        traceback.print_exc()
        return False


def test_phase1_stats():
    """Test that Phase 1 statistics are available."""
    print("\n" + "="*80)
    print("TEST 3: Phase 1 Statistics")
    print("="*80)

    settings = EngineSettings()

    pipeline = EnhancedRLPipeline(
        settings=settings,
        dsn="postgresql://localhost/test_db",
        enable_advanced_rewards=True,
        enable_higher_order_features=True,
        enable_granger_causality=True,
        enable_regime_prediction=True,
    )

    # Get Phase 1 stats
    stats = pipeline.get_phase1_stats()

    print(f"✓ Phase 1 stats retrieved:")
    for key, value in stats.items():
        print(f"  - {key}: {value}")

    assert "reward_calculator" in stats, "Missing reward calculator stats"
    assert "causal_graph" in stats, "Missing causal graph stats"
    assert "regime_predictor" in stats, "Missing regime predictor stats"

    print("✅ PASSED: Phase 1 statistics available")
    return True


def test_feature_flags():
    """Test that feature flags work correctly."""
    print("\n" + "="*80)
    print("TEST 4: Feature Flags")
    print("="*80)

    settings = EngineSettings()

    # Test with all features disabled
    pipeline_disabled = EnhancedRLPipeline(
        settings=settings,
        dsn="postgresql://localhost/test_db",
        enable_advanced_rewards=False,
        enable_higher_order_features=False,
        enable_granger_causality=False,
        enable_regime_prediction=False,
    )

    assert not hasattr(pipeline_disabled, "reward_calculator"), "Should not have reward calculator"
    assert not hasattr(pipeline_disabled, "higher_order_builder"), "Should not have higher-order builder"
    assert not hasattr(pipeline_disabled, "granger_detector"), "Should not have Granger detector"
    assert not hasattr(pipeline_disabled, "regime_predictor"), "Should not have regime predictor"

    print("✓ Feature flags work correctly when disabled")

    # Test with selective enablement
    pipeline_selective = EnhancedRLPipeline(
        settings=settings,
        dsn="postgresql://localhost/test_db",
        enable_advanced_rewards=True,
        enable_higher_order_features=False,
        enable_granger_causality=True,
        enable_regime_prediction=False,
    )

    assert hasattr(pipeline_selective, "reward_calculator"), "Should have reward calculator"
    assert not hasattr(pipeline_selective, "higher_order_builder"), "Should not have higher-order builder"
    assert hasattr(pipeline_selective, "granger_detector"), "Should have Granger detector"
    assert not hasattr(pipeline_selective, "regime_predictor"), "Should not have regime predictor"

    print("✓ Feature flags work correctly with selective enablement")

    print("✅ PASSED: Feature flags")
    return True


def test_state_dim_adjustment():
    """Test that state_dim is adjusted for higher-order features."""
    print("\n" + "="*80)
    print("TEST 5: State Dimension Adjustment")
    print("="*80)

    settings = EngineSettings()

    # Without higher-order features
    pipeline_base = EnhancedRLPipeline(
        settings=settings,
        dsn="postgresql://localhost/test_db",
        enable_advanced_rewards=False,
        enable_higher_order_features=False,
        enable_granger_causality=False,
        enable_regime_prediction=False,
    )

    base_state_dim = pipeline_base.agent.state_dim
    print(f"✓ Base state_dim: {base_state_dim}")

    # With higher-order features
    pipeline_enhanced = EnhancedRLPipeline(
        settings=settings,
        dsn="postgresql://localhost/test_db",
        enable_advanced_rewards=False,
        enable_higher_order_features=True,
        enable_granger_causality=False,
        enable_regime_prediction=False,
    )

    enhanced_state_dim = pipeline_enhanced.agent.state_dim
    print(f"✓ Enhanced state_dim: {enhanced_state_dim}")

    assert enhanced_state_dim == 148, f"Enhanced state_dim should be 148, got {enhanced_state_dim}"
    print(f"✓ State dimension correctly adjusted for higher-order features")

    print("✅ PASSED: State dimension adjustment")
    return True


def run_all_tests():
    """Run all integration tests."""
    print("\n" + "="*80)
    print("ENHANCED PIPELINE INTEGRATION TESTS")
    print("="*80)
    print("Testing Phase 1 integration into training pipeline")
    print("="*80)

    results = []

    try:
        results.append(("Initialization", test_enhanced_pipeline_initialization()))
    except Exception as e:
        print(f"❌ FAILED: Initialization - {e}")
        results.append(("Initialization", False))

    try:
        results.append(("Feature Building", test_enhanced_feature_building()))
    except Exception as e:
        print(f"❌ FAILED: Feature Building - {e}")
        results.append(("Feature Building", False))

    try:
        results.append(("Phase1 Stats", test_phase1_stats()))
    except Exception as e:
        print(f"❌ FAILED: Phase1 Stats - {e}")
        results.append(("Phase1 Stats", False))

    try:
        results.append(("Feature Flags", test_feature_flags()))
    except Exception as e:
        print(f"❌ FAILED: Feature Flags - {e}")
        results.append(("Feature Flags", False))

    try:
        results.append(("State Dim", test_state_dim_adjustment()))
    except Exception as e:
        print(f"❌ FAILED: State Dim - {e}")
        results.append(("State Dim", False))

    # Summary
    print("\n" + "="*80)
    print("INTEGRATION TEST SUMMARY")
    print("="*80)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {name}")

    print("="*80)
    print(f"TOTAL: {passed}/{total} tests passed ({100*passed/total:.1f}%)")
    print("="*80)

    if passed == total:
        print("\n🎉 ALL INTEGRATION TESTS PASSED!")
        print("Phase 1 is fully integrated and ready for production.")
        return True
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Review failures above.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
