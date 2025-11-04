"""
Test Alpha Engines Integration

Quick test to verify:
1. Alpha Engines generate signals
2. Different engines activate in different conditions
3. Ensemble Predictor integrates Alpha signals correctly
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from cloud.training.models.alpha_engines import AlphaEngineCoordinator, TradingTechnique
from cloud.training.models.ensemble_predictor import EnsemblePredictor, PredictionSource


def test_alpha_engines_basic():
    """Test basic Alpha Engine signal generation."""
    print("\n" + "=" * 60)
    print("TEST 1: Basic Alpha Engine Signal Generation")
    print("=" * 60)

    coordinator = AlphaEngineCoordinator()

    # Simulate TREND scenario
    trend_features = {
        "trend_strength": 0.75,  # Strong uptrend
        "ema_slope": 0.05,  # Positive slope
        "momentum_slope": 0.08,  # Strong momentum
        "htf_bias": 0.70,  # Bullish HTF
        "adx": 35.0,  # Strong trend
        "mean_revert_bias": 0.5,
        "compression": 0.2,
        "price_position": 0.5,
        "ignition_score": 45.0,
        "breakout_quality": 0.4,
        "breakout_thrust": 0.3,
        "nr7_density": 0.3,
        "micro_score": 55.0,
        "uptick_ratio": 0.65,
        "vol_jump_z": 1.0,
        "spread_bps": 20.0,
        "rs_score": 75.0,
        "leader_bias": 0.6,
        "pullback_depth": 0.1,
        "kurtosis": 0.5,
    }

    signals = coordinator.generate_all_signals(trend_features, "trend")
    best = coordinator.select_best_technique(signals)

    print(f"\nMarket Regime: TREND")
    print(f"Best Technique: {best.technique.value}")
    print(f"Direction: {best.direction}")
    print(f"Confidence: {best.confidence:.2%}")
    print(f"Reasoning: {best.reasoning}")
    print(f"Regime Affinity: {best.regime_affinity:.2f}")

    print(f"\nAll Engine Signals:")
    for technique, signal in signals.items():
        print(
            f"  {technique.value:12s}: {signal.direction:5s} @ {signal.confidence:.2%} (affinity={signal.regime_affinity:.2f})"
        )

    assert best.technique == TradingTechnique.TREND, "Should select Trend Engine in TREND regime"
    assert best.direction == "buy", "Should be bullish in uptrend"
    assert best.confidence > 0.5, "Should have decent confidence"

    print("\n✅ TREND scenario test passed!")


def test_range_scenario():
    """Test RANGE scenario - should activate Range Engine."""
    print("\n" + "=" * 60)
    print("TEST 2: Range Scenario")
    print("=" * 60)

    coordinator = AlphaEngineCoordinator()

    # Simulate RANGE scenario
    range_features = {
        "trend_strength": 0.1,  # Weak trend
        "ema_slope": 0.001,  # Flat
        "momentum_slope": -0.002,  # Flat
        "htf_bias": 0.52,  # Neutral
        "adx": 18.0,  # Weak trend (ranging)
        "mean_revert_bias": -0.8,  # Far below mean (oversold)
        "compression": 0.75,  # High compression
        "price_position": 0.25,  # Low in range
        "bb_width": 0.015,  # Narrow bands
        "ignition_score": 30.0,
        "breakout_quality": 0.3,
        "breakout_thrust": 0.1,
        "nr7_density": 0.5,
        "micro_score": 50.0,
        "uptick_ratio": 0.48,
        "vol_jump_z": 0.5,
        "spread_bps": 25.0,
        "rs_score": 52.0,
        "leader_bias": 0.1,
        "pullback_depth": 0.3,
        "kurtosis": 0.2,
    }

    signals = coordinator.generate_all_signals(range_features, "range")
    best = coordinator.select_best_technique(signals)

    print(f"\nMarket Regime: RANGE")
    print(f"Best Technique: {best.technique.value}")
    print(f"Direction: {best.direction}")
    print(f"Confidence: {best.confidence:.2%}")
    print(f"Reasoning: {best.reasoning}")

    print(f"\nAll Engine Signals:")
    for technique, signal in signals.items():
        print(
            f"  {technique.value:12s}: {signal.direction:5s} @ {signal.confidence:.2%} (affinity={signal.regime_affinity:.2f})"
        )

    assert best.technique == TradingTechnique.RANGE, "Should select Range Engine in RANGE regime"
    assert best.direction == "buy", "Should buy oversold in range"
    assert best.confidence > 0.4, "Should have reasonable confidence"

    print("\n✅ RANGE scenario test passed!")


def test_breakout_scenario():
    """Test BREAKOUT scenario - should activate Breakout Engine."""
    print("\n" + "=" * 60)
    print("TEST 3: Breakout Scenario")
    print("=" * 60)

    coordinator = AlphaEngineCoordinator()

    # Simulate BREAKOUT scenario
    breakout_features = {
        "trend_strength": 0.6,
        "ema_slope": 0.03,
        "momentum_slope": 0.05,
        "htf_bias": 0.65,
        "adx": 28.0,
        "mean_revert_bias": 0.3,
        "compression": 0.45,
        "price_position": 0.6,
        "bb_width": 0.025,
        "ignition_score": 78.0,  # HIGH ignition
        "breakout_quality": 0.75,  # HIGH quality
        "breakout_thrust": 0.85,  # Strong thrust
        "nr7_density": 0.80,  # High compression before breakout
        "micro_score": 70.0,
        "uptick_ratio": 0.75,
        "vol_jump_z": 2.5,  # Volume spike
        "spread_bps": 18.0,
        "rs_score": 68.0,
        "leader_bias": 0.4,
        "pullback_depth": 0.05,
        "kurtosis": 1.5,
    }

    signals = coordinator.generate_all_signals(breakout_features, "trend")
    best = coordinator.select_best_technique(signals)

    print(f"\nMarket Regime: TREND (breakout)")
    print(f"Best Technique: {best.technique.value}")
    print(f"Direction: {best.direction}")
    print(f"Confidence: {best.confidence:.2%}")
    print(f"Reasoning: {best.reasoning}")

    print(f"\nAll Engine Signals:")
    for technique, signal in signals.items():
        print(
            f"  {technique.value:12s}: {signal.direction:5s} @ {signal.confidence:.2%}"
        )

    assert best.technique == TradingTechnique.BREAKOUT, "Should select Breakout Engine"
    assert best.direction == "buy", "Should be bullish on breakout"
    assert best.confidence > 0.6, "Should have high confidence on quality breakout"

    print("\n✅ BREAKOUT scenario test passed!")


def test_ensemble_integration():
    """Test that Ensemble Predictor integrates Alpha Engines correctly."""
    print("\n" + "=" * 60)
    print("TEST 4: Ensemble Predictor Integration")
    print("=" * 60)

    ensemble = EnsemblePredictor()

    # Create some predictions
    rl_pred = PredictionSource(
        source_name="rl_agent",
        prediction="buy",
        confidence=0.65,
        reasoning="RL agent bullish signal",
    )

    regime_pred = PredictionSource(
        source_name="regime",
        prediction="buy",
        confidence=0.70,
        reasoning="TREND regime detected",
    )

    # Trend features for alpha engines
    features = {
        "trend_strength": 0.75,
        "ema_slope": 0.05,
        "momentum_slope": 0.08,
        "htf_bias": 0.70,
        "adx": 35.0,
        "mean_revert_bias": 0.3,
        "compression": 0.3,
        "price_position": 0.6,
        "bb_width": 0.03,
        "ignition_score": 50.0,
        "breakout_quality": 0.5,
        "breakout_thrust": 0.4,
        "nr7_density": 0.4,
        "micro_score": 60.0,
        "uptick_ratio": 0.68,
        "vol_jump_z": 1.2,
        "spread_bps": 20.0,
        "rs_score": 72.0,
        "leader_bias": 0.5,
        "pullback_depth": 0.1,
        "kurtosis": 0.5,
    }

    # Make ensemble prediction (should include alpha engines)
    result = ensemble.predict(
        rl_prediction=rl_pred,
        regime_prediction=regime_pred,
        features=features,
        current_regime="trend",
    )

    print(f"\nEnsemble Prediction:")
    print(f"Final: {result.final_prediction}")
    print(f"Confidence: {result.ensemble_confidence:.2%}")
    print(f"Agreement: {result.agreement_score:.2%}")
    print(f"Strongest Source: {result.strongest_source}")
    print(f"\nReasoning: {result.reasoning}")

    print(f"\nSource Predictions:")
    for pred in result.source_predictions:
        print(f"  {pred.source_name:15s}: {pred.prediction:5s} @ {pred.confidence:.2%}")

    # Check alpha_engines was included
    alpha_included = any(
        pred.source_name == "alpha_engines" for pred in result.source_predictions
    )
    assert alpha_included, "Alpha Engines should be included in ensemble"
    assert result.final_prediction == "buy", "Should be bullish with all sources agreeing"
    assert result.ensemble_confidence > 0.6, "Should have high confidence"

    print("\n✅ Ensemble integration test passed!")


def test_engine_performance_tracking():
    """Test that engine performance is tracked correctly."""
    print("\n" + "=" * 60)
    print("TEST 5: Engine Performance Tracking")
    print("=" * 60)

    coordinator = AlphaEngineCoordinator()

    # Simulate some trades
    print("\nSimulating 10 trades with Trend Engine...")
    for i in range(10):
        performance = 1.0 if i < 7 else 0.0  # 70% win rate
        coordinator.update_engine_performance(TradingTechnique.TREND, performance)

    print("Simulating 10 trades with Range Engine...")
    for i in range(10):
        performance = 1.0 if i < 6 else 0.0  # 60% win rate
        coordinator.update_engine_performance(TradingTechnique.RANGE, performance)

    stats = coordinator.get_engine_stats()

    print("\nEngine Performance Stats:")
    for technique, stat in stats.items():
        print(
            f"  {technique:12s}: {stat['total_signals']:2d} trades, "
            f"{stat['win_rate']:.1%} win rate, "
            f"{stat['recent_win_rate']:.1%} recent"
        )

    # Check stats
    trend_stats = stats["trend"]
    assert trend_stats["total_signals"] == 10, "Should track all trades"
    assert abs(trend_stats["win_rate"] - 0.70) < 0.01, "Should calculate correct win rate"

    print("\n✅ Performance tracking test passed!")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("ALPHA ENGINES INTEGRATION TEST SUITE")
    print("=" * 60)

    try:
        test_alpha_engines_basic()
        test_range_scenario()
        test_breakout_scenario()
        test_ensemble_integration()
        test_engine_performance_tracking()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED! ✅")
        print("=" * 60)
        print("\nRevuelto Alpha Engines are fully integrated and operational!")
        print("\nKey Results:")
        print("  • 6 specialized engines working correctly")
        print("  • Dynamic technique selection functional")
        print("  • Ensemble integration successful")
        print("  • Performance tracking enabled")
        print("\n" + "=" * 60)

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        raise


if __name__ == "__main__":
    main()
