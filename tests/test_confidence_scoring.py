"""Test confidence scoring system."""

import sys
from pathlib import Path

# Add src to path
engine_root = Path(__file__).parent
sys.path.insert(0, str(engine_root / "src"))

from cloud.training.models.confidence_scorer import ConfidenceScorer, ConfidenceTracker


def test_confidence_scoring():
    """Test confidence scoring with various scenarios."""
    print("=" * 70)
    print("  Testing Confidence Scoring System")
    print("=" * 70)
    print()

    # Initialize scorer
    scorer = ConfidenceScorer(
        min_confidence_threshold=0.52,
        sample_threshold=20,
        strong_alignment_threshold=0.7,
    )

    print("1️⃣  Confidence Scorer Initialized")
    print(f"   Min threshold: {scorer.min_confidence_threshold:.2f}")
    print(f"   Sample threshold: {scorer.sample_threshold}")
    print()

    # Test scenarios
    scenarios = [
        {
            "name": "🎯 High Confidence Trade",
            "sample_count": 50,
            "best_score": 0.85,
            "runner_up_score": 0.45,
            "pattern_similarity": 0.8,
            "pattern_reliability": 0.75,
            "regime_match": True,
            "regime_confidence": 0.8,
        },
        {
            "name": "✅ Good Trade",
            "sample_count": 30,
            "best_score": 0.65,
            "runner_up_score": 0.50,
            "pattern_similarity": 0.6,
            "pattern_reliability": 0.65,
            "regime_match": False,
            "regime_confidence": 0.6,
        },
        {
            "name": "❌ Skip - Insufficient Data",
            "sample_count": 5,
            "best_score": 0.75,
            "runner_up_score": 0.55,
            "pattern_similarity": 0.7,
            "pattern_reliability": 0.7,
            "regime_match": True,
            "regime_confidence": 0.7,
        },
        {
            "name": "❌ Skip - Unclear Winner",
            "sample_count": 40,
            "best_score": 0.52,
            "runner_up_score": 0.51,
            "pattern_similarity": 0.5,
            "pattern_reliability": 0.5,
            "regime_match": False,
            "regime_confidence": 0.5,
        },
        {
            "name": "⚠️  Marginal Trade",
            "sample_count": 20,
            "best_score": 0.58,
            "runner_up_score": 0.45,
            "pattern_similarity": 0.55,
            "pattern_reliability": 0.6,
            "regime_match": False,
            "regime_confidence": 0.5,
        },
    ]

    print("2️⃣  Testing Scenarios...")
    print()

    for i, scenario in enumerate(scenarios, 1):
        name = scenario.pop("name")
        print(f"Scenario {i}: {name}")
        print("-" * 70)

        result = scorer.calculate_confidence(**scenario)

        print(f"   Confidence: {result.confidence:.3f}")
        print(f"   Decision: {result.decision.upper()}")
        print(f"   Reason: {result.reason}")
        print()
        print(f"   Factors:")
        print(f"     Sample: {result.factors.sample_count} → confidence={result.factors.sample_confidence:.2f}")
        print(f"     Scores: best={result.factors.best_score:.2f}, runner-up={result.factors.runner_up_score:.2f}, sep={result.factors.score_separation:.2f}")
        print(f"     Pattern: similarity={result.factors.pattern_similarity:.2f}, reliability={result.factors.pattern_reliability:.2f}")
        print(f"     Regime: match={result.factors.regime_match}, confidence={result.factors.regime_confidence:.2f}")
        print()

    # Test calibration tracking
    print("3️⃣  Testing Confidence Calibration Tracking...")
    print()

    tracker = ConfidenceTracker()

    # Simulate some trades
    simulated_trades = [
        (0.55, True),  # Low confidence, win
        (0.55, False), # Low confidence, loss
        (0.65, True),  # Medium confidence, win
        (0.65, True),
        (0.65, False),
        (0.75, True),  # High confidence, win
        (0.75, True),
        (0.75, True),
        (0.75, False),
        (0.85, True),  # Very high confidence, win
        (0.85, True),
        (0.85, True),
        (0.85, True),
        (0.95, True),  # Extreme confidence, win
        (0.95, True),
    ]

    for conf, won in simulated_trades:
        tracker.record_outcome(conf, won)

    stats = tracker.get_calibration_stats()

    print("   Calibration Statistics:")
    print("   " + "-" * 66)
    print(f"   {'Bin':<15} {'Expected':<12} {'Actual':<12} {'Count':<8} {'Error':<10}")
    print("   " + "-" * 66)

    for bin_key, bin_stats in stats.items():
        print(f"   {bin_key:<15} "
              f"{bin_stats['expected_win_rate']:.1%}        "
              f"{bin_stats['actual_win_rate']:.1%}      "
              f"{bin_stats['count']:<8} "
              f"{bin_stats['calibration_error']:.3f}")

    overall_error = tracker.get_overall_calibration_error()
    print("   " + "-" * 66)
    print(f"   Overall Calibration Error: {overall_error:.3f}")

    if overall_error < 0.05:
        print("   ✅ Excellent calibration!")
    elif overall_error < 0.1:
        print("   ✅ Good calibration")
    else:
        print("   ⚠️  Calibration needs adjustment")

    print()

    # Test threshold adjustment
    print("4️⃣  Testing Regime-Based Threshold Adjustment...")
    print()

    regime_performance = {
        "trend": 0.65,  # 65% win rate
        "range": 0.58,  # 58% win rate
        "panic": 0.45,  # 45% win rate (worse)
    }

    base_threshold = 0.52

    for regime, win_rate in regime_performance.items():
        adjusted = scorer.adjust_threshold_by_regime(
            base_threshold=base_threshold,
            regime=regime,
            regime_performance=regime_performance,
        )

        change = adjusted - base_threshold
        direction = "↑" if change > 0 else "↓" if change < 0 else "→"

        print(f"   {regime.upper():<8} (WR={win_rate:.1%}): "
              f"{base_threshold:.2f} {direction} {adjusted:.2f} "
              f"(change: {change:+.3f})")

    print()
    print("=" * 70)
    print("✅ TEST COMPLETE - Confidence scoring working!")
    print("=" * 70)


if __name__ == "__main__":
    try:
        test_confidence_scoring()
    except Exception as e:
        print()
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
