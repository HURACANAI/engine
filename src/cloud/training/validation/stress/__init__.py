"""
Enhanced Stress Tests

Tests models against extreme market conditions and adversarial scenarios.

Scenarios:
- Flash crashes (-20% in 5 minutes)
- Stuck positions (can't exit for 30 minutes)
- Partial fills (only 30% of order fills)
- Exchange halts (trading paused mid-position)
- Funding rate flips (sudden -0.5% → +0.5%)
- Liquidity evaporation (bid-ask spread 10x normal)
- Fee spikes (10x normal fees)
- Data gaps (30-minute missing data)

Usage:
    from src.cloud.training.validation.stress import StressTester

    tester = StressTester()

    # Run all stress tests
    results = tester.run_all_tests(
        model=model,
        test_data=test_df,
        baseline_metrics=baseline
    )

    if not results.all_passed:
        print(f"Failed {results.num_failed}/{results.num_tests} stress tests")
        raise ModelValidationError("Model failed stress tests!")

    # Model must pass ALL stress tests to be publishable
"""

from .stress_tester import StressTester, StressTestResults, StressScenario
from .scenarios import (
    FlashCrashScenario,
    StuckPositionScenario,
    PartialFillScenario,
    ExchangeHaltScenario,
    FundingFlipScenario,
    LiquidityEvaporationScenario,
    FeeSpikeScenario,
    DataGapScenario
)

__all__ = [
    "StressTester",
    "StressTestResults",
    "StressScenario",
    "FlashCrashScenario",
    "StuckPositionScenario",
    "PartialFillScenario",
    "ExchangeHaltScenario",
    "FundingFlipScenario",
    "LiquidityEvaporationScenario",
    "FeeSpikeScenario",
    "DataGapScenario",
]
