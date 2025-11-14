"""
Stress Tester

Runs all stress test scenarios and evaluates model performance.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import structlog

from .scenarios import (
    StressScenario,
    FlashCrashScenario,
    StuckPositionScenario,
    PartialFillScenario,
    ExchangeHaltScenario,
    FundingFlipScenario,
    LiquidityEvaporationScenario,
    FeeSpikeScenario,
    DataGapScenario
)

logger = structlog.get_logger(__name__)


@dataclass
class ScenarioResult:
    """Result from a single stress test scenario"""
    scenario_name: str
    passed: bool
    baseline_metrics: Dict[str, float]
    stress_metrics: Dict[str, float]
    degradation_pct: float  # How much performance degraded
    error: Optional[str] = None


@dataclass
class StressTestResults:
    """
    Complete stress test results

    Contains results from all scenarios and overall pass/fail.
    """
    timestamp: datetime
    all_passed: bool
    num_tests: int
    num_passed: int
    num_failed: int
    scenario_results: List[ScenarioResult]

    def get_failed_scenarios(self) -> List[ScenarioResult]:
        """Get list of failed scenarios"""
        return [r for r in self.scenario_results if not r.passed]

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "all_passed": self.all_passed,
            "summary": {
                "num_tests": self.num_tests,
                "num_passed": self.num_passed,
                "num_failed": self.num_failed
            },
            "scenarios": [
                {
                    "name": r.scenario_name,
                    "passed": r.passed,
                    "degradation_pct": r.degradation_pct,
                    "baseline_sharpe": r.baseline_metrics.get('sharpe_ratio'),
                    "stress_sharpe": r.stress_metrics.get('sharpe_ratio'),
                    "error": r.error
                }
                for r in self.scenario_results
            ]
        }


class StressTester:
    """
    Stress Tester

    Runs models through extreme market scenarios to test robustness.

    A model must pass ALL stress tests to be publishable.

    Example:
        tester = StressTester()

        results = tester.run_all_tests(
            model=model,
            test_data=test_df,
            baseline_metrics={"sharpe_ratio": 1.5, "max_drawdown_pct": 12}
        )

        if not results.all_passed:
            failed = results.get_failed_scenarios()
            for f in failed:
                print(f"FAILED: {f.scenario_name} - {f.error}")

            raise ModelValidationError("Model failed stress tests!")
    """

    def __init__(self, scenarios: Optional[List[StressScenario]] = None):
        """
        Initialize stress tester

        Args:
            scenarios: List of stress scenarios (None = all default scenarios)
        """
        if scenarios is None:
            # Default: All stress scenarios
            self.scenarios = [
                FlashCrashScenario(crash_pct=0.20, recovery_pct=0.10),
                StuckPositionScenario(stuck_duration_candles=6),
                PartialFillScenario(fill_ratio=0.3),
                ExchangeHaltScenario(halt_duration_candles=12),
                FundingFlipScenario(funding_swing_bps=100),
                LiquidityEvaporationScenario(spread_multiplier=10.0),
                FeeSpikeScenario(fee_multiplier=10.0),
                DataGapScenario(gap_duration_candles=6)
            ]
        else:
            self.scenarios = scenarios

    def run_all_tests(
        self,
        model: Any,
        test_data: pd.DataFrame,
        baseline_metrics: Dict[str, float],
        model_predict_fn: Optional[callable] = None
    ) -> StressTestResults:
        """
        Run all stress tests

        Args:
            model: Model to test
            test_data: Test data (will be modified per scenario)
            baseline_metrics: Baseline performance metrics
            model_predict_fn: Optional custom prediction function

        Returns:
            StressTestResults
        """
        logger.info(
            "starting_stress_tests",
            num_scenarios=len(self.scenarios),
            test_data_rows=len(test_data)
        )

        scenario_results = []

        for scenario in self.scenarios:
            logger.info("running_stress_scenario", scenario=scenario.name)

            try:
                # Apply scenario to data
                stress_data = scenario.apply(test_data.copy())

                # Run model on stressed data
                stress_metrics = self._evaluate_model(
                    model,
                    stress_data,
                    model_predict_fn
                )

                # Check if passed
                passed = scenario.evaluate(baseline_metrics, stress_metrics)

                # Calculate performance degradation
                baseline_sharpe = baseline_metrics.get('sharpe_ratio', 0)
                stress_sharpe = stress_metrics.get('sharpe_ratio', 0)

                if baseline_sharpe > 0:
                    degradation_pct = ((baseline_sharpe - stress_sharpe) / baseline_sharpe) * 100
                else:
                    degradation_pct = 0.0

                result = ScenarioResult(
                    scenario_name=scenario.name,
                    passed=passed,
                    baseline_metrics=baseline_metrics,
                    stress_metrics=stress_metrics,
                    degradation_pct=degradation_pct
                )

                logger.info(
                    "stress_scenario_complete",
                    scenario=scenario.name,
                    passed=passed,
                    degradation_pct=degradation_pct
                )

            except Exception as e:
                # Scenario failed with error
                logger.error(
                    "stress_scenario_error",
                    scenario=scenario.name,
                    error=str(e)
                )

                result = ScenarioResult(
                    scenario_name=scenario.name,
                    passed=False,
                    baseline_metrics=baseline_metrics,
                    stress_metrics={},
                    degradation_pct=100.0,
                    error=str(e)
                )

            scenario_results.append(result)

        # Calculate overall results
        num_passed = sum(1 for r in scenario_results if r.passed)
        num_failed = len(scenario_results) - num_passed
        all_passed = num_failed == 0

        results = StressTestResults(
            timestamp=datetime.utcnow(),
            all_passed=all_passed,
            num_tests=len(scenario_results),
            num_passed=num_passed,
            num_failed=num_failed,
            scenario_results=scenario_results
        )

        logger.info(
            "stress_tests_complete",
            all_passed=all_passed,
            num_passed=num_passed,
            num_failed=num_failed
        )

        return results

    def _evaluate_model(
        self,
        model: Any,
        data: pd.DataFrame,
        predict_fn: Optional[callable]
    ) -> Dict[str, float]:
        """
        Evaluate model on data and return metrics

        Args:
            model: Model to evaluate
            data: Data to evaluate on
            predict_fn: Optional custom prediction function

        Returns:
            Dict of metrics
        """
        # This is a simplified evaluation - in practice, integrate with
        # your existing backtest/evaluation framework

        try:
            if predict_fn:
                predictions = predict_fn(model, data)
            elif hasattr(model, 'predict'):
                # Get features (assuming numeric columns except price columns)
                feature_cols = [c for c in data.columns if c not in ['open', 'high', 'low', 'close', 'timestamp']]
                features = data[feature_cols].fillna(0)
                predictions = model.predict(features)
            else:
                # Can't evaluate
                return {
                    'sharpe_ratio': 0.0,
                    'max_drawdown_pct': 100.0,
                    'pnl_bps': 0.0,
                    'trades_oos': 0
                }

            # Simple backtest
            if 'close' in data.columns:
                returns = data['close'].pct_change()

                # Assume predictions are signals (-1, 0, 1)
                if hasattr(predictions, 'flatten'):
                    signals = predictions.flatten()
                else:
                    signals = predictions

                # Strategy returns
                strategy_returns = returns * signals

                # Calculate metrics
                sharpe = self._calculate_sharpe(strategy_returns)
                drawdown = self._calculate_max_drawdown(strategy_returns)
                total_pnl_bps = strategy_returns.sum() * 10000
                num_trades = (signals != 0).sum()

                return {
                    'sharpe_ratio': sharpe,
                    'max_drawdown_pct': drawdown * 100,
                    'pnl_bps': total_pnl_bps,
                    'trades_oos': num_trades
                }

        except Exception as e:
            logger.error("model_evaluation_failed", error=str(e))

        # Return default poor metrics on error
        return {
            'sharpe_ratio': 0.0,
            'max_drawdown_pct': 100.0,
            'pnl_bps': -1000.0,
            'trades_oos': 0
        }

    def _calculate_sharpe(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio"""
        if returns.std() == 0:
            return 0.0
        return (returns.mean() / returns.std()) * (252 ** 0.5)  # Annualized

    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.expanding().max()
        drawdown = (cum_returns - running_max) / running_max
        return abs(drawdown.min())

    def generate_report(self, results: StressTestResults) -> str:
        """
        Generate human-readable stress test report

        Args:
            results: StressTestResults

        Returns:
            Report string
        """
        report_lines = [
            "=" * 80,
            "STRESS TEST REPORT",
            "=" * 80,
            "",
            f"Overall: {'✅ PASS' if results.all_passed else '❌ FAIL'}",
            f"Passed: {results.num_passed}/{results.num_tests}",
            f"Failed: {results.num_failed}/{results.num_tests}",
            "",
            "SCENARIO RESULTS:",
            "-" * 80,
        ]

        for result in results.scenario_results:
            status = "✅ PASS" if result.passed else "❌ FAIL"

            baseline_sharpe = result.baseline_metrics.get('sharpe_ratio', 0)
            stress_sharpe = result.stress_metrics.get('sharpe_ratio', 0)

            report_lines.append(
                f"{status} {result.scenario_name}: "
                f"Sharpe {baseline_sharpe:.2f} → {stress_sharpe:.2f} "
                f"({result.degradation_pct:+.1f}% degradation)"
            )

            if result.error:
                report_lines.append(f"     Error: {result.error}")

        report_lines.append("")
        report_lines.append("=" * 80)

        if not results.all_passed:
            report_lines.append("")
            report_lines.append("⚠️  MODEL MUST PASS ALL STRESS TESTS TO BE PUBLISHABLE")
            report_lines.append("")

        return "\n".join(report_lines)
