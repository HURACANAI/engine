"""
Individual Gate Implementations

Each gate checks specific quality criteria.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class GateResult:
    """Result from a single gate evaluation"""
    gate_name: str
    passed: bool
    score: Optional[float] = None  # Optional score [0-1]
    reason: str = ""
    details: Dict[str, Any] = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}


class BaseGate(ABC):
    """Base class for all gates"""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def evaluate(self, **kwargs) -> GateResult:
        """
        Evaluate gate

        Returns:
            GateResult
        """
        pass


class PerformanceGate(BaseGate):
    """
    Performance Gate

    Checks if model meets minimum performance thresholds.

    Requirements:
    - Sharpe ratio >= min_sharpe
    - Win rate >= min_win_rate
    - Profit factor >= min_profit_factor
    """

    def __init__(
        self,
        min_sharpe: float = 1.0,
        min_win_rate: float = 0.50,
        min_profit_factor: float = 1.5
    ):
        super().__init__("Performance")
        self.min_sharpe = min_sharpe
        self.min_win_rate = min_win_rate
        self.min_profit_factor = min_profit_factor

    def evaluate(
        self,
        sharpe_ratio: float,
        win_rate: float,
        profit_factor: float,
        **kwargs
    ) -> GateResult:
        """
        Evaluate performance gate

        Args:
            sharpe_ratio: Sharpe ratio
            win_rate: Win rate [0-1]
            profit_factor: Profit factor (gross profit / gross loss)

        Returns:
            GateResult
        """
        failures = []

        if sharpe_ratio < self.min_sharpe:
            failures.append(
                f"Sharpe {sharpe_ratio:.2f} < {self.min_sharpe:.2f}"
            )

        if win_rate < self.min_win_rate:
            failures.append(
                f"Win rate {win_rate:.2%} < {self.min_win_rate:.2%}"
            )

        if profit_factor < self.min_profit_factor:
            failures.append(
                f"Profit factor {profit_factor:.2f} < {self.min_profit_factor:.2f}"
            )

        passed = len(failures) == 0

        if passed:
            reason = (
                f"✅ Sharpe={sharpe_ratio:.2f}, "
                f"WinRate={win_rate:.2%}, "
                f"PF={profit_factor:.2f}"
            )
        else:
            reason = " | ".join(failures)

        return GateResult(
            gate_name=self.name,
            passed=passed,
            score=(sharpe_ratio / 3.0) if passed else 0.0,  # Normalized score
            reason=reason,
            details={
                "sharpe_ratio": sharpe_ratio,
                "win_rate": win_rate,
                "profit_factor": profit_factor
            }
        )


class RiskGate(BaseGate):
    """
    Risk Gate

    Checks if model risk metrics are acceptable.

    Requirements:
    - Max drawdown <= max_acceptable_drawdown
    - Volatility <= max_acceptable_volatility
    - Max consecutive losses <= max_consecutive_losses
    """

    def __init__(
        self,
        max_acceptable_drawdown: float = 15.0,
        max_acceptable_volatility: float = 30.0,
        max_consecutive_losses: int = 5
    ):
        super().__init__("Risk")
        self.max_acceptable_drawdown = max_acceptable_drawdown
        self.max_acceptable_volatility = max_acceptable_volatility
        self.max_consecutive_losses = max_consecutive_losses

    def evaluate(
        self,
        max_drawdown_pct: float,
        volatility_pct: float,
        max_consecutive_losses: int,
        **kwargs
    ) -> GateResult:
        """
        Evaluate risk gate

        Args:
            max_drawdown_pct: Maximum drawdown percentage
            volatility_pct: Annualized volatility percentage
            max_consecutive_losses: Max consecutive losing trades

        Returns:
            GateResult
        """
        failures = []

        if max_drawdown_pct > self.max_acceptable_drawdown:
            failures.append(
                f"Drawdown {max_drawdown_pct:.1f}% > {self.max_acceptable_drawdown:.1f}%"
            )

        if volatility_pct > self.max_acceptable_volatility:
            failures.append(
                f"Volatility {volatility_pct:.1f}% > {self.max_acceptable_volatility:.1f}%"
            )

        if max_consecutive_losses > self.max_consecutive_losses:
            failures.append(
                f"Consecutive losses {max_consecutive_losses} > {self.max_consecutive_losses}"
            )

        passed = len(failures) == 0

        if passed:
            reason = (
                f"✅ DD={max_drawdown_pct:.1f}%, "
                f"Vol={volatility_pct:.1f}%, "
                f"MaxLoss={max_consecutive_losses}"
            )
        else:
            reason = " | ".join(failures)

        return GateResult(
            gate_name=self.name,
            passed=passed,
            score=1.0 - (max_drawdown_pct / 100.0) if passed else 0.0,
            reason=reason,
            details={
                "max_drawdown_pct": max_drawdown_pct,
                "volatility_pct": volatility_pct,
                "max_consecutive_losses": max_consecutive_losses
            }
        )


class CalibrationGate(BaseGate):
    """
    Calibration Gate

    Checks if model confidence scores are well-calibrated.

    Requirements:
    - Brier score <= max_brier
    - ECE <= max_ece
    """

    def __init__(
        self,
        max_brier: float = 0.25,
        max_ece: float = 0.10
    ):
        super().__init__("Calibration")
        self.max_brier = max_brier
        self.max_ece = max_ece

    def evaluate(
        self,
        brier_score: float,
        ece: float,
        **kwargs
    ) -> GateResult:
        """
        Evaluate calibration gate

        Args:
            brier_score: Brier score
            ece: Expected Calibration Error

        Returns:
            GateResult
        """
        failures = []

        if brier_score > self.max_brier:
            failures.append(
                f"Brier {brier_score:.3f} > {self.max_brier:.3f}"
            )

        if ece > self.max_ece:
            failures.append(
                f"ECE {ece:.3f} > {self.max_ece:.3f}"
            )

        passed = len(failures) == 0

        if passed:
            reason = f"✅ Brier={brier_score:.3f}, ECE={ece:.3f}"
        else:
            reason = " | ".join(failures)

        return GateResult(
            gate_name=self.name,
            passed=passed,
            score=1.0 - brier_score if passed else 0.0,
            reason=reason,
            details={
                "brier_score": brier_score,
                "ece": ece
            }
        )


class StressTestGate(BaseGate):
    """
    Stress Test Gate

    Checks if model passes stress test scenarios.

    Requirements:
    - Must pass ALL stress test scenarios
    """

    def __init__(self, required_pass_rate: float = 1.0):
        super().__init__("StressTest")
        self.required_pass_rate = required_pass_rate

    def evaluate(
        self,
        stress_test_results: Dict[str, bool],
        **kwargs
    ) -> GateResult:
        """
        Evaluate stress test gate

        Args:
            stress_test_results: Dict of {scenario_name: passed}

        Returns:
            GateResult
        """
        if len(stress_test_results) == 0:
            return GateResult(
                gate_name=self.name,
                passed=False,
                score=0.0,
                reason="No stress test results provided"
            )

        total_tests = len(stress_test_results)
        passed_tests = sum(1 for result in stress_test_results.values() if result)

        pass_rate = passed_tests / total_tests

        passed = pass_rate >= self.required_pass_rate

        if passed:
            reason = f"✅ Passed {passed_tests}/{total_tests} stress tests"
        else:
            failed_scenarios = [
                name for name, result in stress_test_results.items()
                if not result
            ]
            reason = f"❌ Failed scenarios: {', '.join(failed_scenarios)}"

        return GateResult(
            gate_name=self.name,
            passed=passed,
            score=pass_rate,
            reason=reason,
            details={
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "pass_rate": pass_rate,
                "results": stress_test_results
            }
        )


class LeakageGate(BaseGate):
    """
    Leakage Gate

    Checks for data leakage issues.

    Requirements:
    - No leakage detected
    """

    def __init__(self):
        super().__init__("Leakage")

    def evaluate(
        self,
        leakage_detected: bool,
        leakage_issues: Optional[list] = None,
        **kwargs
    ) -> GateResult:
        """
        Evaluate leakage gate

        Args:
            leakage_detected: Whether leakage was detected
            leakage_issues: List of leakage issues found

        Returns:
            GateResult
        """
        passed = not leakage_detected

        if passed:
            reason = "✅ No leakage detected"
        else:
            if leakage_issues:
                reason = f"❌ Leakage detected: {', '.join(leakage_issues)}"
            else:
                reason = "❌ Leakage detected"

        return GateResult(
            gate_name=self.name,
            passed=passed,
            score=1.0 if passed else 0.0,
            reason=reason,
            details={
                "leakage_detected": leakage_detected,
                "leakage_issues": leakage_issues or []
            }
        )


class DriftGate(BaseGate):
    """
    Drift Gate

    Checks for critical data drift.

    Requirements:
    - No critical drift detected
    """

    def __init__(self, max_acceptable_psi: float = 0.2):
        super().__init__("Drift")
        self.max_acceptable_psi = max_acceptable_psi

    def evaluate(
        self,
        critical_drift: bool,
        max_psi: float,
        drifted_features: Optional[list] = None,
        **kwargs
    ) -> GateResult:
        """
        Evaluate drift gate

        Args:
            critical_drift: Whether critical drift detected
            max_psi: Maximum PSI value observed
            drifted_features: List of drifted feature names

        Returns:
            GateResult
        """
        # Critical drift automatically fails
        if critical_drift:
            passed = False
        else:
            # Check PSI threshold
            passed = max_psi <= self.max_acceptable_psi

        if passed:
            reason = f"✅ Max PSI={max_psi:.3f}"
        else:
            if drifted_features:
                reason = (
                    f"❌ Critical drift detected: "
                    f"{', '.join(drifted_features)} (PSI={max_psi:.3f})"
                )
            else:
                reason = f"❌ PSI {max_psi:.3f} > {self.max_acceptable_psi:.3f}"

        return GateResult(
            gate_name=self.name,
            passed=passed,
            score=1.0 - min(max_psi, 1.0) if passed else 0.0,
            reason=reason,
            details={
                "critical_drift": critical_drift,
                "max_psi": max_psi,
                "drifted_features": drifted_features or []
            }
        )


class LiveConsistencyGate(BaseGate):
    """
    Live Consistency Gate

    Checks if backtest performance is consistent with live performance.

    Requirements:
    - Live Sharpe >= backtest_sharpe * min_consistency_ratio
    """

    def __init__(self, min_consistency_ratio: float = 0.7):
        super().__init__("LiveConsistency")
        self.min_consistency_ratio = min_consistency_ratio

    def evaluate(
        self,
        backtest_sharpe: float,
        live_sharpe: Optional[float] = None,
        **kwargs
    ) -> GateResult:
        """
        Evaluate live consistency gate

        Args:
            backtest_sharpe: Backtest Sharpe ratio
            live_sharpe: Live Sharpe ratio (None if no live data yet)

        Returns:
            GateResult
        """
        if live_sharpe is None:
            # No live data yet - pass with warning
            return GateResult(
                gate_name=self.name,
                passed=True,
                score=0.5,
                reason="⚠️  No live data yet - unable to verify consistency"
            )

        expected_min_sharpe = backtest_sharpe * self.min_consistency_ratio

        passed = live_sharpe >= expected_min_sharpe

        consistency_ratio = live_sharpe / backtest_sharpe if backtest_sharpe > 0 else 0.0

        if passed:
            reason = (
                f"✅ Live Sharpe {live_sharpe:.2f} vs "
                f"Backtest {backtest_sharpe:.2f} "
                f"({consistency_ratio:.0%} consistency)"
            )
        else:
            reason = (
                f"❌ Live Sharpe {live_sharpe:.2f} < "
                f"Expected {expected_min_sharpe:.2f} "
                f"({consistency_ratio:.0%} of backtest)"
            )

        return GateResult(
            gate_name=self.name,
            passed=passed,
            score=min(consistency_ratio, 1.0),
            reason=reason,
            details={
                "backtest_sharpe": backtest_sharpe,
                "live_sharpe": live_sharpe,
                "consistency_ratio": consistency_ratio
            }
        )
