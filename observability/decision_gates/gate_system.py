"""
Gate System

Orchestrates all decision gates for model approval.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

import structlog

from .gates import (
    BaseGate,
    PerformanceGate,
    RiskGate,
    CalibrationGate,
    StressTestGate,
    LeakageGate,
    DriftGate,
    LiveConsistencyGate,
    GateResult
)

logger = structlog.get_logger(__name__)


class GateName(str, Enum):
    """Gate names"""
    PERFORMANCE = "Performance"
    RISK = "Risk"
    CALIBRATION = "Calibration"
    STRESS_TEST = "StressTest"
    LEAKAGE = "Leakage"
    DRIFT = "Drift"
    LIVE_CONSISTENCY = "LiveConsistency"


@dataclass
class GateVerdict:
    """
    Final verdict from gate system

    Determines if model is approved for production deployment.
    """
    model_id: str
    timestamp: datetime

    approved: bool
    total_gates: int
    num_gates_passed: int

    gate_results: Dict[str, GateResult]
    failed_gates: List[str]
    failure_reasons: Dict[str, str]

    overall_score: float  # Average score across gates [0-1]

    # Metadata
    evaluator: str = "GateSystem"
    version: str = "1.0"


class GateSystem:
    """
    Gate System

    Multi-stage decision gates for model approval.

    NO AI, NO LLMs - Pure rule-based validation.

    Example:
        gate_system = GateSystem()

        verdict = gate_system.evaluate_model(
            model_id="btc_trend_v48",
            sharpe_ratio=1.5,
            win_rate=0.55,
            profit_factor=2.0,
            max_drawdown_pct=10.0,
            volatility_pct=25.0,
            max_consecutive_losses=3,
            brier_score=0.15,
            ece=0.08,
            stress_test_results={
                "flash_crash": True,
                "stuck_position": True,
                "partial_fill": True
            },
            leakage_detected=False,
            critical_drift=False,
            max_psi=0.12
        )

        if verdict.approved:
            deploy_to_production(model_id)
        else:
            print(f"Failed gates: {verdict.failed_gates}")
    """

    def __init__(
        self,
        # Performance thresholds
        min_sharpe: float = 1.0,
        min_win_rate: float = 0.50,
        min_profit_factor: float = 1.5,

        # Risk thresholds
        max_drawdown: float = 15.0,
        max_volatility: float = 30.0,
        max_consecutive_losses: int = 5,

        # Calibration thresholds
        max_brier: float = 0.25,
        max_ece: float = 0.10,

        # Stress test
        required_stress_pass_rate: float = 1.0,

        # Drift
        max_psi: float = 0.2,

        # Live consistency
        min_live_consistency: float = 0.7,

        # Approval criteria
        require_all_gates: bool = True,
        min_gates_to_pass: int = 6
    ):
        """
        Initialize gate system

        Args:
            min_sharpe: Minimum Sharpe ratio
            min_win_rate: Minimum win rate
            min_profit_factor: Minimum profit factor
            max_drawdown: Maximum drawdown percentage
            max_volatility: Maximum volatility percentage
            max_consecutive_losses: Max consecutive losses
            max_brier: Maximum Brier score
            max_ece: Maximum ECE
            required_stress_pass_rate: Required stress test pass rate
            max_psi: Maximum PSI for drift
            min_live_consistency: Minimum live/backtest consistency
            require_all_gates: If True, all gates must pass
            min_gates_to_pass: Minimum gates to pass (if not require_all_gates)
        """
        self.require_all_gates = require_all_gates
        self.min_gates_to_pass = min_gates_to_pass

        # Initialize gates
        self.gates: Dict[str, BaseGate] = {
            GateName.PERFORMANCE: PerformanceGate(
                min_sharpe=min_sharpe,
                min_win_rate=min_win_rate,
                min_profit_factor=min_profit_factor
            ),
            GateName.RISK: RiskGate(
                max_acceptable_drawdown=max_drawdown,
                max_acceptable_volatility=max_volatility,
                max_consecutive_losses=max_consecutive_losses
            ),
            GateName.CALIBRATION: CalibrationGate(
                max_brier=max_brier,
                max_ece=max_ece
            ),
            GateName.STRESS_TEST: StressTestGate(
                required_pass_rate=required_stress_pass_rate
            ),
            GateName.LEAKAGE: LeakageGate(),
            GateName.DRIFT: DriftGate(max_acceptable_psi=max_psi),
            GateName.LIVE_CONSISTENCY: LiveConsistencyGate(
                min_consistency_ratio=min_live_consistency
            )
        }

    def evaluate_model(
        self,
        model_id: str,
        # Performance metrics
        sharpe_ratio: float,
        win_rate: float,
        profit_factor: float,
        # Risk metrics
        max_drawdown_pct: float,
        volatility_pct: float,
        max_consecutive_losses: int,
        # Calibration metrics
        brier_score: float,
        ece: float,
        # Stress test results
        stress_test_results: Dict[str, bool],
        # Leakage check
        leakage_detected: bool,
        leakage_issues: Optional[List[str]] = None,
        # Drift check
        critical_drift: bool,
        max_psi: float,
        drifted_features: Optional[List[str]] = None,
        # Live consistency (optional)
        backtest_sharpe: Optional[float] = None,
        live_sharpe: Optional[float] = None
    ) -> GateVerdict:
        """
        Evaluate model through all gates

        Args:
            model_id: Model identifier
            sharpe_ratio: Sharpe ratio
            win_rate: Win rate
            profit_factor: Profit factor
            max_drawdown_pct: Max drawdown percentage
            volatility_pct: Volatility percentage
            max_consecutive_losses: Max consecutive losses
            brier_score: Brier score
            ece: Expected Calibration Error
            stress_test_results: Stress test results
            leakage_detected: Leakage detected flag
            leakage_issues: List of leakage issues
            critical_drift: Critical drift flag
            max_psi: Maximum PSI
            drifted_features: List of drifted features
            backtest_sharpe: Backtest Sharpe (for consistency check)
            live_sharpe: Live Sharpe (for consistency check)

        Returns:
            GateVerdict
        """
        logger.info(
            "evaluating_model_through_gates",
            model_id=model_id,
            num_gates=len(self.gates)
        )

        gate_results = {}
        failed_gates = []
        failure_reasons = {}

        # Run each gate
        gate_results[GateName.PERFORMANCE] = self.gates[GateName.PERFORMANCE].evaluate(
            sharpe_ratio=sharpe_ratio,
            win_rate=win_rate,
            profit_factor=profit_factor
        )

        gate_results[GateName.RISK] = self.gates[GateName.RISK].evaluate(
            max_drawdown_pct=max_drawdown_pct,
            volatility_pct=volatility_pct,
            max_consecutive_losses=max_consecutive_losses
        )

        gate_results[GateName.CALIBRATION] = self.gates[GateName.CALIBRATION].evaluate(
            brier_score=brier_score,
            ece=ece
        )

        gate_results[GateName.STRESS_TEST] = self.gates[GateName.STRESS_TEST].evaluate(
            stress_test_results=stress_test_results
        )

        gate_results[GateName.LEAKAGE] = self.gates[GateName.LEAKAGE].evaluate(
            leakage_detected=leakage_detected,
            leakage_issues=leakage_issues
        )

        gate_results[GateName.DRIFT] = self.gates[GateName.DRIFT].evaluate(
            critical_drift=critical_drift,
            max_psi=max_psi,
            drifted_features=drifted_features
        )

        # Live consistency (optional - uses backtest Sharpe if no live data)
        if backtest_sharpe is None:
            backtest_sharpe = sharpe_ratio

        gate_results[GateName.LIVE_CONSISTENCY] = self.gates[GateName.LIVE_CONSISTENCY].evaluate(
            backtest_sharpe=backtest_sharpe,
            live_sharpe=live_sharpe
        )

        # Collect failures
        for gate_name, result in gate_results.items():
            if not result.passed:
                failed_gates.append(gate_name)
                failure_reasons[gate_name] = result.reason

        # Calculate overall metrics
        num_passed = sum(1 for result in gate_results.values() if result.passed)
        total_gates = len(gate_results)

        # Calculate overall score (average of gate scores)
        scores = [r.score for r in gate_results.values() if r.score is not None]
        overall_score = sum(scores) / len(scores) if scores else 0.0

        # Determine approval
        if self.require_all_gates:
            approved = num_passed == total_gates
        else:
            approved = num_passed >= self.min_gates_to_pass

        verdict = GateVerdict(
            model_id=model_id,
            timestamp=datetime.utcnow(),
            approved=approved,
            total_gates=total_gates,
            num_gates_passed=num_passed,
            gate_results=gate_results,
            failed_gates=failed_gates,
            failure_reasons=failure_reasons,
            overall_score=overall_score
        )

        if approved:
            logger.info(
                "model_approved",
                model_id=model_id,
                gates_passed=num_passed,
                total_gates=total_gates,
                overall_score=overall_score
            )
        else:
            logger.warning(
                "model_rejected",
                model_id=model_id,
                gates_passed=num_passed,
                total_gates=total_gates,
                failed_gates=failed_gates,
                failure_reasons=failure_reasons
            )

        return verdict

    def generate_report(self, verdict: GateVerdict) -> str:
        """
        Generate human-readable gate report

        Args:
            verdict: Gate verdict

        Returns:
            Report string
        """
        lines = [
            "=" * 80,
            "DECISION GATE REPORT",
            "=" * 80,
            "",
            f"Model ID: {verdict.model_id}",
            f"Timestamp: {verdict.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Evaluator: {verdict.evaluator} v{verdict.version}",
            "",
            f"VERDICT: {'✅ APPROVED' if verdict.approved else '❌ REJECTED'}",
            f"Gates Passed: {verdict.num_gates_passed}/{verdict.total_gates}",
            f"Overall Score: {verdict.overall_score:.2%}",
            "",
            "GATE RESULTS:",
            "-" * 80
        ]

        for gate_name, result in verdict.gate_results.items():
            status = "✅ PASS" if result.passed else "❌ FAIL"
            score_str = f"({result.score:.2%})" if result.score is not None else ""

            lines.append(f"{status:10s} {gate_name:20s} {score_str:8s} {result.reason}")

        if len(verdict.failed_gates) > 0:
            lines.extend([
                "",
                "FAILED GATES:",
                "-" * 80
            ])

            for gate_name in verdict.failed_gates:
                lines.append(f"  ❌ {gate_name}: {verdict.failure_reasons[gate_name]}")

        lines.append("")
        lines.append("=" * 80)

        return "\n".join(lines)

    def get_gate_thresholds(self) -> Dict[str, Dict]:
        """
        Get current gate threshold configurations

        Returns:
            Dict of gate configurations
        """
        thresholds = {}

        for gate_name, gate in self.gates.items():
            if hasattr(gate, '__dict__'):
                thresholds[gate_name] = {
                    k: v for k, v in gate.__dict__.items()
                    if not k.startswith('_') and k != 'name'
                }

        return thresholds
