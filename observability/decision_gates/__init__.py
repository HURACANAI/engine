"""
Rule-Based Decision Gates

Multi-stage gates that models must pass before production deployment.

CRITICAL: This replaces the AI Council with deterministic rule-based checks.
No LLMs, no AI - pure rule-based validation.

Gates:
1. Performance Gate: Sharpe > threshold, win rate > threshold
2. Risk Gate: Max drawdown < threshold, volatility < threshold
3. Calibration Gate: Brier score < threshold, ECE < threshold
4. Stress Test Gate: Must pass all stress scenarios
5. Leakage Gate: No data leakage detected
6. Drift Gate: No critical drift in validation data
7. Live Consistency Gate: Backtest vs live performance alignment

Usage:
    from observability.decision_gates import GateSystem

    gate_system = GateSystem()

    # Evaluate model through all gates
    verdict = gate_system.evaluate_model(
        model_id="btc_trend_v48",
        validation_metrics=val_metrics,
        stress_test_results=stress_results,
        calibration_metrics=calibration,
        leakage_check_results=leakage_results
    )

    if verdict.approved:
        print(f"✅ Model APPROVED for production")
        print(f"Passed {verdict.num_gates_passed}/{verdict.total_gates} gates")
        deploy_to_production(model_id)
    else:
        print(f"❌ Model REJECTED")
        print(f"Failed gates: {verdict.failed_gates}")
        for gate, reason in verdict.failure_reasons.items():
            print(f"  {gate}: {reason}")
"""

from .gate_system import (
    GateSystem,
    GateVerdict,
    GateResult,
    GateName
)
from .gates import (
    PerformanceGate,
    RiskGate,
    CalibrationGate,
    StressTestGate,
    LeakageGate,
    DriftGate
)

__all__ = [
    # System
    "GateSystem",
    "GateVerdict",
    "GateResult",
    "GateName",

    # Individual gates
    "PerformanceGate",
    "RiskGate",
    "CalibrationGate",
    "StressTestGate",
    "LeakageGate",
    "DriftGate",
]
