"""
Integration Test Suite

Run: pytest tests/test_integration.py -v
"""
import pytest
import pandas as pd
import numpy as np

def test_gate_system():
    """Test gate system approves good models"""
    from observability.decision_gates import GateSystem
    
    gates = GateSystem()
    verdict = gates.evaluate_model(
        model_id="test_v1",
        sharpe_ratio=1.5, win_rate=0.55, profit_factor=2.0,
        max_drawdown_pct=10.0, volatility_pct=25.0, max_consecutive_losses=3,
        brier_score=0.15, ece=0.08,
        stress_test_results={"flash_crash": True, "stuck_position": True},
        leakage_detected=False, critical_drift=False, max_psi=0.12
    )
    assert verdict.approved == True
    print("✅ Gate system test passed")

def test_calibration():
    """Test calibration works"""
    from models.calibration import ConfidenceCalibrator
    
    cal = ConfidenceCalibrator()
    probs = np.random.random(100)
    labels = np.random.randint(0, 2, 100)
    cal.fit(probs, labels)
    calibrated = cal.calibrate(probs)
    assert len(calibrated) == len(probs)
    print("✅ Calibration test passed")

def test_position_sizing():
    """Test position sizing"""
    from models.position_sizing import BayesianPositionSizer
    
    sizer = BayesianPositionSizer()
    result = sizer.calculate_position_size(
        signal_confidence=0.75,
        current_regime="trending",
        account_balance=100000
    )
    assert 0 <= result.position_fraction <= 1.0
    print("✅ Position sizing test passed")

if __name__ == "__main__":
    test_gate_system()
    test_calibration()
    test_position_sizing()
    print("\n✅ All integration tests passed!")
