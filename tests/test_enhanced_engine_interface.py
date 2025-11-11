"""
Unit Tests for Enhanced Engine Interface

Tests for enhanced engine interface with swing trading support.
"""

from __future__ import annotations

import pytest
from datetime import datetime, timezone, timedelta

from src.shared.engines.enhanced_engine_interface import (
    BaseEnhancedEngine,
    EnhancedEngineInput,
    EnhancedEngineOutput,
    EnhancedEngineRegistry,
    TradingHorizon,
    Direction,
    minutes_to_horizon_type,
    horizon_type_to_minutes,
)


class TestEngine(BaseEnhancedEngine):
    """Test engine implementation."""
    
    def __init__(self, engine_id: str, supported_regimes: list[str], supported_horizons: list[TradingHorizon]):
        super().__init__(
            engine_id=engine_id,
            name=f"Test Engine {engine_id}",
            supported_regimes=supported_regimes,
            supported_horizons=supported_horizons,
            default_horizon=TradingHorizon.SCALP,
        )
    
    def infer(self, input_data: EnhancedEngineInput) -> EnhancedEngineOutput:
        """Test inference."""
        return EnhancedEngineOutput(
            direction=Direction.BUY,
            edge_bps_before_costs=100.0,
            confidence_0_1=0.75,
            horizon_minutes=60,
            horizon_type=self.default_horizon,
            metadata={"test": True},
        )


def test_enhanced_engine_initialization():
    """Test enhanced engine initialization."""
    engine = TestEngine(
        engine_id="test_engine_1",
        supported_regimes=["TREND", "RANGE"],
        supported_horizons=[TradingHorizon.SWING, TradingHorizon.POSITION],
    )
    
    assert engine.engine_id == "test_engine_1"
    assert engine.name == "Test Engine test_engine_1"
    assert engine.supported_regimes == ["TREND", "RANGE"]
    assert len(engine.supported_horizons) == 2
    assert TradingHorizon.SWING in engine.supported_horizons
    assert TradingHorizon.POSITION in engine.supported_horizons


def test_enhanced_engine_is_supported_regime():
    """Test regime support check."""
    engine = TestEngine(
        engine_id="test_engine_1",
        supported_regimes=["TREND", "RANGE"],
        supported_horizons=[TradingHorizon.SWING],
    )
    
    assert engine.is_supported_regime("TREND") is True
    assert engine.is_supported_regime("RANGE") is True
    assert engine.is_supported_regime("PANIC") is False


def test_enhanced_engine_is_supported_horizon():
    """Test horizon support check."""
    engine = TestEngine(
        engine_id="test_engine_1",
        supported_regimes=["TREND"],
        supported_horizons=[TradingHorizon.SWING, TradingHorizon.POSITION],
    )
    
    assert engine.is_supported_horizon(TradingHorizon.SWING) is True
    assert engine.is_supported_horizon(TradingHorizon.POSITION) is True
    assert engine.is_supported_horizon(TradingHorizon.SCALP) is False


def test_enhanced_engine_infer():
    """Test engine inference."""
    engine = TestEngine(
        engine_id="test_engine_1",
        supported_regimes=["TREND"],
        supported_horizons=[TradingHorizon.SWING],
    )
    
    input_data = EnhancedEngineInput(
        symbol="BTCUSDT",
        timestamp=datetime.now(timezone.utc),
        features={"rsi": 50.0, "ema": 45000.0},
        regime="TREND",
        costs={"fees_bps": 4.0, "spread_bps": 5.0, "slippage_bps": 2.0},
    )
    
    output = engine.infer(input_data)
    
    assert output.direction == Direction.BUY
    assert output.edge_bps_before_costs == 100.0
    assert output.confidence_0_1 == 0.75
    assert output.horizon_minutes == 60
    assert output.horizon_type == TradingHorizon.SCALP
    assert output.metadata["test"] is True


def test_enhanced_engine_output_to_dict():
    """Test engine output to dictionary conversion."""
    output = EnhancedEngineOutput(
        direction=Direction.BUY,
        edge_bps_before_costs=150.0,
        confidence_0_1=0.80,
        horizon_minutes=24 * 60,  # 1 day
        horizon_type=TradingHorizon.SWING,
        stop_loss_bps=200.0,
        take_profit_bps=400.0,
        trailing_stop_bps=100.0,
        position_size_multiplier=1.0,
        max_holding_hours=48.0,
        funding_cost_estimate_bps=10.0,
        metadata={"test": True},
    )
    
    output_dict = output.to_dict()
    
    assert output_dict["direction"] == "buy"
    assert output_dict["edge_bps_before_costs"] == 150.0
    assert output_dict["confidence_0_1"] == 0.80
    assert output_dict["horizon_minutes"] == 24 * 60
    assert output_dict["horizon_type"] == "swing"
    assert output_dict["stop_loss_bps"] == 200.0
    assert output_dict["take_profit_bps"] == 400.0
    assert output_dict["trailing_stop_bps"] == 100.0
    assert output_dict["position_size_multiplier"] == 1.0
    assert output_dict["max_holding_hours"] == 48.0
    assert output_dict["funding_cost_estimate_bps"] == 10.0
    assert output_dict["metadata"]["test"] is True


def test_enhanced_engine_output_from_dict():
    """Test engine output from dictionary conversion."""
    output_dict = {
        "direction": "buy",
        "edge_bps_before_costs": 150.0,
        "confidence_0_1": 0.80,
        "horizon_minutes": 24 * 60,
        "horizon_type": "swing",
        "stop_loss_bps": 200.0,
        "take_profit_bps": 400.0,
        "trailing_stop_bps": 100.0,
        "position_size_multiplier": 1.0,
        "max_holding_hours": 48.0,
        "funding_cost_estimate_bps": 10.0,
        "metadata": {"test": True},
    }
    
    output = EnhancedEngineOutput.from_dict(output_dict)
    
    assert output.direction == Direction.BUY
    assert output.edge_bps_before_costs == 150.0
    assert output.confidence_0_1 == 0.80
    assert output.horizon_minutes == 24 * 60
    assert output.horizon_type == TradingHorizon.SWING
    assert output.stop_loss_bps == 200.0
    assert output.take_profit_bps == 400.0
    assert output.trailing_stop_bps == 100.0
    assert output.position_size_multiplier == 1.0
    assert output.max_holding_hours == 48.0
    assert output.funding_cost_estimate_bps == 10.0
    assert output.metadata["test"] is True


def test_enhanced_engine_output_is_swing_trade():
    """Test swing trade detection."""
    swing_output = EnhancedEngineOutput(
        direction=Direction.BUY,
        edge_bps_before_costs=100.0,
        confidence_0_1=0.75,
        horizon_minutes=24 * 60,
        horizon_type=TradingHorizon.SWING,
    )
    
    scalp_output = EnhancedEngineOutput(
        direction=Direction.BUY,
        edge_bps_before_costs=100.0,
        confidence_0_1=0.75,
        horizon_minutes=60,
        horizon_type=TradingHorizon.SCALP,
    )
    
    assert swing_output.is_swing_trade() is True
    assert scalp_output.is_swing_trade() is False


def test_enhanced_engine_output_get_holding_duration_days():
    """Test holding duration calculation."""
    output = EnhancedEngineOutput(
        direction=Direction.BUY,
        edge_bps_before_costs=100.0,
        confidence_0_1=0.75,
        horizon_minutes=24 * 60 * 7,  # 7 days
        horizon_type=TradingHorizon.POSITION,
    )
    
    assert output.get_holding_duration_days() == 7.0


def test_enhanced_engine_registry():
    """Test enhanced engine registry."""
    registry = EnhancedEngineRegistry()
    
    engine1 = TestEngine(
        engine_id="engine_1",
        supported_regimes=["TREND"],
        supported_horizons=[TradingHorizon.SWING],
    )
    
    engine2 = TestEngine(
        engine_id="engine_2",
        supported_regimes=["RANGE"],
        supported_horizons=[TradingHorizon.SCALP],
    )
    
    registry.register(engine1)
    registry.register(engine2)
    
    assert registry.get_engine_count() == 2
    assert registry.get_engine("engine_1") == engine1
    assert registry.get_engine("engine_2") == engine2
    assert registry.get_engine("engine_3") is None


def test_enhanced_engine_registry_get_engines_for_regime():
    """Test getting engines for a regime."""
    registry = EnhancedEngineRegistry()
    
    engine1 = TestEngine(
        engine_id="engine_1",
        supported_regimes=["TREND"],
        supported_horizons=[TradingHorizon.SWING],
    )
    
    engine2 = TestEngine(
        engine_id="engine_2",
        supported_regimes=["RANGE"],
        supported_horizons=[TradingHorizon.SCALP],
    )
    
    registry.register(engine1)
    registry.register(engine2)
    
    trend_engines = registry.get_engines_for_regime("TREND")
    assert len(trend_engines) == 1
    assert trend_engines[0] == engine1
    
    range_engines = registry.get_engines_for_regime("RANGE")
    assert len(range_engines) == 1
    assert range_engines[0] == engine2


def test_enhanced_engine_registry_get_engines_for_horizon():
    """Test getting engines for a horizon."""
    registry = EnhancedEngineRegistry()
    
    engine1 = TestEngine(
        engine_id="engine_1",
        supported_regimes=["TREND"],
        supported_horizons=[TradingHorizon.SWING],
    )
    
    engine2 = TestEngine(
        engine_id="engine_2",
        supported_regimes=["RANGE"],
        supported_horizons=[TradingHorizon.SCALP],
    )
    
    registry.register(engine1)
    registry.register(engine2)
    
    swing_engines = registry.get_engines_for_horizon(TradingHorizon.SWING)
    assert len(swing_engines) == 1
    assert swing_engines[0] == engine1
    
    scalp_engines = registry.get_engines_for_horizon(TradingHorizon.SCALP)
    assert len(scalp_engines) == 1
    assert scalp_engines[0] == engine2


def test_enhanced_engine_registry_get_engines_for_regime_and_horizon():
    """Test getting engines for regime and horizon."""
    registry = EnhancedEngineRegistry()
    
    engine1 = TestEngine(
        engine_id="engine_1",
        supported_regimes=["TREND"],
        supported_horizons=[TradingHorizon.SWING],
    )
    
    engine2 = TestEngine(
        engine_id="engine_2",
        supported_regimes=["TREND"],
        supported_horizons=[TradingHorizon.SCALP],
    )
    
    registry.register(engine1)
    registry.register(engine2)
    
    engines = registry.get_engines_for_regime_and_horizon("TREND", TradingHorizon.SWING)
    assert len(engines) == 1
    assert engines[0] == engine1
    
    engines = registry.get_engines_for_regime_and_horizon("TREND", TradingHorizon.SCALP)
    assert len(engines) == 1
    assert engines[0] == engine2
    
    engines = registry.get_engines_for_regime_and_horizon("RANGE", TradingHorizon.SWING)
    assert len(engines) == 0


def test_minutes_to_horizon_type():
    """Test minutes to horizon type conversion."""
    assert minutes_to_horizon_type(30) == TradingHorizon.SCALP
    assert minutes_to_horizon_type(60) == TradingHorizon.SCALP
    assert minutes_to_horizon_type(12 * 60) == TradingHorizon.SWING
    assert minutes_to_horizon_type(24 * 60) == TradingHorizon.SWING
    assert minutes_to_horizon_type(3 * 24 * 60) == TradingHorizon.POSITION
    assert minutes_to_horizon_type(7 * 24 * 60) == TradingHorizon.POSITION
    assert minutes_to_horizon_type(14 * 24 * 60) == TradingHorizon.CORE
    assert minutes_to_horizon_type(30 * 24 * 60) == TradingHorizon.CORE


def test_horizon_type_to_minutes():
    """Test horizon type to minutes conversion."""
    assert horizon_type_to_minutes(TradingHorizon.SCALP) == 60
    assert horizon_type_to_minutes(TradingHorizon.SWING) == 24 * 60
    assert horizon_type_to_minutes(TradingHorizon.POSITION) == 7 * 24 * 60
    assert horizon_type_to_minutes(TradingHorizon.CORE) == 30 * 24 * 60


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

