"""
Unit Tests for Enhanced Regime Classifier

Tests for enhanced regime classifier with swing trading support.
"""

from __future__ import annotations

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.shared.regime.enhanced_regime_classifier import (
    EnhancedRegimeClassifier,
    RegimeGatingConfig,
    RegimeClassification,
    Regime,
)
from src.shared.engines.enhanced_engine_interface import TradingHorizon, BaseEnhancedEngine


class TestEngine(BaseEnhancedEngine):
    """Test engine for regime filtering."""
    
    def __init__(self, engine_id: str, supported_regimes: list[str], supported_horizons: list[TradingHorizon]):
        super().__init__(
            engine_id=engine_id,
            name=f"Test Engine {engine_id}",
            supported_regimes=supported_regimes,
            supported_horizons=supported_horizons,
        )
    
    def infer(self, input_data):
        pass


def create_sample_candles(n: int = 100, trend: bool = False, volatility: float = 0.02) -> pd.DataFrame:
    """Create sample candle data."""
    dates = pd.date_range(start=datetime.now() - timedelta(days=n), periods=n, freq='1h')
    base_price = 45000.0
    
    if trend:
        prices = base_price + np.arange(n) * 10.0 + np.random.randn(n) * base_price * volatility
    else:
        prices = base_price + np.random.randn(n) * base_price * volatility
    
    volumes = np.random.uniform(1000, 10000, n)
    
    return pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': prices * 1.01,
        'low': prices * 0.99,
        'close': prices,
        'volume': volumes,
    })


def test_regime_gating_config_initialization():
    """Test regime gating config initialization."""
    config = RegimeGatingConfig(
        panic_allows_swing=False,
        panic_allows_position=False,
        illiquid_allows_swing=False,
        illiquid_allows_position=False,
        panic_risk_multiplier=2.0,
        illiquid_risk_multiplier=1.5,
        high_volatility_risk_multiplier=1.3,
    )
    
    assert config.panic_allows_swing is False
    assert config.panic_allows_position is False
    assert config.panic_risk_multiplier == 2.0
    assert config.illiquid_risk_multiplier == 1.5


def test_enhanced_regime_classifier_initialization():
    """Test enhanced regime classifier initialization."""
    config = RegimeGatingConfig()
    classifier = EnhancedRegimeClassifier(config)
    
    assert classifier.config == config


def test_enhanced_regime_classifier_classify_trend():
    """Test regime classification for trend."""
    classifier = EnhancedRegimeClassifier()
    
    # Create trending candles
    candles_df = create_sample_candles(n=100, trend=True, volatility=0.02)
    
    classification = classifier.classify(candles_df, "BTCUSDT")
    
    assert classification.regime in [Regime.TREND, Regime.RANGE]  # Could be either
    assert classification.confidence > 0.0
    assert classification.allows_swing_trading is True
    assert classification.allows_position_trading is True


def test_enhanced_regime_classifier_classify_range():
    """Test regime classification for range."""
    classifier = EnhancedRegimeClassifier()
    
    # Create ranging candles
    candles_df = create_sample_candles(n=100, trend=False, volatility=0.01)
    
    classification = classifier.classify(candles_df, "BTCUSDT")
    
    assert classification.regime in [Regime.RANGE, Regime.LOW_VOLATILITY]
    assert classification.confidence > 0.0
    assert classification.allows_swing_trading is True


def test_enhanced_regime_classifier_classify_panic():
    """Test regime classification for panic."""
    classifier = EnhancedRegimeClassifier()
    
    # Create high volatility candles
    candles_df = create_sample_candles(n=100, trend=True, volatility=0.10)
    
    classification = classifier.classify(candles_df, "BTCUSDT")
    
    # Could be PANIC or HIGH_VOLATILITY
    assert classification.regime in [Regime.PANIC, Regime.HIGH_VOLATILITY, Regime.TREND]
    assert classification.confidence > 0.0


def test_enhanced_regime_classifier_classify_illiquid():
    """Test regime classification for illiquid."""
    classifier = EnhancedRegimeClassifier()
    
    # Create low volume candles
    candles_df = create_sample_candles(n=100, trend=False, volatility=0.02)
    candles_df['volume'] = candles_df['volume'] * 0.1  # Low volume
    
    classification = classifier.classify(candles_df, "BTCUSDT")
    
    # Could be ILLIQUID or RANGE
    assert classification.regime in [Regime.ILLIQUID, Regime.RANGE]
    assert classification.confidence > 0.0


def test_enhanced_regime_classifier_insufficient_data():
    """Test regime classification with insufficient data."""
    classifier = EnhancedRegimeClassifier()
    
    # Create minimal candles
    candles_df = create_sample_candles(n=10)
    
    classification = classifier.classify(candles_df, "BTCUSDT")
    
    assert classification.regime == Regime.RANGE
    assert classification.confidence == 0.5
    assert classification.metadata["reason"] == "insufficient_data"


def test_enhanced_regime_classifier_allows_swing_trading():
    """Test swing trading permission check."""
    config = RegimeGatingConfig(
        panic_allows_swing=False,
        illiquid_allows_swing=False,
    )
    classifier = EnhancedRegimeClassifier(config)
    
    # Panic regime
    classification = RegimeClassification(
        regime=Regime.PANIC,
        confidence=1.0,
        metadata={},
        allows_swing_trading=classifier._allows_swing_trading(Regime.PANIC),
        allows_position_trading=classifier._allows_position_trading(Regime.PANIC),
        allows_core_holding=classifier._allows_core_holding(Regime.PANIC),
        risk_multiplier=classifier._get_risk_multiplier(Regime.PANIC),
    )
    
    assert classification.allows_swing_trading is False
    
    # Trend regime
    classification = RegimeClassification(
        regime=Regime.TREND,
        confidence=1.0,
        metadata={},
        allows_swing_trading=classifier._allows_swing_trading(Regime.TREND),
        allows_position_trading=classifier._allows_position_trading(Regime.TREND),
        allows_core_holding=classifier._allows_core_holding(Regime.TREND),
        risk_multiplier=classifier._get_risk_multiplier(Regime.TREND),
    )
    
    assert classification.allows_swing_trading is True


def test_enhanced_regime_classifier_is_safe_for_horizon():
    """Test horizon safety check."""
    classification = RegimeClassification(
        regime=Regime.TREND,
        confidence=1.0,
        metadata={},
        allows_swing_trading=True,
        allows_position_trading=True,
        allows_core_holding=True,
        risk_multiplier=1.0,
    )
    
    assert classification.is_safe_for_horizon(TradingHorizon.SCALP) is True
    assert classification.is_safe_for_horizon(TradingHorizon.SWING) is True
    assert classification.is_safe_for_horizon(TradingHorizon.POSITION) is True
    assert classification.is_safe_for_horizon(TradingHorizon.CORE) is True
    
    # Panic regime
    classification = RegimeClassification(
        regime=Regime.PANIC,
        confidence=1.0,
        metadata={},
        allows_swing_trading=False,
        allows_position_trading=False,
        allows_core_holding=True,
        risk_multiplier=2.0,
    )
    
    assert classification.is_safe_for_horizon(TradingHorizon.SCALP) is True
    assert classification.is_safe_for_horizon(TradingHorizon.SWING) is False
    assert classification.is_safe_for_horizon(TradingHorizon.POSITION) is False
    assert classification.is_safe_for_horizon(TradingHorizon.CORE) is True


def test_enhanced_regime_classifier_filter_engines_by_regime():
    """Test filtering engines by regime."""
    classifier = EnhancedRegimeClassifier()
    
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
    
    engines = [engine1, engine2]
    
    # Filter by TREND regime
    filtered = classifier.filter_engines_by_regime(engines, Regime.TREND)
    assert len(filtered) == 1
    assert filtered[0] == engine1
    
    # Filter by RANGE regime
    filtered = classifier.filter_engines_by_regime(engines, Regime.RANGE)
    assert len(filtered) == 1
    assert filtered[0] == engine2
    
    # Filter by PANIC regime
    filtered = classifier.filter_engines_by_regime(engines, Regime.PANIC)
    assert len(filtered) == 0


def test_enhanced_regime_classifier_filter_engines_by_regime_and_horizon():
    """Test filtering engines by regime and horizon."""
    classifier = EnhancedRegimeClassifier()
    
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
    
    engines = [engine1, engine2]
    
    # Filter by TREND regime and SWING horizon
    filtered = classifier.filter_engines_by_regime(engines, Regime.TREND, TradingHorizon.SWING)
    assert len(filtered) == 1
    assert filtered[0] == engine1
    
    # Filter by TREND regime and SCALP horizon
    filtered = classifier.filter_engines_by_regime(engines, Regime.TREND, TradingHorizon.SCALP)
    assert len(filtered) == 1
    assert filtered[0] == engine2


def test_enhanced_regime_classifier_should_trade_horizon():
    """Test should trade horizon check."""
    config = RegimeGatingConfig(
        panic_allows_swing=False,
        panic_allows_position=False,
    )
    classifier = EnhancedRegimeClassifier(config)
    
    # Panic regime
    should_trade = classifier.should_trade_horizon(Regime.PANIC, TradingHorizon.SCALP)
    assert should_trade is True  # Scalps can trade in panic
    
    should_trade = classifier.should_trade_horizon(Regime.PANIC, TradingHorizon.SWING)
    assert should_trade is False  # Swings cannot trade in panic
    
    # Trend regime
    should_trade = classifier.should_trade_horizon(Regime.TREND, TradingHorizon.SWING)
    assert should_trade is True  # Swings can trade in trend


def test_enhanced_regime_classifier_risk_multiplier():
    """Test risk multiplier calculation."""
    config = RegimeGatingConfig(
        panic_risk_multiplier=2.0,
        illiquid_risk_multiplier=1.5,
        high_volatility_risk_multiplier=1.3,
    )
    classifier = EnhancedRegimeClassifier(config)
    
    assert classifier._get_risk_multiplier(Regime.PANIC) == 2.0
    assert classifier._get_risk_multiplier(Regime.ILLIQUID) == 1.5
    assert classifier._get_risk_multiplier(Regime.HIGH_VOLATILITY) == 1.3
    assert classifier._get_risk_multiplier(Regime.TREND) == 1.0
    assert classifier._get_risk_multiplier(Regime.RANGE) == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

