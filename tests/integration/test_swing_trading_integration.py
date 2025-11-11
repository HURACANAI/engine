"""
Integration Tests for Swing Trading Workflow

Tests the complete swing trading workflow from feature building to position management.
"""

from __future__ import annotations

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

from src.shared.engines.enhanced_engine_interface import (
    EnhancedEngineInput,
    TradingHorizon,
    Direction,
)
from src.shared.engines.plugins import TrendEngine, RangeEngine, BreakoutEngine
from src.shared.costs.enhanced_cost_calculator import EnhancedCostCalculator, EnhancedCostModel
from src.shared.trading.swing_position_manager import SwingPositionManager, SwingPositionConfig
from src.shared.portfolio.horizon_portfolio_allocator import HorizonPortfolioAllocator, HorizonPortfolioConfig
from src.shared.regime.enhanced_regime_classifier import EnhancedRegimeClassifier
from src.shared.meta.enhanced_meta_combiner import EnhancedMetaCombiner
from src.shared.features.enhanced_feature_builder import EnhancedFeatureBuilder, AlternativeData


def create_sample_candles(n: int = 100, trend: bool = True) -> pd.DataFrame:
    """Create sample candle data."""
    dates = pd.date_range(start=datetime.now(timezone.utc) - timedelta(days=n), periods=n, freq='1h')
    base_price = 45000.0
    
    if trend:
        prices = base_price + np.arange(n) * 10.0 + np.random.randn(n) * base_price * 0.02
    else:
        prices = base_price + np.random.randn(n) * base_price * 0.02
    
    volumes = np.random.uniform(1000, 10000, n)
    
    return pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': prices * 1.01,
        'low': prices * 0.99,
        'close': prices,
        'volume': volumes,
    })


def test_swing_trading_workflow():
    """Test complete swing trading workflow."""
    symbol = "BTCUSDT"
    
    # 1. Initialize components
    trend_engine = TrendEngine(config={"window": 20, "momentum_threshold": 0.02})
    range_engine = RangeEngine(config={"window": 30, "range_threshold": 0.02})
    breakout_engine = BreakoutEngine(config={"window": 20, "breakout_threshold": 0.02})
    
    cost_calculator = EnhancedCostCalculator()
    cost_model = EnhancedCostModel(
        symbol=symbol,
        taker_fee_bps=4.0,
        maker_fee_bps=2.0,
        median_spread_bps=5.0,
        slippage_bps_per_sigma=2.0,
        min_notional=10.0,
        step_size=0.01,
        last_updated_utc=datetime.now(timezone.utc),
        funding_rate_bps_per_8h=1.0,
    )
    cost_calculator.register_cost_model(cost_model)
    
    position_manager = SwingPositionManager(SwingPositionConfig())
    portfolio_allocator = HorizonPortfolioAllocator(HorizonPortfolioConfig())
    portfolio_allocator.initialize(total_portfolio_value=10000.0)
    
    regime_classifier = EnhancedRegimeClassifier()
    meta_combiner = EnhancedMetaCombiner(symbol=symbol)
    feature_builder = EnhancedFeatureBuilder(config={
        "indicator_set": {
            "rsi": {"window": 14},
            "ema": {"window": 20},
            "volatility": {"window": 20},
            "momentum": {"window": 10},
        },
        "fill_rules": {"forward_fill_max_gaps": 5},
        "normalization": {"type": "min_max"},
    })
    
    # 2. Build features
    candles_df = create_sample_candles(n=100, trend=True)
    alternative_data = AlternativeData(
        order_book_imbalance=0.1,
        funding_rate=0.01,
    )
    features_df = feature_builder.build_features(candles_df, symbol, alternative_data)
    
    # 3. Classify regime
    regime_classification = regime_classifier.classify(candles_df, symbol)
    assert regime_classification.regime.value in ["TREND", "RANGE", "PANIC", "ILLIQUID"]
    
    # 4. Get engine outputs
    latest_features = features_df.iloc[-1].to_dict()
    input_data = EnhancedEngineInput(
        symbol=symbol,
        timestamp=datetime.now(timezone.utc),
        features=latest_features,
        regime=regime_classification.regime.value,
        costs=cost_calculator.get_costs(
            symbol=symbol,
            timestamp=datetime.now(timezone.utc),
            order_type="maker",
            holding_hours=24.0,
            horizon_type=TradingHorizon.SWING,
        ).__dict__,
    )
    
    trend_output = trend_engine.infer(input_data)
    range_output = range_engine.infer(input_data)
    breakout_output = breakout_engine.infer(input_data)
    
    # 5. Combine engine outputs
    combined_output = meta_combiner.combine(
        engine_outputs=[trend_output, range_output, breakout_output],
        engine_ids=["trend_engine", "range_engine", "breakout_engine"],
        regime=regime_classification.regime.value,
        horizon=TradingHorizon.SWING,
    )
    
    # 6. Check if should trade
    if combined_output.direction != "wait":
        # Check portfolio allocation
        position_size_usd = 1000.0
        can_open, reason = portfolio_allocator.can_open_position(
            horizon=TradingHorizon.SWING,
            position_size_usd=position_size_usd,
        )
        
        if can_open:
            # Calculate net edge with costs
            net_edge = cost_calculator.calculate_net_edge(
                symbol=symbol,
                edge_bps_before_costs=combined_output.edge_bps_before_costs,
                timestamp=datetime.now(timezone.utc),
                order_type="maker",
                holding_hours=24.0,
                horizon_type=TradingHorizon.SWING,
                include_funding=True,
            )
            
            # Check if net edge is sufficient
            if net_edge >= 3.0:  # 3 bps minimum
                # Allocate position
                portfolio_allocator.allocate_position(
                    horizon=TradingHorizon.SWING,
                    position_size_usd=position_size_usd,
                )
                
                # Open position
                current_price = candles_df["close"].iloc[-1]
                entry_size = position_size_usd / current_price
                
                position = position_manager.open_position(
                    symbol=symbol,
                    direction=Direction.BUY if combined_output.direction == "buy" else Direction.SELL,
                    entry_price=current_price,
                    entry_size=entry_size,
                    horizon_type=TradingHorizon.SWING,
                    stop_loss_bps=combined_output.stop_loss_bps or 200.0,
                    take_profit_levels=[(200.0, 0.30), (400.0, 0.40)] if combined_output.take_profit_bps else None,
                    trailing_stop_bps=combined_output.trailing_stop_bps,
                    max_holding_hours=combined_output.max_holding_hours,
                )
                
                assert position is not None
                assert position.symbol == symbol
                assert position.horizon_type == TradingHorizon.SWING
                
                # 7. Update position
                new_price = current_price * 1.02  # 2% profit
                exit_action = position_manager.update_position(
                    symbol=symbol,
                    current_price=new_price,
                    current_regime=regime_classification.regime.value,
                    funding_cost_bps=5.0,
                )
                
                # Should trigger take profit
                if exit_action:
                    assert exit_action["exit_reason"] in ["take_profit", "stop_loss", "trailing_stop"]
                
                # 8. Update performance
                meta_combiner.update_performance(
                    engine_id="trend_engine",
                    accuracy=0.65,
                    net_edge_bps=net_edge,
                    sharpe_ratio=1.5,
                    win_rate=0.60,
                    avg_win_bps=200.0,
                    avg_loss_bps=100.0,
                    total_trades=1,
                    timestamp=datetime.now(timezone.utc),
                )


def test_regime_gating():
    """Test regime gating for swing trading."""
    regime_classifier = EnhancedRegimeClassifier()
    
    # Create panic candles
    candles_df = create_sample_candles(n=100, trend=True)
    candles_df["close"] = candles_df["close"] * (1 + np.random.randn(len(candles_df)) * 0.1)  # High volatility
    
    regime_classification = regime_classifier.classify(candles_df, "BTCUSDT")
    
    # Check if swing trading is allowed
    if regime_classification.regime.value == "PANIC":
        assert regime_classification.allows_swing_trading is False
        assert regime_classification.is_safe_for_horizon(TradingHorizon.SWING) is False
        assert regime_classification.is_safe_for_horizon(TradingHorizon.SCALP) is True


def test_portfolio_allocation():
    """Test portfolio allocation across horizons."""
    config = HorizonPortfolioConfig(
        scalp_max_allocation_pct=20.0,
        swing_max_allocation_pct=40.0,
        position_max_allocation_pct=30.0,
    )
    allocator = HorizonPortfolioAllocator(config)
    allocator.initialize(total_portfolio_value=10000.0)
    
    # Allocate scalp position
    can_open, _ = allocator.can_open_position(TradingHorizon.SCALP, 500.0)
    assert can_open is True
    allocator.allocate_position(TradingHorizon.SCALP, 500.0)
    
    # Allocate swing position
    can_open, _ = allocator.can_open_position(TradingHorizon.SWING, 2000.0)
    assert can_open is True
    allocator.allocate_position(TradingHorizon.SWING, 2000.0)
    
    # Check allocations
    scalp_allocation = allocator.get_allocation(TradingHorizon.SCALP)
    assert scalp_allocation.current_allocation_pct == 5.0  # 500 / 10000 * 100
    
    swing_allocation = allocator.get_allocation(TradingHorizon.SWING)
    assert swing_allocation.current_allocation_pct == 20.0  # 2000 / 10000 * 100


def test_cost_calculation_with_holding():
    """Test cost calculation with holding context."""
    calculator = EnhancedCostCalculator()
    cost_model = EnhancedCostModel(
        symbol="BTCUSDT",
        taker_fee_bps=4.0,
        maker_fee_bps=2.0,
        median_spread_bps=5.0,
        slippage_bps_per_sigma=2.0,
        min_notional=10.0,
        step_size=0.01,
        last_updated_utc=datetime.now(timezone.utc),
        funding_rate_bps_per_8h=1.0,
    )
    calculator.register_cost_model(cost_model)
    
    # Scalp trade (no holding)
    cost_scalp = calculator.get_costs(
        symbol="BTCUSDT",
        timestamp=datetime.now(timezone.utc),
        order_type="maker",
        holding_hours=0.0,
        horizon_type=TradingHorizon.SCALP,
        include_funding=False,
    )
    
    # Swing trade (24 hours holding)
    cost_swing = calculator.get_costs(
        symbol="BTCUSDT",
        timestamp=datetime.now(timezone.utc),
        order_type="maker",
        holding_hours=24.0,
        horizon_type=TradingHorizon.SWING,
        include_funding=True,
    )
    
    # Swing trade should have higher costs due to funding
    assert cost_swing.total_cost_bps > cost_scalp.total_cost_bps
    assert cost_swing.funding_cost_bps > 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

