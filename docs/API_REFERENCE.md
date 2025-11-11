# API Reference

## Overview

This document provides a comprehensive API reference for all swing trading enhancement modules.

## Enhanced Engine Interface

### `BaseEnhancedEngine`

Base class for all engines with swing trading support.

```python
from src.shared.engines.enhanced_engine_interface import BaseEnhancedEngine

class MyEngine(BaseEnhancedEngine):
    def __init__(self, engine_id: str, supported_regimes: List[str], supported_horizons: List[TradingHorizon]):
        super().__init__(
            engine_id=engine_id,
            name="My Engine",
            supported_regimes=supported_regimes,
            supported_horizons=supported_horizons,
            default_horizon=TradingHorizon.SWING,
        )
    
    def infer(self, input_data: EnhancedEngineInput) -> EnhancedEngineOutput:
        # Your engine logic
        pass
```

### `EnhancedEngineInput`

Input data for engine inference.

```python
input_data = EnhancedEngineInput(
    symbol="BTCUSDT",
    timestamp=datetime.now(timezone.utc),
    features={"rsi": 50.0, "ema": 45000.0},
    regime="TREND",
    costs={"fees_bps": 4.0, "spread_bps": 5.0, "funding_bps": 1.0},
    current_position=None,
    holding_duration_hours=0.0,
    portfolio_allocation=None,
)
```

### `EnhancedEngineOutput`

Output from engine inference.

```python
output = EnhancedEngineOutput(
    direction=Direction.BUY,
    edge_bps_before_costs=150.0,
    confidence_0_1=0.75,
    horizon_minutes=24 * 60,
    horizon_type=TradingHorizon.SWING,
    stop_loss_bps=200.0,
    take_profit_bps=400.0,
    trailing_stop_bps=100.0,
    position_size_multiplier=1.0,
    max_holding_hours=48.0,
    funding_cost_estimate_bps=10.0,
)
```

## Enhanced Cost Calculator

### `EnhancedCostCalculator`

Cost calculator with holding context.

```python
from src.shared.costs.enhanced_cost_calculator import EnhancedCostCalculator, EnhancedCostModel

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
    borrow_rate_bps_per_day=0.0,
    overnight_risk_multiplier=1.2,
    liquidity_decay_factor=0.95,
    spread_widening_bps_per_day=0.5,
)

calculator.register_cost_model(cost_model)

# Calculate costs
cost_breakdown = calculator.get_costs(
    symbol="BTCUSDT",
    timestamp=datetime.now(timezone.utc),
    order_type="maker",
    order_size_sigma=1.0,
    holding_hours=48.0,
    horizon_type=TradingHorizon.SWING,
    include_funding=True,
    include_borrow=False,
)

# Calculate net edge
net_edge = calculator.calculate_net_edge(
    symbol="BTCUSDT",
    edge_bps_before_costs=150.0,
    timestamp=datetime.now(timezone.utc),
    order_type="maker",
    holding_hours=48.0,
    horizon_type=TradingHorizon.SWING,
    include_funding=True,
)
```

## Swing Position Manager

### `SwingPositionManager`

Manages swing trading positions.

```python
from src.shared.trading.swing_position_manager import SwingPositionManager, SwingPositionConfig

config = SwingPositionConfig(
    default_stop_loss_bps=200.0,
    use_trailing_stop=True,
    trailing_stop_distance_bps=100.0,
    take_profit_levels=[(200.0, 0.30), (400.0, 0.40), (600.0, 0.20)],
    max_holding_hours=48.0,
    max_funding_cost_bps=500.0,
    exit_on_panic=True,
)

manager = SwingPositionManager(config)

# Open position
position = manager.open_position(
    symbol="BTCUSDT",
    direction=Direction.BUY,
    entry_price=45000.0,
    entry_size=0.1,
    horizon_type=TradingHorizon.SWING,
    stop_loss_bps=200.0,
    take_profit_levels=[(200.0, 0.30), (400.0, 0.40)],
    trailing_stop_bps=100.0,
    max_holding_hours=48.0,
    max_funding_cost_bps=500.0,
)

# Update position
exit_action = manager.update_position(
    symbol="BTCUSDT",
    current_price=46000.0,
    current_regime="TREND",
    funding_cost_bps=5.0,
)

# Close position
exit_details = manager.close_position("BTCUSDT", "manual")
```

## Horizon Portfolio Allocator

### `HorizonPortfolioAllocator`

Manages portfolio allocation across horizons.

```python
from src.shared.portfolio.horizon_portfolio_allocator import HorizonPortfolioAllocator, HorizonPortfolioConfig

config = HorizonPortfolioConfig(
    scalp_max_allocation_pct=20.0,
    swing_max_allocation_pct=40.0,
    position_max_allocation_pct=30.0,
    core_max_allocation_pct=50.0,
    scalp_max_positions=5,
    swing_max_positions=3,
    position_max_positions=2,
    core_max_positions=3,
)

allocator = HorizonPortfolioAllocator(config)
allocator.initialize(total_portfolio_value=10000.0)

# Check if can open position
can_open, reason = allocator.can_open_position(
    horizon=TradingHorizon.SWING,
    position_size_usd=1000.0,
)

# Allocate position
if can_open:
    allocator.allocate_position(
        horizon=TradingHorizon.SWING,
        position_size_usd=1000.0,
    )

# Deallocate position
allocator.deallocate_position(
    horizon=TradingHorizon.SWING,
    position_size_usd=1000.0,
)

# Get available capacity
available_capacity = allocator.get_available_capacity(TradingHorizon.SWING)

# Get max position size
max_position_size = allocator.get_max_position_size(TradingHorizon.SWING)
```

## Enhanced Regime Classifier

### `EnhancedRegimeClassifier`

Regime classifier with swing trading support.

```python
from src.shared.regime.enhanced_regime_classifier import EnhancedRegimeClassifier, RegimeGatingConfig

config = RegimeGatingConfig(
    panic_allows_swing=False,
    panic_allows_position=False,
    illiquid_allows_swing=False,
    illiquid_allows_position=False,
    panic_risk_multiplier=2.0,
    illiquid_risk_multiplier=1.5,
)

classifier = EnhancedRegimeClassifier(config)

# Classify regime
classification = classifier.classify(candles_df, "BTCUSDT")

# Check if swing trading is allowed
if classification.allows_swing_trading:
    # Swing trading allowed
    pass

# Check if horizon is safe
if classification.is_safe_for_horizon(TradingHorizon.SWING):
    # Safe for swing trading
    pass

# Filter engines by regime and horizon
engines = classifier.filter_engines_by_regime(
    engines=all_engines,
    regime=classification.regime,
    horizon=TradingHorizon.SWING,
)
```

## Enhanced Meta Combiner

### `EnhancedMetaCombiner`

Meta combiner with adaptive weighting and hyperparameter tuning.

```python
from src.shared.meta.enhanced_meta_combiner import EnhancedMetaCombiner, HyperparameterConfig

hyperparameter_config = HyperparameterConfig(
    ema_alpha_min=0.05,
    ema_alpha_max=0.3,
    confidence_multiplier_min=0.5,
    confidence_multiplier_max=2.0,
    tuning_frequency_days=7,
    performance_window_days=30,
)

combiner = EnhancedMetaCombiner(
    symbol="BTCUSDT",
    ema_alpha=0.1,
    accuracy_threshold=0.45,
    hyperparameter_config=hyperparameter_config,
)

# Update performance
combiner.update_performance(
    engine_id="trend_engine",
    accuracy=0.65,
    net_edge_bps=150.0,
    sharpe_ratio=1.5,
    win_rate=0.60,
    avg_win_bps=200.0,
    avg_loss_bps=100.0,
    total_trades=100,
    timestamp=datetime.now(timezone.utc),
)

# Combine outputs
output = combiner.combine(
    engine_outputs=[output1, output2, output3],
    engine_ids=["trend_engine", "range_engine", "breakout_engine"],
    regime="TREND",
    horizon=TradingHorizon.SWING,
)

# Tune hyperparameters
if combiner.should_tune_hyperparameters():
    performance_history = [...]  # Get performance history
    best_params = combiner.tune_hyperparameters("trend_engine", performance_history)

# Get top engines
top_engines = combiner.get_top_engines(n=10)
```

## Enhanced Feature Builder

### `EnhancedFeatureBuilder`

Feature builder with alternative data sources.

```python
from src.shared.features.enhanced_feature_builder import EnhancedFeatureBuilder, AlternativeData

builder = EnhancedFeatureBuilder(config={
    "indicator_set": {
        "rsi": {"window": 14},
        "ema": {"window": 20},
        "volatility": {"window": 20},
        "momentum": {"window": 10},
        "trend_strength": {"window": 20},
        "support_resistance": {"window": 20},
    },
    "fill_rules": {"forward_fill_max_gaps": 5},
    "normalization": {"type": "min_max"},
})

# Build features with alternative data
alternative_data = AlternativeData(
    order_book_imbalance=0.1,
    large_order_flow=1000.0,
    iceberg_orders=5,
    wallet_transfers=5000.0,
    exchange_inflows=10000.0,
    exchange_outflows=8000.0,
    whale_movements=20000.0,
    funding_rate=0.01,
    funding_rate_skew=0.005,
    long_short_ratio=1.2,
    news_sentiment=0.6,
)

features_df = builder.build_features(
    df=candles_df,
    symbol="BTCUSDT",
    alternative_data=alternative_data,
)
```

## Engine Plugins

### `TrendEngine`

Trend engine with momentum breakout filters.

```python
from src.shared.engines.plugins import TrendEngine

engine = TrendEngine(config={
    "window": 20,
    "momentum_threshold": 0.02,
    "volatility_threshold": 0.05,
    "trailing_entry": True,
})

output = engine.infer(input_data)
```

### `RangeEngine`

Range engine with adaptive support/resistance detection.

```python
from src.shared.engines.plugins import RangeEngine

engine = RangeEngine(config={
    "window": 30,
    "range_threshold": 0.02,
    "support_resistance_tolerance": 0.005,
    "adaptive_detection": True,
})

output = engine.infer(input_data)
```

### `BreakoutEngine`

Breakout engine with volume-flow confirmation.

```python
from src.shared.engines.plugins import BreakoutEngine

engine = BreakoutEngine(config={
    "window": 20,
    "breakout_threshold": 0.02,
    "volume_multiplier": 1.5,
    "volatility_contraction_threshold": 0.01,
    "depth_breakout": True,
})

output = engine.infer(input_data)
```

## Configuration

### Config.yaml

See `config.yaml` for complete configuration options:

```yaml
swing_trading:
  enabled: true
  default_stop_loss_bps: 200.0
  use_trailing_stop: true
  trailing_stop_distance_bps: 100.0
  take_profit_levels:
    - [200.0, 0.30]
    - [400.0, 0.40]
    - [600.0, 0.20]
  max_holding_hours: null
  max_funding_cost_bps: 500.0
  exit_on_panic: true

portfolio_allocation:
  scalp_max_allocation_pct: 20.0
  swing_max_allocation_pct: 40.0
  position_max_allocation_pct: 30.0
  core_max_allocation_pct: 50.0
  scalp_max_positions: 5
  swing_max_positions: 3
  position_max_positions: 2
  core_max_positions: 3

enhanced_costs:
  funding_rate_bps_per_8h: 1.0
  borrow_rate_bps_per_day: 0.0
  overnight_risk_multiplier: 1.2
  liquidity_decay_factor: 0.95
  spread_widening_bps_per_day: 0.5
  include_funding_for_swing: true
  include_funding_for_position: true
  include_funding_for_core: true

regime_classifier:
  panic_allows_swing: false
  panic_allows_position: false
  illiquid_allows_swing: false
  illiquid_allows_position: false
  panic_risk_multiplier: 2.0
  illiquid_risk_multiplier: 1.5
  high_volatility_risk_multiplier: 1.3
```

## Testing

### Unit Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_enhanced_engine_interface.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

### Integration Tests

```bash
# Run integration tests
pytest tests/integration/ -v

# Run specific integration test
pytest tests/integration/test_swing_trading_integration.py -v
```

## Examples

See `tests/integration/test_swing_trading_integration.py` for complete workflow examples.

## Conclusion

This API reference provides comprehensive documentation for all swing trading enhancement modules. For more details, see the individual module documentation and usage examples.

