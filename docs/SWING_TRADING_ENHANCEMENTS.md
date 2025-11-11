# Swing Trading Enhancements Guide

## Overview

This document describes the comprehensive enhancements to support swing trading (hours to days/weeks) in the Huracan trading system. The enhancements include:

1. **Enhanced Engine Interface**: Support for long horizons (hours to days/weeks)
2. **Enhanced Cost Calculator**: Holding context with funding costs, overnight risk, liquidity decay
3. **Swing Position Manager**: Stop-loss, take-profit curves, holding logic
4. **Horizon-Based Portfolio Allocator**: Allocation across scalps, swings, positions, core
5. **Enhanced Regime Classifier**: Regime gating for swing trading
6. **Configuration Schema**: Comprehensive configuration for swing trading

## Key Features

### 1. Time Horizons

The system now supports four trading horizons:

- **SCALP**: Minutes to hours (≤ 1 hour)
- **SWING**: Hours to days (≤ 1 day)
- **POSITION**: Days to weeks (≤ 1 week)
- **CORE**: Long-term holds (weeks to months, > 1 week)

### 2. Enhanced Cost Calculator

The enhanced cost calculator includes:

- **Funding Costs**: Calculated based on holding period (8-hour periods)
- **Overnight Risk**: Multiplier for positions held > 24 hours
- **Liquidity Decay**: Spread and slippage increase over time
- **Borrow Costs**: For margin trading (optional)

### 3. Swing Position Manager

Features:

- **Stop-Loss**: Fixed or trailing stop-loss levels
- **Take-Profit Curves**: Multiple partial exit levels
- **Holding Logic**: No forced exit due to short-term stop losses
- **Time Limits**: Optional maximum holding time
- **Funding Cost Limits**: Exit if funding cost exceeds threshold
- **Regime-Based Exits**: Exit on panic/illiquid regimes

### 4. Horizon-Based Portfolio Allocation

Allocation limits by horizon:

- **Scalp**: Max 20% allocation, 5 positions
- **Swing**: Max 40% allocation, 3 positions
- **Position**: Max 30% allocation, 2 positions
- **Core**: Max 50% allocation, 3 positions (can overlap)

### 5. Enhanced Regime Classifier

Regime gating rules:

- **Panic Regime**: Blocks swing/position trading, allows scalps
- **Illiquid Regime**: Blocks swing/position trading, allows scalps
- **Trend Regime**: Allows all horizons
- **Range Regime**: Allows all horizons
- **High Volatility**: Allows all horizons with risk multiplier

## Usage

### 1. Initialize Enhanced Engine Interface

```python
from src.shared.engines.enhanced_engine_interface import (
    BaseEnhancedEngine,
    EnhancedEngineInput,
    EnhancedEngineOutput,
    TradingHorizon,
    Direction,
)

class MySwingEngine(BaseEnhancedEngine):
    def __init__(self):
        super().__init__(
            engine_id="swing_engine_1",
            name="Swing Engine",
            supported_regimes=["TREND", "RANGE"],
            supported_horizons=[TradingHorizon.SWING, TradingHorizon.POSITION],
            default_horizon=TradingHorizon.SWING,
        )
    
    def infer(self, input_data: EnhancedEngineInput) -> EnhancedEngineOutput:
        # Your swing trading logic
        return EnhancedEngineOutput(
            direction=Direction.BUY,
            edge_bps_before_costs=150.0,
            confidence_0_1=0.75,
            horizon_minutes=24 * 60,  # 1 day
            horizon_type=TradingHorizon.SWING,
            stop_loss_bps=200.0,  # 2% stop loss
            take_profit_bps=400.0,  # 4% take profit
            trailing_stop_bps=100.0,  # 1% trailing stop
            position_size_multiplier=1.0,
            max_holding_hours=48.0,  # 2 days max
            funding_cost_estimate_bps=10.0,  # Estimated funding cost
        )
```

### 2. Use Enhanced Cost Calculator

```python
from src.shared.costs.enhanced_cost_calculator import (
    EnhancedCostCalculator,
    EnhancedCostModel,
    TradingHorizon,
)

calculator = EnhancedCostCalculator()

# Register cost model
cost_model = EnhancedCostModel(
    symbol="BTCUSDT",
    taker_fee_bps=4.0,
    maker_fee_bps=2.0,
    median_spread_bps=5.0,
    slippage_bps_per_sigma=2.0,
    min_notional=10.0,
    step_size=0.01,
    last_updated_utc=datetime.now(),
    funding_rate_bps_per_8h=1.0,
    borrow_rate_bps_per_day=0.0,
    overnight_risk_multiplier=1.2,
    liquidity_decay_factor=0.95,
    spread_widening_bps_per_day=0.5,
)

calculator.register_cost_model(cost_model)

# Calculate costs for swing trade
cost_breakdown = calculator.get_costs(
    symbol="BTCUSDT",
    timestamp=datetime.now(),
    order_type="maker",
    order_size_sigma=1.0,
    holding_hours=48.0,  # 2 days
    horizon_type=TradingHorizon.SWING,
    include_funding=True,
    include_borrow=False,
)

# Calculate net edge
net_edge = calculator.calculate_net_edge(
    symbol="BTCUSDT",
    edge_bps_before_costs=150.0,
    timestamp=datetime.now(),
    order_type="maker",
    order_size_sigma=1.0,
    holding_hours=48.0,
    horizon_type=TradingHorizon.SWING,
    include_funding=True,
)
```

### 3. Manage Swing Positions

```python
from src.shared.trading.swing_position_manager import (
    SwingPositionManager,
    SwingPositionConfig,
    TradingHorizon,
    Direction,
)

config = SwingPositionConfig(
    default_stop_loss_bps=200.0,
    use_trailing_stop=True,
    trailing_stop_distance_bps=100.0,
    take_profit_levels=[
        (200.0, 0.30),  # 30% at 2% profit
        (400.0, 0.40),  # 40% at 4% profit
        (600.0, 0.20),  # 20% at 6% profit
    ],
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
    take_profit_levels=[
        (200.0, 0.30),
        (400.0, 0.40),
        (600.0, 0.20),
    ],
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

if exit_action:
    # Execute exit
    print(f"Exit triggered: {exit_action['exit_reason']}")
```

### 4. Portfolio Allocation

```python
from src.shared.portfolio.horizon_portfolio_allocator import (
    HorizonPortfolioAllocator,
    HorizonPortfolioConfig,
    TradingHorizon,
)

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

# Initialize allocation
allocation = allocator.initialize(
    total_portfolio_value=10000.0,
    current_allocations={
        TradingHorizon.SCALP: 10.0,  # 10% allocated to scalps
        TradingHorizon.SWING: 20.0,  # 20% allocated to swings
    },
)

# Check if can open position
can_open, reason = allocator.can_open_position(
    horizon=TradingHorizon.SWING,
    position_size_usd=1000.0,
)

if can_open:
    # Allocate position
    allocator.allocate_position(
        horizon=TradingHorizon.SWING,
        position_size_usd=1000.0,
    )
```

### 5. Enhanced Regime Classifier

```python
from src.shared.regime.enhanced_regime_classifier import (
    EnhancedRegimeClassifier,
    RegimeGatingConfig,
    TradingHorizon,
)

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

## Configuration

### Config.yaml

```yaml
# Swing trading configuration
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
  exit_on_illiquid: false

# Horizon-based portfolio allocation
portfolio_allocation:
  scalp_max_allocation_pct: 20.0
  swing_max_allocation_pct: 40.0
  position_max_allocation_pct: 30.0
  core_max_allocation_pct: 50.0
  scalp_max_positions: 5
  swing_max_positions: 3
  position_max_positions: 2
  core_max_positions: 3

# Enhanced cost calculator
enhanced_costs:
  funding_rate_bps_per_8h: 1.0
  borrow_rate_bps_per_day: 0.0
  overnight_risk_multiplier: 1.2
  liquidity_decay_factor: 0.95
  spread_widening_bps_per_day: 0.5
  include_funding_for_swing: true
  include_funding_for_position: true
  include_funding_for_core: true
```

## Integration with Existing System

### 1. Engine Integration

Engines should use the enhanced engine interface:

```python
from src.shared.engines.enhanced_engine_interface import BaseEnhancedEngine

class MyEngine(BaseEnhancedEngine):
    # Implement infer() method
    pass
```

### 2. Cost Calculation

Use enhanced cost calculator for swing trades:

```python
from src.shared.costs.enhanced_cost_calculator import EnhancedCostCalculator

calculator = EnhancedCostCalculator()
net_edge = calculator.calculate_net_edge(
    symbol=symbol,
    edge_bps_before_costs=edge_bps,
    holding_hours=holding_hours,
    horizon_type=TradingHorizon.SWING,
)
```

### 3. Position Management

Use swing position manager for swing trades:

```python
from src.shared.trading.swing_position_manager import SwingPositionManager

manager = SwingPositionManager(config)
position = manager.open_position(...)
exit_action = manager.update_position(...)
```

### 4. Portfolio Allocation

Use horizon portfolio allocator:

```python
from src.shared.portfolio.horizon_portfolio_allocator import HorizonPortfolioAllocator

allocator = HorizonPortfolioAllocator(config)
can_open, reason = allocator.can_open_position(horizon, size)
```

## Best Practices

### 1. Horizon Selection

- Use **SCALP** for high-frequency, low-risk trades
- Use **SWING** for medium-term trends (hours to days)
- Use **POSITION** for longer-term trends (days to weeks)
- Use **CORE** for long-term holds (weeks to months)

### 2. Risk Management

- Set appropriate stop-loss levels (2% for swing, 3% for position)
- Use trailing stops for swing trades
- Set take-profit levels to lock in profits
- Monitor funding costs for longer holds
- Exit on panic/illiquid regimes

### 3. Portfolio Allocation

- Allocate 20% to scalps (high turnover)
- Allocate 40% to swings (medium turnover)
- Allocate 30% to positions (low turnover)
- Allocate 50% to core (very low turnover, can overlap)

### 4. Cost Awareness

- Include funding costs for swing/position trades
- Account for liquidity decay over time
- Use maker orders when possible (lower fees)
- Monitor overnight risk for positions > 24 hours

## Testing

### Unit Tests

```python
import pytest
from src.shared.trading.swing_position_manager import SwingPositionManager, SwingPositionConfig

def test_swing_position_manager():
    config = SwingPositionConfig()
    manager = SwingPositionManager(config)
    
    # Open position
    position = manager.open_position(
        symbol="BTCUSDT",
        direction=Direction.BUY,
        entry_price=45000.0,
        entry_size=0.1,
        horizon_type=TradingHorizon.SWING,
    )
    
    # Update position
    exit_action = manager.update_position(
        symbol="BTCUSDT",
        current_price=46000.0,
        current_regime="TREND",
    )
    
    assert exit_action is None  # No exit yet
```

## Conclusion

The swing trading enhancements provide a comprehensive framework for managing longer-term positions with proper risk management, cost awareness, and portfolio allocation. The system is designed to be modular, extensible, and integrated with the existing trading system.

