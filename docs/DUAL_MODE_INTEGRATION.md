# Dual-Mode Trading Integration Guide

This guide explains how to integrate the dual-mode trading system into your Huracan Engine.

## Overview

The dual-mode system allows you to run **short-hold (scalp)** and **long-hold (swing)** trading modes concurrently:

- **Short-Hold**: Fast scalps targeting £1-£2 profits with tight stops
- **Long-Hold**: Swing trades that hold through dips and maximize leaders (ETH/SOL/BTC)

## Quick Start

### Option 1: Standalone Demo (Recommended First)

Run the standalone demo to see the system in action without any integration:

```bash
cd /home/user/engine
python examples/dual_mode_standalone_demo.py
```

This will simulate:
1. Opening scalp and swing positions on ETH
2. Managing concurrent positions
3. Adding to positions on dips
4. Scaling out in profit
5. Safety rail monitoring

### Option 2: Integration Example

See how to integrate with your existing pipeline:

```bash
python examples/dual_mode_integration_example.py
```

## Architecture

```
                    ┌─────────────────────────────┐
                    │   Dual-Mode Coordinator     │
                    └──────────┬──────────────────┘
                               │
              ┌────────────────┴────────────────┐
              │                                  │
    ┌─────────▼─────────┐            ┌─────────▼─────────┐
    │  Short-Hold Gate  │            │  Long-Hold Gate   │
    │   (Scalp Mode)    │            │   (Swing Mode)    │
    └─────────┬─────────┘            └─────────┬─────────┘
              │                                  │
    ┌─────────▼─────────┐            ┌─────────▼─────────┐
    │   book_short      │            │   book_long       │
    │                   │            │                   │
    │ - Fast entries    │            │ - HTF bias req    │
    │ - £1-£2 targets   │            │ - Add on dips     │
    │ - Tight stops     │            │ - Scale outs      │
    │ - Maker bias      │            │ - Trailing stops  │
    └───────────────────┘            └───────────────────┘
```

## Integration Steps

### Step 1: Add the Adapter

```python
from src.cloud.training.integrations.dual_mode_adapter import (
    DualModeAdapter,
    DualModeConfig
)

# Create configuration
config = DualModeConfig(
    enabled=True,
    total_capital_gbp=10000.0,
    max_short_heat_pct=0.20,  # 20% for scalps
    max_long_heat_pct=0.50,   # 50% for swings
)

# Initialize adapter
dual_mode = DualModeAdapter(config=config)
```

### Step 2: Evaluate Signals

```python
# When you have a trading signal
should_enter, mode, reason = dual_mode.evaluate_for_entry(
    symbol="ETH",
    features=features_dict,
    regime="trend",
    confidence=0.70,
    current_price=2000.0,
)

if should_enter:
    print(f"Enter {mode.value} position: {reason}")
```

### Step 3: Open Positions

```python
if mode == TradingMode.SHORT_HOLD:
    # Scalp parameters
    size_gbp = 200.0
    stop_loss_bps = -10.0
    take_profit_bps = 15.0
else:  # LONG_HOLD
    # Swing parameters
    size_gbp = 1000.0
    stop_loss_bps = -150.0
    take_profit_bps = None  # Will scale out instead

dual_mode.open_position(
    symbol=symbol,
    mode=mode,
    entry_price=current_price,
    size_gbp=size_gbp,
    stop_loss_bps=stop_loss_bps,
    take_profit_bps=take_profit_bps,
)
```

### Step 4: Update Positions

```python
# On each tick/candle
actions = dual_mode.update_positions(
    symbol=symbol,
    current_price=current_price,
    features=features_dict,
    regime=regime,
)

# Actions might include:
# - CLOSED_long_hold_SAFETY
# - ADDED_long_hold
# - SCALED_OUT_long_hold
# - EXITED_short_hold_tp
# - TRAIL_UPDATED_long_hold
```

### Step 5: Get Statistics

```python
stats = dual_mode.get_statistics()

print(f"Short-hold P&L: £{stats['short_hold']['realized_pnl_gbp']}")
print(f"Long-hold P&L: £{stats['long_hold']['realized_pnl_gbp']}")
print(f"Total trades: {stats['routing']['total_signals']}")
```

## Configuration

### Asset Profiles

Configure which assets run which modes in `config/dual_mode.yaml`:

```yaml
dual_mode:
  assets:
    ETH:
      mode: both  # Run both scalp and swing
      short_hold:
        max_book_pct: 0.10
        target_profit_bps: 15.0
      long_hold:
        max_book_pct: 0.35
        add_grid_bps: [-150.0, -300.0]  # Add at -1.5% and -3%
        trail_style: chandelier_atr_3
        tp_multipliers: [1.0, 1.8, 2.8]  # Scale out levels

    DOGE:
      mode: short_hold  # Scalps only
```

### Safety Rails

Prevent "bag-holding forever" with automatic fail-safes:

```yaml
safety_rails:
  max_floating_dd_bps: 500.0  # Max -5% adverse move
  max_hold_days: 7.0           # Time stop
  drift_check_enabled: true    # Monitor signal quality
  vol_spike_threshold: 2.0     # Clamp adds on vol spikes
```

## PPO Integration

### Enhanced State

The system automatically enhances your trading state with dual-mode fields:

```python
state = dual_mode.get_trading_state_for_rl(
    symbol=symbol,
    market_features=features_array,
    base_state=base_state,
)

# State now includes:
# - state.trading_mode: "short_hold" or "long_hold"
# - state.has_short_position: bool
# - state.has_long_position: bool
# - state.short_position_pnl_bps: float
# - state.long_position_pnl_bps: float
# - state.long_position_age_hours: float
# - state.num_adds: int
# - state.be_lock_active: bool
# - state.trail_active: bool
```

### New Actions

The PPO agent now has additional actions:

- `SCRATCH`: Fast exit for short-hold (minimize loss)
- `ADD_GRID`: Add to position (DCA for long-hold)
- `SCALE_OUT`: Partial exit (long-hold)
- `TRAIL_RUNNER`: Activate trailing stop (long-hold)

Execute them through the adapter:

```python
success, message = dual_mode.execute_rl_action(
    symbol=symbol,
    action=TradingAction.SCALE_OUT,
    current_price=current_price,
    features=features_dict,
)
```

## Risk Management

### Per-Mode Limits

The system enforces separate limits for each mode:

- **Short-hold book**: Max 20% of capital
- **Long-hold book**: Max 50% of capital
- **Per-asset**: Max 45% combined across both books

### Safety Rails

Continuous monitoring prevents runaway losses:

1. **Drawdown Rail**: Force reduce if adverse move exceeds limits + regime/HTF breaks
2. **Time Rail**: Exit positions that don't resolve within max hold time
3. **Feature Drift**: Exit if signal quality degrades during hold
4. **Event Guards**: Clamp adds and tighten trails on volatility spikes

### Conflict Resolution

When both modes want the same asset:

- Calculate total exposure across both books
- Enforce per-asset cap (default 45%)
- Prioritize based on confidence and capacity
- Log all conflict resolutions for analysis

## Performance Tracking

### Per-Mode Statistics

```python
stats = dual_mode.get_statistics()

# Short-hold metrics
short_stats = stats['short_hold']
print(f"Scalps:")
print(f"  Trades: {short_stats['num_trades']}")
print(f"  Win rate: {short_stats['win_rate']:.1%}")
print(f"  Avg P&L: £{stats['adapter']['avg_pnl_short']:.2f}")

# Long-hold metrics
long_stats = stats['long_hold']
print(f"Swings:")
print(f"  Trades: {long_stats['num_trades']}")
print(f"  Win rate: {long_stats['win_rate']:.1%}")
print(f"  Avg P&L: £{stats['adapter']['avg_pnl_long']:.2f}")
```

### Routing Analysis

```python
routing = stats['routing']
print(f"Routing efficiency:")
print(f"  Total signals: {routing['total_signals']}")
print(f"  Short routed: {routing['short_routed']}")
print(f"  Long routed: {routing['long_routed']}")
print(f"  No route: {routing['no_route']}")
```

### Safety Rail Monitoring

```python
safety = stats['safety_rails']
print(f"Safety violations:")
print(f"  Total: {safety['total']}")
print(f"  By type: {safety['by_type']}")
print(f"  By severity: {safety['by_severity']}")
```

## Testing

### Unit Tests

Run the comprehensive test suite:

```bash
pytest tests/test_dual_mode_trading.py -v
```

### Integration Tests

Test with your existing pipeline:

```bash
python examples/dual_mode_integration_example.py
```

## Best Practices

### 1. Start Conservative

Begin with conservative settings and gradually increase:

```python
config = DualModeConfig(
    enabled=True,
    total_capital_gbp=5000.0,  # Start smaller
    max_short_heat_pct=0.10,    # 10% scalps
    max_long_heat_pct=0.30,     # 30% swings
)
```

### 2. Monitor Safety Rails

Check safety rail violations regularly:

```python
safety_summary = coordinator.safety_monitor.get_violation_summary(hours=24)
if safety_summary['by_severity'].get('critical', 0) > 5:
    print("⚠️  Multiple critical violations - review system")
```

### 3. Calibrate Per Asset

Different assets need different parameters:

- ETH: Larger swings, more adds
- BTC: Smaller swings, no adds
- Alts: Scalps only initially

### 4. Review Routing Decisions

Log and analyze routing decisions:

```python
config = DualModeConfig(
    log_routing_decisions=True,
    log_conflicts=True,
)
```

### 5. Daily Resets

Reset statistics daily for clean metrics:

```python
dual_mode.reset_daily()  # Call at start of trading day
```

## Troubleshooting

### Issue: No Signals Routing

**Possible causes:**
- Gates too strict (confidence thresholds, regime requirements)
- Book capacity full
- Asset not enabled for mode

**Solution:**
```python
# Check signal evaluation
signal = coordinator.evaluate_signal(context)
print(f"Short OK: {signal.short_ok} - {signal.short_reason}")
print(f"Long OK: {signal.long_ok} - {signal.long_reason}")
```

### Issue: Positions Not Adding

**Possible causes:**
- Add grid levels not reached
- Safety rails clamping adds
- Max adds already executed

**Solution:**
```python
# Check add conditions
should_add, reason, price = coordinator.should_add_to_position(symbol, context)
print(f"Should add: {should_add} - {reason}")

# Check safety rails
if coordinator.safety_monitor.should_clamp_adds(symbol):
    print("Adds clamped by safety rails")
```

### Issue: High Safety Violations

**Possible causes:**
- Thresholds too tight
- Volatile market conditions
- Holding against trend

**Solution:**
- Review and adjust safety rail config
- Check regime detection accuracy
- Ensure HTF bias is reliable

## Advanced Usage

### Custom Policies

Create custom policies for specific assets:

```python
from src.cloud.training.models.mode_policies import LongHoldPolicy, LongHoldConfig

# Create custom ETH swing policy
eth_config = LongHoldConfig(
    max_book_pct=0.40,
    add_grid_bps=[-100.0, -200.0, -300.0],  # 3 add levels
    trail_style=TrailStyle.CHANDELIER_ATR_3,
    tp_multipliers=[0.8, 1.5, 2.5, 4.0],    # 4 scale-out levels
)

eth_policy = LongHoldPolicy(eth_config)
```

### Dynamic Configuration

Adjust parameters based on market conditions:

```python
if regime == "high_volatility":
    # Tighten safety rails
    coordinator.safety_monitor.config.max_floating_dd_bps = 300.0
elif regime == "trending":
    # Relax for trends
    coordinator.safety_monitor.config.max_floating_dd_bps = 600.0
```

### Multi-Asset Coordination

Use MultiSymbolCoordinator for cross-asset logic:

```python
from src.cloud.training.models.multi_symbol_coordinator import MultiSymbolCoordinator

multi_coord = MultiSymbolCoordinator()
# Check correlated exposure before entering
can_enter, reason = multi_coord.can_enter_position(
    symbol="SOL",
    position_size_gbp=1000.0,
    total_capital_gbp=10000.0,
)
```

## Support

For issues or questions:
- Check the examples in `/examples`
- Review test cases in `/tests/test_dual_mode_trading.py`
- Read the source code documentation
- Open an issue on GitHub

## Next Steps

1. ✅ Run the standalone demo
2. ✅ Review the integration example
3. ✅ Run the test suite
4. ⏭️  Integrate into your pipeline
5. ⏭️  Backtest with historical data
6. ⏭️  Paper trade for validation
7. ⏭️  Deploy to production

🚀 **You're ready to run both scalps and swings concurrently!**
