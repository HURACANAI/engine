# CoreBook Strategy Module

## Overview

The CoreBook strategy module handles the top three coins (BTC, ETH, SOL) with separate CoreBook logic. It never sells at a loss, triggers DCA on drops, and executes partial sells on profit thresholds.

## Key Features

1. **Separate CoreBook**: Distinct from short-term trading book
2. **No Loss Sales**: Never sells coins at a loss
3. **Dollar-Cost Averaging (DCA)**: Triggers DCA buys when price drops below average cost
4. **Partial Sells**: Locks in profits when profit threshold is met
5. **Exposure Caps**: Max exposure per coin, max incremental buys, cooldown periods
6. **Risk Safeguards**: Stop DCA on panic, check liquidity and spread
7. **Telegram Control**: Commands for status, cap, add, trim, auto

## Architecture

### Core Components

1. **CoreBookStrategy** (`core_strategy.py`)
   - Main strategy class
   - Evaluates trading actions
   - Manages CoreBook state
   - Enforces rules and safeguards

2. **CoreBookEntry** (`core_strategy.py`)
   - Per-coin state
   - Units held, average cost, exposure limits
   - DCA triggers and partial sell targets

3. **CoreBookState** (`core_strategy.py`)
   - Overall state for all coins
   - Auto trading enabled/disabled
   - State persistence

4. **TelegramHandler** (`telegram_handler.py`)
   - Handles Telegram commands
   - Status, cap, add, trim, auto

## Configuration

### CoreBookConfig

```python
config = CoreBookConfig(
    default_coins=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
    max_exposure_pct_per_coin=10.0,  # Max exposure as % of portfolio
    max_incremental_buy_pct=2.0,  # Max incremental buy as % of portfolio
    dca_drop_pct=5.0,  # DCA trigger when price drops this % below average cost
    max_dca_buys=10,  # Maximum number of DCA buys
    dca_cooldown_minutes=60,  # Cooldown between DCA buys
    profit_threshold_absolute=1.0,  # Absolute profit threshold (e.g., £1)
    profit_threshold_pct=5.0,  # Percentage profit threshold
    partial_sell_pct=25.0,  # Percentage of position to sell on profit
    action_cooldown_minutes=15,  # Cooldown between actions
    stop_dca_on_panic=True,  # Stop DCA on panic regime
    min_liquidity_usd=1_000_000.0,  # Minimum liquidity to trade
    max_spread_bps=50.0,  # Maximum spread in basis points
    allowable_coins=["BTCUSDT", "ETHUSDT", "SOLUSDT"],  # Allowable coins
)
```

## Usage

### Initialize Strategy

```python
from src.cloud.trading.core_strategy.core_strategy import CoreBookStrategy, CoreBookConfig

config = CoreBookConfig()
strategy = CoreBookStrategy(config=config, state_file="core_book.json")
```

### Evaluate Trading Actions

```python
from src.cloud.trading.core_strategy.core_strategy import MarketRegime

# Evaluate for a coin
action = strategy.evaluate(
    symbol="BTCUSDT",
    current_price=45000.0,
    portfolio_value=10000.0,
    market_regime=MarketRegime.NORMAL,
    liquidity_usd=10_000_000.0,
    spread_bps=5.0,
)

if action:
    print(f"Action: {action.action_type.value}, Units: {action.units}, Price: {action.price}")
```

### Execute Manual Actions

```python
# Execute buy
strategy.execute_buy("BTCUSDT", units=0.05, price=45000.0, portfolio_value=10000.0)

# Execute sell (only if profitable)
strategy.execute_sell("BTCUSDT", units=0.01, price=46000.0)

# Set exposure cap
strategy.set_exposure_cap("BTCUSDT", exposure_pct=15.0)

# Add coin
strategy.add_coin("ADAUSDT", exposure_pct=5.0)

# Trim position
strategy.trim_position("BTCUSDT", trim_pct=25.0)

# Set auto trading
strategy.set_auto_trading(True)
```

### Telegram Commands

```python
from src.cloud.trading.core_strategy.telegram_handler import CoreBookTelegramHandler

handler = CoreBookTelegramHandler(strategy)

# Handle commands
response = handler.handle_command("status", [])
response = handler.handle_command("cap", ["BTCUSDT", "15.0"])
response = handler.handle_command("add", ["ADAUSDT", "5.0"])
response = handler.handle_command("trim", ["BTCUSDT", "25.0"])
response = handler.handle_command("auto", ["on"])
```

## Core Book JSON Schema

### Example: `core_book.json`

```json
{
  "coins": {
    "BTCUSDT": {
      "symbol": "BTCUSDT",
      "units_held": 0.05,
      "average_cost_price": 45000.0,
      "total_cost_basis": 2250.0,
      "next_dca_trigger_price": 42750.0,
      "partial_sell_target_price": 47250.0,
      "total_exposure_limit_pct": 10.0,
      "last_action_timestamp": "2025-01-01T02:00:00Z",
      "last_action_type": "buy",
      "cooldown_until": null,
      "dca_count": 0,
      "max_dca_buys": 10,
      "metadata": {}
    }
  },
  "auto_trading_enabled": true,
  "last_updated": "2025-01-01T02:00:00Z",
  "metadata": {}
}
```

## Strategy Logic

### 1. DCA Buy Trigger

- **Condition**: Price drops below average cost by `dca_drop_pct`
- **Action**: Buy additional units (scaled by drop %)
- **Constraints**:
  - Must not exceed exposure cap
  - Must not exceed max DCA buys
  - Must respect cooldown period
  - Must pass risk safeguards (liquidity, spread, regime)

### 2. Partial Sell Trigger

- **Condition**: Price rises above profit threshold (absolute or %)
- **Action**: Sell `partial_sell_pct` of position
- **Constraints**:
  - Never sell at a loss
  - Must respect cooldown period
  - Must not sell entire position

### 3. Exposure Caps

- **Per Coin**: Max exposure as % of portfolio
- **Incremental Buy**: Max incremental buy as % of portfolio
- **Enforcement**: Blocks actions that would exceed caps

### 4. Risk Safeguards

- **Panic Regime**: Stop DCA on panic
- **Liquidity**: Minimum liquidity required
- **Spread**: Maximum spread allowed
- **Hedging**: Optional hedge on drawdown (future)

## Integration

### With Existing System

```python
from src.cloud.trading.core_strategy.core_strategy import CoreBookStrategy, CoreBookConfig
from src.shared.regime import RegimeClassifier
from src.shared.costs import CostCalculator

# Initialize components
config = CoreBookConfig()
strategy = CoreBookStrategy(config=config)
regime_classifier = RegimeClassifier()
cost_calculator = CostCalculator()

# Get current price and portfolio state
current_price = get_current_price("BTCUSDT")
portfolio_value = get_portfolio_value()
regime = regime_classifier.classify(candles_df, "BTCUSDT")
liquidity = get_liquidity("BTCUSDT")
spread = get_spread("BTCUSDT")

# Evaluate action
action = strategy.evaluate(
    symbol="BTCUSDT",
    current_price=current_price,
    portfolio_value=portfolio_value,
    market_regime=regime.regime,
    liquidity_usd=liquidity,
    spread_bps=spread,
)

# Execute action if needed
if action:
    execute_trade(action)
```

## Telegram Commands

### `/core status [coin]`

Get CoreBook status for all coins or a specific coin.

**Examples**:
```
/core status
/core status BTCUSDT
```

### `/core cap <coin> <percent>`

Set exposure cap for a coin.

**Examples**:
```
/core cap BTCUSDT 15.0
/core cap ETHUSDT 10.0
```

### `/core add <coin> <percent>`

Add a coin to CoreBook.

**Examples**:
```
/core add ADAUSDT 5.0
/core add SOLUSDT 10.0
```

### `/core trim <coin> [percent]`

Trim a position (sell a percentage).

**Examples**:
```
/core trim BTCUSDT
/core trim ETHUSDT 25.0
```

### `/core auto on/off`

Enable or disable auto trading.

**Examples**:
```
/core auto on
/core auto off
```

## Testing

### Run Tests

```bash
# Run all tests
pytest tests/test_core_strategy.py -v

# Run specific test
pytest tests/test_core_strategy.py::test_no_sell_at_loss -v
```

### Test Coverage

- ✅ No sell at loss enforcement
- ✅ DCA triggering logic
- ✅ Partial sell logic
- ✅ Exposure cap enforcement
- ✅ Cooldown periods
- ✅ Risk safeguards
- ✅ State save and load
- ✅ Average cost updates
- ✅ Trigger updates

## Configuration Parameters

### Default Values

- **Default Coins**: ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
- **Max Exposure per Coin**: 10.0% of portfolio
- **Max Incremental Buy**: 2.0% of portfolio
- **DCA Drop %**: 5.0% below average cost
- **Max DCA Buys**: 10
- **DCA Cooldown**: 60 minutes
- **Profit Threshold Absolute**: £1.0
- **Profit Threshold %**: 5.0%
- **Partial Sell %**: 25.0% of position
- **Action Cooldown**: 15 minutes
- **Stop DCA on Panic**: True
- **Min Liquidity**: $1,000,000 USD
- **Max Spread**: 50.0 basis points

## Deployment

### 1. Install Dependencies

```bash
pip install structlog
```

### 2. Configure

Edit `config.yaml` or create `CoreBookConfig` with your settings.

### 3. Initialize Strategy

```python
from src.cloud.trading.core_strategy.core_strategy import CoreBookStrategy, CoreBookConfig

config = CoreBookConfig(
    default_coins=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
    max_exposure_pct_per_coin=10.0,
    # ... other settings
)

strategy = CoreBookStrategy(config=config, state_file="core_book.json")
```

### 4. Integrate with System

Add CoreBook evaluation to your trading loop:

```python
# In your trading loop
for symbol in core_coins:
    action = strategy.evaluate(
        symbol=symbol,
        current_price=get_current_price(symbol),
        portfolio_value=get_portfolio_value(),
        market_regime=get_market_regime(symbol),
        liquidity_usd=get_liquidity(symbol),
        spread_bps=get_spread(symbol),
    )
    
    if action:
        execute_trade(action)
```

### 5. Telegram Integration

```python
from src.cloud.trading.core_strategy.telegram_handler import CoreBookTelegramHandler

handler = CoreBookTelegramHandler(strategy)

# In your Telegram bot
@bot.message_handler(commands=['core'])
def handle_core_command(message):
    response = handler.process_message(message.text)
    bot.reply_to(message, response)
```

## File Structure

```
src/cloud/trading/core_strategy/
├── __init__.py
├── core_strategy.py          # Main strategy implementation
├── telegram_handler.py       # Telegram command handler
├── core_book_example.json    # Example core book state
└── README.md                 # This file

tests/
└── test_core_strategy.py     # Unit tests
```

## Safety Features

1. **No Loss Sales**: Never sells at a loss
2. **Exposure Caps**: Limits total exposure per coin
3. **Cooldowns**: Prevents excessive trading
4. **Risk Safeguards**: Checks liquidity, spread, regime
5. **State Persistence**: Saves state after each action
6. **Input Validation**: Validates all inputs
7. **Error Handling**: Comprehensive error handling

## Extensibility

### Adding New Coins

```python
# Add coin to allowable list
config.allowable_coins.append("ADAUSDT")

# Add coin to CoreBook
strategy.add_coin("ADAUSDT", exposure_pct=5.0)
```

### Customizing Configuration

```python
config = CoreBookConfig(
    default_coins=["BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT"],
    max_exposure_pct_per_coin=15.0,
    dca_drop_pct=3.0,
    profit_threshold_pct=3.0,
    # ... other customizations
)
```

## Conclusion

The CoreBook strategy module provides a robust, safe, and extensible system for managing top coins with DCA and profit-taking logic. It integrates seamlessly with the existing system and provides Telegram control for manual overrides.

