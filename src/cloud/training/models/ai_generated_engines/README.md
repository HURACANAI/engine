# AI-Generated Alpha Engines

This directory contains AlphaEngines automatically generated from the strategy-research pipeline.

## How They Get Here

1. **RBI Agent** researches and backtests trading strategies
2. **Strategy Translator** converts backtests into AlphaEngine code
3. **Engines land here** as Python files
4. **Engine orchestrator** can load them dynamically

## File Naming Convention

```
{engine_name}_{timestamp}.py
```

Example:
```
aigeneratedengine_rsi_reversal_20251108_143022.py
```

## Engine Structure

Each file contains a single AlphaEngine subclass:

```python
from cloud.training.models.alpha_engines import AlphaEngine, AlphaSignal

class AIGeneratedEngine_RSI_Reversal(AlphaEngine):
    def __init__(self):
        super().__init__(name="rsi_reversal")

    def generate_signal(self, symbol, df, regime, meta):
        # Strategy logic here
        return AlphaSignal(direction=..., confidence=...)
```

## Integration with Engine

### Manual Integration

Add to `alpha_engines.py`:

```python
from .ai_generated_engines.aigeneratedengine_rsi_reversal_20251108_143022 import AIGeneratedEngine_RSI_Reversal

# In get_all_engines():
AIGeneratedEngine_RSI_Reversal(),
```

### Dynamic Loading (Coming Soon)

```python
from .ai_generated_engines import load_ai_engines

def get_all_engines():
    return [
        # ... existing engines ...
    ] + load_ai_engines()
```

## Validation Requirements

Before an AI-generated engine goes into production:

1. ✅ **Backtest performance** >5% return
2. ✅ **Walk-forward validation** passes
3. ✅ **Code review** - manual inspection
4. ✅ **Paper trading** - 2-4 weeks shadow trading
5. ✅ **Regime testing** - works in TREND, RANGE, PANIC
6. ✅ **Risk metrics** - Sharpe >1.5, max DD <20%

## Performance Tracking

Track AI-generated engines separately:

```python
metadata = {
    "engine_type": "ai_generated",
    "source": "rbi_agent",
    "backtest_return": 8.5,
    "generation_date": "2025-11-08",
    "validation_status": "pending"
}
```

## Best Practices

- **Test thoroughly** before production
- **Monitor performance** vs. baseline
- **Version control** engine files
- **Document** what makes each strategy unique
- **Deprecate** underperforming engines

## Current Engines

<!-- Auto-updated by deployment script -->

| Engine Name | Generated | Status | Backtest Return | Live Performance |
|-------------|-----------|--------|-----------------|------------------|
| (None yet)  | -         | -      | -               | -                |

## Notes

- Engines are **NOT** automatically deployed to production
- Manual review required for each engine
- Failed engines are archived, not deleted
- Successful patterns feed back into RBI agent
