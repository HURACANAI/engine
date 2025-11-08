# Data Contract Schemas

This directory contains sample exports of all data contracts used in the Huracan Engine.

## Available Schemas

### 1. Candle Schema (`candles.json`)
OHLCV market data candles.

**Fields:**
- `timestamp`: Candle timestamp (UTC)
- `symbol`: Trading symbol (e.g., "BTC")
- `open`, `high`, `low`, `close`: OHLC prices
- `volume`: Trading volume
- `timeframe`: Candle timeframe (e.g., "5m", "1h")

**Example:**
```json
{
  "timestamp": "2025-11-08T12:00:00Z",
  "symbol": "BTC",
  "open": 50000.0,
  "high": 50500.0,
  "low": 49800.0,
  "close": 50200.0,
  "volume": 1250.5,
  "timeframe": "5m"
}
```

---

### 2. Feature Schema (`features.json`)
Computed feature vectors.

**Fields:**
- `timestamp`: Feature timestamp (UTC)
- `symbol`: Trading symbol
- `features`: Dict of feature values
- `feature_set_id`: Feature set identifier
- `metadata`: Optional metadata

**Example:**
```json
{
  "timestamp": "2025-11-08T12:00:00Z",
  "symbol": "BTC",
  "features": {
    "rsi_14": 65.5,
    "macd": 0.023,
    "bb_width": 0.015
  },
  "feature_set_id": "fs_20251108_abc123",
  "metadata": {"regime": "trending"}
}
```

---

### 3. Signal Schema (`signals.json`)
Trading signals from models.

**Fields:**
- `timestamp`: Signal generation time
- `symbol`: Trading symbol
- `direction`: "long", "short", or "neutral"
- `confidence`: Model confidence [0-1]
- `stop_loss_bps`: Stop loss in basis points
- `take_profit_bps`: Take profit in basis points
- `model_id`: Model identifier
- `regime`: Market regime
- `metadata`: Optional metadata

**Example:**
```json
{
  "timestamp": "2025-11-08T12:00:00Z",
  "symbol": "BTC",
  "direction": "long",
  "confidence": 0.75,
  "stop_loss_bps": 50,
  "take_profit_bps": 20,
  "model_id": "btc_trend_v47",
  "regime": "trending"
}
```

---

### 4. Gate Verdict Schema (`gate_verdict.json`)
Model evaluation results from rule-based gates.

**Fields:**
- `model_id`: Model identifier
- `evaluated_at`: Evaluation timestamp
- `status`: "publish", "shadow", "reject", or "pending"
- `meta_weight`: Computed meta weight [0-1]
- `passed_gates`: List of gates that passed
- `failed_gates`: List of gates that failed
- `warnings`: Warning messages
- `gate_details`: Detailed results per gate

**Example:**
```json
{
  "model_id": "btc_trend_v47",
  "evaluated_at": "2025-11-08T12:00:00Z",
  "status": "publish",
  "meta_weight": 0.15,
  "passed_gates": ["minimum_sharpe", "maximum_drawdown", "stress_tests"],
  "failed_gates": [],
  "warnings": [],
  "gate_details": {
    "minimum_sharpe": {"value": 1.5, "threshold": 0.5, "passed": true}
  }
}
```

---

## Usage

### Python

```python
from shared.contracts import (
    validate_candles,
    validate_features,
    validate_signals,
    validate_gate_verdict,
    Signal,
    GateVerdict
)

# Validate dataframes
candles_df = pd.read_parquet("candles.parquet")
validated = validate_candles(candles_df)  # Raises ValidationError if invalid

# Use typed models
signal = Signal(
    timestamp=datetime.now(),
    symbol="BTC",
    direction="long",
    confidence=0.75,
    stop_loss_bps=50,
    take_profit_bps=20,
    model_id="btc_v47"
)
```

### Validation Rules

All contracts enforce:
- **Required fields**: Must be present
- **Type checking**: Correct types (str, float, datetime, etc.)
- **Range validation**: Values within valid ranges
- **Business logic**: Domain-specific rules (e.g., high >= low)
- **Hard fail on mismatch**: Invalid data blocks processing

---

## Schema Versioning

Schemas are versioned alongside the code. Breaking changes require:
1. New schema version
2. Migration script
3. Backward compatibility period

---

## Sample Files

- `candles_sample.json`: Sample candle data
- `features_sample.json`: Sample feature data
- `signals_sample.json`: Sample signal data
- `gate_verdict_sample.json`: Sample gate verdict

These files are auto-generated from the Pydantic schemas.
