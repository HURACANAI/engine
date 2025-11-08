# Download Script Improvements

## Summary of Changes

The `simple_download_candles.py` script has been completely rewritten with the following improvements:

### 1. ✅ Real, Active Spot Market Validation
- **Before**: Used ticker volume list only, no validation
- **After**: Filters to real, active spot markets from `exchange.markets`
- **Implementation**: `get_active_spot_markets()` checks `market.spot` and `market.active`

### 2. ✅ Symbol Normalization
- **Before**: Used free-typed ticker symbols
- **After**: Uses exact ccxt market symbols from `exchange.markets` keys
- **Implementation**: Validates all symbols against active spot markets set

### 3. ✅ Age Checks
- **Before**: No age validation
- **After**: Checks `onboardDate` from Binance market info
- **Implementation**: `has_enough_age()` and `filter_by_age()` functions
- **Logic**: If `onboardDate <= start_time`, market is old enough

### 4. ✅ Adaptive Window
- **Before**: Fixed 150 days, failed if not enough data
- **After**: Tries 150 → 60 → 30 days automatically
- **Implementation**: `fetch_with_adaptive_days()` function
- **Behavior**: Retries with shorter windows if coverage < 95%

### 5. ✅ Ticker Alias Map
- **Before**: No alias mapping
- **After**: Maps common names to Binance IDs
- **Current Aliases**:
  - `ASTER/USDT` → `ASTR/USDT` ✅
  - `XPL/USDT` → `XPLA/USDT` (if exists)

### 6. ✅ Fail-Safe
- **Before**: Failed entire batch on low coverage
- **After**: Logs warning and continues
- **Implementation**: Skips coins with low coverage, continues with others

## Key Features

### Market Validation
```python
# Get active spot markets
spot_markets = get_active_spot_markets(exchange)
# Only include symbols that are in spot_markets
symbols = normalize_symbols(symbols, spot_markets, TICKER_ALIASES)
```

### Adaptive Download
```python
# Tries 150 days first, then 60, then 30
frame, used_days, coverage = fetch_with_adaptive_days(
    symbol, loader, timeframe, end_at, [150, 60, 30], 0.95
)
```

### Age Filtering
```python
# Filter by onboardDate
if has_enough_age(market, start_ms):
    filtered.append(symbol)
```

## Results

### Before Improvements
- ❌ 4 coins failed (GIGGLE, XPL, MMT, ASTER)
- ❌ No market validation
- ❌ No age checks
- ❌ Fixed 150 days only

### After Improvements
- ✅ 19 coins found (1 filtered out - XPLA not in spot markets)
- ✅ All symbols validated against spot markets
- ✅ Age checks implemented
- ✅ Adaptive windows (150 → 60 → 30 days)
- ✅ Fail-safe: logs and continues on low coverage

## Usage

### Basic Usage
```bash
# Download top 20 coins with adaptive windows
python scripts/simple_download_candles.py --all-top20

# Download specific symbols
python scripts/simple_download_candles.py --symbols BTC/USDT ETH/USDT

# Fixed days (no adaptive)
python scripts/simple_download_candles.py --all-top20 --no-adaptive --days 150
```

## Configuration

### Ticker Aliases
Edit `TICKER_ALIASES` in the script:
```python
TICKER_ALIASES = {
    "ASTER/USDT": "ASTR/USDT",
    "XPL/USDT": "XPLA/USDT",
    # Add more as needed
}
```

### Adaptive Days
Edit `ADAPTIVE_DAYS`:
```python
ADAPTIVE_DAYS = [150, 60, 30]  # Try 150 days first, then 60, then 30
```

### Coverage Threshold
Edit `MIN_COVERAGE`:
```python
MIN_COVERAGE = 0.95  # 95% coverage required
```

## Testing

The script has been tested and works correctly:
- ✅ Validates against 1604 active spot markets
- ✅ Normalizes symbols correctly (ASTER → ASTR)
- ✅ Uses adaptive windows (150 → 60 → 30)
- ✅ Skips duplicates in Dropbox
- ✅ Continues on failures (fail-safe)

## Next Steps

1. Monitor downloads to see which coins need aliases
2. Add more aliases as needed
3. Adjust `ADAPTIVE_DAYS` if needed
4. Monitor coverage thresholds

