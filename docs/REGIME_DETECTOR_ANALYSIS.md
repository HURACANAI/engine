# Regime Detector Analysis

## Current Situation

Your training shows:
- **66% PANIC** (1,718 trades)
- **25% UNKNOWN** (635 trades)
- **9% TREND** (239 trades)
- **0% RANGE** (0 trades)

## Why This Happens

### 1. Panic Detection is Too Sensitive

The panic detector triggers when:
```python
if features.atr_pct > 3.0:  # > 3% ATR
```

**Problem**: For crypto (especially SOL), **3% ATR is actually normal**!

Your data shows:
- Panic regime avg volatility: 78.69 bps
- Trend regime avg volatility: 86.86 bps
- Unknown regime avg volatility: 70.39 bps

**The volatility is similar across all regimes!** This means the panic detector is mis-classifying normal crypto volatility as panic.

### 2. Panic Takes Priority

Look at [regime_detector.py:184-196](../src/cloud/training/models/regime_detector.py#L184-L196):

```python
# Panic takes priority!
if blended_scores["panic"] >= 0.7:
    regime = MarketRegime.PANIC
elif blended_scores["trend"] >= 0.6:
    regime = MarketRegime.TREND
elif blended_scores["range"] >= 0.6:
    regime = MarketRegime.RANGE
```

**Once panic score hits 0.7, it wins** - even if trend or range scores are higher!

### 3. Thresholds Too High

- Panic threshold: **0.7** (too low - triggers too easily)
- Trend threshold: **0.6** (reasonable)
- Range threshold: **0.6** (too high for crypto - needs tight consolidation)

For volatile crypto markets, these need adjustment.

### 4. Range Detection Rarely Triggers

Range detection needs (line 345-361):
- Low ADX (< 25)
- **High compression** (tight Bollinger Bands)
- Low volatility ratio
- Low trend strength

**Problem**: Crypto rarely has tight compression! SOL can be ranging but still have 2-4% daily moves.

## Impact on Your 23 Engines

Each of your 23 engines is likely experiencing:
1. **Over-classification as panic** - treating normal volatility as crisis
2. **Missing range opportunities** - not learning sideways strategies
3. **Limited trend detection** - panic priority blocks it
4. **High "unknown" rate** - mixed signals when thresholds don't match

## Recommended Fixes

### Quick Fix (Adjust Thresholds)

Create a crypto-optimized regime detector:

```python
detector = RegimeDetector(
    trend_threshold=0.55,      # Slightly easier to trigger
    range_threshold=0.50,      # Much easier for crypto
    panic_threshold=0.80,      # Much harder - only real panic
)
```

### Medium Fix (Adjust ATR Threshold)

In `regime_detector.py` line 376, change:
```python
if features.atr_pct > 3.0:  # > 3% ATR
```

To:
```python
if features.atr_pct > 5.0:  # > 5% ATR for crypto
```

### Better Fix (Remove Panic Priority)

Change the classification logic to use **highest score wins**:

```python
# Get highest score
max_regime = max(blended_scores.items(), key=lambda x: x[1])

if max_regime[1] >= 0.6:  # Minimum confidence
    if max_regime[0] == "panic":
        regime = MarketRegime.PANIC
    elif max_regime[0] == "trend":
        regime = MarketRegime.TREND
    elif max_regime[0] == "range":
        regime = MarketRegime.RANGE
else:
    regime = MarketRegime.UNKNOWN
```

### Best Fix (Crypto-Specific Calibration)

Create a separate `CryptoRegimeDetector` with:
1. **Higher panic thresholds** (5-7% ATR for panic)
2. **Lower range thresholds** (allow 2-3% daily range)
3. **Equal priority scoring** (highest wins, no panic priority)
4. **Volatility percentile adjustment** (compare to crypto baseline)

## What You Should See

With proper calibration for crypto markets:
- **~35% TREND** - Clear directional moves
- **~30% RANGE** - Consolidation/sideways
- **~20% PANIC** - Actual high volatility events
- **~15% UNKNOWN** - Mixed/transitional periods

## Testing the Hypothesis

Let's check if the problem is real:

```sql
-- Check ATR distribution for "panic" trades
SELECT
    market_regime,
    PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY volatility_bps) as p25,
    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY volatility_bps) as median,
    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY volatility_bps) as p75,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY volatility_bps) as p95
FROM trade_memory
WHERE symbol = 'SOL/USDT'
GROUP BY market_regime;
```

If p50 (median) for panic is similar to other regimes, **we've confirmed the detector is too sensitive**.

## Immediate Action

To improve your 23 engines' learning:

1. **Retrain with adjusted thresholds**:
   ```python
   self.regime_detector = RegimeDetector(
       trend_threshold=0.55,
       range_threshold=0.50,
       panic_threshold=0.80,
   )
   ```

2. **Or modify panic detection** to be crypto-aware:
   ```python
   # In _calculate_panic_score, change ATR threshold
   if features.atr_pct > 5.0:  # Crypto needs higher threshold
   ```

3. **Monitor new distribution** - you should see more balanced regimes

## Why This Matters

Your models learn different strategies for each regime:
- **Trend**: Ride momentum, use momentum indicators
- **Range**: Mean reversion, buy support/sell resistance
- **Panic**: Reduce position, tighter stops, wait for clarity

If **66% of trades are classified as panic**, your models are:
- ❌ **Not learning trend strategies** (only 9% exposure)
- ❌ **Not learning range strategies** (0% exposure!)
- ❌ **Over-applying panic rules** to normal conditions
- ❌ **Missing opportunities** in trending/ranging markets

## Summary

**You have 23 engines but only 2-3 effective regimes** because:
1. Panic detector is too sensitive for crypto (3% ATR is normal)
2. Panic takes priority over other regimes
3. Range detection needs unrealistically tight conditions
4. This creates 66% panic / 25% unknown / 9% trend / 0% range

**Fix**: Adjust thresholds to crypto reality:
- Panic: 0.80 threshold, 5%+ ATR
- Trend: 0.55 threshold
- Range: 0.50 threshold
- Remove panic priority, use highest score

This will give you **proper regime diversity** and better model learning across all 23 engines.
