# Training Test Results - Top 20 Coins

## Summary

**Test Run Date**: November 11, 2025
**Coins Trained**: Top 20 coins by 24h volume from Binance
**Training Period**: 180 days (as configured in `config.yaml`)
**Status**: ✅ **17 out of 20 coins trained successfully**

## Training Statistics

- **Successful**: 17/20 coins (85%)
- **Failed**: 3/20 coins (15%)
- **Average days per coin**: 164.1 days
- **Total days across all coins**: 2,790 days
- **Data Source**: Cached candle data (1d timeframe)
- **Models Uploaded**: All successful models uploaded to Dropbox

## Data Usage

### Training Days
- **Configured lookback**: 180 days (from `config.yaml`)
- **Most coins**: Trained on **180 days** of data (May 15, 2025 to November 11, 2025)
- **Some coins**: Had less data available (e.g., ASTER: 29 days, PUMP: 61 days)
- **Total**: 2,790 days of training data across all coins

### Data Source
- All coins used **cached candle data** from `data/candles/`
- No downloads required (all data already cached from previous 250-coin download)
- Data format: Daily candles (1d timeframe)
- Date range: Most coins had 1,095 days cached, training used most recent 180 days

## Coins Trained

### Successful (17 coins)

1. **BTC/USDT**: ✅ 180 days, Sharpe=-0.1237, Hit Rate=53.12%
2. **ETH/USDT**: ✅ 180 days, Sharpe=-0.0638, Hit Rate=43.75%
3. **ZEC/USDT**: ✅ 180 days, Sharpe=0.2943, Hit Rate=65.62%
4. **UNI/USDT**: ✅ 180 days, Sharpe=0.1763, Hit Rate=53.12%
5. **SOL/USDT**: ✅ 180 days, Sharpe=-0.0957, Hit Rate=56.25%
6. **XRP/USDT**: ✅ 180 days, Sharpe=-0.0598, Hit Rate=56.25%
7. **BNB/USDT**: ✅ 180 days, Sharpe=-0.0797, Hit Rate=43.75%
8. **DOGE/USDT**: ✅ 180 days, Sharpe=-0.0498, Hit Rate=40.62%
9. **ASTER/USDT**: ✅ 29 days, Sharpe=-0.0049, Hit Rate=47.94%
10. **TRUMP/USDT**: ✅ 180 days, Sharpe=0.2251, Hit Rate=50.00%
11. **LTC/USDT**: ✅ 180 days, Sharpe=0.0443, Hit Rate=37.50%
12. **FIL/USDT**: ✅ 180 days, Sharpe=0.1213, Hit Rate=59.38%
13. **ICP/USDT**: ✅ 180 days, Sharpe=0.2632, Hit Rate=56.25%
14. **TRX/USDT**: ✅ 180 days, Sharpe=-0.1004, Hit Rate=62.50%
15. **SUI/USDT**: ✅ 180 days, Sharpe=-0.1234, Hit Rate=40.62%
16. **PUMP/USDT**: ✅ 61 days, Sharpe=0.0593, Hit Rate=55.56%
17. **NEAR/USDT**: ✅ 180 days, Sharpe=0.0781, Hit Rate=68.75%

### Failed (3 coins)

1. **GIGGLE/USDT**: ❌ Feature building failed (17 days of data)
2. **ALLO/USDT**: ❌ Feature building failed (0 days of data)
3. **MMT/USDT**: ❌ Feature building failed (7 days of data)

## Key Findings

### Data Availability
- **All 20 coins** had cached data available
- **Most coins** had full 1,095 days cached (3 years)
- **Training used** the most recent 180 days (as configured)
- **Some newer coins** had less data (ASTER: 29 days, PUMP: 61 days)

### Training Results
- **17 coins** successfully trained and uploaded to Dropbox
- **3 coins** failed due to insufficient data for feature building
- **Average training samples**: ~161 samples per coin (after feature engineering)
- **Models uploaded**: All successful models uploaded to `/Runpodhuracan/huracan/models/baselines/20251111/`

### Performance Metrics
- **Best Sharpe**: ZEC/USDT (0.2943), ICP/USDT (0.2632), TRUMP/USDT (0.2251)
- **Best Hit Rate**: NEAR/USDT (68.75%), ZEC/USDT (65.62%), TRX/USDT (62.50%)
- **Average Hit Rate**: ~52% across successful coins
- **Average Sharpe**: ~0.02 (slightly positive, but low)

## Answer to Your Question

**"How many days does it train them on?"**

✅ **The engine trains on 180 days per coin** (as configured in `config.yaml` under `engine.lookback_days`)

### Details:
- **Configured**: 180 days lookback
- **Actual usage**: Most coins trained on exactly **180 days** (May 15, 2025 to November 11, 2025)
- **Data available**: Most coins have 1,095 days cached, but only the most recent 180 days are used for training
- **Exceptions**: Some newer coins had less data available (ASTER: 29 days, PUMP: 61 days)

## Output Files

### Dropbox Location
- **Models**: `/Runpodhuracan/huracan/models/baselines/20251111/{SYMBOL}/model.bin`
- **Metrics**: `/Runpodhuracan/huracan/models/baselines/20251111/{SYMBOL}/metrics.json`

### Local Cache
- **Candle Data**: `data/candles/{SYMBOL}/{SYMBOL}-USDT_1d_*.parquet`
- **Models**: `models/{SYMBOL}/{TIMESTAMP}/model.bin`

## Next Steps

1. ✅ **Training Complete**: 17/20 coins trained successfully
2. ✅ **Models Uploaded**: All models uploaded to Dropbox
3. ✅ **Metrics Available**: All metrics saved and uploaded
4. ⚠️  **3 coins failed**: GIGGLE, ALLO, MMT - need more data or different timeframe

## Recommendations

1. **For failed coins**: Consider using a shorter lookback period or different timeframe (e.g., 1h instead of 1d)
2. **For better performance**: Consider tuning hyperparameters or using more sophisticated models
3. **For scaling**: The system is ready to scale to more coins (250+ coins already downloaded)

## Conclusion

✅ **Training test successful!** The engine successfully trained on **17 out of 20 top coins** using **180 days of data per coin**. All models and metrics have been uploaded to Dropbox and are ready for use.

