# Quick Start: Download 250 Coins, Train Top 3

## 🎯 Goal

1. **Download candle data for top 250 coins** (for future scaling)
2. **Train models only on top 3 coins** (BTC, ETH, SOL) right now
3. **Scale to more coins later** when ready

---

## 🚀 Step 1: Download Top 250 Coins

### Command:
```bash
export DROPBOX_ACCESS_TOKEN="your_token_here"

python scripts/simple_download_candles.py \
  --top 250 \
  --days 1095 \
  --timeframe 1h
```

### What This Does:
- ✅ Fetches top 250 coins by 24h volume from Binance
- ✅ Downloads 3 years (1095 days) of 1-hour candle data
- ✅ Uses adaptive window (tries 150, 60, 30 days if needed)
- ✅ Saves to local `data/candles/` directory
- ✅ Uploads to Dropbox at `/Runpodhuracan/data/candles/`
- ✅ Skips coins that are already cached/uploaded

### Expected Time:
- **Total**: ~30-60 minutes (depending on network and existing cache)
- **Per coin**: ~10-30 seconds (download + upload)
- **Rate limits**: Script handles delays automatically

### Output:
- **Local**: `data/candles/{SYMBOL}/{SYMBOL}-USDT_1h_*.parquet`
- **Dropbox**: `/Runpodhuracan/data/candles/{SYMBOL}/{SYMBOL}-USDT_1h_*.parquet`

---

## 🎯 Step 2: Train on Top 3 Coins Only

### Command:
```bash
export DROPBOX_ACCESS_TOKEN="your_token_here"

python scripts/test_end_to_end_training.py \
  --symbols BTC/USDT ETH/USDT SOL/USDT \
  --days 1095 \
  --timeframe 1h
```

### What This Does:
- ✅ Loads data from cache (from Step 1)
- ✅ Trains XGBoost models on BTC, ETH, SOL
- ✅ Uses 3 years (1095 days) of data
- ✅ Uploads models and metrics to Dropbox
- ✅ Uploads candle data to Dropbox (if not already uploaded)

### Expected Time:
- **Per coin**: ~5-10 minutes (training + upload)
- **Total for 3 coins**: ~15-30 minutes

### Output:
- **Models**: `/Runpodhuracan/huracan/models/baselines/20251111/{SYMBOL}/model.bin`
- **Metrics**: `/Runpodhuracan/huracan/models/baselines/20251111/{SYMBOL}/metrics.json`

---

## 📊 Complete Workflow

### Phase 1: Data Collection (Run Once)
```bash
# Download top 250 coins (3 years of data)
python scripts/simple_download_candles.py \
  --top 250 \
  --days 1095 \
  --timeframe 1h
```

**Result**: All 250 coins' data in Dropbox, ready for training

### Phase 2: Training (Run Now)
```bash
# Train on top 3 coins only
python scripts/test_end_to_end_training.py \
  --symbols BTC/USDT ETH/USDT SOL/USDT \
  --days 1095 \
  --timeframe 1h
```

**Result**: Trained models for BTC, ETH, SOL

### Phase 3: Scaling (Run Later)
```bash
# Train on more coins when ready (data already in Dropbox!)
python scripts/test_end_to_end_training.py \
  --symbols BTC/USDT ETH/USDT SOL/USDT ADA/USDT DOT/USDT MATIC/USDT AVAX/USDT LINK/USDT UNI/USDT ATOM/USDT \
  --days 1095 \
  --timeframe 1h
```

**Result**: Trained models for more coins (no download needed!)

---

## 📁 Dropbox Structure After Completion

```
/Runpodhuracan/
├── data/
│   └── candles/                    ← All 250 coins (from Step 1)
│       ├── BTC/
│       │   └── BTC-USDT_1h_*.parquet
│       ├── ETH/
│       │   └── ETH-USDT_1h_*.parquet
│       ├── SOL/
│       │   └── SOL-USDT_1h_*.parquet
│       ├── ADA/
│       │   └── ADA-USDT_1h_*.parquet
│       └── ... (250 coins total)
└── huracan/
    └── models/
        └── baselines/
            └── 20251111/
                ├── BTCUSDT/        ← Only top 3 trained (from Step 2)
                │   ├── model.bin
                │   └── metrics.json
                ├── ETHUSDT/
                │   ├── model.bin
                │   └── metrics.json
                └── SOLUSDT/
                    ├── model.bin
                    └── metrics.json
```

---

## ⚡ Benefits

### 1. **Data Ready for Scaling**
- ✅ All 250 coins' data in Dropbox
- ✅ Can train on any coin instantly (no download needed)
- ✅ RunPod can restore from Dropbox quickly

### 2. **Efficient Training**
- ✅ Train only on coins you need right now
- ✅ Save compute time and costs
- ✅ Scale gradually as needed

### 3. **Flexible Scaling**
- ✅ Add more coins to training anytime
- ✅ Data already downloaded and cached
- ✅ Just run training script with more symbols

---

## 🎯 Commands Summary

### Download Top 250 Coins:
```bash
export DROPBOX_ACCESS_TOKEN="your_token_here"

python scripts/simple_download_candles.py \
  --top 250 \
  --days 1095 \
  --timeframe 1h
```

### Train Top 3 Coins:
```bash
python scripts/test_end_to_end_training.py \
  --symbols BTC/USDT ETH/USDT SOL/USDT \
  --days 1095 \
  --timeframe 1h
```

### Train More Coins Later:
```bash
# Add more symbols to --symbols argument
python scripts/test_end_to_end_training.py \
  --symbols BTC/USDT ETH/USDT SOL/USDT ADA/USDT DOT/USDT ... \
  --days 1095 \
  --timeframe 1h
```

---

## 📝 Notes

### Rate Limits:
- Binance has rate limits (1200 requests per minute)
- Script uses adaptive delays automatically
- If you hit rate limits, the script will retry with backoff

### Data Size:
- **250 coins × 3 years × 1h candles**: ~500 MB - 2 GB total
- Depends on coin activity and data compression
- Dropbox has plenty of space

### Training Time:
- **Top 3 coins**: ~15-30 minutes total
- **Top 10 coins**: ~50-90 minutes total
- **Top 250 coins**: ~8-24 hours total (when ready)

### Storage:
- **Local**: ~500 MB - 2 GB (candle data)
- **Dropbox**: ~500 MB - 2 GB (candle data)
- **Models**: ~1-5 MB per coin (small)

---

## ✅ Checklist

- [ ] Set `DROPBOX_ACCESS_TOKEN` environment variable
- [ ] Run download script for top 250 coins
- [ ] Verify data in Dropbox (`/Runpodhuracan/data/candles/`)
- [ ] Run training script for top 3 coins
- [ ] Verify models in Dropbox (`/Runpodhuracan/huracan/models/baselines/`)
- [ ] Ready to scale to more coins when needed!

---

## 🔗 Related Files

- Download Script: `scripts/simple_download_candles.py`
- Training Script: `scripts/test_end_to_end_training.py`
- Scaling Plan: `SCALING_PLAN.md`
- What The Bot Did: `WHAT_THE_BOT_DID.md`
