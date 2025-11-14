# Scaling Plan: Download 250 Coins, Train Top 3

## 🎯 Strategy

**Download data for top 250 coins** (for future scaling)  
**Train models only on top 3 coins** (BTC, ETH, SOL) right now  
**Scale to more coins later** when ready

---

## 🚀 Quick Start

### Option 1: Use the Script (Easiest)
```bash
export DROPBOX_ACCESS_TOKEN="your_token_here"
cd "/Users/haq/ENGINE (HF1) Crsor/engine"
./scripts/download_250_train_3.sh
```

### Option 2: Run Commands Manually

**Step 1: Download Top 250 Coins**
```bash
export DROPBOX_ACCESS_TOKEN="your_token_here"

python scripts/simple_download_candles.py \
  --top 250 \
  --days 1095 \
  --timeframe 1h \
  --no-adaptive
```

**Step 2: Train Top 3 Coins**
```bash
python scripts/test_end_to_end_training.py \
  --symbols BTC/USDT ETH/USDT SOL/USDT \
  --days 1095 \
  --timeframe 1h
```

---

## 📊 What Happens

### Step 1: Download Top 250 Coins
1. ✅ Fetches top 250 coins by 24h volume from Binance
2. ✅ Downloads 3 years (1095 days) of 1-hour candle data
3. ✅ Uses cached data if available (skips download)
4. ✅ Uploads to Dropbox at `/Runpodhuracan/data/candles/`
5. ✅ Takes ~30-60 minutes (depending on cache and network)

### Step 2: Train Top 3 Coins
1. ✅ Loads data from cache (from Step 1)
2. ✅ Trains XGBoost models on BTC, ETH, SOL
3. ✅ Uses 3 years (1095 days) of data
4. ✅ Uploads models and metrics to Dropbox
5. ✅ Uploads candle data to Dropbox (if not already uploaded)
6. ✅ Takes ~15-30 minutes for 3 coins

---

## 📁 Dropbox Structure

### After Step 1 (Download):
```
/Runpodhuracan/data/candles/
├── BTC/
│   └── BTC-USDT_1h_*.parquet
├── ETH/
│   └── ETH-USDT_1h_*.parquet
├── SOL/
│   └── SOL-USDT_1h_*.parquet
└── ... (250 coins total)
```

### After Step 2 (Training):
```
/Runpodhuracan/huracan/models/baselines/20251111/
├── BTCUSDT/
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

## 🎯 Scaling Later

When ready to train on more coins:

```bash
# Train on top 10 coins
python scripts/test_end_to_end_training.py \
  --symbols BTC/USDT ETH/USDT SOL/USDT ADA/USDT DOT/USDT MATIC/USDT AVAX/USDT LINK/USDT UNI/USDT ATOM/USDT \
  --days 1095 \
  --timeframe 1h
```

**Data is already in Dropbox, so training is fast!** 🚀

---

## 📝 Notes

### Training Data:
- **3 years** = 1095 days
- **1-hour candles** = ~26,280 candles per coin
- **Training samples** = ~21,000 samples per coin (after feature building)
- **Better models** = More data = Better performance

### Data Size:
- **250 coins × 3 years × 1h candles**: ~500 MB - 2 GB total
- **Models**: ~1-5 MB per coin (small)

### Time Estimates:
- **Download 250 coins**: ~30-60 minutes
- **Train 3 coins**: ~15-30 minutes
- **Train 10 coins**: ~50-90 minutes
- **Train 250 coins**: ~8-24 hours (when ready)

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

- Quick Start: `QUICK_START.md`
- Scaling Plan: `SCALING_PLAN.md`
- What The Bot Did: `WHAT_THE_BOT_DID.md`
- Run Now: `RUN_NOW.md`

