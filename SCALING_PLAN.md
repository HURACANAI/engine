# Scaling Plan: Download 250 Coins, Train Top 3

## 🎯 Strategy

1. **Download data for top 250 coins** (for future scaling)
2. **Train models only on top 3 coins** (BTC, ETH, SOL) right now
3. **Scale to more coins later** when ready

---

## 📥 Step 1: Download Top 250 Coins

### Command:
```bash
export DROPBOX_ACCESS_TOKEN="your_token_here"

python scripts/download_top_coins.py \
  --top 250 \
  --days 1095 \
  --timeframe 1h \
  --delay 0.5
```

### What This Does:
- Fetches top 250 coins by 24h volume from Binance
- Downloads 3 years (1095 days) of 1-hour candle data
- Saves to local `data/candles/` directory
- Uploads to Dropbox at `/Runpodhuracan/data/candles/`
- Uses 0.5 second delay between downloads to avoid rate limits

### Expected Time:
- ~250 coins × 0.5 seconds = ~2 minutes for delays
- Plus download time: ~10-30 minutes total (depending on network)
- Plus upload time: ~5-15 minutes total

### Output:
- Local: `data/candles/{SYMBOL}/{SYMBOL}-USDT_1h_*.parquet`
- Dropbox: `/Runpodhuracan/data/candles/{SYMBOL}/{SYMBOL}-USDT_1h_*.parquet`

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
- Trains models on BTC, ETH, SOL only
- Uses 3 years (1095 days) of data
- Uploads models and metrics to Dropbox
- Does NOT re-download data (uses cached data from Step 1)

### Output:
- Models: `/Runpodhuracan/huracan/models/baselines/20251111/{SYMBOL}/model.bin`
- Metrics: `/Runpodhuracan/huracan/models/baselines/20251111/{SYMBOL}/metrics.json`

---

## 📊 Workflow Summary

### Phase 1: Data Collection (Now)
```bash
# Download top 250 coins
python scripts/download_top_coins.py --top 250 --days 1095 --timeframe 1h
```
**Result**: All 250 coins' data in Dropbox, ready for training

### Phase 2: Training (Now)
```bash
# Train on top 3 coins only
python scripts/test_end_to_end_training.py \
  --symbols BTC/USDT ETH/USDT SOL/USDT \
  --days 1095 \
  --timeframe 1h
```
**Result**: Trained models for BTC, ETH, SOL

### Phase 3: Scaling (Later)
```bash
# Train on more coins when ready
python scripts/test_end_to_end_training.py \
  --symbols BTC/USDT ETH/USDT SOL/USDT ADA/USDT DOT/USDT ... \
  --days 1095 \
  --timeframe 1h
```
**Result**: Trained models for more coins (data already in Dropbox!)

---

## 🔍 What Gets Stored Where

### Local Storage:
```
data/candles/
├── BTC/
│   └── BTC-USDT_1h_*.parquet
├── ETH/
│   └── ETH-USDT_1h_*.parquet
├── SOL/
│   └── SOL-USDT_1h_*.parquet
├── ADA/
│   └── ADA-USDT_1h_*.parquet
└── ... (250 coins total)
```

### Dropbox Storage:
```
/Runpodhuracan/
├── data/
│   └── candles/                    ← All 250 coins
│       ├── BTC/
│       ├── ETH/
│       ├── SOL/
│       ├── ADA/
│       └── ... (250 coins)
└── huracan/
    └── models/
        └── baselines/
            └── 20251111/
                ├── BTCUSDT/        ← Only top 3 trained
                ├── ETHUSDT/
                └── SOLUSDT/
```

---

## ⚡ Benefits of This Approach

### 1. **Data Ready for Scaling**
- All 250 coins' data in Dropbox
- Can train on any coin instantly (no download needed)
- RunPod can restore from Dropbox quickly

### 2. **Efficient Training**
- Train only on coins you need right now
- Save compute time and costs
- Scale gradually as needed

### 3. **Flexible Scaling**
- Add more coins to training anytime
- Data already downloaded and cached
- Just run training script with more symbols

### 4. **Cost Effective**
- Download data once (cheap)
- Train models as needed (more expensive)
- Pay for training only when ready

---

## 🚀 Quick Start

### 1. Download Top 250 Coins (One Time):
```bash
export DROPBOX_ACCESS_TOKEN="your_token_here"

python scripts/download_top_coins.py \
  --top 250 \
  --days 1095 \
  --timeframe 1h
```

### 2. Train Top 3 Coins (Now):
```bash
python scripts/test_end_to_end_training.py \
  --symbols BTC/USDT ETH/USDT SOL/USDT \
  --days 1095 \
  --timeframe 1h
```

### 3. Scale Later (When Ready):
```bash
# Train on top 10 coins
python scripts/test_end_to_end_training.py \
  --symbols BTC/USDT ETH/USDT SOL/USDT ADA/USDT DOT/USDT MATIC/USDT AVAX/USDT LINK/USDT UNI/USDT ATOM/USDT \
  --days 1095 \
  --timeframe 1h

# Or train on all 250 coins (when ready)
# (modify script to accept list of all symbols)
```

---

## 📝 Notes

### Rate Limits:
- Binance has rate limits (1200 requests per minute)
- Script uses 0.5 second delay between downloads
- If you hit rate limits, increase `--delay` to 1.0 or 2.0 seconds

### Data Size:
- 250 coins × 3 years × 1h candles ≈ 250 MB - 2 GB total
- Depends on coin activity and data compression
- Dropbox has plenty of space

### Training Time:
- Top 3 coins: ~10-30 minutes total
- Top 10 coins: ~30-90 minutes total
- Top 250 coins: ~8-24 hours total (when ready)

### Storage:
- Local: ~250 MB - 2 GB
- Dropbox: ~250 MB - 2 GB (same data)
- Models: ~1-5 MB per coin (small)

---

## 🎯 Next Steps

1. ✅ **Download top 250 coins** (run once)
2. ✅ **Train top 3 coins** (run now)
3. ⏳ **Monitor performance** (watch metrics)
4. ⏳ **Scale to more coins** (when ready)
5. ⏳ **Scale to all 250 coins** (when ready)

---

## 🔗 Related Files

- Download Script: `scripts/download_top_coins.py`
- Training Script: `scripts/test_end_to_end_training.py`
- Data Loader: `src/cloud/training/datasets/data_loader.py`
- Dropbox Sync: `src/cloud/training/integrations/dropbox_sync.py`

