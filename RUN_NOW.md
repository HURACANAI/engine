# Run Now: Download 250 Coins, Train Top 3

## 🎯 Your Request

1. **Download top 250 coins** (for future scaling)
2. **Train only on top 3** (BTC, ETH, SOL) right now
3. **Scale to more coins later** when ready

---

## ✅ Ready to Run Commands

### Step 1: Download Top 250 Coins (3 Years of Data)

```bash
export DROPBOX_ACCESS_TOKEN="sl.u.AGFGiCwLVyXLDRrc0RKgYtd7uZu_IeqNtSVpJqG6NbY0yz6TkUIqns1hsXi8PtW8_tBNRYMgPRvju_zii5BddOlZdZS_9F-c81TGvG5wu1LkspNaNHnGEWB1Djxv-Oe4o4shFAC-JutB5utpaSoVkFaJlX6iaKMJTif3DJajVzESj7NumwYIMmLqn_4y1LQw3pah-6DdqgtDNSfUAfh5WqWiEC8DOzTTP2-zFvBYnbaoOH0f6ZjLXg_-BolLqeOXnDI-3Ee5aj05jyBXrBwmYITZpECMqFIPlTXo_tc8j4LdMNnHgaojzXvb6_NWy6Lmsbeh1tSFWRpwI-JtJSTQgfDhh59S7jWsqWU-mj3F8Z_uXBrYDtbynAec-H2c1mF0hm09Zgk3d6N4l2WjWLiNskbTmKHGK7Qeyqoq5dkS00zdrgpJfCvo8-RS0zPQ44b2ck9tADF_N7Q9uwmsdh_h2fc4axWQbO_w4ZQBeYH_9HHz5mB0M6u1W_Z1PCW4Tu5pj4l_JA3WLAIKOGVs1LD8kPPAEcSA63gDc_KMZItd5NdP2ePzUMJIdp-CeRGBOqRhjLCkJufWa7doy1fsWexbBitt_A7I96GW-1T8FeZ2ltZCLCATffRDAYh2LHp0Iigi5LLIVVr5MCIlioHYR7hu0xLdd-G9KEfA-PUyN3K2e4Vy7BPLe0Z1a7g1XYfy9yYLDX5hsPQsSGCuljFfhvhNud-F_zFrkmJb94vE3vpmK16afZe06ZQelZm84_aXkK4pzgmJzVLUX2kWWfkmApXlZC39jKZRXFbdZ3VIUrGo1ZWXS8kqNKxguwNX-RQ7d3H2fw5cZZuWiKUB443IdVmVF_eaDbPQFZqa2YaICQk6oMonhMvDOR8iOc6AFgMwPaMRk93gyXeipbPrYYI5DSdafoWWsI4u143m1-apCJpVbDJRvBUznOzlb66SKBsTq471b-ImsX9vlmCGJ6ugdaPrXaHAzB1UsnwPjWovCi6Wc_nSSTVsXvGYveiCAwpyKCVuK4Ybuqmfon6Dt_zRGGWKISi4OyrpEbo3tGld0FJB9BXEVixprM8blrBN-hHN26ZyS1q_ad8hmPmiwqvlZ3InoTj7SFcS2gf_JQJfqIa7A-_djO49D_4oGdvESCciIAqjb1CtEbzn1uXyNIOlC4xAh5xf6ScISIANh_eR7UhqXsA5eXDSd5LRbL7O-89y-arY_ymxUOvCyXN2SB9Q2vZ6FXv76ud9y8Uljw0IkeYnrgikMSJnBgMz2ZDYDovTLtto4Zij736hratutNcWXKcSoL5k"

cd "/Users/haq/ENGINE (HF1) Crsor/engine"

# Download top 250 coins (uses cached data if available, downloads if needed)
python scripts/simple_download_candles.py \
  --top 250 \
  --days 1095 \
  --timeframe 1h \
  --no-adaptive
```

**What this does:**
- Fetches top 250 coins by 24h volume
- Downloads 3 years (1095 days) of 1-hour candle data
- Uses cached data if available (skips download)
- Uploads to Dropbox at `/Runpodhuracan/data/candles/`
- Takes ~30-60 minutes (depending on cache)

---

### Step 2: Train on Top 3 Coins Only

```bash
export DROPBOX_ACCESS_TOKEN="sl.u.AGFGiCwLVyXLDRrc0RKgYtd7uZu_IeqNtSVpJqG6NbY0yz6TkUIqns1hsXi8PtW8_tBNRYMgPRvju_zii5BddOlZdZS_9F-c81TGvG5wu1LkspNaNHnGEWB1Djxv-Oe4o4shFAC-JutB5utpaSoVkFaJlX6iaKMJTif3DJajVzESj7NumwYIMmLqn_4y1LQw3pah-6DdqgtDNSfUAfh5WqWiEC8DOzTTP2-zFvBYnbaoOH0f6ZjLXg_-BolLqeOXnDI-3Ee5aj05jyBXrBwmYITZpECMqFIPlTXo_tc8j4LdMNnHgaojzXvb6_NWy6Lmsbeh1tSFWRpwI-JtJSTQgfDhh59S7jWsqWU-mj3F8Z_uXBrYDtbynAec-H2c1mF0hm09Zgk3d6N4l2WjWLiNskbTmKHGK7Qeyqoq5dkS00zdrgpJfCvo8-RS0zPQ44b2ck9tADF_N7Q9uwmsdh_h2fc4axWQbO_w4ZQBeYH_9HHz5mB0M6u1W_Z1PCW4Tu5pj4l_JA3WLAIKOGVs1LD8kPPAEcSA63gDc_KMZItd5NdP2ePzUMJIdp-CeRGBOqRhjLCkJufWa7doy1fsWexbBitt_A7I96GW-1T8FeZ2ltZCLCATffRDAYh2LHp0Iigi5LLIVVr5MCIlioHYR7hu0xLdd-G9KEfA-PUyN3K2e4Vy7BPLe0Z1a7g1XYfy9yYLDX5hsPQsSGCuljFfhvhNud-F_zFrkmJb94vE3vpmK16afZe06ZQelZm84_aXkK4pzgmJzVLUX2kWWfkmApXlZC39jKZRXFbdZ3VIUrGo1ZWXS8kqNKxguwNX-RQ7d3H2fw5cZZuWiKUB443IdVmVF_eaDbPQFZqa2YaICQk6oMonhMvDOR8iOc6AFgMwPaMRk93gyXeipbPrYYI5DSdafoWWsI4u143m1-apCJpVbDJRvBUznOzlb66SKBsTq471b-ImsX9vlmCGJ6ugdaPrXaHAzB1UsnwPjWovCi6Wc_nSSTVsXvGYveiCAwpyKCVuK4Ybuqmfon6Dt_zRGGWKISi4OyrpEbo3tGld0FJB9BXEVixprM8blrBN-hHN26ZyS1q_ad8hmPmiwqvlZ3InoTj7SFcS2gf_JQJfqIa7A-_djO49D_4oGdvESCciIAqjb1CtEbzn1uXyNIOlC4xAh5xf6ScISIANh_eR7UhqXsA5eXDSd5LRbL7O-89y-arY_ymxUOvCyXN2SB9Q2vZ6FXv76ud9y8Uljw0IkeYnrgikMSJnBgMz2ZDYDovTLtto4Zij736hratutNcWXKcSoL5k"

cd "/Users/haq/ENGINE (HF1) Crsor/engine"

# Train only on top 3 coins (BTC, ETH, SOL)
python scripts/test_end_to_end_training.py \
  --symbols BTC/USDT ETH/USDT SOL/USDT \
  --days 1095 \
  --timeframe 1h
```

**What this does:**
- Trains models on BTC, ETH, SOL only
- Uses 3 years (1095 days) of data
- Loads from cache (from Step 1)
- Uploads models and metrics to Dropbox
- Uploads candle data to Dropbox (if not already uploaded)
- Takes ~15-30 minutes for 3 coins

---

## 📊 What You'll Get

### Dropbox Structure:
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

## ⚡ Quick Summary

### Step 1: Download 250 Coins
- **Command**: `python scripts/simple_download_candles.py --top 250 --days 1095 --timeframe 1h --no-adaptive`
- **Time**: ~30-60 minutes
- **Result**: All 250 coins' data in Dropbox

### Step 2: Train Top 3 Coins
- **Command**: `python scripts/test_end_to_end_training.py --symbols BTC/USDT ETH/USDT SOL/USDT --days 1095 --timeframe 1h`
- **Time**: ~15-30 minutes
- **Result**: Trained models for BTC, ETH, SOL

---

## 🎯 What Happens

### Step 1: Download Script
1. Fetches top 250 coins by 24h volume from Binance
2. For each coin:
   - Checks if data exists in cache
   - If cached: Uses cached data, uploads to Dropbox
   - If not cached: Downloads from exchange, saves locally, uploads to Dropbox
3. Skips coins that are already in Dropbox (saves time)

### Step 2: Training Script
1. For each of the 3 coins (BTC, ETH, SOL):
   - Loads data from cache (from Step 1)
   - Builds features (15 features)
   - Trains XGBoost model (3 years of data)
   - Evaluates model performance
   - Saves artifacts (model.bin, metrics.json, config.json, sha256.txt)
   - Uploads to Dropbox (model.bin, metrics.json, candle data)

---

## 📝 Notes

### Data Already Cached?
- If you already have cached data (from previous runs), the script will use it
- It will upload cached data to Dropbox if not already there
- No need to re-download if data is already cached

### Training Data
- **3 years** = 1095 days
- **1-hour candles** = ~26,280 candles per coin
- **Training samples** = ~21,000 samples per coin (after feature building)
- **Better models** = More data = Better performance

### Scaling Later
- All 250 coins' data is in Dropbox
- Just add more symbols to `--symbols` argument
- Data is already downloaded, so training is fast

---

## 🚀 Ready to Run?

### Copy and paste these commands:

```bash
# Set token
export DROPBOX_ACCESS_TOKEN="sl.u.AGFGiCwLVyXLDRrc0RKgYtd7uZu_IeqNtSVpJqG6NbY0yz6TkUIqns1hsXi8PtW8_tBNRYMgPRvju_zii5BddOlZdZS_9F-c81TGvG5wu1LkspNaNHnGEWB1Djxv-Oe4o4shFAC-JutB5utpaSoVkFaJlX6iaKMJTif3DJajVzESj7NumwYIMmLqn_4y1LQw3pah-6DdqgtDNSfUAfh5WqWiEC8DOzTTP2-zFvBYnbaoOH0f6ZjLXg_-BolLqeOXnDI-3Ee5aj05jyBXrBwmYITZpECMqFIPlTXo_tc8j4LdMNnHgaojzXvb6_NWy6Lmsbeh1tSFWRpwI-JtJSTQgfDhh59S7jWsqWU-mj3F8Z_uXBrYDtbynAec-H2c1mF0hm09Zgk3d6N4l2WjWLiNskbTmKHGK7Qeyqoq5dkS00zdrgpJfCvo8-RS0zPQ44b2ck9tADF_N7Q9uwmsdh_h2fc4axWQbO_w4ZQBeYH_9HHz5mB0M6u1W_Z1PCW4Tu5pj4l_JA3WLAIKOGVs1LD8kPPAEcSA63gDc_KMZItd5NdP2ePzUMJIdp-CeRGBOqRhjLCkJufWa7doy1fsWexbBitt_A7I96GW-1T8FeZ2ltZCLCATffRDAYh2LHp0Iigi5LLIVVr5MCIlioHYR7hu0xLdd-G9KEfA-PUyN3K2e4Vy7BPLe0Z1a7g1XYfy9yYLDX5hsPQsSGCuljFfhvhNud-F_zFrkmJb94vE3vpmK16afZe06ZQelZm84_aXkK4pzgmJzVLUX2kWWfkmApXlZC39jKZRXFbdZ3VIUrGo1ZWXS8kqNKxguwNX-RQ7d3H2fw5cZZuWiKUB443IdVmVF_eaDbPQFZqa2YaICQk6oMonhMvDOR8iOc6AFgMwPaMRk93gyXeipbPrYYI5DSdafoWWsI4u143m1-apCJpVbDJRvBUznOzlb66SKBsTq471b-ImsX9vlmCGJ6ugdaPrXaHAzB1UsnwPjWovCi6Wc_nSSTVsXvGYveiCAwpyKCVuK4Ybuqmfon6Dt_zRGGWKISi4OyrpEbo3tGld0FJB9BXEVixprM8blrBN-hHN26ZyS1q_ad8hmPmiwqvlZ3InoTj7SFcS2gf_JQJfqIa7A-_djO49D_4oGdvESCciIAqjb1CtEbzn1uXyNIOlC4xAh5xf6ScISIANh_eR7UhqXsA5eXDSd5LRbL7O-89y-arY_ymxUOvCyXN2SB9Q2vZ6FXv76ud9y8Uljw0IkeYnrgikMSJnBgMz2ZDYDovTLtto4Zij736hratutNcWXKcSoL5k"

cd "/Users/haq/ENGINE (HF1) Crsor/engine"

# Step 1: Download top 250 coins
python scripts/simple_download_candles.py \
  --top 250 \
  --days 1095 \
  --timeframe 1h \
  --no-adaptive

# Step 2: Train top 3 coins
python scripts/test_end_to_end_training.py \
  --symbols BTC/USDT ETH/USDT SOL/USDT \
  --days 1095 \
  --timeframe 1h
```

---

## ✅ Expected Results

### After Step 1:
- ✅ 250 coin data files in Dropbox
- ✅ Local cache updated
- ✅ Ready for training

### After Step 2:
- ✅ 3 trained models in Dropbox
- ✅ 3 metrics files in Dropbox
- ✅ Candle data uploaded (if not already)
- ✅ Ready for production use

---

## 🎯 Next Steps (Later)

When ready to scale:
```bash
# Train on top 10 coins
python scripts/test_end_to_end_training.py \
  --symbols BTC/USDT ETH/USDT SOL/USDT ADA/USDT DOT/USDT MATIC/USDT AVAX/USDT LINK/USDT UNI/USDT ATOM/USDT \
  --days 1095 \
  --timeframe 1h
```

Data is already in Dropbox, so training is fast! 🚀

