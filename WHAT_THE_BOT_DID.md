# What Did The Bot Do? - Explained

## 🤔 Your Questions Answered

### 1. **Did it train on the past 3 years of the coin?**
**Answer: NO** - It only trained on **30 days** of data (1-hour candles)

### 2. **Is it supposed to download candle data to Dropbox too?**
**Answer: YES** - But the test script didn't do this. It should upload candle data to Dropbox.

---

## 📊 What Actually Happened

### Training Data Used
- **Days**: 30 days (default in test script)
- **Timeframe**: 1 hour candles
- **Data Source**: Loaded from local cache (existing parquet files)
- **Total Rows**: ~216,000 rows per coin (30 days × 24 hours × ~300 samples per hour)

### What Got Uploaded to Dropbox
✅ **Model files** (model.bin) - 3 files  
✅ **Metrics files** (metrics.json) - 3 files  
❌ **Candle data** - NOT uploaded (should be uploaded)

---

## 🔍 Detailed Breakdown

### Step-by-Step What Happened:

1. **Data Loading**:
   - Script looked for cached data in `data/candles/`
   - Found existing parquet files (from previous downloads)
   - Loaded ~216,000 rows of 1-hour candle data
   - **Did NOT download fresh data from exchange**
   - **Did NOT upload data to Dropbox**

2. **Feature Building**:
   - Built 15 features from candle data:
     - Returns (1h, 5h, 20h)
     - Price ratios (high/low, close/open)
     - Moving averages (SMA 20, SMA 50)
     - Volume ratios
     - RSI, volatility
     - Time features (hour, day, weekend)

3. **Model Training**:
   - Split data: 80% train (~172K samples), 20% test (~43K samples)
   - Trained XGBoost model on training set
   - Evaluated on test set
   - Generated metrics (Sharpe, hit rate, R²)

4. **Artifact Creation**:
   - Saved model.bin (trained model)
   - Saved metrics.json (performance stats)
   - Saved config.json (model configuration)
   - Saved sha256.txt (file hash)

5. **Dropbox Upload**:
   - ✅ Uploaded model.bin
   - ✅ Uploaded metrics.json
   - ❌ Did NOT upload candle data
   - ❌ Did NOT upload config.json
   - ❌ Did NOT upload sha256.txt

---

## ❌ What's Missing (What Should Happen)

### 1. **More Training Data**
- **Current**: 30 days
- **Recommended**: 150-365 days (or more for better models)
- **For 3 years**: Use `--days 1095` (3 years × 365 days)

### 2. **Candle Data Upload to Dropbox**
- **Current**: Candle data only stored locally
- **Should**: Upload to Dropbox at `/Runpodhuracan/data/candles/`
- **Purpose**: RunPod engine can restore from Dropbox instead of downloading from exchange

### 3. **Fresh Data Download**
- **Current**: Used cached data (may be old)
- **Should**: Download fresh data from exchange
- **Then**: Upload to Dropbox for future use

---

## ✅ Proper Workflow (What Should Happen)

### Step 1: Download Candle Data
```bash
# Download 3 years of data (1095 days)
python scripts/simple_download_candles.py \
  --symbols BTC/USDT ETH/USDT SOL/USDT \
  --days 1095 \
  --timeframe 1h
```

This will:
- Download fresh data from exchange
- Save to local `data/candles/`
- **Upload to Dropbox** at `/Runpodhuracan/data/candles/`

### Step 2: Train Models
```bash
# Train on 3 years of data
python scripts/test_end_to_end_training.py \
  --symbols BTC/USDT ETH/USDT SOL/USDT \
  --days 1095 \
  --timeframe 1h
```

This will:
- Load data from cache (or download if missing)
- Train models on 3 years of data
- Upload models and metrics to Dropbox

---

## 🔧 Fix: Update Test Script to Upload Candle Data

The test script should also upload candle data to Dropbox. Here's what needs to be added:

### Current Behavior:
1. ✅ Load candle data (from cache)
2. ✅ Train model
3. ✅ Upload model.bin
4. ✅ Upload metrics.json
5. ❌ **Missing**: Upload candle data to Dropbox

### Should Be:
1. ✅ Download candle data (if not in cache)
2. ✅ **Upload candle data to Dropbox** (NEW)
3. ✅ Train model
4. ✅ Upload model.bin
5. ✅ Upload metrics.json
6. ✅ Upload config.json
7. ✅ Upload sha256.txt

---

## 📁 Expected Dropbox Structure

### What You Should See:

```
/Runpodhuracan/
├── data/
│   └── candles/                    ← Candle data (MISSING)
│       ├── BTC/
│       │   └── BTC-USDT_1h_*.parquet
│       ├── ETH/
│       │   └── ETH-USDT_1h_*.parquet
│       └── SOL/
│           └── SOL-USDT_1h_*.parquet
└── huracan/
    └── models/
        └── baselines/
            └── 20251111/
                ├── BTCUSDT/
                │   ├── model.bin    ← ✅ Uploaded
                │   └── metrics.json ← ✅ Uploaded
                ├── ETHUSDT/
                │   ├── model.bin    ← ✅ Uploaded
                │   └── metrics.json ← ✅ Uploaded
                └── SOLUSDT/
                    ├── model.bin    ← ✅ Uploaded
                    └── metrics.json ← ✅ Uploaded
```

### What's Actually There:
- ✅ Models and metrics (in `huracan/models/baselines/`)
- ❌ Candle data (missing from `data/candles/`)

---

## 🚀 How to Do It Properly

### Option 1: Use Separate Scripts (Recommended)

**Step 1: Download and Upload Candle Data**
```bash
python scripts/simple_download_candles.py \
  --symbols BTC/USDT ETH/USDT SOL/USDT \
  --days 1095 \
  --timeframe 1h
```

This script:
- Downloads data from exchange
- Saves to local cache
- **Uploads to Dropbox** automatically

**Step 2: Train Models**
```bash
python scripts/test_end_to_end_training.py \
  --symbols BTC/USDT ETH/USDT SOL/USDT \
  --days 1095 \
  --timeframe 1h
```

This script:
- Loads data from cache
- Trains models
- Uploads models and metrics

### Option 2: Enhanced Test Script (Future)

Update `test_end_to_end_training.py` to:
1. Download candle data (if not in cache)
2. Upload candle data to Dropbox
3. Train model
4. Upload all artifacts (model, metrics, config, hash)

---

## 📊 Data Comparison

### What You Got (30 Days):
- **Training Samples**: ~172K per coin
- **Test Samples**: ~43K per coin
- **Time Range**: Last 30 days
- **Data Quality**: Good, but limited history

### What You Should Get (3 Years):
- **Training Samples**: ~2.1M per coin (3 years × 365 days × 24 hours × 0.8)
- **Test Samples**: ~525K per coin (3 years × 365 days × 24 hours × 0.2)
- **Time Range**: Last 3 years
- **Data Quality**: Much better, captures long-term patterns

### Impact on Model Quality:
- **30 days**: Limited patterns, may overfit to recent trends
- **3 years**: Captures multiple market cycles, better generalization

---

## 🎯 Summary

### What Happened:
1. ✅ Loaded 30 days of cached candle data
2. ✅ Trained XGBoost models
3. ✅ Uploaded models and metrics to Dropbox
4. ❌ Did NOT upload candle data to Dropbox
5. ❌ Only used 30 days (not 3 years)

### What Should Happen:
1. ✅ Download 3 years of fresh candle data
2. ✅ Upload candle data to Dropbox
3. ✅ Train models on 3 years of data
4. ✅ Upload all artifacts (model, metrics, config, hash)

### Next Steps:
1. **Download 3 years of data**:
   ```bash
   python scripts/simple_download_candles.py \
     --symbols BTC/USDT ETH/USDT SOL/USDT \
     --days 1095 \
     --timeframe 1h
   ```

2. **Train on 3 years**:
   ```bash
   python scripts/test_end_to_end_training.py \
     --symbols BTC/USDT ETH/USDT SOL/USDT \
     --days 1095 \
     --timeframe 1h
   ```

3. **Verify Dropbox** has:
   - Candle data in `/Runpodhuracan/data/candles/`
   - Models in `/Runpodhuracan/huracan/models/baselines/`

---

## 🔗 Related Files

- Test Script: `scripts/test_end_to_end_training.py`
- Download Script: `scripts/simple_download_candles.py`
- Upload Script: `scripts/download_and_upload_candles.py`
- Data Loader: `src/cloud/training/datasets/data_loader.py`
- Dropbox Sync: `src/cloud/training/integrations/dropbox_sync.py`

