# How The Bot Works - Complete Deep Dive

## 📖 What This Bot Does (And Doesn't Do)

This bot is an automated trading system that learns from historical market data to make trading decisions. It studies price patterns, calculates features, trains machine learning models, and uses those models to generate trading signals. The bot will attempt to make profitable trades by buying and selling cryptocurrencies based on its learned patterns. However, **trading involves significant risk and losses can and will happen**. The bot is not a guarantee of profits, and past performance does not guarantee future results. You should only trade with money you can afford to lose.

---

## 🎯 How It Actually Decides

**Consensus Score (S):** The bot combines votes from multiple trading engines (typically 23 different strategies) into a single consensus score that ranges from negative (sell) to positive (buy).

**Confidence:** Each decision has a confidence level (0-100%) based on how strongly the engines agree and how reliable they've been recently.

**Threshold:** The bot only trades when the consensus score exceeds a threshold that adapts to market volatility - typically it needs to be at least 0.75 times the recent volatility of consensus scores.

---

## ⚠️ Risk Rules

**Per Trade Risk:** Each trade risks a small percentage of total equity (typically 0.5-1.5%, configurable).

**Daily Stop:** If losses exceed a daily limit (typically 2.5% of equity), trading stops for the day.

**Max Leverage:** Total exposure is capped at a maximum leverage (typically 2-3x, configurable).

**Exposure Caps:** No single coin can exceed 25% of equity, and total open risk (sum of stop losses) cannot exceed the daily loss limit.

---

## 💰 Costs

**Fees:** Every trade pays exchange fees (typically 2-5 basis points for maker orders, 4-5 bps for taker orders).

**Spread:** The difference between buy and sell prices costs money (typically 1-10 basis points).

**Funding:** Trading perpetual contracts incurs funding costs (typically 1 basis point per 8 hours).

**Slippage:** Large orders move prices, causing additional costs (varies with order size and market conditions).

**The bot only trades when the expected edge (profit) beats all these costs by a safety margin (typically 3 basis points).**

---

## 🌅 STEP 1: Wake Up and Initialize

### What Happens:

1. **System Startup**
   - Bot wakes up at scheduled time (typically 02:00 UTC, configurable)
   - Initializes logging system (structured JSON logs)
   - Loads configuration from settings files
   - Connects to PostgreSQL database (Brain Library)
   - Initializes Ray for parallel processing

2. **Create Dropbox Dated Folder**
   - Creates folder: `/Runpodhuracan/YYYY-MM-DD/` (e.g., `/Runpodhuracan/2025-11-08/`)
   - This is the FIRST action - happens immediately on startup
   - All today's data will be organized under this folder
   - Purpose: Daily organization, easy access by date, complete history tracking

3. **Start Background Sync Threads**
   - Launches 4 separate background threads for continuous Dropbox sync:
     - **Learning Data Thread**: Syncs every 5 minutes
     - **Logs & Monitoring Thread**: Syncs every 5 minutes
     - **Models Thread**: Syncs every 30 minutes
     - **Data Cache Thread**: Full sync every 2 hours, quick check every 5 minutes

### What Gets Saved to Dropbox:

**At this step:**
- Nothing yet (folder is created, but empty)

**Purpose of dated folder:**
- Organizes all daily outputs in one place
- Makes it easy to find data from any specific day
- Enables historical analysis and comparison
- Provides complete backup of daily operations

---

## 📚 STEP 2: Get Tools Ready

### What Happens:

1. **Load Configuration**
   - Reads `config/settings.yaml` or environment variables
   - Loads trading parameters, risk limits, feature settings
   - Configures exchange API credentials
   - Sets up database connection strings

2. **Connect to Exchange**
   - Initializes exchange client (Binance, Coinbase, etc.)
   - Tests API connection
   - Fetches exchange metadata (available markets, fees, limits)
   - Purpose: Need exchange connection to download price data

3. **Connect to Database (Brain Library)**
   - Connects to PostgreSQL database
   - Initializes connection pool (typically 2-10 connections)
   - Verifies database schema exists
   - Purpose: Store all training data, trades, models, metrics

4. **Set Up Telegram Notifications**
   - Connects to Telegram Bot API
   - Tests message sending
   - Purpose: Send real-time updates about training progress

5. **Initialize Health Monitor**
   - Sets up health check system
   - Configures monitoring intervals
   - Purpose: Track system health, detect issues early

### What Gets Saved to Dropbox:

**At this step:**
- `config/` folder (if config files exist)
  - `settings.yaml` - All configuration parameters
  - `*.yaml`, `*.yml`, `*.json`, `*.toml` - Any config files
  - **Purpose**: Reproducibility - know exactly what settings were used
  - **Sync frequency**: On startup (one-time)

---

## 🪙 STEP 3: Pick Which Coins to Study

### What Happens:

1. **Fetch All Available Markets**
   - Queries exchange for all trading pairs
   - Gets metadata: volume, fees, spread, age
   - Filters by exchange-specific criteria

2. **Apply Universe Selection Rules**
   - **Liquidity Filter**: Minimum daily volume (typically $1M USD, configurable)
   - **Age Filter**: Coin must be at least 30 days old (configurable)
   - **Market Type**: Only spot markets (or perps, depending on config)
   - **Exclude Delisted**: Removes coins that are no longer trading
   - **Purpose**: Focus on liquid, established coins that are safe to trade

3. **Rank and Select Top Coins**
   - Ranks by: volume, liquidity score, trading activity
   - Selects top N coins (typically 20, configurable)
   - Creates universe list: `["BTC/USDT", "ETH/USDT", "SOL/USDT", ...]`

4. **Store Universe Selection**
   - Saves selected coins to database
   - Logs selection criteria and results
   - Purpose: Track which coins were studied, enable reproducibility

### What Gets Saved to Dropbox:

**At this step:**
- Nothing directly (universe selection is logged, not exported separately)

**Later in exports:**
- Universe selection criteria and results are included in comprehensive exports

---

## 📥 STEP 4: Download Coin History

### What Happens (For Each Coin):

1. **Check Local Cache First**
   - Looks in `data/candles/` directory for existing data
   - Checks if data is recent enough (within last 24 hours)
   - If found and fresh, uses cached data (saves time!)

2. **Restore from Dropbox (If Cache Empty)**
   - Checks Dropbox shared location: `/Runpodhuracan/data/candles/`
   - Downloads all historical coin data files (`.parquet` format)
   - Saves to local cache: `data/candles/BTC-USDT.parquet`, etc.
   - **Purpose**: Avoid re-downloading data every day - saves time and API calls

3. **Download Missing/New Data**
   - If coin data doesn't exist locally or in Dropbox:
     - Connects to exchange API
     - Downloads OHLCV (Open, High, Low, Close, Volume) data
     - Timeframe: Typically 120-150 days of daily candles (configurable)
     - Data points: ~120-150 rows per coin (one per day)
   - If data exists but is incomplete:
     - Downloads only missing date ranges
     - Merges with existing data

4. **Save to Local Cache**
   - Saves as Parquet file: `data/candles/{SYMBOL}.parquet`
   - Parquet format: Compressed, columnar storage (efficient)
   - Updates file modification time to current time
   - **Purpose**: Fast local access, efficient storage

5. **Immediate Dropbox Sync (Within 5 Minutes)**
   - Background thread detects newly modified files
   - Automatically syncs to Dropbox within 5 minutes
   - Location: `/Runpodhuracan/data/candles/{SYMBOL}.parquet` (shared location, not dated folder)
   - **Purpose**: Backup historical data, share across days, restore on next startup

### What Gets Saved to Dropbox:

**File Structure:**
```
/Runpodhuracan/data/candles/          # Shared location (persists across days)
├── BTC-USDT.parquet                  # Bitcoin price history
├── ETH-USDT.parquet                  # Ethereum price history
├── SOL-USDT.parquet                  # Solana price history
└── ... (one file per coin)
```

**File Contents (Parquet Format):**
- **Columns**: `timestamp`, `open`, `high`, `low`, `close`, `volume`
- **Rows**: One per day (typically 120-150 rows per file)
- **Data Type**: OHLCV (Open, High, Low, Close, Volume) candle data
- **Time Range**: Last 120-150 days (configurable)

**Purpose of Each File:**
- **Historical Price Data**: Complete price history for training
- **Backup**: Never lose historical data
- **Restore on Startup**: Next day's run can restore from Dropbox instead of re-downloading
- **Shared Across Days**: Same data used every day, stored once in shared location

**Sync Details:**
- **Newly downloaded files**: Synced within 5 minutes ⚡
- **Full sync**: Every 2 hours (all files)
- **Quick check**: Every 5 minutes (catches new downloads)

---

## 🧮 STEP 5: Calculate Features

### What Happens (For Each Coin):

1. **Load Price Data**
   - Reads Parquet file: `data/candles/{SYMBOL}.parquet`
   - Converts to Polars DataFrame
   - Validates data quality (no missing values, correct format)

2. **Calculate Technical Indicators (50+ Features)**
   The bot calculates many different "features" - measurements that describe the market state:

   **Price-Based Features:**
   - `ret_1`, `ret_5`, `ret_20` - Returns over 1, 5, 20 periods
   - `log_ret_1`, `log_ret_5` - Logarithmic returns
   - `price_change` - Absolute price change
   - `high_low_ratio` - High/Low ratio
   - `close_open_ratio` - Close/Open ratio

   **Moving Averages:**
   - `sma_5`, `sma_10`, `sma_20`, `sma_50` - Simple moving averages
   - `ema_5`, `ema_10`, `ema_20` - Exponential moving averages
   - `price_vs_sma_20` - Price relative to moving average

   **Volatility Features:**
   - `volatility_5`, `volatility_20` - Rolling volatility
   - `atr_14` - Average True Range (14 periods)
   - `bb_upper`, `bb_lower`, `bb_width` - Bollinger Bands
   - `price_vs_bb` - Price position within Bollinger Bands

   **Momentum Features:**
   - `rsi_14` - Relative Strength Index
   - `macd`, `macd_signal`, `macd_histogram` - MACD indicators
   - `stoch_k`, `stoch_d` - Stochastic oscillator
   - `adx_14` - Average Directional Index (trend strength)

   **Volume Features:**
   - `volume_sma_20` - Volume moving average
   - `volume_ratio` - Current volume vs average
   - `price_volume_trend` - Price-volume trend
   - `on_balance_volume` - OBV indicator

   **Cross-Asset Features (If Available):**
   - `btc_correlation` - Correlation with Bitcoin
   - `eth_correlation` - Correlation with Ethereum
   - `market_beta` - Beta relative to market

   **Regime Features:**
   - `trend_strength` - How strong is the trend (0-1)
   - `volatility_regime` - Low/Medium/High volatility
   - `market_regime` - TREND/RANGE/PANIC/ILLIQUID

   **Time-Based Features:**
   - `hour_of_day` - Hour of day (0-23)
   - `day_of_week` - Day of week (0-6)
   - `is_weekend` - Boolean (weekend or not)

   **Higher-Order Features:**
   - `feature_interactions` - Combinations of features
   - `lagged_features` - Features from previous periods
   - `rolling_statistics` - Rolling min/max/mean/std

3. **Store Features in Database**
   - Saves calculated features to PostgreSQL `features` table
   - Stores: symbol, timestamp, feature_name, feature_value
   - Purpose: Fast retrieval, historical tracking

4. **Create Feature DataFrame**
   - Combines all features into single DataFrame
   - One row per timestamp, one column per feature
   - Ready for model training

### What Gets Saved to Dropbox:

**At this step:**
- Features are NOT directly saved to Dropbox (stored in database)
- Feature calculations are logged in learning snapshots

**Later in exports:**
- Feature importance rankings are included in model metadata
- Feature values are included in trade history exports

**Learning Snapshots** (synced every 5 minutes):
- `logs/learning/learning_snapshot_{timestamp}.json`
- Contains: Feature importance changes, new patterns discovered, insights
- **Purpose**: Track what the engine is learning in real-time

---

## 🎓 STEP 6: Learn and Test (Per-Coin Training with Hybrid Scheduler)

### What Happens:

The Engine uses a **hybrid training scheduler** to train one model per coin efficiently. This section describes the training process.

#### 6A. Initialize Hybrid Training Scheduler

1. **Select Training Mode**
   - **Sequential mode**: Train one coin at a time (safe, simple, slower)
   - **Parallel mode**: Train all coins at once (fast, requires more resources)
   - **Hybrid mode** (default): Train coins in batches with concurrency cap (balanced)
   - Auto-detects GPU availability and sets defaults:
     - GPU nodes: 12 concurrent workers
     - CPU-only nodes: 2 concurrent workers

2. **Initialize Scheduler**
   - Creates work queue for symbols
   - Sets up resume ledger: `runs/YYYYMMDDZ/status.json`
   - Configures timeout (default: 45 minutes per coin)
   - Sets up retry logic (up to 2 retries with jittered backoff)
   - Initializes storage client (Dropbox or S3)

3. **Load Symbols**
   - Reads symbol list from config or command line
   - Supports: `topN`, CSV file, or comma-separated list
   - Filters completed symbols (unless `--force` flag)

#### 6B. Train Each Coin (Per-Coin Training Loop)

For each coin in the queue:

1. **Fetch Costs Before Training**
   - Fetches per-symbol costs: fees, spread, slippage
   - Saves to `costs.json` in work directory
   - Purpose: Use costs for after-cost evaluation

2. **Create Unique Work Directory**
   - Creates: `models/{SYMBOL}/YYYYMMDD_HHMMSSZ/`
   - Purpose: Isolate each training run, prevent conflicts

3. **Load Data and Build Features**
   - Loads historical price data
   - Calculates features (50+ features)
   - Saves partial outputs early:
     - `features.parquet` - Feature data
     - `split_indices.json` - Train/test split indices
     - `training_log.json` - Training progress log

4. **Train Shared Encoder (First Run)**
   - Trains shared encoder on all coin features
   - Captures common patterns across coins
   - Saves to `meta/shared_encoder.pkl`
   - Purpose: Cross-coin learning without coupling

5. **Train Per-Coin Model**
   - Combines shared encoder features + coin-specific features
   - Trains XGBoost/LightGBM model per coin
   - Uses walk-forward validation
   - Optimizes hyperparameters
   - Calculates feature importance

6. **Evaluate Model (After-Cost Metrics)**
   - Calculates performance metrics:
     - Sharpe ratio, hit rate, net PnL, max drawdown
     - Applies costs (fees, spread, slippage)
     - Net PnL after costs
   - Validates against quality gates:
     - Sharpe > 0.5
     - Hit rate > 0.45
     - Net PnL > 0
     - Max drawdown < 20%
     - Sample size > 100

7. **Save Final Artifacts**
   - `model.bin` - Trained model
   - `config.json` - Model configuration
   - `metrics.json` - Performance metrics
   - `costs.json` - Cost model
   - `features.json` - Feature recipe
   - `sha256.txt` - Integrity hash

8. **Upload to Storage**
   - Uploads to Dropbox (or S3) as training completes
   - Verifies upload success
   - Logs upload events

9. **Update Resume Ledger**
   - Marks symbol as completed/failed/skipped
   - Records metrics, output path, error (if any)
   - Purpose: Enable resumability

#### 6C. Handle Failures and Retries

1. **Timeout Protection**
   - Each coin has a timeout (default: 45 minutes)
   - If timeout exceeded, marks as failed
   - Retries up to 2 times with jittered backoff

2. **Error Handling**
   - Captures exceptions and error types
   - Logs error details
   - Sends Telegram alert (if configured)
   - Marks symbol as failed in resume ledger

3. **Resume on Interruption**
   - If training is interrupted, resume ledger tracks progress
   - On restart, skips completed symbols (unless `--force`)
   - Resumes from checkpoints if available

#### 6D. Generate Summary and Roster

1. **Generate Summary**
   - Creates `summary/YYYYMMDDZ/engine_summary.json`
   - Contains: total_symbols, succeeded, failed, skipped, avg_train_minutes, median_train_minutes, total_wall_minutes, by_symbol metrics

2. **Generate Roster**
   - Creates `champions/roster.json`
   - Ranks symbols by liquidity, cost, and recent net edge
   - Fields: symbol, model_path, rank, spread_bps, fee_bps, avg_slip_bps, last_7d_net_bps, trade_ok
   - Purpose: Hamilton uses this for trading decisions

3. **Update Champion Pointers**
   - Updates `champions/{SYMBOL}.json` for each coin
   - Points to best model for each coin
   - Purpose: Quick lookup of best model per coin

### What Gets Saved to Dropbox:

**Per-Coin Artifacts** (uploaded as training completes):
```
/Huracan/models/{SYMBOL}/YYYYMMDD_HHMMSSZ/
├── model.bin                        # Trained model
├── config.json                      # Model configuration
├── metrics.json                     # Performance metrics
├── costs.json                       # Cost model (fees, spread, slippage)
├── features.json                    # Feature recipe
├── sha256.txt                       # Integrity hash
├── features.parquet                 # Feature data (partial)
├── split_indices.json               # Train/test split (partial)
└── training_log.json                # Training progress log (partial)
```

**Champion Pointers**:
```
/Huracan/champions/
├── BTCUSDT.json                     # Champion pointer for BTC
├── ETHUSDT.json                     # Champion pointer for ETH
└── roster.json                      # Ranked roster for Hamilton
```

**Resume Ledger**:
```
/Huracan/runs/YYYYMMDDZ/
└── status.json                      # Training status per symbol
```

**Summary**:
```
/Huracan/summary/YYYYMMDDZ/
└── engine_summary.json              # Overall training statistics
```

**Model File Contents:**
- **model.bin**: Trained ML model (XGBoost/LightGBM)
- **config.json**: Model configuration, hyperparameters
- **metrics.json**: Performance metrics (Sharpe, hit rate, net PnL, etc.)
- **costs.json**: Cost model (fees, spread, slippage)
- **features.json**: Feature recipe with hash
- **sha256.txt**: Integrity hash for verification
- **Purpose**: Hamilton (trading bot) loads these models to make real trading decisions

**Model Metadata** (in `models/{SYMBOL}/metadata.json`):
```json
{
  "model_id": "uuid-string",
  "symbol": "BTC-USDT",
  "version": "1.0.0",
  "training_date": "2025-11-08",
  "performance_metrics": {
    "sharpe_ratio": 1.25,
    "win_rate": 0.52,
    "max_drawdown": 0.08,
    "total_return": 0.15
  },
  "feature_importance": {
    "rsi_14": 0.15,
    "volatility_20": 0.12,
    "macd": 0.10,
    ...
  },
  "training_parameters": {
    "lookback_days": 150,
    "train_test_split": 0.8,
    "model_type": "LightGBM"
  }
}
```
- **Purpose**: Complete metadata about model - performance, features, parameters

**Learning Snapshots** (synced every 5 minutes):
```
/Runpodhuracan/YYYY-MM-DD/learning/
├── learning_snapshot_20251108_020000.json
├── learning_snapshot_20251108_030000.json
└── ... (one per learning event)
```

**Learning Snapshot Contents (.json):**
```json
{
  "timestamp": "2025-11-08T02:00:00Z",
  "symbol": "BTC-USDT",
  "insights": [
    {
      "type": "pattern_discovery",
      "description": "High RSI + Low volatility = 65% win rate",
      "confidence": 0.85
    },
    {
      "type": "feature_importance_change",
      "feature": "macd",
      "old_importance": 0.08,
      "new_importance": 0.12,
      "change": "+50%"
    }
  ],
  "patterns_learned": 5,
  "model_improvements": 2
}
```
- **Purpose**: Track what the engine is learning in real-time, capture insights

---

## 💾 STEP 7: Save Everything

### What Happens:

1. **Save Model Files**
   - Saves `.pkl` model files to `models/` directory
   - Saves metadata JSON files to `models/{SYMBOL}/metadata.json`
   - Purpose: Models ready for Hamilton to use

2. **Save Trade History**
   - All simulated trades saved to database `trade_memory` table
   - Includes: entry/exit, P&L, costs, features, regime
   - Purpose: Complete history for analysis

3. **Save Performance Metrics**
   - Model performance saved to `model_performance` table
   - Daily metrics saved to `daily_metrics` table
   - Purpose: Track performance over time

4. **Save Learning Insights**
   - Learning snapshots saved to `logs/learning/*.json`
   - Pattern library saved to `pattern_library` table
   - Purpose: Capture all learnings

5. **Generate Reports**
   - Creates performance reports
   - Creates analytics reports
   - Saves to `reports/` directory
   - Purpose: Human-readable summaries

### What Gets Saved to Dropbox:

**Logs** (synced every 5 minutes):
```
/Runpodhuracan/YYYY-MM-DD/logs/
├── engine_monitoring_20251108_020000.log
├── training.log
├── error.log
└── ... (all log files)
```

**Log File Contents (.log):**
- **Structured JSON logs**: Every action, decision, error
- **Training progress**: "Training BTC-USDT...", "Model trained, Sharpe: 1.25"
- **Errors and warnings**: Any issues encountered
- **Performance metrics**: Real-time metrics during training
- **Purpose**: Complete audit trail, debugging, monitoring

**Monitoring Data** (synced every 5 minutes):
```
/Runpodhuracan/YYYY-MM-DD/monitoring/
├── health_check_20251108.json
├── performance_metrics.json
├── system_status.json
└── ... (monitoring JSON files)
```

**Monitoring File Contents (.json):**
```json
{
  "timestamp": "2025-11-08T02:00:00Z",
  "component": "database",
  "status": "healthy",
  "latency_ms": 12.5,
  "error_rate": 0.0,
  "data_gaps": 0,
  "model_load_time_ms": 245.3
}
```
- **Purpose**: System health monitoring, detect issues early

**Reports** (synced every 5 minutes):
```
/Runpodhuracan/YYYY-MM-DD/reports/
├── training_report.json
├── performance_analysis.csv
├── model_comparison.html
└── ... (reports in various formats)
```

**Report Contents:**
- **Training Report**: Summary of training session, models created, performance
- **Performance Analysis**: Detailed performance metrics, charts, comparisons
- **Model Comparison**: Compare different models, versions
- **Purpose**: Human-readable summaries, analysis, decision-making

---

## 📊 STEP 8: Export Everything (Comprehensive A-Z Export)

### What Happens:

The bot runs a **comprehensive data exporter** that exports EVERYTHING from the database and file system to CSV/JSON files.

#### 8A. PostgreSQL Database Exports

1. **Trade History Export**
   - Exports all trades from `trade_memory` table
   - File: `exports/trade_history_{date}.csv`
   - Columns: trade_id, symbol, entry_timestamp, entry_price, exit_timestamp, exit_price, position_size, direction, gross_profit_bps, net_profit_gbp, fees_gbp, slippage_bps, market_regime, volatility_bps, is_winner, model_version, etc.
   - **Purpose**: Complete trade history for analysis

2. **All Trades Complete Export**
   - Exports ALL trades (not just today's) - complete history
   - File: `exports/all_trades_complete_{date}.csv`
   - Same columns as trade history
   - **Purpose**: Complete historical trade database

3. **Model Performance Export**
   - Exports from `model_performance` table
   - File: `exports/model_performance_{date}.csv`
   - Columns: model_id, symbol, evaluation_date, sharpe_ratio, win_rate, total_return, max_drawdown, num_trades, etc.
   - **Purpose**: Track model performance over time

4. **Win/Loss Analysis Export**
   - Exports from `win_analysis` and `loss_analysis` tables
   - File: `exports/win_loss_analysis_{date}.json`
   - Contains: Detailed analysis of every win and loss, patterns, reasons
   - **Purpose**: Understand what makes trades win or lose

5. **Pattern Library Export**
   - Exports from `pattern_library` table
   - File: `exports/pattern_library_{date}.csv`
   - Columns: pattern_id, pattern_description, win_rate, total_occurrences, avg_profit_bps, etc.
   - **Purpose**: All learned patterns with performance metrics

6. **Pattern Performance Export**
   - Exports pattern performance metrics
   - File: `exports/pattern_performance_{date}.csv`
   - **Purpose**: Track which patterns are most profitable

7. **Post-Exit Tracking Export**
   - Exports from `post_exit_tracking` table
   - File: `exports/post_exit_tracking_{date}.csv`
   - Contains: What happened after we exited trades (did price continue moving?)
   - **Purpose**: Learn if we're exiting too early or too late

8. **Regime Analysis Export**
   - Aggregates performance by market regime
   - File: `exports/regime_analysis_{date}.csv`
   - Columns: market_regime, total_trades, wins, losses, win_rate, avg_profit_gbp, total_profit_gbp, etc.
   - **Purpose**: Understand performance in different market conditions

9. **Model Evolution Export**
   - Exports model performance over time
   - File: `exports/model_evolution_{date}.csv`
   - **Purpose**: Track how models improve (or degrade) over time

#### 8B. SQLite Observability Exports

1. **Observability Trades Export**
   - Exports from SQLite `trades` table
   - File: `exports/observability_trades_{date}.csv`
   - **Purpose**: Trade records from observability journal

2. **Observability Models Export**
   - Exports from SQLite `models` table
   - File: `exports/observability_models_{date}.csv`
   - **Purpose**: Model records from observability journal

3. **Observability Model Deltas Export**
   - Exports from SQLite `model_deltas` table
   - File: `exports/observability_model_deltas_{date}.csv`
   - **Purpose**: Track changes in models over time

#### 8C. File System Exports

1. **Learning Snapshots Export**
   - Copies all learning snapshot JSON files
   - Files: `exports/learning_*.json`
   - **Purpose**: Complete learning history

2. **Backtest Results Export**
   - Copies all backtest CSV/JSON files
   - Files: `exports/backtest_*.csv`, `exports/backtest_*.json`
   - **Purpose**: All backtest outcomes

3. **Training Artifacts Export**
   - Copies model metadata files
   - Files: `exports/artifact_*_metadata.json`
   - **Purpose**: Complete model metadata

#### 8D. Metrics & Summary Exports

1. **Comprehensive Metrics Export**
   - Aggregates all metrics into single JSON file
   - File: `exports/comprehensive_metrics_{date}.json`
   - Contains: total_trades, win_rate, total_profit_gbp, patterns_learned, models_trained, etc.
   - **Purpose**: Quick overview of all metrics

2. **Performance Summary Export**
   - Export summary and metadata
   - File: `exports/performance_summary_{date}.json`
   - Contains: List of all exports, export timestamp, summary info
   - **Purpose**: Index of all exported data

### What Gets Saved to Dropbox:

**Exports Folder** (synced every 30 minutes):
```
/Runpodhuracan/YYYY-MM-DD/exports/
├── trade_history_2025-11-08.csv
├── all_trades_complete_2025-11-08.csv
├── model_performance_2025-11-08.csv
├── win_loss_analysis_2025-11-08.json
├── pattern_library_2025-11-08.csv
├── pattern_performance_2025-11-08.csv
├── post_exit_tracking_2025-11-08.csv
├── regime_analysis_2025-11-08.csv
├── model_evolution_2025-11-08.csv
├── observability_trades_2025-11-08.csv
├── observability_models_2025-11-08.csv
├── observability_model_deltas_2025-11-08.csv
├── learning_*.json (multiple files)
├── backtest_*.csv (multiple files)
├── artifact_*_metadata.json (multiple files)
├── comprehensive_metrics_2025-11-08.json
└── performance_summary_2025-11-08.json
```

**Purpose of Each Export File:**
- **Trade History**: Complete record of all trades for analysis
- **All Trades Complete**: Full historical database (not just today)
- **Model Performance**: Track how well models perform
- **Win/Loss Analysis**: Understand what makes trades win or lose
- **Pattern Library**: All learned patterns with performance
- **Pattern Performance**: Which patterns are most profitable
- **Post-Exit Tracking**: Learn about exit timing
- **Regime Analysis**: Performance in different market conditions
- **Model Evolution**: How models change over time
- **Observability Data**: System observability records
- **Learning Snapshots**: Everything the engine learned
- **Backtest Results**: All backtest outcomes
- **Training Artifacts**: Complete model metadata
- **Comprehensive Metrics**: Quick overview of all metrics
- **Performance Summary**: Index of all exports

**Sync Frequency**: Every 30 minutes (exports are large files, don't need real-time sync)

---

## 🔄 STEP 9: Continuous Background Sync

### What Happens:

Four background threads run continuously, syncing different data types at different intervals:

1. **Learning Data Thread** (Every 5 minutes)
   - Syncs: `logs/learning/*.json`
   - Purpose: Capture insights quickly, never lose learnings

2. **Logs & Monitoring Thread** (Every 5 minutes)
   - Syncs: `logs/*.log`, `logs/*.json`, `reports/*`, `exports/*`
   - Purpose: Real-time monitoring, debugging, complete backup

3. **Models Thread** (Every 30 minutes)
   - Syncs: `models/*.pkl`, `models/*/metadata.json`
   - Purpose: Models don't change often, but need backups

4. **Data Cache Thread** (Full sync every 2 hours, quick check every 5 minutes)
   - Full sync: All files in `data/candles/`
   - Quick check: Files modified in last 10 minutes (catches new downloads)
   - Purpose: Backup historical data, restore on startup

### What Gets Saved to Dropbox:

**Everything is continuously synced!** No data is lost, even if the system crashes.

---

## 🏥 STEP 10: Safety Checks

### What Happens:

1. **Health Checks** (Continuous)
   - Checks database connection
   - Checks exchange API connection
   - Checks all services
   - Monitors latency, error rates
   - **Purpose**: Detect issues early

2. **Circuit Breakers** (During Trading)
   - **Daily Drawdown Breaker**: Stops trading if losses exceed daily limit (typically 3% of equity)
   - **Streak Breaker**: Reduces position size after 3 consecutive losses
   - **Volatility Breaker**: Switches to defense mode when volatility exceeds 95th percentile
   - **Purpose**: Protect capital from runaway losses

3. **Kill Switch** (Emergency)
   - Can immediately stop all trading
   - Cancels all open orders
   - Sets position sizes to zero
   - **Purpose**: Emergency stop in case of critical issues

### What Gets Saved to Dropbox:

**Health Check Data** (synced every 5 minutes):
- Health check results in monitoring JSON files
- Circuit breaker activations logged
- Kill switch activations logged (critical events)

---

## 📱 STEP 11: Send Updates

### What Happens:

1. **Telegram Notifications**
   - Sends messages at key milestones:
     - "Starting training..."
     - "Training BTC-USDT... (1/20)"
     - "Model trained: Sharpe 1.25, Win Rate 52%"
     - "Training complete: 20 models created, 15 passed, 5 failed"
   - **Purpose**: Real-time updates, know what's happening

### What Gets Saved to Dropbox:

- Telegram messages are NOT saved to Dropbox (they're sent, not stored)
- Training progress is logged in log files (which ARE saved to Dropbox)

---

## 🎉 STEP 12: Clean Up and Finish

### What Happens:

1. **Save Final Log File**
   - Saves complete log of entire session
   - **Purpose**: Complete audit trail

2. **Verify Dropbox Sync**
   - Checks that all data has been synced
   - **Purpose**: Ensure nothing is lost

3. **Shut Down Ray Workers**
   - Closes parallel processing workers
   - **Purpose**: Clean up resources

4. **Final Summary**
   - Logs final summary: models created, performance, time taken
   - **Purpose**: Quick overview of session

### What Gets Saved to Dropbox:

- Final log file
- Any remaining unsynced files
- Complete session summary

---

## 📊 Complete Dropbox Folder Structure

```
Dropbox/
└── Runpodhuracan/
    ├── 2025-11-08/                          # Today's dated folder
    │   ├── data/
    │   │   └── candles/                     # Historical coin data (shared location)
    │   │       ├── BTC-USDT.parquet
    │   │       ├── ETH-USDT.parquet
    │   │       └── ...
    │   ├── learning/                        # Learning snapshots
    │   │   ├── learning_snapshot_20251108_020000.json
    │   │   └── ...
    │   ├── models/                          # Trained models
    │   │   ├── BTC-USDT_model.pkl
    │   │   ├── ETH-USDT_model.pkl
    │   │   └── ...
    │   ├── logs/                            # All logs
    │   │   ├── engine_monitoring_20251108_020000.log
    │   │   ├── training.log
    │   │   └── ...
    │   ├── monitoring/                      # Monitoring data
    │   │   ├── health_check_20251108.json
    │   │   └── ...
    │   ├── reports/                         # Reports
    │   │   ├── training_report.json
    │   │   └── ...
    │   ├── exports/                         # COMPREHENSIVE EXPORTS (A-Z)
    │   │   ├── trade_history_2025-11-08.csv
    │   │   ├── all_trades_complete_2025-11-08.csv
    │   │   ├── model_performance_2025-11-08.csv
    │   │   ├── win_loss_analysis_2025-11-08.json
    │   │   ├── pattern_library_2025-11-08.csv
    │   │   ├── pattern_performance_2025-11-08.csv
    │   │   ├── post_exit_tracking_2025-11-08.csv
    │   │   ├── regime_analysis_2025-11-08.csv
    │   │   ├── model_evolution_2025-11-08.csv
    │   │   ├── observability_trades_2025-11-08.csv
    │   │   ├── observability_models_2025-11-08.csv
    │   │   ├── observability_model_deltas_2025-11-08.csv
    │   │   ├── learning_*.json
    │   │   ├── backtest_*.csv
    │   │   ├── artifact_*_metadata.json
    │   │   ├── comprehensive_metrics_2025-11-08.json
    │   │   └── performance_summary_2025-11-08.json
    │   └── config/                          # Configuration files
    │       └── settings.yaml
    └── data/                                # Shared historical data (persists across days)
        └── candles/
            ├── BTC-USDT.parquet
            ├── ETH-USDT.parquet
            └── ...
```

---

## 📊 Outcomes We Track

**Win Rate:** The percentage of trades that make money (typically tracked per engine and overall).

**Sharpe Ratio:** A measure of risk-adjusted returns - higher is better (typically we aim for > 1.0).

**Max Drawdown:** The largest peak-to-trough decline in equity - lower is better (typically we limit to 2.5-3.5% daily).

**Cost Share:** What percentage of gross profits are eaten by costs (fees, slippage, funding) - lower is better (typically 20-40%).

---

## ⏰ How Long Does It Take?

- **Total time:** Typically 1-2 hours (varies with number of coins and models)
- **Most time spent on:** Downloading coin data and training models
- **Runs:** Once per day (typically at 2:00 AM UTC, configurable)

---

## 🎉 That's It!

The bot is like a smart student that:
- Studies every day
- Learns from history
- Practices trading
- Creates models (brains)
- Saves everything (to Dropbox)
- Tells you what it did
- Stays safe with circuit breakers and kill switches

And then the trading system uses those models to make real trades! 🚀

**Remember: Trading involves risk. Losses can and will happen. Only trade with money you can afford to lose.**

