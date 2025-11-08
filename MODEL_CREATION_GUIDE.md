# Model Creation Guide - How Engine Creates Models for Hamilton

## Overview

The Engine creates **trained ML models** daily for Hamilton to use in live trading. This guide explains how models are created, what information they contain, and how often they're generated.

## 🔄 Model Creation Schedule

### **Frequency: Daily**
- **Schedule**: Every day at **02:00 UTC**
- **Training Window**: **150 days** of historical data (configurable)
- **Model Type**: Per-symbol models (one model per trading pair)

### **Training Process**
1. **Download Historical Data**: Last 150 days of OHLCV data
2. **Feature Engineering**: Calculate 50+ features
3. **Labeling**: Triple-barrier labeling (profit/loss targets)
4. **Walk-Forward Validation**: Train on 20 days, test on 5 days
5. **Model Training**: Train LightGBM/XGBoost ensemble
6. **Validation**: Check performance metrics (Sharpe, win rate, etc.)
7. **Save Model**: Save as `.pkl` file with metadata
8. **Sync to Dropbox**: Automatically synced every 30 minutes

## 📊 What Information Models Contain

### **1. Model File (.pkl)**
The actual trained model contains:
- **Trained ML Model**: LightGBM/XGBoost/Random Forest ensemble
- **Feature Weights**: Learned feature importance
- **Model Parameters**: Hyperparameters used for training
- **Scalers**: Feature normalization scalers (if used)

### **2. Model Metadata (ModelManifest)**
Each model includes complete metadata:

```python
{
    "model_id": "uuid-string",           # Unique model identifier
    "created_at": "2025-11-08T02:00:00Z", # When model was created
    "symbol": "BTC/USDT",                # Trading pair
    "training_window_days": 150,         # Days of data used
    "metrics": {
        "total_trades": 1250,            # Total shadow trades
        "wins": 750,                     # Winning trades
        "losses": 500,                   # Losing trades
        "win_rate": 0.60,                # Win rate (60%)
        "total_profit_gbp": 1250.50,     # Total profit
        "sharpe_ratio": 1.85,            # Risk-adjusted returns
        "patterns_learned": 45,          # Patterns discovered
    },
    "agent_config": {...},               # RL agent configuration
    "action_space": ["BUY", "SELL", "HOLD"], # Available actions
    "feature_count": 80,                 # Number of features
    "scenario_tests": {...},             # Performance scenarios
}
```

### **3. Feature Information**
Models use **50+ features** from `FeatureRecipe`:

#### **Momentum Features**
- `ret_1`, `ret_3`, `ret_5` - Returns over 1, 3, 5 periods
- `zscore_ret_1`, `zscore_ret_3` - Normalized returns
- `momentum_slope` - Momentum acceleration

#### **Trend Features**
- `trend_strength` - Trend strength (-1 to 1)
- `ema_5`, `ema_21`, `ema_8`, `ema_34` - Exponential moving averages
- `ema_diff_5_21`, `ema_diff_8_34` - EMA crossovers
- `adx` - Average Directional Index (trend strength)
- `htf` - Higher TimeFrame bias

#### **Volatility Features**
- `realized_sigma_30`, `realized_sigma_60` - Realized volatility
- `atr` - Average True Range
- `vol_regime` - Volatility regime indicator
- `kurt` - Kurtosis (tail risk)

#### **Range/Compression Features**
- `compression` - Range compression score (0-1)
- `nr7_dens` - NR7 (Narrow Range 7) density
- `compress_rank` - Compression percentile rank
- `mean_rev` - Mean reversion bias
- `pullback` - Pullback depth from recent high

#### **Breakout Features**
- `ignition_score` - Breakout ignition score (0-100)
- `breakout_thrust` - Breakout momentum
- `breakout_qual` - Breakout quality

#### **RSI & Oscillators**
- `rsi_7`, `rsi_14` - Relative Strength Index
- `bb_upper`, `bb_lower`, `bb_width` - Bollinger Bands

#### **Microstructure Features**
- `micro` - Microstructure score (0-100)
- `uptick` - Uptick ratio (0-1)
- `spread` - Estimated bid-ask spread (bps)
- `orderbook_imbalance` - Order book imbalance (if available)

#### **Volume Features**
- `vol_jump` - Volume jump Z-score
- `volume_ma_ratio` - Volume vs moving average
- `volume_trend` - Volume trend

#### **Cross-Asset Features**
- `leader` - Leader bias (-1 to 1)
- `rs` - Relative strength score (0-100)
- `btc_beta` - BTC correlation
- `btc_divergence` - BTC divergence
- `cross_momentum` - Cross-asset momentum

#### **Temporal Features**
- `tod_fraction` - Time of day (0-1)
- `tod_sin`, `tod_cos` - Time of day (cyclical)
- `dow` - Day of week (0-6)
- `dow_sin`, `dow_cos` - Day of week (cyclical)

#### **Liquidity Features**
- `liquidity_score` - Liquidity score
- `spread_bps` - Spread in basis points
- `volume_score` - Volume score

### **4. Model Predictions**
Models predict:
- **Target**: `net_edge_bps` - Expected profit after costs (basis points)
- **Output**: Predicted profit for a trade
- **Confidence**: Model confidence score (0-1)

### **5. Performance Metrics**
Each model includes:
- **Sharpe Ratio**: Risk-adjusted returns
- **Profit Factor**: Total wins / Total losses
- **Win Rate**: Percentage of winning trades
- **Max Drawdown**: Largest peak-to-trough decline
- **Total Profit**: Total profit in GBP
- **Trade Count**: Number of trades
- **Costs**: Fee, spread, slippage breakdown

## 🎯 Model Training Process

### **Step 1: Data Download**
```python
# Download 150 days of historical data
loader = CandleDataLoader(exchange_client=exchange)
query = CandleQuery(
    symbol="BTC/USDT",
    start_at=now - 150 days,
    end_at=now,
)
raw_frame = loader.load(query)  # Downloads and caches data
```

### **Step 2: Feature Engineering**
```python
# Calculate 50+ features
recipe = FeatureRecipe()
feature_frame = recipe.build(raw_frame)  # Creates all features
```

### **Step 3: Labeling**
```python
# Triple-barrier labeling (profit/loss targets)
label_builder = LabelBuilder(LabelingConfig(horizon_minutes=4))
labeled = label_builder.build(feature_frame, costs)
# Creates: net_edge_bps (target variable)
```

### **Step 4: Walk-Forward Validation**
```python
# Train on 20 days, test on 5 days (repeated)
splits = walk_forward_masks(ts_series, train_days=20, test_days=5)
for train_mask, test_mask in splits:
    X_train = dataset[train_mask]
    y_train = dataset[train_mask]["net_edge_bps"]
    model.fit(X_train, y_train)
    # Evaluate on test set
```

### **Step 5: Model Training**
```python
# Train final model on all data
model = LGBMRegressor(
    objective="regression",
    learning_rate=0.05,
    n_estimators=300,
    max_depth=6,
    ...
)
model.fit(X_train, y_train)  # Trains on all features
```

### **Step 6: Save Model**
```python
# Save model as .pkl file
model_bytes = pickle.dumps(final_model)
# Save to: models/{symbol}_model.pkl
# Also saves: feature metadata, hyperparameters, metrics
```

## 📁 Model Files Structure

### **Local Storage**
```
models/
├── BTC-USDT_model.pkl          # Trained model
├── BTC-USDT_metadata.json       # Model metadata
├── ETH-USDT_model.pkl
├── ETH-USDT_metadata.json
└── ...
```

### **Dropbox Storage**
```
/Runpodhuracan/2025-11-08/
└── models/
    ├── BTC-USDT_model.pkl
    ├── BTC-USDT_metadata.json
    ├── ETH-USDT_model.pkl
    └── ...
```

## 🔍 Model Contents (Detailed)

### **What Hamilton Gets**
When Hamilton loads a model, it receives:

1. **Trained Model Object** (`.pkl` file):
   - LightGBM/XGBoost model
   - Feature weights
   - Prediction function
   - Feature importance

2. **Feature Metadata**:
   - List of feature names (50+ features)
   - Feature order (for prediction)
   - Feature types (numeric, categorical)

3. **Model Configuration**:
   - Hyperparameters
   - Training window
   - Feature count
   - Model version

4. **Performance Metrics**:
   - Sharpe ratio
   - Win rate
   - Profit factor
   - Max drawdown
   - Expected edge (bps)

5. **Trading Parameters**:
   - Recommended edge threshold
   - Cost breakdown (fees, spread, slippage)
   - Max trades per day
   - Cooldown seconds

## 📈 Model Types

### **1. Single Model (Current)**
- **Type**: LightGBM Regressor
- **Predicts**: `net_edge_bps` (expected profit)
- **Features**: 50+ features
- **Training**: Daily on 150 days of data

### **2. Multi-Model Ensemble (Available)**
- **Types**: XGBoost, Random Forest, LightGBM, Logistic Regression
- **Method**: Weighted voting or stacking
- **Benefits**: Better performance, more robust
- **Status**: Implemented, can be enabled

### **3. RL Agent Models (Available)**
- **Type**: PPO (Proximal Policy Optimization)
- **Predicts**: Trading actions (BUY, SELL, HOLD)
- **Features**: 80+ features
- **Status**: Implemented, can be enabled

## 🎯 How Hamilton Uses Models

### **1. Load Model**
```python
# Hamilton loads model from Dropbox
model = pickle.load(open("models/BTC-USDT_model.pkl", "rb"))
```

### **2. Calculate Features**
```python
# Hamilton calculates same features as Engine
recipe = FeatureRecipe()
features = recipe.build(current_market_data)
```

### **3. Predict Edge**
```python
# Predict expected profit
predicted_edge_bps = model.predict(features)
```

### **4. Make Trading Decision**
```python
# If predicted edge > threshold, take trade
if predicted_edge_bps > recommended_edge_threshold:
    execute_trade()
```

## 📊 Model Performance Tracking

### **Metrics Stored**
- **Sharpe Ratio**: Risk-adjusted returns
- **Win Rate**: Percentage of winning trades
- **Profit Factor**: Total wins / Total losses
- **Max Drawdown**: Largest decline
- **Total Profit**: Profit in GBP
- **Trade Count**: Number of trades

### **Validation Results**
- **Walk-Forward**: Train/test split results
- **Out-of-Sample**: Performance on unseen data
- **Gate Results**: Pass/fail for each gate
- **Publish Status**: Published or rejected

## 🔄 Model Update Frequency

### **Daily Training**
- **Schedule**: Every day at 02:00 UTC
- **Duration**: ~1-2 hours (depending on data size)
- **Models Created**: One per symbol in universe (typically 20 models)

### **Model Sync**
- **Frequency**: Every 30 minutes
- **Location**: Dropbox `/Runpodhuracan/YYYY-MM-DD/models/`
- **Format**: `.pkl` files + metadata JSON

### **Model Registry**
- **Database**: PostgreSQL
- **Tables**: `models`, `model_metrics`, `publish_log`
- **Tracking**: Model version, performance, publish status

## 🎯 Model Quality Gates

Models must pass these gates to be published:

1. **Sharpe Ratio** ≥ 0.7
2. **Profit Factor** ≥ 1.1
3. **Max Drawdown** ≤ 1.2 × median drawdown
4. **Win Rate** ≥ median win rate - 1%
5. **Trade Count** ≥ 300 trades

### **If Model Fails Gates**
- Model is **not published** (not sent to Hamilton)
- Reason is logged
- Previous model continues to be used
- Next day's training may improve model

## 📝 Summary

### **Model Creation**
- ✅ **Frequency**: Daily at 02:00 UTC
- ✅ **Training Data**: 150 days of historical data
- ✅ **Features**: 50+ features from FeatureRecipe
- ✅ **Model Type**: LightGBM/XGBoost ensemble
- ✅ **Output**: Expected profit (net_edge_bps)

### **Model Contents**
- ✅ **Trained Model**: `.pkl` file with ML model
- ✅ **Metadata**: Model ID, creation date, metrics
- ✅ **Features**: 50+ features (momentum, trend, volatility, etc.)
- ✅ **Performance**: Sharpe, win rate, profit factor, etc.
- ✅ **Parameters**: Hyperparameters, thresholds, costs

### **Model Storage**
- ✅ **Local**: `models/{symbol}_model.pkl`
- ✅ **Dropbox**: `/Runpodhuracan/YYYY-MM-DD/models/`
- ✅ **Database**: PostgreSQL (model registry)
- ✅ **Sync**: Every 30 minutes to Dropbox

### **Hamilton Usage**
- ✅ **Load**: Loads model from Dropbox
- ✅ **Predict**: Calculates features and predicts edge
- ✅ **Trade**: Uses prediction to make trading decisions
- ✅ **Update**: Gets new model daily (if published)

**Models are created daily with the latest learning, ensuring Hamilton always has the most up-to-date trading intelligence!** 🚀

