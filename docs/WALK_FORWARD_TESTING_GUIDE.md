# Walk-Forward Testing Guide

## Overview

This guide explains the enhanced walk-forward testing system implemented in the Huracan Engine. The system follows strict time order with no peeking, mimics live trading with realistic costs, and provides comprehensive analytics.

## Key Principles

### 1. Strict Time Order
- **Never use future data**: Features and labels are generated only from data available at decision time
- **Event time**: Train on what was known at order time only
- **Chronological validation**: Data must be sorted by timestamp before testing

### 2. Expanding and Sliding Windows
- **Expanding window**: Train on all data up to time T, predict T+1
- **Sliding window**: Train on fixed window ending at T (forgets old regimes)
- **Configurable step size**: Predict one step at a time (default) or multiple steps

### 3. Realistic Costs
- **Fees**: Maker/taker fees based on order type
- **Slippage**: Market impact and liquidity-based slippage
- **Spread**: Bid-ask spread costs
- **Funding**: Funding rate costs for perpetuals
- **Missed fills**: Partial fills and failed orders

### 4. Version Tracking
- **Model versions**: Track model code hash and parameters
- **Data versions**: Track data source and preprocessing
- **Feature versions**: Hash features to detect changes

### 5. Separate Research/Evaluation Datasets
- **Research set**: Used for model development (locked)
- **Evaluation set**: Used for walk-forward testing (locked until end)

## Architecture

### Components

#### 1. EnhancedWalkForwardTester
Main walk-forward testing engine that:
- Manages expanding/sliding windows
- Enforces strict chronology
- Tracks model and data versions
- Generates predictions and executes trades
- Records all results

#### 2. FeatureDriftDetector
Detects feature drift using statistical tests:
- **Mean shift**: t-test for mean changes
- **Variance shift**: F-test for variance changes
- **Distribution shift**: KS test for distribution changes
- **PSI**: Population Stability Index

#### 3. RegimeAwareModelSystem
Maintains separate sub-models per regime:
- **Regime classification**: Bull, bear, sideways, high/low volatility
- **Sub-models**: One model per regime type
- **Top-level classifier**: Selects appropriate sub-model
- **Time-decay weighting**: Recent samples influence more

#### 4. DailyWinLossAnalytics
Analyzes wins and losses to:
- Calculate hit rates by regime and error type
- Analyze slippage by pair and hour
- Generate calibration analysis
- Recommend risk preset updates

## Usage

### Basic Walk-Forward Test

```python
from cloud.training.validation import EnhancedWalkForwardTester, WalkForwardConfig, WindowType

# Create configuration
config = WalkForwardConfig(
    window_type=WindowType.EXPANDING,
    step_size=1,
    time_decay_weight=0.99,
    min_train_samples=100
)

# Create tester
tester = EnhancedWalkForwardTester(config=config)

# Define training function
def train_fn(data, config):
    # Train model on data
    # Return trained model
    pass

# Define prediction function
def predict_fn(model, features):
    # Make prediction
    # Return {prediction, confidence, signal_type, ...}
    pass

# Define execution function
def execute_fn(prediction, market_data):
    # Execute trade
    # Return TradeRecord
    pass

# Run walk-forward test
results = tester.run_walk_forward(
    data=historical_data,
    train_fn=train_fn,
    predict_fn=predict_fn,
    execute_fn=execute_fn,
    evaluation_start_idx=1000  # Lock evaluation set
)
```

### Regime-Aware Models

```python
from cloud.training.models.regime_aware_models import (
    RegimeAwareModelSystem,
    RegimeClassifier
)

# Create regime classifier
classifier = RegimeClassifier()

# Create regime-aware model system
system = RegimeAwareModelSystem(
    train_fn=train_fn,
    predict_fn=predict_fn,
    time_decay_weight=0.99
)

# Classify regimes
regime_labels = [
    classifier.classify(data, idx)
    for idx in range(len(data))
]

# Train sub-models
system.train_sub_models(data, regime_labels)

# Predict with regime awareness
regime = classifier.classify(data, current_idx)
prediction = system.predict(features, regime)
```

### Feature Drift Detection

```python
from cloud.training.validation import FeatureDriftDetector

# Create detector
detector = FeatureDriftDetector(
    mean_shift_threshold=0.05,
    variance_shift_threshold=0.10,
    psi_threshold=0.25
)

# Establish baseline
detector.establish_baseline(baseline_data, features=["feature1", "feature2"])

# Check drift
drift_results = detector.check_drift(current_data, features=["feature1", "feature2"])

# Get drifted features
drifted_features = detector.get_drifted_features(drift_results)
```

### Daily Analytics

```python
from cloud.training.analytics import DailyWinLossAnalytics

# Create analytics
analytics = DailyWinLossAnalytics()

# Analyze trades
analysis = analytics.analyze_trades(trades, predictions, date)

# Generate calibration analysis
calibration = analytics.analyze_calibration(predictions, trades)

# Generate risk preset updates
updates = analytics.generate_risk_preset_updates(
    analysis,
    calibration,
    current_presets={"preset1": 0.02, "preset2": 0.01}
)
```

## Database Schema

### Tables

1. **trades**: Trade records with PnL, costs, exit reasons
2. **predictions**: Prediction records with confidence, features
3. **features**: Feature snapshots at decision time
4. **attribution**: Trade attribution with SHAP/permutation importance
5. **regimes**: Regime labels (bull, bear, sideways, volatility)
6. **models**: Model metadata with versions and performance
7. **data_provenance**: Data versioning and source tracking

### Key Fields

- **trade_id**: Unique trade identifier
- **pred_id**: Prediction identifier
- **model_id**: Model identifier
- **model_version**: Model version string
- **data_version**: Data version hash
- **features_hash**: Hash of features used
- **regime**: Market regime label
- **error_type**: Error classification (direction_wrong, timing_late, etc.)

## Process Flow

1. **Clean and align data**: Remove lookahead fields, lock timestamps
2. **Generate features**: Only from past values at each step
3. **Walk forward train**: Predict one step, reveal, log, update
4. **Execute realistic fill**: Add spread, slippage, fees
5. **Write artifacts**: Store all records in database
6. **Run daily analytics**: Generate win/loss analysis
7. **Auto-update risk presets**: Lower size in bad regimes, cooldown after losses
8. **Retrain schedule**: Nightly light retrain, weekly full retrain
9. **Canary test**: Paper trade or tiny size before promotion
10. **Monitor live vs backtest gap**: Stop if slippage exceeds threshold

## Key Metrics

### Performance Metrics
- **Win rate**: Percentage of winning trades
- **Profit factor**: Average winner / average loser
- **Payoff ratio**: Average winner / |average loser|
- **Sharpe ratio**: Risk-adjusted returns
- **Sortino ratio**: Downside risk-adjusted returns
- **Max drawdown**: Maximum peak-to-trough decline

### Calibration Metrics
- **Brier score**: Calibration quality
- **Confidence buckets**: Predicted vs actual win rate by confidence
- **Calibration error**: Mean squared error of calibration

### Attribution Metrics
- **Feature importance**: SHAP/permutation importance
- **Error classification**: Direction wrong, timing late, stop too tight, etc.
- **Regime performance**: Hit rate by regime type

## Common Pitfalls

1. **Leakage from target-aligned features**: Using high/low of current candle to decide inside that candle
2. **Overfitting through repeated hyperparameter search**: Searching on same test slice multiple times
3. **Survivorship bias**: Using coin lists that exclude delisted coins
4. **Ignoring failed fills**: Not modeling partial fills or failed orders
5. **Comparing backtests with mid prices**: Trading at touch while backtesting at mid

## Best Practices

1. **Enforce chronology**: Always validate data is sorted by timestamp
2. **Model real costs**: Include fees, slippage, spread, funding
3. **Use event time**: Train on what was known at order time only
4. **Track versions**: Model and data versions in every log row
5. **Separate datasets**: Lock research and evaluation sets
6. **Monitor drift**: Detect feature drift and drop drifted features
7. **Regime awareness**: Use separate models per regime
8. **Time decay**: Weight recent samples more heavily
9. **Daily analytics**: Generate win/loss analysis daily
10. **Auto-update risk**: Adjust risk presets based on performance

## Integration with Existing Systems

### Backtesting Framework
The enhanced walk-forward tester integrates with the existing backtesting framework:
- Uses `ExecutionSimulator` for realistic execution
- Uses `TradingCoordinator` for signal processing
- Stores results in database schema

### Continuous Learning
The walk-forward tester supports online learning:
- Update model after each prediction
- Time-decay weighting for recent samples
- Model versioning and rollback

### Risk Management
Risk presets are automatically updated based on:
- Win rate by regime
- Profit factor
- Calibration quality
- Error classification

## Future Enhancements

1. **SHAP integration**: Add SHAP values for feature attribution
2. **Permutation importance**: Add permutation-based importance
3. **Monte Carlo simulation**: Add Monte Carlo for robustness testing
4. **Visualization**: Add charts for equity curves, regime transitions
5. **Real-time monitoring**: Add real-time monitoring of live vs backtest gap

