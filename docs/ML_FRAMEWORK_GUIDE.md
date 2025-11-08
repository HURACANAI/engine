# ML Framework Guide - Modular ML System for Huracan Engine

## Overview

The ML Framework is a comprehensive, production-ready modular ML system that integrates seamlessly with the Huracan Engine. It provides:

- **Pre-processing Layer**: Normalization, feature engineering, PCA
- **Baseline Models**: Linear/Logistic Regression, KNN, SVM
- **Core Learners**: Decision Trees, Random Forest, XGBoost
- **Neural Networks**: LSTM, GRU (PyTorch)
- **Meta-Layer**: Ensemble blending with dynamic weighting
- **Feedback Loop**: Performance tracking and auto-tuning

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│              ML Engine Orchestrator                      │
└─────────────────────────────────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
┌───────▼──────┐ ┌──────▼──────┐ ┌─────▼──────┐
│ Preprocessing│ │   Models    │ │  Ensemble  │
│   Pipeline   │ │  (Multiple) │ │  Blender   │
└───────┬──────┘ └──────┬──────┘ └─────┬──────┘
        │               │               │
        └───────────────┼───────────────┘
                        │
                ┌───────▼───────┐
                │   Feedback    │
                │     Loop      │
                └───────────────┘
```

## Quick Start

### 1. Configuration

Create a configuration file (`config/ml_framework.yaml`):

```yaml
preprocessing:
  enabled: true
  normalize: true
  use_pca: true
  pca_variance_threshold: 0.95

baseline_models:
  linear_regression:
    enabled: true
    model_type: "regression"
  
  logistic_regression:
    enabled: true
    model_type: "classification"

core_models:
  random_forest:
    enabled: true
    model_type: "regression"
    hyperparameters:
      n_estimators: 100
      max_depth: 10
  
  xgboost:
    enabled: true
    model_type: "regression"
    hyperparameters:
      n_estimators: 100
      learning_rate: 0.1

ensemble:
  enabled: true
  method: "weighted_voting"

feedback:
  enabled: true
  auto_retrain_enabled: true
```

### 2. Basic Usage

```python
from src.cloud.training.ml_framework import MLEngineOrchestrator
import pandas as pd

# Initialize orchestrator
orchestrator = MLEngineOrchestrator("config/ml_framework.yaml")

# Load data
data = pd.read_csv("data/training_data.csv")
X = data.drop(columns=["target"])
y = data["target"]

# Train all models
results = orchestrator.train_all_models(X, y)

# Make predictions
predictions = orchestrator.predict(X_test, use_ensemble=True)

# Evaluate models
metrics = orchestrator.evaluate(X_val, y_val)

# Auto-tune based on performance
orchestrator.auto_tune()

# Get performance report
report = orchestrator.get_performance_report()
```

### 3. Command Line Usage

```bash
# Train models
python -m src.cloud.training.ml_framework.engine_main \
    --config config/ml_framework.yaml \
    --mode train \
    --data data/training_data.csv \
    --target net_edge_bps

# Make predictions
python -m src.cloud.training.ml_framework.engine_main \
    --config config/ml_framework.yaml \
    --mode predict \
    --data data/test_data.csv \
    --output predictions.csv

# Evaluate models
python -m src.cloud.training.ml_framework.engine_main \
    --config config/ml_framework.yaml \
    --mode evaluate \
    --data data/validation_data.csv \
    --target net_edge_bps
```

## Components

### 1. Pre-processing Layer

**File**: `src/cloud/training/ml_framework/preprocessing.py`

**Features**:
- Data normalization (StandardScaler or MinMaxScaler)
- Feature engineering (moving averages, RSI, MACD, Bollinger bands)
- PCA-based dimensionality reduction (95% variance)
- Missing data handling
- Outlier detection and handling
- Timestamp alignment

**Usage**:
```python
from src.cloud.training.ml_framework import PreprocessingPipeline, PreprocessingConfig

config = PreprocessingConfig(
    normalize=True,
    scaler_type="standard",
    use_pca=True,
    pca_variance_threshold=0.95,
    engineer_features=True,
)

pipeline = PreprocessingPipeline(config)
X_processed = pipeline.fit_transform(X_train)
X_test_processed = pipeline.transform(X_test)
```

### 2. Baseline Models

**File**: `src/cloud/training/ml_framework/baseline.py`

**Models**:
- `LinearRegressionModel`: Linear regression for continuous targets
- `LogisticRegressionModel`: Logistic regression for binary classification
- `KNNModel`: K-Nearest Neighbors (classification or regression)
- `SVMModel`: Support Vector Machine (classification or regression)

**Usage**:
```python
from src.cloud.training.ml_framework import LinearRegressionModel, ModelConfig

config = ModelConfig(
    name="linear_regression",
    model_type="regression",
    hyperparameters={"fit_intercept": True},
)

model = LinearRegressionModel(config)
metrics = model.fit(X_train, y_train, X_val, y_val)
predictions = model.predict(X_test)
```

### 3. Core Learners

**File**: `src/cloud/training/ml_framework/core.py`

**Models**:
- `DecisionTreeModel`: Decision tree (classification or regression)
- `RandomForestModel`: Random forest ensemble (classification or regression)
- `XGBoostModel`: XGBoost gradient boosting (classification or regression)

**Usage**:
```python
from src.cloud.training.ml_framework import RandomForestModel, ModelConfig

config = ModelConfig(
    name="random_forest",
    model_type="regression",
    hyperparameters={
        "n_estimators": 100,
        "max_depth": 10,
    },
)

model = RandomForestModel(config)
metrics = model.fit(X_train, y_train, X_val, y_val)
predictions = model.predict(X_test)

# Get feature importance
importance = model.get_feature_importance()
```

### 4. Neural Networks

**File**: `src/cloud/training/ml_framework/neural.py`

**Models**:
- `LSTMModel`: LSTM neural network for time-series forecasting
- `GRUModel`: GRU neural network for time-series forecasting

**Requirements**: PyTorch (`pip install torch`)

**Usage**:
```python
from src.cloud.training.ml_framework import LSTMModel, ModelConfig

config = ModelConfig(
    name="lstm",
    model_type="regression",
    hyperparameters={
        "sequence_length": 60,
        "hidden_units": 64,
        "num_layers": 2,
        "dropout": 0.2,
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 50,
        "device": "cuda",  # or "cpu"
    },
)

model = LSTMModel(config)
metrics = model.fit(X_train, y_train, X_val, y_val)
predictions = model.predict(X_test)
```

### 5. Meta-Layer (Ensemble Blending)

**File**: `src/cloud/training/ml_framework/meta.py`

**Features**:
- Weighted voting based on recent performance
- Stacking with meta-learner
- Simple averaging
- Dynamic weight adjustment

**Usage**:
```python
from src.cloud.training.ml_framework import EnsembleBlender, EnsembleConfig

config = EnsembleConfig(
    method="weighted_voting",
    performance_window_days=7,
    use_sharpe_for_weighting=True,
)

ensemble = EnsembleBlender(config)

# Add models
ensemble.add_model("random_forest", rf_model)
ensemble.add_model("xgboost", xgb_model)
ensemble.add_model("lstm", lstm_model)

# Update performance (automatically adjusts weights)
ensemble.update_performance("random_forest", metrics_rf)
ensemble.update_performance("xgboost", metrics_xgb)
ensemble.update_performance("lstm", metrics_lstm)

# Make ensemble predictions
predictions = ensemble.predict(X_test)

# Get current weights
weights = ensemble.get_weights()
```

### 6. Feedback Loop

**File**: `src/cloud/training/ml_framework/feedback.py`

**Features**:
- Track model performance over time
- Detect underperforming models
- Automatically retrain or reweight models
- Prune consistently poor performers
- Store metrics in database

**Usage**:
```python
from src.cloud.training.ml_framework import ModelFeedback, FeedbackConfig

config = FeedbackConfig(
    sharpe_threshold=0.5,
    rmse_threshold=100.0,
    win_rate_threshold=0.45,
    auto_retrain_enabled=True,
    auto_prune_enabled=True,
)

feedback = ModelFeedback(config)

# Record performance
feedback.record_performance("random_forest", metrics)

# Get retrain queue
retrain_queue = feedback.get_retrain_queue()

# Get prune candidates
prune_candidates = feedback.get_prune_candidates()

# Get performance summary
summary = feedback.get_performance_summary("random_forest")
```

## Integration with Existing Engine

### Integration Point

The ML Framework integrates with the existing `FeatureRecipe` and training pipeline:

```python
# In orchestration.py
from src.cloud.training.ml_framework import MLEngineOrchestrator

# After feature engineering
feature_frame = recipe.build(raw_frame)

# Initialize ML Framework
ml_engine = MLEngineOrchestrator("config/ml_framework.yaml")

# Train models
results = ml_engine.train_all_models(
    X_train=dataset[feature_cols],
    y_train=dataset["net_edge_bps"],
    X_val=X_val,
    y_val=y_val,
)

# Make predictions
predictions = ml_engine.predict(X_test, use_ensemble=True)
```

## Configuration

### Model Toggling

Enable/disable models in `config/ml_framework.yaml`:

```yaml
baseline_models:
  linear_regression:
    enabled: true  # Set to false to disable
  
  logistic_regression:
    enabled: false  # Disabled

core_models:
  random_forest:
    enabled: true
  
  xgboost:
    enabled: true

neural_models:
  lstm:
    enabled: false  # Disabled (requires PyTorch)
  
  gru:
    enabled: false  # Disabled (requires PyTorch)
```

### Hyperparameter Tuning

Configure hyperparameters in `config/ml_framework.yaml`:

```yaml
core_models:
  xgboost:
    hyperparameters:
      n_estimators: 200
      max_depth: 8
      learning_rate: 0.05
      subsample: 0.8
      colsample_bytree: 0.8
```

## File Structure

```
src/cloud/training/ml_framework/
├── __init__.py              # Module exports
├── base.py                  # Base model interface
├── preprocessing.py         # Pre-processing pipeline
├── baseline.py              # Baseline models
├── core.py                  # Core learners
├── neural.py                # Neural networks (LSTM, GRU)
├── meta.py                  # Ensemble blending
├── feedback.py              # Feedback loop
├── orchestrator.py          # Main orchestrator
└── engine_main.py           # Command-line entry point

config/
└── ml_framework.yaml        # Configuration file
```

## Examples

### Example 1: Train All Models

```python
from src.cloud.training.ml_framework import MLEngineOrchestrator
import pandas as pd

# Initialize
orchestrator = MLEngineOrchestrator("config/ml_framework.yaml")

# Load data
data = pd.read_csv("data/training_data.csv")
X = data.drop(columns=["net_edge_bps"])
y = data["net_edge_bps"]

# Train
results = orchestrator.train_all_models(X, y)

# Print results
for name, metrics in results.items():
    print(f"{name}: Sharpe={metrics.sharpe_ratio:.4f}, RMSE={metrics.rmse:.4f}")
```

### Example 2: Use Ensemble for Prediction

```python
# Train models (as above)
results = orchestrator.train_all_models(X_train, y_train)

# Make ensemble predictions
predictions = orchestrator.predict(X_test, use_ensemble=True)

# Evaluate ensemble
metrics = orchestrator.evaluate(X_val, y_val)
```

### Example 3: Auto-Tuning

```python
# Train and evaluate
results = orchestrator.train_all_models(X_train, y_train)
metrics = orchestrator.evaluate(X_val, y_val)

# Auto-tune (retrains underperforming models, prunes poor performers)
orchestrator.auto_tune()

# Get performance report
report = orchestrator.get_performance_report()
print(f"Retrain Queue: {report['retrain_queue']}")
print(f"Prune Candidates: {report['prune_candidates']}")
```

## Performance Tracking

### Metrics Tracked

- **MAE**: Mean Absolute Error
- **MSE**: Mean Squared Error
- **RMSE**: Root Mean Squared Error
- **Accuracy**: Classification accuracy
- **Precision**: Classification precision
- **Recall**: Classification recall
- **F1 Score**: Classification F1 score
- **Sharpe Ratio**: Risk-adjusted returns
- **Win Rate**: Percentage of winning predictions
- **R2 Score**: R-squared (regression)
- **AUC**: Area Under Curve (classification)

### Performance Storage

Performance metrics are stored in:
- **Local**: `models/ml_framework/` directory
- **Database**: PostgreSQL (if configured)
- **S3**: Amazon S3 (if configured)

## Best Practices

1. **Start with Baseline Models**: Enable baseline models first to establish performance baseline
2. **Add Core Learners**: Enable Random Forest and XGBoost for better performance
3. **Use Ensemble**: Enable ensemble blending for robust predictions
4. **Enable Feedback Loop**: Enable auto-tuning for continuous improvement
5. **Monitor Performance**: Regularly check performance reports
6. **Tune Hyperparameters**: Adjust hyperparameters based on performance
7. **Use PCA**: Enable PCA to reduce dimensionality and speed up training
8. **GPU Acceleration**: Use GPU for LSTM/GRU models (set `device: "cuda"`)

## Troubleshooting

### PyTorch Not Available

If PyTorch is not installed, LSTM/GRU models will be disabled. Install with:
```bash
pip install torch
```

### XGBoost Not Available

If XGBoost is not installed, XGBoost model will be disabled. Install with:
```bash
pip install xgboost
```

### Out of Memory

If running out of memory:
1. Reduce `batch_size` for neural networks
2. Enable PCA to reduce feature dimensionality
3. Reduce `n_estimators` for Random Forest/XGBoost
4. Use fewer models in ensemble

## Summary

The ML Framework provides a complete, production-ready ML system with:
- ✅ **Modular Design**: Each component is independent and reusable
- ✅ **Unified Interface**: All models implement the same interface
- ✅ **Configuration-Driven**: Enable/disable models via YAML
- ✅ **Performance Tracking**: Comprehensive metrics and auto-tuning
- ✅ **Ensemble Support**: Dynamic weighting based on performance
- ✅ **Production-Ready**: Logging, error handling, persistence

**Ready to use!** 🚀

