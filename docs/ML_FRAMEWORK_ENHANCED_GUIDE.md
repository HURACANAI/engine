# ML Framework Enhanced Guide - Complete ML Concepts Integration

## Overview

The enhanced ML Framework now incorporates all key concepts from modern ML pipelines:
- ✅ Data ingestion and cleaning
- ✅ Feature scaling and selection
- ✅ Bias-variance diagnostics
- ✅ Cross-validation
- ✅ Learning rate scheduling
- ✅ Unsupervised learning (clustering)
- ✅ Visualization utilities
- ✅ Comprehensive evaluation

## New Components

### 1. Feature Selection

**File**: `src/cloud/training/ml_framework/feature_selection.py`

**Methods**:
- **Importance-based**: Use model feature importance (Random Forest, XGBoost)
- **Correlation**: Select features based on correlation with target
- **Mutual Information**: Select features using mutual information
- **F-test**: Select features using F-test

**Usage**:
```python
from src.cloud.training.ml_framework import FeatureSelector

selector = FeatureSelector(
    method="importance",
    n_features=50,  # Select top 50 features
)

X_selected = selector.fit_transform(X_train, y_train, model=rf_model)
X_test_selected = selector.transform(X_test)

# Get feature scores
scores = selector.get_feature_scores()
```

### 2. Cross-Validation

**File**: `src/cloud/training/ml_framework/validation.py`

**Features**:
- K-fold cross-validation
- Time-series cross-validation (preserves temporal order)
- Bias-variance diagnostics
- Train/validation/test splitting

**Usage**:
```python
from src.cloud.training.ml_framework import CrossValidator, create_train_val_test_split

# Create train/val/test split
X_train, X_val, X_test, y_train, y_val, y_test = create_train_val_test_split(
    X, y, test_size=0.2, val_size=0.1
)

# Cross-validation
validator = CrossValidator(cv_folds=5, use_time_series_split=True)
cv_results = validator.cross_validate(model, X_train, y_train)

# Bias-variance diagnosis
diagnostics = validator.bias_variance_diagnosis(
    model, X_train, y_train, X_val, y_val, X_test, y_test
)

if diagnostics.overfitting_detected:
    print("Overfitting detected! Add regularization or reduce complexity.")
```

### 3. Learning Rate Scheduling

**File**: `src/cloud/training/ml_framework/scheduler.py`

**Schedulers**:
- **Step**: Step decay (reduce LR every N epochs)
- **Cosine**: Cosine annealing
- **Exponential**: Exponential decay
- **Plateau**: Reduce on plateau (based on validation loss)

**Usage**:
```python
# In neural network config
hyperparameters:
  learning_rate: 0.001
  scheduler:
    type: "step"
    step_size: 10
    gamma: 0.1  # Reduce by 10x every 10 epochs
```

### 4. Clustering (Unsupervised Learning)

**File**: `src/cloud/training/ml_framework/clustering.py`

**Use Cases**:
- Market regime detection (bullish, bearish, neutral)
- Volatility clustering
- Feature space exploration

**Usage**:
```python
from src.cloud.training.ml_framework import KMeansClustering, ModelConfig

config = ModelConfig(
    name="kmeans",
    hyperparameters={
        "n_clusters": 3,  # Bullish, Bearish, Neutral
    },
)

clustering = KMeansClustering(config)
clustering.fit(X_train)

# Predict clusters
clusters = clustering.predict(X_test)

# Get cluster statistics
stats = clustering.get_cluster_statistics(X_train)
```

### 5. Visualization

**File**: `src/cloud/training/ml_framework/visualizer.py`

**Plots**:
- Predictions vs Actual
- Residuals plot
- Feature importance
- Training curves
- Confusion matrix
- ROC curve
- Model comparison
- Bias-variance tradeoff

**Usage**:
```python
from src.cloud.training.ml_framework import ModelVisualizer

visualizer = ModelVisualizer(output_dir=Path("plots"))

# Plot predictions
visualizer.plot_predictions_vs_actual(y_true, y_pred)

# Plot feature importance
visualizer.plot_feature_importance(feature_importance, top_n=20)

# Plot training curve
visualizer.plot_training_curve(train_losses, val_losses)

# Plot confusion matrix
visualizer.plot_confusion_matrix(y_true, y_pred)

# Plot ROC curve
visualizer.plot_roc_curve(y_true, y_proba)
```

## Enhanced Configuration

### Feature Selection

```yaml
feature_selection:
  enabled: true
  method: "importance"  # "importance", "correlation", "mutual_info", "f_test"
  n_features: 50  # Select top 50 features
  percentile: null
  threshold: null
```

### Cross-Validation

```yaml
training:
  use_cross_validation: true
  cv_folds: 5
  use_time_series_split: true  # Preserve temporal order
```

### Learning Rate Scheduling

```yaml
neural_models:
  lstm:
    hyperparameters:
      scheduler:
        type: "step"  # "step", "cosine", "exponential", "plateau"
        step_size: 10
        gamma: 0.1
```

### Clustering

```yaml
clustering_models:
  kmeans:
    enabled: true
    hyperparameters:
      n_clusters: 3  # Bullish, Bearish, Neutral
```

## Complete Pipeline Example

```python
from src.cloud.training.ml_framework import (
    MLEngineOrchestrator,
    FeatureSelector,
    CrossValidator,
    ModelVisualizer,
    create_train_val_test_split,
)
import pandas as pd

# Initialize orchestrator
orchestrator = MLEngineOrchestrator("config/ml_framework.yaml")

# Load data
data = pd.read_csv("data/training_data.csv")
X = data.drop(columns=["net_edge_bps"])
y = data["net_edge_bps"]

# Create train/val/test split
X_train, X_val, X_test, y_train, y_val, y_test = create_train_val_test_split(
    X, y, test_size=0.2, val_size=0.1
)

# Feature selection (optional)
selector = FeatureSelector(method="importance", n_features=50)
X_train_selected = selector.fit_transform(X_train, y_train)
X_val_selected = selector.transform(X_val)
X_test_selected = selector.transform(X_test)

# Train models
results = orchestrator.train_all_models(X_train_selected, y_train, X_val_selected, y_val)

# Cross-validation
validator = CrossValidator(cv_folds=5, use_time_series_split=True)
for model_name, model in orchestrator.models.items():
    cv_results = validator.cross_validate(model, X_train_selected, y_train)
    print(f"{model_name}: CV Score = {cv_results['mean_score']:.4f} ± {cv_results['std_score']:.4f}")

# Bias-variance diagnostics
for model_name, model in orchestrator.models.items():
    diagnostics = validator.bias_variance_diagnosis(
        model, X_train_selected, y_train, X_val_selected, y_val, X_test_selected, y_test
    )
    print(f"{model_name}: Overfitting={diagnostics.overfitting_detected}")

# Evaluate models
metrics = orchestrator.evaluate(X_test_selected, y_test)

# Visualization
visualizer = ModelVisualizer(output_dir=Path("plots"))

# Plot predictions for best model
best_model_name = max(metrics.keys(), key=lambda k: metrics[k].sharpe_ratio)
best_model = orchestrator.models[best_model_name]
y_pred = best_model.predict(X_test_selected)

visualizer.plot_predictions_vs_actual(y_test.values, y_pred)
visualizer.plot_residuals(y_test.values, y_pred)

# Plot feature importance
if hasattr(best_model, 'get_feature_importance'):
    importance = best_model.get_feature_importance()
    if importance:
        visualizer.plot_feature_importance(importance, top_n=20)

# Plot model comparison
model_metrics_dict = {name: m.to_dict() for name, m in metrics.items()}
visualizer.plot_model_comparison(model_metrics_dict, metric="sharpe_ratio")

# Auto-tune
orchestrator.auto_tune()

# Get performance report
report = orchestrator.get_performance_report()
```

## Bias-Variance Control

### Overfitting Detection

The framework automatically detects overfitting:
- **High variance**: Validation error >> Training error
- **Recommendations**: Add regularization, reduce model complexity, add dropout

### Underfitting Detection

The framework automatically detects underfitting:
- **High bias**: Training error is high, variance is low
- **Recommendations**: Increase model complexity, add more features, reduce regularization

### Regularization

Add L1/L2 regularization to models:

```yaml
core_models:
  random_forest:
    hyperparameters:
      max_depth: 10  # Limit depth to reduce overfitting
      min_samples_split: 5  # Require more samples to split
      min_samples_leaf: 2  # Require more samples in leaf

  xgboost:
    hyperparameters:
      reg_alpha: 0.1  # L1 regularization
      reg_lambda: 1.0  # L2 regularization
      max_depth: 6  # Limit depth
```

## Model Optimization

### Learning Rate Scheduling

For neural networks, use learning rate scheduling:

```yaml
neural_models:
  lstm:
    hyperparameters:
      learning_rate: 0.001
      scheduler:
        type: "cosine"  # Cosine annealing
        T_max: 50
        eta_min: 0.0
```

### Batch Size and Epochs

```yaml
neural_models:
  lstm:
    hyperparameters:
      batch_size: 32  # Adjust based on memory
      epochs: 50  # Adjust based on convergence
```

### Early Stopping

Early stopping is automatically used in XGBoost and can be added to neural networks:

```python
# XGBoost automatically uses early stopping with validation data
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=10)
```

## Evaluation & Feedback

### Standardized Metrics

The framework tracks all standard metrics:
- **Regression**: MAE, MSE, RMSE, R², Sharpe ratio, Win rate
- **Classification**: Accuracy, Precision, Recall, F1, AUC, Win rate

### Evaluation History

All evaluation results are stored and can be used for feedback:
```python
# Record performance
feedback.record_performance("random_forest", metrics)

# Get performance summary
summary = feedback.get_performance_summary("random_forest")
print(f"Average Sharpe: {summary['avg_sharpe']:.4f}")
```

## Feature Engineering & Selection

### Feature Creation

The preprocessing layer automatically creates features:
- Moving averages (SMA, EMA)
- Momentum indicators (RSI, MACD)
- Volatility metrics (ATR, Bollinger bands)
- Correlation features
- Lag features

### Feature Selection

Use feature selection to reduce dimensionality:

```python
selector = FeatureSelector(method="importance", n_features=50)
X_selected = selector.fit_transform(X_train, y_train, model=rf_model)

# Get selected features
selected_features = selector.get_selected_features()
print(f"Selected {len(selected_features)} features")

# Get feature scores
scores = selector.get_feature_scores()
```

## Advanced Features

### Market Regime Detection

Use clustering to detect market regimes:

```python
# Train clustering model
clustering = KMeansClustering(config)
clustering.fit(X_train)

# Predict regimes
regimes = clustering.predict(X_test)

# Analyze regime statistics
stats = clustering.get_cluster_statistics(X_train)
for regime, data in stats.items():
    print(f"{regime}: {data['percentage']:.1f}% of data")
```

### Ensemble Pruning

The feedback loop automatically identifies models to prune:

```python
# Get prune candidates
prune_candidates = feedback.get_prune_candidates()
print(f"Models to prune: {prune_candidates}")

# Prune models
for model_name in prune_candidates:
    feedback.prune_model(model_name)
```

## Summary

The enhanced ML Framework now includes:
- ✅ **Feature Selection**: Multiple methods (importance, correlation, mutual info, F-test)
- ✅ **Cross-Validation**: K-fold and time-series CV
- ✅ **Bias-Variance Diagnostics**: Automatic overfitting/underfitting detection
- ✅ **Learning Rate Scheduling**: Step, cosine, exponential, plateau
- ✅ **Clustering**: K-Means for market regime detection
- ✅ **Visualization**: Comprehensive plotting utilities
- ✅ **Regularization**: L1/L2 for overfitting control
- ✅ **Early Stopping**: Automatic for XGBoost and neural networks
- ✅ **Comprehensive Evaluation**: All standard metrics tracked

**Ready for production use!** 🚀

