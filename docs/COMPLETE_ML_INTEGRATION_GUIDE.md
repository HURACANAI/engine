# Complete ML Integration Guide - Huracan Engine

## Overview

The Huracan Engine now includes a complete, production-ready ML framework that integrates all layers from preprocessing to MLOps.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│              Unified ML Pipeline                        │
└─────────────────────────────────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
┌───────▼──────┐ ┌──────▼──────┐ ┌─────▼──────┐
│ Preprocessing│ │   Baselines  │ │Core Learners│
│  (Enhanced)  │ │  (A/B Test)  │ │(All Models)│
└───────┬──────┘ └──────┬──────┘ └─────┬──────┘
        │               │               │
        └───────────────┼───────────────┘
                        │
                ┌───────▼───────┐
                │  Meta-Layer   │
                │ (AutoML/Stack)│
                └───────┬───────┘
                        │
                ┌───────▼───────┐
                │  Feedback     │
                │(Drift/Retrain)│
                └───────┬───────┘
                        │
                ┌───────▼───────┐
                │    MLOps      │
                │(Distributed)  │
                └───────────────┘
```

## Components

### 1. Enhanced Pre-processing

**File**: `preprocessing/enhanced_preprocessing.py`

**Features**:
- ✅ EDA and data quality checks
- ✅ Feature cleaning and normalization
- ✅ Outlier detection and handling
- ✅ Feature engineering (returns, volatility, moving averages, RSI, MACD, Bollinger bands)
- ✅ Rolling-window normalization
- ✅ Trend decomposition
- ✅ Feature lagging

**Usage**:
```python
from src.cloud.training.ml_framework.preprocessing.enhanced_preprocessing import EnhancedPreprocessor

preprocessor = EnhancedPreprocessor()
X_processed = preprocessor.process(X_train, fit=True)
```

### 2. Baselines with A/B Testing

**File**: `baselines/ab_testing.py`

**Features**:
- ✅ Statistical hypothesis testing (t-test, Mann-Whitney)
- ✅ Confidence intervals
- ✅ Effect size calculation
- ✅ Multiple comparison correction
- ✅ Model comparison framework

**Usage**:
```python
from src.cloud.training.ml_framework.baselines.ab_testing import ABTestingFramework

ab_tester = ABTestingFramework(alpha=0.05)
result = ab_tester.t_test(model_a_results, model_b_results, metric_name="sharpe_ratio")
```

### 3. Core Learners

All models are available:
- ✅ **Random Forest, XGBoost**: Ensemble learners
- ✅ **CNN**: Visual pattern detection
- ✅ **LSTM, GRU**: Sequential pattern detection
- ✅ **Transformer**: Sequence understanding
- ✅ **GAN**: Synthetic data generation
- ✅ **RL Agent**: Adaptive strategy optimization

### 4. Reinforcement Learning

**File**: `reinforcement/rl_agent.py`

**Purpose**: Adaptive strategy optimization
**Ideal dataset shape**: `(num_episodes, episode_length, state_dim)`
**Feature requirements**: State features (price, volume, indicators, position)
**Output schema**: Action (buy, hold, sell) and Q-values

**Usage**:
```python
from src.cloud.training.ml_framework.reinforcement.rl_agent import RLAgent

agent = RLAgent(state_dim=128, action_dim=3)
rewards = agent.train(episodes=1000)
action, q_values = agent.predict(state)
```

### 5. AutoML Engine

**File**: `automl/automl_engine.py`

**Features**:
- ✅ Automated model selection
- ✅ Hyperparameter optimization (Optuna)
- ✅ Cross-validation
- ✅ Best model selection

**Usage**:
```python
from src.cloud.training.ml_framework.automl.automl_engine import AutoMLEngine

automl = AutoMLEngine(models=[model1, model2, model3], n_trials=100)
best_model, best_params = automl.optimize(X_train, y_train, X_val, y_val)
```

### 6. MLOps - Drift Detection

**File**: `mlops/drift_detector.py`

**Features**:
- ✅ Data distribution drift detection
- ✅ Concept drift detection
- ✅ Statistical tests (KS test, PSI)
- ✅ Automated retraining triggers

**Usage**:
```python
from src.cloud.training.ml_framework.mlops.drift_detector import DriftDetector

detector = DriftDetector(threshold=0.05)
detector.set_reference(X_train)
drift_results = detector.detect_data_drift(X_test)
should_retrain = detector.should_retrain(X_test, y_pred, y_true, reference_performance)
```

### 7. Distributed Training

**File**: `distributed/distributed_trainer.py`

**Features**:
- ✅ Multi-GPU training
- ✅ Multi-node training
- ✅ Model parallelism
- ✅ Data parallelism

**Usage**:
```python
from src.cloud.training.ml_framework.distributed.distributed_trainer import DistributedTrainer

trainer = DistributedTrainer(model, backend="nccl")
trainer.setup_distributed(rank=0, world_size=4)
losses = trainer.train_distributed(train_loader, optimizer, loss_fn, epochs=10)
```

### 8. Model Registry

**File**: `model_registry.py`

**Features**:
- ✅ Unified model information
- ✅ Purpose, dataset shape, feature requirements
- ✅ Output schema for each model
- ✅ Market regime mapping
- ✅ Dynamic model selection

**Usage**:
```python
from src.cloud.training.ml_framework.model_registry import get_registry

registry = get_registry()
registry.register_model(model, metadata)
models_for_regime = registry.get_models_by_regime("trending")
all_models_info = registry.get_all_models_info()
```

## Complete Pipeline

**File**: `integration/unified_pipeline.py`

**Usage**:
```python
from src.cloud.training.ml_framework.integration.unified_pipeline import UnifiedMLPipeline

pipeline = UnifiedMLPipeline("config/ml_framework.yaml")

results = pipeline.run_complete_pipeline(
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    X_test=X_test,
    y_test=y_test,
)

# Get models for specific regime
models = pipeline.get_models_for_regime("trending")

# Get all models info
models_info = pipeline.get_all_models_info()
```

## Model Information for Mechanic

Each model provides:
- **Purpose**: What the model is designed for
- **Ideal dataset shape**: Expected input shape
- **Feature requirements**: Required features
- **Output schema**: Output format
- **Market regimes**: When to use the model

Example:
```python
{
    "purpose": "Adaptive strategy optimization",
    "ideal_dataset_shape": "(num_episodes, episode_length, state_dim)",
    "feature_requirements": ["price", "volume", "indicators", "position"],
    "output_schema": {"action": "int", "q_values": "array"},
    "market_regimes": ["trending", "volatile"]
}
```

## Integration with Mechanic

The Mechanic can:
1. **Dynamically select models** based on market regime
2. **Understand feature requirements** for each model
3. **Know output schemas** for proper integration
4. **Trigger retraining** based on drift detection
5. **A/B test models** for performance validation
6. **Use AutoML** for hyperparameter optimization

## Summary

✅ **Complete Integration**: All ML layers integrated
✅ **Production-Ready**: Error handling, logging, monitoring
✅ **Modular Design**: Each component is independent
✅ **Mechanic Integration**: Unified interface for dynamic selection
✅ **MLOps**: Drift detection, automated retraining, distributed training
✅ **AutoML**: Automated model selection and hyperparameter optimization

**The Complete ML Integration is ready for production!** 🚀

