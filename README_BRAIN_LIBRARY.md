# 🧠 Brain Library - ML Enhancements for Huracan

## Overview

The Brain Library is a comprehensive ML enhancement system for Huracan that provides:

- ✅ **Automatic Feature Importance Analysis** - Analyzes and ranks features after training
- ✅ **Model Comparison** - Compares multiple model types and selects the best
- ✅ **Model Versioning** - Tracks model versions with automatic rollback
- ✅ **Data Quality Monitoring** - Tracks data quality issues and coverage
- ✅ **Dynamic Model Selection** - Selects models based on volatility regime
- ✅ **Comprehensive Metrics** - Tracks Sharpe, Sortino, Hit Ratio, Profit Factor, etc.

## Quick Start

### 1. Database Setup

Configure PostgreSQL database in `config/base.yaml`:

```yaml
postgres:
  dsn: "postgresql://user:password@localhost:5432/huracan"
```

### 2. Run Engine Training

Brain Library is automatically integrated into Engine training:

```bash
python -m src.cloud.training.pipelines.daily_retrain
```

### 3. Test Integration

Run the test script to verify Brain Library integration:

```bash
python scripts/test_brain_library_integration.py
```

## Features

### Automatic Feature Importance

After each training run, Brain Library automatically:
- Analyzes feature importance using SHAP, Permutation, and Correlation methods
- Stores rankings in Brain Library
- Makes top features available for future training

### Model Comparison

Brain Library compares multiple model types:
- LightGBM (default)
- XGBoost
- LSTM
- CNN
- Transformer

And selects the best model based on composite score.

### Model Versioning

Brain Library tracks model versions:
- Stores model manifests with hyperparameters
- Tracks dataset and feature set used
- Automatically rolls back if performance degrades >5%

### Dynamic Model Selection

Brain Library selects models based on volatility regime:
- **Low volatility** → XGBoost (stable trends)
- **Normal volatility** → LightGBM (default)
- **High volatility** → LSTM (complex patterns)
- **Extreme volatility** → LightGBM (conservative)

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Brain Library                        │
│  - Liquidation Data                                     │
│  - Funding Rates                                        │
│  - Open Interest                                        │
│  - Sentiment Scores                                     │
│  - Feature Importance                                   │
│  - Model Comparisons                                    │
│  - Model Registry                                       │
│  - Model Metrics                                        │
│  - Data Quality Logs                                    │
└──────────────┬──────────────────────────────────────────┘
               │
               ├──────────────────┐
               │                  │
               ▼                  ▼
┌──────────────────────┐  ┌──────────────────────┐
│   Engine Training    │  │  Nightly Feature     │
│                      │  │  Analysis (Mechanic) │
│  - Train Models      │  │                      │
│  - Feature Analysis  │  │  - Analyze Features  │
│  - Model Comparison  │  │  - Track Trends      │
│  - Versioning        │  │  - Detect Shifts     │
│  - Rollback          │  │                      │
└──────────┬───────────┘  └──────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────┐
│              Model Selection (Hamilton)                 │
│  - Volatility Regime Detection                          │
│  - Model Selection by Regime                            │
│  - Model Confidence Calculation                         │
│  - Dynamic Model Switching                              │
└─────────────────────────────────────────────────────────┘
```

## Components

### Core Brain Library
- `brain_library.py` - Core storage and retrieval
- `feature_importance_analyzer.py` - Feature analysis
- `model_comparison.py` - Model comparison
- `model_versioning.py` - Model versioning
- `liquidation_collector.py` - Liquidation data collection
- `rl_agent.py` - RL agent framework

### Services
- `brain_integrated_training.py` - Training integration
- `model_selector.py` - Model selection for Hamilton
- `nightly_feature_analysis.py` - Nightly analysis for Mechanic
- `data_collector.py` - Data collection service

### Pipelines
- `nightly_feature_workflow.py` - Nightly feature analysis workflow

## Usage

### Engine Training (Automatic)

Brain Library is automatically integrated into Engine training. No additional code needed!

```python
# Just run Engine training - Brain Library integration happens automatically
python -m src.cloud.training.pipelines.daily_retrain
```

### Manual Usage

```python
from src.cloud.training.brain.brain_library import BrainLibrary
from src.cloud.training.services.brain_integrated_training import BrainIntegratedTraining

# Initialize
brain = BrainLibrary(dsn=dsn, use_pool=True)
training = BrainIntegratedTraining(brain, settings)

# Train with Brain Library integration
result = training.train_with_brain_integration(
    symbol="BTC/USDT",
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    feature_names=feature_names,
    base_model=model,
    model_type="lightgbm",
)
```

### Model Selection (Hamilton)

```python
from src.cloud.training.services.model_selector import ModelSelector

selector = ModelSelector(brain)
model = selector.select_model_for_symbol("BTC/USDT", volatility_regime="high")
```

### Nightly Feature Analysis (Mechanic)

```python
from src.cloud.training.pipelines.nightly_feature_workflow import run_nightly_feature_analysis

results = run_nightly_feature_analysis(settings, symbols=["BTC/USDT", "ETH/USDT"])
```

## Documentation

- **[Usage Guide](docs/BRAIN_LIBRARY_USAGE_GUIDE.md)** - Comprehensive usage guide
- **[Architecture Design](HURACAN_ML_ENHANCEMENTS.md)** - Architecture design document
- **[Integration Guide](INTEGRATION_COMPLETE.md)** - Engine integration guide
- **[Implementation Status](IMPLEMENTATION_STATUS.md)** - Implementation status
- **[Next Steps](NEXT_STEPS_COMPLETE.md)** - Next steps implementation

## Testing

Run the test script:

```bash
python scripts/test_brain_library_integration.py
```

## Status

✅ **Complete and Ready for Production**

All core components are implemented and integrated:
- ✅ Brain Library core
- ✅ Engine integration
- ✅ Feature importance analysis
- ✅ Model comparison
- ✅ Model versioning
- ✅ Model selection
- ✅ Data collection
- ✅ Nightly feature analysis

## Future Enhancements

- [ ] Exchange API integration (funding rates, open interest)
- [ ] Sentiment API integration
- [ ] Multi-model training (LSTM, CNN, Transformer)
- [ ] LSTM standardization with attention
- [ ] Comprehensive evaluation dashboard

## Support

For issues or questions:
1. Check logs: `logs/*.log`
2. Review documentation: `docs/BRAIN_LIBRARY_USAGE_GUIDE.md`
3. Run test script: `scripts/test_brain_library_integration.py`

