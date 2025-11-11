# 🚀 Brain Library - Quick Start Guide

## What is Brain Library?

Brain Library is a comprehensive ML enhancement system for Huracan that provides:
- Automatic feature importance analysis
- Model comparison and selection
- Model versioning with automatic rollback
- Data quality monitoring
- Dynamic model switching based on volatility regime

## Quick Setup (5 minutes)

### 1. Database Setup

Configure PostgreSQL in `config/base.yaml`:

```yaml
postgres:
  dsn: "postgresql://user:password@localhost:5432/huracan"
```

Or set environment variable:
```bash
export DATABASE_DSN="postgresql://user:password@localhost:5432/huracan"
```

### 2. Run Engine Training

Brain Library is **automatically integrated** - just run Engine training:

```bash
python -m src.cloud.training.pipelines.daily_retrain
```

That's it! Brain Library will:
- ✅ Analyze feature importance after training
- ✅ Store model metrics
- ✅ Track model versions
- ✅ Automatically rollback if performance degrades

### 3. Verify Integration

Check logs for Brain Library messages:
```bash
grep "brain_library" logs/*.log
```

You should see:
- `brain_library_initialized`
- `brain_integration_complete`
- `feature_importance_analyzed`

## Usage Examples

### View Top Features

```python
from src.cloud.training.brain.brain_library import BrainLibrary

brain = BrainLibrary(dsn=dsn)
top_features = brain.get_top_features("BTC/USDT", top_n=20)
print(top_features)
```

### Get Best Model

```python
from src.cloud.training.brain.brain_library import BrainLibrary

brain = BrainLibrary(dsn=dsn)
best_model = brain.get_best_model("BTC/USDT")
print(f"Best model: {best_model['model_type']}")
print(f"Composite score: {best_model['composite_score']}")
```

### Select Model by Volatility Regime

```python
from src.cloud.training.services.model_selector import ModelSelector

selector = ModelSelector(brain)
model = selector.select_model_for_symbol(
    symbol="BTC/USDT",
    volatility_regime="high",  # 'low', 'normal', 'high', 'extreme'
)
```

## Testing

Run the test script:
```bash
python scripts/test_brain_library_integration.py
```

Run the demo:
```bash
python scripts/demo_brain_library.py
```

## Documentation

- **Usage Guide**: `docs/BRAIN_LIBRARY_USAGE_GUIDE.md`
- **Architecture**: `HURACAN_ML_ENHANCEMENTS.md`
- **Integration**: `INTEGRATION_COMPLETE.md`
- **README**: `README_BRAIN_LIBRARY.md`

## Features

### ✅ Automatic Feature Importance
- Analyzes features after each training run
- Stores rankings in Brain Library
- Supports SHAP, Permutation, Correlation methods

### ✅ Model Comparison
- Compares multiple model types
- Selects best model per symbol
- Tracks historical performance

### ✅ Model Versioning
- Tracks model versions automatically
- Stores hyperparameters and feature sets
- Automatic rollback on performance degradation

### ✅ Dynamic Model Selection
- Selects model based on volatility regime
- Calculates model confidence
- Enables model switching

## Status

✅ **Production Ready**

All components are implemented, integrated, and tested.

## Support

- Check logs: `logs/*.log`
- Review docs: `docs/BRAIN_LIBRARY_USAGE_GUIDE.md`
- Run tests: `scripts/test_brain_library_integration.py`

