# Brain Library Usage Guide

## Overview

The Brain Library is a centralized storage system for all Huracan ML enhancements. It provides:

- Feature importance tracking
- Model comparison and selection
- Model versioning with rollback
- Data quality monitoring
- Liquidation data storage
- Funding rates and open interest storage
- Sentiment data storage

## Quick Start

### 1. Database Setup

First, ensure you have a PostgreSQL database configured:

```yaml
# config/base.yaml
postgres:
  dsn: "postgresql://user:password@localhost:5432/huracan"
```

Or set environment variable:
```bash
export DATABASE_DSN="postgresql://user:password@localhost:5432/huracan"
```

### 2. Initialize Brain Library

```python
from src.cloud.training.brain.brain_library import BrainLibrary

# Initialize with database connection
brain = BrainLibrary(dsn=dsn, use_pool=True)
```

### 3. Use in Engine Training

Brain Library is automatically integrated into Engine training. No additional code needed!

```python
# Engine training automatically uses Brain Library if database is available
python -m src.cloud.training.pipelines.daily_retrain
```

## Usage Examples

### Feature Importance Analysis

```python
from src.cloud.training.brain.feature_importance_analyzer import FeatureImportanceAnalyzer
from src.cloud.training.brain.brain_library import BrainLibrary

# Initialize
brain = BrainLibrary(dsn=dsn)
analyzer = FeatureImportanceAnalyzer(brain)

# Analyze features
results = analyzer.analyze_feature_importance(
    symbol="BTC/USDT",
    model=trained_model,
    X=X_train,
    y=y_train,
    feature_names=feature_names,
    methods=['shap', 'permutation'],
)

# Get top features
top_features = analyzer.get_top_features_for_symbol("BTC/USDT", top_n=20)
```

### Model Comparison

```python
from src.cloud.training.brain.model_comparison import ModelComparisonFramework
from src.cloud.training.brain.brain_library import BrainLibrary

# Initialize
brain = BrainLibrary(dsn=dsn)
comparison = ModelComparisonFramework(brain)

# Compare models
results = comparison.compare_models(
    symbol="BTC/USDT",
    models={
        "lightgbm": lightgbm_model,
        "xgboost": xgboost_model,
        "lstm": lstm_model,
    },
    X_test=X_test,
    y_test=y_test,
)

# Get best model
best_model = comparison.get_best_model_for_symbol("BTC/USDT")
```

### Model Versioning

```python
from src.cloud.training.brain.model_versioning import ModelVersioning
from src.cloud.training.brain.brain_library import BrainLibrary

# Initialize
brain = BrainLibrary(dsn=dsn)
versioning = ModelVersioning(brain)

# Register model version
versioning.register_model_version(
    model_id="BTC/USDT_lightgbm_20250108_120000",
    symbol="BTC/USDT",
    version=1,
    hyperparameters={"n_estimators": 100, "learning_rate": 0.05},
    dataset_id="dataset_BTC/USDT_20250108",
    feature_set=["feature_1", "feature_2", ...],
    training_metrics={"sharpe_ratio": 1.5},
    validation_metrics={"sharpe_ratio": 1.4},
)

# Check for rollback
rollback_occurred = versioning.check_and_rollback(
    model_id="BTC/USDT_lightgbm_20250108_120000",
    symbol="BTC/USDT",
    new_metrics={"sharpe_ratio": 1.3},
)
```

### Model Selection (Hamilton)

```python
from src.cloud.training.services.model_selector import ModelSelector
from src.cloud.training.brain.brain_library import BrainLibrary

# Initialize
brain = BrainLibrary(dsn=dsn)
selector = ModelSelector(brain)

# Select model based on volatility regime
model = selector.select_model_for_symbol(
    symbol="BTC/USDT",
    volatility_regime="high",  # 'low', 'normal', 'high', 'extreme'
)

# Get model confidence
confidence = selector.get_model_confidence("BTC/USDT")

# Check if should switch models
should_switch = selector.should_switch_model(
    symbol="BTC/USDT",
    current_model_id="model_123",
    volatility_regime="high",
)
```

### Nightly Feature Analysis (Mechanic)

```python
from src.cloud.training.services.nightly_feature_analysis import NightlyFeatureAnalysis
from src.cloud.training.brain.brain_library import BrainLibrary

# Initialize
brain = BrainLibrary(dsn=dsn)
feature_analysis = NightlyFeatureAnalysis(brain)

# Run nightly analysis
results = feature_analysis.run_nightly_analysis(
    symbols=["BTC/USDT", "ETH/USDT"],
    models={
        "BTC/USDT": btc_model,
        "ETH/USDT": eth_model,
    },
    datasets={
        "BTC/USDT": {"X": X_btc, "y": y_btc, "feature_names": features},
        "ETH/USDT": {"X": X_eth, "y": y_eth, "feature_names": features},
    },
)

# Get feature importance trends
trends = feature_analysis.get_feature_importance_trends("BTC/USDT", days=30)

# Identify feature shifts
shifts = feature_analysis.identify_feature_shifts("BTC/USDT", threshold=0.1)
```

### Data Collection

```python
from src.cloud.training.services.data_collector import DataCollector
from src.cloud.training.brain.brain_library import BrainLibrary

# Initialize
brain = BrainLibrary(dsn=dsn)
collector = DataCollector(brain, exchanges=['binance', 'bybit', 'okx'])

# Collect all data
results = collector.collect_all_data(
    symbols=["BTC/USDT", "ETH/USDT"],
    hours=24,
)

# Get liquidation features
features = collector.get_liquidation_features("BTC/USDT", hours=24)
```

### Brain-Integrated Training

```python
from src.cloud.training.services.brain_integrated_training import BrainIntegratedTraining
from src.cloud.training.brain.brain_library import BrainLibrary

# Initialize
brain = BrainLibrary(dsn=dsn)
settings = EngineSettings.load()
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

# Get top features
top_features = training.get_top_features("BTC/USDT", top_n=20)

# Get active model
active_model = training.get_active_model("BTC/USDT")
```

## Database Schema

Brain Library automatically creates the following tables:

- `liquidations` - Liquidation event data
- `funding_rates` - Funding rate data
- `open_interest` - Open interest data
- `sentiment_scores` - Sentiment data
- `feature_importance` - Feature importance rankings
- `model_comparisons` - Model comparison results
- `model_registry` - Active model registry
- `model_metrics` - Model evaluation metrics
- `data_quality_logs` - Data quality issue tracking
- `model_manifests` - Model versioning manifests
- `rollback_logs` - Rollback event logs

## Testing

Run the test script to verify Brain Library integration:

```bash
python scripts/test_brain_library_integration.py
```

## Troubleshooting

### Database Connection Issues

If you see database connection errors:

1. Check database DSN is configured:
   ```bash
   echo $DATABASE_DSN
   ```

2. Test database connection:
   ```bash
   psql $DATABASE_DSN -c "SELECT 1"
   ```

3. Check Brain Library initialization in logs:
   ```bash
   grep "brain_library_initialized" logs/*.log
   ```

### Feature Importance Analysis Fails

If feature importance analysis fails:

1. Check if SHAP is installed:
   ```bash
   pip install shap
   ```

2. Check if sklearn is installed:
   ```bash
   pip install scikit-learn
   ```

3. Check logs for specific error messages

### Model Comparison Fails

If model comparison fails:

1. Ensure models have `predict` method
2. Check test data is not empty
3. Verify metrics calculation

### Model Versioning Issues

If model versioning fails:

1. Check database connection
2. Verify model ID format
3. Check rollback threshold (default: 5%)

## Best Practices

1. **Always use connection pooling**: Set `use_pool=True` when initializing Brain Library
2. **Handle errors gracefully**: Brain Library operations are non-blocking
3. **Monitor data quality**: Check data quality logs regularly
4. **Track model versions**: Use model versioning for all production models
5. **Analyze features regularly**: Run nightly feature analysis after training

## API Reference

See individual module documentation:

- `src/cloud/training/brain/brain_library.py` - Core Brain Library
- `src/cloud/training/brain/feature_importance_analyzer.py` - Feature Analysis
- `src/cloud/training/brain/model_comparison.py` - Model Comparison
- `src/cloud/training/brain/model_versioning.py` - Model Versioning
- `src/cloud/training/services/brain_integrated_training.py` - Training Integration
- `src/cloud/training/services/model_selector.py` - Model Selection
- `src/cloud/training/services/nightly_feature_analysis.py` - Nightly Analysis
- `src/cloud/training/services/data_collector.py` - Data Collection

## Support

For issues or questions:

1. Check logs: `logs/*.log`
2. Review documentation: `docs/BRAIN_LIBRARY_USAGE_GUIDE.md`
3. Check test script: `scripts/test_brain_library_integration.py`

