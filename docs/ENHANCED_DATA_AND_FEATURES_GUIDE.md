# Enhanced Data Pipeline and Feature Engineering Guide

## Overview

This guide explains the enhanced data pipeline and dynamic feature engineering system implemented in the Huracan Engine. The system ensures clean, scaled, and ready-to-use data, with dynamic feature generation and Brain Library storage.

## Enhanced Data Pipeline

### Key Features

1. **Automated Data Cleaning**
   - Remove duplicates
   - Remove outliers using Z-score
   - Handle NaN values with forward/backward fill
   - Validate data integrity

2. **Missing Candle Handling**
   - Detect gaps in timestamp data
   - Interpolate missing values
   - Configurable threshold for gap detection
   - Multiple interpolation methods (linear, forward fill, backward fill)

3. **Timestamp Alignment**
   - Align timestamps to consistent intervals
   - Validate chronological order
   - Ensure no future data leakage

4. **Feature Scaling**
   - Standard scaling (Z-score normalization)
   - Min-max scaling
   - Robust scaling (median and IQR)
   - Configurable scaling per feature

5. **Data Validation**
   - Chronology validation
   - No future data validation
   - Required columns validation

### Usage

```python
from cloud.training.datasets.enhanced_data_pipeline import (
    EnhancedDataPipeline,
    DataPipelineConfig,
    ScalingMethod
)

# Create configuration
config = DataPipelineConfig(
    remove_duplicates=True,
    remove_outliers=True,
    outlier_threshold=3.0,
    handle_missing_candles=True,
    missing_candle_threshold_minutes=5,
    scaling_method=ScalingMethod.STANDARD,
    validate_chronology=True,
    validate_no_future_data=True
)

# Create pipeline
pipeline = EnhancedDataPipeline(config=config)

# Process data
cleaned_data = pipeline.process(raw_data)

# Get statistics
stats = pipeline.get_statistics(cleaned_data)

# Get scaling statistics
scaling_stats = pipeline.get_scaling_stats()

# Inverse scale (for visualization)
original_data = pipeline.inverse_scale(cleaned_data, features=["close", "volume"])
```

### Configuration Options

- **remove_duplicates**: Remove duplicate rows
- **remove_outliers**: Remove outliers using Z-score
- **outlier_threshold**: Z-score threshold for outliers (default: 3.0)
- **handle_missing_candles**: Handle missing candles by interpolation
- **missing_candle_threshold_minutes**: Max gap before interpolation (default: 5 minutes)
- **interpolation_method**: Interpolation method ("linear", "forward_fill", "backward_fill")
- **scaling_method**: Scaling method (STANDARD, MINMAX, ROBUST, NONE)
- **scale_features**: List of features to scale (None = all numeric features)
- **align_timestamps**: Align timestamps to consistent intervals
- **validate_chronology**: Validate data is sorted by timestamp
- **validate_no_future_data**: Validate no future data leakage

## Dynamic Feature Engineering

### Key Features

1. **Dynamic Feature Generation**
   - Generate features per symbol
   - Dependency tracking
   - Automatic dependency resolution
   - Feature versioning

2. **Brain Library Storage**
   - Store feature sets in Brain Library
   - Feature caching
   - Feature set versioning
   - Metadata tracking

3. **Integration with FeatureRecipe**
   - Uses existing FeatureRecipe for consistency
   - Adds custom features dynamically
   - Maintains feature parity across components

4. **Feature Sets**
   - Create feature sets for different strategies
   - Symbol-specific feature sets
   - Feature set validation

### Usage

```python
from cloud.training.features import DynamicFeatureEngine

# Create feature engine
engine = DynamicFeatureEngine(brain_library_path="brain_library/features")

# Register custom feature
engine.register_feature(
    name="custom_momentum",
    description="Custom momentum feature",
    dependencies=["close", "volume"],
    generator=lambda df: (pl.col("close").pct_change(10) / pl.col("volume").rolling_mean(10)).alias("custom_momentum"),
    version=1,
    tags=["momentum", "custom"],
    cached=True
)

# Create feature set
feature_set = engine.create_feature_set(
    name="momentum_strategy",
    description="Features for momentum strategy",
    features=["rsi_14", "ema_20", "custom_momentum"],
    version=1
)

# Generate features for symbol
features = engine.generate_features(
    data=raw_data,
    symbol="BTC-USD",
    feature_set="momentum_strategy"
)

# Get feature dependencies
dependencies = engine.get_feature_dependencies("custom_momentum")

# Validate feature set
validation = engine.validate_feature_set("momentum_strategy")
```

### Feature Registration

Features can be registered with:
- **name**: Feature name
- **description**: Feature description
- **dependencies**: Required input columns
- **generator**: Feature generator function (takes DataFrame, returns Expr)
- **version**: Feature version
- **tags**: Feature tags for categorization
- **cached**: Whether to cache this feature

### Feature Sets

Feature sets allow grouping features for different strategies:
- **name**: Feature set name
- **description**: Feature set description
- **features**: List of feature names
- **symbol**: Symbol (optional, for symbol-specific feature sets)
- **version**: Feature set version
- **metadata**: Additional metadata

## Trade Attribution

### Key Features

1. **SHAP Attribution**
   - SHAP values for feature attribution
   - Baseline comparison
   - Feature contribution analysis

2. **Permutation Importance**
   - Permutation-based importance
   - Impact on prediction
   - Feature ranking

3. **Error Classification**
   - Direction wrong
   - Timing late
   - Magnitude wrong
   - None (correct prediction)

4. **Batch Attribution**
   - Compute attributions for multiple trades
   - Feature importance summary
   - Attribution by error type

### Usage

```python
from cloud.training.attribution import (
    TradeAttributionSystem,
    AttributionMethod
)

# Create attribution system
attribution_system = TradeAttributionSystem(
    top_k_features=10,
    num_permutations=100
)

# Compute attribution for single trade
attribution = attribution_system.compute_attribution(
    model=model,
    features=features,
    prediction=prediction,
    actual_outcome=actual_outcome,
    method=AttributionMethod.PERMUTATION
)

# Compute batch attributions
attributions = attribution_system.compute_batch_attributions(
    model=model,
    trades=trades,
    predictions=predictions,
    method=AttributionMethod.SHAP
)

# Get feature importance summary
importance_summary = attribution_system.get_feature_importance_summary(attributions)

# Get attribution by error type
by_error_type = attribution_system.get_attribution_by_error_type(attributions)
```

### Attribution Methods

- **SHAP**: SHAP values for feature attribution
- **PERMUTATION**: Permutation-based importance
- **GRADIENT**: Gradient-based attribution (future)
- **INTEGRATED_GRADIENT**: Integrated gradient attribution (future)

## Integration

### Data Pipeline → Feature Engineering → Attribution

```python
# 1. Process raw data
pipeline = EnhancedDataPipeline(config=config)
cleaned_data = pipeline.process(raw_data)

# 2. Generate features
engine = DynamicFeatureEngine()
features = engine.generate_features(cleaned_data, symbol="BTC-USD")

# 3. Train model and make predictions
model = train_model(features)
predictions = model.predict(features)

# 4. Compute attributions
attribution_system = TradeAttributionSystem()
attributions = attribution_system.compute_batch_attributions(
    model=model,
    trades=trades,
    predictions=predictions
)
```

## Best Practices

1. **Data Cleaning**
   - Always validate chronology before processing
   - Handle missing candles appropriately
   - Remove outliers cautiously (may remove important signals)

2. **Feature Engineering**
   - Register features with proper dependencies
   - Use feature sets for organization
   - Validate feature sets before use
   - Cache frequently used features

3. **Attribution**
   - Use permutation importance for interpretability
   - Use SHAP for detailed feature contributions
   - Analyze attributions by error type
   - Track feature importance over time

4. **Versioning**
   - Version all features and feature sets
   - Track feature changes
   - Maintain backward compatibility

## Future Enhancements

1. **Advanced Attribution**
   - Integrated gradients
   - LIME (Local Interpretable Model-agnostic Explanations)
   - Feature interaction analysis

2. **Feature Engineering**
   - Automated feature generation
   - Feature selection
   - Feature importance-based selection

3. **Data Pipeline**
   - Real-time data processing
   - Streaming data support
   - Distributed processing

