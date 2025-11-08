# ML Framework Guide

## Overview

This guide explains the PyTorch-based ML framework implemented in the Huracan Engine. The framework includes model factory, auto-training, and compiled inference layers for fast model training and deployment.

## PyTorch Model Factory

### Key Features

1. **Multiple Architectures**
   - Feed-forward networks with dropout and batch normalization
   - LSTM networks for sequential data
   - Hybrid architectures (CNN+LSTM)

2. **Regularization**
   - Dropout layers for preventing overfitting
   - Batch normalization for stable training
   - Weight initialization (Xavier, Orthogonal)

3. **Architecture Testing**
   - Performance comparison across architectures
   - Hyperparameter tuning support
   - Model versioning

4. **ONNX Export**
   - Export models to ONNX format
   - Support for ONNX Runtime inference
   - Dynamic batch size support

### Usage

```python
from cloud.training.ml_framework import (
    PyTorchModelFactory,
    ModelConfig,
    ArchitectureType
)

# Create model factory
factory = PyTorchModelFactory(device="cuda")

# Configure model
config = ModelConfig(
    architecture_type=ArchitectureType.FEED_FORWARD,
    input_size=100,
    hidden_sizes=[64, 32, 16],
    output_size=1,
    dropout_rate=0.2,
    use_batch_norm=True,
    activation="relu"
)

# Create model
model = factory.create_model(config)

# Test architecture
train_data = (X_train, y_train)
val_data = (X_val, y_val)

performance = factory.test_architecture(
    config=config,
    train_data=train_data,
    val_data=val_data,
    epochs=10,
    batch_size=32,
    learning_rate=0.001
)

# Export to ONNX
factory.export_to_onnx(
    model=model,
    input_shape=(1, 100),  # (batch_size, input_size)
    output_path="model.onnx"
)
```

### Architecture Types

#### Feed-Forward Network
- Standard fully connected layers
- Dropout and batch normalization
- Configurable activation functions
- Best for: Tabular data, feature-based predictions

#### LSTM Network
- Long Short-Term Memory layers
- Bidirectional support
- Feed-forward layers after LSTM
- Best for: Sequential data, time series

#### Hybrid Network
- CNN layers for feature extraction
- LSTM layers for sequence modeling
- Feed-forward layers for output
- Best for: Complex sequential patterns

### Configuration Options

- **architecture_type**: Architecture type (FEED_FORWARD, LSTM, HYBRID, CNN_LSTM)
- **input_size**: Input feature size
- **hidden_sizes**: List of hidden layer sizes
- **output_size**: Output size (default: 1)
- **dropout_rate**: Dropout rate (default: 0.2)
- **use_batch_norm**: Use batch normalization (default: True)
- **activation**: Activation function ("relu", "tanh", "sigmoid", "gelu")
- **lstm_hidden_size**: LSTM hidden size (for LSTM/Hybrid)
- **lstm_num_layers**: Number of LSTM layers (default: 2)
- **lstm_bidirectional**: Use bidirectional LSTM (default: False)
- **cnn_channels**: CNN channel sizes (for Hybrid)
- **cnn_kernel_sizes**: CNN kernel sizes (for Hybrid)

## AutoTrainer

### Key Features

1. **Hyperparameter Optimization**
   - Grid search
   - Random search
   - Bayesian optimization (future)

2. **Hyperparameters**
   - Learning rate
   - Batch size
   - Optimizer (Adam, AdamW, SGD, RMSprop)
   - Dropout rate
   - Weight decay

3. **Training Features**
   - Early stopping
   - Cross-validation support
   - Per-coin optimization
   - Multiple metrics (Sharpe, Sortino, accuracy, loss)

4. **Automatic Selection**
   - Best model selection
   - Performance tracking
   - Training time monitoring

### Usage

```python
from cloud.training.ml_framework import (
    AutoTrainer,
    AutoTrainerConfig,
    HyperparameterSpace,
    OptimizerType
)

# Create configuration
config = AutoTrainerConfig(
    search_method="grid",  # "grid", "random", "bayesian"
    max_trials=50,
    metric="sharpe",  # "sharpe", "sortino", "accuracy", "loss"
    maximize_metric=True,
    early_stopping_patience=5,
    min_epochs=10,
    max_epochs=100
)

# Create hyperparameter space
hyperparameter_space = HyperparameterSpace(
    learning_rates=[1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
    batch_sizes=[16, 32, 64, 128, 256],
    optimizers=[OptimizerType.ADAM, OptimizerType.ADAMW, OptimizerType.SGD],
    dropout_rates=[0.0, 0.1, 0.2, 0.3],
    weight_decay=[0.0, 1e-5, 1e-4]
)

config.hyperparameter_space = hyperparameter_space

# Create trainer
trainer = AutoTrainer(config=config)

# Define model factory function
def create_model(dropout_rate=0.2, **kwargs):
    from cloud.training.ml_framework import ModelConfig, ArchitectureType
    config = ModelConfig(
        architecture_type=ArchitectureType.FEED_FORWARD,
        input_size=100,
        hidden_sizes=[64, 32, 16],
        output_size=1,
        dropout_rate=dropout_rate
    )
    factory = PyTorchModelFactory()
    return factory.create_model(config)

# Optimize
best_result = trainer.optimize(
    model_factory=create_model,
    train_data=(X_train, y_train),
    val_data=(X_val, y_val),
    symbol="BTC-USD"
)

print(f"Best learning rate: {best_result.learning_rate}")
print(f"Best batch size: {best_result.batch_size}")
print(f"Best optimizer: {best_result.optimizer}")
print(f"Best validation metric: {best_result.val_metric}")
```

### Search Methods

#### Grid Search
- Exhaustive search over all combinations
- Guaranteed to find best in search space
- Can be computationally expensive

#### Random Search
- Random sampling of hyperparameter space
- More efficient for large spaces
- Good for exploration

#### Bayesian Optimization (Future)
- Adaptive search based on previous results
- Most efficient for expensive evaluations
- Requires additional dependencies

### Metrics

- **sharpe**: Sharpe ratio (risk-adjusted returns)
- **sortino**: Sortino ratio (downside risk-adjusted returns)
- **accuracy**: Classification accuracy
- **loss**: Validation loss (minimized)

## Compiled Inference Layer

### Key Features

1. **Multiple Backends**
   - ONNX Runtime (recommended)
   - TorchScript JIT
   - PyTorch (fallback)

2. **Performance Optimizations**
   - Model pre-loading
   - Warm-up iterations
   - Batch inference
   - Hardware-aware scheduling

3. **Latency Monitoring**
   - Per-inference latency tracking
   - Statistical analysis (mean, median, p95, p99)
   - Performance metrics

4. **Device Support**
   - CPU inference
   - CUDA/GPU inference (when available)
   - Automatic device selection

### Usage

```python
from cloud.training.ml_framework import (
    CompiledInferenceLayer,
    InferenceBackend
)

# Create inference layer
inference = CompiledInferenceLayer(
    backend=InferenceBackend.ONNX,
    device="cuda"  # or "cpu"
)

# Load model
inference.load_model("model.onnx")

# Warm up
inference.warm_up(num_iterations=10, batch_size=1)

# Single prediction
features = np.random.randn(1, 100).astype(np.float32)
result = inference.predict(features)

print(f"Prediction: {result.prediction}")
print(f"Latency: {result.latency_ms:.2f} ms")
print(f"Backend: {result.backend.value}")

# Batch prediction
features_list = [np.random.randn(1, 100).astype(np.float32) for _ in range(10)]
results = inference.batch_predict(features_list, batch_size=4)

# Get latency statistics
stats = inference.get_latency_stats()
print(f"Mean latency: {stats['mean_ms']:.2f} ms")
print(f"P95 latency: {stats['p95_ms']:.2f} ms")
```

### Backends

#### ONNX Runtime (Recommended)
- Fast inference
- Cross-platform support
- Optimized execution providers
- Best for: Production deployment

#### TorchScript
- JIT compilation
- PyTorch-native
- Good for: PyTorch-specific optimizations

#### PyTorch (Fallback)
- Standard PyTorch inference
- Less optimized
- Good for: Development and testing

### Performance Tips

1. **Warm-up**: Always warm up the model before production use
2. **Batch Inference**: Use batch inference for multiple predictions
3. **Device Selection**: Use GPU for large models or high throughput
4. **Model Optimization**: Export to ONNX for best performance
5. **Latency Monitoring**: Monitor latency to detect performance degradation

## Integration

### Complete Workflow

```python
# 1. Create model factory
factory = PyTorchModelFactory(device="cuda")

# 2. Create model configuration
config = ModelConfig(
    architecture_type=ArchitectureType.FEED_FORWARD,
    input_size=100,
    hidden_sizes=[64, 32, 16],
    output_size=1,
    dropout_rate=0.2,
    use_batch_norm=True
)

# 3. Create model
model = factory.create_model(config)

# 4. Auto-train with hyperparameter optimization
trainer = AutoTrainer(config=AutoTrainerConfig())
best_result = trainer.optimize(
    model_factory=lambda **kwargs: factory.create_model(config),
    train_data=(X_train, y_train),
    val_data=(X_val, y_val)
)

# 5. Export to ONNX
factory.export_to_onnx(
    model=model,
    input_shape=(1, 100),
    output_path="model.onnx"
)

# 6. Deploy with compiled inference
inference = CompiledInferenceLayer(backend=InferenceBackend.ONNX)
inference.load_model("model.onnx")
inference.warm_up()

# 7. Run inference
result = inference.predict(features)
```

## Best Practices

1. **Model Architecture**
   - Start with feed-forward for tabular data
   - Use LSTM for sequential data
   - Use hybrid for complex patterns
   - Test multiple architectures

2. **Hyperparameter Tuning**
   - Use grid search for small spaces
   - Use random search for large spaces
   - Use early stopping to prevent overfitting
   - Track multiple metrics

3. **Inference Optimization**
   - Always warm up models
   - Use ONNX Runtime for production
   - Batch predictions when possible
   - Monitor latency

4. **Model Deployment**
   - Export to ONNX for cross-platform support
   - Test on target hardware
   - Monitor performance in production
   - Version models for rollback

## Future Enhancements

1. **Bayesian Optimization**: Add Bayesian optimization for hyperparameter tuning
2. **Model Ensembles**: Support for ensemble models
3. **Quantization**: Model quantization for faster inference
4. **Distributed Training**: Support for distributed training
5. **AutoML**: Automated architecture search
