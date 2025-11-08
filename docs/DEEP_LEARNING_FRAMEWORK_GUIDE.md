# Deep Learning Framework Guide

## Overview

The Deep Learning Framework extends the ML Framework with advanced neural network capabilities based on "Deep Learning in 100 Seconds" concepts.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│              Deep Learning Framework                    │
└─────────────────────────────────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
┌───────▼──────┐ ┌──────▼──────┐ ┌─────▼──────┐
│ Neural Core  │ │  Training   │ │  Analysis  │
│  (Layers)    │ │ (Backprop)  │ │(Explain)   │
└───────┬──────┘ └──────┬──────┘ └─────┬──────┘
        │               │               │
        └───────────────┼───────────────┘
                        │
                ┌───────▼───────┐
                │   Models      │
                │ (CNN/Transformer/GAN) │
                └───────────────┘
```

## Components

### 1. Neural Network Core

**Files**:
- `core/neural_base.py`: Base neural network components
- `core/activation_functions.py`: Activation functions
- `core/layer_manager.py`: Dynamic layer building

**Features**:
- `Neuron`: Individual neuron (specialized expert)
- `Layer`: Neural network layer
- `BaseNeuralNetwork`: Base architecture
- Dynamic layer building from config

**Usage**:
```python
from src.cloud.training.ml_framework.core import Layer, BaseNeuralNetwork

# Create layer
layer = Layer(
    in_features=128,
    out_features=64,
    activation="relu",
    dropout=0.2,
)

# Create network
network = BaseNeuralNetwork(
    input_size=128,
    output_size=1,
    hidden_layers=[
        {"out_features": 64, "activation": "relu"},
        {"out_features": 32, "activation": "relu"},
    ],
)
```

### 2. Training Components

**Files**:
- `training/backpropagation.py`: Backpropagation engine
- `training/optimizers.py`: Optimizers (SGD, Adam, RMSProp)
- `training/gpu_handler.py`: GPU management
- `training/trainer.py`: Main training orchestrator

**Features**:
- Backpropagation with gradient clipping
- Multiple optimizers
- GPU acceleration
- Distributed training support
- Checkpoint saving

**Usage**:
```python
from src.cloud.training.ml_framework.training import Trainer, TrainingConfig

config = TrainingConfig(
    epochs=50,
    batch_size=32,
    learning_rate=0.001,
    optimizer="adam",
    use_gpu=True,
)

trainer = Trainer(model, config)
history = trainer.train(train_loader, val_loader)
```

### 3. Specialized Models

**Files**:
- `models/cnn_model.py`: Convolutional Neural Network
- `models/transformer_model.py`: Transformer
- `models/gan_model.py`: Generative Adversarial Network
- `models/autoencoder_model.py`: Autoencoder

**CNN Model**:
```python
from src.cloud.training.ml_framework.models import CNNModel

config = ModelConfig(
    name="cnn",
    hyperparameters={
        "input_channels": 1,
        "conv_layers": [
            {"out_channels": 32, "kernel_size": 3},
            {"out_channels": 64, "kernel_size": 3},
        ],
    },
)

cnn = CNNModel(config)
```

**Transformer Model**:
```python
from src.cloud.training.ml_framework.models import TransformerModel

config = ModelConfig(
    name="transformer",
    hyperparameters={
        "d_model": 512,
        "nhead": 8,
        "num_layers": 6,
    },
)

transformer = TransformerModel(config)
```

**GAN Model**:
```python
from src.cloud.training.ml_framework.models import GANModel

config = ModelConfig(
    name="gan",
    hyperparameters={
        "latent_dim": 100,
        "data_dim": 784,
    },
)

gan = GANModel(config)
# Generate synthetic data
synthetic_data = gan.generate(num_samples=100, device=device)
```

### 4. Analysis Components

**Files**:
- `analysis/explainability.py`: Model interpretability
- `analysis/adversarial_test.py`: Adversarial testing

**Explainability**:
```python
from src.cloud.training.ml_framework.analysis import ModelExplainability

explainer = ModelExplainability(model)

# Get activations
activations = explainer.get_activations(x)

# Analyze layer activations
stats = explainer.analyze_layer_activations(x)

# Detect dead neurons
dead_neurons = explainer.detect_dead_neurons(x)

# Detect bias
bias_stats = explainer.detect_bias(x)
```

**Adversarial Testing**:
```python
from src.cloud.training.ml_framework.analysis import AdversarialTester

tester = AdversarialTester(model, device)

# Test robustness
robustness = tester.test_robustness(x, y, attack_type="fgsm", epsilon=0.1)

# Test noise robustness
noise_robustness = tester.test_noise_robustness(x, y, noise_levels=[0.01, 0.05, 0.1])
```

## Complete Pipeline

```python
from src.cloud.training.ml_framework.training import Trainer, TrainingConfig
from src.cloud.training.ml_framework.training import GPUHandler
from src.cloud.training.ml_framework.analysis import ModelExplainability, AdversarialTester

# Initialize GPU
gpu_handler = GPUHandler(device="cuda")

# Create model
model = YourModel()

# Training configuration
config = TrainingConfig(
    epochs=50,
    batch_size=32,
    learning_rate=0.001,
    optimizer="adam",
    use_gpu=True,
)

# Train
trainer = Trainer(model, config)
history = trainer.train(train_loader, val_loader)

# Explainability
explainer = ModelExplainability(model)
activations = explainer.get_activations(x_test)

# Adversarial testing
tester = AdversarialTester(model, gpu_handler.get_device())
robustness = tester.test_robustness(x_test, y_test)
```

## Integration with Mechanic

The Deep Learning Framework integrates with the Mechanic feedback loop:

```python
# After training
metrics = model.evaluate(X_test, y_test)

# Feed to Mechanic
mechanic.record_performance("deep_model", metrics)

# Adversarial testing
tester = AdversarialTester(model, device)
robustness = tester.test_robustness(X_test, y_test)

# If vulnerable, retrain with adversarial training
if robustness["is_vulnerable"]:
    # Generate adversarial examples
    x_adv = tester.generate_adversarial_dataset(X_train, y_train)
    
    # Retrain with adversarial examples
    trainer.train(adversarial_loader, val_loader)
```

## Summary

✅ **Neural Network Core**: Base architecture with dynamic layer building
✅ **Training Components**: Backpropagation, optimizers, GPU support
✅ **Specialized Models**: CNN, Transformer, GAN, Autoencoder
✅ **Analysis**: Explainability and adversarial testing
✅ **Integration**: Works with existing ML Framework and Mechanic

**Ready for deep learning!** 🚀

