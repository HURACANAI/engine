"""
ML Framework Module

PyTorch integration, model factory, auto-training, and compiled inference.
"""

from .model_factory_pytorch import (
    PyTorchModelFactory,
    ModelConfig,
    ArchitectureType,
    FeedForwardNetwork,
    LSTMNetwork,
    HybridNetwork,
)

from .training.auto_trainer import (
    AutoTrainer,
    AutoTrainerConfig,
    HyperparameterSpace,
    TrainingResult,
    OptimizerType,
)

from .inference.compiled_inference import (
    CompiledInferenceLayer,
    InferenceBackend,
    InferenceResult,
)

__all__ = [
    "PyTorchModelFactory",
    "ModelConfig",
    "ArchitectureType",
    "FeedForwardNetwork",
    "LSTMNetwork",
    "HybridNetwork",
    "AutoTrainer",
    "AutoTrainerConfig",
    "HyperparameterSpace",
    "TrainingResult",
    "OptimizerType",
    "CompiledInferenceLayer",
    "InferenceBackend",
    "InferenceResult",
]
