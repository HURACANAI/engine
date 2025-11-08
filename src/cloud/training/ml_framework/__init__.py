"""
Modular ML Framework for Huracan Engine

A comprehensive, production-ready ML framework with:
- Pre-processing layer (normalization, PCA, feature engineering)
- Baseline models (Linear/Logistic Regression, KNN, SVM)
- Core learners (Decision Trees, Random Forest, XGBoost, LSTM, GRU)
- Meta-layer (ensemble blending with dynamic weighting)
- Feedback loop (performance tracking and auto-tuning)
"""

from .base import BaseModel, ModelConfig, ModelMetrics
from .preprocessing import PreprocessingPipeline, PreprocessingConfig
from .baseline import (
    KNNModel,
    LinearRegressionModel,
    LogisticRegressionModel,
    SVMModel,
)
from .core import (
    DecisionTreeModel,
    RandomForestModel,
    XGBoostModel,
)
from .clustering import KMeansClustering
from .neural import GRUModel, LSTMModel
from .meta import EnsembleBlender, EnsembleConfig
from .feedback import FeedbackConfig, ModelFeedback
from .feature_selection import FeatureSelector
from .validation import (
    BiasVarianceDiagnostics,
    CrossValidator,
    create_train_val_test_split,
)
from .scheduler import LearningRateScheduler, create_scheduler
from .visualizer import ModelVisualizer
from .orchestrator import MLEngineOrchestrator
from .preprocessing import EnhancedPreprocessor
from .baselines import ABTestResult, ABTestingFramework
from .reinforcement import DQNAgent, RLAgent, ReplayBuffer
from .automl import AutoMLEngine
from .mlops import DriftDetector
from .distributed import DistributedTrainer
from .integration import MathematicalPipeline, UnifiedMLPipeline
from .mathematics import (
    ContinuousLearningCycle,
    DataUnderstanding,
    HuracanCore,
    MathematicalReasoning,
    MathematicalReasoningEngine,
    MathematicalValidator,
    PredictionWithReasoning,
    UncertaintyQuantifier,
)
from .model_registry import ModelMetadata, ModelRegistry, get_registry

__all__ = [
    # Base
    "BaseModel",
    "ModelConfig",
    "ModelMetrics",
    # Preprocessing
    "PreprocessingPipeline",
    "PreprocessingConfig",
    "EnhancedPreprocessor",
    # Baseline Models
    "LinearRegressionModel",
    "LogisticRegressionModel",
    "KNNModel",
    "SVMModel",
    # Core Learners
    "DecisionTreeModel",
    "RandomForestModel",
    "XGBoostModel",
    "LSTMModel",  # Optional (requires PyTorch)
    "GRUModel",  # Optional (requires PyTorch)
    "KMeansClustering",
    # Reinforcement Learning
    "DQNAgent",
    "RLAgent",
    "ReplayBuffer",
    # Ensemble
    "EnsembleBlender",
    "EnsembleConfig",
    # Feedback
    "ModelFeedback",
    "FeedbackConfig",
    # Feature Selection
    "FeatureSelector",
    # Validation
    "CrossValidator",
    "BiasVarianceDiagnostics",
    "create_train_val_test_split",
    # Learning Rate Scheduling
    "LearningRateScheduler",
    "create_scheduler",
    # Visualization
    "ModelVisualizer",
    # A/B Testing
    "ABTestResult",
    "ABTestingFramework",
    # AutoML
    "AutoMLEngine",
    # MLOps
    "DriftDetector",
    # Distributed Training
    "DistributedTrainer",
    # Orchestration
    "MLEngineOrchestrator",
    # Integration
    "UnifiedMLPipeline",
    "MathematicalPipeline",
    # Model Registry
    "ModelMetadata",
    "ModelRegistry",
    "get_registry",
    # Mathematical Reasoning
    "HuracanCore",
    "MathematicalReasoningEngine",
    "MathematicalReasoning",
    "PredictionWithReasoning",
    "DataUnderstanding",
    "UncertaintyQuantifier",
    "MathematicalValidator",
    "ContinuousLearningCycle",
]

