"""
Optimization modules for performance improvements.
"""

from .computation_cache import ComputationCache, cached, get_cache
from .efficiency_cache import (
    cache_features,
    cache_predictions,
    cache_regime,
    cache_gate_decision,
    clear_all_caches,
    get_cache_stats,
)
from .parallel_processor import ParallelSignalProcessor
from .query_optimizer import DatabaseQueryOptimizer
from .hyperparameter_tuner import AdvancedHyperparameterTuner, HyperparameterTuningResult
from .feature_selector import AutomatedFeatureSelector, FeatureSelectionResult
from .model_calibration import ModelCalibrator, CalibrationResult
from .advanced_scaling import AdvancedFeatureScaler
from .adaptive_lr_scheduler import AdaptiveLRScheduler, LRState

__all__ = [
    'ComputationCache',
    'cached',
    'get_cache',
    'cache_features',
    'cache_predictions',
    'cache_regime',
    'cache_gate_decision',
    'clear_all_caches',
    'get_cache_stats',
    'ParallelSignalProcessor',
    'DatabaseQueryOptimizer',
    'AdvancedHyperparameterTuner',
    'HyperparameterTuningResult',
    'AutomatedFeatureSelector',
    'FeatureSelectionResult',
    'ModelCalibrator',
    'CalibrationResult',
    'AdvancedFeatureScaler',
    'AdaptiveLRScheduler',
    'LRState',
]
