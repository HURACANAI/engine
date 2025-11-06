"""
Efficiency Caching System

Caches expensive computations to improve performance:
- Feature engineering (60s TTL)
- Model predictions (30s TTL)
- Regime detection (5min TTL)
- Gate decisions (10s TTL)

Same results, faster execution.
"""

from typing import Dict, Any, Callable, Optional
from datetime import datetime, timedelta
from functools import wraps
import hashlib
import json

from src.cloud.training.optimization.computation_cache import ComputationCache, cached

# Global caches with appropriate TTLs
_feature_cache = ComputationCache(max_size=1000, default_ttl=60)  # 60 seconds
_prediction_cache = ComputationCache(max_size=500, default_ttl=30)  # 30 seconds
_regime_cache = ComputationCache(max_size=100, default_ttl=300)  # 5 minutes
_gate_cache = ComputationCache(max_size=500, default_ttl=10)  # 10 seconds


def cache_features(ttl_seconds: int = 60):
    """
    Cache feature engineering results.
    
    Usage:
        @cache_features(ttl_seconds=60)
        def compute_features(symbol: str, data: pl.DataFrame) -> Dict:
            return features
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from function name and arguments
            key_parts = [func.__name__]
            
            # Add args (skip first if it's self)
            for arg in args[1:] if args and hasattr(args[0], '__class__') else args:
                if isinstance(arg, (str, int, float, bool)):
                    key_parts.append(str(arg))
                elif hasattr(arg, '__hash__'):
                    try:
                        key_parts.append(str(hash(arg)))
                    except TypeError:
                        key_parts.append(str(id(arg)))
            
            # Add kwargs
            for k, v in sorted(kwargs.items()):
                if isinstance(v, (str, int, float, bool)):
                    key_parts.append(f"{k}:{v}")
                elif hasattr(v, '__hash__'):
                    try:
                        key_parts.append(f"{k}:{hash(v)}")
                    except TypeError:
                        key_parts.append(f"{k}:{id(v)}")
            
            cache_key = "_".join(key_parts)
            
            return _feature_cache.get_or_compute(
                key=cache_key,
                compute_fn=lambda: func(*args, **kwargs),
                ttl_seconds=ttl_seconds,
            )
        
        return wrapper
    return decorator


def cache_predictions(ttl_seconds: int = 30):
    """
    Cache model predictions.
    
    Usage:
        @cache_predictions(ttl_seconds=30)
        def predict(features: Dict) -> float:
            return model.predict(features)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from function name and features
            key_parts = [func.__name__]
            
            # Hash features dict
            features = args[0] if args else kwargs.get('features', {})
            if isinstance(features, dict):
                features_str = json.dumps(features, sort_keys=True)
                key_parts.append(hashlib.md5(features_str.encode()).hexdigest()[:16])
            else:
                key_parts.append(str(hash(features)) if hasattr(features, '__hash__') else str(id(features)))
            
            cache_key = "_".join(key_parts)
            
            return _prediction_cache.get_or_compute(
                key=cache_key,
                compute_fn=lambda: func(*args, **kwargs),
                ttl_seconds=ttl_seconds,
            )
        
        return wrapper
    return decorator


def cache_regime(ttl_seconds: int = 300):
    """
    Cache regime detection results.
    
    Usage:
        @cache_regime(ttl_seconds=300)
        def detect_regime(symbol: str, features: Dict) -> str:
            return regime
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from symbol and features hash
            symbol = args[0] if args else kwargs.get('symbol', 'unknown')
            features = args[1] if len(args) > 1 else kwargs.get('features', {})
            
            # Hash features
            if isinstance(features, dict):
                features_str = json.dumps(features, sort_keys=True)
                features_hash = hashlib.md5(features_str.encode()).hexdigest()[:16]
            else:
                features_hash = str(hash(features)) if hasattr(features, '__hash__') else str(id(features))
            
            cache_key = f"{func.__name__}_{symbol}_{features_hash}"
            
            return _regime_cache.get_or_compute(
                key=cache_key,
                compute_fn=lambda: func(*args, **kwargs),
                ttl_seconds=ttl_seconds,
            )
        
        return wrapper
    return decorator


def cache_gate_decision(ttl_seconds: int = 10):
    """
    Cache gate decision results.
    
    Usage:
        @cache_gate_decision(ttl_seconds=10)
        def evaluate_gates(signal: AlphaSignal, regime: str) -> GateDecision:
            return gate_decision
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from signal and regime
            signal = args[0] if args else kwargs.get('signal')
            regime = args[1] if len(args) > 1 else kwargs.get('regime', 'unknown')
            
            # Create key from signal attributes
            if hasattr(signal, 'confidence') and hasattr(signal, 'technique'):
                signal_key = f"{signal.technique}_{signal.confidence:.3f}"
            else:
                signal_key = str(hash(signal)) if hasattr(signal, '__hash__') else str(id(signal))
            
            cache_key = f"{func.__name__}_{signal_key}_{regime}"
            
            return _gate_cache.get_or_compute(
                key=cache_key,
                compute_fn=lambda: func(*args, **kwargs),
                ttl_seconds=ttl_seconds,
            )
        
        return wrapper
    return decorator


def clear_all_caches():
    """Clear all efficiency caches."""
    _feature_cache.clear()
    _prediction_cache.clear()
    _regime_cache.clear()
    _gate_cache.clear()


def get_cache_stats() -> Dict[str, Dict[str, Any]]:
    """Get statistics for all caches."""
    return {
        'feature_cache': {
            'size': len(_feature_cache.cache),
            'hits': _feature_cache.hits,
            'misses': _feature_cache.misses,
            'hit_rate': _feature_cache.hits / (_feature_cache.hits + _feature_cache.misses) if (_feature_cache.hits + _feature_cache.misses) > 0 else 0.0,
        },
        'prediction_cache': {
            'size': len(_prediction_cache.cache),
            'hits': _prediction_cache.hits,
            'misses': _prediction_cache.misses,
            'hit_rate': _prediction_cache.hits / (_prediction_cache.hits + _prediction_cache.misses) if (_prediction_cache.hits + _prediction_cache.misses) > 0 else 0.0,
        },
        'regime_cache': {
            'size': len(_regime_cache.cache),
            'hits': _regime_cache.hits,
            'misses': _regime_cache.misses,
            'hit_rate': _regime_cache.hits / (_regime_cache.hits + _regime_cache.misses) if (_regime_cache.hits + _regime_cache.misses) > 0 else 0.0,
        },
        'gate_cache': {
            'size': len(_gate_cache.cache),
            'hits': _gate_cache.hits,
            'misses': _gate_cache.misses,
            'hit_rate': _gate_cache.hits / (_gate_cache.hits + _gate_cache.misses) if (_gate_cache.hits + _gate_cache.misses) > 0 else 0.0,
        },
    }

