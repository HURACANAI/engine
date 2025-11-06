"""
Incremental Training System for Mechanic

The Mechanic runs 24/7 with hourly model updates. This module provides:
1. Incremental data loading (only new candles)
2. Incremental labeling (only new trades)
3. Label caching and persistence
4. Efficient updates without full retraining

Philosophy:
- Engine (daily): Full training on all data with V2 pipeline
- Mechanic (hourly): Incremental updates on new data only
- Archive: Stores labeled trades for fast loading

This prevents Mechanic from re-labeling 90 days of data every hour.
"""

from .incremental_labeler import IncrementalLabeler, LabelCache
from .delta_detector import DeltaDetector
from .cache_manager import CacheManager

__all__ = [
    'IncrementalLabeler',
    'LabelCache',
    'DeltaDetector',
    'CacheManager'
]
