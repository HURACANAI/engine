"""
[FUTURE/MECHANIC - NOT USED IN ENGINE]

Incremental Training System for Mechanic

This module is for Mechanic (Cloud Updater Box) hourly incremental updates.
The Engine does NOT use this - it does full daily retraining instead.

DO NOT USE in Engine daily training pipeline.
This will be used when building Mechanic component.

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
# FUTURE/MECHANIC - CacheManager may not exist yet, will be used when building Mechanic
try:
    from .cache_manager import CacheManager
    __all__ = [
        'IncrementalLabeler',
        'LabelCache',
        'DeltaDetector',
        'CacheManager'
    ]
except ImportError:
    # CacheManager not implemented yet (future Mechanic component)
    __all__ = [
        'IncrementalLabeler',
        'LabelCache',
        'DeltaDetector',
    ]
