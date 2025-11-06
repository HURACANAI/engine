"""
Multi-Window Training for Huracan V2

Different components need different historical windows:
- Scalp core (1m): 60 days - microstructure changes fast
- Confirm filter (5m): 120 days - need more regime variety
- Regime classifier (1m): 365 days - need full market cycles
- Risk context (1d): 730 days - long-term correlations

This module provides component-specific training with appropriate:
1. Historical windows
2. Recency weighting
3. Walk-forward validation
4. Data preparation
"""

from .component_configs import (
    ComponentConfig,
    ScalpCoreConfig,
    ConfirmFilterConfig,
    RegimeClassifierConfig,
    RiskContextConfig,
    create_all_component_configs
)
from .window_manager import TrainingWindowManager
from .multi_window_trainer import MultiWindowTrainer

__all__ = [
    'ComponentConfig',
    'ScalpCoreConfig',
    'ConfirmFilterConfig',
    'RegimeClassifierConfig',
    'RiskContextConfig',
    'create_all_component_configs',
    'TrainingWindowManager',
    'MultiWindowTrainer'
]
