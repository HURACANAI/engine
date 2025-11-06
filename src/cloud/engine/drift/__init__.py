"""
Drift Detection System

Monitors data distribution changes that indicate:
1. Market regime shifts (volatility, trend changes)
2. Data quality degradation (gaps, outliers increasing)
3. Label distribution shifts (profitable % changing)
4. Cost structure changes (spreads widening, fees changing)

When drift detected, Mechanic should:
- Alert operators
- Trigger full retrain
- Adjust risk limits
- Pause trading (if severe)

This prevents models from trading on stale patterns.
"""

from .distribution_monitor import DistributionMonitor, DriftReport
from .alert_manager import AlertManager, Alert, AlertLevel
from .drift_detector import DriftDetector, DriftMetrics

__all__ = [
    'DistributionMonitor',
    'DriftReport',
    'AlertManager',
    'Alert',
    'AlertLevel',
    'DriftDetector',
    'DriftMetrics'
]
