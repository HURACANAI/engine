"""
Attribution Module

Trade attribution with SHAP and permutation importance.
"""

from .trade_attribution import (
    TradeAttributionSystem,
    TradeAttribution,
    FeatureAttribution,
    AttributionMethod,
)

__all__ = [
    "TradeAttributionSystem",
    "TradeAttribution",
    "FeatureAttribution",
    "AttributionMethod",
]

