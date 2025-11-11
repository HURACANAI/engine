"""
Engine Plugins

Plugin implementations for all 23 engines.
"""

from .trend_engine import TrendEngine
from .range_engine import RangeEngine
from .breakout_engine import BreakoutEngine

__all__ = [
    "TrendEngine",
    "RangeEngine",
    "BreakoutEngine",
]

