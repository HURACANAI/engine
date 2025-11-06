"""
Data Quality Pipeline for Huracan V2

Ensures clean, reliable historical data for training:
- Deduplication (remove duplicate trades)
- Timestamp fixing (enforce monotonic increasing)
- Outlier removal (filter bad prints, flash crashes)
- Gap handling (exchange outages)
- Historical fee tracking (date-locked fee schedules)
"""

from .fee_schedule import FeeSchedule, HistoricalFeeManager
from .gap_handler import GapHandler
from .sanity_pipeline import DataSanityPipeline

__all__ = [
    'DataSanityPipeline',
    'FeeSchedule',
    'HistoricalFeeManager',
    'GapHandler',
]
