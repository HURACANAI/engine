"""
[FUTURE/MECHANIC - NOT USED IN ENGINE]

Delta Detector

This module is for Mechanic (Cloud Updater Box) hourly incremental updates.
The Engine does NOT use this - it does full daily retraining instead.

DO NOT USE in Engine daily training pipeline.
This will be used when building Mechanic component.

Detects what changed between training runs:
- New candles available
- Labels that need updating
- Config changes requiring full retrain

This helps Mechanic decide: incremental update vs full retrain.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional

import polars as pl
import structlog

from ..labeling import LabeledTrade

logger = structlog.get_logger(__name__)


@dataclass
class Delta:
    """Changes detected between runs."""

    has_new_candles: bool
    new_candles_count: int
    new_candle_start: Optional[datetime]
    new_candle_end: Optional[datetime]

    has_incomplete_labels: bool
    incomplete_labels_count: int

    config_changed: bool
    requires_full_retrain: bool

    time_since_last_run: Optional[timedelta] = None


class DeltaDetector:
    """
    Detects changes between training runs.

    Usage:
        detector = DeltaDetector()

        delta = detector.detect(
            new_candles=current_candles,
            last_candle_timestamp=cache.last_candle_timestamp,
            cached_labels=cache.labeled_trades,
            current_config_hash=labeler.config_hash(),
            cached_config_hash=cache.config_hash
        )

        if delta.requires_full_retrain:
            # Full retrain
        elif delta.has_new_candles:
            # Incremental update
        else:
            # No changes
    """

    def detect(
        self,
        new_candles: pl.DataFrame,
        last_candle_timestamp: Optional[datetime],
        cached_labels: Optional[List[LabeledTrade]],
        current_config_hash: str,
        cached_config_hash: Optional[str],
        last_run_time: Optional[datetime] = None
    ) -> Delta:
        """
        Detect changes.

        Args:
            new_candles: Current candle data
            last_candle_timestamp: Last processed candle timestamp
            cached_labels: Previously cached labels
            current_config_hash: Current labeler config hash
            cached_config_hash: Cached labeler config hash
            last_run_time: When last run occurred

        Returns:
            Delta object with detected changes
        """
        logger.debug("detecting_delta")

        # Check config change
        config_changed = False
        if cached_config_hash and current_config_hash != cached_config_hash:
            config_changed = True
            logger.info(
                "config_change_detected",
                old_hash=cached_config_hash,
                new_hash=current_config_hash
            )

        # Check new candles
        has_new_candles = False
        new_candles_count = 0
        new_candle_start = None
        new_candle_end = None

        if last_candle_timestamp:
            new_candles_filtered = new_candles.filter(
                pl.col('timestamp') > last_candle_timestamp
            )
            new_candles_count = len(new_candles_filtered)
            has_new_candles = new_candles_count > 0

            if has_new_candles:
                new_candle_start = new_candles_filtered['timestamp'].min()
                new_candle_end = new_candles_filtered['timestamp'].max()

                logger.info(
                    "new_candles_detected",
                    count=new_candles_count,
                    start=new_candle_start,
                    end=new_candle_end
                )
        else:
            # No previous run - all candles are new
            has_new_candles = True
            new_candles_count = len(new_candles)
            new_candle_start = new_candles['timestamp'].min()
            new_candle_end = new_candles['timestamp'].max()

            logger.info(
                "first_run_all_candles_new",
                count=new_candles_count
            )

        # Check incomplete labels
        has_incomplete_labels = False
        incomplete_labels_count = 0

        if cached_labels:
            # Labels that exited on timeout might now hit TP/SL
            timeout_labels = [
                t for t in cached_labels
                if t.exit_reason == 'timeout'
            ]
            incomplete_labels_count = len(timeout_labels)
            has_incomplete_labels = incomplete_labels_count > 0

            if has_incomplete_labels:
                logger.info(
                    "incomplete_labels_detected",
                    count=incomplete_labels_count
                )

        # Determine if full retrain needed
        requires_full_retrain = config_changed

        # Time since last run
        time_since_last_run = None
        if last_run_time:
            time_since_last_run = datetime.now() - last_run_time

        delta = Delta(
            has_new_candles=has_new_candles,
            new_candles_count=new_candles_count,
            new_candle_start=new_candle_start,
            new_candle_end=new_candle_end,
            has_incomplete_labels=has_incomplete_labels,
            incomplete_labels_count=incomplete_labels_count,
            config_changed=config_changed,
            requires_full_retrain=requires_full_retrain,
            time_since_last_run=time_since_last_run
        )

        logger.info(
            "delta_detection_complete",
            has_new_candles=has_new_candles,
            new_candles=new_candles_count,
            config_changed=config_changed,
            requires_full_retrain=requires_full_retrain
        )

        return delta

    def estimate_incremental_speedup(
        self,
        delta: Delta,
        full_dataset_size: int
    ) -> dict:
        """
        Estimate speed improvement from incremental vs full.

        Args:
            delta: Detected changes
            full_dataset_size: Total candles in full dataset

        Returns:
            Dictionary with speedup estimates
        """
        if delta.requires_full_retrain:
            return {
                'speedup': 1.0,
                'reason': 'full_retrain_required',
                'candles_to_process': full_dataset_size
            }

        if not delta.has_new_candles:
            return {
                'speedup': float('inf'),
                'reason': 'no_new_data',
                'candles_to_process': 0
            }

        # Incremental processes new candles + small lookback
        # Assume lookback = 2x timeout window (conservative)
        estimated_candles_to_process = delta.new_candles_count * 3

        speedup = full_dataset_size / max(estimated_candles_to_process, 1)

        return {
            'speedup': speedup,
            'reason': 'incremental_update',
            'candles_to_process': estimated_candles_to_process,
            'new_candles': delta.new_candles_count,
            'full_dataset_size': full_dataset_size
        }


def format_delta_summary(delta: Delta) -> str:
    """
    Format delta as human-readable summary.

    Args:
        delta: Delta object

    Returns:
        Formatted string

    Example:
        summary = format_delta_summary(delta)
        print(summary)
    """
    lines = []
    lines.append("=" * 60)
    lines.append("DELTA DETECTION SUMMARY")
    lines.append("=" * 60)

    # New candles
    if delta.has_new_candles:
        lines.append(f"\n✅ New Candles: {delta.new_candles_count}")
        lines.append(f"   Start: {delta.new_candle_start}")
        lines.append(f"   End:   {delta.new_candle_end}")
    else:
        lines.append("\n❌ No new candles")

    # Incomplete labels
    if delta.has_incomplete_labels:
        lines.append(f"\n⚠️  Incomplete Labels: {delta.incomplete_labels_count}")
        lines.append("   (Timeout exits that might now hit TP/SL)")
    else:
        lines.append("\n✅ No incomplete labels")

    # Config change
    if delta.config_changed:
        lines.append("\n⚠️  Config Changed")
        lines.append("   Full retrain required!")
    else:
        lines.append("\n✅ Config unchanged")

    # Action
    lines.append("\n" + "-" * 60)
    if delta.requires_full_retrain:
        lines.append("ACTION: Full retrain required")
    elif delta.has_new_candles:
        lines.append("ACTION: Incremental update")
    else:
        lines.append("ACTION: No update needed")

    # Time info
    if delta.time_since_last_run:
        hours = delta.time_since_last_run.total_seconds() / 3600
        lines.append(f"Time since last run: {hours:.1f} hours")

    lines.append("=" * 60)

    return "\n".join(lines)
