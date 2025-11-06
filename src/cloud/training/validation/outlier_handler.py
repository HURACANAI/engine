"""
Outlier Detection and Handling System

Detects outliers using multiple methods:
1. Statistical (Z-score, IQR)
2. Domain-based (price relationships)
3. Volume-based (unusual volume spikes)
4. Time-based (sudden changes)

Handles outliers by:
1. Flagging for review
2. Capping at thresholds
3. Removing extreme outliers
4. Imputing with median/mean
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class OutlierDetection:
    """Outlier detection result."""

    column: str
    method: str  # 'z_score', 'iqr', 'domain', 'volume'
    outlier_count: int
    outlier_indices: List[int]
    outlier_values: List[float]
    threshold: float
    message: str


@dataclass
class OutlierHandlingResult:
    """Result of outlier handling."""

    original_count: int
    outliers_detected: int
    outliers_handled: int
    method_used: str
    data_cleaned: pl.DataFrame
    report: str


class OutlierDetector:
    """
    Outlier detection using multiple methods.

    Methods:
    1. Z-score: Statistical outliers (>3 std devs)
    2. IQR: Interquartile range method
    3. Domain-based: Price relationship violations
    4. Volume-based: Unusual volume spikes
    5. Time-based: Sudden price changes

    Usage:
        detector = OutlierDetector(
            z_threshold=3.0,
            iqr_multiplier=1.5,
        )

        detections = detector.detect_all(data, symbol="BTC/USDT")

        handler = OutlierHandler(method="cap")
        result = handler.handle(data, detections)
    """

    def __init__(
        self,
        z_threshold: float = 3.0,
        iqr_multiplier: float = 1.5,
        volume_spike_threshold: float = 5.0,  # 5x average volume
        price_change_threshold: float = 0.20,  # 20% change
    ):
        """
        Initialize outlier detector.

        Args:
            z_threshold: Z-score threshold for outliers
            iqr_multiplier: IQR multiplier for outlier detection
            volume_spike_threshold: Volume spike threshold (x average)
            price_change_threshold: Price change threshold (percentage)
        """
        self.z_threshold = z_threshold
        self.iqr_multiplier = iqr_multiplier
        self.volume_spike_threshold = volume_spike_threshold
        self.price_change_threshold = price_change_threshold

        logger.info("outlier_detector_initialized")

    def detect_z_score(self, data: pl.DataFrame, column: str) -> OutlierDetection:
        """Detect outliers using Z-score method."""
        values = data[column].to_numpy()
        mean = np.mean(values)
        std = np.std(values)

        if std == 0:
            return OutlierDetection(
                column=column,
                method="z_score",
                outlier_count=0,
                outlier_indices=[],
                outlier_values=[],
                threshold=self.z_threshold,
                message="No variance - cannot detect outliers",
            )

        z_scores = np.abs((values - mean) / std)
        outlier_mask = z_scores > self.z_threshold
        outlier_indices = np.where(outlier_mask)[0].tolist()
        outlier_values = values[outlier_mask].tolist()

        return OutlierDetection(
            column=column,
            method="z_score",
            outlier_count=len(outlier_indices),
            outlier_indices=outlier_indices,
            outlier_values=outlier_values,
            threshold=self.z_threshold,
            message=f"Z-score: {len(outlier_indices)} outliers detected",
        )

    def detect_iqr(self, data: pl.DataFrame, column: str) -> OutlierDetection:
        """Detect outliers using IQR method."""
        values = data[column].to_numpy()
        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1

        lower_bound = q1 - self.iqr_multiplier * iqr
        upper_bound = q3 + self.iqr_multiplier * iqr

        outlier_mask = (values < lower_bound) | (values > upper_bound)
        outlier_indices = np.where(outlier_mask)[0].tolist()
        outlier_values = values[outlier_mask].tolist()

        return OutlierDetection(
            column=column,
            method="iqr",
            outlier_count=len(outlier_indices),
            outlier_indices=outlier_indices,
            outlier_values=outlier_values,
            threshold=self.iqr_multiplier,
            message=f"IQR: {len(outlier_indices)} outliers detected",
        )

    def detect_domain_based(self, data: pl.DataFrame) -> List[OutlierDetection]:
        """Detect domain-based outliers (price relationship violations)."""
        detections = []

        # Check: high < low
        if "high" in data.columns and "low" in data.columns:
            invalid_mask = data["high"] < data["low"]
            invalid_indices = np.where(invalid_mask.to_numpy())[0].tolist()

            if len(invalid_indices) > 0:
                detections.append(
                    OutlierDetection(
                        column="high_low",
                        method="domain",
                        outlier_count=len(invalid_indices),
                        outlier_indices=invalid_indices,
                        outlier_values=[],
                        threshold=0.0,
                        message=f"Domain: {len(invalid_indices)} rows with high < low",
                    )
                )

        # Check: close outside high/low range
        if all(col in data.columns for col in ["high", "low", "close"]):
            invalid_mask = (data["close"] > data["high"]) | (data["close"] < data["low"])
            invalid_indices = np.where(invalid_mask.to_numpy())[0].tolist()

            if len(invalid_indices) > 0:
                detections.append(
                    OutlierDetection(
                        column="close_range",
                        method="domain",
                        outlier_count=len(invalid_indices),
                        outlier_indices=invalid_indices,
                        outlier_values=[],
                        threshold=0.0,
                        message=f"Domain: {len(invalid_indices)} rows with close outside range",
                    )
                )

        return detections

    def detect_volume_spikes(self, data: pl.DataFrame) -> Optional[OutlierDetection]:
        """Detect unusual volume spikes."""
        if "volume" not in data.columns:
            return None

        volumes = data["volume"].to_numpy()
        avg_volume = np.mean(volumes)
        std_volume = np.std(volumes)

        if std_volume == 0:
            return None

        # Detect spikes (>threshold x average)
        spike_threshold = avg_volume * self.volume_spike_threshold
        spike_mask = volumes > spike_threshold
        spike_indices = np.where(spike_mask)[0].tolist()
        spike_values = volumes[spike_mask].tolist()

        return OutlierDetection(
            column="volume",
            method="volume_spike",
            outlier_count=len(spike_indices),
            outlier_indices=spike_indices,
            outlier_values=spike_values,
            threshold=self.volume_spike_threshold,
            message=f"Volume: {len(spike_indices)} spikes detected",
        )

    def detect_all(self, data: pl.DataFrame, symbol: str) -> List[OutlierDetection]:
        """Detect outliers using all methods."""
        detections = []

        # Z-score for price columns
        price_columns = ["open", "high", "low", "close"]
        for col in price_columns:
            if col in data.columns:
                detections.append(self.detect_z_score(data, col))

        # IQR for price columns
        for col in price_columns:
            if col in data.columns:
                detections.append(self.detect_iqr(data, col))

        # Domain-based
        detections.extend(self.detect_domain_based(data))

        # Volume spikes
        volume_detection = self.detect_volume_spikes(data)
        if volume_detection:
            detections.append(volume_detection)

        logger.info(
            "outlier_detection_complete",
            symbol=symbol,
            total_detections=len(detections),
            total_outliers=sum(d.outlier_count for d in detections),
        )

        return detections


class OutlierHandler:
    """
    Handle detected outliers.

    Methods:
    1. 'flag': Flag for review (no changes)
    2. 'cap': Cap at threshold
    3. 'remove': Remove extreme outliers
    4. 'impute': Impute with median/mean
    """

    def __init__(self, method: str = "cap"):
        """
        Initialize outlier handler.

        Args:
            method: Handling method ('flag', 'cap', 'remove', 'impute')
        """
        self.method = method
        logger.info("outlier_handler_initialized", method=method)

    def handle(
        self, data: pl.DataFrame, detections: List[OutlierDetection]
    ) -> OutlierHandlingResult:
        """
        Handle detected outliers.

        Args:
            data: Original data
            detections: Outlier detections

        Returns:
            OutlierHandlingResult with cleaned data
        """
        original_count = data.height
        total_outliers = sum(d.outlier_count for d in detections)
        cleaned_data = data.clone()

        handled_count = 0

        for detection in detections:
            if detection.outlier_count == 0:
                continue

            if self.method == "cap":
                # Cap outliers at threshold
                if detection.column in cleaned_data.columns:
                    values = cleaned_data[detection.column].to_numpy()
                    mean = np.mean(values)
                    std = np.std(values)

                    if std > 0:
                        upper_bound = mean + self.z_threshold * std
                        lower_bound = mean - self.z_threshold * std

                        # Cap values
                        capped_values = np.clip(values, lower_bound, upper_bound)
                        cleaned_data = cleaned_data.with_columns(
                            pl.Series(detection.column, capped_values)
                        )
                        handled_count += detection.outlier_count

            elif self.method == "remove":
                # Remove rows with outliers
                outlier_rows = set(detection.outlier_indices)
                if outlier_rows:
                    # Create mask to keep non-outlier rows
                    keep_mask = ~pl.Series([i in outlier_rows for i in range(len(cleaned_data))])
                    cleaned_data = cleaned_data.filter(keep_mask)
                    handled_count += len(outlier_rows)

            elif self.method == "impute":
                # Impute with median
                if detection.column in cleaned_data.columns:
                    median_value = cleaned_data[detection.column].median()
                    for idx in detection.outlier_indices:
                        cleaned_data = cleaned_data.with_columns(
                            pl.when(pl.int_range(0, len(cleaned_data)) == idx)
                            .then(pl.lit(median_value))
                            .otherwise(pl.col(detection.column))
                            .alias(detection.column)
                        )
                    handled_count += detection.outlier_count

        report = f"Handled {handled_count}/{total_outliers} outliers using {self.method} method"

        return OutlierHandlingResult(
            original_count=original_count,
            outliers_detected=total_outliers,
            outliers_handled=handled_count,
            method_used=self.method,
            data_cleaned=cleaned_data,
            report=report,
        )

    def __init__(self, method: str = "cap", z_threshold: float = 3.0):
        """Initialize with z_threshold for capping."""
        self.method = method
        self.z_threshold = z_threshold
        logger.info("outlier_handler_initialized", method=method)

