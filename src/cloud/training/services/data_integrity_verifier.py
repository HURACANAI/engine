"""Data integrity verification and automatic repair."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import structlog  # type: ignore[reportMissingImports]

logger = structlog.get_logger(__name__)


class DataIntegrityVerifier:
    """
    Verifies data integrity and automatically repairs issues.
    
    Detects:
    - NaN values
    - Timestamp drifts
    - Irregular intervals
    - Duplicate entries
    - Missing candles
    
    Scores dataset reliability (0-100).
    Only trains if reliability > threshold.
    """

    def __init__(
        self,
        reliability_threshold: float = 80.0,
        timestamp_tolerance_seconds: int = 120,  # 2 minutes
        expected_interval_seconds: int = 60,  # 1 minute
    ) -> None:
        """
        Initialize data integrity verifier.
        
        Args:
            reliability_threshold: Minimum reliability score to allow training
            timestamp_tolerance_seconds: Tolerance for timestamp drift
            expected_interval_seconds: Expected interval between candles
        """
        self.reliability_threshold = reliability_threshold
        self.timestamp_tolerance = timestamp_tolerance_seconds
        self.expected_interval = expected_interval_seconds
        
        logger.info(
            "data_integrity_verifier_initialized",
            reliability_threshold=reliability_threshold,
            timestamp_tolerance=timestamp_tolerance_seconds,
        )

    def verify_data(
        self,
        data: pd.DataFrame,
        symbol: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Verify data integrity.
        
        Args:
            data: DataFrame to verify
            symbol: Optional symbol for logging
            
        Returns:
            Verification results dictionary
        """
        issues = []
        scores = {}
        
        # Check for required columns
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            issues.append({
                "type": "missing_columns",
                "severity": "critical",
                "details": f"Missing columns: {missing_columns}",
            })
            scores["missing_columns"] = 0.0
        else:
            scores["missing_columns"] = 100.0
        
        # Check for NaN values
        nan_issues = self._check_nans(data)
        if nan_issues:
            issues.extend(nan_issues)
            scores["nan_check"] = 100.0 - (len(nan_issues) * 10.0)  # Penalize 10 points per NaN issue
        else:
            scores["nan_check"] = 100.0
        
        # Check timestamp drift
        timestamp_issues = self._check_timestamp_drift(data)
        if timestamp_issues:
            issues.extend(timestamp_issues)
            scores["timestamp_check"] = 100.0 - (len(timestamp_issues) * 5.0)
        else:
            scores["timestamp_check"] = 100.0
        
        # Check irregular intervals
        interval_issues = self._check_irregular_intervals(data)
        if interval_issues:
            issues.extend(interval_issues)
            scores["interval_check"] = 100.0 - (len(interval_issues) * 3.0)
        else:
            scores["interval_check"] = 100.0
        
        # Check duplicates
        duplicate_issues = self._check_duplicates(data)
        if duplicate_issues:
            issues.extend(duplicate_issues)
            scores["duplicate_check"] = 100.0 - (len(duplicate_issues) * 5.0)
        else:
            scores["duplicate_check"] = 100.0
        
        # Calculate overall reliability score
        reliability = np.mean(list(scores.values()))
        
        # Determine if training should proceed
        should_train = reliability >= self.reliability_threshold
        
        result = {
            "reliability": float(reliability),
            "should_train": should_train,
            "issues": issues,
            "scores": scores,
            "symbol": symbol,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        }
        
        if should_train:
            logger.info(
                "data_verification_passed",
                symbol=symbol,
                reliability=reliability,
                num_issues=len(issues),
            )
        else:
            logger.warning(
                "data_verification_failed",
                symbol=symbol,
                reliability=reliability,
                num_issues=len(issues),
                threshold=self.reliability_threshold,
            )
        
        return result

    def _check_nans(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Check for NaN values."""
        issues = []
        
        # Check for NaN in numeric columns
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            nan_count = data[col].isna().sum()
            if nan_count > 0:
                nan_percentage = (nan_count / len(data)) * 100
                issues.append({
                    "type": "nan_values",
                    "severity": "critical" if nan_percentage > 5 else "warning",
                    "column": col,
                    "count": int(nan_count),
                    "percentage": float(nan_percentage),
                    "details": f"Column {col} has {nan_count} NaN values ({nan_percentage:.2f}%)",
                })
        
        return issues

    def _check_timestamp_drift(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Check for timestamp drift."""
        issues = []
        
        if 'timestamp' not in data.columns:
            return issues
        
        # Convert timestamp to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(data['timestamp']):
            try:
                data['timestamp'] = pd.to_datetime(data['timestamp'])
            except Exception:
                issues.append({
                    "type": "timestamp_format",
                    "severity": "critical",
                    "details": "Timestamp column cannot be converted to datetime",
                })
                return issues
        
        # Check for monotonic timestamps
        if not data['timestamp'].is_monotonic_increasing:
            issues.append({
                "type": "timestamp_not_monotonic",
                "severity": "critical",
                "details": "Timestamp column is not monotonically increasing",
            })
        
        # Check for large gaps
        if len(data) > 1:
            time_diffs = data['timestamp'].diff().dt.total_seconds()
            large_gaps = time_diffs[time_diffs > self.timestamp_tolerance]
            
            if len(large_gaps) > 0:
                max_gap = float(large_gaps.max())
                issues.append({
                    "type": "timestamp_gaps",
                    "severity": "warning" if max_gap < 3600 else "critical",
                    "count": int(len(large_gaps)),
                    "max_gap_seconds": max_gap,
                    "details": f"Found {len(large_gaps)} timestamp gaps > {self.timestamp_tolerance}s",
                })
        
        return issues

    def _check_irregular_intervals(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Check for irregular intervals."""
        issues = []
        
        if 'timestamp' not in data.columns or len(data) < 2:
            return issues
        
        # Convert timestamp to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(data['timestamp']):
            try:
                data['timestamp'] = pd.to_datetime(data['timestamp'])
            except Exception:
                return issues
        
        # Calculate intervals
        time_diffs = data['timestamp'].diff().dt.total_seconds().dropna()
        
        if len(time_diffs) == 0:
            return issues
        
        # Check for irregular intervals (more than 20% deviation from expected)
        expected_interval = self.expected_interval
        tolerance = expected_interval * 0.2  # 20% tolerance
        
        irregular_intervals = time_diffs[
            (time_diffs < expected_interval - tolerance) |
            (time_diffs > expected_interval + tolerance)
        ]
        
        if len(irregular_intervals) > 0:
            irregular_percentage = (len(irregular_intervals) / len(time_diffs)) * 100
            issues.append({
                "type": "irregular_intervals",
                "severity": "warning" if irregular_percentage < 10 else "critical",
                "count": int(len(irregular_intervals)),
                "percentage": float(irregular_percentage),
                "expected_interval": expected_interval,
                "details": f"Found {len(irregular_intervals)} irregular intervals ({irregular_percentage:.2f}%)",
            })
        
        return issues

    def _check_duplicates(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Check for duplicate entries."""
        issues = []
        
        # Check for duplicate timestamps
        if 'timestamp' in data.columns:
            duplicate_timestamps = data['timestamp'].duplicated().sum()
            if duplicate_timestamps > 0:
                issues.append({
                    "type": "duplicate_timestamps",
                    "severity": "critical",
                    "count": int(duplicate_timestamps),
                    "details": f"Found {duplicate_timestamps} duplicate timestamps",
                })
        
        # Check for completely duplicate rows
        duplicate_rows = data.duplicated().sum()
        if duplicate_rows > 0:
            issues.append({
                "type": "duplicate_rows",
                "severity": "warning",
                "count": int(duplicate_rows),
                "details": f"Found {duplicate_rows} duplicate rows",
            })
        
        return issues

    def repair_data(
        self,
        data: pd.DataFrame,
        verification_result: Optional[Dict[str, Any]] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Attempt to repair data issues.
        
        Args:
            data: DataFrame to repair
            verification_result: Optional verification result (if already verified)
            
        Returns:
            Tuple of (repaired_data, repair_report)
        """
        if verification_result is None:
            verification_result = self.verify_data(data)
        
        repaired_data = data.copy()
        repairs = []
        
        # Repair NaN values (forward fill, then backward fill)
        if any(issue["type"] == "nan_values" for issue in verification_result.get("issues", [])):
            original_nan_count = repaired_data.isna().sum().sum()
            repaired_data = repaired_data.fillna(method='ffill').fillna(method='bfill')
            new_nan_count = repaired_data.isna().sum().sum()
            
            if new_nan_count < original_nan_count:
                repairs.append({
                    "type": "nan_repair",
                    "repaired": int(original_nan_count - new_nan_count),
                    "remaining": int(new_nan_count),
                })
        
        # Remove duplicate timestamps (keep first)
        if any(issue["type"] == "duplicate_timestamps" for issue in verification_result.get("issues", [])):
            if 'timestamp' in repaired_data.columns:
                original_len = len(repaired_data)
                repaired_data = repaired_data.drop_duplicates(subset=['timestamp'], keep='first')
                new_len = len(repaired_data)
                
                if new_len < original_len:
                    repairs.append({
                        "type": "duplicate_timestamp_removal",
                        "removed": int(original_len - new_len),
                    })
        
        # Remove completely duplicate rows
        if any(issue["type"] == "duplicate_rows" for issue in verification_result.get("issues", [])):
            original_len = len(repaired_data)
            repaired_data = repaired_data.drop_duplicates()
            new_len = len(repaired_data)
            
            if new_len < original_len:
                repairs.append({
                    "type": "duplicate_row_removal",
                    "removed": int(original_len - new_len),
                })
        
        # Sort by timestamp
        if 'timestamp' in repaired_data.columns:
            repaired_data = repaired_data.sort_values('timestamp').reset_index(drop=True)
        
        repair_report = {
            "repairs": repairs,
            "original_length": len(data),
            "repaired_length": len(repaired_data),
        }
        
        logger.info("data_repair_complete", repairs=repairs)
        
        return repaired_data, repair_report

    def calculate_reliability_score(self, issues: List[Dict[str, Any]]) -> float:
        """
        Calculate reliability score from issues.
        
        Args:
            issues: List of issues
            
        Returns:
            Reliability score (0-100)
        """
        if not issues:
            return 100.0
        
        # Penalty weights
        weights = {
            "critical": 20.0,
            "warning": 5.0,
            "info": 1.0,
        }
        
        total_penalty = 0.0
        for issue in issues:
            severity = issue.get("severity", "info")
            penalty = weights.get(severity, 1.0)
            total_penalty += penalty
        
        reliability = max(0.0, 100.0 - total_penalty)
        
        return reliability

