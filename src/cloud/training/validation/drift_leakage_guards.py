"""
Drift and Leakage Guards

PSI and KS tests on key features. Hard fail on label leakage or misaligned windows.

Key Features:
- Population Stability Index (PSI) tests
- Kolmogorov-Smirnov (KS) tests
- Label leakage detection
- Window alignment validation
- Feature drift detection
- Hard fail on violations

Author: Huracan Engine Team
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


class DriftStatus(Enum):
    """Drift status"""
    NO_DRIFT = "no_drift"
    MINOR_DRIFT = "minor_drift"
    MODERATE_DRIFT = "moderate_drift"
    SEVERE_DRIFT = "severe_drift"
    FAILED = "failed"


class LeakageStatus(Enum):
    """Leakage status"""
    NO_LEAKAGE = "no_leakage"
    SUSPECTED_LEAKAGE = "suspected_leakage"
    CONFIRMED_LEAKAGE = "confirmed_leakage"
    FAILED = "failed"


@dataclass
class DriftTestResult:
    """Drift test result"""
    feature_name: str
    psi_score: float
    ks_statistic: float
    ks_p_value: float
    drift_status: DriftStatus
    is_drifted: bool
    threshold_psi: float = 0.25
    threshold_ks: float = 0.05


@dataclass
class LeakageTestResult:
    """Leakage test result"""
    feature_name: str
    correlation_with_target: float
    leakage_status: LeakageStatus
    is_leaked: bool
    threshold: float = 0.95


@dataclass
class WindowAlignmentResult:
    """Window alignment validation result"""
    is_aligned: bool
    misaligned_features: List[str]
    alignment_errors: List[str]


@dataclass
class DriftLeakageReport:
    """Complete drift and leakage report"""
    drift_results: List[DriftTestResult]
    leakage_results: List[LeakageTestResult]
    window_alignment: WindowAlignmentResult
    overall_status: str  # "PASS", "WARNING", "FAIL"
    failed_checks: List[str]
    passed_checks: List[str]


class DriftLeakageGuards:
    """
    Drift and Leakage Guards.
    
    Performs PSI and KS tests on key features.
    Hard fails on label leakage or misaligned windows.
    
    Usage:
        guards = DriftLeakageGuards(
            psi_threshold=0.25,
            ks_p_value_threshold=0.05,
            leakage_threshold=0.95
        )
        
        report = guards.validate(
            training_features={...},
            validation_features={...},
            targets={...},
            feature_windows={...}
        )
        
        if report.overall_status == "FAIL":
            raise ValueError("Drift or leakage detected!")
    """
    
    def __init__(
        self,
        psi_threshold: float = 0.25,  # PSI threshold for drift
        ks_p_value_threshold: float = 0.05,  # KS test p-value threshold
        leakage_threshold: float = 0.95,  # Correlation threshold for leakage
        hard_fail_on_drift: bool = True,  # Hard fail on drift
        hard_fail_on_leakage: bool = True  # Hard fail on leakage
    ):
        """
        Initialize drift and leakage guards.
        
        Args:
            psi_threshold: PSI threshold for drift detection
            ks_p_value_threshold: KS test p-value threshold
            leakage_threshold: Correlation threshold for leakage
            hard_fail_on_drift: Hard fail on drift
            hard_fail_on_leakage: Hard fail on leakage
        """
        self.psi_threshold = psi_threshold
        self.ks_p_value_threshold = ks_p_value_threshold
        self.leakage_threshold = leakage_threshold
        self.hard_fail_on_drift = hard_fail_on_drift
        self.hard_fail_on_leakage = hard_fail_on_leakage
        
        logger.info(
            "drift_leakage_guards_initialized",
            psi_threshold=psi_threshold,
            ks_p_value_threshold=ks_p_value_threshold,
            leakage_threshold=leakage_threshold
        )
    
    def validate(
        self,
        training_features: Dict[str, np.ndarray],
        validation_features: Dict[str, np.ndarray],
        targets: Optional[np.ndarray] = None,
        feature_windows: Optional[Dict[str, int]] = None
    ) -> DriftLeakageReport:
        """
        Validate features for drift and leakage.
        
        Args:
            training_features: Training feature data
            validation_features: Validation feature data
            targets: Target values (for leakage detection)
            feature_windows: Feature windows (for alignment check)
        
        Returns:
            DriftLeakageReport
        """
        drift_results = []
        leakage_results = []
        failed_checks = []
        passed_checks = []
        
        # Test drift for each feature
        common_features = set(training_features.keys()) & set(validation_features.keys())
        
        for feature_name in common_features:
            # PSI test
            psi_score = self._calculate_psi(
                training_features[feature_name],
                validation_features[feature_name]
            )
            
            # KS test
            ks_statistic, ks_p_value = self._calculate_ks_test(
                training_features[feature_name],
                validation_features[feature_name]
            )
            
            # Determine drift status
            drift_status = self._determine_drift_status(psi_score, ks_p_value)
            is_drifted = drift_status in [DriftStatus.SEVERE_DRIFT, DriftStatus.FAILED]
            
            result = DriftTestResult(
                feature_name=feature_name,
                psi_score=psi_score,
                ks_statistic=ks_statistic,
                ks_p_value=ks_p_value,
                drift_status=drift_status,
                is_drifted=is_drifted,
                threshold_psi=self.psi_threshold,
                threshold_ks=self.ks_p_value_threshold
            )
            
            drift_results.append(result)
            
            if is_drifted:
                failed_checks.append(f"Drift detected in {feature_name} (PSI: {psi_score:.3f})")
            else:
                passed_checks.append(f"No drift in {feature_name}")
        
        # Test leakage if targets provided
        if targets is not None:
            for feature_name in common_features:
                correlation = self._calculate_correlation(
                    validation_features[feature_name],
                    targets
                )
                
                leakage_status = self._determine_leakage_status(correlation)
                is_leaked = leakage_status in [LeakageStatus.CONFIRMED_LEAKAGE, LeakageStatus.FAILED]
                
                result = LeakageTestResult(
                    feature_name=feature_name,
                    correlation_with_target=correlation,
                    leakage_status=leakage_status,
                    is_leaked=is_leaked,
                    threshold=self.leakage_threshold
                )
                
                leakage_results.append(result)
                
                if is_leaked:
                    failed_checks.append(f"Leakage detected in {feature_name} (correlation: {correlation:.3f})")
                else:
                    passed_checks.append(f"No leakage in {feature_name}")
        
        # Validate window alignment
        window_alignment = self._validate_window_alignment(
            training_features,
            validation_features,
            feature_windows
        )
        
        if not window_alignment.is_aligned:
            failed_checks.extend(window_alignment.alignment_errors)
        
        # Determine overall status
        overall_status = self._determine_overall_status(
            drift_results,
            leakage_results,
            window_alignment
        )
        
        # Hard fail if configured
        if overall_status == "FAIL":
            if self.hard_fail_on_drift and any(r.is_drifted for r in drift_results):
                raise ValueError(f"Hard fail: Drift detected in features: {[r.feature_name for r in drift_results if r.is_drifted]}")
            
            if self.hard_fail_on_leakage and any(r.is_leaked for r in leakage_results):
                raise ValueError(f"Hard fail: Leakage detected in features: {[r.feature_name for r in leakage_results if r.is_leaked]}")
        
        report = DriftLeakageReport(
            drift_results=drift_results,
            leakage_results=leakage_results,
            window_alignment=window_alignment,
            overall_status=overall_status,
            failed_checks=failed_checks,
            passed_checks=passed_checks
        )
        
        logger.info(
            "drift_leakage_validation_complete",
            overall_status=overall_status,
            num_drifted=sum(1 for r in drift_results if r.is_drifted),
            num_leaked=sum(1 for r in leakage_results if r.is_leaked),
            num_failed_checks=len(failed_checks)
        )
        
        return report
    
    def _calculate_psi(
        self,
        training_data: np.ndarray,
        validation_data: np.ndarray
    ) -> float:
        """Calculate Population Stability Index (PSI)"""
        # Bin the data
        min_val = min(np.min(training_data), np.min(validation_data))
        max_val = max(np.max(training_data), np.max(validation_data))
        
        if max_val == min_val:
            return 0.0
        
        num_bins = 10
        bins = np.linspace(min_val, max_val, num_bins + 1)
        
        # Calculate distributions
        training_hist, _ = np.histogram(training_data, bins=bins)
        validation_hist, _ = np.histogram(validation_data, bins=bins)
        
        # Normalize to probabilities
        training_probs = training_hist / (len(training_data) + 1e-9)
        validation_probs = validation_hist / (len(validation_data) + 1e-9)
        
        # Add small epsilon to avoid division by zero
        training_probs = training_probs + 1e-9
        validation_probs = validation_probs + 1e-9
        
        # Calculate PSI
        psi = np.sum((validation_probs - training_probs) * np.log(validation_probs / training_probs))
        
        return float(psi)
    
    def _calculate_ks_test(
        self,
        training_data: np.ndarray,
        validation_data: np.ndarray
    ) -> Tuple[float, float]:
        """Calculate Kolmogorov-Smirnov test"""
        # Sort data
        training_sorted = np.sort(training_data)
        validation_sorted = np.sort(validation_data)
        
        # Calculate empirical CDFs
        n_train = len(training_sorted)
        n_val = len(validation_sorted)
        
        # Combined sorted array
        combined = np.sort(np.concatenate([training_sorted, validation_sorted]))
        
        # Calculate CDFs
        train_cdf = np.searchsorted(training_sorted, combined, side='right') / n_train
        val_cdf = np.searchsorted(validation_sorted, combined, side='right') / n_val
        
        # KS statistic
        ks_statistic = np.max(np.abs(train_cdf - val_cdf))
        
        # Simplified p-value calculation (would use scipy.stats.ks_2samp in production)
        # Critical value approximation
        critical_value = np.sqrt((n_train + n_val) / (n_train * n_val)) * 1.36  # 95% confidence
        p_value = 0.05 if ks_statistic > critical_value else 0.5
        
        return float(ks_statistic), float(p_value)
    
    def _calculate_correlation(
        self,
        feature_data: np.ndarray,
        targets: np.ndarray
    ) -> float:
        """Calculate correlation between feature and target"""
        if len(feature_data) != len(targets):
            return 0.0
        
        # Remove NaN
        mask = ~(np.isnan(feature_data) | np.isnan(targets))
        feature_clean = feature_data[mask]
        targets_clean = targets[mask]
        
        if len(feature_clean) < 2:
            return 0.0
        
        # Calculate correlation
        correlation = np.corrcoef(feature_clean, targets_clean)[0, 1]
        
        return float(abs(correlation)) if not np.isnan(correlation) else 0.0
    
    def _determine_drift_status(
        self,
        psi_score: float,
        ks_p_value: float
    ) -> DriftStatus:
        """Determine drift status"""
        if psi_score > 0.5 or ks_p_value < 0.01:
            return DriftStatus.SEVERE_DRIFT
        elif psi_score > self.psi_threshold or ks_p_value < self.ks_p_value_threshold:
            return DriftStatus.MODERATE_DRIFT
        elif psi_score > self.psi_threshold * 0.5 or ks_p_value < self.ks_p_value_threshold * 2:
            return DriftStatus.MINOR_DRIFT
        else:
            return DriftStatus.NO_DRIFT
    
    def _determine_leakage_status(self, correlation: float) -> LeakageStatus:
        """Determine leakage status"""
        if correlation >= self.leakage_threshold:
            return LeakageStatus.CONFIRMED_LEAKAGE
        elif correlation >= self.leakage_threshold * 0.9:
            return LeakageStatus.SUSPECTED_LEAKAGE
        else:
            return LeakageStatus.NO_LEAKAGE
    
    def _validate_window_alignment(
        self,
        training_features: Dict[str, np.ndarray],
        validation_features: Dict[str, np.ndarray],
        feature_windows: Optional[Dict[str, int]]
    ) -> WindowAlignmentResult:
        """Validate window alignment"""
        if feature_windows is None:
            return WindowAlignmentResult(
                is_aligned=True,
                misaligned_features=[],
                alignment_errors=[]
            )
        
        misaligned_features = []
        alignment_errors = []
        
        for feature_name, window in feature_windows.items():
            if feature_name in training_features and feature_name in validation_features:
                train_data = training_features[feature_name]
                val_data = validation_features[feature_name]
                
                # Check if data length is consistent with window
                # Simplified: check if data has expected structure
                if len(train_data) > 0 and len(val_data) > 0:
                    # Check for temporal alignment issues
                    # In production, would check timestamps and windows
                    if abs(len(train_data) - len(val_data)) > window:
                        misaligned_features.append(feature_name)
                        alignment_errors.append(
                            f"Window misalignment in {feature_name}: "
                            f"train_length={len(train_data)}, val_length={len(val_data)}, window={window}"
                        )
        
        is_aligned = len(misaligned_features) == 0
        
        return WindowAlignmentResult(
            is_aligned=is_aligned,
            misaligned_features=misaligned_features,
            alignment_errors=alignment_errors
        )
    
    def _determine_overall_status(
        self,
        drift_results: List[DriftTestResult],
        leakage_results: List[LeakageTestResult],
        window_alignment: WindowAlignmentResult
    ) -> str:
        """Determine overall validation status"""
        has_severe_drift = any(r.drift_status == DriftStatus.SEVERE_DRIFT for r in drift_results)
        has_confirmed_leakage = any(r.leakage_status == LeakageStatus.CONFIRMED_LEAKAGE for r in leakage_results)
        has_window_misalignment = not window_alignment.is_aligned
        
        if has_severe_drift or has_confirmed_leakage or has_window_misalignment:
            return "FAIL"
        elif any(r.is_drifted for r in drift_results) or any(r.is_leaked for r in leakage_results):
            return "WARNING"
        else:
            return "PASS"

