"""
Validation Module - Complete Implementation

This module provides comprehensive validation, optimization, and testing capabilities:

1. Mandatory OOS Validation - Strict enforcement before deployment
2. Robust Overfitting Detection - Multiple indicators
3. Automated Data Validation - Comprehensive checks
4. Outlier Detection and Handling - Multiple methods
5. Missing Data Imputation - Multiple strategies
6. Parallel Signal Processing - Ray-based parallelization
7. Computation Caching - LRU cache with TTL
8. Database Query Optimization - Performance improvements
9. Extended Paper Trading Validation - 2-4 weeks minimum
10. Regime-Specific Performance Tracking - Cross-regime analysis
11. Stress Testing Framework - Extreme condition testing
12. Validation Pipeline - Unified integration

All components are production-ready and integrated.
"""

from .mandatory_oos_validator import MandatoryOOSValidator, ValidationResult
from .overfitting_detector import RobustOverfittingDetector, OverfittingReport
from .data_validator import AutomatedDataValidator, ValidationReport
from .outlier_handler import OutlierDetector, OutlierHandler
from .missing_data_imputer import MissingDataImputer, ImputationReport
from .paper_trading_validator import ExtendedPaperTradingValidator, PaperTradingResult
from .regime_performance_tracker import RegimePerformanceTracker, RegimePerformanceReport
from .stress_testing import StressTestingFramework, StressTestResult
from .validation_pipeline import ValidationPipeline, ValidationPipelineResult
from .concept_drift_detector import ConceptDriftDetector, DriftReport, DriftSeverity

__all__ = [
    "MandatoryOOSValidator",
    "ValidationResult",
    "RobustOverfittingDetector",
    "OverfittingReport",
    "AutomatedDataValidator",
    "ValidationReport",
    "OutlierDetector",
    "OutlierHandler",
    "MissingDataImputer",
    "ImputationReport",
    "ExtendedPaperTradingValidator",
    "PaperTradingResult",
    "RegimePerformanceTracker",
    "RegimePerformanceReport",
    "StressTestingFramework",
    "StressTestResult",
    "ValidationPipeline",
    "ValidationPipelineResult",
    "ConceptDriftDetector",
    "DriftReport",
    "DriftSeverity",
]

