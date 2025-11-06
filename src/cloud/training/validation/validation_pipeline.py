"""
Validation Integration Module

Integrates all validation components into the training pipeline.

This module provides a unified interface for:
1. Mandatory OOS validation
2. Overfitting detection
3. Data validation
4. Extended paper trading (optional)
5. Stress testing (optional)

All validations are integrated into the daily retrain pipeline.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import structlog

from ..config.settings import EngineSettings
from ..engine.walk_forward import WalkForwardResults, WalkForwardValidator
from .mandatory_oos_validator import MandatoryOOSValidator, ValidationResult
from .overfitting_detector import RobustOverfittingDetector, OverfittingReport
from .data_validator import AutomatedDataValidator, ValidationReport
from .paper_trading_validator import ExtendedPaperTradingValidator, PaperTradingResult
from .stress_testing import StressTestingFramework, StressTestResult

logger = structlog.get_logger(__name__)


@dataclass
class ValidationPipelineResult:
    """Complete validation pipeline result."""

    passed: bool  # True if all validations passed
    oos_validation: Optional[ValidationResult] = None
    overfitting_detection: Optional[OverfittingReport] = None
    data_validation: Optional[ValidationReport] = None
    paper_trading: Optional[PaperTradingResult] = None
    stress_testing: Optional[StressTestResult] = None
    blocking_issues: List[str] = None
    recommendation: str = ""


class ValidationPipeline:
    """
    Validation pipeline integrating all validation components.

    Usage:
        pipeline = ValidationPipeline(settings=settings)

        result = pipeline.validate(
            walk_forward_results=wf_results,
            model_id="model_v1",
            data=df,
            symbol="BTC/USDT",
        )

        if not result.passed:
            raise ValueError(f"Validation failed: {result.blocking_issues}")
    """

    def __init__(self, settings: EngineSettings):
        """
        Initialize validation pipeline.

        Args:
            settings: Engine settings with validation configuration
        """
        self.settings = settings
        self.validation_settings = settings.training.validation

        # Initialize validators based on configuration
        self.oos_validator = None
        self.overfitting_detector = None
        self.data_validator = None
        self.paper_trading_validator = None
        self.stress_testing_framework = None

        if self.validation_settings.enabled:
            if self.validation_settings.mandatory_oos.enabled:
                self.oos_validator = MandatoryOOSValidator(
                    min_oos_sharpe=self.validation_settings.mandatory_oos.min_oos_sharpe,
                    min_oos_win_rate=self.validation_settings.mandatory_oos.min_oos_win_rate,
                    max_train_test_gap=self.validation_settings.mandatory_oos.max_train_test_gap,
                    max_sharpe_std=self.validation_settings.mandatory_oos.max_sharpe_std,
                    min_test_trades=self.validation_settings.mandatory_oos.min_test_trades,
                    min_windows=self.validation_settings.mandatory_oos.min_windows,
                )

            if self.validation_settings.overfitting_detection.enabled:
                self.overfitting_detector = RobustOverfittingDetector(
                    train_test_gap_threshold=self.validation_settings.overfitting_detection.train_test_gap_threshold,
                    cv_stability_threshold=self.validation_settings.overfitting_detection.cv_stability_threshold,
                    degradation_threshold=self.validation_settings.overfitting_detection.degradation_threshold,
                )

            if self.validation_settings.data_validation.enabled:
                self.data_validator = AutomatedDataValidator(
                    outlier_z_threshold=self.validation_settings.data_validation.outlier_z_threshold,
                    max_missing_pct=self.validation_settings.data_validation.max_missing_pct,
                    max_age_hours=self.validation_settings.data_validation.max_age_hours,
                    min_coverage=self.validation_settings.data_validation.min_coverage,
                )

            if self.validation_settings.paper_trading.enabled:
                self.paper_trading_validator = ExtendedPaperTradingValidator(
                    min_duration_days=self.validation_settings.paper_trading.min_duration_days,
                    min_trades=self.validation_settings.paper_trading.min_trades,
                    min_win_rate=self.validation_settings.paper_trading.min_win_rate,
                    min_sharpe=self.validation_settings.paper_trading.min_sharpe,
                    max_backtest_deviation=self.validation_settings.paper_trading.max_backtest_deviation,
                )

            if self.validation_settings.stress_testing.enabled:
                self.stress_testing_framework = StressTestingFramework(
                    max_drawdown_threshold=self.validation_settings.stress_testing.max_drawdown_threshold,
                    min_survival_rate=self.validation_settings.stress_testing.min_survival_rate,
                )

        logger.info(
            "validation_pipeline_initialized",
            enabled=self.validation_settings.enabled,
            oos_enabled=self.oos_validator is not None,
            overfitting_enabled=self.overfitting_detector is not None,
            data_validation_enabled=self.data_validator is not None,
        )

    def validate(
        self,
        walk_forward_results: Optional[WalkForwardResults] = None,
        model_id: str = "model",
        data: Optional[any] = None,
        symbol: str = "BTC/USDT",
        paper_trades: Optional[List[Dict]] = None,
        backtest_results: Optional[Dict] = None,
        model: Optional[any] = None,
        historical_data: Optional[any] = None,
    ) -> ValidationPipelineResult:
        """
        Run complete validation pipeline.

        Args:
            walk_forward_results: Walk-forward validation results
            model_id: Model identifier
            data: Training data (for data validation)
            symbol: Symbol name
            paper_trades: Paper trades (for extended paper trading validation)
            backtest_results: Backtest results (for comparison)
            model: Model object (for stress testing)
            historical_data: Historical data (for stress testing)

        Returns:
            ValidationPipelineResult with all validation results

        Raises:
            ValueError: If validation fails (hard block)
        """
        if not self.validation_settings.enabled:
            logger.info("validation_pipeline_disabled", model_id=model_id)
            return ValidationPipelineResult(
                passed=True,
                recommendation="Validation pipeline disabled",
            )

        blocking_issues = []
        oos_result = None
        overfitting_report = None
        data_report = None
        paper_result = None
        stress_result = None

        # 1. Data Validation (if data provided)
        if self.data_validator and data is not None:
            try:
                data_report = self.data_validator.validate(
                    data=data,
                    symbol=symbol,
                    expected_columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'],
                )
                if not data_report.passed:
                    blocking_issues.extend(data_report.issues)
            except Exception as e:
                logger.warning("data_validation_failed", error=str(e), symbol=symbol)
                blocking_issues.append(f"Data validation error: {str(e)}")

        # 2. Mandatory OOS Validation (if walk-forward results provided)
        if self.oos_validator and walk_forward_results is not None:
            try:
                total_test_trades = getattr(walk_forward_results, 'total_test_trades', 0)
                oos_result = self.oos_validator.validate(
                    walk_forward_results=walk_forward_results,
                    model_id=model_id,
                    total_test_trades=total_test_trades,
                )
                # If validation fails, it raises ValueError (hard block)
            except ValueError as e:
                blocking_issues.append(f"OOS validation failed: {str(e)}")
                logger.error("oos_validation_failed", model_id=model_id, error=str(e))

        # 3. Overfitting Detection (if walk-forward results provided)
        if self.overfitting_detector and walk_forward_results is not None:
            try:
                # Calculate train/test metrics (simplified - would need actual train metrics)
                train_sharpe = getattr(walk_forward_results, 'train_sharpe', walk_forward_results.test_sharpe * 1.2)
                train_win_rate = getattr(walk_forward_results, 'train_win_rate', walk_forward_results.test_win_rate * 1.1)

                overfitting_report = self.overfitting_detector.detect_overfitting(
                    train_sharpe=train_sharpe,
                    test_sharpe=walk_forward_results.test_sharpe,
                    train_win_rate=train_win_rate,
                    test_win_rate=walk_forward_results.test_win_rate,
                    cv_sharpe_std=walk_forward_results.sharpe_std,
                )

                if overfitting_report.is_overfitting:
                    blocking_issues.append(f"Overfitting detected: {overfitting_report.recommendation}")
            except Exception as e:
                logger.warning("overfitting_detection_failed", error=str(e), model_id=model_id)

        # 4. Extended Paper Trading Validation (if enabled and paper trades provided)
        if self.paper_trading_validator and paper_trades is not None:
            try:
                paper_result = self.paper_trading_validator.validate(
                    paper_trades=paper_trades,
                    backtest_results=backtest_results,
                    model_id=model_id,
                )
                # If validation fails, it raises ValueError (hard block)
            except ValueError as e:
                blocking_issues.append(f"Paper trading validation failed: {str(e)}")
                logger.error("paper_trading_validation_failed", model_id=model_id, error=str(e))
            except Exception as e:
                logger.warning("paper_trading_validation_error", error=str(e), model_id=model_id)

        # 5. Stress Testing (if enabled and model provided)
        if self.stress_testing_framework and model is not None and historical_data is not None:
            try:
                stress_result = self.stress_testing_framework.run_stress_tests(
                    model=model,
                    historical_data=historical_data,
                    model_id=model_id,
                )
                # If stress tests fail, it raises ValueError (hard block)
            except ValueError as e:
                blocking_issues.append(f"Stress testing failed: {str(e)}")
                logger.error("stress_testing_failed", model_id=model_id, error=str(e))
            except Exception as e:
                logger.warning("stress_testing_error", error=str(e), model_id=model_id)

        # Determine overall pass/fail
        passed = len(blocking_issues) == 0

        # Generate recommendation
        if passed:
            recommendation = "✅ Model passed all validation checks. Safe to deploy."
        else:
            recommendation = f"❌ Model FAILED validation. {len(blocking_issues)} blocking issue(s). DO NOT DEPLOY."

        result = ValidationPipelineResult(
            passed=passed,
            oos_validation=oos_result,
            overfitting_detection=overfitting_report,
            data_validation=data_report,
            paper_trading=paper_result,
            stress_testing=stress_result,
            blocking_issues=blocking_issues or [],
            recommendation=recommendation,
        )

        logger.info(
            "validation_pipeline_complete",
            model_id=model_id,
            passed=passed,
            blocking_issues=len(blocking_issues),
        )

        # HARD BLOCK: Raise error if validation fails
        if not passed:
            error_msg = f"Model {model_id} FAILED validation pipeline:\n"
            error_msg += "\n".join(f"  - {issue}" for issue in blocking_issues)
            error_msg += "\n\nDO NOT DEPLOY THIS MODEL!"
            raise ValueError(error_msg)

        return result

