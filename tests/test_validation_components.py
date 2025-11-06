"""
Unit Tests for Validation Components

Tests for:
1. Mandatory OOS Validator
2. Overfitting Detector
3. Data Validator
4. Outlier Handler
5. Missing Data Imputer
6. Paper Trading Validator
7. Regime Performance Tracker
8. Stress Testing Framework
9. Validation Pipeline
"""

import pytest
from datetime import datetime, timedelta
from typing import Dict, List

import numpy as np
import polars as pl

from src.cloud.training.validation import (
    MandatoryOOSValidator,
    RobustOverfittingDetector,
    AutomatedDataValidator,
    OutlierDetector,
    OutlierHandler,
    MissingDataImputer,
    ExtendedPaperTradingValidator,
    RegimePerformanceTracker,
    StressTestingFramework,
    ValidationPipeline,
)
from src.cloud.engine.walk_forward import WalkForwardResults, WalkForwardWindow
from src.cloud.training.config.settings import EngineSettings


class TestMandatoryOOSValidator:
    """Tests for MandatoryOOSValidator."""

    def test_validation_passes(self):
        """Test validation passes with good metrics."""
        validator = MandatoryOOSValidator(
            min_oos_sharpe=1.0,
            min_oos_win_rate=0.55,
            max_train_test_gap=0.3,
            max_sharpe_std=0.2,
            min_test_trades=100,
        )

        wf_results = WalkForwardResults(
            windows=[],
            total_windows=5,
            test_sharpe=1.5,
            test_win_rate=0.65,
            test_avg_pnl_bps=50.0,
            sharpe_std=0.15,
            win_rate_std=0.05,
            train_test_sharpe_diff=0.2,
            train_test_wr_diff=0.05,
        )

        result = validator.validate(wf_results, model_id="test_model", total_test_trades=150)
        assert result.passed
        assert len(result.blocking_issues) == 0

    def test_validation_fails_low_sharpe(self):
        """Test validation fails with low Sharpe."""
        validator = MandatoryOOSValidator(min_oos_sharpe=1.0)

        wf_results = WalkForwardResults(
            windows=[],
            total_windows=5,
            test_sharpe=0.8,  # Below threshold
            test_win_rate=0.65,
            test_avg_pnl_bps=50.0,
            sharpe_std=0.15,
            win_rate_std=0.05,
            train_test_sharpe_diff=0.2,
            train_test_wr_diff=0.05,
        )

        with pytest.raises(ValueError, match="FAILED mandatory OOS validation"):
            validator.validate(wf_results, model_id="test_model", total_test_trades=150)


class TestRobustOverfittingDetector:
    """Tests for RobustOverfittingDetector."""

    def test_detects_overfitting(self):
        """Test overfitting detection."""
        detector = RobustOverfittingDetector()

        report = detector.detect_overfitting(
            train_sharpe=2.5,
            test_sharpe=1.2,  # Large gap
            train_win_rate=0.85,
            test_win_rate=0.62,
            cv_sharpe_std=0.35,
        )

        assert report.is_overfitting
        assert report.confidence > 0.5

    def test_no_overfitting(self):
        """Test no overfitting detected."""
        detector = RobustOverfittingDetector()

        report = detector.detect_overfitting(
            train_sharpe=1.3,
            test_sharpe=1.2,  # Small gap
            train_win_rate=0.65,
            test_win_rate=0.62,
            cv_sharpe_std=0.15,
        )

        assert not report.is_overfitting


class TestAutomatedDataValidator:
    """Tests for AutomatedDataValidator."""

    def test_validation_passes(self):
        """Test validation passes with good data."""
        validator = AutomatedDataValidator()

        data = pl.DataFrame({
            'timestamp': [datetime.now() - timedelta(hours=i) for i in range(100)],
            'open': [100.0 + i * 0.1 for i in range(100)],
            'high': [101.0 + i * 0.1 for i in range(100)],
            'low': [99.0 + i * 0.1 for i in range(100)],
            'close': [100.5 + i * 0.1 for i in range(100)],
            'volume': [1000.0 for _ in range(100)],
        })

        report = validator.validate(
            data=data,
            symbol="BTC/USDT",
            expected_columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'],
        )

        assert report.passed
        assert report.quality_score > 0.8

    def test_validation_fails_missing_columns(self):
        """Test validation fails with missing columns."""
        validator = AutomatedDataValidator()

        data = pl.DataFrame({
            'timestamp': [datetime.now()],
            'open': [100.0],
            # Missing 'high', 'low', 'close', 'volume'
        })

        report = validator.validate(
            data=data,
            symbol="BTC/USDT",
            expected_columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'],
        )

        assert not report.passed


class TestOutlierDetector:
    """Tests for OutlierDetector."""

    def test_detects_outliers(self):
        """Test outlier detection."""
        detector = OutlierDetector(z_threshold=3.0)

        data = pl.DataFrame({
            'open': [100.0] * 98 + [200.0, 300.0],  # Two outliers
            'close': [100.0] * 100,
        })

        detections = detector.detect_all(data, symbol="BTC/USDT")

        assert len(detections) > 0
        assert any(d.outlier_count > 0 for d in detections)


class TestMissingDataImputer:
    """Tests for MissingDataImputer."""

    def test_imputes_missing_data(self):
        """Test missing data imputation."""
        imputer = MissingDataImputer(default_method="forward_fill")

        data = pl.DataFrame({
            'price': [100.0, None, None, 103.0, None, 105.0],
        })

        report = imputer.impute(data, symbol="BTC/USDT", method="forward_fill")

        assert report.total_imputed > 0
        assert report.quality_score > 0.0


class TestExtendedPaperTradingValidator:
    """Tests for ExtendedPaperTradingValidator."""

    def test_validation_passes(self):
        """Test validation passes with good paper trading results."""
        validator = ExtendedPaperTradingValidator(
            min_duration_days=14,
            min_trades=100,
            min_win_rate=0.55,
            min_sharpe=1.0,
        )

        paper_trades = [
            {
                'entry_timestamp': datetime.now() - timedelta(days=15) + timedelta(hours=i),
                'is_winner': i % 3 != 0,  # ~67% win rate
                'net_profit_bps': 20.0 if i % 3 != 0 else -10.0,
            }
            for i in range(150)
        ]

        backtest_results = {
            'test_win_rate': 0.65,
            'test_sharpe': 1.2,
            'test_avg_pnl_bps': 50.0,
        }

        result = validator.validate(
            paper_trades=paper_trades,
            backtest_results=backtest_results,
            model_id="test_model",
        )

        assert result.passed

    def test_validation_fails_insufficient_trades(self):
        """Test validation fails with insufficient trades."""
        validator = ExtendedPaperTradingValidator(min_trades=100)

        paper_trades = [
            {
                'entry_timestamp': datetime.now() - timedelta(days=15) + timedelta(hours=i),
                'is_winner': True,
                'net_profit_bps': 20.0,
            }
            for i in range(50)  # Only 50 trades
        ]

        with pytest.raises(ValueError, match="FAILED extended paper trading validation"):
            validator.validate(
                paper_trades=paper_trades,
                backtest_results=None,
                model_id="test_model",
            )


class TestRegimePerformanceTracker:
    """Tests for RegimePerformanceTracker."""

    def test_tracks_regime_performance(self):
        """Test regime performance tracking."""
        tracker = RegimePerformanceTracker()

        trades = [
            {
                'market_regime': 'TREND',
                'is_winner': True,
                'net_profit_bps': 20.0,
                'hold_duration_minutes': 60,
            },
            {
                'market_regime': 'RANGE',
                'is_winner': False,
                'net_profit_bps': -10.0,
                'hold_duration_minutes': 30,
            },
        ] * 50

        report = tracker.track_performance(trades=trades, model_id="test_model")

        assert 'TREND' in report.regime_performance
        assert 'RANGE' in report.regime_performance


class TestStressTestingFramework:
    """Tests for StressTestingFramework."""

    def test_stress_tests_pass(self):
        """Test stress tests pass (mock implementation)."""
        framework = StressTestingFramework()

        # Mock model and data
        class MockModel:
            pass

        class MockData:
            pass

        result = framework.run_stress_tests(
            model=MockModel(),
            historical_data=MockData(),
            model_id="test_model",
        )

        # Note: Current implementation uses placeholders
        # In real implementation, would test actual stress scenarios
        assert isinstance(result, type(framework.run_stress_tests.__annotations__['return']))


class TestValidationPipeline:
    """Tests for ValidationPipeline."""

    def test_pipeline_passes(self):
        """Test validation pipeline passes."""
        settings = EngineSettings.load()
        pipeline = ValidationPipeline(settings=settings)

        wf_results = WalkForwardResults(
            windows=[],
            total_windows=5,
            test_sharpe=1.5,
            test_win_rate=0.65,
            test_avg_pnl_bps=50.0,
            sharpe_std=0.15,
            win_rate_std=0.05,
            train_test_sharpe_diff=0.2,
            train_test_wr_diff=0.05,
        )

        data = pl.DataFrame({
            'timestamp': [datetime.now() - timedelta(hours=i) for i in range(100)],
            'open': [100.0 + i * 0.1 for i in range(100)],
            'high': [101.0 + i * 0.1 for i in range(100)],
            'low': [99.0 + i * 0.1 for i in range(100)],
            'close': [100.5 + i * 0.1 for i in range(100)],
            'volume': [1000.0 for _ in range(100)],
        })

        result = pipeline.validate(
            walk_forward_results=wf_results,
            model_id="test_model",
            data=data,
            symbol="BTC/USDT",
        )

        assert result.passed

    def test_pipeline_fails_on_validation_error(self):
        """Test pipeline fails when validation raises error."""
        settings = EngineSettings.load()
        pipeline = ValidationPipeline(settings=settings)

        wf_results = WalkForwardResults(
            windows=[],
            total_windows=5,
            test_sharpe=0.5,  # Below threshold
            test_win_rate=0.45,  # Below threshold
            test_avg_pnl_bps=10.0,
            sharpe_std=0.15,
            win_rate_std=0.05,
            train_test_sharpe_diff=0.2,
            train_test_wr_diff=0.05,
        )

        with pytest.raises(ValueError, match="FAILED validation pipeline"):
            pipeline.validate(
                walk_forward_results=wf_results,
                model_id="test_model",
                symbol="BTC/USDT",
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

