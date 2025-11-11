"""
Tests for Walk-Forward Purged Cross-Validation

Example test suite for walk-forward validation.
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta, timezone

from src.cloud.training.validation.walk_forward_purged import (
    WalkForwardPurgedCV,
    WalkForwardConfig,
)


@pytest.fixture
def config():
    """Test configuration."""
    return WalkForwardConfig(
        train_days=20,
        test_days=5,
        purge_days=2,
        min_windows=3,
        min_test_trades=10,
    )


@pytest.fixture
def validator(config):
    """Test validator."""
    return WalkForwardPurgedCV(config=config)


@pytest.fixture
def sample_data():
    """Sample time series data."""
    start_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
    dates = [start_date + timedelta(days=i) for i in range(100)]
    
    return pd.DataFrame({
        "timestamp": dates,
        "close": [100 + i * 0.1 for i in range(100)],
        "volume": [1000 + i * 10 for i in range(100)],
    })


def test_generate_windows(validator):
    """Test window generation."""
    data_start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    data_end = datetime(2024, 4, 1, tzinfo=timezone.utc)
    
    windows = validator.generate_windows(data_start, data_end)
    
    assert len(windows) >= 3  # At least min_windows
    
    # Check first window
    first = windows[0]
    assert first.train_start == data_start
    assert first.train_end == first.train_start + timedelta(days=20)
    assert first.purge_end == first.test_start
    assert first.test_end == first.test_start + timedelta(days=5)
    
    # Check no overlap between windows
    for i in range(len(windows) - 1):
        assert windows[i].test_end <= windows[i + 1].train_start


def test_split_data(validator, sample_data):
    """Test data splitting."""
    data_start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    data_end = datetime(2024, 4, 1, tzinfo=timezone.utc)
    
    windows = validator.generate_windows(data_start, data_end)
    window = windows[0]
    
    train_data, test_data = validator.split_data(sample_data, window)
    
    assert len(train_data) > 0
    assert len(test_data) > 0
    
    # Check no overlap
    if len(train_data) > 0 and len(test_data) > 0:
        max_train = train_data["timestamp"].max()
        min_test = test_data["timestamp"].min()
        assert min_test > max_train


def test_leakage_detection(validator):
    """Test leakage detection."""
    # Similar metrics - no leakage
    train_metrics = {"sharpe_ratio": 1.5, "hit_rate": 0.55}
    test_metrics = {"sharpe_ratio": 1.4, "hit_rate": 0.54}
    
    detected, score = validator.detect_leakage(train_metrics, test_metrics)
    assert not detected
    assert score < 0.3
    
    # Very different metrics - leakage
    train_metrics = {"sharpe_ratio": 2.0, "hit_rate": 0.70}
    test_metrics = {"sharpe_ratio": 0.5, "hit_rate": 0.45}
    
    detected, score = validator.detect_leakage(train_metrics, test_metrics)
    assert detected
    assert score > 0.3

