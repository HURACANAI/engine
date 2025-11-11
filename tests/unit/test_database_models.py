"""
Unit Tests for Database Models

Tests the database client and all save operations.
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, MagicMock, patch, call
from sqlalchemy.exc import SQLAlchemyError

from src.shared.database.models import (
    DatabaseClient,
    ModelRecord,
    ModelMetrics,
    Promotion,
    LiveTrade,
    DailyEquity,
)


@pytest.fixture
def mock_engine():
    """Create a mock SQLAlchemy engine."""
    engine = MagicMock()
    connection = MagicMock()
    engine.begin.return_value.__enter__.return_value = connection
    engine.begin.return_value.__exit__.return_value = None
    return engine


@pytest.fixture
def db_client(mock_engine):
    """Create a database client with mocked engine."""
    with patch('src.shared.database.models.create_engine', return_value=mock_engine):
        client = DatabaseClient("postgresql://test:test@localhost:5432/test")
        client._engine = mock_engine
    return client


class TestModelRecord:
    """Tests for ModelRecord dataclass."""

    def test_to_dict(self):
        """Test ModelRecord.to_dict()."""
        record = ModelRecord(
            model_id="test_model_001",
            parent_id="parent_001",
            kind="baseline",
            created_at=datetime(2025, 1, 1, 12, 0, 0),
            s3_path="s3://bucket/models/test_model_001.pkl",
            features_used=["rsi", "ema", "volatility"],
            params={"max_depth": 6, "n_estimators": 100},
        )

        result = record.to_dict()

        assert result["model_id"] == "test_model_001"
        assert result["parent_id"] == "parent_001"
        assert result["kind"] == "baseline"
        assert result["created_at"] == "2025-01-01T12:00:00"
        assert result["s3_path"] == "s3://bucket/models/test_model_001.pkl"
        assert result["features_used"] == ["rsi", "ema", "volatility"]
        assert result["params"] == {"max_depth": 6, "n_estimators": 100}


class TestModelMetrics:
    """Tests for ModelMetrics dataclass."""

    def test_to_dict(self):
        """Test ModelMetrics.to_dict()."""
        metrics = ModelMetrics(
            model_id="test_model_001",
            sharpe=2.5,
            hit_rate=0.58,
            drawdown=0.15,
            net_bps=45.0,
            window="test",
            cost_bps=8.0,
            promoted=False,
        )

        result = metrics.to_dict()

        assert result["model_id"] == "test_model_001"
        assert result["sharpe"] == 2.5
        assert result["hit_rate"] == 0.58
        assert result["drawdown"] == 0.15
        assert result["net_bps"] == 45.0
        assert result["window"] == "test"
        assert result["cost_bps"] == 8.0
        assert result["promoted"] is False


class TestDatabaseClient:
    """Tests for DatabaseClient."""

    def test_init_success(self, mock_engine):
        """Test successful database client initialization."""
        with patch('src.shared.database.models.create_engine', return_value=mock_engine):
            client = DatabaseClient("postgresql://user:pass@host:5432/db")
            assert client.connection_string == "postgresql://user:pass@host:5432/db"
            assert client._engine == mock_engine

    def test_init_failure(self):
        """Test database client initialization failure."""
        with patch('src.shared.database.models.create_engine', side_effect=SQLAlchemyError("Connection failed")):
            with pytest.raises(ConnectionError) as exc_info:
                DatabaseClient("postgresql://bad:connection@nowhere:5432/db")
            assert "Failed to connect to database" in str(exc_info.value)

    def test_save_model_success(self, db_client, mock_engine):
        """Test successful model save."""
        model = ModelRecord(
            model_id="test_model_001",
            parent_id=None,
            kind="baseline",
            created_at=datetime(2025, 1, 1),
            s3_path="s3://bucket/model.pkl",
            features_used=["rsi"],
            params={"depth": 6},
        )

        result = db_client.save_model(model)

        assert result is True
        # Verify execute was called
        mock_engine.begin.return_value.__enter__.return_value.execute.assert_called_once()

    def test_save_model_failure(self, db_client, mock_engine):
        """Test model save failure."""
        # Make execute raise an error
        mock_engine.begin.return_value.__enter__.return_value.execute.side_effect = SQLAlchemyError("Save failed")

        model = ModelRecord(
            model_id="test_model_001",
            parent_id=None,
            kind="baseline",
            created_at=datetime(2025, 1, 1),
            s3_path="s3://bucket/model.pkl",
            features_used=["rsi"],
            params={},
        )

        with pytest.raises(RuntimeError) as exc_info:
            db_client.save_model(model)
        assert "Failed to save model test_model_001" in str(exc_info.value)

    def test_save_metrics_success(self, db_client, mock_engine):
        """Test successful metrics save."""
        metrics = ModelMetrics(
            model_id="test_model_001",
            sharpe=2.5,
            hit_rate=0.58,
            drawdown=0.15,
            net_bps=45.0,
            window="test",
            cost_bps=8.0,
            promoted=False,
        )

        result = db_client.save_metrics(metrics)

        assert result is True
        mock_engine.begin.return_value.__enter__.return_value.execute.assert_called_once()

    def test_save_metrics_failure(self, db_client, mock_engine):
        """Test metrics save failure."""
        mock_engine.begin.return_value.__enter__.return_value.execute.side_effect = SQLAlchemyError("Save failed")

        metrics = ModelMetrics(
            model_id="test_model_001",
            sharpe=2.5,
            hit_rate=0.58,
            drawdown=0.15,
            net_bps=45.0,
            window="test",
            cost_bps=8.0,
            promoted=False,
        )

        with pytest.raises(RuntimeError) as exc_info:
            db_client.save_metrics(metrics)
        assert "Failed to save metrics for test_model_001" in str(exc_info.value)

    def test_save_promotion_success(self, db_client, mock_engine):
        """Test successful promotion save."""
        promotion = Promotion(
            from_model_id="model_001",
            to_model_id="model_002",
            reason="Better Sharpe ratio",
            at=datetime(2025, 1, 1),
            snapshot={"sharpe": 2.5, "win_rate": 0.58},
        )

        result = db_client.save_promotion(promotion)

        assert result is True
        mock_engine.begin.return_value.__enter__.return_value.execute.assert_called_once()

    def test_save_live_trade_success(self, db_client, mock_engine):
        """Test successful live trade save."""
        trade = LiveTrade(
            trade_id="trade_001",
            time=datetime(2025, 1, 1, 12, 0, 0),
            symbol="BTCUSDT",
            side="buy",
            size=0.1,
            entry=50000.0,
            exit=50500.0,
            fees=10.0,
            net_pnl=40.0,
            model_id="model_001",
        )

        result = db_client.save_live_trade(trade)

        assert result is True
        mock_engine.begin.return_value.__enter__.return_value.execute.assert_called_once()

    def test_save_live_trade_failure(self, db_client, mock_engine):
        """Test live trade save failure."""
        mock_engine.begin.return_value.__enter__.return_value.execute.side_effect = SQLAlchemyError("Save failed")

        trade = LiveTrade(
            trade_id="trade_001",
            time=datetime(2025, 1, 1),
            symbol="BTCUSDT",
            side="buy",
            size=0.1,
            entry=50000.0,
            exit=50500.0,
            fees=10.0,
            net_pnl=40.0,
            model_id="model_001",
        )

        with pytest.raises(RuntimeError) as exc_info:
            db_client.save_live_trade(trade)
        assert "Failed to save live trade trade_001" in str(exc_info.value)

    def test_save_daily_equity_success(self, db_client, mock_engine):
        """Test successful daily equity save."""
        equity = DailyEquity(
            date="2025-01-01",
            nav=100000.0,
            max_dd=0.05,
            turnover=0.25,
            fees_bps=8.0,
        )

        result = db_client.save_daily_equity(equity)

        assert result is True
        mock_engine.begin.return_value.__enter__.return_value.execute.assert_called_once()

    def test_save_daily_equity_failure(self, db_client, mock_engine):
        """Test daily equity save failure."""
        mock_engine.begin.return_value.__enter__.return_value.execute.side_effect = SQLAlchemyError("Save failed")

        equity = DailyEquity(
            date="2025-01-01",
            nav=100000.0,
            max_dd=0.05,
            turnover=0.25,
            fees_bps=8.0,
        )

        with pytest.raises(RuntimeError) as exc_info:
            db_client.save_daily_equity(equity)
        assert "Failed to save daily equity for 2025-01-01" in str(exc_info.value)


class TestIntegration:
    """Integration tests for database operations."""

    @pytest.mark.integration
    def test_full_workflow(self, db_client, mock_engine):
        """Test complete workflow: save model, metrics, promotion."""
        # Save model
        model = ModelRecord(
            model_id="workflow_model",
            parent_id=None,
            kind="baseline",
            created_at=datetime(2025, 1, 1),
            s3_path="s3://bucket/model.pkl",
            features_used=["rsi", "ema"],
            params={"depth": 6},
        )
        assert db_client.save_model(model) is True

        # Save metrics
        metrics = ModelMetrics(
            model_id="workflow_model",
            sharpe=2.5,
            hit_rate=0.58,
            drawdown=0.15,
            net_bps=45.0,
            window="test",
            cost_bps=8.0,
            promoted=False,
        )
        assert db_client.save_metrics(metrics) is True

        # Save promotion
        promotion = Promotion(
            from_model_id="old_model",
            to_model_id="workflow_model",
            reason="Better performance",
            at=datetime(2025, 1, 1),
            snapshot={"sharpe": 2.5},
        )
        assert db_client.save_promotion(promotion) is True

        # Verify all operations called execute
        assert mock_engine.begin.return_value.__enter__.return_value.execute.call_count == 3
