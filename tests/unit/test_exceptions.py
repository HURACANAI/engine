"""
Unit Tests for Custom Exceptions

Tests the exception hierarchy and context handling.
"""

import pytest
from src.shared.exceptions import (
    HuracanError,
    DatabaseError,
    StorageError,
    DropboxError,
    S3Error,
    DataError,
    DataQualityError,
    DataLoadError,
    ModelError,
    ModelLoadError,
    ModelSaveError,
    TrainingError,
    ValidationError,
    ConfigurationError,
    ExchangeError,
    OrderError,
    TelegramError,
    FeatureError,
    ContractError,
    SerializationError,
)


class TestHuracanError:
    """Tests for base HuracanError."""

    def test_basic_error(self):
        """Test basic error without context."""
        error = HuracanError("Something went wrong")
        assert str(error) == "Something went wrong"
        assert error.context == {}

    def test_error_with_context(self):
        """Test error with context."""
        context = {"file": "test.py", "line": 42}
        error = HuracanError("Error occurred", context=context)

        assert str(error) == "Error occurred"
        assert error.context == context
        assert error.context["file"] == "test.py"
        assert error.context["line"] == 42

    def test_error_inheritance(self):
        """Test that HuracanError is an Exception."""
        error = HuracanError("Test")
        assert isinstance(error, Exception)


class TestDatabaseErrors:
    """Tests for database-related errors."""

    def test_database_error(self):
        """Test DatabaseError."""
        error = DatabaseError("Connection failed", context={"host": "localhost"})
        assert isinstance(error, HuracanError)
        assert str(error) == "Connection failed"
        assert error.context["host"] == "localhost"


class TestStorageErrors:
    """Tests for storage-related errors."""

    def test_storage_error(self):
        """Test base StorageError."""
        error = StorageError("Storage failed")
        assert isinstance(error, HuracanError)

    def test_dropbox_error(self):
        """Test DropboxError."""
        error = DropboxError(
            "Upload failed",
            context={"file": "model.pkl", "path": "/models/"}
        )
        assert isinstance(error, StorageError)
        assert isinstance(error, HuracanError)
        assert error.context["file"] == "model.pkl"
        assert error.context["path"] == "/models/"

    def test_s3_error(self):
        """Test S3Error."""
        error = S3Error(
            "Bucket not found",
            context={"bucket": "huracan", "key": "model.pkl"}
        )
        assert isinstance(error, StorageError)
        assert isinstance(error, HuracanError)


class TestDataErrors:
    """Tests for data-related errors."""

    def test_data_error(self):
        """Test base DataError."""
        error = DataError("Data issue")
        assert isinstance(error, HuracanError)

    def test_data_quality_error(self):
        """Test DataQualityError."""
        error = DataQualityError(
            "Data quality check failed",
            context={
                "checks_failed": ["missing_values", "outliers"],
                "symbol": "BTCUSDT"
            }
        )
        assert isinstance(error, DataError)
        assert len(error.context["checks_failed"]) == 2

    def test_data_load_error(self):
        """Test DataLoadError."""
        error = DataLoadError(
            "Failed to load data",
            context={"source": "s3://bucket/data.parquet"}
        )
        assert isinstance(error, DataError)


class TestModelErrors:
    """Tests for model-related errors."""

    def test_model_error(self):
        """Test base ModelError."""
        error = ModelError("Model issue")
        assert isinstance(error, HuracanError)

    def test_model_load_error(self):
        """Test ModelLoadError."""
        error = ModelLoadError(
            "Failed to load model",
            context={"model_id": "btc_model_001", "path": "s3://bucket/model.pkl"}
        )
        assert isinstance(error, ModelError)
        assert error.context["model_id"] == "btc_model_001"

    def test_model_save_error(self):
        """Test ModelSaveError."""
        error = ModelSaveError(
            "Failed to save model",
            context={"model_id": "btc_model_001"}
        )
        assert isinstance(error, ModelError)


class TestTrainingAndValidationErrors:
    """Tests for training and validation errors."""

    def test_training_error(self):
        """Test TrainingError."""
        error = TrainingError(
            "Training failed",
            context={"epoch": 10, "loss": float('inf')}
        )
        assert isinstance(error, HuracanError)
        assert error.context["epoch"] == 10

    def test_validation_error(self):
        """Test ValidationError."""
        error = ValidationError(
            "Validation failed",
            context={"metric": "sharpe", "threshold": 1.0, "actual": 0.5}
        )
        assert isinstance(error, HuracanError)


class TestConfigurationError:
    """Tests for ConfigurationError."""

    def test_configuration_error(self):
        """Test ConfigurationError."""
        error = ConfigurationError(
            "Invalid configuration",
            context={"field": "engine.lookback_days", "value": -1}
        )
        assert isinstance(error, HuracanError)
        assert error.context["field"] == "engine.lookback_days"


class TestExchangeErrors:
    """Tests for exchange-related errors."""

    def test_exchange_error(self):
        """Test base ExchangeError."""
        error = ExchangeError("Exchange connection failed")
        assert isinstance(error, HuracanError)

    def test_order_error(self):
        """Test OrderError."""
        error = OrderError(
            "Order placement failed",
            context={
                "symbol": "BTCUSDT",
                "side": "buy",
                "quantity": 0.1,
                "reason": "Insufficient funds"
            }
        )
        assert isinstance(error, ExchangeError)
        assert error.context["symbol"] == "BTCUSDT"


class TestTelegramError:
    """Tests for TelegramError."""

    def test_telegram_error(self):
        """Test TelegramError."""
        error = TelegramError(
            "Failed to send message",
            context={"chat_id": "123456", "message": "Test"}
        )
        assert isinstance(error, HuracanError)


class TestFeatureError:
    """Tests for FeatureError."""

    def test_feature_error(self):
        """Test FeatureError."""
        error = FeatureError(
            "Feature computation failed",
            context={"feature": "rsi_14", "reason": "Missing data"}
        )
        assert isinstance(error, HuracanError)


class TestContractError:
    """Tests for ContractError."""

    def test_contract_error(self):
        """Test ContractError."""
        error = ContractError(
            "Contract validation failed",
            context={"contract_type": "RunManifest"}
        )
        assert isinstance(error, HuracanError)


class TestSerializationError:
    """Tests for SerializationError."""

    def test_serialization_error(self):
        """Test SerializationError."""
        error = SerializationError(
            "Failed to serialize object",
            context={"object_type": "ModelRecord", "format": "json"}
        )
        assert isinstance(error, HuracanError)


class TestErrorChaining:
    """Tests for error chaining with 'from'."""

    def test_error_chaining(self):
        """Test exception chaining."""
        try:
            try:
                raise ValueError("Original error")
            except ValueError as e:
                raise ModelLoadError(
                    "Failed to load model",
                    context={"model_id": "test"}
                ) from e
        except ModelLoadError as caught:
            assert isinstance(caught, ModelLoadError)
            assert isinstance(caught.__cause__, ValueError)
            assert str(caught.__cause__) == "Original error"


class TestErrorRaising:
    """Tests for raising and catching errors."""

    def test_catch_specific_error(self):
        """Test catching specific error type."""
        with pytest.raises(DropboxError) as exc_info:
            raise DropboxError("Test error", context={"file": "test.txt"})

        assert "Test error" in str(exc_info.value)
        assert exc_info.value.context["file"] == "test.txt"

    def test_catch_base_error(self):
        """Test catching via base HuracanError."""
        with pytest.raises(HuracanError):
            raise DataQualityError("Data issue")

    def test_multiple_error_types(self):
        """Test catching multiple error types."""
        def risky_operation(error_type: str):
            if error_type == "dropbox":
                raise DropboxError("Dropbox failed")
            elif error_type == "s3":
                raise S3Error("S3 failed")
            elif error_type == "data":
                raise DataLoadError("Data load failed")

        with pytest.raises(StorageError):
            risky_operation("dropbox")

        with pytest.raises(StorageError):
            risky_operation("s3")

        with pytest.raises(DataError):
            risky_operation("data")


class TestContextManipulation:
    """Tests for context manipulation."""

    def test_empty_context(self):
        """Test error with empty context."""
        error = HuracanError("Error")
        assert error.context == {}

    def test_none_context(self):
        """Test error with None context."""
        error = HuracanError("Error", context=None)
        assert error.context == {}

    def test_modify_context(self):
        """Test modifying context after creation."""
        error = HuracanError("Error", context={"initial": "value"})
        error.context["added"] = "new_value"

        assert error.context["initial"] == "value"
        assert error.context["added"] == "new_value"

    def test_complex_context(self):
        """Test error with complex context data."""
        context = {
            "symbol": "BTCUSDT",
            "metrics": {
                "sharpe": 2.5,
                "win_rate": 0.58
            },
            "features": ["rsi", "ema", "volatility"],
            "metadata": {
                "version": "1.0",
                "created_at": "2025-01-01"
            }
        }
        error = ModelError("Model failed", context=context)

        assert error.context["symbol"] == "BTCUSDT"
        assert error.context["metrics"]["sharpe"] == 2.5
        assert len(error.context["features"]) == 3
        assert error.context["metadata"]["version"] == "1.0"
