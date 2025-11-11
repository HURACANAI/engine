"""
Unit Tests for Configuration System

Tests Pydantic configuration validation and loading.
"""

import os
import pytest
import tempfile
from pathlib import Path
from pydantic import ValidationError

from src.shared.config.schema import (
    HuracanConfig,
    GeneralConfig,
    EngineConfig,
    MechanicConfig,
    DatabaseConfig,
    ModelType,
    EncoderType,
    SchedulerMode,
)
from src.shared.config.loader import (
    resolve_env_vars,
    load_yaml_config,
    load_config,
)
from src.shared.exceptions import ConfigurationError


class TestGeneralConfig:
    """Tests for GeneralConfig."""

    def test_default_values(self):
        """Test GeneralConfig with defaults."""
        config = GeneralConfig()
        assert config.version == "2.0"
        assert config.timezone == "UTC"
        assert config.dropbox_root == "/Huracan/"
        assert config.s3_bucket == "huracan"
        assert config.symbols == ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

    def test_custom_values(self):
        """Test GeneralConfig with custom values."""
        config = GeneralConfig(
            version="3.0",
            timezone="America/New_York",
            symbols=["BTCUSDT"]
        )
        assert config.version == "3.0"
        assert config.timezone == "America/New_York"
        assert config.symbols == ["BTCUSDT"]


class TestEngineConfig:
    """Tests for EngineConfig."""

    def test_default_values(self):
        """Test EngineConfig with defaults."""
        config = EngineConfig()
        assert config.lookback_days == 180
        assert config.model_type == ModelType.XGBOOST
        assert config.parallel_tasks == 8

    def test_model_type_enum(self):
        """Test ModelType enum."""
        config = EngineConfig(model_type=ModelType.LIGHTGBM)
        assert config.model_type == ModelType.LIGHTGBM
        assert config.model_type.value == "lightgbm"

    def test_validation_lookback_days(self):
        """Test lookback_days validation."""
        # Valid range
        config = EngineConfig(lookback_days=100)
        assert config.lookback_days == 100

        # Invalid: too small
        with pytest.raises(ValidationError):
            EngineConfig(lookback_days=0)

        # Invalid: too large
        with pytest.raises(ValidationError):
            EngineConfig(lookback_days=2000)

    def test_validation_parallel_tasks(self):
        """Test parallel_tasks validation."""
        # Valid range
        config = EngineConfig(parallel_tasks=16)
        assert config.parallel_tasks == 16

        # Invalid: too small
        with pytest.raises(ValidationError):
            EngineConfig(parallel_tasks=0)

        # Invalid: too large
        with pytest.raises(ValidationError):
            EngineConfig(parallel_tasks=200)


class TestMechanicConfig:
    """Tests for MechanicConfig."""

    def test_default_values(self):
        """Test MechanicConfig with defaults."""
        config = MechanicConfig()
        assert config.fine_tune_hours == 6
        assert config.challengers_per_symbol == 3
        assert config.promote_if_net_pnl_above_pct == 1.0

    def test_validation_fine_tune_hours(self):
        """Test fine_tune_hours validation."""
        # Valid range
        config = MechanicConfig(fine_tune_hours=12)
        assert config.fine_tune_hours == 12

        # Invalid: too small
        with pytest.raises(ValidationError):
            MechanicConfig(fine_tune_hours=0)

        # Invalid: too large
        with pytest.raises(ValidationError):
            MechanicConfig(fine_tune_hours=200)


class TestDatabaseConfig:
    """Tests for DatabaseConfig."""

    def test_required_fields(self):
        """Test that connection_string is required."""
        with pytest.raises(ValidationError):
            DatabaseConfig()

    def test_valid_config(self):
        """Test valid DatabaseConfig."""
        config = DatabaseConfig(
            connection_string="postgresql://user:pass@localhost:5432/db"
        )
        assert config.connection_string == "postgresql://user:pass@localhost:5432/db"
        assert config.pool_size == 10
        assert config.max_overflow == 20

    def test_validation_pool_size(self):
        """Test pool_size validation."""
        # Valid range
        config = DatabaseConfig(
            connection_string="postgresql://test",
            pool_size=20
        )
        assert config.pool_size == 20

        # Invalid: too small
        with pytest.raises(ValidationError):
            DatabaseConfig(
                connection_string="postgresql://test",
                pool_size=0
            )

        # Invalid: too large
        with pytest.raises(ValidationError):
            DatabaseConfig(
                connection_string="postgresql://test",
                pool_size=200
            )


class TestEnums:
    """Tests for configuration enums."""

    def test_model_type_enum(self):
        """Test ModelType enum values."""
        assert ModelType.XGBOOST.value == "xgboost"
        assert ModelType.LIGHTGBM.value == "lightgbm"
        assert ModelType.CATBOOST.value == "catboost"
        assert ModelType.RANDOM_FOREST.value == "random_forest"

    def test_encoder_type_enum(self):
        """Test EncoderType enum values."""
        assert EncoderType.PCA.value == "pca"
        assert EncoderType.AUTOENCODER.value == "autoencoder"

    def test_scheduler_mode_enum(self):
        """Test SchedulerMode enum values."""
        assert SchedulerMode.SEQUENTIAL.value == "sequential"
        assert SchedulerMode.PARALLEL.value == "parallel"
        assert SchedulerMode.HYBRID.value == "hybrid"


class TestResolveEnvVars:
    """Tests for resolve_env_vars function."""

    def test_simple_string(self):
        """Test resolving simple environment variable."""
        os.environ["TEST_VAR"] = "test_value"
        config = {"key": "${TEST_VAR}"}
        result = resolve_env_vars(config)
        assert result["key"] == "test_value"
        del os.environ["TEST_VAR"]

    def test_multiple_vars(self):
        """Test resolving multiple environment variables."""
        os.environ["VAR1"] = "value1"
        os.environ["VAR2"] = "value2"
        config = {
            "key1": "${VAR1}",
            "key2": "${VAR2}"
        }
        result = resolve_env_vars(config)
        assert result["key1"] == "value1"
        assert result["key2"] == "value2"
        del os.environ["VAR1"]
        del os.environ["VAR2"]

    def test_nested_dict(self):
        """Test resolving environment variables in nested dict."""
        os.environ["NESTED_VAR"] = "nested_value"
        config = {
            "outer": {
                "inner": "${NESTED_VAR}"
            }
        }
        result = resolve_env_vars(config)
        assert result["outer"]["inner"] == "nested_value"
        del os.environ["NESTED_VAR"]

    def test_list_values(self):
        """Test resolving environment variables in lists."""
        os.environ["LIST_VAR"] = "list_value"
        config = {
            "items": ["${LIST_VAR}", "static"]
        }
        result = resolve_env_vars(config)
        assert result["items"][0] == "list_value"
        assert result["items"][1] == "static"
        del os.environ["LIST_VAR"]

    def test_missing_env_var(self):
        """Test handling of missing environment variable."""
        config = {"key": "${MISSING_VAR}"}
        result = resolve_env_vars(config)
        # Should keep placeholder if variable not found
        assert result["key"] == "${MISSING_VAR}"

    def test_partial_replacement(self):
        """Test partial string replacement."""
        os.environ["HOST"] = "localhost"
        os.environ["PORT"] = "5432"
        config = {"url": "postgresql://${HOST}:${PORT}/db"}
        result = resolve_env_vars(config)
        assert result["url"] == "postgresql://localhost:5432/db"
        del os.environ["HOST"]
        del os.environ["PORT"]


class TestLoadYamlConfig:
    """Tests for load_yaml_config function."""

    def test_load_valid_yaml(self):
        """Test loading valid YAML file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("key: value\nnumber: 42\n")
            temp_path = f.name

        try:
            config = load_yaml_config(Path(temp_path))
            assert config["key"] == "value"
            assert config["number"] == 42
        finally:
            os.unlink(temp_path)

    def test_file_not_found(self):
        """Test loading non-existent file."""
        with pytest.raises(ConfigurationError) as exc_info:
            load_yaml_config(Path("/nonexistent/config.yaml"))
        assert "Configuration file not found" in str(exc_info.value)

    def test_invalid_yaml(self):
        """Test loading invalid YAML."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: syntax:")
            temp_path = f.name

        try:
            with pytest.raises(ConfigurationError) as exc_info:
                load_yaml_config(Path(temp_path))
            assert "Invalid YAML" in str(exc_info.value)
        finally:
            os.unlink(temp_path)


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_minimal_config(self):
        """Test loading minimal valid configuration."""
        yaml_content = """
general:
  symbols: ["BTCUSDT"]
engine:
  lookback_days: 180
mechanic:
  fine_tune_hours: 6
hamilton:
  edge_threshold_bps: 10
costs:
  taker_fee_bps: 4.0
regime_classifier:
  trend_threshold: 0.6
database:
  connection_string: "postgresql://localhost/test"
s3:
  bucket: "test"
  access_key: "test_key"
  secret_key: "test_secret"
telegram:
  token: "test_token"
  chat_id: "test_chat"
scheduler:
  mode: "hybrid"
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            config = load_config(Path(temp_path))
            assert isinstance(config, HuracanConfig)
            assert config.engine.lookback_days == 180
            assert config.database.connection_string == "postgresql://localhost/test"
        finally:
            os.unlink(temp_path)

    def test_load_config_with_validation_errors(self):
        """Test loading configuration with validation errors."""
        yaml_content = """
general:
  symbols: []
engine:
  lookback_days: -1
database:
  connection_string: "postgresql://localhost/test"
s3:
  bucket: "test"
  access_key: "key"
  secret_key: "secret"
telegram:
  token: "token"
  chat_id: "chat"
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            with pytest.raises(ConfigurationError) as exc_info:
                load_config(Path(temp_path))
            assert "Configuration validation failed" in str(exc_info.value)
        finally:
            os.unlink(temp_path)


class TestHuracanConfigValidation:
    """Tests for HuracanConfig validation."""

    def test_validate_general_symbols(self):
        """Test general.symbols validation."""
        # Empty symbols allowed in GeneralConfig directly
        config = GeneralConfig(symbols=[])
        assert config.symbols == []

    def test_validate_engine_symbol_counts(self):
        """Test engine symbol count validation."""
        # Invalid counts allowed in EngineConfig directly
        config = EngineConfig(
            start_with_symbols=200,
            target_symbols=100
        )
        assert config.start_with_symbols == 200
        assert config.target_symbols == 100

    def test_to_dict(self):
        """Test HuracanConfig.to_dict()."""
        yaml_content = """
general:
  symbols: ["BTCUSDT"]
engine:
  lookback_days: 180
mechanic:
  fine_tune_hours: 6
hamilton:
  edge_threshold_bps: 10
costs:
  taker_fee_bps: 4.0
regime_classifier:
  trend_threshold: 0.6
database:
  connection_string: "postgresql://localhost/test"
s3:
  bucket: "test"
  access_key: "key"
  secret_key: "secret"
telegram:
  token: "token"
  chat_id: "chat"
scheduler:
  mode: "hybrid"
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            config = load_config(Path(temp_path))
            config_dict = config.to_dict()

            assert isinstance(config_dict, dict)
            assert config_dict["engine"]["lookback_days"] == 180
            assert config_dict["database"]["connection_string"] == "postgresql://localhost/test"
        finally:
            os.unlink(temp_path)
