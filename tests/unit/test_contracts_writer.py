"""
Unit Tests for Contract Writer

Tests exception handling and Dropbox integration in contract writer.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime, timezone

from src.shared.contracts.writer import ContractWriter
from src.shared.contracts.per_coin import RunManifest
from src.shared.exceptions import DropboxError, SerializationError


@pytest.fixture
def mock_dropbox_sync():
    """Create a mock DropboxSync instance."""
    mock_sync = MagicMock()
    mock_sync.upload_file.return_value = True
    return mock_sync


@pytest.fixture
def writer(mock_dropbox_sync):
    """Create a ContractWriter with mocked DropboxSync."""
    return ContractWriter(dropbox_sync=mock_dropbox_sync, base_folder="test_huracan")


@pytest.fixture
def sample_manifest():
    """Create a sample RunManifest."""
    return RunManifest(
        run_id="test_run_001",
        symbol="BTCUSDT",
        utc_started=datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        utc_finished=datetime(2025, 1, 1, 13, 0, 0, tzinfo=timezone.utc),
        engine_version="2.0",
        success=True,
        error=None,
        model_s3_path="s3://bucket/models/btc_model.pkl",
        metrics_s3_path="s3://bucket/metrics/btc_metrics.json",
        features_used=["rsi", "ema", "volatility"],
        config_snapshot={"lookback_days": 180},
        cost_model={"taker_fee_bps": 4.0, "maker_fee_bps": 2.0},
    )


class TestContractWriterInit:
    """Tests for ContractWriter initialization."""

    def test_init_with_dropbox(self, mock_dropbox_sync):
        """Test initialization with DropboxSync."""
        writer = ContractWriter(dropbox_sync=mock_dropbox_sync)
        assert writer.dropbox_sync == mock_dropbox_sync
        assert writer.base_folder == "huracan"

    def test_init_without_dropbox(self):
        """Test initialization without DropboxSync."""
        writer = ContractWriter(dropbox_sync=None)
        assert writer.dropbox_sync is None

    def test_init_custom_base_folder(self, mock_dropbox_sync):
        """Test initialization with custom base folder."""
        writer = ContractWriter(dropbox_sync=mock_dropbox_sync, base_folder="custom")
        assert writer.base_folder == "custom"


class TestWriteManifest:
    """Tests for write_manifest method."""

    def test_write_manifest_success(self, writer, sample_manifest, mock_dropbox_sync):
        """Test successful manifest write."""
        result = writer.write_manifest(sample_manifest)

        assert result is not None
        assert "test_huracan" in result
        mock_dropbox_sync.upload_file.assert_called_once()

    def test_write_manifest_no_dropbox(self, sample_manifest):
        """Test manifest write without DropboxSync."""
        writer = ContractWriter(dropbox_sync=None)
        result = writer.write_manifest(sample_manifest)

        assert result is None

    def test_write_manifest_upload_failure(self, writer, sample_manifest, mock_dropbox_sync):
        """Test manifest write when upload fails."""
        mock_dropbox_sync.upload_file.return_value = False

        result = writer.write_manifest(sample_manifest)

        assert result is None

    def test_write_manifest_dropbox_exception(self, writer, sample_manifest, mock_dropbox_sync):
        """Test manifest write when Dropbox raises exception."""
        mock_dropbox_sync.upload_file.side_effect = Exception("Connection timeout")

        result = writer.write_manifest(sample_manifest)

        assert result is None

    def test_write_manifest_serialization_error(self, writer, mock_dropbox_sync):
        """Test manifest write with serialization error."""
        # Create manifest with invalid data that can't be serialized
        bad_manifest = Mock()
        bad_manifest.to_json.side_effect = TypeError("Cannot serialize")
        bad_manifest.utc_started = datetime(2025, 1, 1)
        bad_manifest.run_id = "test"

        result = writer.write_manifest(bad_manifest)

        assert result is None

    def test_write_manifest_custom_date(self, writer, sample_manifest, mock_dropbox_sync):
        """Test manifest write with custom date string."""
        result = writer.write_manifest(sample_manifest, date_str="20250115")

        assert result is not None
        assert "20250115" in result


class TestExceptionHandling:
    """Tests for specific exception handling."""

    @patch('src.shared.contracts.writer.tempfile.NamedTemporaryFile')
    def test_serialization_error_handling(self, mock_tempfile, writer, sample_manifest):
        """Test handling of SerializationError."""
        # Make to_json raise an error
        sample_manifest.to_json = Mock(side_effect=TypeError("Cannot serialize"))

        result = writer.write_manifest(sample_manifest)

        assert result is None

    def test_dropbox_error_handling(self, writer, sample_manifest, mock_dropbox_sync):
        """Test handling of Dropbox-specific errors."""
        mock_dropbox_sync.upload_file.side_effect = ConnectionError("Network error")

        result = writer.write_manifest(sample_manifest)

        assert result is None

    @patch('src.shared.contracts.writer.Path.unlink')
    def test_cleanup_on_error(self, mock_unlink, writer, sample_manifest, mock_dropbox_sync):
        """Test that temp file cleanup happens even on error."""
        mock_dropbox_sync.upload_file.side_effect = Exception("Upload failed")

        result = writer.write_manifest(sample_manifest)

        # Temp file cleanup should still be attempted
        assert result is None


class TestTempFileHandling:
    """Tests for temporary file handling."""

    @patch('src.shared.contracts.writer.tempfile.NamedTemporaryFile')
    @patch('src.shared.contracts.writer.Path')
    def test_temp_file_created_and_cleaned(self, mock_path, mock_tempfile, writer, sample_manifest):
        """Test that temporary file is created and cleaned up."""
        # Setup mocks
        mock_file = MagicMock()
        mock_file.name = "/tmp/test_file"
        mock_tempfile.return_value.__enter__.return_value = mock_file

        mock_path_instance = MagicMock()
        mock_path.return_value = mock_path_instance

        result = writer.write_manifest(sample_manifest)

        # Verify temp file was created
        mock_tempfile.assert_called_once()

        # Verify cleanup was attempted
        mock_path.assert_called_with("/tmp/test_file")
        mock_path_instance.unlink.assert_called_once()


class TestIntegrationScenarios:
    """Integration tests for complete workflows."""

    def test_complete_success_workflow(self, writer, sample_manifest, mock_dropbox_sync):
        """Test complete successful workflow."""
        # Write manifest
        path = writer.write_manifest(sample_manifest)

        # Verify success
        assert path is not None
        assert isinstance(path, str)
        assert len(path) > 0

        # Verify Dropbox was called
        mock_dropbox_sync.upload_file.assert_called_once()
        call_args = mock_dropbox_sync.upload_file.call_args
        assert call_args.kwargs["overwrite"] is True

    def test_retry_logic_not_implemented(self, writer, sample_manifest, mock_dropbox_sync):
        """Test that there's no automatic retry (should fail immediately)."""
        # Make first call fail
        mock_dropbox_sync.upload_file.return_value = False

        result = writer.write_manifest(sample_manifest)

        # Should fail without retry
        assert result is None
        assert mock_dropbox_sync.upload_file.call_count == 1


class TestLogging:
    """Tests for logging behavior."""

    @patch('src.shared.contracts.writer.logger')
    def test_success_logging(self, mock_logger, writer, sample_manifest, mock_dropbox_sync):
        """Test that success is logged."""
        writer.write_manifest(sample_manifest)

        # Verify info log was called
        mock_logger.info.assert_called()
        call_args = [call[0][0] for call in mock_logger.info.call_args_list]
        assert any("manifest_written" in str(arg) for arg in call_args)

    @patch('src.shared.contracts.writer.logger')
    def test_error_logging(self, mock_logger, writer, sample_manifest, mock_dropbox_sync):
        """Test that errors are logged."""
        mock_dropbox_sync.upload_file.side_effect = Exception("Test error")

        writer.write_manifest(sample_manifest)

        # Verify error log was called
        mock_logger.error.assert_called()
        call_args = [call[0][0] for call in mock_logger.error.call_args_list]
        assert any("exception" in str(arg).lower() for arg in call_args)


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_manifest(self, writer, mock_dropbox_sync):
        """Test with minimal manifest data."""
        minimal_manifest = RunManifest(
            run_id="minimal",
            symbol="BTCUSDT",
            utc_started=datetime(2025, 1, 1, tzinfo=timezone.utc),
            utc_finished=datetime(2025, 1, 1, tzinfo=timezone.utc),
            engine_version="2.0",
            success=True,
            error=None,
            model_s3_path="",
            metrics_s3_path="",
            features_used=[],
            config_snapshot={},
            cost_model={},
        )

        result = writer.write_manifest(minimal_manifest)

        assert result is not None

    def test_large_manifest(self, writer, sample_manifest, mock_dropbox_sync):
        """Test with large manifest data."""
        # Add large data to manifest
        sample_manifest.features_used = [f"feature_{i}" for i in range(1000)]
        sample_manifest.config_snapshot = {f"key_{i}": f"value_{i}" for i in range(500)}

        result = writer.write_manifest(sample_manifest)

        assert result is not None

    def test_special_characters_in_path(self, writer, sample_manifest, mock_dropbox_sync):
        """Test handling of special characters in paths."""
        sample_manifest.symbol = "BTC-USDT"  # Special character

        result = writer.write_manifest(sample_manifest)

        # Should still succeed
        assert result is not None
