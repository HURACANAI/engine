"""Dropbox sync service for automatic cloud backup of logs, models, and training data."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import structlog  # type: ignore[reportMissingImports]

try:
    import dropbox  # type: ignore[reportMissingImports]
    from dropbox.exceptions import ApiError, AuthError  # type: ignore[reportMissingImports]
    DROPBOX_AVAILABLE = True
except ImportError:
    DROPBOX_AVAILABLE = False
    dropbox = None  # type: ignore[assignment]
    ApiError = Exception  # type: ignore[assignment]
    AuthError = Exception  # type: ignore[assignment]

logger = structlog.get_logger(__name__)


class DropboxSync:
    """Automatic Dropbox sync for engine data.
    
    Syncs:
    - Training logs
    - Trained models
    - Monitoring data
    - Training results
    - Configuration files
    
    Usage:
        sync = DropboxSync(
            access_token="your_access_token",
            app_folder="Runpodhuracan"
        )
        sync.upload_file("logs/engine.log", "/logs/engine.log")
        sync.sync_directory("models/", "/models/")
    """
    
    def __init__(
        self,
        access_token: str,
        app_folder: str = "Runpodhuracan",
        enabled: bool = True,
        create_dated_folder: bool = True,
    ) -> None:
        """Initialize Dropbox sync.
        
        Args:
            access_token: Dropbox access token
            app_folder: App folder name in Dropbox
            enabled: Whether sync is enabled
            create_dated_folder: Whether to create a dated folder (YYYY-MM-DD) for today's data
        """
        if not DROPBOX_AVAILABLE:
            raise ImportError(
                "dropbox package not installed. Install with: pip install dropbox"
            )
        
        if not enabled:
            logger.info("dropbox_sync_disabled")
            self._enabled = False
            return
        
        self._enabled = True
        # Clean and validate token (remove any whitespace, newlines, quotes, etc.)
        if not access_token:
            raise ValueError("Dropbox access token is required")
        
        # Clean token: remove whitespace, newlines, quotes, etc.
        self._access_token = access_token.strip().strip('"').strip("'").replace("\n", "").replace("\r", "")
        self._app_folder = app_folder
        
        # Validate token format (should start with 'sl.')
        if not self._access_token.startswith("sl."):
            logger.error(
                "dropbox_token_invalid_format",
                token_prefix=self._access_token[:20] if len(self._access_token) > 20 else self._access_token,
                token_length=len(self._access_token),
                message="Token should start with 'sl.'",
            )
            raise ValueError(f"Invalid Dropbox token format - token should start with 'sl.' (got: {self._access_token[:20]}...)")
        
        # Validate token length (Dropbox tokens are typically very long - 1000+ characters)
        if len(self._access_token) < 100:
            logger.error(
                "dropbox_token_too_short",
                token_length=len(self._access_token),
                message=(
                    f"Token is too short ({len(self._access_token)} chars). "
                    f"Dropbox tokens are typically 1000+ characters. "
                    f"The token may have been truncated or not copied completely."
                ),
            )
            raise ValueError(
                f"Dropbox token is too short ({len(self._access_token)} characters). "
                f"Valid tokens are typically 1000+ characters. "
                f"Please ensure you copied the entire token from Dropbox App Console."
            )
        elif len(self._access_token) < 500:
            logger.warning(
                "dropbox_token_short",
                token_length=len(self._access_token),
                message=(
                    f"Token seems shorter than expected ({len(self._access_token)} chars). "
                    f"Valid tokens are typically 1000+ characters. "
                    f"This may still work, but verify the token is complete."
                ),
            )
        
        # Log token prefix for debugging (not full token for security)
        logger.info(
            "dropbox_token_received",
            token_prefix=self._access_token[:30],
            token_length=len(self._access_token),
            message="Token received and cleaned",
        )
        
        # Initialize Dropbox client
        # Type checker note: dropbox is guaranteed to be not None here because DROPBOX_AVAILABLE is True
        try:
            self._dbx = dropbox.Dropbox(self._access_token)  # type: ignore[misc]
        except Exception as e:
            error_msg = str(e)
            logger.error(
                "dropbox_client_init_failed",
                error=error_msg,
                token_prefix=self._access_token[:30],
                message="Failed to initialize Dropbox client",
            )
            raise
        
        # Test connection
        try:
            account_info = self._dbx.users_get_current_account()
            logger.info(
                "dropbox_sync_initialized",
                account_email=account_info.email,
                app_folder=app_folder,
            )
        except AuthError as e:
            error_msg = str(e)
            # Check if it's an expired token error
            if "expired_access_token" in error_msg or "expired" in error_msg.lower():
                logger.error(
                    "dropbox_token_expired",
                    error=error_msg,
                    token_prefix=self._access_token[:20],
                    token_length=len(self._access_token),
                    message=(
                        "Dropbox access token has expired. "
                        "To fix this:\n"
                        "1. Go to https://www.dropbox.com/developers/apps\n"
                        "2. Select your app (or create a new one)\n"
                        "3. Generate a new access token\n"
                        "4. Update DROPBOX_ACCESS_TOKEN environment variable or settings file\n"
                        "5. Restart the engine"
                    ),
                    help_url="https://www.dropbox.com/developers/apps",
                )
            elif "invalid_access_token" in error_msg or "invalid" in error_msg.lower():
                logger.error(
                    "dropbox_token_invalid",
                    error=error_msg,
                    token_prefix=self._access_token[:20],
                    token_length=len(self._access_token),
                    token_starts_with_sl=self._access_token.startswith("sl."),
                    message=(
                        "Dropbox access token is invalid. "
                        "Possible causes:\n"
                        "1. Token format is incorrect (should start with 'sl.')\n"
                        "2. Token was truncated or corrupted\n"
                        "3. Token was revoked in Dropbox App Console\n"
                        "4. Token has special characters that need escaping\n"
                        "To fix:\n"
                        "1. Go to https://www.dropbox.com/developers/apps\n"
                        "2. Generate a NEW access token\n"
                        "3. Copy the ENTIRE token (it's very long, ~1000+ characters)\n"
                        "4. Set it as: export DROPBOX_ACCESS_TOKEN='your_full_token_here'\n"
                        "5. Restart the engine"
                    ),
                    help_url="https://www.dropbox.com/developers/apps",
                )
            else:
                logger.error(
                    "dropbox_auth_failed",
                    error=error_msg,
                    token_prefix=self._access_token[:20],
                    token_length=len(self._access_token),
                    message="Token authentication failed - check if token is valid and not expired",
                )
            raise
        except Exception as e:
            logger.error(
                "dropbox_connection_failed",
                error=str(e),
                token_prefix=self._access_token[:20],
            )
            raise
        
        # Create dated folder at startup (first thing we do)
        if create_dated_folder:
            self._dated_folder = self._create_dated_folder()
        else:
            self._dated_folder = ""
    
    def upload_file(
        self,
        local_path: str | Path,
        remote_path: str,
        overwrite: bool = True,
        use_dated_folder: bool = True,
    ) -> bool:
        """Upload a single file to Dropbox.
        
        Args:
            local_path: Local file path
            remote_path: Remote Dropbox path (relative to app folder or already normalized)
            overwrite: Whether to overwrite existing files
            use_dated_folder: Whether to use dated folder (default: True). 
                            Set to False if remote_path is already fully normalized.
            
        Returns:
            True if successful, False otherwise
        """
        if not self._enabled:
            return False
        
        local_path = Path(local_path)
        if not local_path.exists():
            logger.warning("file_not_found", path=str(local_path))
            return False
        
        # Normalize remote path
        # If use_dated_folder=False, assume path is already fully normalized and check if it needs normalization
        if not use_dated_folder:
            # Path should already be fully normalized - just ensure it starts with /
            if not remote_path.startswith("/"):
                remote_path = f"/{remote_path}"
            # Log for debugging
            logger.debug(
                "upload_file_using_pre_normalized_path",
                remote_path=remote_path,
                dated_folder=getattr(self, "_dated_folder", "not_set"),
            )
        else:
            # Normalize path (will add dated folder if available)
            remote_path = self._normalize_path(remote_path, use_dated_folder=use_dated_folder)
            logger.debug(
                "upload_file_normalized_path",
                original_path=remote_path,
                normalized_path=remote_path,
                use_dated_folder=use_dated_folder,
                dated_folder=getattr(self, "_dated_folder", "not_set"),
            )
        
        try:
            with open(local_path, "rb") as f:
                file_data = f.read()
            
            # Always check if file exists first (even if overwrite=True)
            # This prevents duplicates and ensures we know what we're overwriting
            try:
                existing = self._dbx.files_get_metadata(remote_path)
                if isinstance(existing, dropbox.files.FileMetadata):  # type: ignore[misc]
                    # Compare hashes to see if file is identical
                    local_hash = hashlib.sha256(file_data).hexdigest()
                    # Dropbox content_hash is base64 encoded, we need to compare properly
                    # For now, if file exists and overwrite=True, we'll overwrite it
                    # If overwrite=False and file exists, skip
                    if not overwrite:
                        logger.debug(
                            "file_exists_skip",
                            path=remote_path,
                            message="File exists and overwrite=False, skipping",
                        )
                        return True
                    # If overwrite=True, continue to upload (will overwrite)
            except ApiError:
                # File doesn't exist, proceed with upload
                pass
            
            # Upload file (will overwrite if exists and overwrite=True)
            mode = dropbox.files.WriteMode.overwrite if overwrite else dropbox.files.WriteMode.add  # type: ignore[misc]
            self._dbx.files_upload(
                file_data,
                remote_path,
                mode=mode,
            )
            
            logger.info(
                "file_uploaded",
                local_path=str(local_path),
                remote_path=remote_path,
                size_bytes=len(file_data),
            )
            return True
            
        except Exception as e:
            logger.error(
                "file_upload_failed",
                local_path=str(local_path),
                remote_path=remote_path,
                error=str(e),
            )
            return False
    
    def sync_directory(
        self,
        local_dir: str | Path,
        remote_dir: str,
        pattern: str = "*",
        recursive: bool = True,
        use_dated_folder: bool = True,
    ) -> int:
        """Sync a directory to Dropbox.
        
        Args:
            local_dir: Local directory path
            remote_dir: Remote Dropbox directory (relative to app folder)
            pattern: File pattern to match (e.g., "*.log", "*.pkl")
            recursive: Whether to sync recursively
            use_dated_folder: Whether to use dated folder (default: True)
            
        Returns:
            Number of files synced
        """
        if not self._enabled:
            return 0
        
        local_dir = Path(local_dir)
        if not local_dir.exists() or not local_dir.is_dir():
            logger.warning("directory_not_found", path=str(local_dir))
            return 0
        
        remote_dir = self._normalize_path(remote_dir, use_dated_folder=use_dated_folder)
        
        # Log the normalized remote directory for debugging
        logger.info(
            "sync_directory_normalized_path",
            normalized_remote_dir=remote_dir,
            use_dated_folder=use_dated_folder,
            dated_folder=getattr(self, "_dated_folder", "not_set"),
        )
        
        # Ensure remote directory exists
        try:
            self._dbx.files_get_metadata(remote_dir)
        except ApiError:
            # Directory doesn't exist, create it
            try:
                self._dbx.files_create_folder_v2(remote_dir)
                logger.info("directory_created", path=remote_dir)
            except ApiError as e:
                logger.warning("directory_creation_failed", path=remote_dir, error=str(e))
        
        synced_count = 0
        
        # Find files matching pattern
        if recursive:
            files = list(local_dir.rglob(pattern))
        else:
            files = list(local_dir.glob(pattern))
        
        for local_file in files:
            if not local_file.is_file():
                continue
            
            # Calculate relative path
            rel_path = local_file.relative_to(local_dir)
            # Construct full remote path (remote_dir is already normalized)
            if remote_dir.endswith("/"):
                remote_path = f"{remote_dir}{rel_path.as_posix()}"
            else:
                remote_path = f"{remote_dir}/{rel_path.as_posix()}"
            
            # Upload file (don't normalize again - remote_path is already fully normalized)
            if self.upload_file(local_file, remote_path, use_dated_folder=False):
                synced_count += 1
        
        logger.info(
            "directory_synced",
            local_dir=str(local_dir),
            remote_dir=remote_dir,
            files_synced=synced_count,
            total_files=len(files),
        )
        
        return synced_count
    
    def upload_logs(self, logs_dir: str | Path = "logs", use_dated_folder: bool = True) -> int:
        """Upload all log files to Dropbox.
        
        Args:
            logs_dir: Local logs directory
            use_dated_folder: Whether to use dated folder (default: True)
            
        Returns:
            Number of log files uploaded
        """
        return self.sync_directory(
            local_dir=logs_dir,
            remote_dir="/logs",
            pattern="*.log",
            recursive=True,
            use_dated_folder=use_dated_folder,
        )
    
    def upload_models(self, models_dir: str | Path = "models") -> int:
        """Upload all model files to Dropbox.
        
        Args:
            models_dir: Local models directory
            
        Returns:
            Number of model files uploaded
        """
        return self.sync_directory(
            local_dir=models_dir,
            remote_dir="/models",
            pattern="*.pkl",
            recursive=True,
        )
    
    def upload_monitoring_data(self, monitoring_dir: str | Path = "logs") -> int:
        """Upload monitoring data to Dropbox.
        
        Args:
            monitoring_dir: Local monitoring data directory
            
        Returns:
            Number of files uploaded
        """
        return self.sync_directory(
            local_dir=monitoring_dir,
            remote_dir="/monitoring",
            pattern="*.json",
            recursive=True,
        )
    
    def upload_reports(self, reports_dir: str | Path = "reports") -> int:
        """Upload reports and analytics to Dropbox.
        
        Args:
            reports_dir: Local reports directory
            
        Returns:
            Number of files uploaded
        """
        reports_dir = Path(reports_dir)
        if not reports_dir.exists():
            return 0
        
        total_synced = 0
        # Sync multiple file types
        for pattern in ["*.json", "*.csv", "*.html", "*.pdf", "*.txt"]:
            count = self.sync_directory(
                local_dir=reports_dir,
                remote_dir="/reports",
                pattern=pattern,
                recursive=True,
            )
            total_synced += count
        
        return total_synced
    
    def upload_exports(self, exports_dir: str | Path = "exports") -> int:
        """Upload exported data files to Dropbox.
        
        Args:
            exports_dir: Local exports directory
            
        Returns:
            Number of files uploaded
        """
        exports_dir = Path(exports_dir)
        if not exports_dir.exists():
            return 0
        
        total_synced = 0
        # Sync all exported files (CSV, JSON, etc.)
        for pattern in ["*.csv", "*.json", "*.parquet"]:
            count = self.sync_directory(
                local_dir=exports_dir,
                remote_dir="/exports",
                pattern=pattern,
                recursive=True,
            )
            total_synced += count
        
        return total_synced
    
    def upload_configs(self, config_dir: str | Path = "config") -> int:
        """Upload configuration files to Dropbox.
        
        Args:
            config_dir: Local config directory
            
        Returns:
            Number of files uploaded
        """
        config_dir = Path(config_dir)
        if not config_dir.exists():
            return 0
        
        total_synced = 0
        # Sync multiple file types
        for pattern in ["*.yaml", "*.yml", "*.json", "*.toml"]:
            count = self.sync_directory(
                local_dir=config_dir,
                remote_dir="/config",
                pattern=pattern,
                recursive=True,
            )
            total_synced += count
        
        return total_synced
    
    def sync_all(
        self,
        logs_dir: str | Path = "logs",
        models_dir: str | Path = "models",
        monitoring_dir: str | Path = "logs",
        learning_dir: str | Path = "logs/learning",
        data_cache_dir: str | Path = "data/candles",
        reports_dir: str | Path = "reports",
        config_dir: str | Path = "config",
        exports_dir: str | Path = "exports",
    ) -> dict[str, int]:
        """Sync all data to Dropbox.
        
        Args:
            logs_dir: Local logs directory
            models_dir: Local models directory
            monitoring_dir: Local monitoring data directory
            learning_dir: Local learning data directory
            data_cache_dir: Local historical data cache directory
            reports_dir: Local reports directory
            config_dir: Local config directory
            
        Returns:
            Dictionary with sync counts
        """
        results = {
            "logs": self.upload_logs(logs_dir),
            "models": self.upload_models(models_dir),
            "monitoring": self.upload_monitoring_data(monitoring_dir),
        }
        
        # Sync learning data if directory exists
        if Path(learning_dir).exists():
            results["learning"] = self.sync_directory(
                local_dir=learning_dir,
                remote_dir="/learning",
                pattern="*.json",
                recursive=True,
            )
        
        # Sync data cache if directory exists
        if Path(data_cache_dir).exists():
            results["data_cache"] = self.upload_data_cache(
                data_cache_dir=data_cache_dir,
                use_dated_folder=True,
            )
        
        # Sync reports if directory exists
        if Path(reports_dir).exists():
            results["reports"] = self.upload_reports(reports_dir)
        
        # Sync configs if directory exists
        if Path(config_dir).exists():
            results["config"] = self.upload_configs(config_dir)
        
        # Sync exported data (trade history, performance metrics, etc.)
        if Path(exports_dir).exists():
            results["exports"] = self.upload_exports(exports_dir)
        
        logger.info(
            "dropbox_sync_complete",
            **results,
            total_files=sum(results.values()),
        )
        
        return results
    
    def _create_dated_folder(self) -> str:
        """Create a dated folder in Dropbox (YYYY-MM-DD format).
        
        Returns:
            Path to the dated folder (e.g., "/Runpodhuracan/2025-11-06/")
        """
        if not self._enabled:
            return ""
        
        # Get today's date in YYYY-MM-DD format
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        dated_path = f"/{self._app_folder}/{today}"
        
        try:
            # Try to create the folder (will not error if it already exists)
            try:
                self._dbx.files_create_folder_v2(dated_path)
                logger.info(
                    "dropbox_dated_folder_created",
                    folder_path=dated_path,
                    date=today,
                )
            except ApiError as e:
                # Folder already exists - that's fine
                error_msg = str(getattr(e, 'error', e))
                if "path/conflict/folder" in error_msg:
                    logger.debug(
                        "dropbox_dated_folder_exists",
                        folder_path=dated_path,
                        date=today,
                    )
                else:
                    raise
            
            return dated_path
        except Exception as e:
            logger.error(
                "dropbox_dated_folder_creation_failed",
                folder_path=dated_path,
                error=str(e),
            )
            # Fallback to app folder root if dated folder creation fails
            return f"/{self._app_folder}"
    
    def _normalize_path(self, path: str, use_dated_folder: bool = True) -> str:
        """Normalize Dropbox path.
        
        All paths are organized under the dated folder created at startup.
        Structure: /Runpodhuracan/YYYY-MM-DD/data_type/file
        
        Args:
            path: Path to normalize
            use_dated_folder: If True, use dated folder (default). If False, use app folder root.
            
        Returns:
            Normalized path (always starts with /)
        """
        # Remove leading slash for processing
        path_no_slash = path.lstrip("/")
        
        # Get dated folder path (without leading slash for comparison)
        dated_folder_no_slash = ""
        if use_dated_folder and hasattr(self, "_dated_folder") and self._dated_folder:
            dated_folder_no_slash = self._dated_folder.lstrip("/")
        
        # Check if path already contains the dated folder
        if dated_folder_no_slash and path_no_slash.startswith(dated_folder_no_slash):
            # Path already uses dated folder - just ensure leading slash
            return f"/{path_no_slash}"
        
        # Check if path starts with app folder
        if path_no_slash.startswith(f"{self._app_folder}/"):
            # Path has app folder but not dated folder
            if use_dated_folder and dated_folder_no_slash:
                # Replace app folder root with dated folder
                # Example: Runpodhuracan/logs/file -> Runpodhuracan/2025-11-08/logs/file
                path_no_slash = path_no_slash.replace(f"{self._app_folder}/", f"{dated_folder_no_slash}/", 1)
            # Ensure leading slash
            return f"/{path_no_slash}"
        
        # Path doesn't start with app folder - prepend dated folder or app folder
        if use_dated_folder and dated_folder_no_slash:
            # Use dated folder
            return f"/{dated_folder_no_slash}/{path_no_slash}"
        else:
            # Use app folder root
            return f"/{self._app_folder}/{path_no_slash}"
    
    def list_files(self, remote_dir: str = "/") -> list[str]:
        """List files in Dropbox directory.
        
        Args:
            remote_dir: Remote directory path
            
        Returns:
            List of file paths
        """
        if not self._enabled:
            return []
        
        remote_dir = self._normalize_path(remote_dir)
        
        try:
            result = self._dbx.files_list_folder(remote_dir)
            files = []
            
            for entry in result.entries:
                if isinstance(entry, dropbox.files.FileMetadata):  # type: ignore[misc]
                    files.append(entry.path_display)
            
            return files
        except Exception as e:
            logger.error("list_files_failed", path=remote_dir, error=str(e))
            return []
    
    def start_continuous_sync(
        self,
        logs_dir: str | Path = "logs",
        models_dir: str | Path = "models",
        learning_dir: str | Path = "logs/learning",
        monitoring_dir: str | Path = "logs",
        data_cache_dir: str | Path = "data/candles",
        sync_intervals: Optional[dict[str, int]] = None,
    ) -> list[threading.Thread]:
        """Start continuous sync with different intervals for different data types.
        
        Args:
            logs_dir: Local logs directory
            models_dir: Local models directory
            learning_dir: Local learning data directory
            monitoring_dir: Local monitoring data directory
            data_cache_dir: Local historical data cache directory
            sync_intervals: Dict with sync intervals in seconds for each data type:
                - learning: interval for learning data (default: 300 = 5 min)
                - logs: interval for logs (default: 300 = 5 min)
                - models: interval for models (default: 1800 = 30 min)
                - data_cache: interval for historical data (default: 7200 = 2 hours)
            
        Returns:
            List of background threads running continuous sync
        """
        if not self._enabled:
            logger.warning("continuous_sync_disabled")
            return []
        
        # Default sync intervals (can be overridden)
        intervals = sync_intervals or {}
        learning_interval = intervals.get("learning", 300)  # 5 minutes
        logs_interval = intervals.get("logs", 300)  # 5 minutes
        models_interval = intervals.get("models", 1800)  # 30 minutes
        data_cache_interval = intervals.get("data_cache", 7200)  # 2 hours
        
        threads = []
        
        # Sync learning data (frequently - captures insights quickly)
        def sync_learning_loop():
            logger.info("continuous_sync_learning_started", interval_seconds=learning_interval)
            while True:
                try:
                    if Path(learning_dir).exists():
                        count = self.sync_directory(
                            local_dir=learning_dir,
                            remote_dir="/learning",
                            pattern="*.json",
                            recursive=True,
                        )
                        if count > 0:
                            logger.info(
                                "learning_data_synced",
                                files_synced=count,
                                interval_seconds=learning_interval,
                            )
                except Exception as e:
                    logger.error("learning_sync_error", error=str(e))
                time.sleep(learning_interval)
        
        # Sync logs & monitoring (frequently - for real-time monitoring)
        def sync_logs_loop():
            logger.info("continuous_sync_logs_started", interval_seconds=logs_interval)
            while True:
                try:
                    # Sync logs
                    log_count = self.upload_logs(logs_dir) if Path(logs_dir).exists() else 0
                    
                    # Sync monitoring data
                    monitoring_count = (
                        self.upload_monitoring_data(monitoring_dir)
                        if Path(monitoring_dir).exists()
                        else 0
                    )
                    
                    # Sync reports if they exist
                    reports_count = 0
                    if Path("reports").exists():
                        reports_count = self.upload_reports("reports")
                    
                    # Sync exports (comprehensive data export) - less frequently but still regularly
                    exports_count = 0
                    if Path("exports").exists():
                        # Only sync exports every 30 minutes (not every 5 minutes)
                        # Exports are large files, don't need real-time sync
                        exports_count = self.upload_exports("exports")
                    
                    if log_count > 0 or monitoring_count > 0 or reports_count > 0 or exports_count > 0:
                        logger.info(
                            "logs_synced",
                            logs_synced=log_count,
                            monitoring_synced=monitoring_count,
                            reports_synced=reports_count,
                            exports_synced=exports_count,
                            interval_seconds=logs_interval,
                        )
                except Exception as e:
                    logger.error("logs_sync_error", error=str(e))
                time.sleep(logs_interval)
        
        # Sync models (less frequently - models don't change often)
        def sync_models_loop():
            logger.info("continuous_sync_models_started", interval_seconds=models_interval)
            while True:
                try:
                    if Path(models_dir).exists():
                        count = self.upload_models(models_dir)
                        if count > 0:
                            logger.info(
                                "models_synced",
                                files_synced=count,
                                interval_seconds=models_interval,
                            )
                except Exception as e:
                    logger.error("models_sync_error", error=str(e))
                time.sleep(models_interval)
        
        # Sync historical data cache
        # Store in dated folder: /Runpodhuracan/YYYY-MM-DD/data/candles/
        # Also checks for recently modified files and syncs them immediately
        def sync_data_cache_loop():
            logger.info(
                "continuous_sync_data_cache_started",
                interval_seconds=data_cache_interval,
            )
            last_sync_time = time.time()
            # Check for new files every 5 minutes (more frequent than full sync)
            quick_check_interval = 300  # 5 minutes
            
            while True:
                try:
                    if Path(data_cache_dir).exists():
                        current_time = time.time()
                        time_since_last_sync = current_time - last_sync_time
                        
                        # If it's been more than the full interval, do a full sync
                        # Otherwise, check for recently modified files and sync those immediately
                        if time_since_last_sync >= data_cache_interval:
                            # Full sync of all files
                            # Full sync: sync all data cache files to SHARED location (not dated folder)
                            count = self.upload_data_cache(
                                data_cache_dir=data_cache_dir,
                                use_dated_folder=False,  # Use shared location (persists across days)
                            )
                            if count > 0:
                                logger.info(
                                    "data_cache_synced_full",
                                    files_synced=count,
                                    interval_seconds=data_cache_interval,
                                )
                            last_sync_time = current_time
                        else:
                            # Quick check: sync only files modified in the last 10 minutes
                            recent_files = []
                            cutoff_time = current_time - 600  # 10 minutes ago
                            
                            for parquet_file in Path(data_cache_dir).rglob("*.parquet"):
                                try:
                                    if parquet_file.stat().st_mtime > cutoff_time:
                                        recent_files.append(parquet_file)
                                except (OSError, FileNotFoundError):
                                    continue
                            
                            if recent_files:
                                # Sync recently modified files immediately
                                synced_count = 0
                                for file_path in recent_files:
                                    try:
                                        # Get relative path
                                        rel_path = file_path.relative_to(Path(data_cache_dir))
                                        # Use shared location: /Runpodhuracan/data/candles/ (not dated folder)
                                        remote_path = f"/{self._app_folder}/data/candles/{rel_path.as_posix()}"
                                        remote_path = self._normalize_path(remote_path, use_dated_folder=False)
                                        
                                        # remote_path is already normalized to shared location, don't normalize again
                                        if self.upload_file(file_path, remote_path, use_dated_folder=False):
                                            synced_count += 1
                                            logger.info(
                                                "coin_data_synced_immediately",
                                                file=str(file_path),
                                                remote_path=remote_path,
                                                message="Newly downloaded coin data synced to Dropbox",
                                            )
                                    except Exception as file_error:
                                        logger.warning(
                                            "recent_file_sync_failed",
                                            file=str(file_path),
                                            error=str(file_error),
                                        )
                                
                                if synced_count > 0:
                                    logger.info(
                                        "recent_coin_data_synced",
                                        files_synced=synced_count,
                                        total_recent_files=len(recent_files),
                                        message="Recently downloaded coin data synced immediately",
                                    )
                    
                    # Sleep for quick check interval (5 minutes) instead of full interval
                    # This allows us to catch newly downloaded files faster
                    time.sleep(min(quick_check_interval, data_cache_interval))
                    
                except Exception as e:
                    logger.error("data_cache_sync_error", error=str(e))
                    time.sleep(60)  # Short sleep on error
        
        # Start all sync threads
        if learning_interval > 0:
            thread = threading.Thread(
                target=sync_learning_loop, daemon=True, name="DropboxSync-Learning"
            )
            thread.start()
            threads.append(thread)
        
        if logs_interval > 0:
            thread = threading.Thread(
                target=sync_logs_loop, daemon=True, name="DropboxSync-Logs"
            )
            thread.start()
            threads.append(thread)
        
        if models_interval > 0:
            thread = threading.Thread(
                target=sync_models_loop, daemon=True, name="DropboxSync-Models"
            )
            thread.start()
            threads.append(thread)
        
        if data_cache_interval > 0:
            thread = threading.Thread(
                target=sync_data_cache_loop, daemon=True, name="DropboxSync-DataCache"
            )
            thread.start()
            threads.append(thread)
        
        logger.info(
            "continuous_sync_threads_started",
            learning_interval=learning_interval,
            logs_interval=logs_interval,
            models_interval=models_interval,
            data_cache_interval=data_cache_interval,
            total_threads=len(threads),
        )
        return threads
    
    def upload_data_cache(
        self,
        data_cache_dir: str | Path = "data/candles",
        use_dated_folder: bool = False,  # Changed default: Historical data should be in shared location
    ) -> int:
        """Upload historical data cache to Dropbox.
        
        Historical coin data should be stored in a SHARED location (not dated folders)
        so it persists across days and can be restored on startup without re-downloading.
        
        Args:
            data_cache_dir: Local data cache directory
            use_dated_folder: If False (default), use shared location. If True, store in dated folder.
            
        Returns:
            Number of files uploaded
        """
        if use_dated_folder:
            # Store in dated folder: /Runpodhuracan/YYYY-MM-DD/data/candles/
            remote_dir = "/data/candles"
        else:
            # Use shared location: /Runpodhuracan/data/candles/ (persists across days)
            remote_dir = f"/{self._app_folder}/data/candles"
            # Normalize without dated folder to ensure it goes to shared location
            remote_dir = self._normalize_path(remote_dir, use_dated_folder=False)
        
        return self.sync_directory(
            local_dir=data_cache_dir,
            remote_dir=remote_dir,
            pattern="*.parquet",
            recursive=True,
            use_dated_folder=False,  # Always use shared location for historical data
        )
    
    def restore_data_cache(
        self,
        data_cache_dir: str | Path = "data/candles",
        remote_dir: Optional[str] = None,
        use_latest_dated_folder: bool = False,  # Changed default: Always use shared location
    ) -> int:
        """Restore historical data cache from Dropbox.
        
        Historical coin data is stored in a SHARED location (not dated folders)
        so it persists across days. This function restores ALL historical data
        from the shared location to avoid re-downloading on every startup.
        
        If Dropbox is empty (first startup), this returns 0 and training will
        download data normally from the exchange.
        
        Args:
            data_cache_dir: Local data cache directory
            remote_dir: Remote Dropbox directory (if None, uses shared location)
            use_latest_dated_folder: If True, try to restore from latest dated folder first (not recommended)
            
        Returns:
            Number of files restored (0 if Dropbox is empty - this is OK for first startup)
        """
        if not self._enabled:
            return 0
        
        local_dir = Path(data_cache_dir)
        local_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine remote directory - ALWAYS use shared location for historical data
        if remote_dir is None:
            # Always use shared location: /Runpodhuracan/data/candles/
            # This ensures data persists across days
            remote_dir = f"/{self._app_folder}/data/candles"
            # Normalize without dated folder
            remote_dir = self._normalize_path(remote_dir, use_dated_folder=False)
        elif use_latest_dated_folder:
            # Only if explicitly requested, try dated folder first
            if hasattr(self, "_dated_folder") and self._dated_folder:
                remote_dir = f"{self._dated_folder}/data/candles"
            else:
                # Fallback to shared location
                remote_dir = f"/{self._app_folder}/data/candles"
                remote_dir = self._normalize_path(remote_dir, use_dated_folder=False)
        else:
            # Don't normalize - use as-is (caller specifies exact path)
            pass
        
        try:
            # List files in Dropbox (recursively)
            # This will raise ApiError if folder doesn't exist (first startup)
            restored_count = 0
            
            def _restore_folder(folder_path: str, local_base: Path) -> int:
                """Recursively restore files from a Dropbox folder."""
                count = 0
                try:
                    result = self._dbx.files_list_folder(folder_path)
                except ApiError as e:
                    # Folder doesn't exist in Dropbox - this is OK for first startup
                    error_msg = str(getattr(e, 'error', e))
                    if "not_found" in error_msg:
                        logger.debug(
                            "data_cache_subfolder_not_found",
                            remote_dir=folder_path,
                        )
                        return 0
                    else:
                        raise
                
                # Process all entries (files and folders)
                for entry in result.entries:
                    if isinstance(entry, dropbox.files.FileMetadata):  # type: ignore[misc]
                        # It's a file - download it
                        # Remove the remote_dir prefix to get relative path
                        if entry.path_display.startswith(remote_dir):
                            rel_path = entry.path_display[len(remote_dir):].lstrip("/")
                        else:
                            # Fallback: try to extract relative path
                            rel_path = entry.path_display.split("/data/candles/")[-1] if "/data/candles/" in entry.path_display else entry.name
                        local_path = local_base / rel_path
                        
                        # Create parent directory if needed
                        local_path.parent.mkdir(parents=True, exist_ok=True)
                        
                        # Download file if it doesn't exist locally or is older
                        if not local_path.exists() or local_path.stat().st_mtime < entry.server_modified.timestamp():
                            try:
                                metadata, response = self._dbx.files_download(entry.path_display)
                                with open(local_path, "wb") as f:
                                    f.write(response.content)
                                count += 1
                                logger.debug(
                                    "data_cache_file_restored",
                                    remote_path=entry.path_display,
                                    local_path=str(local_path),
                                    size_bytes=len(response.content),
                                )
                            except Exception as e:
                                logger.warning(
                                    "data_cache_file_restore_failed",
                                    remote_path=entry.path_display,
                                    error=str(e),
                                )
                    elif isinstance(entry, dropbox.files.FolderMetadata):  # type: ignore[misc]
                        # It's a folder - recurse into it
                        count += _restore_folder(entry.path_display, local_base)
                
                # Check if there are more entries (pagination)
                while result.has_more:
                    result = self._dbx.files_list_folder_continue(result.cursor)
                    for entry in result.entries:
                        if isinstance(entry, dropbox.files.FileMetadata):  # type: ignore[misc]
                            # Remove the remote_dir prefix to get relative path
                            if entry.path_display.startswith(remote_dir):
                                rel_path = entry.path_display[len(remote_dir):].lstrip("/")
                            else:
                                # Fallback: try to extract relative path
                                rel_path = entry.path_display.split("/data/candles/")[-1] if "/data/candles/" in entry.path_display else entry.name
                            local_path = local_base / rel_path
                            local_path.parent.mkdir(parents=True, exist_ok=True)
                            
                            if not local_path.exists() or local_path.stat().st_mtime < entry.server_modified.timestamp():
                                try:
                                    metadata, response = self._dbx.files_download(entry.path_display)
                                    with open(local_path, "wb") as f:
                                        f.write(response.content)
                                    count += 1
                                    logger.debug(
                                        "data_cache_file_restored",
                                        remote_path=entry.path_display,
                                        local_path=str(local_path),
                                        size_bytes=len(response.content),
                                    )
                                except Exception as e:
                                    logger.warning(
                                        "data_cache_file_restore_failed",
                                        remote_path=entry.path_display,
                                        error=str(e),
                                    )
                        elif isinstance(entry, dropbox.files.FolderMetadata):  # type: ignore[misc]
                            count += _restore_folder(entry.path_display, local_base)
                
                return count
            
            # Start recursive restoration
            try:
                restored_count = _restore_folder(remote_dir, local_dir)
            except ApiError as e:
                # Root folder doesn't exist in Dropbox - this is OK for first startup
                error_msg = str(getattr(e, 'error', e))
                if "not_found" in error_msg:
                    logger.info(
                        "data_cache_folder_not_found",
                        remote_dir=remote_dir,
                        message="Dropbox folder doesn't exist yet (first startup) - data will be downloaded during training",
                    )
                    return 0
                else:
                    raise
            
            if restored_count > 0:
                logger.info(
                    "data_cache_restore_complete",
                    files_restored=restored_count,
                    local_dir=str(local_dir),
                )
            else:
                logger.info(
                    "data_cache_restore_empty",
                    remote_dir=remote_dir,
                    message="Dropbox folder exists but is empty - data will be downloaded during training",
                )
            
            return restored_count
            
        except Exception as e:
            logger.error("data_cache_restore_failed", error=str(e))
            # Return 0 on error - training will download data normally
            return 0
    
    def write_text(
        self,
        remote_path: str,
        text_content: str,
        use_dated_folder: bool = False,
        overwrite: bool = True,
    ) -> bool:
        """Write text content directly to Dropbox.
        
        Useful for writing JSON manifests, heartbeats, and other text files.
        
        Args:
            remote_path: Remote Dropbox path (e.g., "champion/latest.json")
            text_content: Text content to write
            use_dated_folder: Whether to use dated folder (default: False for shared files)
            overwrite: Whether to overwrite existing files
            
        Returns:
            True if successful, False otherwise
        """
        if not self._enabled:
            return False
        
        # Normalize path
        if not use_dated_folder:
            # For shared files (champion, heartbeats, registry), use app folder root
            if not remote_path.startswith("/"):
                remote_path = f"/{self._app_folder}/{remote_path}"
            else:
                # Path already starts with /, ensure it has app folder
                if not remote_path.startswith(f"/{self._app_folder}"):
                    remote_path = f"/{self._app_folder}{remote_path}"
        else:
            # Use dated folder
            remote_path = self._normalize_path(remote_path, use_dated_folder=True)
        
        try:
            # Convert text to bytes
            file_data = text_content.encode('utf-8')
            
            # Write mode
            mode = dropbox.files.WriteMode.overwrite if overwrite else dropbox.files.WriteMode.add  # type: ignore[misc]
            
            # Upload file
            self._dbx.files_upload(
                file_data,
                remote_path,
                mode=mode,
            )
            
            logger.info(
                "text_written_to_dropbox",
                remote_path=remote_path,
                size_bytes=len(file_data),
            )
            return True
            
        except Exception as e:
            logger.error(
                "text_write_failed",
                remote_path=remote_path,
                error=str(e),
            )
            return False

