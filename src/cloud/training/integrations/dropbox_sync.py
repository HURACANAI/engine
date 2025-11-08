"""Dropbox sync service for automatic cloud backup of logs, models, and training data."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import structlog

try:
    import dropbox
    from dropbox.exceptions import ApiError, AuthError
    DROPBOX_AVAILABLE = True
except ImportError:
    DROPBOX_AVAILABLE = False
    dropbox = None  # type: ignore

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
        
        # Validate token length (Dropbox tokens are typically long)
        if len(self._access_token) < 50:
            logger.warning(
                "dropbox_token_short",
                token_length=len(self._access_token),
                message="Token seems unusually short - may be invalid",
            )
        
        # Log token prefix for debugging (not full token for security)
        logger.info(
            "dropbox_token_received",
            token_prefix=self._access_token[:30],
            token_length=len(self._access_token),
            message="Token received and cleaned",
        )
        
        # Initialize Dropbox client
        try:
            self._dbx = dropbox.Dropbox(self._access_token)
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
            else:
                logger.error(
                    "dropbox_auth_failed",
                    error=error_msg,
                    token_prefix=self._access_token[:20],
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
    ) -> bool:
        """Upload a single file to Dropbox.
        
        Args:
            local_path: Local file path
            remote_path: Remote Dropbox path (relative to app folder)
            overwrite: Whether to overwrite existing files
            
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
        remote_path = self._normalize_path(remote_path)
        
        try:
            with open(local_path, "rb") as f:
                file_data = f.read()
            
            # Check if file exists and is same
            if not overwrite:
                try:
                    existing = self._dbx.files_get_metadata(remote_path)
                    if isinstance(existing, dropbox.files.FileMetadata):
                        # Compare hashes
                        local_hash = hashlib.sha256(file_data).hexdigest()
                        if existing.content_hash == local_hash:
                            logger.debug(
                                "file_unchanged",
                                path=remote_path,
                                message="Skipping upload - file unchanged",
                            )
                            return True
                except ApiError:
                    # File doesn't exist, proceed with upload
                    pass
            
            # Upload file
            mode = dropbox.files.WriteMode.overwrite if overwrite else dropbox.files.WriteMode.add
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
    ) -> int:
        """Sync a directory to Dropbox.
        
        Args:
            local_dir: Local directory path
            remote_dir: Remote Dropbox directory (relative to app folder)
            pattern: File pattern to match (e.g., "*.log", "*.pkl")
            recursive: Whether to sync recursively
            
        Returns:
            Number of files synced
        """
        if not self._enabled:
            return 0
        
        local_dir = Path(local_dir)
        if not local_dir.exists() or not local_dir.is_dir():
            logger.warning("directory_not_found", path=str(local_dir))
            return 0
        
        remote_dir = self._normalize_path(remote_dir)
        
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
            remote_path = f"{remote_dir}/{rel_path.as_posix()}"
            
            if self.upload_file(local_file, remote_path):
                synced_count += 1
        
        logger.info(
            "directory_synced",
            local_dir=str(local_dir),
            remote_dir=remote_dir,
            files_synced=synced_count,
            total_files=len(files),
        )
        
        return synced_count
    
    def upload_logs(self, logs_dir: str | Path = "logs") -> int:
        """Upload all log files to Dropbox.
        
        Args:
            logs_dir: Local logs directory
            
        Returns:
            Number of log files uploaded
        """
        return self.sync_directory(
            local_dir=logs_dir,
            remote_dir="/logs",
            pattern="*.log",
            recursive=True,
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
                if "path/conflict/folder" in str(e.error):
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
            Normalized path
        """
        # Remove leading slash if present
        if path.startswith("/"):
            path = path[1:]
        
        # If path doesn't start with app folder, prepend it
        if not path.startswith(f"{self._app_folder}/"):
            # Use dated folder if available and requested, otherwise use app folder
            if use_dated_folder and hasattr(self, "_dated_folder") and self._dated_folder:
                # Combine with dated folder
                path = f"{self._dated_folder}/{path}"
            else:
                path = f"/{self._app_folder}/{path}"
        else:
            # Path already has app folder, but ensure it uses dated folder if requested
            if use_dated_folder and hasattr(self, "_dated_folder") and self._dated_folder:
                # Replace app folder with dated folder
                path = path.replace(f"/{self._app_folder}/", f"{self._dated_folder}/", 1)
            else:
                # Ensure leading slash
                if not path.startswith("/"):
                    path = f"/{path}"
        
        return path
    
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
                if isinstance(entry, dropbox.files.FileMetadata):
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
                            count = self.sync_directory(
                                local_dir=data_cache_dir,
                                remote_dir="/data/candles",  # Will be normalized to dated folder
                                pattern="*.parquet",
                                recursive=True,
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
                                        remote_path = f"/data/candles/{rel_path.as_posix()}"
                                        remote_path = self._normalize_path(remote_path)
                                        
                                        if self.upload_file(file_path, remote_path):
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
        use_dated_folder: bool = True,
    ) -> int:
        """Upload historical data cache to Dropbox.
        
        Args:
            data_cache_dir: Local data cache directory
            use_dated_folder: If True, store in dated folder (default). If False, use shared location.
            
        Returns:
            Number of files uploaded
        """
        if use_dated_folder:
            # Store in dated folder: /Runpodhuracan/YYYY-MM-DD/data/candles/
            remote_dir = "/data/candles"
        else:
            # Use shared location: /Runpodhuracan/data/candles/
            remote_dir = f"/{self._app_folder}/data/candles"
            # Normalize without dated folder
            remote_dir = self._normalize_path(remote_dir, use_dated_folder=False)
        
        return self.sync_directory(
            local_dir=data_cache_dir,
            remote_dir=remote_dir,
            pattern="*.parquet",
            recursive=True,
        )
    
    def restore_data_cache(
        self,
        data_cache_dir: str | Path = "data/candles",
        remote_dir: Optional[str] = None,
        use_latest_dated_folder: bool = True,
    ) -> int:
        """Restore historical data cache from Dropbox.
        
        This is a convenience function - if Dropbox is empty (first startup),
        the training pipeline will download data normally from the exchange.
        
        Args:
            data_cache_dir: Local data cache directory
            remote_dir: Remote Dropbox directory (if None, uses latest dated folder or shared location)
            use_latest_dated_folder: If True, try to restore from latest dated folder first
            
        Returns:
            Number of files restored (0 if Dropbox is empty - this is OK for first startup)
        """
        if not self._enabled:
            return 0
        
        local_dir = Path(data_cache_dir)
        local_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine remote directory
        if remote_dir is None:
            if use_latest_dated_folder:
                # Try to restore from latest dated folder first
                # This will be set by the caller or use current dated folder
                if hasattr(self, "_dated_folder") and self._dated_folder:
                    remote_dir = f"{self._dated_folder}/data/candles"
                else:
                    # Fallback to shared location
                    remote_dir = f"/{self._app_folder}/data/candles"
            else:
                # Use shared location
                remote_dir = f"/{self._app_folder}/data/candles"
        else:
            # Don't normalize - use as-is (caller specifies exact path)
            pass
        
        try:
            # List files in Dropbox
            # This will raise ApiError if folder doesn't exist (first startup)
            try:
                result = self._dbx.files_list_folder(remote_dir)
            except ApiError as e:
                # Folder doesn't exist in Dropbox - this is OK for first startup
                if "not_found" in str(e.error):
                    logger.info(
                        "data_cache_folder_not_found",
                        remote_dir=remote_dir,
                        message="Dropbox folder doesn't exist yet (first startup) - data will be downloaded during training",
                    )
                    return 0
                else:
                    raise
            
            restored_count = 0
            
            for entry in result.entries:
                if isinstance(entry, dropbox.files.FileMetadata):
                    # Get relative path
                    rel_path = entry.path_display.replace(remote_dir, "").lstrip("/")
                    local_path = local_dir / rel_path
                    
                    # Create parent directory if needed
                    local_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Download file if it doesn't exist locally or is older
                    if not local_path.exists() or local_path.stat().st_mtime < entry.server_modified.timestamp():
                        try:
                            metadata, response = self._dbx.files_download(entry.path_display)
                            with open(local_path, "wb") as f:
                                f.write(response.content)
                            restored_count += 1
                            logger.info(
                                "data_cache_file_restored",
                                remote_path=entry.path_display,
                                local_path=str(local_path),
                            )
                        except Exception as e:
                            logger.warning(
                                "data_cache_file_restore_failed",
                                remote_path=entry.path_display,
                                error=str(e),
                            )
            
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

