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
            logger.error(
                "dropbox_client_init_failed",
                error=str(e),
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
            logger.error(
                "dropbox_auth_failed",
                error=str(e),
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
    
    def sync_all(
        self,
        logs_dir: str | Path = "logs",
        models_dir: str | Path = "models",
        monitoring_dir: str | Path = "logs",
    ) -> dict[str, int]:
        """Sync all data to Dropbox.
        
        Args:
            logs_dir: Local logs directory
            models_dir: Local models directory
            monitoring_dir: Local monitoring data directory
            
        Returns:
            Dictionary with sync counts
        """
        results = {
            "logs": self.upload_logs(logs_dir),
            "models": self.upload_models(models_dir),
            "monitoring": self.upload_monitoring_data(monitoring_dir),
        }
        
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
        interval_seconds: int = 60,
        logs_dir: str | Path = "logs",
        models_dir: str | Path = "models",
        learning_dir: str | Path = "logs/learning",
        monitoring_dir: str | Path = "logs",
        data_cache_dir: str | Path = "data/candles",
    ) -> threading.Thread:
        """Start continuous sync in background thread.
        
        Args:
            interval_seconds: Sync interval in seconds (default: 60 = 1 minute)
            logs_dir: Local logs directory
            models_dir: Local models directory
            learning_dir: Local learning data directory
            monitoring_dir: Local monitoring data directory
            data_cache_dir: Local historical data cache directory
            
        Returns:
            Background thread running continuous sync
        """
        if not self._enabled:
            logger.warning("continuous_sync_disabled")
            return None
        
        def sync_loop():
            logger.info(
                "continuous_sync_started",
                interval_seconds=interval_seconds,
            )
            
            while True:
                try:
                    # Sync all data
                    self.sync_all(
                        logs_dir=logs_dir,
                        models_dir=models_dir,
                        monitoring_dir=monitoring_dir,
                    )
                    
                    # Sync learning data
                    if Path(learning_dir).exists():
                        self.sync_directory(
                            local_dir=learning_dir,
                            remote_dir="/learning",
                            pattern="*.json",
                            recursive=True,
                        )
                    
                    # Sync historical data cache (so we don't need to redownload)
                    # Note: Historical data goes to a shared location, not dated folder
                    # (so it can be reused across days)
                    if Path(data_cache_dir).exists():
                        # Use shared location for historical data (not dated folder)
                        shared_data_path = f"/{self._app_folder}/data/candles"
                        self.sync_directory(
                            local_dir=data_cache_dir,
                            remote_dir=shared_data_path,
                            pattern="*.parquet",
                            recursive=True,
                        )
                    
                    logger.debug("continuous_sync_complete", interval_seconds=interval_seconds)
                    
                except Exception as e:
                    logger.error("continuous_sync_error", error=str(e))
                
                # Wait for next interval
                time.sleep(interval_seconds)
        
        thread = threading.Thread(target=sync_loop, daemon=True, name="DropboxSync")
        thread.start()
        
        logger.info("continuous_sync_thread_started", interval_seconds=interval_seconds)
        return thread
    
    def upload_data_cache(
        self,
        data_cache_dir: str | Path = "data/candles",
        use_shared_location: bool = True,
    ) -> int:
        """Upload historical data cache to Dropbox.
        
        Args:
            data_cache_dir: Local data cache directory
            use_shared_location: If True, use shared location (not dated folder) for reuse
            
        Returns:
            Number of files uploaded
        """
        if use_shared_location:
            # Use shared location for historical data (not dated folder)
            remote_dir = f"/{self._app_folder}/data/candles"
        else:
            remote_dir = "/data/candles"
        
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
    ) -> int:
        """Restore historical data cache from Dropbox.
        
        Args:
            data_cache_dir: Local data cache directory
            remote_dir: Remote Dropbox directory
            
        Returns:
            Number of files restored
        """
        if not self._enabled:
            return 0
        
        local_dir = Path(data_cache_dir)
        local_dir.mkdir(parents=True, exist_ok=True)
        
        # Use shared location for historical data (not dated folder)
        if remote_dir is None:
            remote_dir = f"/{self._app_folder}/data/candles"
        else:
            # Don't use dated folder for historical data (use shared location)
            remote_dir = self._normalize_path(remote_dir, use_dated_folder=False)
        
        try:
            # List files in Dropbox
            result = self._dbx.files_list_folder(remote_dir)
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
            
            logger.info(
                "data_cache_restore_complete",
                files_restored=restored_count,
                local_dir=str(local_dir),
            )
            return restored_count
            
        except Exception as e:
            logger.error("data_cache_restore_failed", error=str(e))
            return 0

