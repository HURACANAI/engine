"""
Storage Client Abstraction

Provides storage client abstraction with put_file and put_json.
Supports Dropbox now, S3 later via env DRIVER=dropbox or s3
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

import structlog

logger = structlog.get_logger(__name__)


class StorageClient(ABC):
    """Abstract storage client."""
    
    @abstractmethod
    def put_file(self, local_path: str, remote_path: str, overwrite: bool = True) -> bool:
        """Upload a file.
        
        Args:
            local_path: Local file path
            remote_path: Remote file path
            overwrite: Whether to overwrite existing file
            
        Returns:
            True if successful
        """
        pass
    
    @abstractmethod
    def put_json(self, data: Dict[str, Any], remote_path: str, overwrite: bool = True) -> bool:
        """Upload JSON data.
        
        Args:
            data: JSON data dictionary
            remote_path: Remote file path
            overwrite: Whether to overwrite existing file
            
        Returns:
            True if successful
        """
        pass
    
    @abstractmethod
    def exists(self, remote_path: str) -> bool:
        """Check if file exists.
        
        Args:
            remote_path: Remote file path
            
        Returns:
            True if file exists
        """
        pass
    
    @abstractmethod
    def checksum(self, remote_path: str) -> Optional[str]:
        """Get file checksum.
        
        Args:
            remote_path: Remote file path
            
        Returns:
            SHA256 hash if available, None otherwise
        """
        pass


class DropboxStorageClient(StorageClient):
    """Dropbox storage client."""
    
    def __init__(
        self,
        access_token: str,
        base_path: str = "/Huracan",
    ):
        """Initialize Dropbox storage client.
        
        Args:
            access_token: Dropbox access token
            base_path: Base path in Dropbox
        """
        self.access_token = access_token
        self.base_path = base_path.rstrip("/")
        
        try:
            import dropbox  # type: ignore[import-untyped]
            self.client = dropbox.Dropbox(access_token)
            self.dropbox_module = dropbox  # Store for later use
            logger.info("dropbox_storage_client_initialized", base_path=base_path)
        except ImportError:
            raise ImportError("dropbox package not installed. Install with: pip install dropbox")
        except Exception as e:
            logger.error("dropbox_client_init_failed", error=str(e))
            raise
    
    def put_file(self, local_path: str, remote_path: str, overwrite: bool = True) -> bool:
        """Upload a file to Dropbox.
        
        Args:
            local_path: Local file path
            remote_path: Remote file path
            overwrite: Whether to overwrite existing file
            
        Returns:
            True if successful
        """
        try:
            # Make path absolute
            if not remote_path.startswith("/"):
                remote_path = f"{self.base_path}/{remote_path}"
            else:
                remote_path = f"{self.base_path}{remote_path}"
            
            # Read file
            with open(local_path, 'rb') as f:
                file_data = f.read()
            
            # Upload
            mode = self.dropbox_module.files.WriteMode.overwrite if overwrite else self.dropbox_module.files.WriteMode.add  # type: ignore[misc]
            self.client.files_upload(file_data, remote_path, mode=mode)
            
            logger.info("file_uploaded", local_path=local_path, remote_path=remote_path, size_bytes=len(file_data))
            return True
            
        except Exception as e:
            logger.error("file_upload_failed", local_path=local_path, remote_path=remote_path, error=str(e))
            return False
    
    def put_json(self, data: Dict[str, Any], remote_path: str, overwrite: bool = True) -> bool:
        """Upload JSON data to Dropbox.
        
        Args:
            data: JSON data dictionary
            remote_path: Remote file path
            overwrite: Whether to overwrite existing file
            
        Returns:
            True if successful
        """
        try:
            # Write to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(data, f, indent=2)
                temp_path = f.name
            
            # Upload
            success = self.put_file(temp_path, remote_path, overwrite)
            
            # Clean up
            Path(temp_path).unlink()
            
            return success
            
        except Exception as e:
            logger.error("json_upload_failed", remote_path=remote_path, error=str(e))
            return False
    
    def exists(self, remote_path: str) -> bool:
        """Check if file exists in Dropbox.
        
        Args:
            remote_path: Remote file path
            
        Returns:
            True if file exists
        """
        try:
            if not remote_path.startswith("/"):
                remote_path = f"{self.base_path}/{remote_path}"
            else:
                remote_path = f"{self.base_path}{remote_path}"
            
            self.client.files_get_metadata(remote_path)
            return True
            
        except Exception:
            return False
    
    def checksum(self, remote_path: str) -> Optional[str]:
        """Get file checksum from Dropbox.
        
        Args:
            remote_path: Remote file path
            
        Returns:
            SHA256 hash if available, None otherwise
        """
        try:
            if not remote_path.startswith("/"):
                remote_path = f"{self.base_path}/{remote_path}"
            else:
                remote_path = f"{self.base_path}{remote_path}"
            
            metadata = self.client.files_get_metadata(remote_path)
            if hasattr(metadata, 'content_hash'):
                return metadata.content_hash
            return None
            
        except Exception as e:
            logger.warning("checksum_failed", remote_path=remote_path, error=str(e))
            return None


class S3StorageClient(StorageClient):
    """S3 storage client (placeholder for future implementation)."""
    
    def __init__(
        self,
        bucket: str,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        endpoint_url: Optional[str] = None,
    ):
        """Initialize S3 storage client.
        
        Args:
            bucket: S3 bucket name
            access_key: AWS access key
            secret_key: AWS secret key
            endpoint_url: S3 endpoint URL (for S3-compatible storage)
        """
        self.bucket = bucket
        self.access_key = access_key
        self.secret_key = secret_key
        self.endpoint_url = endpoint_url
        
        # TODO: Implement S3 client
        logger.warning("s3_storage_client_not_implemented", bucket=bucket)
        raise NotImplementedError("S3 storage client not yet implemented")
    
    def put_file(self, local_path: str, remote_path: str, overwrite: bool = True) -> bool:
        """Upload a file to S3."""
        raise NotImplementedError("S3 storage client not yet implemented")
    
    def put_json(self, data: Dict[str, Any], remote_path: str, overwrite: bool = True) -> bool:
        """Upload JSON data to S3."""
        raise NotImplementedError("S3 storage client not yet implemented")
    
    def exists(self, remote_path: str) -> bool:
        """Check if file exists in S3."""
        raise NotImplementedError("S3 storage client not yet implemented")
    
    def checksum(self, remote_path: str) -> Optional[str]:
        """Get file checksum from S3."""
        raise NotImplementedError("S3 storage client not yet implemented")


def create_storage_client(
    driver: str = "dropbox",
    **kwargs: Any,
) -> StorageClient:
    """Create storage client based on driver.
    
    Args:
        driver: Storage driver ("dropbox" or "s3")
        **kwargs: Driver-specific arguments
        
    Returns:
        StorageClient instance
    """
    if driver == "dropbox":
        access_token = kwargs.get("access_token")
        if not access_token:
            raise ValueError("access_token required for Dropbox driver")
        return DropboxStorageClient(access_token=access_token, base_path=kwargs.get("base_path", "/Huracan"))
    elif driver == "s3":
        bucket = kwargs.get("bucket")
        if not bucket:
            raise ValueError("bucket required for S3 driver")
        return S3StorageClient(
            bucket=bucket,
            access_key=kwargs.get("access_key"),
            secret_key=kwargs.get("secret_key"),
            endpoint_url=kwargs.get("endpoint_url"),
        )
    else:
        raise ValueError(f"Unknown storage driver: {driver}")

