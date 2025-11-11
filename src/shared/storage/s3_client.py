"""
S3 Storage Client

S3 client for storing models, champions, summaries, and live logs.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional

import structlog

logger = structlog.get_logger(__name__)

try:
    import boto3
    from botocore.exceptions import ClientError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    boto3 = None
    ClientError = Exception


class S3Client:
    """S3 client for storage."""
    
    def __init__(
        self,
        bucket: str,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        region: str = "us-east-1",
    ):
        """Initialize S3 client.
        
        Args:
            bucket: S3 bucket name
            access_key: AWS access key (optional, uses credentials if not provided)
            secret_key: AWS secret key (optional, uses credentials if not provided)
            endpoint_url: S3 endpoint URL (for S3-compatible storage like R2)
            region: AWS region
        """
        if not BOTO3_AVAILABLE:
            raise ImportError("boto3 not installed. Install with: pip install boto3")
        
        self.bucket = bucket
        self.region = region
        
        # Create S3 client
        s3_kwargs = {
            "region_name": region,
        }
        
        if access_key and secret_key:
            s3_kwargs["aws_access_key_id"] = access_key
            s3_kwargs["aws_secret_access_key"] = secret_key
        
        if endpoint_url:
            s3_kwargs["endpoint_url"] = endpoint_url
        
        self.s3_client = boto3.client("s3", **s3_kwargs)
        logger.info("s3_client_initialized", bucket=bucket, region=region, endpoint_url=endpoint_url)
    
    def put_file(self, local_path: str, s3_path: str, overwrite: bool = True) -> bool:
        """Upload a file to S3.
        
        Args:
            local_path: Local file path
            s3_path: S3 object path
            overwrite: Whether to overwrite existing file
            
        Returns:
            True if successful
        """
        try:
            self.s3_client.upload_file(local_path, self.bucket, s3_path)
            logger.info("file_uploaded", local_path=local_path, s3_path=s3_path, bucket=self.bucket)
            return True
        except ClientError as e:
            logger.error("file_upload_failed", local_path=local_path, s3_path=s3_path, error=str(e))
            return False
    
    def put_json(self, data: Dict[str, Any], s3_path: str, overwrite: bool = True) -> bool:
        """Upload JSON data to S3.
        
        Args:
            data: JSON data dictionary
            s3_path: S3 object path
            overwrite: Whether to overwrite existing file
            
        Returns:
            True if successful
        """
        try:
            json_str = json.dumps(data, indent=2)
            self.s3_client.put_object(
                Bucket=self.bucket,
                Key=s3_path,
                Body=json_str.encode('utf-8'),
                ContentType='application/json',
            )
            logger.info("json_uploaded", s3_path=s3_path, bucket=self.bucket)
            return True
        except ClientError as e:
            logger.error("json_upload_failed", s3_path=s3_path, error=str(e))
            return False
    
    def get_file(self, s3_path: str, local_path: str) -> bool:
        """Download a file from S3.
        
        Args:
            s3_path: S3 object path
            local_path: Local file path
            
        Returns:
            True if successful
        """
        try:
            self.s3_client.download_file(self.bucket, s3_path, local_path)
            logger.info("file_downloaded", s3_path=s3_path, local_path=local_path, bucket=self.bucket)
            return True
        except ClientError as e:
            logger.error("file_download_failed", s3_path=s3_path, error=str(e))
            return False
    
    def get_json(self, s3_path: str) -> Optional[Dict[str, Any]]:
        """Download JSON data from S3.
        
        Args:
            s3_path: S3 object path
            
        Returns:
            JSON data dictionary, or None if not found
        """
        try:
            response = self.s3_client.get_object(Bucket=self.bucket, Key=s3_path)
            json_str = response['Body'].read().decode('utf-8')
            return json.loads(json_str)
        except ClientError as e:
            logger.error("json_download_failed", s3_path=s3_path, error=str(e))
            return None
    
    def exists(self, s3_path: str) -> bool:
        """Check if file exists in S3.
        
        Args:
            s3_path: S3 object path
            
        Returns:
            True if file exists
        """
        try:
            self.s3_client.head_object(Bucket=self.bucket, Key=s3_path)
            return True
        except ClientError:
            return False
    
    def get_signed_url(self, s3_path: str, expiration: int = 3600) -> Optional[str]:
        """Get signed URL for S3 object.
        
        Args:
            s3_path: S3 object path
            expiration: URL expiration time in seconds
            
        Returns:
            Signed URL, or None if failed
        """
        try:
            url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.bucket, 'Key': s3_path},
                ExpiresIn=expiration,
            )
            return url
        except ClientError as e:
            logger.error("signed_url_failed", s3_path=s3_path, error=str(e))
            return None

