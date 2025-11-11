"""
Hash Utilities for Integrity

Computes and verifies file hashes for integrity checks.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

import structlog  # type: ignore[import-untyped]

logger = structlog.get_logger(__name__)


def compute_file_hash(file_path: str, algorithm: str = "sha256") -> Optional[str]:
    """Compute file hash.
    
    Args:
        file_path: Path to file
        algorithm: Hash algorithm ("sha256", "md5", etc.)
        
    Returns:
        Hash as hex string, or None if file not found
    """
    try:
        hash_obj = hashlib.new(algorithm)
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_obj.update(chunk)
        
        return hash_obj.hexdigest()
        
    except Exception as e:
        logger.warning("hash_computation_failed", file_path=file_path, error=str(e))
        return None


def write_hash_file(file_path: str, hash_value: str, hash_file_path: Optional[str] = None) -> bool:
    """Write hash to file.
    
    Args:
        file_path: Path to file that was hashed
        hash_value: Hash value
        hash_file_path: Path to hash file (defaults to file_path + ".sha256")
        
    Returns:
        True if successful
    """
    try:
        if hash_file_path is None:
            hash_file_path = f"{file_path}.sha256"
        
        with open(hash_file_path, 'w') as f:
            f.write(f"{hash_value}  {Path(file_path).name}\n")
        
        logger.debug("hash_file_written", hash_file_path=hash_file_path)
        return True
        
    except Exception as e:
        logger.error("hash_file_write_failed", hash_file_path=hash_file_path, error=str(e))
        return False


def verify_file_hash(file_path: str, expected_hash: str, algorithm: str = "sha256") -> bool:
    """Verify file hash.
    
    Args:
        file_path: Path to file
        expected_hash: Expected hash value
        algorithm: Hash algorithm ("sha256", "md5", etc.)
        
    Returns:
        True if hash matches
    """
    actual_hash = compute_file_hash(file_path, algorithm)
    if actual_hash is None:
        return False
    
    match = actual_hash == expected_hash
    if not match:
        logger.warning("hash_verification_failed", file_path=file_path, expected=expected_hash, actual=actual_hash)
    
    return match


def read_hash_file(hash_file_path: str) -> Optional[str]:
    """Read hash from file.
    
    Args:
        hash_file_path: Path to hash file
        
    Returns:
        Hash value, or None if not found
    """
    try:
        with open(hash_file_path, 'r') as f:
            line = f.readline().strip()
            # Format: "hash  filename" or just "hash"
            parts = line.split()
            if parts:
                return parts[0]
        return None
        
    except Exception as e:
        logger.warning("hash_file_read_failed", hash_file_path=hash_file_path, error=str(e))
        return None

