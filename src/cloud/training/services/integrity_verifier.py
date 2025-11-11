"""
Integrity Verifier Service

Verifies that all model paths in latest.json exist and hashes match.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import structlog

from typing import TYPE_CHECKING

from src.shared.contracts.per_coin import ChampionPointer
from src.shared.contracts.paths import get_champion_pointer_path

if TYPE_CHECKING:
    from ..integrations.dropbox_sync import DropboxSync  # type: ignore[import-untyped]

logger = structlog.get_logger(__name__)


class IntegrityVerifier:
    """Verifies integrity of models and artifacts."""
    
    def __init__(
        self,
        dropbox_sync: Optional["DropboxSync"] = None,
        base_folder: str = "huracan",
    ):
        """Initialize integrity verifier.
        
        Args:
            dropbox_sync: DropboxSync instance for reading files
            base_folder: Base folder name in Dropbox (default: "huracan")
        """
        self.dropbox_sync = dropbox_sync
        self.base_folder = base_folder
        logger.info("integrity_verifier_initialized", base_folder=base_folder)
    
    def compute_file_hash(self, file_path: str) -> Optional[str]:
        """Compute SHA256 hash of a file.
        
        Args:
            file_path: Path to file (local or Dropbox)
            
        Returns:
            SHA256 hash as hex string, or None if file not found
        """
        try:
            if self.dropbox_sync and file_path.startswith("/"):
                # Download from Dropbox
                _, response = self.dropbox_sync._dbx.files_download(file_path)
                file_data = response.content
            else:
                # Read from local filesystem
                with open(file_path, 'rb') as f:
                    file_data = f.read()
            
            return hashlib.sha256(file_data).hexdigest()
        except Exception as e:
            logger.warning("file_hash_computation_failed", file_path=file_path, error=str(e))
            return None
    
    def verify_champion_pointer(
        self,
        champion_pointer: Optional[ChampionPointer] = None,
    ) -> Tuple[bool, List[str], List[str]]:
        """Verify integrity of champion pointer and all referenced models.
        
        Args:
            champion_pointer: ChampionPointer instance (will load if not provided)
            
        Returns:
            Tuple of (is_valid, errors, warnings)
        """
        errors = []
        warnings = []
        
        # Load champion pointer if not provided
        if champion_pointer is None:
            if not self.dropbox_sync:
                errors.append("dropbox_sync_not_available")
                return False, errors, warnings
            
            dropbox_path = get_champion_pointer_path(self.base_folder)
            try:
                _, response = self.dropbox_sync._dbx.files_download(dropbox_path)
                json_str = response.content.decode('utf-8')
                champion_pointer = ChampionPointer.from_json(json_str)
            except Exception as e:
                errors.append(f"failed_to_load_champion_pointer: {str(e)}")
                return False, errors, warnings
        
        if not champion_pointer:
            errors.append("champion_pointer_is_none")
            return False, errors, warnings
        
        # Verify each model path exists
        for symbol, model_path in champion_pointer.models.items():
            # Check if model file exists
            try:
                if self.dropbox_sync and model_path.startswith("/"):
                    # Check Dropbox file
                    metadata = self.dropbox_sync._dbx.files_get_metadata(model_path)
                    if not metadata:
                        errors.append(f"model_not_found_{symbol}: {model_path}")
                        continue
                else:
                    # Check local file
                    if not Path(model_path).exists():
                        errors.append(f"model_not_found_{symbol}: {model_path}")
                        continue
                
                # Compute hash if code_hash is present
                if champion_pointer.code_hash:
                    file_hash = self.compute_file_hash(model_path)
                    if file_hash:
                        # Compare with code_hash (first 16 chars)
                        if file_hash[:16] != champion_pointer.code_hash[:16]:
                            errors.append(f"model_hash_mismatch_{symbol}: expected {champion_pointer.code_hash[:16]}, got {file_hash[:16]}")
                    else:
                        warnings.append(f"could_not_compute_hash_{symbol}")
                
            except Exception as e:
                errors.append(f"model_verification_failed_{symbol}: {str(e)}")
        
        is_valid = len(errors) == 0
        
        logger.info(
            "champion_pointer_verified",
            is_valid=is_valid,
            errors=len(errors),
            warnings=len(warnings),
            symbols_checked=len(champion_pointer.models),
        )
        
        return is_valid, errors, warnings
    
    def verify_model_artifact(
        self,
        symbol: str,
        model_path: str,
        expected_hash: Optional[str] = None,
    ) -> Tuple[bool, Optional[str]]:
        """Verify a single model artifact.
        
        Args:
            symbol: Trading symbol
            model_path: Path to model file
            expected_hash: Expected hash (optional)
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Check if file exists
            if self.dropbox_sync and model_path.startswith("/"):
                metadata = self.dropbox_sync._dbx.files_get_metadata(model_path)
                if not metadata:
                    return False, f"model_file_not_found: {model_path}"
            else:
                if not Path(model_path).exists():
                    return False, f"model_file_not_found: {model_path}"
            
            # Verify hash if provided
            if expected_hash:
                file_hash = self.compute_file_hash(model_path)
                if file_hash:
                    if file_hash[:16] != expected_hash[:16]:
                        return False, f"hash_mismatch: expected {expected_hash[:16]}, got {file_hash[:16]}"
                else:
                    return False, "could_not_compute_hash"
            
            return True, None
            
        except Exception as e:
            return False, f"verification_failed: {str(e)}"
    
    def verify_all_artifacts(
        self,
        manifest_data: Dict[str, Any],
    ) -> Dict[str, Tuple[bool, List[str]]]:
        """Verify all artifacts referenced in a manifest.
        
        Args:
            manifest_data: Manifest dictionary
            
        Returns:
            Dictionary mapping symbol to (is_valid, errors)
        """
        results = {}
        
        artifacts_map = manifest_data.get("artifacts_map", {})
        
        for symbol, model_path in artifacts_map.items():
            is_valid, error = self.verify_model_artifact(symbol, model_path)
            errors = [error] if error else []
            results[symbol] = (is_valid, errors)
        
        logger.info(
            "all_artifacts_verified",
            total_symbols=len(results),
            valid_count=sum(1 for is_valid, _ in results.values() if is_valid),
        )
        
        return results

