"""
Dropbox Publisher for Training Architecture

Publishes champion models and reports to Dropbox with manifest-driven folder structure.

Author: Huracan Engine Team
Date: 2025-01-27
"""

from __future__ import annotations

import json
import hashlib
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import structlog

logger = structlog.get_logger(__name__)

try:
    import dropbox
    from dropbox.exceptions import ApiError, AuthError
    DROPBOX_AVAILABLE = True
except ImportError:
    DROPBOX_AVAILABLE = False
    logger.warning("dropbox not available, using file system fallback")


@dataclass
class ModelManifest:
    """Model manifest."""
    coin: str
    horizon: str
    version: int
    training_window_start: str
    training_window_end: str
    features_hash: str
    code_hash: str
    timestamp: str
    model_path: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    regime: Optional[str] = None
    engine_type: Optional[str] = None


@dataclass
class ExportBundle:
    """Export bundle for Dropbox."""
    coin: str
    horizon: str
    champion_model: Optional[Path] = None
    metrics_report: Optional[Path] = None
    cost_report: Optional[Path] = None
    decision_logs: Optional[Path] = None
    regime_map: Optional[Path] = None
    data_integrity_report: Optional[Path] = None
    model_manifest: Optional[ModelManifest] = None


class DropboxPublisher:
    """
    Dropbox publisher for model and report exports.
    
    Features:
    - Manifest-driven folder structure
    - Champion model export
    - Report export (metrics, costs, regime, logs, manifest)
    - Versioning
    - Error handling
    """
    
    def __init__(
        self,
        access_token: str,
        base_path: str = "/HuracanEngine",
        dry_run: bool = False,
    ):
        """
        Initialize Dropbox publisher.
        
        Args:
            access_token: Dropbox access token
            base_path: Base path in Dropbox
            dry_run: Whether to perform dry run (no actual upload)
        """
        self.access_token = access_token
        self.base_path = base_path
        self.dry_run = dry_run
        
        if DROPBOX_AVAILABLE and not dry_run:
            self.client = dropbox.Dropbox(access_token)
        else:
            self.client = None
        
        logger.info(
            "dropbox_publisher_initialized",
            base_path=base_path,
            dry_run=dry_run,
        )
    
    def _get_manifest_path(self, coin: str, horizon: str, version: int) -> str:
        """Get manifest path in Dropbox."""
        return f"{self.base_path}/models/{coin}/{horizon}/v{version}/manifest.json"
    
    def _get_model_path(self, coin: str, horizon: str, version: int) -> str:
        """Get model path in Dropbox."""
        return f"{self.base_path}/models/{coin}/{horizon}/v{version}/model.pkl"
    
    def _get_metrics_path(self, coin: str, horizon: str, version: int) -> str:
        """Get metrics path in Dropbox."""
        return f"{self.base_path}/models/{coin}/{horizon}/v{version}/metrics.json"
    
    def _get_cost_report_path(self, coin: str, horizon: str, version: int) -> str:
        """Get cost report path in Dropbox."""
        return f"{self.base_path}/models/{coin}/{horizon}/v{version}/cost_report.json"
    
    def _get_decision_logs_path(self, coin: str, horizon: str, version: int) -> str:
        """Get decision logs path in Dropbox."""
        return f"{self.base_path}/models/{coin}/{horizon}/v{version}/decision_logs.json"
    
    def _get_regime_map_path(self, coin: str, horizon: str, version: int) -> str:
        """Get regime map path in Dropbox."""
        return f"{self.base_path}/models/{coin}/{horizon}/v{version}/regime_map.json"
    
    def _get_data_integrity_path(self, coin: str, horizon: str, version: int) -> str:
        """Get data integrity report path in Dropbox."""
        return f"{self.base_path}/models/{coin}/{horizon}/v{version}/data_integrity.json"
    
    def _upload_file(self, local_path: Path, dropbox_path: str) -> bool:
        """
        Upload a file to Dropbox.
        
        Args:
            local_path: Local file path
            dropbox_path: Dropbox file path
        
        Returns:
            True if successful
        """
        if self.dry_run:
            logger.info("dry_run_upload", local_path=str(local_path), dropbox_path=dropbox_path)
            return True
        
        if not self.client:
            logger.error("dropbox_client_not_available")
            return False
        
        if not local_path.exists():
            logger.error("file_not_found", local_path=str(local_path))
            return False
        
        try:
            with open(local_path, "rb") as f:
                file_content = f.read()
            
            self.client.files_upload(
                file_content,
                dropbox_path,
                mode=dropbox.files.WriteMode.overwrite,
            )
            
            logger.info("file_uploaded", local_path=str(local_path), dropbox_path=dropbox_path)
            return True
        
        except ApiError as e:
            logger.error("dropbox_api_error", error=str(e), dropbox_path=dropbox_path)
            return False
        except Exception as e:
            logger.error("upload_error", error=str(e), local_path=str(local_path))
            return False
    
    def _upload_json(self, data: Dict[str, Any], dropbox_path: str) -> bool:
        """
        Upload JSON data to Dropbox.
        
        Args:
            data: JSON data
            dropbox_path: Dropbox file path
        
        Returns:
            True if successful
        """
        if self.dry_run:
            logger.info("dry_run_upload_json", dropbox_path=dropbox_path, data_keys=list(data.keys()))
            return True
        
        if not self.client:
            logger.error("dropbox_client_not_available")
            return False
        
        try:
            json_content = json.dumps(data, indent=2).encode("utf-8")
            
            self.client.files_upload(
                json_content,
                dropbox_path,
                mode=dropbox.files.WriteMode.overwrite,
            )
            
            logger.info("json_uploaded", dropbox_path=dropbox_path)
            return True
        
        except ApiError as e:
            logger.error("dropbox_api_error", error=str(e), dropbox_path=dropbox_path)
            return False
        except Exception as e:
            logger.error("upload_error", error=str(e), dropbox_path=dropbox_path)
            return False
    
    def publish_bundle(self, bundle: ExportBundle) -> bool:
        """
        Publish export bundle to Dropbox.
        
        Args:
            bundle: Export bundle
        
        Returns:
            True if successful
        """
        if not bundle.model_manifest:
            logger.error("model_manifest_missing", coin=bundle.coin, horizon=bundle.horizon)
            return False
        
        manifest = bundle.model_manifest
        version = manifest.version
        
        logger.info(
            "publishing_bundle",
            coin=bundle.coin,
            horizon=bundle.horizon,
            version=version,
        )
        
        success = True
        
        # Upload champion model
        if bundle.champion_model:
            model_path = self._get_model_path(bundle.coin, bundle.horizon, version)
            if not self._upload_file(bundle.champion_model, model_path):
                success = False
        
        # Upload metrics report
        if bundle.metrics_report:
            metrics_path = self._get_metrics_path(bundle.coin, bundle.horizon, version)
            if bundle.metrics_report.suffix == ".json":
                with open(bundle.metrics_report, "r") as f:
                    metrics_data = json.load(f)
                if not self._upload_json(metrics_data, metrics_path):
                    success = False
            else:
                if not self._upload_file(bundle.metrics_report, metrics_path):
                    success = False
        
        # Upload cost report
        if bundle.cost_report:
            cost_path = self._get_cost_report_path(bundle.coin, bundle.horizon, version)
            if bundle.cost_report.suffix == ".json":
                with open(bundle.cost_report, "r") as f:
                    cost_data = json.load(f)
                if not self._upload_json(cost_data, cost_path):
                    success = False
            else:
                if not self._upload_file(bundle.cost_report, cost_path):
                    success = False
        
        # Upload decision logs
        if bundle.decision_logs:
            logs_path = self._get_decision_logs_path(bundle.coin, bundle.horizon, version)
            if bundle.decision_logs.suffix == ".json":
                with open(bundle.decision_logs, "r") as f:
                    logs_data = json.load(f)
                if not self._upload_json(logs_data, logs_path):
                    success = False
            else:
                if not self._upload_file(bundle.decision_logs, logs_path):
                    success = False
        
        # Upload regime map
        if bundle.regime_map:
            regime_path = self._get_regime_map_path(bundle.coin, bundle.horizon, version)
            if bundle.regime_map.suffix == ".json":
                with open(bundle.regime_map, "r") as f:
                    regime_data = json.load(f)
                if not self._upload_json(regime_data, regime_path):
                    success = False
            else:
                if not self._upload_file(bundle.regime_map, regime_path):
                    success = False
        
        # Upload data integrity report
        if bundle.data_integrity_report:
            integrity_path = self._get_data_integrity_path(bundle.coin, bundle.horizon, version)
            if bundle.data_integrity_report.suffix == ".json":
                with open(bundle.data_integrity_report, "r") as f:
                    integrity_data = json.load(f)
                if not self._upload_json(integrity_data, integrity_path):
                    success = False
            else:
                if not self._upload_file(bundle.data_integrity_report, integrity_path):
                    success = False
        
        # Upload model manifest
        manifest_path = self._get_manifest_path(bundle.coin, bundle.horizon, version)
        manifest_data = asdict(manifest)
        if not self._upload_json(manifest_data, manifest_path):
            success = False
        
        if success:
            logger.info(
                "bundle_published",
                coin=bundle.coin,
                horizon=bundle.horizon,
                version=version,
            )
        else:
            logger.error(
                "bundle_publish_failed",
                coin=bundle.coin,
                horizon=bundle.horizon,
                version=version,
            )
        
        return success
    
    def publish_summary(self, summary: Dict[str, Any]) -> bool:
        """
        Publish daily summary to Dropbox.
        
        Args:
            summary: Summary data
        
        Returns:
            True if successful
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_path = f"{self.base_path}/summaries/{timestamp}_summary.json"
        
        logger.info("publishing_summary", summary_path=summary_path)
        
        return self._upload_json(summary, summary_path)
    
    def list_models(self, coin: Optional[str] = None, horizon: Optional[str] = None) -> List[str]:
        """
        List models in Dropbox.
        
        Args:
            coin: Coin symbol (optional)
            horizon: Horizon (optional)
        
        Returns:
            List of model paths
        """
        if self.dry_run or not self.client:
            return []
        
        try:
            if coin and horizon:
                path = f"{self.base_path}/models/{coin}/{horizon}"
            elif coin:
                path = f"{self.base_path}/models/{coin}"
            else:
                path = f"{self.base_path}/models"
            
            result = self.client.files_list_folder(path)
            return [entry.path_lower for entry in result.entries if isinstance(entry, dropbox.files.FileMetadata)]
        
        except ApiError as e:
            logger.error("list_models_error", error=str(e), path=path)
            return []


def compute_code_hash(code_path: Path) -> str:
    """Compute code hash for reproducibility."""
    if not code_path.exists():
        return hashlib.sha256(str(time.time()).encode()).hexdigest()[:16]
    
    with open(code_path, "rb") as f:
        code_content = f.read()
    
    return hashlib.sha256(code_content).hexdigest()[:16]


def compute_features_hash(features: Dict[str, Any]) -> str:
    """Compute features hash for reproducibility."""
    features_str = json.dumps(features, sort_keys=True)
    return hashlib.sha256(features_str.encode()).hexdigest()[:16]

