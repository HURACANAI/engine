"""
Contract Writer for Per-Coin Training

Writes contracts to Dropbox using DropboxSync.
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import structlog

from .per_coin import (
    RunManifest,
    ChampionPointer,
    FeatureRecipe,
    PerCoinMetrics,
    CostModel,
    Heartbeat,
    FailureReport,
)
from .paths import (
    get_manifest_path,
    get_champion_pointer_path,
    get_model_path,
    get_config_path,
    get_metrics_path,
    get_feature_recipe_path,
    get_heartbeat_path,
    get_failure_report_path,
    format_date_str,
    make_absolute_path,
)

logger = structlog.get_logger(__name__)


class ContractWriter:
    """Writer for per-coin training contracts to Dropbox."""
    
    def __init__(
        self,
        dropbox_sync: Optional[Any] = None,  # DropboxSync from cloud.training.integrations.dropbox_sync
        base_folder: str = "huracan",
    ):
        """Initialize contract writer.
        
        Args:
            dropbox_sync: DropboxSync instance for uploading files
            base_folder: Base folder name in Dropbox (default: "huracan")
        """
        self.dropbox_sync = dropbox_sync
        self.base_folder = base_folder
        logger.info("contract_writer_initialized", base_folder=base_folder)
    
    def write_manifest(
        self,
        manifest: RunManifest,
        date_str: Optional[str] = None,
    ) -> Optional[str]:
        """Write run manifest to Dropbox.
        
        Args:
            manifest: RunManifest instance
            date_str: Date string in YYYYMMDD format (defaults to today)
            
        Returns:
            Dropbox path if successful, None otherwise
        """
        if not self.dropbox_sync:
            logger.warning("dropbox_sync_not_available", message="Cannot write manifest without DropboxSync")
            return None
        
        if date_str is None:
            date_str = format_date_str(manifest.utc_started)
        
        dropbox_path = get_manifest_path(date_str, self.base_folder)
        
        try:
            # Write to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                f.write(manifest.to_json())
                temp_path = f.name
            
            # Upload to Dropbox
            success = self.dropbox_sync.upload_file(
                local_path=temp_path,
                remote_path=dropbox_path,
                overwrite=True,
            )
            
            # Clean up temp file
            Path(temp_path).unlink()
            
            if success:
                logger.info("manifest_written", path=dropbox_path, run_id=manifest.run_id)
                return dropbox_path
            else:
                logger.error("manifest_write_failed", path=dropbox_path)
                return None
                
        except Exception as e:
            logger.error("manifest_write_exception", path=dropbox_path, error=str(e))
            return None
    
    def write_champion_pointer(
        self,
        champion: ChampionPointer,
    ) -> Optional[str]:
        """Write champion pointer to Dropbox.
        
        Args:
            champion: ChampionPointer instance
            
        Returns:
            Dropbox path if successful, None otherwise
        """
        if not self.dropbox_sync:
            logger.warning("dropbox_sync_not_available", message="Cannot write champion pointer without DropboxSync")
            return None
        
        dropbox_path = get_champion_pointer_path(self.base_folder)
        
        try:
            # Update timestamp
            champion.updated_at = datetime.now(timezone.utc)
            
            # Write to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                f.write(champion.to_json())
                temp_path = f.name
            
            # Upload to Dropbox
            success = self.dropbox_sync.upload_file(
                local_path=temp_path,
                remote_path=dropbox_path,
                overwrite=True,
            )
            
            # Clean up temp file
            Path(temp_path).unlink()
            
            if success:
                logger.info("champion_pointer_written", path=dropbox_path, date=champion.date)
                return dropbox_path
            else:
                logger.error("champion_pointer_write_failed", path=dropbox_path)
                return None
                
        except Exception as e:
            logger.error("champion_pointer_write_exception", path=dropbox_path, error=str(e))
            return None
    
    def write_feature_recipe(
        self,
        recipe: FeatureRecipe,
        date_str: Optional[str] = None,
    ) -> Optional[str]:
        """Write feature recipe to Dropbox.
        
        Args:
            recipe: FeatureRecipe instance
            date_str: Date string in YYYYMMDD format (defaults to today)
            
        Returns:
            Dropbox path if successful, None otherwise
        """
        if not self.dropbox_sync:
            logger.warning("dropbox_sync_not_available", message="Cannot write feature recipe without DropboxSync")
            return None
        
        if date_str is None:
            date_str = format_date_str()
        
        dropbox_path = get_feature_recipe_path(date_str, recipe.symbol, self.base_folder)
        
        try:
            # Compute hash if not set
            if not recipe.hash:
                recipe.hash = recipe.compute_hash()
            
            # Write to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                f.write(recipe.to_json())
                temp_path = f.name
            
            # Upload to Dropbox
            success = self.dropbox_sync.upload_file(
                local_path=temp_path,
                remote_path=dropbox_path,
                overwrite=True,
            )
            
            # Clean up temp file
            Path(temp_path).unlink()
            
            if success:
                logger.info("feature_recipe_written", path=dropbox_path, symbol=recipe.symbol)
                return dropbox_path
            else:
                logger.error("feature_recipe_write_failed", path=dropbox_path)
                return None
                
        except Exception as e:
            logger.error("feature_recipe_write_exception", path=dropbox_path, error=str(e))
            return None
    
    def write_metrics(
        self,
        metrics: PerCoinMetrics,
        date_str: Optional[str] = None,
    ) -> Optional[str]:
        """Write metrics to Dropbox.
        
        Args:
            metrics: PerCoinMetrics instance
            date_str: Date string in YYYYMMDD format (defaults to today)
            
        Returns:
            Dropbox path if successful, None otherwise
        """
        if not self.dropbox_sync:
            logger.warning("dropbox_sync_not_available", message="Cannot write metrics without DropboxSync")
            return None
        
        if date_str is None:
            date_str = format_date_str()
        
        dropbox_path = get_metrics_path(date_str, metrics.symbol, self.base_folder)
        
        try:
            # Write to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                f.write(metrics.to_json())
                temp_path = f.name
            
            # Upload to Dropbox
            success = self.dropbox_sync.upload_file(
                local_path=temp_path,
                remote_path=dropbox_path,
                overwrite=True,
            )
            
            # Clean up temp file
            Path(temp_path).unlink()
            
            if success:
                logger.info("metrics_written", path=dropbox_path, symbol=metrics.symbol)
                return dropbox_path
            else:
                logger.error("metrics_write_failed", path=dropbox_path)
                return None
                
        except Exception as e:
            logger.error("metrics_write_exception", path=dropbox_path, error=str(e))
            return None
    
    def write_cost_model(
        self,
        cost_model: CostModel,
        date_str: Optional[str] = None,
    ) -> Optional[str]:
        """Write cost model to Dropbox (stored alongside metrics).
        
        Args:
            cost_model: CostModel instance
            date_str: Date string in YYYYMMDD format (defaults to today)
            
        Returns:
            Dropbox path if successful, None otherwise
        """
        if not self.dropbox_sync:
            logger.warning("dropbox_sync_not_available", message="Cannot write cost model without DropboxSync")
            return None
        
        if date_str is None:
            date_str = format_date_str()
        
        # Store cost model in same directory as metrics
        dropbox_path = get_metrics_path(date_str, cost_model.symbol, self.base_folder).replace("metrics.json", "costs.json")
        
        try:
            # Write to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                f.write(cost_model.to_json())
                temp_path = f.name
            
            # Upload to Dropbox
            success = self.dropbox_sync.upload_file(
                local_path=temp_path,
                remote_path=dropbox_path,
                overwrite=True,
            )
            
            # Clean up temp file
            Path(temp_path).unlink()
            
            if success:
                logger.info("cost_model_written", path=dropbox_path, symbol=cost_model.symbol)
                return dropbox_path
            else:
                logger.error("cost_model_write_failed", path=dropbox_path)
                return None
                
        except Exception as e:
            logger.error("cost_model_write_exception", path=dropbox_path, error=str(e))
            return None
    
    def write_heartbeat(
        self,
        heartbeat: Heartbeat,
    ) -> Optional[str]:
        """Write heartbeat to Dropbox.
        
        Args:
            heartbeat: Heartbeat instance
            
        Returns:
            Dropbox path if successful, None otherwise
        """
        if not self.dropbox_sync:
            logger.warning("dropbox_sync_not_available", message="Cannot write heartbeat without DropboxSync")
            return None
        
        dropbox_path = get_heartbeat_path(self.base_folder)
        
        try:
            # Write to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                f.write(heartbeat.to_json())
                temp_path = f.name
            
            # Upload to Dropbox
            success = self.dropbox_sync.upload_file(
                local_path=temp_path,
                remote_path=dropbox_path,
                overwrite=True,
            )
            
            # Clean up temp file
            Path(temp_path).unlink()
            
            if success:
                logger.debug("heartbeat_written", path=dropbox_path, phase=heartbeat.phase)
                return dropbox_path
            else:
                logger.error("heartbeat_write_failed", path=dropbox_path)
                return None
                
        except Exception as e:
            logger.error("heartbeat_write_exception", path=dropbox_path, error=str(e))
            return None
    
    def write_failure_report(
        self,
        failure_report: FailureReport,
        date_str: Optional[str] = None,
    ) -> Optional[str]:
        """Write failure report to Dropbox.
        
        Args:
            failure_report: FailureReport instance
            date_str: Date string in YYYYMMDD format (defaults to today)
            
        Returns:
            Dropbox path if successful, None otherwise
        """
        if not self.dropbox_sync:
            logger.warning("dropbox_sync_not_available", message="Cannot write failure report without DropboxSync")
            return None
        
        if date_str is None:
            date_str = format_date_str()
        
        dropbox_path = get_failure_report_path(date_str, self.base_folder)
        
        try:
            # Write to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                f.write(failure_report.to_json())
                temp_path = f.name
            
            # Upload to Dropbox
            success = self.dropbox_sync.upload_file(
                local_path=temp_path,
                remote_path=dropbox_path,
                overwrite=True,
            )
            
            # Clean up temp file
            Path(temp_path).unlink()
            
            if success:
                logger.info("failure_report_written", path=dropbox_path, run_id=failure_report.run_id)
                return dropbox_path
            else:
                logger.error("failure_report_write_failed", path=dropbox_path)
                return None
                
        except Exception as e:
            logger.error("failure_report_write_exception", path=dropbox_path, error=str(e))
            return None
    
    def write_model_file(
        self,
        model_path: str,
        symbol: str,
        date_str: Optional[str] = None,
    ) -> Optional[str]:
        """Write model binary file to Dropbox.
        
        Args:
            model_path: Local path to model file
            symbol: Trading symbol
            date_str: Date string in YYYYMMDD format (defaults to today)
            
        Returns:
            Dropbox path if successful, None otherwise
        """
        if not self.dropbox_sync:
            logger.warning("dropbox_sync_not_available", message="Cannot write model file without DropboxSync")
            return None
        
        if date_str is None:
            date_str = format_date_str()
        
        dropbox_path = get_model_path(date_str, symbol, self.base_folder)
        
        try:
            # Upload to Dropbox
            success = self.dropbox_sync.upload_file(
                local_path=model_path,
                remote_path=dropbox_path,
                overwrite=True,
            )
            
            if success:
                logger.info("model_file_written", path=dropbox_path, symbol=symbol)
                return dropbox_path
            else:
                logger.error("model_file_write_failed", path=dropbox_path)
                return None
                
        except Exception as e:
            logger.error("model_file_write_exception", path=dropbox_path, error=str(e))
            return None

