"""
Mechanic Service for Per-Coin Challenger Creation and Promotion

Creates challengers per symbol, shadows them, and promotes based on rules.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import structlog  # type: ignore[import-untyped]

from src.shared.contracts.per_coin import (
    ChampionPointer,
    PerCoinMetrics,
    FeatureRecipe,
)
from src.shared.contracts.paths import (
    get_champion_pointer_path,
    get_promotions_log_path,
    format_date_str,
)
from src.shared.contracts.writer import ContractWriter

if TYPE_CHECKING:
    from ..integrations.dropbox_sync import DropboxSync
    from ..config.settings import EngineSettings

logger = structlog.get_logger(__name__)


@dataclass
class PromotionEntry:
    """Entry in promotions log."""
    timestamp: datetime
    symbol: str
    champion_metrics: Dict[str, Any]
    challenger_metrics: Dict[str, Any]
    promotion_reason: str
    before_model_path: str
    after_model_path: str


class MechanicService:
    """Service for creating challengers and promoting per-coin models."""
    
    def __init__(
        self,
        dropbox_sync: Optional["DropboxSync"] = None,
        base_folder: str = "huracan",
        settings: Optional["EngineSettings"] = None,
    ):
        """Initialize Mechanic service.
        
        Args:
            dropbox_sync: DropboxSync instance for reading/writing files
            base_folder: Base folder name in Dropbox (default: "huracan")
            settings: Engine settings for promotion rules
        """
        self.contract_writer = ContractWriter(dropbox_sync=dropbox_sync, base_folder=base_folder)
        self.dropbox_sync = dropbox_sync
        self.base_folder = base_folder
        self.settings = settings
        
        # Promotion rules from settings
        if settings and settings.training.per_coin.promotion_rules:
            self.promotion_rules = settings.training.per_coin.promotion_rules
        else:
            # Default promotion rules
            self.promotion_rules = type('obj', (object,), {
                'min_hit_rate_improvement': 0.01,
                'min_sharpe_improvement': 0.2,
                'max_drawdown_tolerance': 0.0,
                'min_net_pnl_improvement': 0.01,
            })()
        
        logger.info("mechanic_service_initialized", base_folder=base_folder)
    
    def load_champion_pointer(self) -> Optional[ChampionPointer]:
        """Load champion pointer from Dropbox.
        
        Returns:
            ChampionPointer instance if found, None otherwise
        """
        if not self.dropbox_sync:
            logger.warning("dropbox_sync_not_available", message="Cannot load champion pointer without DropboxSync")
            return None
        
        dropbox_path = get_champion_pointer_path(self.base_folder)
        
        try:
            # Download from Dropbox
            _, response = self.dropbox_sync._dbx.files_download(dropbox_path)
            json_str = response.content.decode('utf-8')
            champion = ChampionPointer.from_json(json_str)
            logger.info("champion_pointer_loaded", path=dropbox_path, symbols=list(champion.models.keys()))
            return champion
        except Exception as e:
            logger.warning("champion_pointer_not_found", path=dropbox_path, error=str(e))
            return None
    
    def create_challengers(
        self,
        symbol: str,
        last_n_hours: int = 24,
        max_challengers: int = 3,
    ) -> List[Dict[str, Any]]:
        """Create up to N challengers per symbol from the last N hours.
        
        Args:
            symbol: Trading symbol
            last_n_hours: Number of hours to look back for challengers
            max_challengers: Maximum number of challengers to create
            
        Returns:
            List of challenger dictionaries with model paths and metadata
        """
        challengers = []
        
        # In a real implementation, this would:
        # 1. Look for recent model artifacts in Dropbox (last N hours)
        # 2. Load model metadata and metrics
        # 3. Filter by symbol
        # 4. Return up to max_challengers
        
        # For now, return empty list (would be implemented by reading from Dropbox)
        logger.info("challengers_created", symbol=symbol, count=len(challengers))
        return challengers
    
    def shadow_test_challenger(
        self,
        symbol: str,
        challenger_model_path: str,
        recent_data: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Shadow test challenger on recent unseen data.
        
        Args:
            symbol: Trading symbol
            challenger_model_path: Path to challenger model
            recent_data: Recent market data for testing
            
        Returns:
            Dictionary with shadow test results (metrics, trades, etc.)
        """
        # In a real implementation, this would:
        # 1. Load challenger model
        # 2. Load recent unseen data
        # 3. Run predictions on recent data
        # 4. Calculate metrics (sharpe, hit_rate, drawdown, net_pnl)
        # 5. Return results
        
        # For now, return empty dict (would be implemented with actual backtesting)
        logger.info("challenger_shadow_tested", symbol=symbol, model_path=challenger_model_path)
        return {
            "sharpe": 0.0,
            "hit_rate": 0.0,
            "max_drawdown_pct": 0.0,
            "net_pnl_pct": 0.0,
            "sample_size": 0,
        }
    
    def should_promote(
        self,
        symbol: str,
        champion_metrics: Dict[str, Any],
        challenger_metrics: Dict[str, Any],
    ) -> tuple[bool, str]:
        """Check if challenger should be promoted based on promotion rules.
        
        Args:
            symbol: Trading symbol
            champion_metrics: Champion metrics dictionary
            challenger_metrics: Challenger metrics dictionary
            
        Returns:
            Tuple of (should_promote, reason)
        """
        champion_hit_rate = champion_metrics.get("hit_rate", 0.0)
        challenger_hit_rate = challenger_metrics.get("hit_rate", 0.0)
        
        champion_sharpe = champion_metrics.get("sharpe", 0.0)
        challenger_sharpe = challenger_metrics.get("sharpe", 0.0)
        
        champion_drawdown = champion_metrics.get("max_drawdown_pct", 0.0)
        challenger_drawdown = challenger_metrics.get("max_drawdown_pct", 0.0)
        
        champion_net_pnl = champion_metrics.get("net_pnl_pct", 0.0)
        challenger_net_pnl = challenger_metrics.get("net_pnl_pct", 0.0)
        
        challenger_sample_size = challenger_metrics.get("sample_size", 0)
        min_sample_size = getattr(self.promotion_rules, 'min_sample_size', 100) if hasattr(self.promotion_rules, 'min_sample_size') else 100
        
        # Check promotion rules
        reasons = []
        
        # Hit rate improvement
        hit_rate_improvement = challenger_hit_rate - champion_hit_rate
        min_hit_rate_improvement = getattr(self.promotion_rules, 'min_hit_rate_improvement', 0.01)
        if hit_rate_improvement >= min_hit_rate_improvement:
            reasons.append(f"hit_rate_improvement: {hit_rate_improvement:.2%}")
        elif hit_rate_improvement < min_hit_rate_improvement:
            return False, f"hit_rate_improvement insufficient: {hit_rate_improvement:.2%} < {min_hit_rate_improvement:.2%}"
        
        # Sharpe improvement
        sharpe_improvement = challenger_sharpe - champion_sharpe
        min_sharpe_improvement = getattr(self.promotion_rules, 'min_sharpe_improvement', 0.2)
        if sharpe_improvement >= min_sharpe_improvement:
            reasons.append(f"sharpe_improvement: {sharpe_improvement:.2f}")
        elif sharpe_improvement < min_sharpe_improvement:
            return False, f"sharpe_improvement insufficient: {sharpe_improvement:.2f} < {min_sharpe_improvement:.2f}"
        
        # Drawdown tolerance
        max_drawdown_tolerance = getattr(self.promotion_rules, 'max_drawdown_tolerance', 0.0)
        drawdown_diff = challenger_drawdown - champion_drawdown
        if drawdown_diff <= max_drawdown_tolerance:
            reasons.append(f"drawdown_acceptable: {challenger_drawdown:.2%} <= {champion_drawdown:.2%}")
        else:
            return False, f"drawdown_too_high: {challenger_drawdown:.2%} > {champion_drawdown:.2%}"
        
        # Net P&L improvement
        net_pnl_improvement = challenger_net_pnl - champion_net_pnl
        min_net_pnl_improvement = getattr(self.promotion_rules, 'min_net_pnl_improvement', 0.01)
        if net_pnl_improvement >= min_net_pnl_improvement:
            reasons.append(f"net_pnl_improvement: {net_pnl_improvement:.2%}")
        elif net_pnl_improvement < min_net_pnl_improvement:
            return False, f"net_pnl_improvement insufficient: {net_pnl_improvement:.2%} < {min_net_pnl_improvement:.2%}"
        
        # Sample size check
        if challenger_sample_size < min_sample_size:
            return False, f"sample_size_too_small: {challenger_sample_size} < {min_sample_size}"
        
        reason = "; ".join(reasons)
        return True, reason
    
    def promote_champion(
        self,
        symbol: str,
        challenger_model_path: str,
        challenger_metrics: Dict[str, Any],
        date_str: Optional[str] = None,
    ) -> bool:
        """Promote challenger to champion for a symbol.
        
        Args:
            symbol: Trading symbol
            challenger_model_path: Path to challenger model
            challenger_metrics: Challenger metrics
            date_str: Date string in YYYYMMDD format (defaults to today)
            
        Returns:
            True if promotion successful, False otherwise
        """
        if not self.dropbox_sync:
            logger.warning("dropbox_sync_not_available", message="Cannot promote champion without DropboxSync")
            return False
        
        if date_str is None:
            date_str = format_date_str()
        
        # Load current champion pointer
        champion_pointer = self.load_champion_pointer()
        
        if not champion_pointer:
            # Create new champion pointer
            champion_pointer = ChampionPointer(
                date=date_str,
                run_id=f"mechanic_{date_str}",
                models={},
                default_costs_bps=15.0,
            )
            champion_metrics = {}
        else:
            # Get current champion metrics for this symbol
            # In a real implementation, would load from metrics.json
            champion_metrics = {}
        
        # Make challenger model path absolute
        absolute_model_path = challenger_model_path
        if not challenger_model_path.startswith("/"):
            absolute_model_path = f"/{self.base_folder}/{challenger_model_path}"
        
        # Check if should promote
        should_promote, reason = self.should_promote(
            symbol=symbol,
            champion_metrics=champion_metrics,
            challenger_metrics=challenger_metrics,
        )
        
        if not should_promote:
            logger.info("challenger_not_promoted", symbol=symbol, reason=reason)
            return False
        
        # Update champion pointer
        old_model_path = champion_pointer.models.get(symbol)
        champion_pointer.models[symbol] = absolute_model_path
        champion_pointer.date = date_str
        champion_pointer.updated_at = datetime.now(timezone.utc)
        
        # Write updated champion pointer
        champion_path = self.contract_writer.write_champion_pointer(champion_pointer)
        
        if champion_path:
            # Write promotion log entry
            promotion_entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "symbol": symbol,
                "champion_metrics": champion_metrics,
                "challenger_metrics": challenger_metrics,
                "promotion_reason": reason,
                "before_model_path": old_model_path or "none",
                "after_model_path": absolute_model_path,
            }
            
            self.append_promotion_log(promotion_entry)
            
            logger.info("champion_promoted", symbol=symbol, reason=reason, model_path=absolute_model_path)
            return True
        else:
            logger.error("champion_promotion_failed", symbol=symbol, reason="failed_to_write_champion_pointer")
            return False
    
    def append_promotion_log(self, promotion_entry: Dict[str, Any]) -> bool:
        """Append promotion entry to promotions log.
        
        Args:
            promotion_entry: Promotion entry dictionary
            
        Returns:
            True if successful, False otherwise
        """
        if not self.dropbox_sync:
            logger.warning("dropbox_sync_not_available", message="Cannot append promotion log without DropboxSync")
            return False
        
        dropbox_path = get_promotions_log_path(self.base_folder)
        
        try:
            # Try to load existing promotions log
            try:
                _, response = self.dropbox_sync._dbx.files_download(dropbox_path)
                existing_log = json.loads(response.content.decode('utf-8'))
                promotions = existing_log.get("promotions", [])
            except Exception:
                # Create new promotions log
                promotions = []
            
            # Append new promotion entry
            promotions.append(promotion_entry)
            
            # Write updated promotions log
            log_data = {
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "promotions": promotions,
            }
            
            # Write to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(log_data, f, indent=2)
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
                logger.info("promotion_log_appended", symbol=promotion_entry.get("symbol"))
                return True
            else:
                logger.error("promotion_log_append_failed", path=dropbox_path)
                return False
                
        except Exception as e:
            logger.error("promotion_log_append_exception", path=dropbox_path, error=str(e))
            return False

