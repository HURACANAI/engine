"""
Per-Coin Training Service

Wraps training logic to export per-coin artifacts and contracts.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import structlog

from src.shared.contracts.per_coin import (
    RunManifest,
    ChampionPointer,
    FeatureRecipe,
    PerCoinMetrics,
    CostModel,
    Heartbeat,
    FailureReport,
)
from src.shared.contracts.paths import (
    format_date_str,
    get_model_path,
    make_absolute_path,
)
from src.shared.contracts.writer import ContractWriter

if TYPE_CHECKING:
    from ..integrations.dropbox_sync import DropboxSync  # type: ignore[import-untyped]
    from ..services.costs import CostBreakdown  # type: ignore[import-untyped]
    from ..services.orchestration import TrainingTaskResult  # type: ignore[import-untyped]

logger = structlog.get_logger(__name__)


class PerCoinTrainingService:
    """Service for per-coin training with contract export."""
    
    def __init__(
        self,
        dropbox_sync: Optional["DropboxSync"] = None,
        base_folder: str = "huracan",
        engine_version: str = "1.0.0",
    ):
        """Initialize per-coin training service.
        
        Args:
            dropbox_sync: DropboxSync instance for uploading files
            base_folder: Base folder name in Dropbox (default: "huracan")
            engine_version: Engine version string
        """
        self.contract_writer = ContractWriter(dropbox_sync=dropbox_sync, base_folder=base_folder)
        self.base_folder = base_folder
        self.engine_version = engine_version
        logger.info("per_coin_training_service_initialized", base_folder=base_folder)
    
    def create_run_manifest(
        self,
        run_id: str,
        utc_started: datetime,
        symbols_trained: List[str],
        artifacts_map: Dict[str, str],
        metrics_map: Dict[str, Dict[str, Any]],
        costs_map: Dict[str, Dict[str, Any]],
        feature_recipe_hash_map: Dict[str, str],
        status: str = "ok",
        failure_reason: Optional[str] = None,
        utc_finished: Optional[datetime] = None,
    ) -> RunManifest:
        """Create run manifest from training results.
        
        Args:
            run_id: Run ID
            utc_started: Start timestamp
            symbols_trained: List of symbols trained
            artifacts_map: Map of symbol to model path
            metrics_map: Map of symbol to metrics
            costs_map: Map of symbol to cost object
            feature_recipe_hash_map: Map of symbol to feature recipe hash
            status: Status ("ok" or "failed")
            failure_reason: Failure reason if status is "failed"
            utc_finished: Finish timestamp
            
        Returns:
            RunManifest instance
        """
        return RunManifest(
            run_id=run_id,
            utc_started=utc_started,
            utc_finished=utc_finished,
            engine_version=self.engine_version,
            symbols_trained=symbols_trained,
            artifacts_map=artifacts_map,
            metrics_map=metrics_map,
            costs_map=costs_map,
            feature_recipe_hash_map=feature_recipe_hash_map,
            status=status,
            failure_reason=failure_reason,
        )
    
    def create_champion_pointer(
        self,
        date_str: str,
        run_id: str,
        models: Dict[str, str],
        default_costs_bps: float = 15.0,
        code_hash: Optional[str] = None,
    ) -> ChampionPointer:
        """Create champion pointer from training results.
        
        Args:
            date_str: Date string in YYYYMMDD format
            run_id: Run ID
            models: Map of symbol to absolute model path
            default_costs_bps: Default costs in basis points
            code_hash: Code hash for reproducibility
            
        Returns:
            ChampionPointer instance
        """
        # Make paths absolute
        absolute_models = {
            symbol: make_absolute_path(path, f"/{self.base_folder}")
            for symbol, path in models.items()
        }
        
        return ChampionPointer(
            date=date_str,
            run_id=run_id,
            models=absolute_models,
            default_costs_bps=default_costs_bps,
            code_hash=code_hash,
        )
    
    def create_feature_recipe(
        self,
        symbol: str,
        timeframes: Optional[List[str]] = None,
        indicators: Optional[Dict[str, Dict[str, Any]]] = None,
        fill_rules: Optional[Dict[str, str]] = None,
        normalization: Optional[Dict[str, Any]] = None,
        window_sizes: Optional[Dict[str, int]] = None,
    ) -> FeatureRecipe:
        """Create feature recipe for a symbol.
        
        Args:
            symbol: Trading symbol
            timeframes: List of timeframes
            indicators: Indicator configuration
            fill_rules: Fill rules
            normalization: Normalization configuration
            window_sizes: Window sizes
            
        Returns:
            FeatureRecipe instance
        """
        return FeatureRecipe(
            symbol=symbol,
            timeframes=timeframes or ["1h"],
            indicators=indicators or {},
            fill_rules=fill_rules or {"strategy": "forward_fill"},
            normalization=normalization or {"type": "standard", "scaler": "StandardScaler"},
            window_sizes=window_sizes or {"lookback": 100, "prediction": 1},
        )
    
    def create_per_coin_metrics(
        self,
        symbol: str,
        sample_size: int,
        gross_pnl_pct: float,
        net_pnl_pct: float,
        sharpe: float,
        hit_rate: float,
        max_drawdown_pct: float,
        avg_trade_bps: float,
        validation_windows: Optional[List[str]] = None,
        costs_bps_used: Optional[Dict[str, float]] = None,
    ) -> PerCoinMetrics:
        """Create per-coin metrics from training results.
        
        Args:
            symbol: Trading symbol
            sample_size: Sample size
            gross_pnl_pct: Gross P&L percentage
            net_pnl_pct: Net P&L percentage
            sharpe: Sharpe ratio
            hit_rate: Hit rate (0.0 to 1.0)
            max_drawdown_pct: Maximum drawdown percentage
            avg_trade_bps: Average trade in basis points
            validation_windows: Validation windows
            costs_bps_used: Cost breakdown
            
        Returns:
            PerCoinMetrics instance
        """
        return PerCoinMetrics(
            symbol=symbol,
            sample_size=sample_size,
            gross_pnl_pct=gross_pnl_pct,
            net_pnl_pct=net_pnl_pct,
            sharpe=sharpe,
            hit_rate=hit_rate,
            max_drawdown_pct=max_drawdown_pct,
            avg_trade_bps=avg_trade_bps,
            validation_windows=validation_windows or [],
            costs_bps_used=costs_bps_used or {},
        )
    
    def create_cost_model(
        self,
        symbol: str,
        taker_fee_bps: float,
        maker_fee_bps: float,
        median_spread_bps: float,
        slippage_bps_per_sigma: float,
        min_notional: float,
        step_size: float,
        last_updated_utc: Optional[datetime] = None,
    ) -> CostModel:
        """Create cost model for a symbol.
        
        Args:
            symbol: Trading symbol
            taker_fee_bps: Taker fee in basis points
            maker_fee_bps: Maker fee in basis points
            median_spread_bps: Median spread in basis points
            slippage_bps_per_sigma: Slippage in basis points per sigma
            min_notional: Minimum notional
            step_size: Step size
            last_updated_utc: Last updated timestamp
            
        Returns:
            CostModel instance
        """
        return CostModel(
            symbol=symbol,
            taker_fee_bps=taker_fee_bps,
            maker_fee_bps=maker_fee_bps,
            median_spread_bps=median_spread_bps,
            slippage_bps_per_sigma=slippage_bps_per_sigma,
            min_notional=min_notional,
            step_size=step_size,
            last_updated_utc=last_updated_utc or datetime.now(timezone.utc),
        )
    
    def cost_breakdown_to_cost_model(
        self,
        symbol: str,
        cost_breakdown: "CostBreakdown",
        min_notional: float = 10.0,
        step_size: float = 0.001,
    ) -> CostModel:
        """Convert CostBreakdown to CostModel.
        
        Args:
            symbol: Trading symbol
            cost_breakdown: CostBreakdown instance
            min_notional: Minimum notional
            step_size: Step size
            
        Returns:
            CostModel instance
        """
        return self.create_cost_model(
            symbol=symbol,
            taker_fee_bps=cost_breakdown.fee_bps if hasattr(cost_breakdown, "fee_bps") else 10.0,
            maker_fee_bps=cost_breakdown.fee_bps * 0.5 if hasattr(cost_breakdown, "fee_bps") else 5.0,
            median_spread_bps=cost_breakdown.spread_bps if hasattr(cost_breakdown, "spread_bps") else 5.0,
            slippage_bps_per_sigma=cost_breakdown.slippage_bps if hasattr(cost_breakdown, "slippage_bps") else 2.0,
            min_notional=min_notional,
            step_size=step_size,
        )
    
    def training_result_to_metrics(
        self,
        symbol: str,
        result: "TrainingTaskResult",
        sample_size: int,
    ) -> PerCoinMetrics:
        """Convert TrainingTaskResult to PerCoinMetrics.
        
        Args:
            symbol: Trading symbol
            result: TrainingTaskResult instance
            sample_size: Sample size
            
        Returns:
            PerCoinMetrics instance
        """
        metrics = result.metrics
        
        # Extract metrics
        sharpe = metrics.get("sharpe", 0.0)
        hit_rate = metrics.get("hit_rate", 0.0)
        max_drawdown_pct = abs(metrics.get("max_dd_bps", 0.0)) / 100.0  # Convert bps to percentage
        pnl_bps = metrics.get("pnl_bps", 0.0)
        gross_pnl_pct = pnl_bps / 100.0  # Convert bps to percentage
        
        # Calculate net P&L after costs
        total_costs_bps = metrics.get("total_costs_bps", 0.0)
        net_pnl_pct = (pnl_bps - total_costs_bps) / 100.0
        
        # Average trade in bps
        trades_oos = metrics.get("trades_oos", 0)
        avg_trade_bps = pnl_bps / trades_oos if trades_oos > 0 else 0.0
        
        # Cost breakdown
        costs_bps_used = {}
        if hasattr(result.costs, "fee_bps"):
            costs_bps_used["fee_bps"] = result.costs.fee_bps
        if hasattr(result.costs, "spread_bps"):
            costs_bps_used["spread_bps"] = result.costs.spread_bps
        if hasattr(result.costs, "slippage_bps"):
            costs_bps_used["slippage_bps"] = result.costs.slippage_bps
        costs_bps_used["total_bps"] = total_costs_bps
        
        return self.create_per_coin_metrics(
            symbol=symbol,
            sample_size=sample_size,
            gross_pnl_pct=gross_pnl_pct,
            net_pnl_pct=net_pnl_pct,
            sharpe=sharpe,
            hit_rate=hit_rate,
            max_drawdown_pct=max_drawdown_pct,
            avg_trade_bps=avg_trade_bps,
            costs_bps_used=costs_bps_used,
        )
    
    def export_training_result(
        self,
        symbol: str,
        result: "TrainingTaskResult",
        date_str: str,
        feature_recipe: Optional[FeatureRecipe] = None,
        sample_size: int = 0,
    ) -> Dict[str, Optional[str]]:
        """Export training result as per-coin artifacts.
        
        Args:
            symbol: Trading symbol
            result: TrainingTaskResult instance
            date_str: Date string in YYYYMMDD format
            feature_recipe: Feature recipe (optional)
            sample_size: Sample size
            
        Returns:
            Dictionary with exported paths
        """
        exported_paths = {
            "model": None,
            "metrics": None,
            "costs": None,
            "feature_recipe": None,
        }
        
        # Export model file if available
        if result.artifacts and result.artifacts.model_path:
            model_path = self.contract_writer.write_model_file(
                model_path=result.artifacts.model_path,
                symbol=symbol,
                date_str=date_str,
            )
            exported_paths["model"] = model_path
        
        # Export metrics
        metrics = self.training_result_to_metrics(symbol, result, sample_size)
        metrics_path = self.contract_writer.write_metrics(metrics, date_str)
        exported_paths["metrics"] = metrics_path
        
        # Export cost model
        cost_model = self.cost_breakdown_to_cost_model(symbol, result.costs)
        costs_path = self.contract_writer.write_cost_model(cost_model, date_str)
        exported_paths["costs"] = costs_path
        
        # Export feature recipe if provided
        if feature_recipe:
            recipe_path = self.contract_writer.write_feature_recipe(feature_recipe, date_str)
            exported_paths["feature_recipe"] = recipe_path
        
        return exported_paths
    
    def write_heartbeat(
        self,
        phase: str,
        current_symbol: Optional[str] = None,
        progress: float = 0.0,
        last_error: Optional[str] = None,
    ) -> Optional[str]:
        """Write heartbeat to Dropbox.
        
        Args:
            phase: Current phase ("loading", "training", "validating", "publishing", "complete", "failed")
            current_symbol: Current symbol being processed
            progress: Progress (0.0 to 1.0)
            last_error: Last error message if any
            
        Returns:
            Dropbox path if successful, None otherwise
        """
        heartbeat = Heartbeat(
            utc_timestamp=datetime.now(timezone.utc),
            phase=phase,
            current_symbol=current_symbol,
            progress=progress,
            last_error=last_error,
        )
        return self.contract_writer.write_heartbeat(heartbeat)
    
    def write_failure_report(
        self,
        run_id: str,
        step: str,
        exception_type: str,
        message: str,
        last_files_written: Optional[List[str]] = None,
        suggestions: Optional[List[str]] = None,
        date_str: Optional[str] = None,
    ) -> Optional[str]:
        """Write failure report to Dropbox.
        
        Args:
            run_id: Run ID
            step: Step where failure occurred
            exception_type: Exception type
            message: Error message
            last_files_written: List of last files written
            suggestions: List of suggestions
            date_str: Date string in YYYYMMDD format (defaults to today)
            
        Returns:
            Dropbox path if successful, None otherwise
        """
        failure_report = FailureReport(
            run_id=run_id,
            step=step,
            exception_type=exception_type,
            message=message,
            last_files_written=last_files_written or [],
            suggestions=suggestions or [],
        )
        return self.contract_writer.write_failure_report(failure_report, date_str)

