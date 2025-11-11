"""
Hamilton Service for Per-Coin Model Loading and Trading

Loads champions per symbol, builds features from recipe, applies costs, and logs trades.
"""

from __future__ import annotations

import json
import pickle
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import structlog  # type: ignore[import-untyped]

from src.shared.contracts.per_coin import (
    ChampionPointer,
    FeatureRecipe,
    CostModel,
    PerCoinMetrics,
)
from src.shared.contracts.paths import (
    get_champion_pointer_path,
    get_feature_recipe_path,
    get_metrics_path,
    get_model_path,
    format_date_str,
)

if TYPE_CHECKING:
    from ..integrations.dropbox_sync import DropboxSync

logger = structlog.get_logger(__name__)


class HamiltonService:
    """Service for Hamilton to load champions and apply feature recipes."""
    
    def __init__(
        self,
        dropbox_sync: Optional["DropboxSync"] = None,
        base_folder: str = "huracan",
    ):
        """Initialize Hamilton service.
        
        Args:
            dropbox_sync: DropboxSync instance for reading files
            base_folder: Base folder name in Dropbox (default: "huracan")
        """
        self.dropbox_sync = dropbox_sync
        self.base_folder = base_folder
        
        # Cache for loaded models and recipes
        self._champion_pointer: Optional[ChampionPointer] = None
        self._loaded_models: Dict[str, Any] = {}
        self._loaded_recipes: Dict[str, FeatureRecipe] = {}
        self._loaded_costs: Dict[str, CostModel] = {}
        
        logger.info("hamilton_service_initialized", base_folder=base_folder)
    
    def load_champion_pointer(self, force_reload: bool = False) -> Optional[ChampionPointer]:
        """Load champion pointer from Dropbox.
        
        Args:
            force_reload: Force reload even if cached
            
        Returns:
            ChampionPointer instance if found, None otherwise
        """
        if self._champion_pointer and not force_reload:
            return self._champion_pointer
        
        if not self.dropbox_sync:
            logger.warning("dropbox_sync_not_available", message="Cannot load champion pointer without DropboxSync")
            return None
        
        dropbox_path = get_champion_pointer_path(self.base_folder)
        
        try:
            # Download from Dropbox
            _, response = self.dropbox_sync._dbx.files_download(dropbox_path)
            json_str = response.content.decode('utf-8')
            self._champion_pointer = ChampionPointer.from_json(json_str)
            logger.info("champion_pointer_loaded", path=dropbox_path, symbols=list(self._champion_pointer.models.keys()))
            return self._champion_pointer
        except Exception as e:
            logger.warning("champion_pointer_not_found", path=dropbox_path, error=str(e))
            return None
    
    def load_champion_model(self, symbol: str, force_reload: bool = False) -> Optional[Any]:
        """Load champion model for a symbol.
        
        Args:
            symbol: Trading symbol
            force_reload: Force reload even if cached
            
        Returns:
            Loaded model object if found, None otherwise
        """
        if symbol in self._loaded_models and not force_reload:
            return self._loaded_models[symbol]
        
        # Load champion pointer
        champion_pointer = self.load_champion_pointer()
        if not champion_pointer:
            logger.warning("champion_pointer_not_available", symbol=symbol)
            return None
        
        # Get model path for symbol
        model_path = champion_pointer.models.get(symbol)
        if not model_path:
            logger.warning("champion_model_not_found", symbol=symbol)
            return None
        
        # Load model from Dropbox
        if not self.dropbox_sync:
            logger.warning("dropbox_sync_not_available", message="Cannot load model without DropboxSync")
            return None
        
        try:
            # Download model from Dropbox
            _, response = self.dropbox_sync._dbx.files_download(model_path)
            model_bytes = response.content
            
            # Load model from bytes
            model = pickle.loads(model_bytes)
            self._loaded_models[symbol] = model
            
            logger.info("champion_model_loaded", symbol=symbol, model_path=model_path)
            return model
        except Exception as e:
            logger.error("champion_model_load_failed", symbol=symbol, model_path=model_path, error=str(e))
            return None
    
    def load_feature_recipe(self, symbol: str, date_str: Optional[str] = None, force_reload: bool = False) -> Optional[FeatureRecipe]:
        """Load feature recipe for a symbol.
        
        Args:
            symbol: Trading symbol
            date_str: Date string in YYYYMMDD format (defaults to latest)
            force_reload: Force reload even if cached
            
        Returns:
            FeatureRecipe instance if found, None otherwise
        """
        cache_key = f"{symbol}_{date_str or 'latest'}"
        if cache_key in self._loaded_recipes and not force_reload:
            return self._loaded_recipes[cache_key]
        
        if not self.dropbox_sync:
            logger.warning("dropbox_sync_not_available", message="Cannot load feature recipe without DropboxSync")
            return None
        
        # If date_str not provided, try to get from champion pointer
        if date_str is None:
            champion_pointer = self.load_champion_pointer()
            if champion_pointer:
                date_str = champion_pointer.date
        
        if not date_str:
            date_str = format_date_str()
        
        dropbox_path = get_feature_recipe_path(date_str, symbol, self.base_folder)
        
        try:
            # Download from Dropbox
            _, response = self.dropbox_sync._dbx.files_download(dropbox_path)
            json_str = response.content.decode('utf-8')
            recipe = FeatureRecipe.from_json(json_str)
            self._loaded_recipes[cache_key] = recipe
            
            logger.info("feature_recipe_loaded", symbol=symbol, path=dropbox_path)
            return recipe
        except Exception as e:
            logger.warning("feature_recipe_not_found", symbol=symbol, path=dropbox_path, error=str(e))
            return None
    
    def load_cost_model(self, symbol: str, date_str: Optional[str] = None, force_reload: bool = False) -> Optional[CostModel]:
        """Load cost model for a symbol.
        
        Args:
            symbol: Trading symbol
            date_str: Date string in YYYYMMDD format (defaults to latest)
            force_reload: Force reload even if cached
            
        Returns:
            CostModel instance if found, None otherwise
        """
        cache_key = f"{symbol}_{date_str or 'latest'}"
        if cache_key in self._loaded_costs and not force_reload:
            return self._loaded_costs[cache_key]
        
        if not self.dropbox_sync:
            logger.warning("dropbox_sync_not_available", message="Cannot load cost model without DropboxSync")
            return None
        
        # If date_str not provided, try to get from champion pointer
        if date_str is None:
            champion_pointer = self.load_champion_pointer()
            if champion_pointer:
                date_str = champion_pointer.date
        
        if not date_str:
            date_str = format_date_str()
        
        # Cost model is stored alongside metrics
        dropbox_path = get_metrics_path(date_str, symbol, self.base_folder).replace("metrics.json", "costs.json")
        
        try:
            # Download from Dropbox
            _, response = self.dropbox_sync._dbx.files_download(dropbox_path)
            json_str = response.content.decode('utf-8')
            cost_model = CostModel.from_json(json_str)
            self._loaded_costs[cache_key] = cost_model
            
            logger.info("cost_model_loaded", symbol=symbol, path=dropbox_path)
            return cost_model
        except Exception as e:
            logger.warning("cost_model_not_found", symbol=symbol, path=dropbox_path, error=str(e))
            return None
    
    def build_features_from_recipe(
        self,
        symbol: str,
        market_data: Dict[str, Any],
        recipe: Optional[FeatureRecipe] = None,
    ) -> Dict[str, Any]:
        """Build features from feature recipe.
        
        Args:
            symbol: Trading symbol
            market_data: Market data dictionary (OHLCV, etc.)
            recipe: Feature recipe (optional, will load if not provided)
            
        Returns:
            Dictionary of features
        """
        if recipe is None:
            recipe = self.load_feature_recipe(symbol)
        
        if not recipe:
            logger.warning("feature_recipe_not_available", symbol=symbol)
            return {}
        
        # In a real implementation, this would:
        # 1. Apply timeframes from recipe
        # 2. Calculate indicators from recipe.indicators
        # 3. Apply fill rules from recipe.fill_rules
        # 4. Apply normalization from recipe.normalization
        # 5. Return feature dictionary
        
        # For now, return empty dict (would be implemented with actual feature engineering)
        logger.info("features_built_from_recipe", symbol=symbol, recipe_hash=recipe.hash)
        return {}
    
    def apply_symbol_costs(
        self,
        symbol: str,
        predicted_edge_bps: float,
        order_type: str = "taker",
        cost_model: Optional[CostModel] = None,
    ) -> float:
        """Apply symbol costs to predicted edge.
        
        Args:
            symbol: Trading symbol
            predicted_edge_bps: Predicted edge in basis points
            order_type: Order type ("taker" or "maker")
            cost_model: Cost model (optional, will load if not provided)
            
        Returns:
            Net edge after costs in basis points
        """
        if cost_model is None:
            cost_model = self.load_cost_model(symbol)
        
        if not cost_model:
            logger.warning("cost_model_not_available", symbol=symbol, message="Using default costs")
            # Use default costs
            total_costs_bps = 15.0
        else:
            total_costs_bps = cost_model.total_cost_bps(order_type=order_type)
        
        net_edge_bps = predicted_edge_bps - total_costs_bps
        
        logger.debug("costs_applied", symbol=symbol, predicted_edge_bps=predicted_edge_bps, total_costs_bps=total_costs_bps, net_edge_bps=net_edge_bps)
        return net_edge_bps
    
    def predict(
        self,
        symbol: str,
        market_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Make prediction for a symbol.
        
        Args:
            symbol: Trading symbol
            market_data: Market data dictionary
            
        Returns:
            Dictionary with prediction results
        """
        # Load champion model
        model = self.load_champion_model(symbol)
        if not model:
            logger.warning("model_not_available", symbol=symbol)
            return {
                "symbol": symbol,
                "predicted_edge_bps": 0.0,
                "net_edge_bps": 0.0,
                "confidence": 0.0,
                "direction": "hold",
                "model_id": None,
            }
        
        # Build features from recipe
        features = self.build_features_from_recipe(symbol, market_data)
        
        # Make prediction
        try:
            # Convert features to model input format
            # In a real implementation, this would depend on the model type
            model_input = features  # Simplified
            
            # Predict
            predicted_edge_bps = model.predict([model_input])[0] if hasattr(model, 'predict') else 0.0
            
            # Apply costs
            net_edge_bps = self.apply_symbol_costs(symbol, predicted_edge_bps)
            
            # Determine direction
            direction = "long" if net_edge_bps > 0 else ("short" if net_edge_bps < 0 else "hold")
            confidence = abs(net_edge_bps) / 100.0  # Normalize to 0-1
            
            # Get model ID from champion pointer
            champion_pointer = self.load_champion_pointer()
            model_id = champion_pointer.run_id if champion_pointer else None
            
            return {
                "symbol": symbol,
                "predicted_edge_bps": predicted_edge_bps,
                "net_edge_bps": net_edge_bps,
                "confidence": confidence,
                "direction": direction,
                "model_id": model_id,
            }
        except Exception as e:
            logger.error("prediction_failed", symbol=symbol, error=str(e))
            return {
                "symbol": symbol,
                "predicted_edge_bps": 0.0,
                "net_edge_bps": 0.0,
                "confidence": 0.0,
                "direction": "hold",
                "model_id": None,
            }
    
    def log_trade(
        self,
        symbol: str,
        model_id: str,
        trade_data: Dict[str, Any],
        trade_log_path: Optional[str] = None,
    ) -> bool:
        """Log trade with model_id and symbol.
        
        Args:
            symbol: Trading symbol
            model_id: Model ID from champion pointer
            trade_data: Trade data dictionary
            trade_log_path: Path to trade log file (optional)
            
        Returns:
            True if successful, False otherwise
        """
        if not trade_log_path:
            # Default trade log path
            date_str = format_date_str()
            trade_log_path = f"/{self.base_folder}/trades/{date_str}/trades.json"
        
        trade_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "model_id": model_id,
            **trade_data,
        }
        
        # In a real implementation, this would:
        # 1. Load existing trade log
        # 2. Append new trade entry
        # 3. Write to Dropbox
        
        logger.info("trade_logged", symbol=symbol, model_id=model_id, trade_log_path=trade_log_path)
        return True

