"""
Hamilton Interface Contract

Provides interface for Hamilton trading system:
- Model loading
- Prediction
- Ranking table
- Do-not-trade list

Author: Huracan Engine Team
Date: 2025-01-27
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any
from enum import Enum

import structlog

logger = structlog.get_logger(__name__)


class ModelLoadError(Exception):
    """Model load error."""
    pass


@dataclass
class ModelMetadata:
    """Model metadata for Hamilton."""
    coin: str
    horizon: str
    version: int
    regime: Optional[str] = None
    net_edge_bps: float = 0.0
    confidence: float = 0.0
    capacity_estimate: float = 0.0
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    created_at: Optional[str] = None


@dataclass
class RankingEntry:
    """Ranking table entry."""
    coin: str
    regime: str
    net_edge_bps: float
    confidence: float
    capacity_estimate: float
    rank: int = 0


@dataclass
class PredictionResult:
    """Prediction result."""
    coin: str
    direction: str  # "long", "short", "hold"
    confidence: float
    edge_bps: float
    metadata: ModelMetadata


class HamiltonModelLoader:
    """
    Model loader for Hamilton interface.
    
    Features:
    - Single call model loading
    - Prediction interface
    - Metadata access
    """
    
    def __init__(self, model_base_path: str = "/models"):
        """
        Initialize model loader.
        
        Args:
            model_base_path: Base path for models
        """
        self.model_base_path = Path(model_base_path)
        self.loaded_models: Dict[str, Any] = {}
        logger.info("hamilton_model_loader_initialized", model_base_path=model_base_path)
    
    def load_model(self, coin: str, horizon: str, version: Optional[int] = None) -> tuple[Any, ModelMetadata]:
        """
        Load model with a single call.
        
        Args:
            coin: Coin symbol
            horizon: Horizon
            version: Model version (None for latest)
        
        Returns:
            Tuple of (model, metadata)
        
        Raises:
            ModelLoadError: If model cannot be loaded
        """
        try:
            # Construct model path
            if version:
                model_path = self.model_base_path / coin / horizon / f"v{version}" / "model.pkl"
                metadata_path = self.model_base_path / coin / horizon / f"v{version}" / "manifest.json"
            else:
                # Find latest version
                horizon_dir = self.model_base_path / coin / horizon
                if not horizon_dir.exists():
                    raise ModelLoadError(f"Model directory not found: {horizon_dir}")
                
                # Find latest version
                versions = [d.name for d in horizon_dir.iterdir() if d.is_dir() and d.name.startswith("v")]
                if not versions:
                    raise ModelLoadError(f"No versions found for {coin}/{horizon}")
                
                latest_version = max(versions, key=lambda v: int(v[1:]))
                model_path = horizon_dir / latest_version / "model.pkl"
                metadata_path = horizon_dir / latest_version / "manifest.json"
            
            # Load model
            if not model_path.exists():
                raise ModelLoadError(f"Model file not found: {model_path}")
            
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            
            # Load metadata
            metadata = self._load_metadata(metadata_path, coin, horizon, version)
            
            # Cache model
            cache_key = f"{coin}_{horizon}_{version or 'latest'}"
            self.loaded_models[cache_key] = (model, metadata)
            
            logger.info("model_loaded", coin=coin, horizon=horizon, version=version)
            
            return model, metadata
        
        except Exception as e:
            logger.error("model_load_failed", coin=coin, horizon=horizon, error=str(e))
            raise ModelLoadError(f"Failed to load model: {str(e)}")
    
    def predict(self, coin: str, horizon: str, features: Dict[str, Any], version: Optional[int] = None) -> PredictionResult:
        """
        Make prediction with model.
        
        Args:
            coin: Coin symbol
            horizon: Horizon
            features: Feature dictionary
            version: Model version (None for latest)
        
        Returns:
            PredictionResult
        """
        # Load model if not cached
        cache_key = f"{coin}_{horizon}_{version or 'latest'}"
        if cache_key not in self.loaded_models:
            model, metadata = self.load_model(coin, horizon, version)
        else:
            model, metadata = self.loaded_models[cache_key]
        
        # Make prediction
        try:
            # Convert features to model input format
            model_input = self._prepare_features(features)
            
            # Predict
            prediction = model.predict(model_input)
            direction = "long" if prediction > 0 else ("short" if prediction < 0 else "hold")
            confidence = abs(prediction)
            edge_bps = prediction * 100.0  # Convert to basis points
            
            return PredictionResult(
                coin=coin,
                direction=direction,
                confidence=confidence,
                edge_bps=edge_bps,
                metadata=metadata,
            )
        
        except Exception as e:
            logger.error("prediction_failed", coin=coin, horizon=horizon, error=str(e))
            raise ModelLoadError(f"Failed to make prediction: {str(e)}")
    
    def _load_metadata(self, metadata_path: Path, coin: str, horizon: str, version: Optional[int]) -> ModelMetadata:
        """Load metadata from manifest."""
        import json
        
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                data = json.load(f)
            
            return ModelMetadata(
                coin=data.get("coin", coin),
                horizon=data.get("horizon", horizon),
                version=data.get("version", version or 1),
                regime=data.get("regime"),
                net_edge_bps=data.get("metrics", {}).get("net_edge_bps", 0.0),
                confidence=data.get("metrics", {}).get("confidence", 0.0),
                capacity_estimate=data.get("metrics", {}).get("capacity_estimate", 0.0),
                sharpe_ratio=data.get("metrics", {}).get("sharpe_ratio", 0.0),
                win_rate=data.get("metrics", {}).get("win_rate", 0.0),
                created_at=data.get("timestamp"),
            )
        else:
            # Return default metadata
            return ModelMetadata(
                coin=coin,
                horizon=horizon,
                version=version or 1,
            )
    
    def _prepare_features(self, features: Dict[str, Any]) -> Any:
        """Prepare features for model input."""
        # Placeholder implementation
        # In production, convert features to model-specific format
        return features


class HamiltonRankingTable:
    """
    Ranking table for Hamilton.
    
    Provides ranking of coins by net edge, confidence, and capacity.
    """
    
    def __init__(self, model_loader: HamiltonModelLoader):
        """
        Initialize ranking table.
        
        Args:
            model_loader: Model loader instance
        """
        self.model_loader = model_loader
        self.rankings: List[RankingEntry] = []
        logger.info("hamilton_ranking_table_initialized")
    
    def build_ranking_table(
        self,
        coins: List[str],
        horizons: List[str],
        regimes: List[str],
    ) -> List[RankingEntry]:
        """
        Build ranking table.
        
        Args:
            coins: List of coins
            horizons: List of horizons
            regimes: List of regimes
        
        Returns:
            List of ranking entries
        """
        entries = []
        
        for coin in coins:
            for horizon in horizons:
                for regime in regimes:
                    try:
                        # Load model metadata
                        _, metadata = self.model_loader.load_model(coin, horizon)
                        
                        # Create ranking entry
                        entry = RankingEntry(
                            coin=coin,
                            regime=regime,
                            net_edge_bps=metadata.net_edge_bps,
                            confidence=metadata.confidence,
                            capacity_estimate=metadata.capacity_estimate,
                        )
                        entries.append(entry)
                    
                    except ModelLoadError as e:
                        logger.warning("model_load_failed_for_ranking", coin=coin, horizon=horizon, error=str(e))
                        continue
        
        # Sort by net edge (descending)
        entries.sort(key=lambda e: e.net_edge_bps, reverse=True)
        
        # Assign ranks
        for i, entry in enumerate(entries):
            entry.rank = i + 1
        
        self.rankings = entries
        logger.info("ranking_table_built", entries=len(entries))
        
        return entries
    
    def get_ranking_table(self) -> List[RankingEntry]:
        """Get current ranking table."""
        return self.rankings.copy()


class HamiltonDoNotTradeList:
    """
    Do-not-trade list for Hamilton.
    
    Provides list of coins that fail liquidity or cost checks.
    """
    
    def __init__(self):
        """Initialize do-not-trade list."""
        self.blocked_coins: Dict[str, str] = {}  # coin -> reason
        logger.info("hamilton_do_not_trade_list_initialized")
    
    def add_coin(self, coin: str, reason: str) -> None:
        """Add coin to do-not-trade list."""
        self.blocked_coins[coin] = reason
        logger.info("coin_added_to_dnt_list", coin=coin, reason=reason)
    
    def remove_coin(self, coin: str) -> None:
        """Remove coin from do-not-trade list."""
        if coin in self.blocked_coins:
            del self.blocked_coins[coin]
            logger.info("coin_removed_from_dnt_list", coin=coin)
    
    def is_blocked(self, coin: str) -> bool:
        """Check if coin is blocked."""
        return coin in self.blocked_coins
    
    def get_blocked_coins(self) -> Dict[str, str]:
        """Get all blocked coins with reasons."""
        return self.blocked_coins.copy()
    
    def get_do_not_trade_list(self) -> List[str]:
        """Get list of blocked coins."""
        return list(self.blocked_coins.keys())


class HamiltonInterface:
    """
    Hamilton interface contract.
    
    Provides unified interface for Hamilton trading system.
    """
    
    def __init__(
        self,
        model_base_path: str = "/models",
        ranking_coins: Optional[List[str]] = None,
        ranking_horizons: Optional[List[str]] = None,
        ranking_regimes: Optional[List[str]] = None,
    ):
        """
        Initialize Hamilton interface.
        
        Args:
            model_base_path: Base path for models
            ranking_coins: Coins for ranking table
            ranking_horizons: Horizons for ranking table
            ranking_regimes: Regimes for ranking table
        """
        self.model_loader = HamiltonModelLoader(model_base_path)
        self.ranking_table = HamiltonRankingTable(self.model_loader)
        self.do_not_trade_list = HamiltonDoNotTradeList()
        
        # Build ranking table if coins provided
        if ranking_coins and ranking_horizons and ranking_regimes:
            self.ranking_table.build_ranking_table(ranking_coins, ranking_horizons, ranking_regimes)
        
        logger.info("hamilton_interface_initialized")
    
    def load_model(self, coin: str, horizon: str, version: Optional[int] = None) -> tuple[Any, ModelMetadata]:
        """Load model with a single call."""
        return self.model_loader.load_model(coin, horizon, version)
    
    def predict(self, coin: str, horizon: str, features: Dict[str, Any], version: Optional[int] = None) -> PredictionResult:
        """Make prediction with model."""
        return self.model_loader.predict(coin, horizon, features, version)
    
    def get_ranking_table(self) -> List[RankingEntry]:
        """Get ranking table."""
        return self.ranking_table.get_ranking_table()
    
    def get_do_not_trade_list(self) -> List[str]:
        """Get do-not-trade list."""
        return self.do_not_trade_list.get_do_not_trade_list()
    
    def is_coin_blocked(self, coin: str) -> bool:
        """Check if coin is blocked."""
        return self.do_not_trade_list.is_blocked(coin)
    
    def add_blocked_coin(self, coin: str, reason: str) -> None:
        """Add coin to do-not-trade list."""
        self.do_not_trade_list.add_coin(coin, reason)
    
    def remove_blocked_coin(self, coin: str) -> None:
        """Remove coin from do-not-trade list."""
        self.do_not_trade_list.remove_coin(coin)

