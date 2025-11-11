"""
Per-Symbol Champion Manager

Keeps a lightweight champion pointer per symbol: champions/{SYMBOL}.json
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import structlog

logger = structlog.get_logger(__name__)


class PerSymbolChampion:
    """Manages champion pointer per symbol."""
    
    def __init__(
        self,
        champions_dir: str = "champions",
    ):
        """Initialize per-symbol champion manager.
        
        Args:
            champions_dir: Directory for champion files
        """
        self.champions_dir = Path(champions_dir)
        self.champions_dir.mkdir(parents=True, exist_ok=True)
        logger.info("per_symbol_champion_initialized", champions_dir=str(self.champions_dir))
    
    def get_champion_path(self, symbol: str) -> Path:
        """Get path to champion file for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Path to champion file
        """
        return self.champions_dir / f"{symbol}.json"
    
    def load_champion(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Load champion for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Champion dictionary, or None if not found
        """
        champion_path = self.get_champion_path(symbol)
        
        if not champion_path.exists():
            return None
        
        with open(champion_path, 'r') as f:
            champion = json.load(f)
        
        return champion
    
    def save_champion(
        self,
        symbol: str,
        model_path: str,
        metrics: Dict[str, Any],
        cost_model: Dict[str, Any],
        feature_recipe_hash: Optional[str] = None,
        date_str: Optional[str] = None,
    ) -> bool:
        """Save champion for a symbol.
        
        Args:
            symbol: Trading symbol
            model_path: Path to model file
            metrics: Metrics dictionary
            cost_model: Cost model dictionary
            feature_recipe_hash: Feature recipe hash (optional)
            date_str: Date string in YYYYMMDD format (defaults to today)
            
        Returns:
            True if successful
        """
        if date_str is None:
            date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
        
        champion = {
            "symbol": symbol,
            "date": date_str,
            "model_path": model_path,
            "metrics": metrics,
            "cost_model": cost_model,
            "feature_recipe_hash": feature_recipe_hash,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        
        champion_path = self.get_champion_path(symbol)
        
        try:
            with open(champion_path, 'w') as f:
                json.dump(champion, f, indent=2)
            
            logger.info("champion_saved", symbol=symbol, path=str(champion_path))
            return True
        except Exception as e:
            logger.error("champion_save_failed", symbol=symbol, error=str(e))
            return False
    
    def update_champion(
        self,
        symbol: str,
        model_path: str,
        metrics: Dict[str, Any],
        cost_model: Dict[str, Any],
        feature_recipe_hash: Optional[str] = None,
    ) -> bool:
        """Update champion if new model is better.
        
        Args:
            symbol: Trading symbol
            model_path: Path to new model file
            metrics: New metrics dictionary
            cost_model: New cost model dictionary
            feature_recipe_hash: Feature recipe hash (optional)
            
        Returns:
            True if champion was updated
        """
        current_champion = self.load_champion(symbol)
        
        # If no current champion, save new one
        if not current_champion:
            return self.save_champion(symbol, model_path, metrics, cost_model, feature_recipe_hash)
        
        # Compare metrics
        current_sharpe = current_champion.get("metrics", {}).get("sharpe", 0.0)
        new_sharpe = metrics.get("sharpe", 0.0)
        
        current_net_pnl = current_champion.get("metrics", {}).get("net_pnl_pct", 0.0)
        new_net_pnl = metrics.get("net_pnl_pct", 0.0)
        
        # Update if new model is better
        # Criteria: sharpe improvement > 0.1 OR net_pnl improvement > 0.5%
        if (new_sharpe > current_sharpe + 0.1) or (new_net_pnl > current_net_pnl + 0.005):
            logger.info("champion_updated", symbol=symbol, 
                       old_sharpe=current_sharpe, new_sharpe=new_sharpe,
                       old_net_pnl=current_net_pnl, new_net_pnl=new_net_pnl)
            return self.save_champion(symbol, model_path, metrics, cost_model, feature_recipe_hash)
        else:
            logger.info("champion_not_updated", symbol=symbol,
                       current_sharpe=current_sharpe, new_sharpe=new_sharpe)
            return False
    
    def get_all_champions(self) -> Dict[str, Dict[str, Any]]:
        """Get all champions.
        
        Returns:
            Dictionary mapping symbol to champion data
        """
        champions = {}
        
        for champion_file in self.champions_dir.glob("*.json"):
            if champion_file.name == "roster.json":
                continue
            
            symbol = champion_file.stem
            champion = self.load_champion(symbol)
            if champion:
                champions[symbol] = champion
        
        return champions

