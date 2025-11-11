"""
Simple Champion Manager

Manages champion.json file with clean, simple structure.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog  # type: ignore[import-untyped]

logger = structlog.get_logger(__name__)


class ChampionManager:
    """Manages champion models."""
    
    def __init__(self, champion_path: str = "champion.json"):
        """Initialize champion manager.
        
        Args:
            champion_path: Path to champion.json file
        """
        self.champion_path = Path(champion_path)
        logger.info("champion_manager_initialized", path=str(self.champion_path))
    
    def load_champion(self) -> Dict[str, Any]:
        """Load champion.json file.
        
        Returns:
            Champion dictionary
        """
        if not self.champion_path.exists():
            return {
                "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                "models": {},
                "summary": {
                    "total_symbols": 0,
                    "updated_today": [],
                },
            }
        
        with open(self.champion_path, 'r') as f:
            champion = json.load(f)
        
        return champion
    
    def save_champion(self, champion: Dict[str, Any]) -> bool:
        """Save champion.json file.
        
        Args:
            champion: Champion dictionary
            
        Returns:
            True if successful
        """
        try:
            self.champion_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.champion_path, 'w') as f:
                json.dump(champion, f, indent=2)
            
            logger.info("champion_saved", path=str(self.champion_path))
            return True
        except Exception as e:
            logger.error("champion_save_failed", path=str(self.champion_path), error=str(e))
            return False
    
    def update_model(self, symbol: str, model_path: str) -> bool:
        """Update model for a symbol.
        
        Args:
            symbol: Trading symbol
            model_path: Path to model file
            
        Returns:
            True if successful
        """
        champion = self.load_champion()
        
        # Update model
        champion["models"][symbol] = model_path
        
        # Update summary
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if champion["date"] != today:
            champion["updated_today"] = []
            champion["date"] = today
        
        if symbol not in champion["updated_today"]:
            champion["updated_today"].append(symbol)
        
        champion["summary"]["total_symbols"] = len(champion["models"])
        champion["summary"]["updated_today"] = champion["updated_today"]
        
        return self.save_champion(champion)
    
    def get_model_path(self, symbol: str) -> Optional[str]:
        """Get model path for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Model path, or None if not found
        """
        champion = self.load_champion()
        return champion["models"].get(symbol)
    
    def get_all_symbols(self) -> List[str]:
        """Get all symbols with champion models.
        
        Returns:
            List of symbols
        """
        champion = self.load_champion()
        return list(champion["models"].keys())

