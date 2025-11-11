"""
Champion Manager

Per coin champion pointer. Latest.json per symbol. Always valid.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import structlog

from ..contracts.model_bundle import ChampionPointer

logger = structlog.get_logger(__name__)


class ChampionManager:
    """Champion manager for per-coin champion pointers."""
    
    def __init__(
        self,
        base_path: str = "champion",
        s3_bucket: Optional[str] = None,
    ):
        """Initialize champion manager.
        
        Args:
            base_path: Base path for champion pointers
            s3_bucket: S3 bucket name (optional)
        """
        self.base_path = Path(base_path)
        self.s3_bucket = s3_bucket
        self.base_path.mkdir(parents=True, exist_ok=True)
        logger.info("champion_manager_initialized", base_path=str(self.base_path), s3_bucket=s3_bucket)
    
    def get_champion_path(self, symbol: str) -> Path:
        """Get path to champion pointer file.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Path to champion pointer file
        """
        symbol_dir = self.base_path / symbol
        symbol_dir.mkdir(parents=True, exist_ok=True)
        return symbol_dir / "latest.json"
    
    def load_champion(self, symbol: str) -> Optional[ChampionPointer]:
        """Load champion pointer for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Champion pointer, or None if not found
        """
        champion_path = self.get_champion_path(symbol)
        
        if not champion_path.exists():
            return None
        
        try:
            with open(champion_path, 'r') as f:
                data = json.load(f)
            
            return ChampionPointer.from_dict(data)
        except Exception as e:
            logger.error("champion_load_failed", symbol=symbol, error=str(e))
            return None
    
    def save_champion(
        self,
        symbol: str,
        model_id: str,
        s3_path: str,
        metadata: Optional[Dict] = None,
    ) -> bool:
        """Save champion pointer for a symbol.
        
        Args:
            symbol: Trading symbol
            model_id: Model ID
            s3_path: S3 path to model bundle
            metadata: Optional metadata
            
        Returns:
            True if successful
        """
        champion_path = self.get_champion_path(symbol)
        
        champion = ChampionPointer(
            symbol=symbol,
            model_id=model_id,
            s3_path=s3_path,
            updated_at=datetime.now(timezone.utc),
            metadata=metadata or {},
        )
        
        try:
            with open(champion_path, 'w') as f:
                json.dump(champion.to_dict(), f, indent=2)
            
            logger.info("champion_saved", symbol=symbol, model_id=model_id, s3_path=s3_path)
            return True
        except Exception as e:
            logger.error("champion_save_failed", symbol=symbol, error=str(e))
            return False
    
    def update_champion(
        self,
        symbol: str,
        model_id: str,
        s3_path: str,
        metadata: Optional[Dict] = None,
    ) -> bool:
        """Update champion pointer if new model is better.
        
        Args:
            symbol: Trading symbol
            model_id: New model ID
            s3_path: S3 path to new model bundle
            metadata: Optional metadata
            
        Returns:
            True if champion was updated
        """
        current_champion = self.load_champion(symbol)
        
        # If no current champion, save new one
        if not current_champion:
            return self.save_champion(symbol, model_id, s3_path, metadata)
        
        # TODO: Compare metrics to determine if new model is better
        # For now, always update
        return self.save_champion(symbol, model_id, s3_path, metadata)
    
    def get_all_champions(self) -> Dict[str, ChampionPointer]:
        """Get all champion pointers.
        
        Returns:
            Dictionary mapping symbol to champion pointer
        """
        champions = {}
        
        for symbol_dir in self.base_path.iterdir():
            if not symbol_dir.is_dir():
                continue
            
            symbol = symbol_dir.name
            champion = self.load_champion(symbol)
            if champion:
                champions[symbol] = champion
        
        return champions

