"""
Symbols Selector

Telegram control for symbol selection. Engine respects a symbols selector file that your bot writes.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import structlog

logger = structlog.get_logger(__name__)


class SymbolsSelector:
    """Symbols selector for Telegram control."""
    
    def __init__(
        self,
        selector_file: str = "symbols_selector.json",
    ):
        """Initialize symbols selector.
        
        Args:
            selector_file: Path to symbols selector file
        """
        self.selector_file = Path(selector_file)
        logger.info("symbols_selector_initialized", selector_file=str(self.selector_file))
    
    def load_symbols(self) -> List[str]:
        """Load symbols from selector file.
        
        Returns:
            List of symbols
        """
        if not self.selector_file.exists():
            logger.warning("selector_file_not_found", selector_file=str(self.selector_file))
            return []
        
        try:
            with open(self.selector_file, 'r') as f:
                data = json.load(f)
            
            symbols = data.get("symbols", [])
            logger.info("symbols_loaded", symbol_count=len(symbols), selector_file=str(self.selector_file))
            return symbols
        except Exception as e:
            logger.error("symbols_load_failed", selector_file=str(self.selector_file), error=str(e))
            return []
    
    def save_symbols(self, symbols: List[str], source: str = "telegram") -> bool:
        """Save symbols to selector file.
        
        Args:
            symbols: List of symbols
            source: Source of symbol selection (e.g., "telegram")
            
        Returns:
            True if successful
        """
        try:
            data = {
                "symbols": symbols,
                "source": source,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            
            with open(self.selector_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info("symbols_saved", symbol_count=len(symbols), source=source, selector_file=str(self.selector_file))
            return True
        except Exception as e:
            logger.error("symbols_save_failed", selector_file=str(self.selector_file), error=str(e))
            return False
    
    def get_top_symbols(self, n: int, meta_weights: Dict[str, float]) -> List[str]:
        """Get top N symbols by meta weight.
        
        Args:
            n: Number of symbols to return
            meta_weights: Dictionary mapping symbol to meta weight
            
        Returns:
            List of top N symbols
        """
        sorted_symbols = sorted(
            meta_weights.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        
        return [symbol for symbol, _ in sorted_symbols[:n]]

