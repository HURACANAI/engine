"""
Feature Bank for Meta Score Table

Tracks feature importance per coin and saves to meta/feature_bank.json for reuse.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog

logger = structlog.get_logger(__name__)


class FeatureBank:
    """Feature bank for tracking feature importance per coin."""
    
    def __init__(
        self,
        output_dir: str = "meta",
    ):
        """Initialize feature bank.
        
        Args:
            output_dir: Output directory for feature bank file
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.feature_bank_path = self.output_dir / "feature_bank.json"
        logger.info("feature_bank_initialized", output_dir=str(self.output_dir))
    
    def load_feature_bank(self) -> Dict[str, Any]:
        """Load feature bank from file.
        
        Returns:
            Feature bank dictionary
        """
        if not self.feature_bank_path.exists():
            return {
                "features": {},
                "symbols": {},
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
        
        with open(self.feature_bank_path, 'r') as f:
            feature_bank = json.load(f)
        
        return feature_bank
    
    def save_feature_bank(self, feature_bank: Dict[str, Any]) -> bool:
        """Save feature bank to file.
        
        Args:
            feature_bank: Feature bank dictionary
            
        Returns:
            True if successful
        """
        try:
            feature_bank["updated_at"] = datetime.now(timezone.utc).isoformat()
            
            with open(self.feature_bank_path, 'w') as f:
                json.dump(feature_bank, f, indent=2)
            
            logger.info("feature_bank_saved", path=str(self.feature_bank_path))
            return True
        except Exception as e:
            logger.error("feature_bank_save_failed", error=str(e))
            return False
    
    def update_feature_importance(
        self,
        symbol: str,
        feature_importance: Dict[str, float],
        model_type: str = "xgboost",
    ) -> bool:
        """Update feature importance for a symbol.
        
        Args:
            symbol: Trading symbol
            feature_importance: Dictionary mapping feature names to importance scores
            model_type: Type of model (e.g., "xgboost", "lightgbm")
            
        Returns:
            True if successful
        """
        feature_bank = self.load_feature_bank()
        
        # Update symbol features
        if "symbols" not in feature_bank:
            feature_bank["symbols"] = {}
        
        feature_bank["symbols"][symbol] = {
            "feature_importance": feature_importance,
            "model_type": model_type,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        
        # Update global feature registry
        if "features" not in feature_bank:
            feature_bank["features"] = {}
        
        for feature_name, importance in feature_importance.items():
            if feature_name not in feature_bank["features"]:
                feature_bank["features"][feature_name] = {
                    "symbols_using": [],
                    "avg_importance": 0.0,
                    "max_importance": 0.0,
                }
            
            feature_info = feature_bank["features"][feature_name]
            if symbol not in feature_info["symbols_using"]:
                feature_info["symbols_using"].append(symbol)
            
            # Update statistics
            symbol_importances = [
                feature_bank["symbols"][s]["feature_importance"].get(feature_name, 0.0)
                for s in feature_bank["symbols"].keys()
                if feature_name in feature_bank["symbols"][s]["feature_importance"]
            ]
            
            if symbol_importances:
                feature_info["avg_importance"] = sum(symbol_importances) / len(symbol_importances)
                feature_info["max_importance"] = max(symbol_importances)
        
        return self.save_feature_bank(feature_bank)
    
    def get_shared_features(self, min_symbols: int = 5) -> List[str]:
        """Get features used by multiple symbols.
        
        Args:
            min_symbols: Minimum number of symbols using the feature
            
        Returns:
            List of shared feature names
        """
        feature_bank = self.load_feature_bank()
        
        shared_features = [
            feature_name
            for feature_name, feature_info in feature_bank.get("features", {}).items()
            if len(feature_info.get("symbols_using", [])) >= min_symbols
        ]
        
        return shared_features
    
    def get_feature_usage(self, feature_name: str) -> Dict[str, Any]:
        """Get feature usage statistics.
        
        Args:
            feature_name: Feature name
            
        Returns:
            Feature usage dictionary
        """
        feature_bank = self.load_feature_bank()
        
        return feature_bank.get("features", {}).get(feature_name, {
            "symbols_using": [],
            "avg_importance": 0.0,
            "max_importance": 0.0,
        })
    
    def get_symbol_features(self, symbol: str) -> Dict[str, float]:
        """Get feature importance for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        feature_bank = self.load_feature_bank()
        
        return feature_bank.get("symbols", {}).get(symbol, {}).get("feature_importance", {})

