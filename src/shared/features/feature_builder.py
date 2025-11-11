"""
Shared Feature Builder

One shared feature builder. Same recipe for cloud and Hamilton.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class FeatureRecipe:
    """Feature recipe definition."""
    timeframes: List[str] = field(default_factory=lambda: ["1h", "4h", "1d"])
    indicators: Dict[str, Any] = field(default_factory=dict)
    fill_rules: Dict[str, Any] = field(default_factory=lambda: {"strategy": "forward_fill"})
    normalization: Dict[str, Any] = field(default_factory=lambda: {"type": "standard", "scaler": "StandardScaler"})
    window_sizes: Dict[str, int] = field(default_factory=lambda: {"short": 20, "medium": 50, "long": 200})
    hash: str = ""
    
    def compute_hash(self) -> str:
        """Compute hash of recipe.
        
        Returns:
            SHA256 hash as hex string
        """
        recipe_str = json.dumps({
            "timeframes": self.timeframes,
            "indicators": self.indicators,
            "fill_rules": self.fill_rules,
            "normalization": self.normalization,
            "window_sizes": self.window_sizes,
        }, sort_keys=True)
        
        return hashlib.sha256(recipe_str.encode()).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timeframes": self.timeframes,
            "indicators": self.indicators,
            "fill_rules": self.fill_rules,
            "normalization": self.normalization,
            "window_sizes": self.window_sizes,
            "hash": self.hash or self.compute_hash(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> FeatureRecipe:
        """Create from dictionary."""
        recipe = cls(
            timeframes=data.get("timeframes", ["1h", "4h", "1d"]),
            indicators=data.get("indicators", {}),
            fill_rules=data.get("fill_rules", {"strategy": "forward_fill"}),
            normalization=data.get("normalization", {"type": "standard", "scaler": "StandardScaler"}),
            window_sizes=data.get("window_sizes", {"short": 20, "medium": 50, "long": 200}),
        )
        recipe.hash = data.get("hash", recipe.compute_hash())
        return recipe


class FeatureBuilder:
    """Shared feature builder for cloud and Hamilton."""
    
    def __init__(self, recipe: Optional[FeatureRecipe] = None):
        """Initialize feature builder.
        
        Args:
            recipe: Feature recipe (defaults to standard recipe)
        """
        self.recipe = recipe or FeatureRecipe()
        if not self.recipe.hash:
            self.recipe.hash = self.recipe.compute_hash()
        logger.info("feature_builder_initialized", recipe_hash=self.recipe.hash)
    
    def build_features(self, candles_df: pd.DataFrame, symbol: str) -> Dict[str, float]:
        """Build features from candles.
        
        Args:
            candles_df: DataFrame with OHLCV data
            symbol: Trading symbol
            
        Returns:
            Dictionary of feature names to values
        """
        features = {}
        
        # Price-based features
        features.update(self._build_price_features(candles_df))
        
        # Volume features
        features.update(self._build_volume_features(candles_df))
        
        # Technical indicators
        features.update(self._build_technical_indicators(candles_df))
        
        # Time-based features
        features.update(self._build_time_features(candles_df))
        
        # Fill missing values
        features = self._fill_missing_values(features)
        
        logger.debug("features_built", symbol=symbol, feature_count=len(features))
        return features
    
    def _build_price_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Build price-based features.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Dictionary of price features
        """
        features = {}
        
        if df.empty:
            return features
        
        latest = df.iloc[-1]
        
        # Returns
        if len(df) > 1:
            features["ret_1"] = (latest["close"] / df.iloc[-2]["close"] - 1) * 100
        if len(df) > 5:
            features["ret_5"] = (latest["close"] / df.iloc[-6]["close"] - 1) * 100
        if len(df) > 20:
            features["ret_20"] = (latest["close"] / df.iloc[-21]["close"] - 1) * 100
        
        # Price ratios
        features["high_low_ratio"] = latest["high"] / latest["low"] if latest["low"] > 0 else 1.0
        features["close_open_ratio"] = latest["close"] / latest["open"] if latest["open"] > 0 else 1.0
        
        # Moving averages
        window_short = self.recipe.window_sizes.get("short", 20)
        window_medium = self.recipe.window_sizes.get("medium", 50)
        
        if len(df) >= window_short:
            features["sma_20"] = df["close"].tail(window_short).mean()
            features["price_vs_sma_20"] = (latest["close"] / features["sma_20"] - 1) * 100
        
        if len(df) >= window_medium:
            features["sma_50"] = df["close"].tail(window_medium).mean()
            features["price_vs_sma_50"] = (latest["close"] / features["sma_50"] - 1) * 100
        
        return features
    
    def _build_volume_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Build volume-based features.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Dictionary of volume features
        """
        features = {}
        
        if df.empty:
            return features
        
        latest = df.iloc[-1]
        window = self.recipe.window_sizes.get("short", 20)
        
        if len(df) >= window:
            avg_volume = df["volume"].tail(window).mean()
            features["volume_ratio"] = latest["volume"] / avg_volume if avg_volume > 0 else 1.0
        else:
            features["volume_ratio"] = 1.0
        
        return features
    
    def _build_technical_indicators(self, df: pd.DataFrame) -> Dict[str, float]:
        """Build technical indicator features.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Dictionary of technical indicator features
        """
        features = {}
        
        if df.empty or len(df) < 14:
            return features
        
        # RSI
        features["rsi_14"] = self._calculate_rsi(df["close"], period=14)
        
        # Volatility
        if len(df) >= 20:
            returns = df["close"].pct_change()
            features["volatility_20"] = returns.tail(20).std() * 100
        
        return features
    
    def _calculate_rsi(self, series: pd.Series, period: int = 14) -> float:
        """Calculate RSI.
        
        Args:
            series: Price series
            period: RSI period
            
        Returns:
            RSI value
        """
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return float(rsi.iloc[-1]) if not rsi.empty else 50.0
    
    def _build_time_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Build time-based features.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Dictionary of time features
        """
        features = {}
        
        if df.empty:
            return features
        
        # Assume timestamp column exists
        if "timestamp" in df.columns:
            latest_timestamp = pd.to_datetime(df["timestamp"].iloc[-1])
            features["hour_of_day"] = latest_timestamp.hour
            features["day_of_week"] = latest_timestamp.dayofweek
            features["is_weekend"] = 1.0 if latest_timestamp.dayofweek >= 5 else 0.0
        
        return features
    
    def _fill_missing_values(self, features: Dict[str, float]) -> Dict[str, float]:
        """Fill missing values according to recipe.
        
        Args:
            features: Feature dictionary
            
        Returns:
            Feature dictionary with missing values filled
        """
        fill_strategy = self.recipe.fill_rules.get("strategy", "forward_fill")
        
        if fill_strategy == "forward_fill":
            # Forward fill with 0.0
            return {k: (v if v is not None and not pd.isna(v) else 0.0) for k, v in features.items()}
        elif fill_strategy == "zero":
            return {k: (v if v is not None and not pd.isna(v) else 0.0) for k, v in features.items()}
        else:
            return features
    
    def load_recipe(self, recipe_path: str) -> None:
        """Load recipe from file.
        
        Args:
            recipe_path: Path to recipe file
        """
        with open(recipe_path, 'r') as f:
            recipe_data = json.load(f)
        
        self.recipe = FeatureRecipe.from_dict(recipe_data)
        logger.info("recipe_loaded", recipe_path=recipe_path, recipe_hash=self.recipe.hash)
    
    def save_recipe(self, recipe_path: str) -> None:
        """Save recipe to file.
        
        Args:
            recipe_path: Path to recipe file
        """
        with open(recipe_path, 'w') as f:
            json.dump(self.recipe.to_dict(), f, indent=2)
        
        logger.info("recipe_saved", recipe_path=recipe_path, recipe_hash=self.recipe.hash)

