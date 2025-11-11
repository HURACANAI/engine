"""
Enhanced Feature Builder

Enhanced feature builder with alternative data sources (order book, on-chain, funding rates).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class AlternativeData:
    """Alternative data sources."""
    # Order book data
    order_book_depth: Optional[Dict[str, float]] = None  # {level: depth}
    order_book_imbalance: Optional[float] = None  # Bid/ask imbalance
    large_order_flow: Optional[float] = None  # Large order flow
    iceberg_orders: Optional[int] = None  # Number of iceberg orders
    
    # On-chain data
    wallet_transfers: Optional[float] = None  # Wallet transfer volume
    exchange_inflows: Optional[float] = None  # Exchange inflows
    exchange_outflows: Optional[float] = None  # Exchange outflows
    whale_movements: Optional[float] = None  # Large wallet movements
    
    # Funding rates
    funding_rate: Optional[float] = None  # Current funding rate
    funding_rate_skew: Optional[float] = None  # Funding rate skew
    long_short_ratio: Optional[float] = None  # Long/short ratio
    
    # News sentiment
    news_sentiment: Optional[float] = None  # News sentiment score
    governance_events: Optional[List[str]] = None  # Governance events


@dataclass
class EnhancedFeatureRecipe:
    """Enhanced feature recipe with alternative data."""
    symbol: str
    timeframes: List[str]
    indicator_set: Dict[str, Any]
    fill_rules: Dict[str, Any]
    normalization: Dict[str, Any]
    window_sizes: Dict[str, Any]
    # Alternative data
    use_order_book: bool = False
    use_on_chain: bool = False
    use_funding_rates: bool = False
    use_news_sentiment: bool = False
    hash: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now())


class EnhancedFeatureBuilder:
    """Enhanced feature builder with alternative data sources."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize enhanced feature builder.
        
        Args:
            config: Feature builder configuration
        """
        self.config = config
        self.recipe: Optional[EnhancedFeatureRecipe] = None
        logger.info("enhanced_feature_builder_initialized")
    
    def build_features(
        self,
        df: pd.DataFrame,
        symbol: str,
        alternative_data: Optional[AlternativeData] = None,
    ) -> pd.DataFrame:
        """Build features with alternative data.
        
        Args:
            df: DataFrame with OHLCV data
            symbol: Trading symbol
            alternative_data: Alternative data sources (optional)
            
        Returns:
            DataFrame with calculated features
        """
        logger.info("building_enhanced_features", symbol=symbol, num_rows=len(df))
        
        # Convert to Pandas if needed
        if not isinstance(df, pd.DataFrame):
            df = df.to_pandas()
        
        # Apply fill rules
        df = self._apply_fill_rules(df)
        
        # Calculate traditional indicators
        features_df = self._calculate_indicators(df)
        
        # Add alternative data features
        if alternative_data:
            features_df = self._add_alternative_data_features(features_df, alternative_data)
        
        # Apply normalization
        features_df = self._apply_normalization(features_df)
        
        logger.info("enhanced_features_built", symbol=symbol, num_features=features_df.shape[1])
        return features_df
    
    def _apply_fill_rules(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply data filling rules."""
        if self.config.get("fill_rules", {}).get("forward_fill_max_gaps"):
            max_gaps = self.config["fill_rules"]["forward_fill_max_gaps"]
            df = df.ffill(limit=max_gaps)
            logger.debug("applied_forward_fill", max_gaps=max_gaps)
        return df
    
    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators."""
        features = pd.DataFrame(index=df.index)
        close = df["close"]
        
        indicator_set = self.config.get("indicator_set", {})
        
        # RSI
        if indicator_set.get("rsi"):
            window = indicator_set["rsi"].get("window", 14)
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            features["rsi"] = 100 - (100 / (1 + rs))
            features["rsi"] = features["rsi"].fillna(50)
            logger.debug("calculated_rsi", window=window)
        
        # EMA
        if indicator_set.get("ema"):
            window = indicator_set["ema"].get("window", 20)
            features["ema"] = close.ewm(span=window, adjust=False).mean()
            # EMA slope
            features["ema_slope"] = features["ema"].diff()
            logger.debug("calculated_ema", window=window)
        
        # Volatility
        if indicator_set.get("volatility"):
            window = indicator_set["volatility"].get("window", 20)
            returns = close.pct_change()
            features["volatility"] = returns.rolling(window=window).std()
            # Volatility contraction
            features["volatility_contraction"] = features["volatility"].diff()
            logger.debug("calculated_volatility", window=window)
        
        # Momentum
        if indicator_set.get("momentum"):
            window = indicator_set["momentum"].get("window", 10)
            features["momentum"] = close.pct_change(window)
            logger.debug("calculated_momentum", window=window)
        
        # Trend strength
        if indicator_set.get("trend_strength"):
            window = indicator_set["trend_strength"].get("window", 20)
            sma = close.rolling(window=window).mean()
            trend_strength = abs((close - sma) / sma)
            features["trend_strength"] = trend_strength
            logger.debug("calculated_trend_strength", window=window)
        
        # Support/Resistance levels
        if indicator_set.get("support_resistance"):
            window = indicator_set["support_resistance"].get("window", 20)
            # Simple support/resistance (high/low of window)
            features["support_level"] = df["low"].rolling(window=window).min()
            features["resistance_level"] = df["high"].rolling(window=window).max()
            # Distance to support/resistance
            features["distance_to_support"] = (close - features["support_level"]) / close
            features["distance_to_resistance"] = (features["resistance_level"] - close) / close
            logger.debug("calculated_support_resistance", window=window)
        
        # Volume features
        if "volume" in df.columns:
            window = indicator_set.get("volume", {}).get("window", 20)
            avg_volume = df["volume"].rolling(window=window).mean()
            features["volume_ratio"] = df["volume"] / avg_volume
            features["volume_ratio"] = features["volume_ratio"].fillna(1.0)
            logger.debug("calculated_volume_ratio", window=window)
        
        # Price change
        features["price_change"] = close.pct_change()
        features["current_price"] = close
        
        # Drop rows with NaN values
        features = features.dropna()
        
        return features
    
    def _add_alternative_data_features(
        self,
        features_df: pd.DataFrame,
        alternative_data: AlternativeData,
    ) -> pd.DataFrame:
        """Add alternative data features.
        
        Args:
            features_df: DataFrame with existing features
            alternative_data: Alternative data sources
            
        Returns:
            DataFrame with alternative data features added
        """
        # Order book features
        if alternative_data.order_book_imbalance is not None:
            features_df["order_book_imbalance"] = alternative_data.order_book_imbalance
        
        if alternative_data.large_order_flow is not None:
            features_df["large_order_flow"] = alternative_data.large_order_flow
        
        if alternative_data.iceberg_orders is not None:
            features_df["iceberg_orders"] = alternative_data.iceberg_orders
        
        # On-chain features
        if alternative_data.wallet_transfers is not None:
            features_df["wallet_transfers"] = alternative_data.wallet_transfers
        
        if alternative_data.exchange_inflows is not None:
            features_df["exchange_inflows"] = alternative_data.exchange_inflows
        
        if alternative_data.exchange_outflows is not None:
            features_df["exchange_outflows"] = alternative_data.exchange_outflows
        
        if alternative_data.whale_movements is not None:
            features_df["whale_movements"] = alternative_data.whale_movements
        
        # Funding rate features
        if alternative_data.funding_rate is not None:
            features_df["funding_rate"] = alternative_data.funding_rate
        
        if alternative_data.funding_rate_skew is not None:
            features_df["funding_rate_skew"] = alternative_data.funding_rate_skew
        
        if alternative_data.long_short_ratio is not None:
            features_df["long_short_ratio"] = alternative_data.long_short_ratio
        
        # News sentiment
        if alternative_data.news_sentiment is not None:
            features_df["news_sentiment"] = alternative_data.news_sentiment
        
        logger.debug("added_alternative_data_features", num_features=len(features_df.columns))
        
        return features_df
    
    def _apply_normalization(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply normalization rules."""
        normalization_config = self.config.get("normalization", {})
        
        if normalization_config.get("type") == "min_max":
            for col in df.columns:
                if df[col].dtype in [float, int]:
                    min_val = df[col].min()
                    max_val = df[col].max()
                    if max_val > min_val:
                        df[col] = (df[col] - min_val) / (max_val - min_val)
                    else:
                        df[col] = 0.5
            logger.debug("applied_min_max_normalization")
        elif normalization_config.get("type") == "z_score":
            for col in df.columns:
                if df[col].dtype in [float, int]:
                    mean = df[col].mean()
                    std = df[col].std()
                    if std > 0:
                        df[col] = (df[col] - mean) / std
                    else:
                        df[col] = 0.0
            logger.debug("applied_z_score_normalization")
        
        return df
    
    def set_recipe(self, recipe: EnhancedFeatureRecipe):
        """Set the feature recipe."""
        self.recipe = recipe
        # Update internal config based on recipe
        self.config["timeframes"] = recipe.timeframes
        self.config["indicator_set"] = recipe.indicator_set
        self.config["fill_rules"] = recipe.fill_rules
        self.config["normalization"] = recipe.normalization
        self.config["window_sizes"] = recipe.window_sizes
        logger.info("enhanced_feature_recipe_set", symbol=recipe.symbol, hash=recipe.hash)
    
    def get_recipe_hash(self) -> str:
        """Compute SHA256 hash of the current feature recipe."""
        if self.recipe:
            import hashlib
            import json
            recipe_dict = self.recipe.__dict__.copy()
            recipe_dict.pop("hash", None)
            recipe_dict.pop("created_at", None)
            sorted_json = json.dumps(recipe_dict, sort_keys=True)
            return hashlib.sha256(sorted_json.encode()).hexdigest()
        return ""


