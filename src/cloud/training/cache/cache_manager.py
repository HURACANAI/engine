"""
Cache Manager - High-level caching abstraction.

Provides caching for:
- Market data (prices, candles)
- Calculated features
- Model predictions
- Performance metrics
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Optional, Dict, List
import json

import pandas as pd
import numpy as np
import structlog

from .redis_client import RedisClient, get_redis_client

logger = structlog.get_logger(__name__)


class CacheManager:
    """
    High-level cache manager for trading data.
    
    Provides caching with automatic serialization/deserialization
    and intelligent TTL management.
    """
    
    def __init__(
        self,
        redis_client: Optional[RedisClient] = None,
        default_ttl: int = 300,  # 5 minutes default
    ) -> None:
        """
        Initialize cache manager.
        
        Args:
            redis_client: Redis client instance (creates new if None)
            default_ttl: Default time-to-live in seconds
        """
        self.redis_client = redis_client
        self.default_ttl = default_ttl
        self._redis_initialized = False
        
        logger.info("cache_manager_initialized", default_ttl=default_ttl)
    
    async def _ensure_redis(self) -> RedisClient:
        """Ensure Redis client is initialized."""
        if self.redis_client is None:
            self.redis_client = await get_redis_client()
            self._redis_initialized = True
        elif not self._redis_initialized:
            await self.redis_client.connect()
            self._redis_initialized = True
        return self.redis_client
    
    # ========================================================================
    # Price Data Caching
    # ========================================================================
    
    async def get_prices(
        self,
        symbol: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> Optional[pd.DataFrame]:
        """
        Get cached price data.
        
        Args:
            symbol: Trading symbol
            start_time: Start time (optional)
            end_time: End time (optional)
        
        Returns:
            Cached DataFrame or None
        """
        redis = await self._ensure_redis()
        
        # Build cache key
        key = f"prices:{symbol}"
        if start_time:
            key += f":{start_time.isoformat()}"
        if end_time:
            key += f":{end_time.isoformat()}"
        
        # Try to get from cache
        cached = await redis.get_json(key)
        if cached:
            try:
                df = pd.read_json(cached, orient='records')
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                logger.debug("cache_hit_prices", symbol=symbol, rows=len(df))
                return df
            except Exception as e:
                logger.warning("cache_deserialize_error", key=key, error=str(e))
                return None
        
        return None
    
    async def cache_prices(
        self,
        symbol: str,
        data: pd.DataFrame,
        ttl: Optional[int] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> bool:
        """
        Cache price data.
        
        Args:
            symbol: Trading symbol
            data: DataFrame with price data
            ttl: Time-to-live in seconds (uses default if None)
            start_time: Start time (optional, for key)
            end_time: End time (optional, for key)
        
        Returns:
            True if cached successfully
        """
        redis = await self._ensure_redis()
        
        # Build cache key
        key = f"prices:{symbol}"
        if start_time:
            key += f":{start_time.isoformat()}"
        if end_time:
            key += f":{end_time.isoformat()}"
        
        # Serialize DataFrame
        try:
            json_data = data.to_json(orient='records', date_format='iso')
            success = await redis.set_json(key, json.loads(json_data), ttl=ttl or self.default_ttl)
            
            if success:
                logger.debug("cache_set_prices", symbol=symbol, rows=len(data), ttl=ttl)
            return success
        except Exception as e:
            logger.error("cache_serialize_error", key=key, error=str(e))
            return False
    
    # ========================================================================
    # Features Caching
    # ========================================================================
    
    async def get_features(
        self,
        symbol: str,
        timestamp: Optional[datetime] = None,
    ) -> Optional[Dict[str, float]]:
        """
        Get cached features.
        
        Args:
            symbol: Trading symbol
            timestamp: Feature timestamp (optional)
        
        Returns:
            Cached features dict or None
        """
        redis = await self._ensure_redis()
        
        key = f"features:{symbol}"
        if timestamp:
            key += f":{timestamp.isoformat()}"
        
        cached = await redis.get_json(key)
        if cached:
            logger.debug("cache_hit_features", symbol=symbol)
            return cached
        return None
    
    async def cache_features(
        self,
        symbol: str,
        features: Dict[str, float],
        ttl: Optional[int] = None,
        timestamp: Optional[datetime] = None,
    ) -> bool:
        """
        Cache calculated features.
        
        Args:
            symbol: Trading symbol
            features: Features dictionary
            ttl: Time-to-live in seconds
            timestamp: Feature timestamp (optional)
        
        Returns:
            True if cached successfully
        """
        redis = await self._ensure_redis()
        
        key = f"features:{symbol}"
        if timestamp:
            key += f":{timestamp.isoformat()}"
        
        # Use longer TTL for features (they don't change as often)
        feature_ttl = ttl or (self.default_ttl * 3)  # 15 minutes default
        
        success = await redis.set_json(key, features, ttl=feature_ttl)
        if success:
            logger.debug("cache_set_features", symbol=symbol, feature_count=len(features))
        return success
    
    # ========================================================================
    # Predictions Caching
    # ========================================================================
    
    async def get_predictions(
        self,
        symbol: str,
        engine: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached model predictions.
        
        Args:
            symbol: Trading symbol
            engine: Engine name (optional)
        
        Returns:
            Cached predictions or None
        """
        redis = await self._ensure_redis()
        
        key = f"predictions:{symbol}"
        if engine:
            key += f":{engine}"
        
        cached = await redis.get_json(key)
        if cached:
            logger.debug("cache_hit_predictions", symbol=symbol, engine=engine)
            return cached
        return None
    
    async def cache_predictions(
        self,
        symbol: str,
        predictions: Dict[str, Any],
        ttl: Optional[int] = None,
        engine: Optional[str] = None,
    ) -> bool:
        """
        Cache model predictions.
        
        Args:
            symbol: Trading symbol
            predictions: Predictions dictionary
            ttl: Time-to-live in seconds
            engine: Engine name (optional)
        
        Returns:
            True if cached successfully
        """
        redis = await self._ensure_redis()
        
        key = f"predictions:{symbol}"
        if engine:
            key += f":{engine}"
        
        # Predictions are valid for shorter time (1-2 minutes)
        prediction_ttl = ttl or (self.default_ttl // 3)  # ~1.5 minutes default
        
        success = await redis.set_json(key, predictions, ttl=prediction_ttl)
        if success:
            logger.debug("cache_set_predictions", symbol=symbol, engine=engine)
        return success
    
    # ========================================================================
    # Metrics Caching
    # ========================================================================
    
    async def get_metrics(
        self,
        symbol: str,
        period: str = "daily",
    ) -> Optional[Dict[str, float]]:
        """
        Get cached performance metrics.
        
        Args:
            symbol: Trading symbol
            period: Time period (daily, weekly, monthly)
        
        Returns:
            Cached metrics or None
        """
        redis = await self._ensure_redis()
        
        key = f"metrics:{symbol}:{period}"
        cached = await redis.get_json(key)
        if cached:
            logger.debug("cache_hit_metrics", symbol=symbol, period=period)
            return cached
        return None
    
    async def cache_metrics(
        self,
        symbol: str,
        metrics: Dict[str, float],
        period: str = "daily",
        ttl: Optional[int] = None,
    ) -> bool:
        """
        Cache performance metrics.
        
        Args:
            symbol: Trading symbol
            metrics: Metrics dictionary
            period: Time period
            ttl: Time-to-live in seconds
        
        Returns:
            True if cached successfully
        """
        redis = await self._ensure_redis()
        
        key = f"metrics:{symbol}:{period}"
        
        # Metrics can be cached longer (1 hour default)
        metrics_ttl = ttl or 3600
        
        success = await redis.set_json(key, metrics, ttl=metrics_ttl)
        if success:
            logger.debug("cache_set_metrics", symbol=symbol, period=period)
        return success
    
    # ========================================================================
    # Regime Caching
    # ========================================================================
    
    async def get_regime(self, symbol: str) -> Optional[str]:
        """
        Get cached market regime.
        
        Args:
            symbol: Trading symbol
        
        Returns:
            Cached regime or None
        """
        redis = await self._ensure_redis()
        
        key = f"regime:{symbol}"
        cached = await redis.get(key)
        return cached
    
    async def cache_regime(
        self,
        symbol: str,
        regime: str,
        ttl: Optional[int] = None,
    ) -> bool:
        """
        Cache market regime.
        
        Args:
            symbol: Trading symbol
            regime: Regime name (TREND, RANGE, PANIC)
            ttl: Time-to-live in seconds
        
        Returns:
            True if cached successfully
        """
        redis = await self._ensure_redis()
        
        key = f"regime:{symbol}"
        regime_ttl = ttl or self.default_ttl
        
        success = await redis.set(key, regime, ttl=regime_ttl)
        if success:
            logger.debug("cache_set_regime", symbol=symbol, regime=regime)
        return success
    
    # ========================================================================
    # Cache Invalidation
    # ========================================================================
    
    async def invalidate_prices(self, symbol: str) -> int:
        """
        Invalidate all cached prices for a symbol.
        
        Args:
            symbol: Trading symbol
        
        Returns:
            Number of keys deleted
        """
        redis = await self._ensure_redis()
        pattern = f"prices:{symbol}*"
        count = await redis.clear_pattern(pattern)
        logger.info("cache_invalidated_prices", symbol=symbol, count=count)
        return count
    
    async def invalidate_features(self, symbol: str) -> int:
        """Invalidate all cached features for a symbol."""
        redis = await self._ensure_redis()
        pattern = f"features:{symbol}*"
        count = await redis.clear_pattern(pattern)
        logger.info("cache_invalidated_features", symbol=symbol, count=count)
        return count
    
    async def invalidate_predictions(self, symbol: str) -> int:
        """Invalidate all cached predictions for a symbol."""
        redis = await self._ensure_redis()
        pattern = f"predictions:{symbol}*"
        count = await redis.clear_pattern(pattern)
        logger.info("cache_invalidated_predictions", symbol=symbol, count=count)
        return count
    
    async def invalidate_all(self, symbol: str) -> int:
        """
        Invalidate all cached data for a symbol.
        
        Args:
            symbol: Trading symbol
        
        Returns:
            Total number of keys deleted
        """
        prices = await self.invalidate_prices(symbol)
        features = await self.invalidate_features(symbol)
        predictions = await self.invalidate_predictions(symbol)
        
        redis = await self._ensure_redis()
        metrics = await redis.clear_pattern(f"metrics:{symbol}*")
        regime = await redis.clear_pattern(f"regime:{symbol}*")
        
        total = prices + features + predictions + metrics + regime
        logger.info("cache_invalidated_all", symbol=symbol, total=total)
        return total
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache stats
        """
        redis = await self._ensure_redis()
        
        try:
            info = await redis.client.info()  # type: ignore
            return {
                "connected": await redis.ping(),
                "used_memory": info.get("used_memory_human", "unknown"),
                "connected_clients": info.get("connected_clients", 0),
                "total_keys": info.get("db0", {}).get("keys", 0),
            }
        except Exception as e:
            logger.error("cache_stats_error", error=str(e))
            return {"error": str(e)}

