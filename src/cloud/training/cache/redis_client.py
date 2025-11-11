"""
Redis Client - Async Redis connection manager.

Provides async Redis operations with connection pooling and error handling.
"""

from __future__ import annotations

import json
from typing import Any, Optional
from datetime import timedelta

import structlog

logger = structlog.get_logger(__name__)

# Try to import redis, but handle gracefully if not available
try:
    import redis.asyncio as aioredis
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False
    logger.warning("redis_not_available", message="Redis not installed. Install with: pip install redis")


class RedisClient:
    """
    Async Redis client wrapper with connection pooling.
    
    Usage:
        client = RedisClient(host='localhost', port=6379)
        await client.set('key', 'value', ttl=300)
        value = await client.get('key')
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        max_connections: int = 50,
        decode_responses: bool = True,
    ) -> None:
        """
        Initialize Redis client.
        
        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
            password: Redis password (optional)
            max_connections: Maximum connection pool size
            decode_responses: Whether to decode responses as strings
        """
        if not HAS_REDIS:
            raise ImportError(
                "Redis is not installed. Install with: pip install redis"
            )
        
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.max_connections = max_connections
        
        # Create connection pool
        self.pool: Optional[aioredis.ConnectionPool] = None
        self.client: Optional[aioredis.Redis] = None
        self._connected = False
        
        logger.info(
            "redis_client_initialized",
            host=host,
            port=port,
            db=db,
            max_connections=max_connections
        )
    
    async def connect(self) -> None:
        """Connect to Redis."""
        if self._connected:
            return
        
        try:
            self.pool = aioredis.ConnectionPool(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                max_connections=self.max_connections,
                decode_responses=True,
            )
            self.client = aioredis.Redis(connection_pool=self.pool)
            
            # Test connection
            await self.client.ping()
            self._connected = True
            
            logger.info("redis_connected", host=self.host, port=self.port)
        except Exception as e:
            logger.error("redis_connection_failed", error=str(e), host=self.host, port=self.port)
            raise
    
    async def disconnect(self) -> None:
        """Disconnect from Redis."""
        if self.client:
            await self.client.close()
        if self.pool:
            await self.pool.disconnect()
        self._connected = False
        logger.info("redis_disconnected")
    
    async def get(self, key: str) -> Optional[str]:
        """
        Get value from Redis.
        
        Args:
            key: Cache key
        
        Returns:
            Cached value or None if not found
        """
        if not self._connected:
            await self.connect()
        
        try:
            value = await self.client.get(key)  # type: ignore
            if value:
                logger.debug("cache_hit", key=key)
            else:
                logger.debug("cache_miss", key=key)
            return value
        except Exception as e:
            logger.error("redis_get_error", key=key, error=str(e))
            return None
    
    async def set(
        self,
        key: str,
        value: str,
        ttl: Optional[int] = None,
    ) -> bool:
        """
        Set value in Redis.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (None = no expiration)
        
        Returns:
            True if successful
        """
        if not self._connected:
            await self.connect()
        
        try:
            if ttl:
                await self.client.setex(key, ttl, value)  # type: ignore
            else:
                await self.client.set(key, value)  # type: ignore
            
            logger.debug("cache_set", key=key, ttl=ttl)
            return True
        except Exception as e:
            logger.error("redis_set_error", key=key, error=str(e))
            return False
    
    async def delete(self, *keys: str) -> int:
        """
        Delete keys from Redis.
        
        Args:
            *keys: Keys to delete
        
        Returns:
            Number of keys deleted
        """
        if not self._connected:
            await self.connect()
        
        try:
            count = await self.client.delete(*keys)  # type: ignore
            logger.debug("cache_delete", keys=keys, count=count)
            return count
        except Exception as e:
            logger.error("redis_delete_error", keys=keys, error=str(e))
            return 0
    
    async def exists(self, key: str) -> bool:
        """
        Check if key exists in Redis.
        
        Args:
            key: Cache key
        
        Returns:
            True if key exists
        """
        if not self._connected:
            await self.connect()
        
        try:
            result = await self.client.exists(key)  # type: ignore
            return bool(result)
        except Exception as e:
            logger.error("redis_exists_error", key=key, error=str(e))
            return False
    
    async def expire(self, key: str, ttl: int) -> bool:
        """
        Set expiration on existing key.
        
        Args:
            key: Cache key
            ttl: Time-to-live in seconds
        
        Returns:
            True if expiration was set
        """
        if not self._connected:
            await self.connect()
        
        try:
            result = await self.client.expire(key, ttl)  # type: ignore
            return bool(result)
        except Exception as e:
            logger.error("redis_expire_error", key=key, ttl=ttl, error=str(e))
            return False
    
    async def get_json(self, key: str) -> Optional[Any]:
        """
        Get JSON value from Redis.
        
        Args:
            key: Cache key
        
        Returns:
            Deserialized JSON value or None
        """
        value = await self.get(key)
        if value is None:
            return None
        
        try:
            return json.loads(value)
        except json.JSONDecodeError as e:
            logger.error("redis_json_decode_error", key=key, error=str(e))
            return None
    
    async def set_json(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
    ) -> bool:
        """
        Set JSON value in Redis.
        
        Args:
            key: Cache key
            value: Value to serialize and cache
            ttl: Time-to-live in seconds
        
        Returns:
            True if successful
        """
        try:
            json_value = json.dumps(value)
            return await self.set(key, json_value, ttl=ttl)
        except (TypeError, ValueError) as e:
            logger.error("redis_json_encode_error", key=key, error=str(e))
            return False
    
    async def clear_pattern(self, pattern: str) -> int:
        """
        Delete all keys matching pattern.
        
        Args:
            pattern: Redis key pattern (e.g., "prices:*")
        
        Returns:
            Number of keys deleted
        """
        if not self._connected:
            await self.connect()
        
        try:
            count = 0
            async for key in self.client.scan_iter(match=pattern):  # type: ignore
                await self.delete(key)
                count += 1
            logger.info("cache_clear_pattern", pattern=pattern, count=count)
            return count
        except Exception as e:
            logger.error("redis_clear_pattern_error", pattern=pattern, error=str(e))
            return 0
    
    async def ping(self) -> bool:
        """
        Ping Redis server to check connection.
        
        Returns:
            True if connected
        """
        if not self._connected:
            await self.connect()
        
        try:
            await self.client.ping()  # type: ignore
            return True
        except Exception:
            return False


# Global Redis client instance
_redis_client: Optional[RedisClient] = None


async def get_redis_client() -> RedisClient:
    """
    Get or create global Redis client instance.
    
    Returns:
        RedisClient instance
    """
    global _redis_client
    
    if _redis_client is None:
        # Get config from environment or use defaults
        import os
        _redis_client = RedisClient(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            db=int(os.getenv("REDIS_DB", "0")),
            password=os.getenv("REDIS_PASSWORD"),
        )
        await _redis_client.connect()
    
    return _redis_client

