"""
Caching Layer - Redis-based caching for frequently accessed data.

Provides:
- Fast in-memory caching
- Automatic expiration
- Cache invalidation strategies
- Async support
"""

from .redis_client import RedisClient, get_redis_client
from .cache_manager import CacheManager

__all__ = [
    "RedisClient",
    "get_redis_client",
    "CacheManager",
]

