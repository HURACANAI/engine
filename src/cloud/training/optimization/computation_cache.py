"""
Computation Caching System

Caches expensive computations to improve performance.

Caches:
1. Feature engineering results
2. Model predictions
3. Database query results
4. Expensive calculations

Uses LRU cache with TTL for automatic expiration.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Any, Callable, Dict, Optional

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class CacheEntry:
    """Single cache entry."""

    key: str
    value: Any
    timestamp: datetime
    ttl_seconds: int
    hit_count: int = 0

    def is_expired(self) -> bool:
        """Check if entry is expired."""
        age = (datetime.now() - self.timestamp).total_seconds()
        return age > self.ttl_seconds


class ComputationCache:
    """
    Computation cache with TTL and LRU eviction.

    Caches expensive computations to improve performance.

    Usage:
        cache = ComputationCache(max_size=1000, default_ttl=3600)

        # Cache a function result
        result = cache.get_or_compute(
            key="feature_engineering_btc",
            compute_fn=lambda: expensive_feature_engineering(symbol="BTC/USDT"),
            ttl_seconds=3600,
        )

        # Clear cache
        cache.clear()
    """

    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        """
        Initialize computation cache.

        Args:
            max_size: Maximum number of cache entries
            default_ttl: Default TTL in seconds
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: Dict[str, CacheEntry] = {}
        self.hits = 0
        self.misses = 0

        logger.info("computation_cache_initialized", max_size=max_size, default_ttl=default_ttl)

    def get_or_compute(
        self,
        key: str,
        compute_fn: Callable[[], Any],
        ttl_seconds: Optional[int] = None,
    ) -> Any:
        """
        Get value from cache or compute if not found.

        Args:
            key: Cache key
            compute_fn: Function to compute value if not in cache
            ttl_seconds: TTL in seconds (uses default if not provided)

        Returns:
            Cached or computed value
        """
        ttl = ttl_seconds or self.default_ttl

        # Check cache
        if key in self.cache:
            entry = self.cache[key]

            # Check if expired
            if entry.is_expired():
                # Remove expired entry
                del self.cache[key]
                self.misses += 1
            else:
                # Cache hit
                entry.hit_count += 1
                self.hits += 1
                logger.debug("cache_hit", key=key, hit_count=entry.hit_count)
                return entry.value

        # Cache miss - compute value
        self.misses += 1
        logger.debug("cache_miss", key=key)

        # Compute value
        value = compute_fn()

        # Store in cache
        self._store(key, value, ttl)

        return value

    def _store(self, key: str, value: Any, ttl_seconds: int) -> None:
        """Store value in cache."""
        # Evict if cache is full
        if len(self.cache) >= self.max_size:
            self._evict_lru()

        # Create entry
        entry = CacheEntry(
            key=key,
            value=value,
            timestamp=datetime.now(),
            ttl_seconds=ttl_seconds,
            hit_count=0,
        )

        self.cache[key] = entry

    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if not self.cache:
            return

        # Find entry with lowest hit count
        lru_key = min(self.cache.keys(), key=lambda k: self.cache[k].hit_count)
        del self.cache[lru_key]

        logger.debug("cache_evicted", key=lru_key)

    def clear(self) -> None:
        """Clear all cache entries."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
        logger.info("cache_cleared")

    def get_statistics(self) -> dict:
        """Get cache statistics."""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0.0

        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'total_requests': total_requests,
        }

    def cleanup_expired(self) -> int:
        """Remove expired entries and return count removed."""
        expired_keys = [key for key, entry in self.cache.items() if entry.is_expired()]

        for key in expired_keys:
            del self.cache[key]

        logger.debug("cache_cleanup", removed=len(expired_keys))

        return len(expired_keys)


# Global cache instance
_global_cache: Optional[ComputationCache] = None


def get_cache() -> ComputationCache:
    """Get global cache instance."""
    global _global_cache
    if _global_cache is None:
        _global_cache = ComputationCache()
    return _global_cache


def cached(ttl_seconds: int = 3600):
    """
    Decorator to cache function results.

    Usage:
        @cached(ttl_seconds=3600)
        def expensive_function(symbol: str) -> Dict:
            # Expensive computation
            return result
    """
    cache = get_cache()

    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            # Create cache key from function name and arguments
            key = f"{func.__name__}_{hash((args, tuple(sorted(kwargs.items()))))}"

            return cache.get_or_compute(
                key=key,
                compute_fn=lambda: func(*args, **kwargs),
                ttl_seconds=ttl_seconds,
            )

        return wrapper

    return decorator

