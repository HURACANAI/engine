"""
Async Database Pool - Async PostgreSQL connection pool using asyncpg.

Provides async database operations with connection pooling.
"""

from __future__ import annotations

from typing import Optional, Any
import structlog

logger = structlog.get_logger(__name__)

# Try to import asyncpg, but handle gracefully if not available
try:
    import asyncpg
    HAS_ASYNCPG = True
except ImportError:
    HAS_ASYNCPG = False
    logger.warning("asyncpg_not_available", message="asyncpg not installed. Install with: pip install asyncpg")


class AsyncDatabasePool:
    """
    Async PostgreSQL connection pool using asyncpg.
    
    Usage:
        pool = AsyncDatabasePool(dsn="postgresql://...", min_size=2, max_size=10)
        async with pool.acquire() as conn:
            rows = await conn.fetch("SELECT * FROM prices WHERE symbol = $1", symbol)
    """
    
    def __init__(
        self,
        dsn: str,
        min_size: int = 2,
        max_size: int = 10,
        max_queries: int = 50000,
        max_inactive_connection_lifetime: float = 300.0,
    ) -> None:
        """
        Initialize async database pool.
        
        Args:
            dsn: Database connection string
            min_size: Minimum pool size
            max_size: Maximum pool size
            max_queries: Maximum queries per connection
            max_inactive_connection_lifetime: Max seconds before closing idle connections
        """
        if not HAS_ASYNCPG:
            raise ImportError(
                "asyncpg is not installed. Install with: pip install asyncpg"
            )
        
        self.dsn = dsn
        self.min_size = min_size
        self.max_size = max_size
        self.max_queries = max_queries
        self.max_inactive_connection_lifetime = max_inactive_connection_lifetime
        
        self.pool: Optional[asyncpg.Pool] = None
        self._initialized = False
        
        logger.info(
            "async_db_pool_initialized",
            min_size=min_size,
            max_size=max_size
        )
    
    async def initialize(self) -> None:
        """Initialize the connection pool."""
        if self._initialized:
            return
        
        try:
            self.pool = await asyncpg.create_pool(
                dsn=self.dsn,
                min_size=self.min_size,
                max_size=self.max_size,
                max_queries=self.max_queries,
                max_inactive_connection_lifetime=self.max_inactive_connection_lifetime,
            )
            self._initialized = True
            logger.info("async_db_pool_connected")
        except Exception as e:
            logger.error("async_db_pool_connection_failed", error=str(e))
            raise
    
    async def close(self) -> None:
        """Close the connection pool."""
        if self.pool:
            await self.pool.close()
            self._initialized = False
            logger.info("async_db_pool_closed")
    
    async def acquire(self) -> asyncpg.Connection:
        """
        Acquire a connection from the pool.
        
        Returns:
            Database connection (use as async context manager)
        
        Example:
            async with pool.acquire() as conn:
                rows = await conn.fetch("SELECT * FROM table")
        """
        if not self._initialized:
            await self.initialize()
        
        if self.pool is None:
            raise RuntimeError("Pool not initialized")
        
        return await self.pool.acquire()
    
    async def release(self, connection: asyncpg.Connection) -> None:
        """
        Release a connection back to the pool.
        
        Args:
            connection: Connection to release
        """
        if self.pool:
            await self.pool.release(connection)
    
    async def execute(self, query: str, *args: Any) -> str:
        """
        Execute a query and return status.
        
        Args:
            query: SQL query
            *args: Query parameters
        
        Returns:
            Status string
        """
        async with self.acquire() as conn:
            return await conn.execute(query, *args)
    
    async def fetch(self, query: str, *args: Any) -> list[asyncpg.Record]:
        """
        Execute a query and return all rows.
        
        Args:
            query: SQL query
            *args: Query parameters
        
        Returns:
            List of records
        """
        async with self.acquire() as conn:
            return await conn.fetch(query, *args)
    
    async def fetchrow(self, query: str, *args: Any) -> Optional[asyncpg.Record]:
        """
        Execute a query and return one row.
        
        Args:
            query: SQL query
            *args: Query parameters
        
        Returns:
            Single record or None
        """
        async with self.acquire() as conn:
            return await conn.fetchrow(query, *args)
    
    async def fetchval(self, query: str, *args: Any) -> Any:
        """
        Execute a query and return a single value.
        
        Args:
            query: SQL query
            *args: Query parameters
        
        Returns:
            Single value
        """
        async with self.acquire() as conn:
            return await conn.fetchval(query, *args)
    
    def get_size(self) -> int:
        """
        Get current pool size.
        
        Returns:
            Number of connections in pool
        """
        if self.pool:
            return self.pool.get_size()
        return 0
    
    def get_idle_size(self) -> int:
        """
        Get number of idle connections.
        
        Returns:
            Number of idle connections
        """
        if self.pool:
            return self.pool.get_idle_size()
        return 0

