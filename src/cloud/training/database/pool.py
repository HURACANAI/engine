"""Database connection pooling for efficient connection management."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Optional
import threading

import psycopg2
from psycopg2 import pool
import structlog

logger = structlog.get_logger(__name__)


class DatabaseConnectionPool:
    """
    Thread-safe database connection pool manager.
    
    Usage:
        pool = DatabaseConnectionPool(dsn="postgresql://...", minconn=2, maxconn=10)
        
        with pool.get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT ...")
            result = cur.fetchall()
    """
    
    _pools: dict[str, pool.ThreadedConnectionPool] = {}
    _lock = threading.Lock()
    
    def __init__(
        self,
        dsn: str,
        minconn: int = 2,
        maxconn: int = 10,
        pool_key: Optional[str] = None,
    ) -> None:
        """
        Initialize connection pool.
        
        Args:
            dsn: Database connection string
            minconn: Minimum number of connections in pool
            maxconn: Maximum number of connections in pool
            pool_key: Optional key to share pool across instances (defaults to dsn)
        """
        self.dsn = dsn
        self.minconn = minconn
        self.maxconn = maxconn
        self.pool_key = pool_key or dsn
        
        with self._lock:
            if self.pool_key not in self._pools:
                try:
                    self._pool = pool.ThreadedConnectionPool(
                        minconn=minconn,
                        maxconn=maxconn,
                        dsn=dsn,
                    )
                    self._pools[self.pool_key] = self._pool
                    logger.info(
                        "connection_pool_created",
                        pool_key=self.pool_key,
                        minconn=minconn,
                        maxconn=maxconn,
                    )
                except Exception as e:
                    logger.error(
                        "connection_pool_creation_failed",
                        pool_key=self.pool_key,
                        error=str(e),
                    )
                    raise
            else:
                self._pool = self._pools[self.pool_key]
                logger.debug("using_existing_pool", pool_key=self.pool_key)
    
    @contextmanager
    def get_connection(self):
        """
        Get a connection from the pool (context manager).
        
        Yields:
            psycopg2.connection: Database connection
            
        Example:
            with pool.get_connection() as conn:
                cur = conn.cursor()
                cur.execute("SELECT ...")
        """
        conn = None
        try:
            conn = self._pool.getconn()
            yield conn
            conn.commit()
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error("database_operation_failed", error=str(e))
            raise
        finally:
            if conn:
                self._pool.putconn(conn)
    
    def close_all(self) -> None:
        """Close all connections in the pool."""
        with self._lock:
            if self.pool_key in self._pools:
                try:
                    self._pools[self.pool_key].closeall()
                    del self._pools[self.pool_key]
                    logger.info("connection_pool_closed", pool_key=self.pool_key)
                except Exception as e:
                    logger.error(
                        "connection_pool_close_failed",
                        pool_key=self.pool_key,
                        error=str(e),
                    )
    
    @classmethod
    def close_all_pools(cls) -> None:
        """Close all connection pools (useful for cleanup)."""
        with cls._lock:
            for pool_key, pool_instance in list(cls._pools.items()):
                try:
                    pool_instance.closeall()
                    logger.info("pool_closed", pool_key=pool_key)
                except Exception as e:
                    logger.error("pool_close_failed", pool_key=pool_key, error=str(e))
            cls._pools.clear()

