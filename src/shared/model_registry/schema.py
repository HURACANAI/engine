"""
Schema initialization and migration helpers
"""

from pathlib import Path
from typing import Optional

import psycopg2
import structlog

logger = structlog.get_logger(__name__)


def init_schema(dsn: str, schema_path: Optional[Path] = None) -> None:
    """
    Initialize database schema from SQL file

    Args:
        dsn: PostgreSQL connection string
        schema_path: Path to schema.sql (default: same directory)
    """
    if schema_path is None:
        schema_path = Path(__file__).parent / "schema.sql"

    if not schema_path.exists():
        raise FileNotFoundError(f"Schema file not found: {schema_path}")

    logger.info("initializing_schema", schema_path=str(schema_path))

    with open(schema_path) as f:
        schema_sql = f.read()

    conn = psycopg2.connect(dsn)
    try:
        with conn.cursor() as cur:
            cur.execute(schema_sql)
        conn.commit()
        logger.info("schema_initialized_successfully")
    except Exception as e:
        conn.rollback()
        logger.error("schema_initialization_failed", error=str(e))
        raise
    finally:
        conn.close()


def migrate_schema(dsn: str, migration_version: str) -> None:
    """
    Apply schema migration

    Args:
        dsn: PostgreSQL connection string
        migration_version: Migration version (e.g., "v1_to_v2")

    Note:
        Migration files should be in migrations/ directory
        Named as: {version}.sql
    """
    migrations_dir = Path(__file__).parent / "migrations"
    migration_path = migrations_dir / f"{migration_version}.sql"

    if not migration_path.exists():
        raise FileNotFoundError(f"Migration not found: {migration_path}")

    logger.info("applying_migration", version=migration_version)

    with open(migration_path) as f:
        migration_sql = f.read()

    conn = psycopg2.connect(dsn)
    try:
        with conn.cursor() as cur:
            cur.execute(migration_sql)
        conn.commit()
        logger.info("migration_applied_successfully", version=migration_version)
    except Exception as e:
        conn.rollback()
        logger.error("migration_failed", version=migration_version, error=str(e))
        raise
    finally:
        conn.close()


def check_schema_exists(dsn: str) -> bool:
    """
    Check if schema is initialized

    Args:
        dsn: PostgreSQL connection string

    Returns:
        True if unified_models table exists
    """
    conn = psycopg2.connect(dsn)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_name = 'unified_models'
                )
                """
            )
            exists = cur.fetchone()[0]
        return exists
    finally:
        conn.close()
