"""
Feature Store Schema Initialization

SQL schema for feature versioning and feature sets.
"""

from pathlib import Path
from typing import Optional

import psycopg2
import structlog

logger = structlog.get_logger(__name__)


# Feature Store SQL Schema
FEATURE_STORE_SCHEMA = """
-- FEATURE STORE SCHEMA
-- Version: 1.0.0

-- Individual features table
CREATE TABLE IF NOT EXISTS features (
    -- Primary identification
    feature_id VARCHAR(200) PRIMARY KEY,  -- {name}_{version}
    name VARCHAR(100) NOT NULL,
    version VARCHAR(50) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Metadata
    description TEXT,
    dependencies TEXT[],  -- List of dependency feature names
    compute_fn_hash VARCHAR(64),  -- SHA256 of function code
    parameters JSONB,  -- Feature parameters

    -- Lifecycle
    deprecated BOOLEAN DEFAULT FALSE,
    deprecated_at TIMESTAMP WITH TIME ZONE,
    deprecated_reason TEXT,
    replaced_by_version VARCHAR(50),  -- Version that replaces this

    -- Unique constraint
    UNIQUE(name, version)
);

-- Feature sets table (snapshots of features)
CREATE TABLE IF NOT EXISTS feature_sets (
    -- Primary identification
    feature_set_id VARCHAR(100) PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Features in this set
    feature_versions JSONB NOT NULL,  -- {feature_name: version}
    feature_checksums JSONB,  -- {feature_name: sha256} for reproducibility

    -- Metadata
    description TEXT,

    -- Lifecycle
    deprecated BOOLEAN DEFAULT FALSE,
    deprecated_at TIMESTAMP WITH TIME ZONE,
    deprecated_reason TEXT
);

-- Indexes for fast queries
CREATE INDEX IF NOT EXISTS idx_features_name ON features(name);
CREATE INDEX IF NOT EXISTS idx_features_name_version ON features(name, version);
CREATE INDEX IF NOT EXISTS idx_features_deprecated ON features(deprecated);

CREATE INDEX IF NOT EXISTS idx_feature_sets_name ON feature_sets(name);
CREATE INDEX IF NOT EXISTS idx_feature_sets_deprecated ON feature_sets(deprecated);
CREATE INDEX IF NOT EXISTS idx_feature_sets_created ON feature_sets(created_at DESC);

-- Views for common queries

-- Active features (non-deprecated)
CREATE OR REPLACE VIEW v_active_features AS
SELECT
    feature_id,
    name,
    version,
    created_at,
    description,
    dependencies,
    parameters
FROM features
WHERE NOT deprecated
ORDER BY name, created_at DESC;

-- Latest version of each feature
CREATE OR REPLACE VIEW v_latest_features AS
SELECT DISTINCT ON (name)
    feature_id,
    name,
    version,
    created_at,
    description,
    dependencies,
    parameters
FROM features
WHERE NOT deprecated
ORDER BY name, created_at DESC;

-- Active feature sets
CREATE OR REPLACE VIEW v_active_feature_sets AS
SELECT * FROM feature_sets
WHERE NOT deprecated
ORDER BY created_at DESC;

-- Comments
COMMENT ON TABLE features IS 'Individual feature registrations with versioning';
COMMENT ON TABLE feature_sets IS 'Feature set snapshots for reproducibility';
COMMENT ON COLUMN features.compute_fn_hash IS 'SHA256 hash of feature computation function for reproducibility';
COMMENT ON COLUMN feature_sets.feature_checksums IS 'Checksums of all features for exact reproducibility';
"""


def init_feature_schema(dsn: str) -> None:
    """
    Initialize feature store schema

    Args:
        dsn: PostgreSQL connection string
    """
    logger.info("initializing_feature_store_schema")

    conn = psycopg2.connect(dsn)
    try:
        with conn.cursor() as cur:
            cur.execute(FEATURE_STORE_SCHEMA)
        conn.commit()
        logger.info("feature_store_schema_initialized_successfully")
    except Exception as e:
        conn.rollback()
        logger.error("feature_store_schema_initialization_failed", error=str(e))
        raise
    finally:
        conn.close()


def check_feature_schema_exists(dsn: str) -> bool:
    """
    Check if feature store schema is initialized

    Args:
        dsn: PostgreSQL connection string

    Returns:
        True if features table exists
    """
    conn = psycopg2.connect(dsn)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_name = 'features'
                )
                """
            )
            exists = cur.fetchone()[0]
        return exists
    finally:
        conn.close()
