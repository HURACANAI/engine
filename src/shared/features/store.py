"""
Feature Store Implementation

Centralized feature registration, versioning, and lifecycle management.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set

import psycopg2
import psycopg2.extras
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class FeatureRecord:
    """Single feature registration"""
    feature_id: str
    name: str
    version: str
    created_at: datetime

    # Metadata
    description: Optional[str] = None
    dependencies: Optional[List[str]] = None
    compute_fn_hash: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None

    # Lifecycle
    deprecated: bool = False
    deprecated_at: Optional[datetime] = None
    deprecated_reason: Optional[str] = None
    replaced_by_version: Optional[str] = None


@dataclass
class FeatureSetRecord:
    """Feature set snapshot (collection of features)"""
    feature_set_id: str
    name: str
    created_at: datetime

    # Features in this set
    feature_versions: Dict[str, str]  # {feature_name: version}
    feature_checksums: Dict[str, str]  # {feature_name: sha256}

    # Lifecycle
    deprecated: bool = False
    deprecated_at: Optional[datetime] = None
    deprecated_reason: Optional[str] = None


class FeatureStore:
    """
    Feature Store - Centralized Feature Management

    Provides versioning, reproducibility, and lifecycle management for features.

    Example:
        store = FeatureStore(dsn="postgresql://...")

        # Register individual features
        store.register_feature(
            name="rsi_14",
            version="v1.0",
            compute_fn=compute_rsi_14,
            dependencies=["close_price"],
            description="14-period RSI"
        )

        # Create feature set (snapshot)
        feature_set_id = store.create_feature_set(
            name="baseline_v1",
            feature_versions={
                "rsi_14": "v1.0",
                "macd": "v2.1",
                "bb_width": "v1.3"
            }
        )

        # Later: Get exact feature set
        features = store.get_feature_set(feature_set_id)

        # Deprecate old version
        store.deprecate_feature(
            name="rsi_14",
            version="v1.0",
            reason="Replaced by v1.1 with bug fix",
            replaced_by="v1.1"
        )
    """

    def __init__(self, dsn: str):
        """
        Initialize feature store

        Args:
            dsn: PostgreSQL connection string
        """
        self.dsn = dsn
        self._conn: Optional[psycopg2.extensions.connection] = None

    def _connect(self) -> psycopg2.extensions.connection:
        """Get database connection"""
        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(self.dsn)
        return self._conn

    def _compute_function_hash(self, compute_fn: Optional[Callable]) -> Optional[str]:
        """
        Compute hash of function code for reproducibility

        Args:
            compute_fn: Function to hash

        Returns:
            SHA256 hash of function code, or None if no function
        """
        if compute_fn is None:
            return None

        try:
            import inspect
            source = inspect.getsource(compute_fn)
            return hashlib.sha256(source.encode()).hexdigest()
        except Exception as e:
            logger.warning("could_not_hash_function", error=str(e))
            return None

    def register_feature(
        self,
        name: str,
        version: str,
        compute_fn: Optional[Callable] = None,
        dependencies: Optional[List[str]] = None,
        parameters: Optional[Dict[str, Any]] = None,
        description: Optional[str] = None,
    ) -> str:
        """
        Register a new feature version

        Args:
            name: Feature name (e.g., "rsi_14")
            version: Version string (e.g., "v1.0", "v2.1")
            compute_fn: Function that computes this feature
            dependencies: List of dependency feature names
            parameters: Feature parameters (e.g., {"period": 14})
            description: Human-readable description

        Returns:
            feature_id (unique ID for this name+version)
        """
        conn = self._connect()

        # Compute function hash for reproducibility
        fn_hash = self._compute_function_hash(compute_fn)

        # Generate feature ID
        feature_id = f"{name}_{version}"

        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO features (
                    feature_id, name, version,
                    description, dependencies, compute_fn_hash, parameters
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s
                )
                ON CONFLICT (feature_id) DO UPDATE
                SET description = EXCLUDED.description,
                    dependencies = EXCLUDED.dependencies,
                    compute_fn_hash = EXCLUDED.compute_fn_hash,
                    parameters = EXCLUDED.parameters
                RETURNING feature_id
                """,
                (
                    feature_id, name, version,
                    description,
                    dependencies,
                    fn_hash,
                    json.dumps(parameters) if parameters else None,
                )
            )
            conn.commit()

        logger.info(
            "feature_registered",
            feature_id=feature_id,
            name=name,
            version=version,
            dependencies=dependencies,
        )

        return feature_id

    def create_feature_set(
        self,
        name: str,
        feature_versions: Dict[str, str],
        description: Optional[str] = None,
    ) -> str:
        """
        Create a feature set (snapshot of features)

        Args:
            name: Feature set name (e.g., "baseline_v1")
            feature_versions: Dict of {feature_name: version}
            description: Human-readable description

        Returns:
            feature_set_id

        Example:
            feature_set_id = store.create_feature_set(
                name="baseline_v1",
                feature_versions={
                    "rsi_14": "v1.0",
                    "macd": "v2.1",
                    "bb_width": "v1.3"
                }
            )
        """
        conn = self._connect()

        # Compute checksums for each feature
        feature_checksums = {}
        for fname, fversion in feature_versions.items():
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT compute_fn_hash FROM features WHERE name = %s AND version = %s",
                    (fname, fversion)
                )
                row = cur.fetchone()
                if row and row[0]:
                    feature_checksums[fname] = row[0]

        # Generate feature set ID with timestamp
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        content_hash = hashlib.sha256(
            json.dumps(feature_versions, sort_keys=True).encode()
        ).hexdigest()[:8]
        feature_set_id = f"fs_{timestamp}_{content_hash}"

        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO feature_sets (
                    feature_set_id, name, feature_versions, feature_checksums, description
                ) VALUES (
                    %s, %s, %s, %s, %s
                )
                RETURNING feature_set_id
                """,
                (
                    feature_set_id,
                    name,
                    json.dumps(feature_versions),
                    json.dumps(feature_checksums),
                    description,
                )
            )
            conn.commit()

        logger.info(
            "feature_set_created",
            feature_set_id=feature_set_id,
            name=name,
            num_features=len(feature_versions),
        )

        return feature_set_id

    def get_feature_set(self, feature_set_id: str) -> Optional[FeatureSetRecord]:
        """
        Get a feature set by ID

        Args:
            feature_set_id: Feature set identifier

        Returns:
            FeatureSetRecord or None if not found
        """
        conn = self._connect()

        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM feature_sets WHERE feature_set_id = %s",
                (feature_set_id,)
            )
            row = cur.fetchone()

        if not row:
            return None

        return FeatureSetRecord(
            feature_set_id=row['feature_set_id'],
            name=row['name'],
            created_at=row['created_at'],
            feature_versions=row['feature_versions'],
            feature_checksums=row['feature_checksums'],
            deprecated=row.get('deprecated', False),
            deprecated_at=row.get('deprecated_at'),
            deprecated_reason=row.get('deprecated_reason'),
        )

    def get_feature(
        self,
        name: str,
        version: Optional[str] = None
    ) -> Optional[FeatureRecord]:
        """
        Get a feature by name and version

        Args:
            name: Feature name
            version: Version (None = latest non-deprecated)

        Returns:
            FeatureRecord or None if not found
        """
        conn = self._connect()

        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            if version:
                cur.execute(
                    "SELECT * FROM features WHERE name = %s AND version = %s",
                    (name, version)
                )
            else:
                # Get latest non-deprecated version
                cur.execute(
                    """
                    SELECT * FROM features
                    WHERE name = %s AND NOT deprecated
                    ORDER BY created_at DESC
                    LIMIT 1
                    """,
                    (name,)
                )

            row = cur.fetchone()

        if not row:
            return None

        return FeatureRecord(
            feature_id=row['feature_id'],
            name=row['name'],
            version=row['version'],
            created_at=row['created_at'],
            description=row.get('description'),
            dependencies=row.get('dependencies'),
            compute_fn_hash=row.get('compute_fn_hash'),
            parameters=row.get('parameters'),
            deprecated=row.get('deprecated', False),
            deprecated_at=row.get('deprecated_at'),
            deprecated_reason=row.get('deprecated_reason'),
            replaced_by_version=row.get('replaced_by_version'),
        )

    def list_versions(self, name: str) -> List[FeatureRecord]:
        """
        List all versions of a feature

        Args:
            name: Feature name

        Returns:
            List of FeatureRecord (newest first)
        """
        conn = self._connect()

        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT * FROM features
                WHERE name = %s
                ORDER BY created_at DESC
                """,
                (name,)
            )
            rows = cur.fetchall()

        return [
            FeatureRecord(
                feature_id=row['feature_id'],
                name=row['name'],
                version=row['version'],
                created_at=row['created_at'],
                description=row.get('description'),
                dependencies=row.get('dependencies'),
                compute_fn_hash=row.get('compute_fn_hash'),
                parameters=row.get('parameters'),
                deprecated=row.get('deprecated', False),
                deprecated_at=row.get('deprecated_at'),
                deprecated_reason=row.get('deprecated_reason'),
                replaced_by_version=row.get('replaced_by_version'),
            )
            for row in rows
        ]

    def deprecate_feature(
        self,
        name: str,
        version: str,
        reason: str,
        replaced_by: Optional[str] = None,
    ) -> None:
        """
        Deprecate a feature version

        Args:
            name: Feature name
            version: Version to deprecate
            reason: Deprecation reason
            replaced_by: Replacement version (if any)
        """
        conn = self._connect()

        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE features
                SET deprecated = TRUE,
                    deprecated_at = NOW(),
                    deprecated_reason = %s,
                    replaced_by_version = %s
                WHERE name = %s AND version = %s
                """,
                (reason, replaced_by, name, version)
            )
            conn.commit()

        logger.info(
            "feature_deprecated",
            name=name,
            version=version,
            reason=reason,
            replaced_by=replaced_by,
        )

    def deprecate_feature_set(
        self,
        feature_set_id: str,
        reason: str,
    ) -> None:
        """
        Deprecate a feature set

        Args:
            feature_set_id: Feature set identifier
            reason: Deprecation reason
        """
        conn = self._connect()

        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE feature_sets
                SET deprecated = TRUE,
                    deprecated_at = NOW(),
                    deprecated_reason = %s
                WHERE feature_set_id = %s
                """,
                (reason, feature_set_id)
            )
            conn.commit()

        logger.info(
            "feature_set_deprecated",
            feature_set_id=feature_set_id,
            reason=reason,
        )

    def close(self):
        """Close database connection"""
        if self._conn and not self._conn.closed:
            self._conn.close()
