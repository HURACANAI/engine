"""
Run Manifest Loader

Load and verify saved manifests.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from typing import Optional

import psycopg2
import psycopg2.extras
import structlog

from .generator import RunManifest, GitInfo, EnvironmentInfo, DatasetInfo, TrainingResults

logger = structlog.get_logger(__name__)


def load_manifest(
    manifest_id: str,
    dsn: str
) -> Optional[RunManifest]:
    """
    Load manifest from database

    Args:
        manifest_id: Run manifest ID
        dsn: PostgreSQL connection string

    Returns:
        RunManifest or None if not found
    """
    conn = psycopg2.connect(dsn)

    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM run_manifests WHERE run_manifest_id = %s",
                (manifest_id,)
            )
            row = cur.fetchone()

        if not row:
            logger.warning("manifest_not_found", manifest_id=manifest_id)
            return None

        # Reconstruct manifest from database row
        git = GitInfo(
            commit=row['git_commit'],
            branch=row['git_branch'],
            dirty=row['git_dirty']
        )

        environment = EnvironmentInfo(
            python_version=row['python_version'],
            platform="unknown",  # Not stored in DB
            platform_version="unknown",
            package_versions=row['package_versions'],
            ray_cluster_info=row.get('ray_cluster_info')
        )

        dataset = DatasetInfo(
            symbols=row['symbol_list'],
            checksums=row['dataset_checksums']
        )

        results = None
        if row.get('models_trained'):
            results = TrainingResults(
                models_trained=row['models_trained'],
                models_published=row['models_published'],
                models_shadowed=row['models_shadowed'],
                models_rejected=row['models_rejected']
            )

        manifest = RunManifest(
            run_manifest_id=row['run_manifest_id'],
            run_date=row['run_date'],
            created_at=row['created_at'],
            git=git,
            settings_hash=row['settings_hash'],
            settings_snapshot=row['settings_snapshot'],
            feature_set_id="unknown",  # Not stored directly, need to extract from manifest_data
            dataset=dataset,
            environment=environment,
            results=results
        )

        logger.info(
            "manifest_loaded",
            manifest_id=manifest_id,
            run_date=manifest.run_date
        )

        return manifest

    finally:
        conn.close()


def verify_manifest(
    manifest: RunManifest,
    strict: bool = False
) -> tuple[bool, list[str]]:
    """
    Verify manifest completeness and consistency

    Args:
        manifest: RunManifest to verify
        strict: If True, warnings become errors

    Returns:
        (is_valid, issues)
    """
    issues = []

    # Check git dirty status
    if manifest.git.dirty:
        issues.append("Git repository was dirty at training time")

    # Check for unknown git state
    if manifest.git.commit == "unknown":
        issues.append("Git commit unknown - cannot reproduce")

    # Check feature set ID
    if not manifest.feature_set_id or manifest.feature_set_id == "unknown":
        issues.append("Feature set ID missing")

    # Check dataset checksums
    if not manifest.dataset.checksums:
        issues.append("Dataset checksums missing")

    # Check for missing package versions
    if not manifest.environment.package_versions:
        issues.append("Package versions not recorded")

    is_valid = len(issues) == 0 or not strict

    if issues:
        logger.warning(
            "manifest_verification_issues",
            manifest_id=manifest.run_manifest_id,
            issues=issues,
            strict=strict
        )

    return is_valid, issues


def get_recent_manifests(
    dsn: str,
    limit: int = 10,
    run_date: Optional[date] = None
) -> list[dict]:
    """
    Get recent run manifests

    Args:
        dsn: PostgreSQL connection string
        limit: Max number of manifests to return
        run_date: Optional filter by run date

    Returns:
        List of manifest summary dicts
    """
    conn = psycopg2.connect(dsn)

    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            if run_date:
                cur.execute(
                    """
                    SELECT
                        run_manifest_id, run_date, created_at,
                        git_commit, git_branch, git_dirty,
                        models_trained, models_published, models_shadowed, models_rejected
                    FROM run_manifests
                    WHERE run_date = %s
                    ORDER BY created_at DESC
                    LIMIT %s
                    """,
                    (run_date, limit)
                )
            else:
                cur.execute(
                    """
                    SELECT
                        run_manifest_id, run_date, created_at,
                        git_commit, git_branch, git_dirty,
                        models_trained, models_published, models_shadowed, models_rejected
                    FROM run_manifests
                    ORDER BY created_at DESC
                    LIMIT %s
                    """,
                    (limit,)
                )

            rows = cur.fetchall()

        return [dict(row) for row in rows]

    finally:
        conn.close()
