"""
Run Manifest Generator

Generates comprehensive manifests for full reproducibility.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass, asdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import psycopg2
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class GitInfo:
    """Git repository state"""
    commit: str
    branch: str
    dirty: bool
    remote_url: Optional[str] = None
    commit_message: Optional[str] = None
    commit_author: Optional[str] = None
    commit_date: Optional[str] = None


@dataclass
class EnvironmentInfo:
    """Environment information"""
    python_version: str
    platform: str
    platform_version: str
    package_versions: Dict[str, str]
    ray_cluster_info: Optional[Dict[str, Any]] = None


@dataclass
class DatasetInfo:
    """Dataset information"""
    symbols: List[str]
    checksums: Dict[str, str]  # {symbol: sha256}
    total_candles: Optional[int] = None
    date_range: Optional[Dict[str, str]] = None  # {start, end}


@dataclass
class TrainingResults:
    """Training run results"""
    models_trained: int
    models_published: int
    models_shadowed: int
    models_rejected: int
    total_runtime_seconds: Optional[float] = None
    errors_encountered: int = 0


@dataclass
class RunManifest:
    """
    Complete run manifest for reproducibility

    Contains everything needed to reproduce a training run.
    """
    # Identity
    run_manifest_id: str
    run_date: date
    created_at: datetime

    # Git state
    git: GitInfo

    # Configuration
    settings_hash: str  # SHA256 of settings yaml
    settings_snapshot: Dict[str, Any]  # Full settings

    # Features
    feature_set_id: str

    # Data
    dataset: DatasetInfo

    # Environment
    environment: EnvironmentInfo

    # Results
    results: Optional[TrainingResults] = None

    # Schema version
    schema_version: str = "1.0.0"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        d = asdict(self)
        # Convert date/datetime to ISO strings
        d['run_date'] = self.run_date.isoformat()
        d['created_at'] = self.created_at.isoformat()
        return d

    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)


class RunManifestGenerator:
    """
    Run Manifest Generator

    Generates comprehensive manifests for training runs.

    Example:
        generator = RunManifestGenerator()

        manifest = generator.generate(
            run_date=date.today(),
            settings=settings,
            feature_set_id="fs_abc123",
            dataset_checksums={"BTC": "sha256..."},
            symbols=["BTC", "ETH"]
        )

        # Save to database
        manifest_id = generator.save(manifest, dsn="postgresql://...")
    """

    def __init__(self, repo_path: Optional[Path] = None):
        """
        Initialize generator

        Args:
            repo_path: Path to git repository (default: current directory)
        """
        self.repo_path = repo_path or Path.cwd()

    def _get_git_info(self) -> GitInfo:
        """
        Get current git state

        Returns:
            GitInfo with commit, branch, dirty status
        """
        try:
            # Get commit hash
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=self.repo_path,
                stderr=subprocess.DEVNULL
            ).decode().strip()

            # Get branch name
            branch = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=self.repo_path,
                stderr=subprocess.DEVNULL
            ).decode().strip()

            # Check if dirty
            status = subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=self.repo_path,
                stderr=subprocess.DEVNULL
            ).decode().strip()
            dirty = len(status) > 0

            # Get remote URL (optional)
            try:
                remote_url = subprocess.check_output(
                    ["git", "config", "--get", "remote.origin.url"],
                    cwd=self.repo_path,
                    stderr=subprocess.DEVNULL
                ).decode().strip()
            except subprocess.CalledProcessError:
                remote_url = None

            # Get commit message
            try:
                commit_message = subprocess.check_output(
                    ["git", "log", "-1", "--pretty=%B"],
                    cwd=self.repo_path,
                    stderr=subprocess.DEVNULL
                ).decode().strip()
            except subprocess.CalledProcessError:
                commit_message = None

            # Get commit author
            try:
                commit_author = subprocess.check_output(
                    ["git", "log", "-1", "--pretty=%an <%ae>"],
                    cwd=self.repo_path,
                    stderr=subprocess.DEVNULL
                ).decode().strip()
            except subprocess.CalledProcessError:
                commit_author = None

            # Get commit date
            try:
                commit_date = subprocess.check_output(
                    ["git", "log", "-1", "--pretty=%aI"],
                    cwd=self.repo_path,
                    stderr=subprocess.DEVNULL
                ).decode().strip()
            except subprocess.CalledProcessError:
                commit_date = None

            return GitInfo(
                commit=commit,
                branch=branch,
                dirty=dirty,
                remote_url=remote_url,
                commit_message=commit_message,
                commit_author=commit_author,
                commit_date=commit_date
            )

        except subprocess.CalledProcessError as e:
            logger.warning("git_info_failed", error=str(e))
            return GitInfo(
                commit="unknown",
                branch="unknown",
                dirty=False
            )

    def _get_settings_hash(self, settings: Any) -> str:
        """
        Compute SHA256 hash of settings

        Args:
            settings: Settings object

        Returns:
            SHA256 hash string
        """
        # Convert settings to dict if needed
        if hasattr(settings, 'model_dump'):
            settings_dict = settings.model_dump()
        elif hasattr(settings, 'dict'):
            settings_dict = settings.dict()
        else:
            settings_dict = vars(settings)

        # Sort keys for deterministic hash
        settings_json = json.dumps(settings_dict, sort_keys=True)
        return hashlib.sha256(settings_json.encode()).hexdigest()

    def _get_environment_info(
        self,
        ray_cluster_info: Optional[Dict[str, Any]] = None
    ) -> EnvironmentInfo:
        """
        Get environment information

        Args:
            ray_cluster_info: Optional Ray cluster info

        Returns:
            EnvironmentInfo
        """
        import importlib.metadata

        # Get Python version
        python_version = sys.version.split()[0]

        # Get platform info
        platform_name = platform.system()
        platform_version = platform.release()

        # Get key package versions
        key_packages = [
            'polars', 'pandas', 'numpy', 'lightgbm', 'ray',
            'psycopg2', 'pydantic', 'structlog', 'ccxt'
        ]

        package_versions = {}
        for pkg in key_packages:
            try:
                version = importlib.metadata.version(pkg)
                package_versions[pkg] = version
            except importlib.metadata.PackageNotFoundError:
                package_versions[pkg] = "not_installed"

        return EnvironmentInfo(
            python_version=python_version,
            platform=platform_name,
            platform_version=platform_version,
            package_versions=package_versions,
            ray_cluster_info=ray_cluster_info
        )

    def generate(
        self,
        run_date: date,
        settings: Any,
        feature_set_id: str,
        dataset_checksums: Dict[str, str],
        symbols: List[str],
        ray_cluster_info: Optional[Dict[str, Any]] = None,
        results: Optional[TrainingResults] = None,
    ) -> RunManifest:
        """
        Generate a run manifest

        Args:
            run_date: Training run date
            settings: Settings object
            feature_set_id: Feature set identifier
            dataset_checksums: Dict of {symbol: sha256_checksum}
            symbols: List of symbols
            ray_cluster_info: Optional Ray cluster info
            results: Optional training results (can be updated later)

        Returns:
            RunManifest
        """
        # Generate manifest ID
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        manifest_id = f"run_{run_date.strftime('%Y%m%d')}_{timestamp}"

        # Get git info
        git = self._get_git_info()

        # Get settings hash and snapshot
        settings_hash = self._get_settings_hash(settings)
        if hasattr(settings, 'model_dump'):
            settings_snapshot = settings.model_dump()
        elif hasattr(settings, 'dict'):
            settings_snapshot = settings.dict()
        else:
            settings_snapshot = vars(settings)

        # Dataset info
        dataset = DatasetInfo(
            symbols=symbols,
            checksums=dataset_checksums
        )

        # Environment info
        environment = self._get_environment_info(ray_cluster_info)

        manifest = RunManifest(
            run_manifest_id=manifest_id,
            run_date=run_date,
            created_at=datetime.now(timezone.utc),
            git=git,
            settings_hash=settings_hash,
            settings_snapshot=settings_snapshot,
            feature_set_id=feature_set_id,
            dataset=dataset,
            environment=environment,
            results=results
        )

        logger.info(
            "manifest_generated",
            manifest_id=manifest_id,
            git_commit=git.commit[:8],
            git_dirty=git.dirty,
            feature_set_id=feature_set_id,
            num_symbols=len(symbols)
        )

        return manifest

    def save(
        self,
        manifest: RunManifest,
        dsn: str
    ) -> str:
        """
        Save manifest to database

        Args:
            manifest: RunManifest to save
            dsn: PostgreSQL connection string

        Returns:
            run_manifest_id
        """
        conn = psycopg2.connect(dsn)

        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO run_manifests (
                        run_manifest_id, run_date,
                        git_commit, git_branch, git_dirty,
                        settings_hash, settings_snapshot,
                        dataset_checksums, symbol_list,
                        python_version, package_versions,
                        ray_cluster_info,
                        models_trained, models_published, models_shadowed, models_rejected,
                        manifest_data
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )
                    ON CONFLICT (run_manifest_id) DO UPDATE
                    SET models_trained = EXCLUDED.models_trained,
                        models_published = EXCLUDED.models_published,
                        models_shadowed = EXCLUDED.models_shadowed,
                        models_rejected = EXCLUDED.models_rejected,
                        manifest_data = EXCLUDED.manifest_data
                    """,
                    (
                        manifest.run_manifest_id,
                        manifest.run_date,
                        manifest.git.commit,
                        manifest.git.branch,
                        manifest.git.dirty,
                        manifest.settings_hash,
                        json.dumps(manifest.settings_snapshot),
                        json.dumps(manifest.dataset.checksums),
                        manifest.dataset.symbols,
                        manifest.environment.python_version,
                        json.dumps(manifest.environment.package_versions),
                        json.dumps(manifest.environment.ray_cluster_info) if manifest.environment.ray_cluster_info else None,
                        manifest.results.models_trained if manifest.results else 0,
                        manifest.results.models_published if manifest.results else 0,
                        manifest.results.models_shadowed if manifest.results else 0,
                        manifest.results.models_rejected if manifest.results else 0,
                        json.dumps(manifest.to_dict())
                    )
                )
            conn.commit()

            logger.info(
                "manifest_saved",
                manifest_id=manifest.run_manifest_id
            )

            return manifest.run_manifest_id

        finally:
            conn.close()

    def update_results(
        self,
        manifest_id: str,
        results: TrainingResults,
        dsn: str
    ) -> None:
        """
        Update manifest with training results

        Args:
            manifest_id: Run manifest ID
            results: TrainingResults
            dsn: PostgreSQL connection string
        """
        conn = psycopg2.connect(dsn)

        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE run_manifests
                    SET models_trained = %s,
                        models_published = %s,
                        models_shadowed = %s,
                        models_rejected = %s
                    WHERE run_manifest_id = %s
                    """,
                    (
                        results.models_trained,
                        results.models_published,
                        results.models_shadowed,
                        results.models_rejected,
                        manifest_id
                    )
                )
            conn.commit()

            logger.info(
                "manifest_results_updated",
                manifest_id=manifest_id,
                trained=results.models_trained,
                published=results.models_published
            )

        finally:
            conn.close()

    def verify_environment(self, manifest: RunManifest) -> tuple[bool, List[str]]:
        """
        Verify current environment matches manifest

        Args:
            manifest: RunManifest to verify against

        Returns:
            (matches, warnings)
        """
        warnings = []

        # Check Python version
        current_python = sys.version.split()[0]
        if current_python != manifest.environment.python_version:
            warnings.append(
                f"Python version mismatch: {current_python} vs {manifest.environment.python_version}"
            )

        # Check key package versions
        import importlib.metadata
        for pkg, expected_version in manifest.environment.package_versions.items():
            try:
                current_version = importlib.metadata.version(pkg)
                if current_version != expected_version:
                    warnings.append(
                        f"{pkg} version mismatch: {current_version} vs {expected_version}"
                    )
            except importlib.metadata.PackageNotFoundError:
                warnings.append(f"{pkg} not installed (expected {expected_version})")

        matches = len(warnings) == 0

        if not matches:
            logger.warning(
                "environment_mismatch",
                manifest_id=manifest.run_manifest_id,
                warnings=warnings
            )

        return matches, warnings
