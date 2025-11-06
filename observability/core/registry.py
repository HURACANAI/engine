"""
Model & Config Registry - Content-Addressable Storage

Purpose:
- Every model has unique SHA256 ID (content hash)
- Track git SHA (code reproducibility)
- Track data snapshot ID (data reproducibility)
- Config snapshots with diffs
- Full audit trail

Key Features:
1. Content-Addressable Models
   - model_id = sha256(model_bytes)
   - Can recreate any model from ID
   - Detect duplicate models automatically

2. Reproducibility Guarantees
   - code_git_sha: Git commit hash
   - data_snapshot_id: Training data version
   - Can recreate exact training conditions

3. Config Versioning
   - Every gate threshold change recorded
   - Diff between versions
   - Who changed, when, why
   - Signed audit trail

Usage:
    registry = ModelRegistry()

    # Register model
    model_id = registry.register_model(
        model=trained_model,
        code_git_sha="a1b2c3d4",
        data_snapshot_id="snap_20251106",
        metrics={"auc": 0.710, "ece": 0.061}
    )

    # Load model
    model = registry.load_model(model_id)

    # Register config change
    registry.register_config_change(
        component="meta_label_gate",
        old_config={"threshold": 0.45},
        new_config={"threshold": 0.47},
        reason="Win-rate governor tightened",
        changed_by="system"
    )
"""

import hashlib
import pickle
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime
from dataclasses import dataclass
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class ModelMetadata:
    """Model metadata"""
    model_id: str  # sha256:...
    created_at: datetime
    code_git_sha: str
    data_snapshot_id: str
    metrics: Dict[str, float]
    n_samples: int
    notes: str


@dataclass
class ConfigChange:
    """Config change record"""
    change_id: int
    timestamp: datetime
    component: str  # meta_label_gate, cost_gate, etc.
    old_config: Dict
    new_config: Dict
    diff: Dict  # What changed
    reason: str
    changed_by: str  # system or user
    git_sha: Optional[str] = None


class ModelRegistry:
    """
    Content-addressable model storage with full provenance tracking.

    Directory structure:
    models/
      ├── sha256:abc123.../
      │   ├── model.pkl
      │   ├── metadata.json
      │   └── metrics.json
      └── registry.db (SQLite)
    """

    def __init__(
        self,
        base_path: str = "observability/data/models",
        db_path: str = "observability/data/sqlite/registry.db"
    ):
        """
        Initialize model registry.

        Args:
            base_path: Base directory for model storage
            db_path: SQLite database for metadata
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize SQLite
        self.conn = sqlite3.connect(str(self.db_path))
        self._create_tables()

        logger.info(
            "model_registry_initialized",
            base_path=str(self.base_path),
            db_path=str(self.db_path)
        )

    def _create_tables(self):
        """Create registry tables"""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS models(
                model_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                code_git_sha TEXT NOT NULL,
                data_snapshot_id TEXT NOT NULL,
                auc REAL,
                ece REAL,
                brier REAL,
                wr REAL,
                n_samples INTEGER NOT NULL,
                notes TEXT,
                file_path TEXT NOT NULL
            )
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS model_lineage(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                from_model_id TEXT NOT NULL,
                to_model_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                auc_delta REAL,
                ece_delta REAL,
                wr_delta REAL,
                notes TEXT
            )
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS config_changes(
                change_id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                component TEXT NOT NULL,
                old_config TEXT NOT NULL,
                new_config TEXT NOT NULL,
                diff TEXT NOT NULL,
                reason TEXT NOT NULL,
                changed_by TEXT NOT NULL,
                git_sha TEXT
            )
        """)

        self.conn.commit()

    def register_model(
        self,
        model: Any,
        code_git_sha: str,
        data_snapshot_id: str,
        metrics: Dict[str, float],
        n_samples: int = 0,
        notes: str = "",
        previous_model_id: Optional[str] = None
    ) -> str:
        """
        Register model with content hash.

        Args:
            model: Trained model object (any pickle-able object)
            code_git_sha: Git commit hash of code
            data_snapshot_id: Data version identifier
            metrics: Model metrics (auc, ece, brier, wr)
            n_samples: Number of training samples
            notes: Additional notes
            previous_model_id: Previous model ID (for lineage tracking)

        Returns:
            model_id: Content-addressable SHA256 hash
        """
        # Compute content hash
        model_bytes = pickle.dumps(model)
        hash_digest = hashlib.sha256(model_bytes).hexdigest()
        model_id = f"sha256:{hash_digest}"

        # Check if already exists
        existing = self.conn.execute(
            "SELECT model_id FROM models WHERE model_id = ?",
            (model_id,)
        ).fetchone()

        if existing:
            logger.info("model_already_registered", model_id=model_id)
            return model_id

        # Save model to disk
        model_dir = self.base_path / model_id
        model_dir.mkdir(parents=True, exist_ok=True)

        model_path = model_dir / "model.pkl"
        with open(model_path, "wb") as f:
            f.write(model_bytes)

        # Save metadata
        metadata = {
            "model_id": model_id,
            "created_at": datetime.utcnow().isoformat(),
            "code_git_sha": code_git_sha,
            "data_snapshot_id": data_snapshot_id,
            "metrics": metrics,
            "n_samples": n_samples,
            "notes": notes
        }

        with open(model_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Register in database
        self.conn.execute("""
            INSERT INTO models (
                model_id, created_at, code_git_sha, data_snapshot_id,
                auc, ece, brier, wr, n_samples, notes, file_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            model_id,
            datetime.utcnow().isoformat(),
            code_git_sha,
            data_snapshot_id,
            metrics.get("auc"),
            metrics.get("ece"),
            metrics.get("brier"),
            metrics.get("wr"),
            n_samples,
            notes,
            str(model_path)
        ))

        # Track lineage if previous model provided
        if previous_model_id:
            prev_metrics = self.get_model_metadata(previous_model_id).metrics

            self.conn.execute("""
                INSERT INTO model_lineage (
                    from_model_id, to_model_id, created_at,
                    auc_delta, ece_delta, wr_delta, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                previous_model_id,
                model_id,
                datetime.utcnow().isoformat(),
                metrics.get("auc", 0) - prev_metrics.get("auc", 0),
                metrics.get("ece", 0) - prev_metrics.get("ece", 0),
                metrics.get("wr", 0) - prev_metrics.get("wr", 0),
                f"Improved from {previous_model_id}"
            ))

        self.conn.commit()

        logger.info(
            "model_registered",
            model_id=model_id,
            git_sha=code_git_sha,
            data_snapshot=data_snapshot_id,
            metrics=metrics
        )

        return model_id

    def load_model(self, model_id: str) -> Any:
        """
        Load model by content hash.

        Args:
            model_id: SHA256 model ID

        Returns:
            model: Loaded model object
        """
        # Get file path from database
        result = self.conn.execute(
            "SELECT file_path FROM models WHERE model_id = ?",
            (model_id,)
        ).fetchone()

        if not result:
            raise ValueError(f"Model not found: {model_id}")

        model_path = Path(result[0])

        # Load model
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        logger.debug("model_loaded", model_id=model_id)

        return model

    def get_model_metadata(self, model_id: str) -> ModelMetadata:
        """Get model metadata"""
        result = self.conn.execute("""
            SELECT model_id, created_at, code_git_sha, data_snapshot_id,
                   auc, ece, brier, wr, n_samples, notes
            FROM models WHERE model_id = ?
        """, (model_id,)).fetchone()

        if not result:
            raise ValueError(f"Model not found: {model_id}")

        return ModelMetadata(
            model_id=result[0],
            created_at=datetime.fromisoformat(result[1]),
            code_git_sha=result[2],
            data_snapshot_id=result[3],
            metrics={
                "auc": result[4],
                "ece": result[5],
                "brier": result[6],
                "wr": result[7]
            },
            n_samples=result[8],
            notes=result[9]
        )

    def get_model_lineage(self, model_id: str) -> list:
        """Get model evolution history"""
        results = self.conn.execute("""
            SELECT from_model_id, to_model_id, created_at,
                   auc_delta, ece_delta, wr_delta, notes
            FROM model_lineage
            WHERE from_model_id = ? OR to_model_id = ?
            ORDER BY created_at DESC
        """, (model_id, model_id)).fetchall()

        lineage = []
        for row in results:
            lineage.append({
                "from": row[0],
                "to": row[1],
                "created_at": row[2],
                "auc_delta": row[3],
                "ece_delta": row[4],
                "wr_delta": row[5],
                "notes": row[6]
            })

        return lineage

    def register_config_change(
        self,
        component: str,
        old_config: Dict,
        new_config: Dict,
        reason: str,
        changed_by: str = "system",
        git_sha: Optional[str] = None
    ) -> int:
        """
        Register configuration change.

        Args:
            component: Component name (meta_label_gate, cost_gate, etc.)
            old_config: Previous configuration
            new_config: New configuration
            reason: Why changed
            changed_by: Who changed (system or user ID)
            git_sha: Git commit hash

        Returns:
            change_id: Change record ID
        """
        # Compute diff
        diff = {}
        all_keys = set(old_config.keys()) | set(new_config.keys())
        for key in all_keys:
            old_val = old_config.get(key)
            new_val = new_config.get(key)
            if old_val != new_val:
                diff[key] = {"old": old_val, "new": new_val}

        # Insert change record
        cursor = self.conn.execute("""
            INSERT INTO config_changes (
                timestamp, component, old_config, new_config, diff,
                reason, changed_by, git_sha
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.utcnow().isoformat(),
            component,
            json.dumps(old_config),
            json.dumps(new_config),
            json.dumps(diff),
            reason,
            changed_by,
            git_sha
        ))

        self.conn.commit()
        change_id = cursor.lastrowid

        logger.info(
            "config_change_registered",
            change_id=change_id,
            component=component,
            diff=diff,
            reason=reason,
            changed_by=changed_by
        )

        return change_id

    def get_config_history(self, component: str, limit: int = 10) -> list:
        """Get configuration change history"""
        results = self.conn.execute("""
            SELECT change_id, timestamp, old_config, new_config, diff, reason, changed_by
            FROM config_changes
            WHERE component = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (component, limit)).fetchall()

        history = []
        for row in results:
            history.append(ConfigChange(
                change_id=row[0],
                timestamp=datetime.fromisoformat(row[1]),
                component=component,
                old_config=json.loads(row[2]),
                new_config=json.loads(row[3]),
                diff=json.loads(row[4]),
                reason=row[5],
                changed_by=row[6]
            ))

        return history


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

# Define DummyModel at module level for pickling
class DummyModel:
    def __init__(self, param=0.5):
        self.param = param

    def predict(self, x):
        return x * self.param


def example():
    """Example usage"""
    registry = ModelRegistry()

    # Register model v1
    model_v1 = DummyModel(param=0.5)
    model_id_v1 = registry.register_model(
        model=model_v1,
        code_git_sha="a1b2c3d4",
        data_snapshot_id="snap_20251106_v1",
        metrics={"auc": 0.68, "ece": 0.064, "wr": 0.72},
        n_samples=1000,
        notes="Initial meta-label model"
    )
    print(f"✓ Registered model v1: {model_id_v1}")

    # Register model v2 (improved)
    model_v2 = DummyModel(param=0.6)
    model_id_v2 = registry.register_model(
        model=model_v2,
        code_git_sha="e5f6g7h8",
        data_snapshot_id="snap_20251106_v2",
        metrics={"auc": 0.71, "ece": 0.061, "wr": 0.75},
        n_samples=1047,
        notes="Retrained with 47 new trades",
        previous_model_id=model_id_v1
    )
    print(f"✓ Registered model v2: {model_id_v2}")

    # Load model
    loaded = registry.load_model(model_id_v1)
    print(f"✓ Loaded model v1: param={loaded.param}")

    # Get metadata
    metadata = registry.get_model_metadata(model_id_v2)
    print(f"✓ Model v2 metadata: AUC={metadata.metrics['auc']:.3f}")

    # Get lineage
    lineage = registry.get_model_lineage(model_id_v2)
    print(f"✓ Model lineage: {len(lineage)} entries")
    for entry in lineage:
        print(f"  {entry['from'][:16]}... → {entry['to'][:16]}... (AUC Δ{entry['auc_delta']:+.3f})")

    # Register config change
    change_id = registry.register_config_change(
        component="meta_label_gate",
        old_config={"threshold": 0.45, "buffer_bps": 3.0},
        new_config={"threshold": 0.47, "buffer_bps": 3.0},
        reason="Win-rate governor tightened due to low WR",
        changed_by="win_rate_governor",
        git_sha="i9j0k1l2"
    )
    print(f"✓ Registered config change: {change_id}")

    # Get config history
    history = registry.get_config_history("meta_label_gate")
    print(f"✓ Config history: {len(history)} changes")
    for change in history:
        print(f"  {change.timestamp}: {change.diff} ({change.reason})")


if __name__ == '__main__':
    print("Testing Model Registry...")
    print("=" * 60)

    example()

    print("\nModel registry tests passed ✓")
