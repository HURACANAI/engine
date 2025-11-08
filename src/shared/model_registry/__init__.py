"""
Unified Model Registry

Single source of truth for all model versioning, tracking, and lifecycle management.
Consolidates scattered registries from services/ and ml_framework/.

Key Features:
- Version tracking with full lineage
- Feature set linking
- Run manifest integration
- Gate verdict storage
- Publish/shadow/reject status
- S3 artifact URIs
- Hamilton integration ready

Usage:
    from shared.model_registry import UnifiedModelRegistry

    registry = UnifiedModelRegistry(dsn="postgresql://...")

    # Register model
    model_id = registry.register_model(
        symbol="BTC",
        version="2025-11-08_02-00",
        feature_set_id="fs_abc123",
        run_manifest_id="run_xyz789",
        artifacts_uri="s3://models/btc_v47/"
    )

    # Update with gate verdict
    registry.update_gate_verdict(
        model_id=model_id,
        verdict={"status": "PUBLISH", "meta_weight": 0.15}
    )

    # Query publishable models
    models = registry.get_publishable_models(symbol="BTC")
"""

from .registry import UnifiedModelRegistry, ModelRecord, ModelStatus
from .schema import init_schema, migrate_schema

__all__ = [
    "UnifiedModelRegistry",
    "ModelRecord",
    "ModelStatus",
    "init_schema",
    "migrate_schema",
]
