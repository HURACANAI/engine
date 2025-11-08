"""
Run Manifest - Full Reproducibility Tracking

Captures everything needed to reproduce a training run:
- Git state (commit, branch, dirty status)
- Configuration (settings hash)
- Data (checksums per symbol)
- Environment (Python, packages, Ray cluster)
- Results (models trained, published, rejected)

Usage:
    from observability.run_manifest import RunManifestGenerator

    generator = RunManifestGenerator()

    # Generate manifest
    manifest = generator.generate(
        run_date=date.today(),
        settings=settings,
        feature_set_id="fs_abc123",
        dataset_checksums={"BTC": "sha256...", "ETH": "sha256..."},
        symbols=["BTC", "ETH"],
        ray_cluster_info={...}
    )

    # Save to database
    manifest_id = generator.save(manifest)

    # Later: Load and reproduce
    loaded = generator.load(manifest_id)
    # Check if environment matches
    if not generator.verify_environment(loaded):
        print("Warning: Environment mismatch!")
"""

from .generator import RunManifestGenerator, RunManifest
from .loader import load_manifest, verify_manifest
from .schema import MANIFEST_SCHEMA_VERSION

__all__ = [
    "RunManifestGenerator",
    "RunManifest",
    "load_manifest",
    "verify_manifest",
    "MANIFEST_SCHEMA_VERSION",
]
