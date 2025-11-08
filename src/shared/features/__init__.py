"""
Feature engineering primitives shared by Engine, Pilot, and Mechanic.

Feature Store with Versioning:
- Centralized feature registration and versioning
- Feature set snapshots for reproducibility
- Checksums and golden snapshot testing
- Deprecation and migration support

Usage:
    from shared.features import FeatureStore

    store = FeatureStore(dsn="postgresql://...")

    # Register features
    store.register_feature(name="rsi_14", version="v1.0", ...)

    # Create feature set
    feature_set_id = store.create_feature_set(...)
"""

# Import existing feature engineering
from .recipe import FeatureRecipe  # noqa: F401

# Import feature store (new)
try:
    from .store import FeatureStore, FeatureRecord, FeatureSetRecord
    from .schema import init_feature_schema
    __all__ = ["FeatureRecipe", "FeatureStore", "FeatureRecord", "FeatureSetRecord", "init_feature_schema"]
except ImportError:
    # Feature store not yet installed (backward compatibility)
    __all__ = ["FeatureRecipe"]
