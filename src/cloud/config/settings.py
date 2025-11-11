"""
Compatibility shim for legacy imports.

The training package holds the canonical EngineSettings definition, but older
code (and several tests) import it from ``src.cloud.config.settings``. This
module re-exports the training config so that existing import paths continue to
work while the codebase stabilises.
"""

from src.cloud.training.config.settings import EngineSettings

__all__ = ["EngineSettings"]
