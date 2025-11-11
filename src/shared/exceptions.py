"""
Custom Exceptions for Huracan Trading Engine

Provides specific exception types for better error handling and debugging.
"""

from __future__ import annotations

from typing import Optional, Any, Dict


class HuracanError(Exception):
    """Base exception for all Huracan errors."""

    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.context = context or {}


class DatabaseError(HuracanError):
    """Raised when database operations fail."""
    pass


class StorageError(HuracanError):
    """Raised when storage operations (Dropbox, S3, etc.) fail."""
    pass


class DropboxError(StorageError):
    """Raised when Dropbox operations fail."""
    pass


class S3Error(StorageError):
    """Raised when S3 operations fail."""
    pass


class DataError(HuracanError):
    """Raised when data operations fail."""
    pass


class DataQualityError(DataError):
    """Raised when data quality checks fail."""
    pass


class DataLoadError(DataError):
    """Raised when data loading fails."""
    pass


class ModelError(HuracanError):
    """Raised when model operations fail."""
    pass


class ModelLoadError(ModelError):
    """Raised when model loading fails."""
    pass


class ModelSaveError(ModelError):
    """Raised when model saving fails."""
    pass


class TrainingError(HuracanError):
    """Raised when training operations fail."""
    pass


class ValidationError(HuracanError):
    """Raised when validation operations fail."""
    pass


class ConfigurationError(HuracanError):
    """Raised when configuration is invalid."""
    pass


class ExchangeError(HuracanError):
    """Raised when exchange operations fail."""
    pass


class OrderError(ExchangeError):
    """Raised when order operations fail."""
    pass


class TelegramError(HuracanError):
    """Raised when Telegram operations fail."""
    pass


class FeatureError(HuracanError):
    """Raised when feature operations fail."""
    pass


class ContractError(HuracanError):
    """Raised when contract operations fail."""
    pass


class SerializationError(HuracanError):
    """Raised when serialization/deserialization fails."""
    pass
