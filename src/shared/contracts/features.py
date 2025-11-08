"""
Feature Data Contract

Schema for computed feature vectors.
"""

from datetime import datetime
from typing import Dict, Any, List

import pandas as pd
import polars as pl
from pydantic import BaseModel, Field

from .validator import DataContractValidator, validate_with_schema


class FeatureSchema(BaseModel):
    """
    Feature Vector Schema

    Represents computed features for a single timestamp/symbol.

    Fields:
        timestamp: Feature timestamp (UTC)
        symbol: Trading symbol
        features: Dict of feature values {feature_name: value}
        feature_set_id: Feature set identifier (for reproducibility)
        metadata: Optional metadata dict
    """

    timestamp: datetime = Field(..., description="Feature timestamp (UTC)")
    symbol: str = Field(..., description="Trading symbol", min_length=1, max_length=20)
    features: Dict[str, float] = Field(..., description="Feature values")
    feature_set_id: str = Field(..., description="Feature set ID", min_length=1)
    metadata: Dict[str, Any] | None = Field(default=None, description="Optional metadata")

    class Config:
        json_schema_extra = {
            "example": {
                "timestamp": "2025-11-08T12:00:00Z",
                "symbol": "BTC",
                "features": {
                    "rsi_14": 65.5,
                    "macd": 0.023,
                    "bb_width": 0.015,
                    "volume_ma_ratio": 1.25
                },
                "feature_set_id": "fs_20251108_120000_abc123",
                "metadata": {"regime": "trending"}
            }
        }


def validate_features(
    df: pd.DataFrame | pl.DataFrame,
    fail_on_error: bool = True,
) -> pd.DataFrame | pl.DataFrame:
    """
    Validate feature data

    Args:
        df: DataFrame with feature data
        fail_on_error: If True, raise ValidationError on failure

    Returns:
        Validated DataFrame

    Raises:
        ValidationError: If validation fails and fail_on_error=True
    """
    return validate_with_schema(df, FeatureSchema, context="features")
