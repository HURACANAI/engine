"""
Signal Data Contract

Schema for trading signals from models.
"""

from datetime import datetime
from typing import Literal, Dict, Any

import pandas as pd
import polars as pl
from pydantic import BaseModel, Field, field_validator

from .validator import DataContractValidator, validate_with_schema


class SignalSchema(BaseModel):
    """
    Trading Signal Schema

    Represents a trading signal from a model.

    Fields:
        timestamp: Signal generation time (UTC)
        symbol: Trading symbol
        direction: Trade direction (long, short, neutral)
        confidence: Model confidence [0-1]
        stop_loss_bps: Stop loss in basis points
        take_profit_bps: Take profit in basis points
        model_id: Model identifier
        regime: Market regime
        metadata: Optional metadata
    """

    timestamp: datetime = Field(..., description="Signal timestamp (UTC)")
    symbol: str = Field(..., description="Trading symbol", min_length=1, max_length=20)
    direction: Literal["long", "short", "neutral"] = Field(..., description="Trade direction")
    confidence: float = Field(..., description="Model confidence", ge=0, le=1)
    stop_loss_bps: float = Field(..., description="Stop loss (bps)", gt=0)
    take_profit_bps: float = Field(..., description="Take profit (bps)", gt=0)
    model_id: str = Field(..., description="Model identifier")
    regime: str | None = Field(default=None, description="Market regime")
    metadata: Dict[str, Any] | None = Field(default=None, description="Optional metadata")

    @field_validator("confidence")
    @classmethod
    def confidence_must_be_valid(cls, v):
        """Confidence must be between 0 and 1"""
        if not (0 <= v <= 1):
            raise ValueError("confidence must be between 0 and 1")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "timestamp": "2025-11-08T12:00:00Z",
                "symbol": "BTC",
                "direction": "long",
                "confidence": 0.75,
                "stop_loss_bps": 50,
                "take_profit_bps": 20,
                "model_id": "btc_trend_v47",
                "regime": "trending",
                "metadata": {"engine": "liquidity_regime"}
            }
        }


# Export Signal as a standalone class for easy use
Signal = SignalSchema


def validate_signals(
    df: pd.DataFrame | pl.DataFrame,
    fail_on_error: bool = True,
) -> pd.DataFrame | pl.DataFrame:
    """
    Validate signal data

    Args:
        df: DataFrame with signal data
        fail_on_error: If True, raise ValidationError on failure

    Returns:
        Validated DataFrame

    Raises:
        ValidationError: If validation fails and fail_on_error=True
    """
    return validate_with_schema(df, SignalSchema, context="signals")
