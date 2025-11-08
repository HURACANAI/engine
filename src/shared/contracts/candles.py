"""
Candle Data Contract

Schema for OHLCV market data candles.
"""

from datetime import datetime
from typing import Literal

import pandas as pd
import polars as pl
from pydantic import BaseModel, Field, field_validator

from .validator import DataContractValidator, validate_with_schema


class CandleSchema(BaseModel):
    """
    OHLCV Candle Schema

    Represents a single market data candle (e.g., 5-minute, 1-hour).

    Fields:
        timestamp: Candle timestamp (UTC)
        symbol: Trading symbol (e.g., "BTC", "ETH")
        open: Opening price
        high: High price
        low: Low price
        close: Closing price
        volume: Trading volume
        timeframe: Candle timeframe (e.g., "5m", "1h")
    """

    timestamp: datetime = Field(..., description="Candle timestamp (UTC)")
    symbol: str = Field(..., description="Trading symbol", min_length=1, max_length=20)
    open: float = Field(..., description="Opening price", gt=0)
    high: float = Field(..., description="High price", gt=0)
    low: float = Field(..., description="Low price", gt=0)
    close: float = Field(..., description="Closing price", gt=0)
    volume: float = Field(..., description="Trading volume", ge=0)
    timeframe: str = Field(default="5m", description="Candle timeframe")

    @field_validator("high")
    @classmethod
    def high_must_be_highest(cls, v, info):
        """High must be >= open, close, low"""
        if 'open' in info.data and v < info.data['open']:
            raise ValueError("high must be >= open")
        if 'close' in info.data and v < info.data['close']:
            raise ValueError("high must be >= close")
        if 'low' in info.data and v < info.data['low']:
            raise ValueError("high must be >= low")
        return v

    @field_validator("low")
    @classmethod
    def low_must_be_lowest(cls, v, info):
        """Low must be <= open, close"""
        if 'open' in info.data and v > info.data['open']:
            raise ValueError("low must be <= open")
        if 'close' in info.data and v > info.data['close']:
            raise ValueError("low must be <= close")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "timestamp": "2025-11-08T12:00:00Z",
                "symbol": "BTC",
                "open": 50000.0,
                "high": 50500.0,
                "low": 49800.0,
                "close": 50200.0,
                "volume": 1250.5,
                "timeframe": "5m"
            }
        }


def validate_candles(
    df: pd.DataFrame | pl.DataFrame,
    fail_on_error: bool = True,
) -> pd.DataFrame | pl.DataFrame:
    """
    Validate candle data

    Args:
        df: DataFrame with candle data
        fail_on_error: If True, raise ValidationError on failure

    Returns:
        Validated DataFrame

    Raises:
        ValidationError: If validation fails and fail_on_error=True

    Example:
        candles_df = pd.read_parquet("candles.parquet")
        validated = validate_candles(candles_df)
    """
    return validate_with_schema(df, CandleSchema, context="candles")
