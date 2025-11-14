"""
Execution Result Schema

Parquet schema for execution results with contract validation.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# Execution result schema for validation
EXECUTION_RESULT_SCHEMA = {
    "trade_id": "string",
    "model_id": "string",
    "symbol": "string",
    "signal_time": "datetime",
    "entry_time": "datetime",
    "exit_time": "datetime",
    "duration_seconds": "float",
    "signal_direction": "string",
    "signal_confidence": "float",
    "expected_pnl_bps": "float",
    "actual_pnl_bps": "float",
    "pnl_error_bps": "float",
    "expected_slippage_bps": "float",
    "actual_slippage_bps": "float",
    "slippage_error_bps": "float",
    "entry_fill_ratio": "float",
    "exit_fill_ratio": "float",
    "timing_delay_ms": "float",
    "regime_at_entry": "string",
    "regime_at_exit": "string",
    "volatility_at_entry": "float",
    "spread_at_entry_bps": "float",
    "total_fees_bps": "float",
}


class ExecutionResultSchema(BaseModel):
    """
    Execution Result Schema for Validation

    Used for contract validation of execution feedback data.
    """

    trade_id: str = Field(..., description="Unique trade identifier")
    model_id: str = Field(..., description="Model that generated signal")
    symbol: str = Field(..., description="Trading symbol", min_length=1, max_length=20)

    signal_time: datetime = Field(..., description="Signal generation time")
    entry_time: datetime = Field(..., description="Trade entry time")
    exit_time: datetime = Field(..., description="Trade exit time")
    duration_seconds: float = Field(..., description="Trade duration", ge=0)

    signal_direction: str = Field(..., description="Signal direction (long/short)")
    signal_confidence: float = Field(..., description="Model confidence", ge=0, le=1)

    expected_pnl_bps: float = Field(..., description="Expected PnL (bps)")
    actual_pnl_bps: float = Field(..., description="Actual PnL (bps)")
    pnl_error_bps: float = Field(..., description="PnL error (actual - expected)")

    expected_slippage_bps: float = Field(..., description="Expected slippage (bps)")
    actual_slippage_bps: float = Field(..., description="Actual slippage (bps)")
    slippage_error_bps: float = Field(..., description="Slippage error")

    entry_fill_ratio: float = Field(..., description="Entry fill ratio", ge=0, le=1)
    exit_fill_ratio: float = Field(..., description="Exit fill ratio", ge=0, le=1)
    timing_delay_ms: float = Field(..., description="Timing delay (ms)", ge=0)

    regime_at_entry: str = Field(..., description="Market regime at entry")
    regime_at_exit: str = Field(..., description="Market regime at exit")

    volatility_at_entry: Optional[float] = Field(default=None, description="Volatility")
    spread_at_entry_bps: Optional[float] = Field(default=None, description="Spread (bps)")
    total_fees_bps: Optional[float] = Field(default=None, description="Total fees (bps)")

    class Config:
        json_schema_extra = {
            "example": {
                "trade_id": "trade_xyz123",
                "model_id": "btc_trend_v47",
                "symbol": "BTC",
                "signal_time": "2025-11-08T12:00:00Z",
                "entry_time": "2025-11-08T12:01:15Z",
                "exit_time": "2025-11-08T14:30:00Z",
                "duration_seconds": 8925.0,
                "signal_direction": "long",
                "signal_confidence": 0.75,
                "expected_pnl_bps": 20.0,
                "actual_pnl_bps": 18.5,
                "pnl_error_bps": -1.5,
                "expected_slippage_bps": 2.0,
                "actual_slippage_bps": 3.5,
                "slippage_error_bps": 1.5,
                "entry_fill_ratio": 1.0,
                "exit_fill_ratio": 1.0,
                "timing_delay_ms": 75.0,
                "regime_at_entry": "trending",
                "regime_at_exit": "trending",
                "volatility_at_entry": 0.025,
                "spread_at_entry_bps": 5.0,
                "total_fees_bps": 4.0
            }
        }
