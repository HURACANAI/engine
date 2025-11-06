"""
Label Schemas - Type-Safe Labeling Configurations

Defines structured configs for different trading modes.
Using Pydantic ensures type safety and validation.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class ExitReason(Enum):
    """Why did the trade exit?"""
    TAKE_PROFIT = "tp"
    STOP_LOSS = "sl"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


class LabelConfig(BaseModel):
    """Base configuration for labeling."""

    tp_bps: float = Field(..., gt=0, description="Take profit in basis points")
    sl_bps: float = Field(..., gt=0, description="Stop loss in basis points")
    timeout_minutes: int = Field(..., gt=0, description="Max hold time in minutes")

    mode_name: str = Field(..., description="Mode identifier (scalp/runner)")

    @field_validator('tp_bps')
    @classmethod
    def validate_tp(cls, v):
        if v <= 0:
            raise ValueError("TP must be positive")
        return v

    @field_validator('sl_bps')
    @classmethod
    def validate_sl(cls, v):
        if v <= 0:
            raise ValueError("SL must be positive")
        return v


class ScalpLabelConfig(LabelConfig):
    """
    Labeling config for scalp mode.

    Philosophy:
    - Quick profits (15-30 bps)
    - Tight stops (10-15 bps)
    - Fast timeout (30-60 minutes)
    - High win rate expected (70%+)
    """

    tp_bps: float = Field(default=15.0, description="Target £1-£2 on £1000 base")
    sl_bps: float = Field(default=10.0, description="Tight stop")
    timeout_minutes: int = Field(default=30, description="Fast turnover")
    mode_name: str = "scalp"

    # Scalp-specific parameters
    scratch_threshold_bps: float = Field(
        default=-3.0,
        description="Exit immediately if < this"
    )


class RunnerLabelConfig(LabelConfig):
    """
    Labeling config for runner mode.

    Philosophy:
    - Bigger profits (80-200 bps)
    - Wider stops (40-80 bps)
    - Longer timeout (days)
    - Lower win rate acceptable (50-60%)
    """

    tp_bps: float = Field(default=80.0, description="First TP level")
    sl_bps: float = Field(default=40.0, description="Wide stop for swings")
    timeout_minutes: int = Field(default=10080, description="7 days = 10080 minutes")
    mode_name: str = "runner"

    # Runner-specific parameters
    tp_ladder: list[float] = Field(
        default=[1.0, 1.8, 2.8],
        description="TP multipliers for scale-outs"
    )
    trail_after_bps: float = Field(
        default=60.0,
        description="Start trailing after this profit"
    )


@dataclass
class LabeledTrade:
    """
    A fully labeled trade with all metadata.

    This is what gets stored in your training dataset.
    """

    # Entry info
    entry_time: datetime
    entry_price: float
    entry_idx: int  # Index in original dataframe

    # Exit info
    exit_time: datetime
    exit_price: float
    exit_reason: ExitReason

    # Performance
    duration_minutes: float
    pnl_gross_bps: float  # Before costs
    costs_bps: float      # Total transaction costs
    pnl_net_bps: float    # After costs

    # Meta-label (THE most important field)
    meta_label: int  # 1 if profitable after costs, 0 otherwise

    # Additional context
    symbol: str
    mode: str  # 'scalp' or 'runner'
    regime: Optional[str] = None  # Market regime at entry
    confidence: Optional[float] = None  # Model confidence

    def to_dict(self) -> dict:
        """Convert to dictionary for Polars/Pandas."""
        return {
            'entry_time': self.entry_time,
            'entry_price': self.entry_price,
            'entry_idx': self.entry_idx,
            'exit_time': self.exit_time,
            'exit_price': self.exit_price,
            'exit_reason': self.exit_reason.value,
            'duration_minutes': self.duration_minutes,
            'pnl_gross_bps': self.pnl_gross_bps,
            'costs_bps': self.costs_bps,
            'pnl_net_bps': self.pnl_net_bps,
            'meta_label': self.meta_label,
            'symbol': self.symbol,
            'mode': self.mode,
            'regime': self.regime,
            'confidence': self.confidence,
        }

    def is_winner(self) -> bool:
        """Was this trade profitable after costs?"""
        return self.meta_label == 1

    def is_tp_exit(self) -> bool:
        """Did this hit take-profit?"""
        return self.exit_reason == ExitReason.TAKE_PROFIT

    def is_sl_exit(self) -> bool:
        """Did this hit stop-loss?"""
        return self.exit_reason == ExitReason.STOP_LOSS

    def is_timeout_exit(self) -> bool:
        """Did this time out?"""
        return self.exit_reason == ExitReason.TIMEOUT


def create_scalp_config(**overrides) -> ScalpLabelConfig:
    """
    Create scalp label config with optional overrides.

    Usage:
        config = create_scalp_config(tp_bps=20.0, timeout_minutes=45)
    """
    return ScalpLabelConfig(**overrides)


def create_runner_config(**overrides) -> RunnerLabelConfig:
    """
    Create runner label config with optional overrides.

    Usage:
        config = create_runner_config(tp_bps=100.0, sl_bps=50.0)
    """
    return RunnerLabelConfig(**overrides)
