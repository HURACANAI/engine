"""
Contracts describing Engine outputs for Pilot and Mechanic.

Data Contracts Layer:
- Schema validation for all data types
- Strong typing with Pydantic
- Automatic validation on read/write
- Sample exports for documentation

Key Contracts:
- Candles: OHLCV market data
- Features: Computed feature vectors
- Signals: Trading signals from models
- GateVerdict: Decision gate evaluation results
- ExecutionResult: Trade execution feedback

Usage:
    from shared.contracts import validate_candles, Signal

    # Validate data
    validate_candles(candles_df)

    # Use typed models
    signal = Signal(timestamp=..., symbol="BTC", direction="long", ...)
"""

# Existing contracts
from .pilot import PilotContract  # noqa: F401
from .mechanic import MechanicContract  # noqa: F401
from .metrics import MetricsPayload  # noqa: F401

# New data validation contracts (will be created)
try:
    from .candles import CandleSchema, validate_candles
    from .features import FeatureSchema, validate_features
    from .signals import SignalSchema, validate_signals, Signal
    from .gate_verdict import GateVerdictSchema, validate_gate_verdict, GateVerdict
    from .validator import DataContractValidator, ValidationError

    __all__ = [
        "PilotContract", "MechanicContract", "MetricsPayload",
        "CandleSchema", "validate_candles",
        "FeatureSchema", "validate_features",
        "SignalSchema", "validate_signals", "Signal",
        "GateVerdictSchema", "validate_gate_verdict", "GateVerdict",
        "DataContractValidator", "ValidationError",
    ]
except ImportError:
    # Validation contracts not yet installed (backward compatibility)
    __all__ = ["PilotContract", "MechanicContract", "MetricsPayload"]
