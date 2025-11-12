"""Pilot contract schema for daily Engine outputs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Any, Dict


@dataclass
class PilotContract:
    date: date
    symbol: str
    recommended_edge_threshold_bps: int
    taker_extra_buffer_bps: int
    max_trades_per_day_hint: int
    cooldown_seconds_hint: int
    costs_breakdown: Dict[str, float]
    validation_summary: Dict[str, float | int | str]
    gate_flags: Dict[str, bool]
    notes: str | None = None

    def to_json(self) -> str:
        """Convert to JSON, handling numpy types and ensuring all values are JSON-serializable."""
        import numpy as np
        
        def convert_to_json_serializable(obj: Any) -> Any:
            """Recursively convert numpy types and other non-serializable types to JSON-serializable types."""
            if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
                val = float(obj)
                # Replace NaN and Inf with None (JSON doesn't support NaN/Inf)
                if np.isnan(val) or np.isinf(val):
                    return None
                return val
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return [convert_to_json_serializable(item) for item in obj.tolist()]
            elif isinstance(obj, dict):
                return {key: convert_to_json_serializable(value) for key, value in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_json_serializable(item) for item in obj]
            elif obj is None:
                return None
            else:
                # Try to serialize, if it fails return string representation
                try:
                    json.dumps(obj, allow_nan=False)
                    return obj
                except (TypeError, ValueError):
                    return str(obj)
        
        payload = asdict(self)
        payload["date"] = self.date.isoformat()
        # Convert all numpy types to native Python types
        serializable_payload = convert_to_json_serializable(payload)
        return json.dumps(serializable_payload, separators=(",", ":"), allow_nan=False)
