"""Metrics schema for storing after-cost validation results."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Dict, Any
import numpy as np


@dataclass
class MetricsPayload:
    symbol: str
    run_id: str
    validation_window: str
    sharpe: float
    profit_factor: float
    hit_rate_pct: float
    max_drawdown_bps: float
    trades: int
    turnover_pct: float
    pnl_bps: float
    costs: Dict[str, float]
    gate_results: Dict[str, bool]

    def to_json(self) -> str:
        """Convert to JSON, handling numpy types and ensuring all values are JSON-serializable."""
        def convert_to_json_serializable(obj: Any) -> Any:
            """Recursively convert numpy types and other non-serializable types to JSON-serializable types."""
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_to_json_serializable(value) for key, value in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_json_serializable(item) for item in obj]
            elif obj is None:
                return None
            else:
                # For other types, try to convert to string if not already serializable
                try:
                    json.dumps(obj)
                    return obj
                except (TypeError, ValueError):
                    return str(obj)
        
        data = asdict(self)
        # Convert all numpy types to native Python types
        serializable_data = convert_to_json_serializable(data)
        return json.dumps(serializable_data, separators=(",", ":"))
