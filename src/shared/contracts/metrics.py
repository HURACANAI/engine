"""Metrics schema for storing after-cost validation results."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Dict


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
        return json.dumps(asdict(self), separators=(",", ":"))
