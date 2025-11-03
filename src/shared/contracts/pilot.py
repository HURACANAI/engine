"""Pilot contract schema for daily Engine outputs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Dict


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
        payload = asdict(self)
        payload["date"] = self.date.isoformat()
        return json.dumps(payload, separators=(",", ":"))
