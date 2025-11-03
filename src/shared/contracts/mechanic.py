"""Mechanic contract schema for challenger coordination."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import date
from typing import Dict, List


@dataclass
class MechanicContract:
    date: date
    symbol: str
    edge_threshold_bps: int
    entry_exit_horizons: Dict[str, List[int] | int]
    taker_policy_hint: str
    taker_cross_buffer_bps: int
    cooldown_seconds_hint: int
    max_trades_hint: int
    costs_components: Dict[str, float]
    promotion_criteria_hint: Dict[str, float]
    confidence_model: Dict[str, float | str]

    def to_json(self) -> str:
        payload = asdict(self)
        payload["date"] = self.date.isoformat()
        return json.dumps(payload, separators=(",", ":"))
