"""Schema for model artifact manifests shared across modules."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ModelManifest:
    """Versioned manifest describing an exported RL policy."""

    model_id: str
    created_at: datetime
    symbol: str
    training_window_days: int
    metrics: Dict[str, Any]
    agent_config: Dict[str, Any]
    action_space: List[str]
    feature_count: int
    replay_buffer_size: int
    scenario_tests: Dict[str, Any] = field(default_factory=dict)
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["created_at"] = self.created_at.isoformat()
        return payload

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.write_text(json.dumps(self.to_dict(), indent=2))

