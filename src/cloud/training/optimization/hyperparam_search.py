from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import structlog  # type: ignore[reportMissingImports]

logger = structlog.get_logger(__name__)


def score_run(
    sharpe: float,
    hit_rate: float,
    trades: int,
    max_drawdown: float,
) -> float:
    """Standard objective scalar combining key walk-forward metrics."""
    if trades <= 0:
        return -1e6
    penalty = max(0.0, -max_drawdown)
    return (sharpe * 2.0) + (hit_rate * 0.5) + (trades / 1000.0) - penalty


@dataclass
class HyperparamSearchResult:
    params: Dict[str, Any]
    rationale: str


class HyperparamSearch:
    """Heuristic hyperparameter tuner that nudges settings before training."""

    def __init__(self, model_type: str) -> None:
        self.model_type = model_type

    def suggest(
        self,
        base_params: Dict[str, Any],
        volatility_hint: float,
    ) -> HyperparamSearchResult:
        params = dict(base_params)
        rationale_parts = []

        if self.model_type.lower() in {"lightgbm", "lgbm"}:
            if volatility_hint > 0.02:
                params["reg_lambda"] = float(params.get("reg_lambda", 5.0)) + 1.0
                params["feature_fraction"] = min(1.0, float(params.get("feature_fraction", 0.8)) + 0.05)
                rationale_parts.append("raised regularization for high volatility")
            else:
                params["learning_rate"] = max(0.005, float(params.get("learning_rate", 0.01)) * 1.1)
                rationale_parts.append("slightly faster learning for calm markets")

        elif self.model_type.lower() == "xgboost":
            if volatility_hint > 0.02:
                params["reg_alpha"] = float(params.get("reg_alpha", 2.0)) + 0.5
                params["max_depth"] = max(3, int(params.get("max_depth", 6)) - 1)
                rationale_parts.append("penalized depth under high variance")
            else:
                params["subsample"] = min(1.0, float(params.get("subsample", 0.8)) + 0.05)
                rationale_parts.append("allow deeper exploration in calm regime")

        elif self.model_type.lower() == "random_forest":
            if volatility_hint > 0.02:
                params["max_depth"] = max(3, int(params.get("max_depth", 5)) - 1)
                rationale_parts.append("reduced depth for noisy regime")
            else:
                params["max_depth"] = int(params.get("max_depth", 5)) + 1
                rationale_parts.append("increased depth for richer structure")

        logger.info(
            "hyperparam_suggestion",
            model_type=self.model_type,
            params=params,
            volatility_hint=volatility_hint,
            rationale=", ".join(rationale_parts) or "base parameters retained",
        )
        return HyperparamSearchResult(params=params, rationale=", ".join(rationale_parts))


