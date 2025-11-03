"""Label computation utilities for the training engine."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from .costs import CostBreakdown


@dataclass
class LabelingConfig:
    horizon_minutes: int = 4


class LabelBuilder:
    """Produces net edge labels and confidences after costs."""

    def __init__(self, config: LabelingConfig) -> None:
        self._config = config

    def build(self, frame: pl.DataFrame, costs: CostBreakdown) -> pl.DataFrame:
        horizon = self._config.horizon_minutes
        future_close = pl.col("close").shift(-horizon)
        net_edge = ((future_close - pl.col("close")) / pl.col("close")) * 10_000 - costs.total_costs_bps
        label_expr = net_edge.alias("net_edge_bps")
        denom = max(costs.total_costs_bps, 1.0)
        confidence_expr = (1.0 / (1.0 + (-net_edge / denom).exp())).alias("edge_confidence")
        labeled = frame.with_columns([label_expr, confidence_expr])
        return labeled.drop_nulls(subset=["net_edge_bps"]).with_columns(
            pl.col("edge_confidence").fill_null(0.5)
        )