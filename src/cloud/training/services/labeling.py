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
        
        # Check if close column exists
        if "close" not in frame.columns:
            raise ValueError(f"Expected 'close' column in frame, but got columns: {frame.columns}")
        
        # Check for NaN/null or zero values in close column
        # Use multiple methods to catch all cases
        close_stats_dict = frame.select([
            pl.col("close").is_null().sum().alias("null_count"),  # Catches both None and NaN
            pl.col("close").null_count().alias("null_count_alt"),  # Alternative method
            (pl.col("close") == 0).sum().alias("zero_count"),
            pl.col("close").count().alias("total_count"),  # Excludes nulls
            pl.col("close").len().alias("total_rows"),  # Includes nulls
        ]).to_dicts()[0]
        
        null_count = close_stats_dict["null_count"]
        null_count_alt = close_stats_dict["null_count_alt"]
        zero_count = close_stats_dict["zero_count"]
        total_count = close_stats_dict["total_count"]
        total_rows = close_stats_dict["total_rows"]
        
        # Check for nulls (includes both None and NaN in Polars)
        # If total_rows > total_count, we have nulls
        if total_rows > total_count or null_count > 0 or null_count_alt > 0:
            nulls_found = max(null_count, null_count_alt, total_rows - total_count, 0)
            raise ValueError(
                f"Found {nulls_found} null/NaN values in 'close' column "
                f"(null_count: {null_count}, null_count_alt: {null_count_alt}, "
                f"total_rows: {total_rows}, total_count: {total_count})"
            )
        
        if zero_count > 0:
            raise ValueError(f"Found {zero_count} zero values in 'close' column (division by zero risk)")
        
        # Calculate future close (shift forward by horizon)
        # Shift is negative to look forward in time
        future_close = pl.col("close").shift(-horizon)
        
        # Calculate net edge: ((future_price - current_price) / current_price) * 10000 - costs
        # This gives basis points (bps) of profit/loss after costs
        net_edge = ((future_close - pl.col("close")) / pl.col("close")) * 10_000 - costs.total_costs_bps
        label_expr = net_edge.alias("net_edge_bps")
        
        # Calculate confidence using sigmoid: 1 / (1 + exp(-net_edge / cost_threshold))
        # Higher net_edge = higher confidence
        denom = max(costs.total_costs_bps, 1.0)
        confidence_expr = (1.0 / (1.0 + (-net_edge / denom).exp())).alias("edge_confidence")
        
        # Add labels to frame
        labeled = frame.with_columns([label_expr, confidence_expr])
        
        # Drop rows where net_edge_bps is null (last 'horizon' rows due to shift)
        # This must happen BEFORE any other operations to avoid issues
        labeled = labeled.drop_nulls(subset=["net_edge_bps"])
        
        # Fill any remaining null confidence values with 0.5 (neutral)
        labeled = labeled.with_columns(
            pl.col("edge_confidence").fill_null(0.5)
        )
        
        # Verify that net_edge_bps column has valid values
        # Polars DataFrames are already materialized (not LazyFrames)
        net_edge_check = labeled.select([
            pl.col("net_edge_bps").null_count().alias("null_count"),
            pl.col("net_edge_bps").count().alias("total_count"),
        ]).row(0)
        
        if net_edge_check[0] > 0:  # If there are still nulls after drop_nulls
            raise ValueError(
                f"Found {net_edge_check[0]} null values in 'net_edge_bps' after drop_nulls. "
                f"This should not happen. Total rows: {net_edge_check[1]}"
            )
        
        # Additional verification: check that we have valid numeric values
        # Only check if we have rows (edge case: horizon > dataset size)
        if len(labeled) > 0:
            # Check a sample to ensure values are not all NaN
            sample_size = min(10, len(labeled))
            net_edge_sample = labeled.select("net_edge_bps").head(sample_size)
            sample_values = net_edge_sample["net_edge_bps"].to_list()
            
            # Check for NaN values (NaN != NaN in Python)
            valid_samples = [v for v in sample_values if v is not None and (not isinstance(v, float) or v == v)]
            if len(valid_samples) == 0 and len(sample_values) > 0:
                raise ValueError(
                    f"Sample check failed: All sampled net_edge_bps values are NaN/invalid. "
                    f"First {sample_size} values: {sample_values}"
                )
        else:
            # Dataset is empty after labeling (horizon too large for dataset)
            raise ValueError(
                f"Labeling produced empty dataset. This can happen if horizon ({horizon}) "
                f"is larger than or equal to the dataset size ({len(frame)})."
            )
        
        return labeled