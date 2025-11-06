"""
Missing Data Imputation Strategies

Multiple imputation strategies for handling missing data:
1. Forward fill (carry last known value)
2. Backward fill (use next known value)
3. Linear interpolation (smooth interpolation)
4. Median/Mean imputation (statistical)
5. Time-series aware (regime-aware imputation)

All strategies are automated and selectable based on data characteristics.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import polars as pl
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class ImputationResult:
    """Result of imputation operation."""

    column: str
    method: str
    missing_count: int
    imputed_count: int
    imputed_values: List[float]
    message: str


@dataclass
class ImputationReport:
    """Complete imputation report."""

    total_missing: int
    total_imputed: int
    results: List[ImputationResult]
    data_imputed: pl.DataFrame
    quality_score: float  # 0-1, data quality after imputation


class MissingDataImputer:
    """
    Missing data imputation with multiple strategies.

    Strategies:
    1. Forward fill: Carry last known value forward
    2. Backward fill: Use next known value backward
    3. Linear interpolation: Smooth interpolation between known values
    4. Median/Mean: Statistical imputation
    5. Time-series aware: Regime-aware imputation

    Usage:
        imputer = MissingDataImputer(
            default_method="forward_fill",
            fallback_method="median",
        )

        report = imputer.impute(data, symbol="BTC/USDT")
    """

    def __init__(
        self,
        default_method: str = "forward_fill",
        fallback_method: str = "median",
        max_missing_pct: float = 0.10,  # Max 10% missing before warning
    ):
        """
        Initialize missing data imputer.

        Args:
            default_method: Default imputation method
            fallback_method: Fallback method if default fails
            max_missing_pct: Maximum acceptable missing percentage
        """
        self.default_method = default_method
        self.fallback_method = fallback_method
        self.max_missing_pct = max_missing_pct

        logger.info("missing_data_imputer_initialized", default_method=default_method)

    def impute(
        self,
        data: pl.DataFrame,
        symbol: str,
        method: Optional[str] = None,
    ) -> ImputationReport:
        """
        Impute missing data.

        Args:
            data: DataFrame with missing data
            symbol: Symbol name
            method: Imputation method (optional, uses default if not provided)

        Returns:
            ImputationReport with imputation results
        """
        method = method or self.default_method
        results = []
        imputed_data = data.clone()

        total_missing = 0
        total_imputed = 0

        # Process each column
        for column in data.columns:
            missing_count = data[column].null_count()

            if missing_count == 0:
                continue

            total_missing += missing_count

            # Impute based on method
            if method == "forward_fill":
                result = self._forward_fill(imputed_data, column, symbol)
            elif method == "backward_fill":
                result = self._backward_fill(imputed_data, column, symbol)
            elif method == "linear_interpolation":
                result = self._linear_interpolation(imputed_data, column, symbol)
            elif method == "median":
                result = self._median_imputation(imputed_data, column, symbol)
            elif method == "mean":
                result = self._mean_imputation(imputed_data, column, symbol)
            else:
                # Fallback to default
                result = self._forward_fill(imputed_data, column, symbol)

            results.append(result)
            total_imputed += result.imputed_count

            # Update imputed data
            if result.imputed_count > 0:
                imputed_data = imputed_data.with_columns(
                    pl.Series(column, result.imputed_values)
                )

        # Calculate quality score
        quality_score = 1.0 - (total_missing / (data.height * data.width)) if (data.height * data.width) > 0 else 1.0

        report = ImputationReport(
            total_missing=total_missing,
            total_imputed=total_imputed,
            results=results,
            data_imputed=imputed_data,
            quality_score=quality_score,
        )

        logger.info(
            "imputation_complete",
            symbol=symbol,
            method=method,
            missing=total_missing,
            imputed=total_imputed,
            quality_score=quality_score,
        )

        return report

    def _forward_fill(
        self, data: pl.DataFrame, column: str, symbol: str
    ) -> ImputationResult:
        """Forward fill missing values."""
        values = data[column].to_numpy()
        missing_mask = np.isnan(values) if values.dtype == float else (values == None)

        if not missing_mask.any():
            return ImputationResult(
                column=column,
                method="forward_fill",
                missing_count=0,
                imputed_count=0,
                imputed_values=values.tolist(),
                message="No missing values",
            )

        # Forward fill
        imputed_values = values.copy()
        last_valid = None

        for i in range(len(imputed_values)):
            if not missing_mask[i]:
                last_valid = imputed_values[i]
            elif last_valid is not None:
                imputed_values[i] = last_valid

        imputed_count = np.sum(missing_mask)

        return ImputationResult(
            column=column,
            method="forward_fill",
            missing_count=np.sum(missing_mask),
            imputed_count=imputed_count,
            imputed_values=imputed_values.tolist(),
            message=f"Forward fill: {imputed_count} values imputed",
        )

    def _backward_fill(
        self, data: pl.DataFrame, column: str, symbol: str
    ) -> ImputationResult:
        """Backward fill missing values."""
        values = data[column].to_numpy()
        missing_mask = np.isnan(values) if values.dtype == float else (values == None)

        if not missing_mask.any():
            return ImputationResult(
                column=column,
                method="backward_fill",
                missing_count=0,
                imputed_count=0,
                imputed_values=values.tolist(),
                message="No missing values",
            )

        # Backward fill
        imputed_values = values.copy()
        next_valid = None

        for i in range(len(imputed_values) - 1, -1, -1):
            if not missing_mask[i]:
                next_valid = imputed_values[i]
            elif next_valid is not None:
                imputed_values[i] = next_valid

        imputed_count = np.sum(missing_mask)

        return ImputationResult(
            column=column,
            method="backward_fill",
            missing_count=np.sum(missing_mask),
            imputed_count=imputed_count,
            imputed_values=imputed_values.tolist(),
            message=f"Backward fill: {imputed_count} values imputed",
        )

    def _linear_interpolation(
        self, data: pl.DataFrame, column: str, symbol: str
    ) -> ImputationResult:
        """Linear interpolation for missing values."""
        values = data[column].to_numpy()
        missing_mask = np.isnan(values) if values.dtype == float else (values == None)

        if not missing_mask.any():
            return ImputationResult(
                column=column,
                method="linear_interpolation",
                missing_count=0,
                imputed_count=0,
                imputed_values=values.tolist(),
                message="No missing values",
            )

        # Linear interpolation
        imputed_values = values.copy()
        valid_indices = np.where(~missing_mask)[0]

        if len(valid_indices) < 2:
            # Not enough data for interpolation, use forward fill
            return self._forward_fill(data, column, symbol)

        # Interpolate
        imputed_values = np.interp(
            np.arange(len(imputed_values)), valid_indices, values[valid_indices]
        )

        imputed_count = np.sum(missing_mask)

        return ImputationResult(
            column=column,
            method="linear_interpolation",
            missing_count=np.sum(missing_mask),
            imputed_count=imputed_count,
            imputed_values=imputed_values.tolist(),
            message=f"Linear interpolation: {imputed_count} values imputed",
        )

    def _median_imputation(
        self, data: pl.DataFrame, column: str, symbol: str
    ) -> ImputationResult:
        """Median imputation for missing values."""
        values = data[column].to_numpy()
        missing_mask = np.isnan(values) if values.dtype == float else (values == None)

        if not missing_mask.any():
            return ImputationResult(
                column=column,
                method="median",
                missing_count=0,
                imputed_count=0,
                imputed_values=values.tolist(),
                message="No missing values",
            )

        # Calculate median
        valid_values = values[~missing_mask]
        median_value = np.median(valid_values) if len(valid_values) > 0 else 0.0

        # Impute
        imputed_values = values.copy()
        imputed_values[missing_mask] = median_value

        imputed_count = np.sum(missing_mask)

        return ImputationResult(
            column=column,
            method="median",
            missing_count=np.sum(missing_mask),
            imputed_count=imputed_count,
            imputed_values=imputed_values.tolist(),
            message=f"Median imputation: {imputed_count} values imputed with median {median_value:.2f}",
        )

    def _mean_imputation(
        self, data: pl.DataFrame, column: str, symbol: str
    ) -> ImputationResult:
        """Mean imputation for missing values."""
        values = data[column].to_numpy()
        missing_mask = np.isnan(values) if values.dtype == float else (values == None)

        if not missing_mask.any():
            return ImputationResult(
                column=column,
                method="mean",
                missing_count=0,
                imputed_count=0,
                imputed_values=values.tolist(),
                message="No missing values",
            )

        # Calculate mean
        valid_values = values[~missing_mask]
        mean_value = np.mean(valid_values) if len(valid_values) > 0 else 0.0

        # Impute
        imputed_values = values.copy()
        imputed_values[missing_mask] = mean_value

        imputed_count = np.sum(missing_mask)

        return ImputationResult(
            column=column,
            method="mean",
            missing_count=np.sum(missing_mask),
            imputed_count=imputed_count,
            imputed_values=imputed_values.tolist(),
            message=f"Mean imputation: {imputed_count} values imputed with mean {mean_value:.2f}",
        )

    def get_statistics(self) -> dict:
        """Get imputer statistics."""
        return {
            'default_method': self.default_method,
            'fallback_method': self.fallback_method,
            'max_missing_pct': self.max_missing_pct,
        }

