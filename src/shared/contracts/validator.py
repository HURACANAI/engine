"""
Data Contract Validator

Base validator for all data contracts with hard fail-on-mismatch logic.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Type

import pandas as pd
import polars as pl
from pydantic import BaseModel, ValidationError as PydanticValidationError
import structlog

logger = structlog.get_logger(__name__)


class ValidationError(Exception):
    """Data contract validation failed"""
    pass


class DataContractValidator:
    """
    Base validator for data contracts

    Provides validation on read and write with hard fail-on-mismatch.

    Usage:
        validator = DataContractValidator(schema=CandleSchema)

        # Validate on read
        candles = validator.validate_read(df, fail_on_error=True)

        # Validate on write
        validator.validate_write(df, path="candles.parquet")
    """

    def __init__(self, schema: Type[BaseModel]):
        """
        Initialize validator

        Args:
            schema: Pydantic schema model
        """
        self.schema = schema

    def validate_dataframe(
        self,
        df: pd.DataFrame | pl.DataFrame,
        fail_on_error: bool = True,
    ) -> tuple[bool, List[str]]:
        """
        Validate DataFrame against schema

        Args:
            df: DataFrame to validate
            fail_on_error: If True, raise ValidationError on failure

        Returns:
            (is_valid, error_messages)

        Raises:
            ValidationError: If fail_on_error=True and validation fails
        """
        errors = []

        # Convert polars to pandas for validation
        if isinstance(df, pl.DataFrame):
            df = df.to_pandas()

        # Get expected columns from schema
        expected_fields = self.schema.model_fields.keys()

        # Check for missing columns
        missing = set(expected_fields) - set(df.columns)
        if missing:
            errors.append(f"Missing columns: {missing}")

        # Check for extra columns (warning, not error)
        extra = set(df.columns) - set(expected_fields)
        if extra:
            logger.warning("extra_columns_found", columns=extra)

        # Validate a sample row
        if not errors and len(df) > 0:
            try:
                # Validate first row
                row_dict = df.iloc[0].to_dict()
                self.schema.model_validate(row_dict)
            except PydanticValidationError as e:
                errors.append(f"Schema validation failed: {e}")

        # Handle failures
        if errors:
            error_msg = "; ".join(errors)
            logger.error(
                "validation_failed",
                schema=self.schema.__name__,
                errors=errors,
                num_rows=len(df),
            )

            if fail_on_error:
                raise ValidationError(error_msg)

            return False, errors

        logger.info(
            "validation_passed",
            schema=self.schema.__name__,
            num_rows=len(df),
            num_cols=len(df.columns),
        )

        return True, []

    def validate_read(
        self,
        df: pd.DataFrame | pl.DataFrame,
        source: Optional[str] = None,
        fail_on_error: bool = True,
    ) -> pd.DataFrame | pl.DataFrame:
        """
        Validate data on read

        Args:
            df: DataFrame to validate
            source: Data source (for logging)
            fail_on_error: If True, raise on validation failure

        Returns:
            Validated DataFrame

        Raises:
            ValidationError: If validation fails and fail_on_error=True
        """
        logger.info(
            "validating_read",
            schema=self.schema.__name__,
            source=source,
            num_rows=len(df),
        )

        is_valid, errors = self.validate_dataframe(df, fail_on_error=fail_on_error)

        if not is_valid and not fail_on_error:
            logger.warning(
                "read_validation_failed_continuing",
                schema=self.schema.__name__,
                errors=errors,
            )

        return df

    def validate_write(
        self,
        df: pd.DataFrame | pl.DataFrame,
        destination: Optional[str] = None,
        fail_on_error: bool = True,
    ) -> None:
        """
        Validate data before write

        Args:
            df: DataFrame to validate
            destination: Write destination (for logging)
            fail_on_error: If True, raise on validation failure

        Raises:
            ValidationError: If validation fails and fail_on_error=True
        """
        logger.info(
            "validating_write",
            schema=self.schema.__name__,
            destination=destination,
            num_rows=len(df),
        )

        is_valid, errors = self.validate_dataframe(df, fail_on_error=fail_on_error)

        if not is_valid:
            logger.error(
                "write_validation_failed",
                schema=self.schema.__name__,
                destination=destination,
                errors=errors,
            )

            if fail_on_error:
                raise ValidationError(
                    f"Cannot write invalid data to {destination}: {'; '.join(errors)}"
                )


def validate_with_schema(
    df: pd.DataFrame | pl.DataFrame,
    schema: Type[BaseModel],
    context: str = "unknown",
) -> pd.DataFrame | pl.DataFrame:
    """
    Convenience function to validate DataFrame

    Args:
        df: DataFrame to validate
        schema: Pydantic schema
        context: Context string for logging

    Returns:
        Validated DataFrame

    Raises:
        ValidationError: If validation fails
    """
    validator = DataContractValidator(schema)
    return validator.validate_read(df, source=context)
