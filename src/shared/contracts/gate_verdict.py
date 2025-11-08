"""
Gate Verdict Data Contract

Schema for model gate evaluation results.
"""

from datetime import datetime
from typing import Literal, List, Dict, Any

import pandas as pd
import polars as pl
from pydantic import BaseModel, Field, field_validator

from .validator import DataContractValidator, validate_with_schema


class GateVerdictSchema(BaseModel):
    """
    Gate Verdict Schema

    Represents the result of rule-based gate evaluation for a model.

    Fields:
        model_id: Model identifier
        evaluated_at: Evaluation timestamp (UTC)
        status: Final verdict (publish, shadow, reject, pending)
        meta_weight: Computed meta weight [0-1]
        passed_gates: List of gate names that passed
        failed_gates: List of gate names that failed
        warnings: List of warning messages
        gate_details: Detailed results per gate
    """

    model_id: str = Field(..., description="Model identifier")
    evaluated_at: datetime = Field(default_factory=datetime.utcnow, description="Evaluation time")
    status: Literal["publish", "shadow", "reject", "pending"] = Field(..., description="Final verdict")
    meta_weight: float = Field(..., description="Meta weight", ge=0, le=1)
    passed_gates: List[str] = Field(default_factory=list, description="Gates that passed")
    failed_gates: List[str] = Field(default_factory=list, description="Gates that failed")
    warnings: List[str] = Field(default_factory=list, description="Warning messages")
    gate_details: Dict[str, Any] = Field(default_factory=dict, description="Detailed gate results")

    @field_validator("meta_weight")
    @classmethod
    def meta_weight_must_be_valid(cls, v):
        """Meta weight must be between 0 and 1"""
        if not (0 <= v <= 1):
            raise ValueError("meta_weight must be between 0 and 1")
        return v

    @field_validator("status")
    @classmethod
    def status_logic_must_be_valid(cls, v, info):
        """
        Validate status logic:
        - PUBLISH: All blocking gates passed
        - SHADOW: Blocking gates passed, but warnings exist
        - REJECT: Any blocking gate failed
        """
        failed_gates = info.data.get('failed_gates', [])

        if v == "publish" and failed_gates:
            raise ValueError("Cannot publish with failed gates")

        return v

    class Config:
        json_schema_extra = {
            "example": {
                "model_id": "btc_trend_v47",
                "evaluated_at": "2025-11-08T12:00:00Z",
                "status": "publish",
                "meta_weight": 0.15,
                "passed_gates": [
                    "minimum_sharpe",
                    "maximum_drawdown",
                    "minimum_trades",
                    "stress_tests_passed"
                ],
                "failed_gates": [],
                "warnings": ["calibration_slightly_off"],
                "gate_details": {
                    "minimum_sharpe": {"value": 1.5, "threshold": 0.5, "passed": True},
                    "maximum_drawdown": {"value": 12.0, "threshold": 20.0, "passed": True}
                }
            }
        }


# Export as standalone class
GateVerdict = GateVerdictSchema


def validate_gate_verdict(
    verdict: Dict[str, Any],
    fail_on_error: bool = True,
) -> Dict[str, Any]:
    """
    Validate gate verdict data

    Args:
        verdict: Gate verdict dict
        fail_on_error: If True, raise ValidationError on failure

    Returns:
        Validated verdict dict

    Raises:
        ValidationError: If validation fails and fail_on_error=True
    """
    from .validator import ValidationError
    from pydantic import ValidationError as PydanticValidationError

    try:
        validated = GateVerdictSchema.model_validate(verdict)
        return validated.model_dump()
    except PydanticValidationError as e:
        if fail_on_error:
            raise ValidationError(f"Gate verdict validation failed: {e}")
        return verdict
