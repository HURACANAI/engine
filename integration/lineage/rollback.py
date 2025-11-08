"""
Model Rollback Manager

Enables safe rollback to previous model versions.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class RollbackRecord:
    """Record of a model rollback"""
    rollback_id: str
    timestamp: datetime

    # Models
    from_model_id: str
    to_model_id: str
    symbol: str

    # Context
    reason: str
    triggered_by: str  # "operator" or "automated"

    # Performance (if available)
    from_model_live_sharpe: Optional[float] = None
    to_model_live_sharpe: Optional[float] = None

    # Verification
    verified: bool = False
    verification_time: Optional[datetime] = None


class RollbackManager:
    """
    Model Rollback Manager

    Enables safe rollback to previous model versions with validation.

    Example:
        manager = RollbackManager(
            registry_dsn="postgresql://...",
            model_dir="/models"
        )

        # Rollback to previous version
        rollback_id = manager.rollback(
            current_model_id="btc_v48",
            target_model_id="btc_v47",
            reason="Live Sharpe degraded from 1.5 to 0.8",
            triggered_by="operator"
        )

        # Verify rollback succeeded
        manager.verify_rollback(rollback_id)

        # Get rollback history
        history = manager.get_rollback_history(symbol="BTC", limit=10)
    """

    def __init__(
        self,
        registry_dsn: Optional[str] = None,
        model_dir: Optional[Path] = None
    ):
        """
        Initialize rollback manager

        Args:
            registry_dsn: PostgreSQL DSN for model registry
            model_dir: Directory containing model artifacts
        """
        self.registry_dsn = registry_dsn
        self.model_dir = Path(model_dir) if model_dir else None

        self.rollback_history: list[RollbackRecord] = []

    def rollback(
        self,
        current_model_id: str,
        target_model_id: str,
        symbol: str,
        reason: str,
        triggered_by: str = "operator",
        from_model_live_sharpe: Optional[float] = None,
        to_model_live_sharpe: Optional[float] = None
    ) -> str:
        """
        Rollback to previous model version

        Args:
            current_model_id: Current (problematic) model
            target_model_id: Target (safe) model to rollback to
            symbol: Trading symbol
            reason: Reason for rollback
            triggered_by: "operator" or "automated"
            from_model_live_sharpe: Live Sharpe of current model
            to_model_live_sharpe: Historical Sharpe of target model

        Returns:
            rollback_id

        Raises:
            ValueError: If target model not found
        """
        logger.warning(
            "initiating_model_rollback",
            current_model_id=current_model_id,
            target_model_id=target_model_id,
            symbol=symbol,
            reason=reason
        )

        # Validate target model exists
        if not self._validate_model_exists(target_model_id):
            raise ValueError(
                f"Target model {target_model_id} not found. "
                f"Cannot rollback."
            )

        # Check if target model artifacts exist
        if self.model_dir:
            target_path = self.model_dir / f"{target_model_id}.pkl"
            if not target_path.exists():
                raise ValueError(
                    f"Target model artifacts not found at {target_path}. "
                    f"Cannot rollback."
                )

        # Create rollback record
        rollback_id = f"rollback_{symbol}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        record = RollbackRecord(
            rollback_id=rollback_id,
            timestamp=datetime.utcnow(),
            from_model_id=current_model_id,
            to_model_id=target_model_id,
            symbol=symbol,
            reason=reason,
            triggered_by=triggered_by,
            from_model_live_sharpe=from_model_live_sharpe,
            to_model_live_sharpe=to_model_live_sharpe,
            verified=False
        )

        self.rollback_history.append(record)

        # Update registry to mark target as active
        if self.registry_dsn:
            self._update_active_model_in_registry(symbol, target_model_id)

        logger.info(
            "model_rollback_complete",
            rollback_id=rollback_id,
            current_model_id=current_model_id,
            target_model_id=target_model_id
        )

        return rollback_id

    def verify_rollback(
        self,
        rollback_id: str,
        verification_passed: bool = True
    ) -> None:
        """
        Verify rollback succeeded

        Args:
            rollback_id: Rollback to verify
            verification_passed: Whether verification passed

        Raises:
            ValueError: If rollback_id not found
        """
        # Find rollback record
        record = None
        for r in self.rollback_history:
            if r.rollback_id == rollback_id:
                record = r
                break

        if record is None:
            raise ValueError(f"Rollback {rollback_id} not found")

        record.verified = verification_passed
        record.verification_time = datetime.utcnow()

        logger.info(
            "rollback_verified",
            rollback_id=rollback_id,
            passed=verification_passed,
            to_model_id=record.to_model_id
        )

        if not verification_passed:
            logger.error(
                "rollback_verification_failed",
                rollback_id=rollback_id,
                to_model_id=record.to_model_id
            )

    def get_rollback_history(
        self,
        symbol: Optional[str] = None,
        limit: int = 100
    ) -> list[RollbackRecord]:
        """
        Get rollback history

        Args:
            symbol: Filter by symbol (None = all)
            limit: Maximum records to return

        Returns:
            List of RollbackRecord (newest first)
        """
        records = self.rollback_history

        if symbol is not None:
            records = [r for r in records if r.symbol == symbol]

        # Sort by timestamp (newest first)
        records.sort(key=lambda r: r.timestamp, reverse=True)

        return records[:limit]

    def can_rollback_to(
        self,
        target_model_id: str,
        min_sharpe: float = 0.5
    ) -> bool:
        """
        Check if safe to rollback to target model

        Args:
            target_model_id: Model to potentially rollback to
            min_sharpe: Minimum acceptable Sharpe ratio

        Returns:
            True if safe to rollback
        """
        # Check if model exists
        if not self._validate_model_exists(target_model_id):
            logger.warning(
                "rollback_target_not_found",
                target_model_id=target_model_id
            )
            return False

        # Check artifacts exist
        if self.model_dir:
            target_path = self.model_dir / f"{target_model_id}.pkl"
            if not target_path.exists():
                logger.warning(
                    "rollback_target_artifacts_missing",
                    target_model_id=target_model_id,
                    path=str(target_path)
                )
                return False

        # Get model performance from registry
        if self.registry_dsn:
            sharpe = self._get_model_sharpe_from_registry(target_model_id)
            if sharpe is not None and sharpe < min_sharpe:
                logger.warning(
                    "rollback_target_low_sharpe",
                    target_model_id=target_model_id,
                    sharpe=sharpe,
                    min_sharpe=min_sharpe
                )
                return False

        return True

    def suggest_rollback_target(
        self,
        current_model_id: str,
        symbol: str,
        max_versions_back: int = 5
    ) -> Optional[str]:
        """
        Suggest best model to rollback to

        Args:
            current_model_id: Current problematic model
            symbol: Trading symbol
            max_versions_back: Maximum versions to look back

        Returns:
            Suggested model_id to rollback to (None if no good option)
        """
        # Get lineage
        from .tracker import LineageTracker

        tracker = LineageTracker(registry_dsn=self.registry_dsn)
        ancestors = tracker.get_ancestors(current_model_id, max_depth=max_versions_back)

        if len(ancestors) == 0:
            logger.warning(
                "no_rollback_candidates",
                current_model_id=current_model_id
            )
            return None

        # Find best ancestor by Sharpe
        best_model_id = None
        best_sharpe = float('-inf')

        for ancestor_id in ancestors:
            if not self.can_rollback_to(ancestor_id):
                continue

            sharpe = self._get_model_sharpe_from_registry(ancestor_id)
            if sharpe is not None and sharpe > best_sharpe:
                best_sharpe = sharpe
                best_model_id = ancestor_id

        if best_model_id:
            logger.info(
                "rollback_target_suggested",
                current_model_id=current_model_id,
                suggested_model_id=best_model_id,
                sharpe=best_sharpe
            )

        return best_model_id

    def generate_rollback_report(
        self,
        symbol: Optional[str] = None,
        days_back: int = 30
    ) -> str:
        """
        Generate rollback history report

        Args:
            symbol: Filter by symbol
            days_back: Report on last N days

        Returns:
            Human-readable report
        """
        cutoff = datetime.utcnow() - pd.Timedelta(days=days_back)

        records = [
            r for r in self.rollback_history
            if r.timestamp >= cutoff
        ]

        if symbol is not None:
            records = [r for r in records if r.symbol == symbol]

        if len(records) == 0:
            return f"No rollbacks in last {days_back} days"

        lines = [
            "=" * 80,
            "ROLLBACK HISTORY REPORT",
            "=" * 80,
            "",
            f"Period: Last {days_back} days",
            f"Total Rollbacks: {len(records)}",
            ""
        ]

        for record in records:
            status = "✅ Verified" if record.verified else "⏳ Pending"

            lines.append(
                f"[{record.timestamp.strftime('%Y-%m-%d %H:%M')}] "
                f"{record.symbol}: {record.from_model_id} → {record.to_model_id}"
            )
            lines.append(f"  Reason: {record.reason}")
            lines.append(f"  Triggered By: {record.triggered_by}")
            lines.append(f"  Status: {status}")

            if record.from_model_live_sharpe is not None:
                lines.append(
                    f"  Performance: {record.from_model_live_sharpe:.2f} → "
                    f"{record.to_model_live_sharpe:.2f} (Sharpe)"
                )

            lines.append("")

        lines.append("=" * 80)

        return "\n".join(lines)

    def _validate_model_exists(self, model_id: str) -> bool:
        """Check if model exists in registry"""
        if self.registry_dsn:
            # TODO: Query registry database
            return True
        return True

    def _update_active_model_in_registry(
        self,
        symbol: str,
        model_id: str
    ) -> None:
        """Update active model in registry"""
        # TODO: Update registry database
        logger.info(
            "registry_active_model_updated",
            symbol=symbol,
            model_id=model_id
        )

    def _get_model_sharpe_from_registry(
        self,
        model_id: str
    ) -> Optional[float]:
        """Get model Sharpe ratio from registry"""
        # TODO: Query registry database
        return None


# Import pandas for Timedelta
import pandas as pd
