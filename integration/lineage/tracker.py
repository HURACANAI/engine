"""
Model Lineage Tracker

Tracks model ancestry, changes, and evolution over time.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

import structlog

logger = structlog.get_logger(__name__)


class ChangeType(str, Enum):
    """Type of change made to create new model version"""
    INITIAL = "initial"  # First model in lineage
    HYPERPARAMETER_TUNING = "hyperparameter_tuning"
    ARCHITECTURE_CHANGE = "architecture_change"
    FEATURE_UPDATE = "feature_update"
    DATA_EXPANSION = "data_expansion"
    TRAINING_METHOD = "training_method"
    CALIBRATION = "calibration"
    BUGFIX = "bugfix"
    EXPERIMENT = "experiment"


@dataclass
class LineageNode:
    """
    Single node in model lineage tree

    Represents one version of a model and its relationship to parent.
    """
    lineage_id: str
    model_id: str
    parent_model_id: Optional[str]
    symbol: str
    created_at: datetime

    change_type: ChangeType
    changes: Dict  # What changed from parent
    reason: str  # Why this change was made

    # Performance comparison
    parent_sharpe: Optional[float] = None
    current_sharpe: Optional[float] = None
    sharpe_improvement: Optional[float] = None

    # Metadata
    created_by: Optional[str] = None
    tags: Optional[List[str]] = None


class LineageTracker:
    """
    Model Lineage Tracker

    Tracks model evolution, ancestry, and enables lineage queries.

    Example:
        tracker = LineageTracker(registry_dsn="postgresql://...")

        # Track new model version
        tracker.track_model(
            model_id="btc_v48",
            parent_model_id="btc_v47",
            change_type=ChangeType.HYPERPARAMETER_TUNING,
            changes={"learning_rate": {"old": 0.001, "new": 0.0005}},
            reason="Better convergence on recent data"
        )

        # Get lineage
        tree = tracker.get_lineage_tree("btc_v48")
        ancestors = tracker.get_ancestors("btc_v48")
        children = tracker.get_children("btc_v47")

        # Compare performance
        comparison = tracker.compare_with_parent("btc_v48")
    """

    def __init__(self, registry_dsn: Optional[str] = None):
        """
        Initialize lineage tracker

        Args:
            registry_dsn: PostgreSQL DSN for model registry
        """
        self.registry_dsn = registry_dsn
        self._lineage: Dict[str, LineageNode] = {}

        # If DSN provided, load from database
        if registry_dsn:
            self._load_from_registry()

    def track_model(
        self,
        model_id: str,
        symbol: str,
        parent_model_id: Optional[str] = None,
        change_type: ChangeType = ChangeType.INITIAL,
        changes: Optional[Dict] = None,
        reason: str = "",
        parent_sharpe: Optional[float] = None,
        current_sharpe: Optional[float] = None,
        created_by: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> str:
        """
        Track new model in lineage

        Args:
            model_id: New model identifier
            symbol: Trading symbol
            parent_model_id: Parent model (None if initial)
            change_type: Type of change
            changes: Dict describing what changed
            reason: Why this change was made
            parent_sharpe: Parent model Sharpe ratio
            current_sharpe: Current model Sharpe ratio
            created_by: Who created this model
            tags: Optional tags

        Returns:
            lineage_id
        """
        # Validate parent exists (if specified)
        if parent_model_id is not None:
            if parent_model_id not in self._lineage:
                logger.warning(
                    "parent_model_not_in_lineage",
                    model_id=model_id,
                    parent_model_id=parent_model_id
                )

        # Calculate improvement
        sharpe_improvement = None
        if parent_sharpe is not None and current_sharpe is not None:
            sharpe_improvement = current_sharpe - parent_sharpe

        # Create lineage node
        lineage_id = f"lineage_{model_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        node = LineageNode(
            lineage_id=lineage_id,
            model_id=model_id,
            parent_model_id=parent_model_id,
            symbol=symbol,
            created_at=datetime.utcnow(),
            change_type=change_type,
            changes=changes or {},
            reason=reason,
            parent_sharpe=parent_sharpe,
            current_sharpe=current_sharpe,
            sharpe_improvement=sharpe_improvement,
            created_by=created_by,
            tags=tags or []
        )

        self._lineage[model_id] = node

        logger.info(
            "model_lineage_tracked",
            model_id=model_id,
            parent_model_id=parent_model_id,
            change_type=change_type.value,
            sharpe_improvement=sharpe_improvement
        )

        # Persist to registry if available
        if self.registry_dsn:
            self._save_to_registry(node)

        return lineage_id

    def get_lineage_tree(
        self,
        model_id: str,
        max_depth: int = 10
    ) -> List[LineageNode]:
        """
        Get full lineage tree (ancestors) for a model

        Args:
            model_id: Model to get lineage for
            max_depth: Maximum depth to traverse

        Returns:
            List of LineageNode from root to current model
        """
        if model_id not in self._lineage:
            logger.warning("model_not_found_in_lineage", model_id=model_id)
            return []

        tree = []
        current_id = model_id
        depth = 0

        while current_id is not None and depth < max_depth:
            if current_id not in self._lineage:
                break

            node = self._lineage[current_id]
            tree.append(node)

            current_id = node.parent_model_id
            depth += 1

        # Reverse to get root -> current order
        tree.reverse()

        return tree

    def get_ancestors(
        self,
        model_id: str,
        max_depth: int = 10
    ) -> List[str]:
        """
        Get ancestor model IDs

        Args:
            model_id: Model to get ancestors for
            max_depth: Maximum depth

        Returns:
            List of ancestor model IDs (nearest to oldest)
        """
        tree = self.get_lineage_tree(model_id, max_depth)

        # Remove the current model itself
        ancestors = [node.model_id for node in tree if node.model_id != model_id]

        return ancestors

    def get_children(self, model_id: str) -> List[str]:
        """
        Get direct children (models derived from this one)

        Args:
            model_id: Model to get children for

        Returns:
            List of child model IDs
        """
        children = [
            node.model_id
            for node in self._lineage.values()
            if node.parent_model_id == model_id
        ]

        return children

    def get_siblings(self, model_id: str) -> List[str]:
        """
        Get siblings (models with same parent)

        Args:
            model_id: Model to get siblings for

        Returns:
            List of sibling model IDs
        """
        if model_id not in self._lineage:
            return []

        node = self._lineage[model_id]

        if node.parent_model_id is None:
            return []

        # Get all children of parent (excluding self)
        siblings = [
            child_id for child_id in self.get_children(node.parent_model_id)
            if child_id != model_id
        ]

        return siblings

    def compare_with_parent(self, model_id: str) -> Optional[Dict]:
        """
        Compare model performance with parent

        Args:
            model_id: Model to compare

        Returns:
            Dict with comparison (None if no parent)
        """
        if model_id not in self._lineage:
            return None

        node = self._lineage[model_id]

        if node.parent_model_id is None:
            return None

        return {
            "model_id": model_id,
            "parent_model_id": node.parent_model_id,
            "change_type": node.change_type.value,
            "changes": node.changes,
            "reason": node.reason,
            "parent_sharpe": node.parent_sharpe,
            "current_sharpe": node.current_sharpe,
            "sharpe_improvement": node.sharpe_improvement,
            "improvement_pct": (
                (node.sharpe_improvement / node.parent_sharpe * 100)
                if node.parent_sharpe and node.sharpe_improvement
                else None
            )
        }

    def find_best_in_lineage(
        self,
        model_id: str,
        metric: str = "sharpe"
    ) -> Optional[str]:
        """
        Find best performing model in lineage tree

        Args:
            model_id: Model to search lineage for
            metric: Metric to optimize (currently only "sharpe")

        Returns:
            Model ID of best performer (None if no data)
        """
        tree = self.get_lineage_tree(model_id)

        if len(tree) == 0:
            return None

        # Find best Sharpe
        best_node = None
        best_sharpe = float('-inf')

        for node in tree:
            if node.current_sharpe is not None:
                if node.current_sharpe > best_sharpe:
                    best_sharpe = node.current_sharpe
                    best_node = node

        return best_node.model_id if best_node else None

    def get_lineage_summary(self, model_id: str) -> str:
        """
        Generate human-readable lineage summary

        Args:
            model_id: Model to summarize

        Returns:
            Summary string
        """
        tree = self.get_lineage_tree(model_id)

        if len(tree) == 0:
            return f"No lineage found for {model_id}"

        lines = [
            "=" * 80,
            f"LINEAGE SUMMARY: {model_id}",
            "=" * 80,
            "",
            f"Total Versions: {len(tree)}",
            ""
        ]

        for i, node in enumerate(tree):
            # Format improvement
            if node.sharpe_improvement is not None:
                improvement = f" ({node.sharpe_improvement:+.2f})"
            else:
                improvement = ""

            lines.append(
                f"[{i+1}] {node.model_id} "
                f"({node.change_type.value}){improvement}"
            )

            if node.reason:
                lines.append(f"    Reason: {node.reason}")

            if node.changes:
                lines.append(f"    Changes: {node.changes}")

            lines.append("")

        # Best model in lineage
        best_model_id = self.find_best_in_lineage(model_id)
        if best_model_id:
            lines.append(f"Best Performer: {best_model_id}")

        lines.append("=" * 80)

        return "\n".join(lines)

    def _load_from_registry(self) -> None:
        """Load lineage from model registry database"""
        # TODO: Implement database loading
        # This would query the model_lineage table in PostgreSQL
        logger.info("lineage_load_from_registry", dsn=self.registry_dsn)

    def _save_to_registry(self, node: LineageNode) -> None:
        """Save lineage node to model registry database"""
        # TODO: Implement database persistence
        # This would insert into model_lineage table
        logger.info(
            "lineage_save_to_registry",
            model_id=node.model_id,
            lineage_id=node.lineage_id
        )
