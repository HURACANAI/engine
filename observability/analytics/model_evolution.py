"""
Model Evolution Tracker

Tracks how models improve over time and determines readiness for Hamilton export.

This answers:
- "How has the model improved since yesterday?"
- "Is the model ready for Hamilton to use?"
- "Which features became more/less important?"
- "Is the model regressing?"

Usage:
    tracker = ModelEvolutionTracker()

    # Compare two models
    comparison = tracker.compare_models(
        old_model_id="sha256:abc...",
        new_model_id="sha256:def..."
    )

    # Check if ready for Hamilton
    ready = tracker.is_ready_for_hamilton(model_id="sha256:def...")

    # Get improvement report
    report = tracker.get_improvement_report(days=7)
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import structlog
from dataclasses import dataclass

from observability.core.registry import ModelRegistry

logger = structlog.get_logger(__name__)


@dataclass
class ModelComparison:
    """Comparison between two models"""
    old_model_id: str
    new_model_id: str

    # Metric improvements
    delta_auc: float
    delta_ece: float
    delta_brier: float

    # Feature importance changes
    top_features_old: Dict[str, float]
    top_features_new: Dict[str, float]
    feature_importance_shift: float  # How much did importance change?

    # Summary
    improved: bool
    regression: bool
    recommendation: str


@dataclass
class HamiltonReadiness:
    """Model readiness assessment for Hamilton export"""
    model_id: str
    ready: bool

    # Criteria checks
    auc_threshold_met: bool  # AUC >= 0.65
    ece_threshold_met: bool  # ECE <= 0.10
    sufficient_data: bool  # Trained on >= 1000 samples
    no_regression: bool  # Not worse than previous model

    # Scores
    auc: float
    ece: float
    brier: float

    # Recommendation
    recommendation: str
    blockers: List[str]  # What's preventing export?


class ModelEvolutionTracker:
    """
    Track model evolution over time.

    Determines:
    - Model improvements (AUC, ECE, Brier)
    - Feature importance changes
    - Readiness for Hamilton export
    """

    def __init__(
        self,
        registry: Optional[ModelRegistry] = None,
        min_auc: float = 0.65,
        max_ece: float = 0.10,
        min_samples: int = 1000
    ):
        """
        Initialize model evolution tracker.

        Args:
            registry: Model registry (or create new one)
            min_auc: Minimum AUC for Hamilton export
            max_ece: Maximum ECE for Hamilton export
            min_samples: Minimum training samples required
        """
        self.registry = registry or ModelRegistry()

        self.min_auc = min_auc
        self.max_ece = max_ece
        self.min_samples = min_samples

        logger.info(
            "model_evolution_tracker_initialized",
            min_auc=min_auc,
            max_ece=max_ece,
            min_samples=min_samples
        )

    def compare_models(
        self,
        old_model_id: str,
        new_model_id: str
    ) -> ModelComparison:
        """
        Compare two models.

        Args:
            old_model_id: Previous model SHA256 ID
            new_model_id: New model SHA256 ID

        Returns:
            ModelComparison with detailed diff
        """
        # Get model metrics from registry
        old_metrics = self._get_model_metrics(old_model_id)
        new_metrics = self._get_model_metrics(new_model_id)

        if not old_metrics or not new_metrics:
            raise ValueError(f"Model not found: {old_model_id if not old_metrics else new_model_id}")

        # Compute deltas
        delta_auc = new_metrics['auc'] - old_metrics['auc']
        delta_ece = old_metrics['ece'] - new_metrics['ece']  # Lower is better
        delta_brier = old_metrics['brier'] - new_metrics['brier']  # Lower is better

        # Check if improved
        improved = delta_auc > 0 or delta_ece > 0
        regression = delta_auc < -0.01  # >1% AUC drop is regression

        # Feature importance (placeholder - would load from learning tracker)
        top_features_old = {}
        top_features_new = {}
        feature_importance_shift = 0.0

        # Recommendation
        if regression:
            recommendation = "⚠️ MODEL REGRESSION - Do not export to Hamilton. Investigate why AUC dropped."
        elif improved and delta_auc >= 0.01:
            recommendation = "✅ SIGNIFICANT IMPROVEMENT - Ready for Hamilton export."
        elif improved:
            recommendation = "✓ Minor improvement - Can export to Hamilton."
        else:
            recommendation = "→ No change - Keep current Hamilton model."

        logger.info(
            "models_compared",
            old_model=old_model_id[:20],
            new_model=new_model_id[:20],
            delta_auc=delta_auc,
            improved=improved,
            regression=regression
        )

        return ModelComparison(
            old_model_id=old_model_id,
            new_model_id=new_model_id,
            delta_auc=delta_auc,
            delta_ece=delta_ece,
            delta_brier=delta_brier,
            top_features_old=top_features_old,
            top_features_new=top_features_new,
            feature_importance_shift=feature_importance_shift,
            improved=improved,
            regression=regression,
            recommendation=recommendation
        )

    def is_ready_for_hamilton(self, model_id: str) -> HamiltonReadiness:
        """
        Check if model is ready for Hamilton export.

        Criteria:
        1. AUC >= 0.65 (minimum acceptable)
        2. ECE <= 0.10 (well-calibrated)
        3. Trained on >= 1000 samples
        4. Not worse than previous model

        Args:
            model_id: Model SHA256 ID

        Returns:
            HamiltonReadiness with detailed assessment
        """
        metrics = self._get_model_metrics(model_id)
        if not metrics:
            raise ValueError(f"Model not found: {model_id}")

        auc = metrics['auc']
        ece = metrics['ece']
        brier = metrics['brier']
        n_samples = metrics.get('n_samples', 0)

        # Check criteria
        auc_threshold_met = auc >= self.min_auc
        ece_threshold_met = ece <= self.max_ece
        sufficient_data = n_samples >= self.min_samples

        # Check for regression (compare to previous model)
        no_regression = True
        previous_model_id = self._get_previous_model_id(model_id)
        if previous_model_id:
            prev_metrics = self._get_model_metrics(previous_model_id)
            if prev_metrics and auc < prev_metrics['auc'] - 0.01:
                no_regression = False

        # Overall readiness
        ready = auc_threshold_met and ece_threshold_met and sufficient_data and no_regression

        # Identify blockers
        blockers = []
        if not auc_threshold_met:
            blockers.append(f"AUC {auc:.3f} < minimum {self.min_auc:.3f}")
        if not ece_threshold_met:
            blockers.append(f"ECE {ece:.3f} > maximum {self.max_ece:.3f}")
        if not sufficient_data:
            blockers.append(f"Only {n_samples} samples (need {self.min_samples})")
        if not no_regression:
            blockers.append("Model regressed from previous version")

        # Recommendation
        if ready:
            recommendation = f"✅ READY FOR HAMILTON - AUC {auc:.3f}, ECE {ece:.3f}"
        else:
            recommendation = f"❌ NOT READY - Blockers: {'; '.join(blockers)}"

        logger.info(
            "hamilton_readiness_checked",
            model_id=model_id[:20],
            ready=ready,
            auc=auc,
            ece=ece,
            blockers=len(blockers)
        )

        return HamiltonReadiness(
            model_id=model_id,
            ready=ready,
            auc_threshold_met=auc_threshold_met,
            ece_threshold_met=ece_threshold_met,
            sufficient_data=sufficient_data,
            no_regression=no_regression,
            auc=auc,
            ece=ece,
            brier=brier,
            recommendation=recommendation,
            blockers=blockers
        )

    def get_improvement_report(self, days: int = 7) -> Dict[str, Any]:
        """
        Get improvement report over last N days.

        Args:
            days: Number of days to look back

        Returns:
            Dict with improvement trends
        """
        cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()

        # Get all models from registry in date range
        models = self._get_models_since(cutoff)

        if len(models) < 2:
            return {
                "period_days": days,
                "models_trained": len(models),
                "message": "Need at least 2 models to compare"
            }

        # First and last model in period
        first_model = models[0]
        last_model = models[-1]

        # Compute improvement
        delta_auc = last_model['auc'] - first_model['auc']
        delta_ece = first_model['ece'] - last_model['ece']  # Lower is better

        # Trend
        if delta_auc > 0.02:
            trend = "📈 Strong improvement"
        elif delta_auc > 0:
            trend = "↗️ Improving"
        elif delta_auc < -0.02:
            trend = "📉 Regressing"
        else:
            trend = "→ Stable"

        return {
            "period_days": days,
            "models_trained": len(models),
            "first_model": {
                "model_id": first_model['model_id'],
                "date": first_model['created_at'],
                "auc": first_model['auc'],
                "ece": first_model['ece']
            },
            "last_model": {
                "model_id": last_model['model_id'],
                "date": last_model['created_at'],
                "auc": last_model['auc'],
                "ece": last_model['ece']
            },
            "improvement": {
                "delta_auc": delta_auc,
                "delta_ece": delta_ece,
                "pct_improvement_auc": (delta_auc / first_model['auc'] * 100) if first_model['auc'] > 0 else 0
            },
            "trend": trend,
            "ready_for_hamilton": last_model['auc'] >= self.min_auc and last_model['ece'] <= self.max_ece
        }

    def _get_model_metrics(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get metrics for a model from registry"""
        self.registry.conn.row_factory = None  # Reset row factory
        row = self.registry.conn.execute("""
            SELECT model_id, created_at, auc, ece, brier, wr, n_samples
            FROM models
            WHERE model_id = ?
        """, (model_id,)).fetchone()

        if row:
            return {
                "model_id": row[0],
                "created_at": row[1],
                "auc": row[2],
                "ece": row[3],
                "brier": row[4],
                "wr": row[5],
                "n_samples": row[6]
            }
        return None

    def _get_previous_model_id(self, model_id: str) -> Optional[str]:
        """Get the model that came before this one"""
        current = self._get_model_metrics(model_id)
        if not current:
            return None

        # Get previous model (by date)
        self.registry.conn.row_factory = None
        row = self.registry.conn.execute("""
            SELECT model_id
            FROM models
            WHERE created_at < ?
            ORDER BY created_at DESC
            LIMIT 1
        """, (current['created_at'],)).fetchone()

        return row[0] if row else None

    def _get_models_since(self, cutoff: str) -> List[Dict[str, Any]]:
        """Get all models since cutoff date"""
        self.registry.conn.row_factory = None
        rows = self.registry.conn.execute("""
            SELECT model_id, created_at, auc, ece, brier, wr, n_samples
            FROM models
            WHERE created_at >= ?
            ORDER BY created_at ASC
        """, (cutoff,)).fetchall()

        return [{
            "model_id": row[0],
            "created_at": row[1],
            "auc": row[2],
            "ece": row[3],
            "brier": row[4],
            "wr": row[5],
            "n_samples": row[6]
        } for row in rows]


if __name__ == '__main__':
    # Example usage
    print("Model Evolution Tracker Example")
    print("=" * 80)

    tracker = ModelEvolutionTracker()

    # Register test models
    from observability.core.registry import ModelRegistry
    import pickle

    registry = ModelRegistry(base_path="observability/data/test_models")

    class DummyModel:
        def __init__(self, threshold=0.5):
            self.threshold = threshold

    # Model v1
    model_v1 = DummyModel(threshold=0.45)
    model_id_v1 = registry.register_model(
        model=model_v1,
        code_git_sha="abc123",
        data_snapshot_id="snapshot_v1",
        metrics={"auc": 0.70, "ece": 0.080, "brier": 0.19, "wr": 0.72},
        notes="Initial model"
    )
    print(f"\n✓ Model v1 registered: {model_id_v1[:20]}...")
    print(f"  AUC: 0.70, ECE: 0.080")

    # Model v2 (improved)
    model_v2 = DummyModel(threshold=0.42)
    model_id_v2 = registry.register_model(
        model=model_v2,
        code_git_sha="def456",
        data_snapshot_id="snapshot_v2",
        metrics={"auc": 0.74, "ece": 0.055, "brier": 0.17, "wr": 0.75},
        notes="Improved model with more training data"
    )
    print(f"\n✓ Model v2 registered: {model_id_v2[:20]}...")
    print(f"  AUC: 0.74, ECE: 0.055")

    # Compare models
    print("\n📊 Comparing models...")
    tracker_with_registry = ModelEvolutionTracker(registry=registry)
    comparison = tracker_with_registry.compare_models(model_id_v1, model_id_v2)

    print(f"\n{comparison.recommendation}")
    print(f"  ΔAUC: {comparison.delta_auc:+.3f} ({comparison.delta_auc/0.70*100:+.1f}%)")
    print(f"  ΔECE: {comparison.delta_ece:+.3f} (lower is better)")
    print(f"  Improved: {comparison.improved}")
    print(f"  Regression: {comparison.regression}")

    # Check Hamilton readiness
    print("\n🎯 Checking Hamilton readiness...")
    readiness = tracker_with_registry.is_ready_for_hamilton(model_id_v2)

    print(f"\n{readiness.recommendation}")
    print(f"  Ready: {readiness.ready}")
    print(f"  AUC threshold ({tracker_with_registry.min_auc:.2f}): {'✓' if readiness.auc_threshold_met else '✗'}")
    print(f"  ECE threshold ({tracker_with_registry.max_ece:.2f}): {'✓' if readiness.ece_threshold_met else '✗'}")
    print(f"  Sufficient data: {'✓' if readiness.sufficient_data else '✗'}")
    print(f"  No regression: {'✓' if readiness.no_regression else '✗'}")

    if readiness.blockers:
        print(f"\n  Blockers:")
        for blocker in readiness.blockers:
            print(f"    • {blocker}")

    # Improvement report
    print("\n📈 Improvement report (last 7 days)...")
    report = tracker_with_registry.get_improvement_report(days=7)

    print(f"  Models trained: {report['models_trained']}")
    print(f"  Trend: {report['trend']}")
    if report['models_trained'] >= 2:
        print(f"  ΔAUC: {report['improvement']['delta_auc']:+.3f} ({report['improvement']['pct_improvement_auc']:+.1f}%)")
        print(f"  Ready for Hamilton: {report['ready_for_hamilton']}")

    print("\n✓ Model evolution tracker ready!")
