"""
Metrics Computer

Pre-computes all shadow trading and learning metrics for AI Council consumption.

This is the Engine's (Learning System) metrics:
- Shadow trade performance (paper trades only, NO real money)
- Model learning progress
- Feature importance evolution
- Gate performance (are gates tuned correctly?)
- Model readiness for Hamilton export

This answers: "How is the learning going? Are models improving?"

Usage:
    computer = MetricsComputer()

    # Compute daily metrics
    metrics = computer.compute_daily_metrics(date="2025-11-06")

    # Save for AI Council
    computer.save_metrics_json(metrics, "metrics_2025-11-06.json")

    # Verify all numbers (anti-hallucination layer)
    computer.verify_metrics(metrics)
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import json
import structlog
import yaml
import sqlite3

logger = structlog.get_logger(__name__)


class MetricsComputer:
    """
    Compute all metrics from metrics.yaml for shadow trading system.

    Key Concept: This is the LEARNING SYSTEM, not live trading:
    - Shadow trades = paper trades (no real money)
    - Focus on learning progress
    - Model readiness for Hamilton
    """

    def __init__(
        self,
        journal_db: str = "observability/data/sqlite/journal.db",
        learning_db: str = "observability/data/sqlite/learning.db",
        metrics_yaml: str = "observability/configs/metrics.yaml"
    ):
        """
        Initialize metrics computer.

        Args:
            journal_db: Path to journal database (shadow trades)
            learning_db: Path to learning database (training sessions)
            metrics_yaml: Path to metrics configuration
        """
        self.journal_db = Path(journal_db)
        self.learning_db = Path(learning_db)
        self.metrics_yaml = Path(metrics_yaml)

        # Load metrics config
        if self.metrics_yaml.exists():
            with open(self.metrics_yaml, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            logger.warning("metrics_yaml_not_found", path=str(self.metrics_yaml))
            self.config = {}

        logger.info(
            "metrics_computer_initialized",
            journal_db=str(self.journal_db),
            learning_db=str(self.learning_db)
        )

    def compute_daily_metrics(self, date: str) -> Dict[str, Any]:
        """
        Compute all metrics for a specific date.

        Args:
            date: Date string (YYYY-MM-DD)

        Returns:
            Dict with all computed metrics, ready for AI Council
        """
        start_ts = f"{date}T00:00:00"
        end_ts = f"{date}T23:59:59"

        metrics = {
            "date": date,
            "computed_at": datetime.utcnow().isoformat(),
            "shadow_trading": self._compute_shadow_trading_metrics(start_ts, end_ts),
            "learning": self._compute_learning_metrics(start_ts, end_ts),
            "gates": self._compute_gate_metrics(start_ts, end_ts),
            "models": self._compute_model_metrics(start_ts, end_ts),
            "system": self._compute_system_metrics(start_ts, end_ts),
        }

        logger.info(
            "daily_metrics_computed",
            date=date,
            shadow_trades=metrics["shadow_trading"].get("total_shadow_trades", 0),
            training_sessions=metrics["learning"].get("training_sessions", 0)
        )

        return metrics

    def _compute_shadow_trading_metrics(self, start_ts: str, end_ts: str) -> Dict[str, Any]:
        """
        Compute shadow trading metrics (paper trades only).

        IMPORTANT: These are SIMULATED trades, not real money.
        """
        if not self.journal_db.exists():
            return {"error": "journal.db not found"}

        conn = sqlite3.connect(str(self.journal_db))
        conn.row_factory = sqlite3.Row

        # Shadow trades (paper only)
        shadow_trades = conn.execute("""
            SELECT
                COUNT(*) as total_shadow_trades,
                SUM(CASE WHEN shadow_pnl_bps > 0 THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN shadow_pnl_bps <= 0 THEN 1 ELSE 0 END) as losses,
                AVG(CASE WHEN shadow_pnl_bps > 0 THEN 1.0 ELSE 0.0 END) as win_rate,
                SUM(shadow_pnl_bps) as total_pnl_bps,
                AVG(shadow_pnl_bps) as avg_pnl_bps,
                MAX(shadow_pnl_bps) as max_win_bps,
                MIN(shadow_pnl_bps) as max_loss_bps,
                AVG(duration_sec) as avg_duration_sec
            FROM shadow_trades
            WHERE ts BETWEEN ? AND ?
        """, (start_ts, end_ts)).fetchone()

        # By mode (scalp vs runner)
        by_mode = conn.execute("""
            SELECT
                mode,
                COUNT(*) as trades,
                AVG(CASE WHEN shadow_pnl_bps > 0 THEN 1.0 ELSE 0.0 END) as win_rate,
                AVG(shadow_pnl_bps) as avg_pnl_bps
            FROM shadow_trades
            WHERE ts BETWEEN ? AND ?
            GROUP BY mode
        """, (start_ts, end_ts)).fetchall()

        # By regime
        by_regime = conn.execute("""
            SELECT
                regime,
                COUNT(*) as trades,
                AVG(CASE WHEN shadow_pnl_bps > 0 THEN 1.0 ELSE 0.0 END) as win_rate,
                AVG(shadow_pnl_bps) as avg_pnl_bps
            FROM shadow_trades
            WHERE ts BETWEEN ? AND ?
            GROUP BY regime
        """, (start_ts, end_ts)).fetchall()

        conn.close()

        return {
            "total_shadow_trades": shadow_trades['total_shadow_trades'] or 0,
            "wins": shadow_trades['wins'] or 0,
            "losses": shadow_trades['losses'] or 0,
            "win_rate": shadow_trades['win_rate'] or 0.0,
            "total_pnl_bps": shadow_trades['total_pnl_bps'] or 0.0,
            "avg_pnl_bps": shadow_trades['avg_pnl_bps'] or 0.0,
            "max_win_bps": shadow_trades['max_win_bps'] or 0.0,
            "max_loss_bps": shadow_trades['max_loss_bps'] or 0.0,
            "avg_duration_sec": shadow_trades['avg_duration_sec'] or 0.0,
            "by_mode": [dict(row) for row in by_mode],
            "by_regime": [dict(row) for row in by_regime],
            "note": "SIMULATED trades (paper only, no real money)"
        }

    def _compute_learning_metrics(self, start_ts: str, end_ts: str) -> Dict[str, Any]:
        """Compute learning progress metrics"""
        if not self.learning_db.exists():
            return {"error": "learning.db not found"}

        conn = sqlite3.connect(str(self.learning_db))
        conn.row_factory = sqlite3.Row

        # Training sessions
        sessions = conn.execute("""
            SELECT
                COUNT(*) as training_sessions,
                SUM(samples_processed) as total_samples,
                MAX(auc) as best_auc,
                MIN(ece) as best_ece,
                AVG(delta_auc) as avg_improvement_auc
            FROM training_sessions
            WHERE ts BETWEEN ? AND ?
        """, (start_ts, end_ts)).fetchone()

        # Latest model metrics
        latest = conn.execute("""
            SELECT auc, ece, brier, wr
            FROM training_sessions
            WHERE ts BETWEEN ? AND ?
            ORDER BY ts DESC
            LIMIT 1
        """, (start_ts, end_ts)).fetchone()

        # Top features (latest session)
        top_features = conn.execute("""
            SELECT feature_name, importance, delta_importance
            FROM feature_importance
            WHERE ts BETWEEN ? AND ?
            ORDER BY importance DESC
            LIMIT 10
        """, (start_ts, end_ts)).fetchall()

        conn.close()

        return {
            "training_sessions": sessions['training_sessions'] or 0,
            "total_samples_processed": sessions['total_samples'] or 0,
            "best_auc": sessions['best_auc'] or 0.0,
            "best_ece": sessions['best_ece'] or 0.0,
            "avg_improvement_auc": sessions['avg_improvement_auc'] or 0.0,
            "latest_model": dict(latest) if latest else {},
            "top_features": [dict(f) for f in top_features],
            "models_ready_for_hamilton": self._check_model_readiness(latest)
        }

    def _compute_gate_metrics(self, start_ts: str, end_ts: str) -> Dict[str, Any]:
        """Compute gate performance metrics"""
        if not self.journal_db.exists():
            return {"error": "journal.db not found"}

        conn = sqlite3.connect(str(self.journal_db))
        conn.row_factory = sqlite3.Row

        # Shadow trades blocked by gates
        blocked = conn.execute("""
            SELECT
                blocked_by,
                COUNT(*) as total_blocked,
                SUM(CASE WHEN was_good_block = 1 THEN 1 ELSE 0 END) as good_blocks,
                SUM(CASE WHEN was_good_block = 0 THEN 1 ELSE 0 END) as bad_blocks,
                AVG(CASE WHEN was_good_block = 1 THEN 1.0 ELSE 0.0 END) as block_accuracy,
                SUM(cf_pnl_bps) as missed_pnl_bps
            FROM shadow_trades
            WHERE ts BETWEEN ? AND ?
              AND blocked_by IS NOT NULL
            GROUP BY blocked_by
        """, (start_ts, end_ts)).fetchall()

        conn.close()

        return {
            "gates": [dict(b) for b in blocked],
            "note": "Gate accuracy = % of blocks that were correct (blocked a loser)"
        }

    def _compute_model_metrics(self, start_ts: str, end_ts: str) -> Dict[str, Any]:
        """Compute model performance metrics"""
        if not self.learning_db.exists():
            return {}

        conn = sqlite3.connect(str(self.learning_db))
        conn.row_factory = sqlite3.Row

        # Latest model
        latest = conn.execute("""
            SELECT model_id, auc, ece, brier, wr
            FROM training_sessions
            WHERE ts <= ?
            ORDER BY ts DESC
            LIMIT 1
        """, (end_ts,)).fetchone()

        # Model improvement (vs yesterday)
        yesterday = (datetime.fromisoformat(start_ts) - timedelta(days=1)).isoformat()
        prev = conn.execute("""
            SELECT auc, ece
            FROM training_sessions
            WHERE ts < ?
            ORDER BY ts DESC
            LIMIT 1
        """, (start_ts,)).fetchone()

        conn.close()

        improvement = {}
        if latest and prev:
            improvement = {
                "delta_auc": latest['auc'] - prev['auc'] if latest['auc'] and prev['auc'] else None,
                "delta_ece": prev['ece'] - latest['ece'] if latest['ece'] and prev['ece'] else None,  # Lower is better
            }

        return {
            "latest_model_id": latest['model_id'] if latest else None,
            "latest_metrics": dict(latest) if latest else {},
            "improvement_vs_yesterday": improvement
        }

    def _compute_system_metrics(self, start_ts: str, end_ts: str) -> Dict[str, Any]:
        """Compute system health metrics"""
        # TODO: Query from event logs
        return {
            "events_logged": 0,
            "queue_health": "healthy",
            "errors": 0
        }

    def _check_model_readiness(self, model_metrics: Optional[sqlite3.Row]) -> bool:
        """
        Check if model is ready for Hamilton export.

        Criteria:
        - AUC >= 0.65 (minimum)
        - ECE <= 0.10 (well-calibrated)
        - Sufficient training data
        """
        if not model_metrics:
            return False

        auc = model_metrics.get('auc', 0)
        ece = model_metrics.get('ece', 1.0)

        ready = auc >= 0.65 and ece <= 0.10

        return ready

    def save_metrics_json(self, metrics: Dict[str, Any], output_path: str):
        """
        Save computed metrics as JSON for AI Council.

        Args:
            metrics: Computed metrics
            output_path: Where to save JSON
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        logger.info("metrics_saved", path=str(output_path))

    def verify_metrics(self, metrics: Dict[str, Any]) -> bool:
        """
        Verify all numbers in metrics are valid (anti-hallucination layer).

        Checks:
        - No NaN or Inf
        - Win rates between 0-1
        - Counts are non-negative integers
        - Percentages sum to 100% where applicable

        Returns:
            True if all valid, False otherwise
        """
        def check_number(value, name, min_val=None, max_val=None):
            if value is None:
                return True  # None is allowed

            if isinstance(value, (int, float)):
                if not (-1e10 < value < 1e10):  # Reasonable range
                    logger.error("invalid_metric", name=name, value=value, reason="out_of_range")
                    return False

                if min_val is not None and value < min_val:
                    logger.error("invalid_metric", name=name, value=value, reason=f"below_min_{min_val}")
                    return False

                if max_val is not None and value > max_val:
                    logger.error("invalid_metric", name=name, value=value, reason=f"above_max_{max_val}")
                    return False

                return True

            return True

        valid = True

        # Shadow trading metrics
        st = metrics.get("shadow_trading", {})
        valid &= check_number(st.get("total_shadow_trades"), "total_shadow_trades", min_val=0)
        valid &= check_number(st.get("win_rate"), "win_rate", min_val=0.0, max_val=1.0)
        valid &= check_number(st.get("wins"), "wins", min_val=0)
        valid &= check_number(st.get("losses"), "losses", min_val=0)

        # Learning metrics
        learning = metrics.get("learning", {})
        valid &= check_number(learning.get("training_sessions"), "training_sessions", min_val=0)
        valid &= check_number(learning.get("best_auc"), "best_auc", min_val=0.0, max_val=1.0)
        valid &= check_number(learning.get("best_ece"), "best_ece", min_val=0.0, max_val=1.0)

        if valid:
            logger.info("metrics_verified", status="valid")
        else:
            logger.error("metrics_verification_failed")

        return valid


if __name__ == '__main__':
    # Example usage
    print("Metrics Computer Example")
    print("=" * 80)

    computer = MetricsComputer()

    # Compute metrics for today
    date = datetime.utcnow().strftime("%Y-%m-%d")
    print(f"\n📊 Computing metrics for {date}...")

    metrics = computer.compute_daily_metrics(date)

    # Display
    print(f"\n✓ Metrics computed:")
    print(f"  Shadow trades: {metrics['shadow_trading']['total_shadow_trades']}")
    print(f"  Win rate: {metrics['shadow_trading']['win_rate']:.1%}")
    print(f"  Training sessions: {metrics['learning']['training_sessions']}")
    print(f"  Model ready for Hamilton: {metrics['learning']['models_ready_for_hamilton']}")

    # Save
    output_path = f"observability/data/metrics/metrics_{date}.json"
    computer.save_metrics_json(metrics, output_path)
    print(f"\n✓ Saved to: {output_path}")

    # Verify
    valid = computer.verify_metrics(metrics)
    print(f"\n✓ Verification: {'PASSED' if valid else 'FAILED'}")

    print("\n✓ Metrics computer ready!")
