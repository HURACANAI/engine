"""
Learning Tracker

Tracks what the trading bot learns over time:
- Model improvements (AUC, ECE, Brier)
- Historical data processed
- Feature importance changes
- Calibration drift
- Performance by regime

This answers: "What did the bot learn today? How much better did it get?"

Usage:
    tracker = LearningTracker()

    # Track new model training
    tracker.record_training(
        model_id="sha256:abc123...",
        samples_processed=5000,
        metrics={"auc": 0.72, "ece": 0.055},
        feature_importance={...}
    )

    # Get daily summary
    summary = tracker.get_daily_summary(date="2025-11-06")
"""

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import structlog
import json

from observability.core.registry import ModelRegistry

logger = structlog.get_logger(__name__)


class LearningTracker:
    """
    Track learning progress over time.

    Stores:
    - Training sessions (when, how many samples, metrics)
    - Feature importance evolution
    - Calibration history
    - Performance by regime
    """

    def __init__(
        self,
        db_path: str = "observability/data/sqlite/learning.db",
        registry: Optional[ModelRegistry] = None
    ):
        """
        Initialize learning tracker.

        Args:
            db_path: Path to learning database
            registry: Model registry (optional, for linking models)
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row  # Return dict-like rows

        self.registry = registry

        self._create_tables()

        logger.info("learning_tracker_initialized", db_path=str(self.db_path))

    def _create_tables(self):
        """Create learning tracking tables"""

        # Training sessions
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS training_sessions(
                session_id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                model_id TEXT NOT NULL,

                -- Data
                samples_processed INTEGER NOT NULL,
                samples_total INTEGER,
                data_start_date TEXT,
                data_end_date TEXT,

                -- Metrics
                auc REAL,
                ece REAL,
                brier REAL,
                wr REAL,

                -- Improvement
                delta_auc REAL,
                delta_ece REAL,
                delta_brier REAL,

                -- Metadata
                duration_sec REAL,
                previous_model_id TEXT,
                notes TEXT
            )
        """)

        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS ix_training_sessions_ts
            ON training_sessions(ts)
        """)

        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS ix_training_sessions_model_id
            ON training_sessions(model_id)
        """)

        # Feature importance history
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS feature_importance(
                importance_id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                model_id TEXT NOT NULL,
                session_id INTEGER,

                -- Feature importance (top features stored individually)
                feature_name TEXT NOT NULL,
                importance REAL NOT NULL,
                importance_rank INTEGER,

                -- Change from previous
                delta_importance REAL,
                delta_rank INTEGER,

                FOREIGN KEY (session_id) REFERENCES training_sessions(session_id)
            )
        """)

        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS ix_feature_importance_ts
            ON feature_importance(ts)
        """)

        # Calibration history
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS calibration_history(
                calibration_id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                model_id TEXT NOT NULL,
                session_id INTEGER,

                -- Calibration metrics
                ece REAL NOT NULL,
                mce REAL,  -- Max calibration error
                brier REAL,

                -- Calibration curve (JSON)
                bin_edges TEXT,  -- JSON array
                bin_counts TEXT,  -- JSON array
                bin_accuracies TEXT,  -- JSON array
                bin_confidences TEXT,  -- JSON array

                -- Method
                calibration_method TEXT,  -- 'platt_scaling', 'isotonic', 'none'

                FOREIGN KEY (session_id) REFERENCES training_sessions(session_id)
            )
        """)

        # Performance by regime
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS regime_performance(
                regime_perf_id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                model_id TEXT NOT NULL,
                session_id INTEGER,

                -- Regime
                regime TEXT NOT NULL,  -- 'TREND', 'RANGE', 'PANIC'

                -- Performance
                auc REAL,
                wr REAL,
                n_samples INTEGER,
                avg_return_bps REAL,

                -- Confidence
                avg_confidence REAL,

                FOREIGN KEY (session_id) REFERENCES training_sessions(session_id)
            )
        """)

        # Daily learning summary
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS daily_summary(
                date TEXT PRIMARY KEY,

                -- Training activity
                num_sessions INTEGER,
                total_samples_processed INTEGER,

                -- Best model metrics
                best_auc REAL,
                best_ece REAL,
                best_model_id TEXT,

                -- Improvement
                improvement_auc REAL,
                improvement_ece REAL,

                -- Top features (JSON)
                top_features_json TEXT,

                -- Generated
                generated_at TEXT,
                summary_text TEXT  -- AI-generated summary
            )
        """)

        self.conn.commit()

    def record_training(
        self,
        model_id: str,
        samples_processed: int,
        metrics: Dict[str, float],
        feature_importance: Optional[Dict[str, float]] = None,
        calibration: Optional[Dict[str, Any]] = None,
        regime_perf: Optional[Dict[str, Dict[str, float]]] = None,
        previous_model_id: Optional[str] = None,
        data_date_range: Optional[tuple] = None,
        duration_sec: Optional[float] = None,
        notes: Optional[str] = None
    ) -> int:
        """
        Record a training session.

        Args:
            model_id: Model SHA256 ID
            samples_processed: Number of samples used for training
            metrics: Model metrics (auc, ece, brier, wr)
            feature_importance: Dict of feature_name -> importance
            calibration: Calibration data
            regime_perf: Performance by regime
            previous_model_id: Previous model ID (for computing deltas)
            data_date_range: (start_date, end_date) tuple
            duration_sec: Training duration
            notes: Optional notes

        Returns:
            session_id
        """
        ts = datetime.utcnow().isoformat()

        # Compute deltas if previous model exists
        delta_auc = None
        delta_ece = None
        delta_brier = None

        if previous_model_id and self.registry:
            prev_metrics = self._get_model_metrics(previous_model_id)
            if prev_metrics:
                delta_auc = metrics.get('auc', 0) - prev_metrics.get('auc', 0)
                delta_ece = metrics.get('ece', 0) - prev_metrics.get('ece', 0)
                delta_brier = metrics.get('brier', 0) - prev_metrics.get('brier', 0)

        # Insert training session
        cursor = self.conn.execute("""
            INSERT INTO training_sessions(
                ts, model_id, samples_processed, samples_total,
                data_start_date, data_end_date,
                auc, ece, brier, wr,
                delta_auc, delta_ece, delta_brier,
                duration_sec, previous_model_id, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            ts, model_id, samples_processed, samples_processed,
            data_date_range[0] if data_date_range else None,
            data_date_range[1] if data_date_range else None,
            metrics.get('auc'), metrics.get('ece'), metrics.get('brier'), metrics.get('wr'),
            delta_auc, delta_ece, delta_brier,
            duration_sec, previous_model_id, notes
        ))
        session_id = cursor.lastrowid

        # Record feature importance
        if feature_importance:
            self._record_feature_importance(session_id, model_id, ts, feature_importance, previous_model_id)

        # Record calibration
        if calibration:
            self._record_calibration(session_id, model_id, ts, calibration)

        # Record regime performance
        if regime_perf:
            self._record_regime_performance(session_id, model_id, ts, regime_perf)

        self.conn.commit()

        logger.info(
            "training_recorded",
            session_id=session_id,
            model_id=model_id[:20] + "...",
            samples=samples_processed,
            auc=metrics.get('auc'),
            delta_auc=delta_auc
        )

        return session_id

    def _record_feature_importance(
        self,
        session_id: int,
        model_id: str,
        ts: str,
        feature_importance: Dict[str, float],
        previous_model_id: Optional[str]
    ):
        """Record feature importance for this session"""

        # Sort features by importance
        sorted_features = sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Get previous importance if exists
        prev_importance = {}
        prev_ranks = {}
        if previous_model_id:
            prev_importance, prev_ranks = self._get_previous_feature_importance(previous_model_id)

        # Insert top features
        for rank, (feature_name, importance) in enumerate(sorted_features[:50], 1):
            delta_importance = importance - prev_importance.get(feature_name, 0)
            delta_rank = prev_ranks.get(feature_name, 999) - rank

            self.conn.execute("""
                INSERT INTO feature_importance(
                    ts, model_id, session_id, feature_name, importance, importance_rank,
                    delta_importance, delta_rank
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                ts, model_id, session_id, feature_name, importance, rank,
                delta_importance, delta_rank
            ))

    def _record_calibration(
        self,
        session_id: int,
        model_id: str,
        ts: str,
        calibration: Dict[str, Any]
    ):
        """Record calibration data"""
        self.conn.execute("""
            INSERT INTO calibration_history(
                ts, model_id, session_id, ece, mce, brier,
                bin_edges, bin_counts, bin_accuracies, bin_confidences,
                calibration_method
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            ts, model_id, session_id,
            calibration.get('ece'), calibration.get('mce'), calibration.get('brier'),
            json.dumps(calibration.get('bin_edges', [])),
            json.dumps(calibration.get('bin_counts', [])),
            json.dumps(calibration.get('bin_accuracies', [])),
            json.dumps(calibration.get('bin_confidences', [])),
            calibration.get('method', 'none')
        ))

    def _record_regime_performance(
        self,
        session_id: int,
        model_id: str,
        ts: str,
        regime_perf: Dict[str, Dict[str, float]]
    ):
        """Record performance by regime"""
        for regime, perf in regime_perf.items():
            self.conn.execute("""
                INSERT INTO regime_performance(
                    ts, model_id, session_id, regime, auc, wr, n_samples, avg_return_bps, avg_confidence
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                ts, model_id, session_id, regime,
                perf.get('auc'), perf.get('wr'), perf.get('n_samples'),
                perf.get('avg_return_bps'), perf.get('avg_confidence')
            ))

    def get_daily_summary(self, date: str) -> Dict[str, Any]:
        """
        Get summary of what was learned on a specific date.

        Args:
            date: Date string (YYYY-MM-DD)

        Returns:
            Dict with:
            - num_sessions: Number of training sessions
            - total_samples: Total samples processed
            - best_metrics: Best model metrics
            - improvement: Improvement from start of day
            - top_features: Top features learned
            - regime_performance: Performance by regime
        """
        start_ts = f"{date}T00:00:00"
        end_ts = f"{date}T23:59:59"

        # Get training sessions for the day
        sessions = self.conn.execute("""
            SELECT * FROM training_sessions
            WHERE ts BETWEEN ? AND ?
            ORDER BY ts
        """, (start_ts, end_ts)).fetchall()

        if not sessions:
            return {
                "date": date,
                "num_sessions": 0,
                "total_samples": 0,
                "message": "No training sessions on this date"
            }

        # Aggregate metrics
        num_sessions = len(sessions)
        total_samples = sum(s['samples_processed'] for s in sessions)

        # Best model
        best_session = max(sessions, key=lambda s: s['auc'] or 0)
        best_metrics = {
            "auc": best_session['auc'],
            "ece": best_session['ece'],
            "brier": best_session['brier'],
            "wr": best_session['wr'],
            "model_id": best_session['model_id']
        }

        # Improvement (first session vs best session)
        first_session = sessions[0]
        improvement = {
            "auc": best_session['auc'] - first_session['auc'] if best_session['auc'] and first_session['auc'] else None,
            "ece": first_session['ece'] - best_session['ece'] if best_session['ece'] and first_session['ece'] else None,  # Lower is better
        }

        # Get latest feature importance
        latest_session_id = sessions[-1]['session_id']
        top_features = self.conn.execute("""
            SELECT feature_name, importance, delta_importance, delta_rank
            FROM feature_importance
            WHERE session_id = ?
            ORDER BY importance_rank
            LIMIT 10
        """, (latest_session_id,)).fetchall()

        # Get regime performance
        regime_perf = self.conn.execute("""
            SELECT regime, AVG(auc) as avg_auc, AVG(wr) as avg_wr, SUM(n_samples) as total_samples
            FROM regime_performance
            WHERE session_id IN ({})
            GROUP BY regime
        """.format(','.join('?' * num_sessions)), tuple(s['session_id'] for s in sessions)).fetchall()

        return {
            "date": date,
            "num_sessions": num_sessions,
            "total_samples": total_samples,
            "best_metrics": best_metrics,
            "improvement": improvement,
            "top_features": [dict(f) for f in top_features],
            "regime_performance": [dict(r) for r in regime_perf],
            "sessions": [dict(s) for s in sessions]
        }

    def get_learning_curve(self, days: int = 30) -> List[Dict[str, Any]]:
        """
        Get learning curve over time.

        Args:
            days: Number of days to look back

        Returns:
            List of daily metrics
        """
        cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()

        sessions = self.conn.execute("""
            SELECT
                DATE(ts) as date,
                COUNT(*) as num_sessions,
                SUM(samples_processed) as total_samples,
                MAX(auc) as best_auc,
                MIN(ece) as best_ece,
                AVG(delta_auc) as avg_improvement
            FROM training_sessions
            WHERE ts >= ?
            GROUP BY DATE(ts)
            ORDER BY date
        """, (cutoff,)).fetchall()

        return [dict(s) for s in sessions]

    def get_feature_evolution(self, feature_name: str, days: int = 30) -> List[Dict[str, Any]]:
        """
        Track how a specific feature's importance changed over time.

        Args:
            feature_name: Name of feature
            days: Number of days to look back

        Returns:
            List of {ts, importance, rank, delta}
        """
        cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()

        evolution = self.conn.execute("""
            SELECT ts, importance, importance_rank, delta_importance, delta_rank
            FROM feature_importance
            WHERE feature_name = ? AND ts >= ?
            ORDER BY ts
        """, (feature_name, cutoff)).fetchall()

        return [dict(e) for e in evolution]

    def _get_model_metrics(self, model_id: str) -> Optional[Dict[str, float]]:
        """Get metrics for a model from registry"""
        if not self.registry:
            return None

        # Query from registry database
        row = self.registry.conn.execute("""
            SELECT auc, ece, brier, wr
            FROM models
            WHERE model_id = ?
        """, (model_id,)).fetchone()

        if row:
            return dict(row)
        return None

    def _get_previous_feature_importance(self, model_id: str) -> tuple:
        """Get feature importance and ranks from previous model"""
        features = self.conn.execute("""
            SELECT feature_name, importance, importance_rank
            FROM feature_importance
            WHERE model_id = ?
        """, (model_id,)).fetchall()

        importance = {f['feature_name']: f['importance'] for f in features}
        ranks = {f['feature_name']: f['importance_rank'] for f in features}

        return importance, ranks


if __name__ == '__main__':
    # Example usage
    print("Learning Tracker Example")
    print("=" * 80)

    tracker = LearningTracker(db_path="observability/data/test_learning.db")

    # Record training session
    session_id = tracker.record_training(
        model_id="sha256:abc123def456",
        samples_processed=5000,
        metrics={"auc": 0.72, "ece": 0.055, "brier": 0.18, "wr": 0.74},
        feature_importance={
            "volatility_1h": 0.25,
            "spread_bps": 0.18,
            "volume_vs_avg": 0.15,
            "recent_trend_30m": 0.12,
            "liquidity_score": 0.10
        },
        calibration={
            "ece": 0.055,
            "mce": 0.12,
            "method": "platt_scaling"
        },
        regime_perf={
            "TREND": {"auc": 0.78, "wr": 0.82, "n_samples": 3000},
            "RANGE": {"auc": 0.68, "wr": 0.68, "n_samples": 2000}
        },
        data_date_range=("2025-11-01", "2025-11-05"),
        duration_sec=120.5,
        notes="First training session after Huracan v4 launch"
    )

    print(f"\n✓ Training session recorded: {session_id}")

    # Get daily summary
    summary = tracker.get_daily_summary(datetime.utcnow().strftime("%Y-%m-%d"))
    print(f"\n✓ Daily summary:")
    print(f"  Sessions: {summary['num_sessions']}")
    print(f"  Samples: {summary['total_samples']}")
    print(f"  Best AUC: {summary['best_metrics']['auc']:.3f}")

    print("\n✓ Learning tracker ready!")
