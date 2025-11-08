"""
Enhanced Learning Tracker

Extends base learning tracker with integration of all new quality systems:
- Calibration tracking (from models/calibration)
- Drift detection (from datasets/drift)
- Stress test results
- Gate evaluation results
- Counterfactual analysis
- Live feedback integration
- Curriculum learning progress

This answers: "What did the bot learn, how well did it pass gates,
and what improvements were made?"

Usage:
    from observability.analytics.enhanced_learning_tracker import EnhancedLearningTracker

    tracker = EnhancedLearningTracker()

    # Record comprehensive training session
    tracker.record_enhanced_training(
        model_id="btc_trend_v48",
        samples_processed=10000,
        validation_metrics=val_metrics,
        gate_verdict=gate_verdict,  # From decision gates
        calibration_snapshot=calibration,  # From calibration tracker
        drift_report=drift_report,  # From drift monitor
        stress_test_results=stress_results,
        curriculum_stage=2,  # Which stage of curriculum
        live_feedback_stats=feedback_stats
    )

    # Get comprehensive report
    report = tracker.generate_comprehensive_report(days=7)
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

import structlog
import pandas as pd

from observability.analytics.learning_tracker import LearningTracker
from models.calibration import CalibrationSnapshot
from datasets.drift import DriftReport
from observability.decision_gates import GateVerdict

logger = structlog.get_logger(__name__)


class EnhancedLearningTracker(LearningTracker):
    """
    Enhanced Learning Tracker

    Extends base LearningTracker with integration of:
    - Decision gate results
    - Calibration quality
    - Drift detection
    - Stress test results
    - Counterfactual analysis
    - Live feedback
    """

    def __init__(self, db_path: str = "observability/data/sqlite/enhanced_learning.db"):
        super().__init__(db_path)
        self._create_enhanced_tables()

    def _create_enhanced_tables(self):
        """Create additional tables for enhanced tracking"""

        # Gate evaluation results
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS gate_evaluations(
                eval_id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                model_id TEXT NOT NULL,
                session_id INTEGER,

                -- Verdict
                approved BOOLEAN NOT NULL,
                total_gates INTEGER,
                gates_passed INTEGER,
                overall_score REAL,

                -- Failed gates (JSON)
                failed_gates_json TEXT,
                failure_reasons_json TEXT,

                FOREIGN KEY (session_id) REFERENCES training_sessions(session_id)
            )
        """)

        # Calibration snapshots
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS calibration_snapshots(
                snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                model_id TEXT NOT NULL,
                session_id INTEGER,
                regime TEXT,

                -- Calibration metrics
                brier_score REAL,
                ece REAL,
                uncalibrated_brier REAL,
                uncalibrated_ece REAL,

                -- Improvement
                brier_improvement REAL,
                ece_improvement REAL,

                num_samples INTEGER,

                FOREIGN KEY (session_id) REFERENCES training_sessions(session_id)
            )
        """)

        # Drift detection results
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS drift_checks(
                drift_id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                model_id TEXT NOT NULL,
                session_id INTEGER,

                -- Drift status
                critical_drift BOOLEAN,
                max_psi REAL,

                -- Drifted features (JSON)
                drifted_features_json TEXT,

                -- Details (JSON)
                drift_details_json TEXT,

                FOREIGN KEY (session_id) REFERENCES training_sessions(session_id)
            )
        """)

        # Stress test results
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS stress_tests(
                test_id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                model_id TEXT NOT NULL,
                session_id INTEGER,

                -- Results
                total_scenarios INTEGER,
                scenarios_passed INTEGER,
                pass_rate REAL,

                -- Scenario results (JSON)
                scenario_results_json TEXT,

                FOREIGN KEY (session_id) REFERENCES training_sessions(session_id)
            )
        """)

        # Counterfactual analysis
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS counterfactual_analysis(
                analysis_id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                model_id TEXT NOT NULL,

                -- Decision quality
                avg_opportunity_cost REAL,
                avg_quality_score REAL,
                pct_suboptimal REAL,

                num_decisions INTEGER
            )
        """)

        # Live feedback stats
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS live_feedback_stats(
                feedback_id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                model_id TEXT NOT NULL,

                -- Performance
                live_sharpe REAL,
                live_win_rate REAL,
                live_avg_pnl_bps REAL,

                -- vs Backtest
                backtest_sharpe REAL,
                consistency_ratio REAL,

                num_trades INTEGER,
                days_live INTEGER
            )
        """)

        # Curriculum progress
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS curriculum_progress(
                progress_id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                model_id TEXT NOT NULL,
                session_id INTEGER,

                -- Curriculum stage
                stage_number INTEGER,
                stage_name TEXT,

                -- Performance
                stage_accuracy REAL,
                stage_sharpe REAL,

                -- Progression
                passed_stage BOOLEAN,

                FOREIGN KEY (session_id) REFERENCES training_sessions(session_id)
            )
        """)

        self.conn.commit()

    def record_enhanced_training(
        self,
        model_id: str,
        samples_processed: int,
        validation_metrics: Dict[str, float],
        gate_verdict: Optional[GateVerdict] = None,
        calibration_snapshot: Optional[CalibrationSnapshot] = None,
        drift_report: Optional[Any] = None,
        stress_test_results: Optional[Dict] = None,
        counterfactual_stats: Optional[Dict] = None,
        live_feedback_stats: Optional[Dict] = None,
        curriculum_stage: Optional[int] = None,
        **kwargs
    ) -> int:
        """
        Record enhanced training session with all new systems

        Args:
            model_id: Model identifier
            samples_processed: Number of samples
            validation_metrics: Validation metrics
            gate_verdict: Gate evaluation verdict
            calibration_snapshot: Calibration snapshot
            drift_report: Drift detection report
            stress_test_results: Stress test results
            counterfactual_stats: Counterfactual analysis stats
            live_feedback_stats: Live feedback statistics
            curriculum_stage: Current curriculum stage
            **kwargs: Additional args for base tracker

        Returns:
            session_id
        """
        # Record base training session
        session_id = self.record_training(
            model_id=model_id,
            samples_processed=samples_processed,
            metrics=validation_metrics,
            **kwargs
        )

        ts = datetime.utcnow().isoformat()

        # Record gate evaluation
        if gate_verdict:
            self._record_gate_evaluation(session_id, model_id, ts, gate_verdict)

        # Record calibration snapshot
        if calibration_snapshot:
            self._record_calibration_snapshot(session_id, model_id, ts, calibration_snapshot)

        # Record drift check
        if drift_report:
            self._record_drift_check(session_id, model_id, ts, drift_report)

        # Record stress test
        if stress_test_results:
            self._record_stress_test(session_id, model_id, ts, stress_test_results)

        # Record counterfactual analysis
        if counterfactual_stats:
            self._record_counterfactual_analysis(model_id, ts, counterfactual_stats)

        # Record live feedback
        if live_feedback_stats:
            self._record_live_feedback(model_id, ts, live_feedback_stats)

        # Record curriculum progress
        if curriculum_stage is not None:
            self._record_curriculum_progress(
                session_id, model_id, ts, curriculum_stage, validation_metrics
            )

        self.conn.commit()

        logger.info(
            "enhanced_training_recorded",
            session_id=session_id,
            model_id=model_id[:20],
            gates_passed=gate_verdict.approved if gate_verdict else None,
            calibrated=calibration_snapshot is not None,
            drift_checked=drift_report is not None
        )

        return session_id

    def _record_gate_evaluation(
        self,
        session_id: int,
        model_id: str,
        ts: str,
        verdict: GateVerdict
    ):
        """Record gate evaluation results"""
        import json

        self.conn.execute("""
            INSERT INTO gate_evaluations(
                ts, model_id, session_id, approved, total_gates, gates_passed,
                overall_score, failed_gates_json, failure_reasons_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            ts, model_id, session_id,
            verdict.approved,
            verdict.total_gates,
            verdict.num_gates_passed,
            verdict.overall_score,
            json.dumps(verdict.failed_gates),
            json.dumps(verdict.failure_reasons)
        ))

    def _record_calibration_snapshot(
        self,
        session_id: int,
        model_id: str,
        ts: str,
        snapshot: CalibrationSnapshot
    ):
        """Record calibration snapshot"""
        brier_improvement = None
        ece_improvement = None

        if snapshot.uncalibrated_brier is not None:
            brier_improvement = snapshot.uncalibrated_brier - snapshot.brier_score

        if snapshot.uncalibrated_ece is not None:
            ece_improvement = snapshot.uncalibrated_ece - snapshot.ece

        self.conn.execute("""
            INSERT INTO calibration_snapshots(
                ts, model_id, session_id, regime,
                brier_score, ece, uncalibrated_brier, uncalibrated_ece,
                brier_improvement, ece_improvement, num_samples
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            ts, model_id, session_id, snapshot.regime,
            snapshot.brier_score, snapshot.ece,
            snapshot.uncalibrated_brier, snapshot.uncalibrated_ece,
            brier_improvement, ece_improvement,
            snapshot.num_samples
        ))

    def _record_drift_check(
        self,
        session_id: int,
        model_id: str,
        ts: str,
        drift_report: Any
    ):
        """Record drift check results"""
        import json

        self.conn.execute("""
            INSERT INTO drift_checks(
                ts, model_id, session_id, critical_drift, max_psi,
                drifted_features_json, drift_details_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            ts, model_id, session_id,
            getattr(drift_report, 'critical_drift', False),
            getattr(drift_report, 'max_psi', 0.0),
            json.dumps(getattr(drift_report, 'drifted_features', [])),
            json.dumps({})  # Placeholder for detailed drift data
        ))

    def _record_stress_test(
        self,
        session_id: int,
        model_id: str,
        ts: str,
        results: Dict
    ):
        """Record stress test results"""
        import json

        total = results.get('total_scenarios', 0)
        passed = results.get('scenarios_passed', 0)
        pass_rate = passed / total if total > 0 else 0.0

        self.conn.execute("""
            INSERT INTO stress_tests(
                ts, model_id, session_id, total_scenarios, scenarios_passed,
                pass_rate, scenario_results_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            ts, model_id, session_id,
            total, passed, pass_rate,
            json.dumps(results.get('scenario_results', {}))
        ))

    def _record_counterfactual_analysis(
        self,
        model_id: str,
        ts: str,
        stats: Dict
    ):
        """Record counterfactual analysis"""
        self.conn.execute("""
            INSERT INTO counterfactual_analysis(
                ts, model_id, avg_opportunity_cost, avg_quality_score,
                pct_suboptimal, num_decisions
            ) VALUES (?, ?, ?, ?, ?, ?)
        """, (
            ts, model_id,
            stats.get('avg_opportunity_cost', 0.0),
            stats.get('avg_quality_score', 0.0),
            stats.get('pct_suboptimal', 0.0),
            stats.get('num_decisions', 0)
        ))

    def _record_live_feedback(
        self,
        model_id: str,
        ts: str,
        stats: Dict
    ):
        """Record live feedback statistics"""
        consistency = None
        if stats.get('backtest_sharpe') and stats.get('live_sharpe'):
            consistency = stats['live_sharpe'] / stats['backtest_sharpe']

        self.conn.execute("""
            INSERT INTO live_feedback_stats(
                ts, model_id, live_sharpe, live_win_rate, live_avg_pnl_bps,
                backtest_sharpe, consistency_ratio, num_trades, days_live
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            ts, model_id,
            stats.get('live_sharpe'),
            stats.get('live_win_rate'),
            stats.get('live_avg_pnl_bps'),
            stats.get('backtest_sharpe'),
            consistency,
            stats.get('num_trades', 0),
            stats.get('days_live', 0)
        ))

    def _record_curriculum_progress(
        self,
        session_id: int,
        model_id: str,
        ts: str,
        stage_number: int,
        metrics: Dict
    ):
        """Record curriculum learning progress"""
        stage_names = ["Easy", "Medium", "Hard", "Expert", "Master"]
        stage_name = stage_names[stage_number] if stage_number < len(stage_names) else f"Stage {stage_number}"

        self.conn.execute("""
            INSERT INTO curriculum_progress(
                ts, model_id, session_id, stage_number, stage_name,
                stage_accuracy, stage_sharpe, passed_stage
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            ts, model_id, session_id,
            stage_number, stage_name,
            metrics.get('accuracy'),
            metrics.get('sharpe'),
            metrics.get('passed_stage', False)
        ))

    def generate_comprehensive_report(self, days: int = 7) -> str:
        """
        Generate comprehensive learning report

        Args:
            days: Days to look back

        Returns:
            Report string
        """
        cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()

        # Get training sessions
        sessions = self.conn.execute("""
            SELECT COUNT(*) as num_sessions, SUM(samples_processed) as total_samples
            FROM training_sessions
            WHERE ts >= ?
        """, (cutoff,)).fetchone()

        # Get gate pass rate
        gates = self.conn.execute("""
            SELECT
                COUNT(*) as total_evaluations,
                SUM(CASE WHEN approved THEN 1 ELSE 0 END) as approved_count,
                AVG(overall_score) as avg_score
            FROM gate_evaluations
            WHERE ts >= ?
        """, (cutoff,)).fetchone()

        # Get calibration improvement
        calibration = self.conn.execute("""
            SELECT
                AVG(brier_improvement) as avg_brier_improvement,
                AVG(ece_improvement) as avg_ece_improvement
            FROM calibration_snapshots
            WHERE ts >= ?
        """, (cutoff,)).fetchone()

        # Get drift checks
        drift = self.conn.execute("""
            SELECT
                COUNT(*) as total_checks,
                SUM(CASE WHEN critical_drift THEN 1 ELSE 0 END) as critical_drift_count,
                AVG(max_psi) as avg_psi
            FROM drift_checks
            WHERE ts >= ?
        """, (cutoff,)).fetchone()

        # Generate report
        lines = [
            "=" * 80,
            f"COMPREHENSIVE LEARNING REPORT (Last {days} Days)",
            "=" * 80,
            "",
            "TRAINING ACTIVITY:",
            f"  Sessions: {sessions['num_sessions'] or 0}",
            f"  Samples Processed: {sessions['total_samples'] or 0:,}",
            "",
            "GATE EVALUATIONS:",
            f"  Total Evaluations: {gates['total_evaluations'] or 0}",
            f"  Approved: {gates['approved_count'] or 0}",
            f"  Pass Rate: {(gates['approved_count'] or 0) / max(gates['total_evaluations'] or 1, 1) * 100:.1f}%",
            f"  Average Score: {gates['avg_score'] or 0:.2%}",
            "",
            "CALIBRATION QUALITY:",
            f"  Avg Brier Improvement: {calibration['avg_brier_improvement'] or 0:.3f}",
            f"  Avg ECE Improvement: {calibration['avg_ece_improvement'] or 0:.3f}",
            "",
            "DRIFT MONITORING:",
            f"  Total Checks: {drift['total_checks'] or 0}",
            f"  Critical Drift Detected: {drift['critical_drift_count'] or 0}",
            f"  Average PSI: {drift['avg_psi'] or 0:.3f}",
            "",
            "=" * 80
        ]

        return "\n".join(lines)
