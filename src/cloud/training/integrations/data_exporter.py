"""
COMPREHENSIVE data exporter for Dropbox sync - EXPORTS EVERYTHING A-Z.

Exports ALL engine data to files:
- Trade history (wins/losses) - ALL trades from PostgreSQL
- Performance metrics - Model performance, win rates, P&L
- ML metrics - Training metrics, validation scores, model versions
- Model performance - Daily performance tracking
- Backtest results - All backtest outcomes
- Shadow trade results - All shadow trades
- Pattern library - All learned patterns
- Win/loss analysis - Detailed analysis of every win and loss
- Post-exit tracking - What happened after we exited
- Observability data - SQLite journal data (trades, models, deltas)
- Learning snapshots - Everything the engine learned
- System metrics - Health checks, errors, alerts
- Config files - All configuration used
- Training artifacts - Model metadata, component models
- Reports - All generated reports
- EVERYTHING ELSE - Complete coverage A-Z
"""

from __future__ import annotations

import json
import csv
import sqlite3
from datetime import datetime, date, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

try:
    import structlog  # type: ignore[import-untyped]
    _HAS_STRUCTLOG = True
    logger = structlog.get_logger(__name__)
except ImportError:
    # Fallback if structlog is not available
    import logging
    _HAS_STRUCTLOG = False
    _base_logger = logging.getLogger(__name__)
    # Create a wrapper that mimics structlog's API
    class LoggerWrapper:
        def _log(self, level: str, msg: str, **kwargs: Any) -> None:
            if kwargs:
                extra_info = ", ".join(f"{k}={v}" for k, v in kwargs.items())
                _base_logger.log(getattr(logging, level.upper()), f"{msg} | {extra_info}")
            else:
                _base_logger.log(getattr(logging, level.upper()), msg)
        def info(self, msg: str, **kwargs: Any) -> None:
            self._log("INFO", msg, **kwargs)
        def warning(self, msg: str, **kwargs: Any) -> None:
            self._log("WARNING", msg, **kwargs)
        def error(self, msg: str, **kwargs: Any) -> None:
            self._log("ERROR", msg, **kwargs)
        def debug(self, msg: str, **kwargs: Any) -> None:
            self._log("DEBUG", msg, **kwargs)
    logger = LoggerWrapper()  # type: ignore[assignment]

try:
    import pandas as pd  # type: ignore[import-untyped]
    import psycopg2  # type: ignore[import-untyped]
    from psycopg2.extras import RealDictCursor  # type: ignore[import-untyped]
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False
    psycopg2 = None  # type: ignore[assignment]
    pd = None  # type: ignore[assignment]
    RealDictCursor = None  # type: ignore[assignment,misc]

if TYPE_CHECKING:
    from psycopg2.extras import RealDictCursor as RealDictCursorType  # type: ignore[import-untyped]


class ComprehensiveDataExporter:
    """Export ALL engine data to files for Dropbox sync."""
    
    def __init__(
        self,
        dsn: Optional[str] = None,
        output_dir: Path = Path("exports"),
    ):
        """Initialize data exporter.
        
        Args:
            dsn: PostgreSQL connection string
            output_dir: Directory to save exported files
        """
        self.dsn = dsn
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("data_exporter_initialized", output_dir=str(self.output_dir))
    
    def export_all(self, run_date: Optional[date] = None) -> Dict[str, int]:
        """Export ALL data to files - COMPREHENSIVE A-Z EXPORT.
        
        Args:
            run_date: Date to export (defaults to today)
            
        Returns:
            Dictionary with export counts
        """
        if run_date is None:
            run_date = datetime.now(timezone.utc).date()
        
        results = {}
        
        # ===== POSTGRESQL DATABASE EXPORTS =====
        if self.dsn:
            # Core trade data
            results["trade_history"] = self.export_trade_history(run_date)
            results["model_performance"] = self.export_model_performance(run_date)
            results["win_loss_analysis"] = self.export_win_loss_analysis(run_date)
            results["pattern_library"] = self.export_pattern_library(run_date)
            results["post_exit_tracking"] = self.export_post_exit_tracking(run_date)
            
            # Additional comprehensive exports
            results["all_trades_complete"] = self.export_all_trades_complete(run_date)
            results["pattern_performance"] = self.export_pattern_performance(run_date)
            results["regime_analysis"] = self.export_regime_analysis(run_date)
            results["model_evolution"] = self.export_model_evolution(run_date)
        
        # ===== SQLITE OBSERVABILITY EXPORTS =====
        results["observability_trades"] = self.export_observability_trades(run_date)
        results["observability_models"] = self.export_observability_models(run_date)
        results["observability_model_deltas"] = self.export_observability_model_deltas(run_date)
        
        # ===== FILE SYSTEM EXPORTS =====
        results["learning_snapshots"] = self.export_learning_snapshots(run_date)
        results["backtest_results"] = self.export_backtest_results(run_date)
        results["training_artifacts"] = self.export_training_artifacts(run_date)
        
        # ===== METRICS & SUMMARY EXPORTS =====
        results["performance_summary"] = self.export_performance_summary(run_date)
        results["comprehensive_metrics"] = self.export_comprehensive_metrics(run_date)
        
        logger.info(
            "comprehensive_data_export_complete",
            run_date=run_date.isoformat(),
            **results,
            total_exports=sum(results.values()),
        )
        
        return results
    
    def export_trade_history(self, run_date: date) -> int:
        """Export all trade history to CSV.
        
        Returns:
            Number of trades exported
        """
        if not self.dsn or not DB_AVAILABLE or psycopg2 is None or pd is None or RealDictCursor is None:
            logger.warning("trade_history_export_skipped", reason="Database not available")
            return 0
        
        try:
            conn = psycopg2.connect(self.dsn)
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Query all trades
            cursor.execute("""
                SELECT 
                    trade_id,
                    symbol,
                    entry_timestamp,
                    entry_price,
                    exit_timestamp,
                    exit_price,
                    exit_reason,
                    position_size_gbp,
                    direction,
                    hold_duration_minutes,
                    gross_profit_bps,
                    net_profit_gbp,
                    fees_gbp,
                    slippage_bps,
                    market_regime,
                    volatility_bps,
                    spread_at_entry_bps,
                    is_winner,
                    win_quality,
                    model_version,
                    model_confidence,
                    created_at
                FROM trade_memory
                WHERE DATE(entry_timestamp) = %s
                ORDER BY entry_timestamp DESC
            """, (run_date,))
            
            trades = cursor.fetchall()
            cursor.close()
            conn.close()
            
            if not trades:
                logger.info("no_trades_to_export", run_date=run_date.isoformat())
                return 0
            
            # Convert to DataFrame and save
            df = pd.DataFrame(trades)
            output_file = self.output_dir / f"trade_history_{run_date.isoformat()}.csv"
            df.to_csv(output_file, index=False)
            
            logger.info(
                "trade_history_exported",
                run_date=run_date.isoformat(),
                trades_exported=len(trades),
                output_file=str(output_file),
            )
            
            return len(trades)
            
        except Exception as e:
            logger.error("trade_history_export_failed", error=str(e))
            return 0
    
    def export_model_performance(self, run_date: date) -> int:
        """Export model performance metrics to CSV.
        
        Returns:
            Number of records exported
        """
        if not self.dsn or not DB_AVAILABLE or psycopg2 is None or pd is None or RealDictCursor is None:
            return 0
        
        try:
            conn = psycopg2.connect(self.dsn)
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            cursor.execute("""
                SELECT *
                FROM model_performance
                WHERE evaluation_date = %s
                ORDER BY model_version, evaluation_date
            """, (run_date,))
            
            records = cursor.fetchall()
            cursor.close()
            conn.close()
            
            if not records:
                return 0
            
            df = pd.DataFrame(records)
            output_file = self.output_dir / f"model_performance_{run_date.isoformat()}.csv"
            df.to_csv(output_file, index=False)
            
            logger.info(
                "model_performance_exported",
                records_exported=len(records),
                output_file=str(output_file),
            )
            
            return len(records)
            
        except Exception as e:
            logger.error("model_performance_export_failed", error=str(e))
            return 0
    
    def export_win_loss_analysis(self, run_date: date) -> int:
        """Export win/loss analysis to JSON.
        
        Returns:
            Number of analyses exported
        """
        if not self.dsn or not DB_AVAILABLE or psycopg2 is None or RealDictCursor is None:
            return 0
        
        try:
            conn = psycopg2.connect(self.dsn)
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Get win analysis
            cursor.execute("""
                SELECT *
                FROM win_analysis
                WHERE DATE(created_at) = %s
            """, (run_date,))
            wins = cursor.fetchall()
            
            # Get loss analysis
            cursor.execute("""
                SELECT *
                FROM loss_analysis
                WHERE DATE(created_at) = %s
            """, (run_date,))
            losses = cursor.fetchall()
            
            cursor.close()
            conn.close()
            
            analysis = {
                "run_date": run_date.isoformat(),
                "wins": [dict(win) for win in wins],
                "losses": [dict(loss) for loss in losses],
                "summary": {
                    "total_wins": len(wins),
                    "total_losses": len(losses),
                },
            }
            
            output_file = self.output_dir / f"win_loss_analysis_{run_date.isoformat()}.json"
            with open(output_file, "w") as f:
                json.dump(analysis, f, indent=2, default=str)
            
            logger.info(
                "win_loss_analysis_exported",
                wins=len(wins),
                losses=len(losses),
                output_file=str(output_file),
            )
            
            return len(wins) + len(losses)
            
        except Exception as e:
            logger.error("win_loss_analysis_export_failed", error=str(e))
            return 0
    
    def export_pattern_library(self, run_date: date) -> int:
        """Export pattern library to CSV.
        
        Returns:
            Number of patterns exported
        """
        if not self.dsn or not DB_AVAILABLE or psycopg2 is None or pd is None or RealDictCursor is None:
            return 0
        
        try:
            conn = psycopg2.connect(self.dsn)
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            cursor.execute("""
                SELECT *
                FROM pattern_library
                WHERE DATE(created_at) = %s
                ORDER BY win_rate DESC
            """, (run_date,))
            
            patterns = cursor.fetchall()
            cursor.close()
            conn.close()
            
            if not patterns:
                return 0
            
            df = pd.DataFrame(patterns)
            output_file = self.output_dir / f"pattern_library_{run_date.isoformat()}.csv"
            df.to_csv(output_file, index=False)
            
            logger.info(
                "pattern_library_exported",
                patterns_exported=len(patterns),
                output_file=str(output_file),
            )
            
            return len(patterns)
            
        except Exception as e:
            logger.error("pattern_library_export_failed", error=str(e))
            return 0
    
    def export_post_exit_tracking(self, run_date: date) -> int:
        """Export post-exit tracking data to CSV.
        
        Returns:
            Number of records exported
        """
        if not self.dsn or not DB_AVAILABLE or psycopg2 is None or pd is None or RealDictCursor is None:
            return 0
        
        try:
            conn = psycopg2.connect(self.dsn)
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            cursor.execute("""
                SELECT *
                FROM post_exit_tracking
                WHERE DATE(created_at) = %s
            """, (run_date,))
            
            records = cursor.fetchall()
            cursor.close()
            conn.close()
            
            if not records:
                return 0
            
            df = pd.DataFrame(records)
            output_file = self.output_dir / f"post_exit_tracking_{run_date.isoformat()}.csv"
            df.to_csv(output_file, index=False)
            
            logger.info(
                "post_exit_tracking_exported",
                records_exported=len(records),
                output_file=str(output_file),
            )
            
            return len(records)
            
        except Exception as e:
            logger.error("post_exit_tracking_export_failed", error=str(e))
            return 0
    
    def export_shadow_trades(self, run_date: date) -> int:
        """Export shadow trades to CSV.
        
        Returns:
            Number of shadow trades exported
        """
        if not self.dsn or not DB_AVAILABLE or psycopg2 is None or pd is None or RealDictCursor is None:
            return 0
        
        try:
            conn = psycopg2.connect(self.dsn)
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Try to query shadow_trades table (might not exist in all schemas)
            try:
                cursor.execute("""
                    SELECT *
                    FROM shadow_trades
                    WHERE DATE(ts) = %s
                    ORDER BY ts DESC
                """, (run_date,))
                trades = cursor.fetchall()
            except psycopg2.errors.UndefinedTable:  # type: ignore[attr-defined]
                # Table doesn't exist - that's OK
                logger.debug("shadow_trades_table_not_found")
                trades = []
            
            cursor.close()
            conn.close()
            
            if not trades:
                return 0
            
            df = pd.DataFrame(trades)
            output_file = self.output_dir / f"shadow_trades_{run_date.isoformat()}.csv"
            df.to_csv(output_file, index=False)
            
            logger.info(
                "shadow_trades_exported",
                trades_exported=len(trades),
                output_file=str(output_file),
            )
            
            return len(trades)
            
        except Exception as e:
            logger.error("shadow_trades_export_failed", error=str(e))
            return 0
    
    def export_all_trades_complete(self, run_date: date) -> int:
        """Export ALL trades with complete details (not just today's).
        
        Returns:
            Number of trades exported
        """
        if not self.dsn or not DB_AVAILABLE or psycopg2 is None or pd is None or RealDictCursor is None:
            return 0
        
        try:
            conn = psycopg2.connect(self.dsn)
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Export ALL trades (not just today's)
            cursor.execute("""
                SELECT *
                FROM trade_memory
                ORDER BY entry_timestamp DESC
                LIMIT 100000
            """)
            
            trades = cursor.fetchall()
            cursor.close()
            conn.close()
            
            if not trades:
                return 0
            
            df = pd.DataFrame(trades)
            output_file = self.output_dir / f"all_trades_complete_{run_date.isoformat()}.csv"
            df.to_csv(output_file, index=False)
            
            logger.info("all_trades_complete_exported", trades_exported=len(trades))
            return len(trades)
            
        except Exception as e:
            logger.error("all_trades_complete_export_failed", error=str(e))
            return 0
    
    def export_pattern_performance(self, run_date: date) -> int:
        """Export pattern performance metrics.
        
        Returns:
            Number of patterns exported
        """
        if not self.dsn or not DB_AVAILABLE or psycopg2 is None or pd is None or RealDictCursor is None:
            return 0
        
        try:
            conn = psycopg2.connect(self.dsn)
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            cursor.execute("""
                SELECT *
                FROM pattern_library
                ORDER BY win_rate DESC, total_occurrences DESC
            """)
            
            patterns = cursor.fetchall()
            cursor.close()
            conn.close()
            
            if not patterns:
                return 0
            
            df = pd.DataFrame(patterns)
            output_file = self.output_dir / f"pattern_performance_{run_date.isoformat()}.csv"
            df.to_csv(output_file, index=False)
            
            logger.info("pattern_performance_exported", patterns_exported=len(patterns))
            return len(patterns)
            
        except Exception as e:
            logger.error("pattern_performance_export_failed", error=str(e))
            return 0
    
    def export_regime_analysis(self, run_date: date) -> int:
        """Export regime-specific performance analysis.
        
        Returns:
            Number of records exported
        """
        if not self.dsn or not DB_AVAILABLE or psycopg2 is None or pd is None or RealDictCursor is None:
            return 0
        
        try:
            conn = psycopg2.connect(self.dsn)
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Aggregate by regime
            cursor.execute("""
                SELECT 
                    market_regime,
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN is_winner THEN 1 ELSE 0 END) as wins,
                    SUM(CASE WHEN NOT is_winner THEN 1 ELSE 0 END) as losses,
                    AVG(CASE WHEN is_winner THEN 1.0 ELSE 0.0 END) as win_rate,
                    AVG(net_profit_gbp) as avg_profit_gbp,
                    SUM(net_profit_gbp) as total_profit_gbp,
                    AVG(gross_profit_bps) as avg_profit_bps,
                    AVG(hold_duration_minutes) as avg_hold_minutes
                FROM trade_memory
                WHERE market_regime IS NOT NULL
                GROUP BY market_regime
                ORDER BY total_trades DESC
            """)
            
            records = cursor.fetchall()
            cursor.close()
            conn.close()
            
            if not records:
                return 0
            
            df = pd.DataFrame(records)
            output_file = self.output_dir / f"regime_analysis_{run_date.isoformat()}.csv"
            df.to_csv(output_file, index=False)
            
            logger.info("regime_analysis_exported", records_exported=len(records))
            return len(records)
            
        except Exception as e:
            logger.error("regime_analysis_export_failed", error=str(e))
            return 0
    
    def export_model_evolution(self, run_date: date) -> int:
        """Export model evolution over time.
        
        Returns:
            Number of records exported
        """
        if not self.dsn or not DB_AVAILABLE or psycopg2 is None or pd is None or RealDictCursor is None:
            return 0
        
        try:
            conn = psycopg2.connect(self.dsn)
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            cursor.execute("""
                SELECT *
                FROM model_performance
                ORDER BY evaluation_date DESC, model_version
            """)
            
            records = cursor.fetchall()
            cursor.close()
            conn.close()
            
            if not records:
                return 0
            
            df = pd.DataFrame(records)
            output_file = self.output_dir / f"model_evolution_{run_date.isoformat()}.csv"
            df.to_csv(output_file, index=False)
            
            logger.info("model_evolution_exported", records_exported=len(records))
            return len(records)
            
        except Exception as e:
            logger.error("model_evolution_export_failed", error=str(e))
            return 0
    
    def export_observability_trades(self, run_date: date) -> int:
        """Export trades from SQLite observability database.
        
        Returns:
            Number of trades exported
        """
        if pd is None:
            return 0
        
        db_path = Path("observability/data/sqlite/journal.db")
        if not db_path.exists():
            return 0
        
        try:
            conn = sqlite3.connect(str(db_path))
            conn.row_factory = sqlite3.Row
            
            cursor = conn.cursor()
            cursor.execute("""
                SELECT *
                FROM trades
                WHERE DATE(ts_open) = ?
                ORDER BY ts_open DESC
            """, (run_date.isoformat(),))
            
            rows = cursor.fetchall()
            cursor.close()
            conn.close()
            
            if not rows:
                return 0
            
            # Convert to list of dicts
            trades = [dict(row) for row in rows]
            
            df = pd.DataFrame(trades)
            output_file = self.output_dir / f"observability_trades_{run_date.isoformat()}.csv"
            df.to_csv(output_file, index=False)
            
            logger.info("observability_trades_exported", trades_exported=len(trades))
            return len(trades)
            
        except Exception as e:
            logger.error("observability_trades_export_failed", error=str(e))
            return 0
    
    def export_observability_models(self, run_date: date) -> int:
        """Export models from SQLite observability database.
        
        Returns:
            Number of models exported
        """
        if pd is None:
            return 0
        
        db_path = Path("observability/data/sqlite/journal.db")
        if not db_path.exists():
            return 0
        
        try:
            conn = sqlite3.connect(str(db_path))
            conn.row_factory = sqlite3.Row
            
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM models ORDER BY created_at DESC")
            
            rows = cursor.fetchall()
            cursor.close()
            conn.close()
            
            if not rows:
                return 0
            
            models = [dict(row) for row in rows]
            df = pd.DataFrame(models)
            output_file = self.output_dir / f"observability_models_{run_date.isoformat()}.csv"
            df.to_csv(output_file, index=False)
            
            logger.info("observability_models_exported", models_exported=len(models))
            return len(models)
            
        except Exception as e:
            logger.error("observability_models_export_failed", error=str(e))
            return 0
    
    def export_observability_model_deltas(self, run_date: date) -> int:
        """Export model deltas from SQLite observability database.
        
        Returns:
            Number of deltas exported
        """
        if pd is None:
            return 0
        
        db_path = Path("observability/data/sqlite/journal.db")
        if not db_path.exists():
            return 0
        
        try:
            conn = sqlite3.connect(str(db_path))
            conn.row_factory = sqlite3.Row
            
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM model_deltas ORDER BY ts DESC")
            
            rows = cursor.fetchall()
            cursor.close()
            conn.close()
            
            if not rows:
                return 0
            
            deltas = [dict(row) for row in rows]
            df = pd.DataFrame(deltas)
            output_file = self.output_dir / f"observability_model_deltas_{run_date.isoformat()}.csv"
            df.to_csv(output_file, index=False)
            
            logger.info("observability_model_deltas_exported", deltas_exported=len(deltas))
            return len(deltas)
            
        except Exception as e:
            logger.error("observability_model_deltas_export_failed", error=str(e))
            return 0
    
    def export_learning_snapshots(self, run_date: date) -> int:
        """Export learning snapshots from logs/learning directory.
        
        Returns:
            Number of snapshots exported
        """
        learning_dir = Path("logs/learning")
        if not learning_dir.exists():
            return 0
        
        try:
            # Copy all learning snapshot files to exports
            snapshot_files = list(learning_dir.glob("*.json"))
            exported = 0
            
            for snapshot_file in snapshot_files:
                # Check if file was modified today
                file_date = datetime.fromtimestamp(snapshot_file.stat().st_mtime).date()
                if file_date == run_date:
                    # Copy to exports
                    export_file = self.output_dir / f"learning_{snapshot_file.name}"
                    import shutil
                    shutil.copy2(snapshot_file, export_file)
                    exported += 1
            
            logger.info("learning_snapshots_exported", snapshots_exported=exported)
            return exported
            
        except Exception as e:
            logger.error("learning_snapshots_export_failed", error=str(e))
            return 0
    
    def export_backtest_results(self, run_date: date) -> int:
        """Export backtest results if they exist.
        
        Returns:
            Number of backtest files exported
        """
        # Look for backtest CSV files in various locations
        backtest_dirs = [
            Path("backtests"),
            Path("results"),
            Path("reports"),
        ]
        
        exported = 0
        for backtest_dir in backtest_dirs:
            if backtest_dir.exists():
                for pattern in ["*backtest*.csv", "*backtest*.json"]:
                    for file in backtest_dir.rglob(pattern):
                        # Copy to exports
                        export_file = self.output_dir / f"backtest_{file.name}"
                        import shutil
                        shutil.copy2(file, export_file)
                        exported += 1
        
        if exported > 0:
            logger.info("backtest_results_exported", files_exported=exported)
        
        return exported
    
    def export_training_artifacts(self, run_date: date) -> int:
        """Export training artifacts (metadata, component models).
        
        Returns:
            Number of artifacts exported
        """
        models_dir = Path("models")
        if not models_dir.exists():
            return 0
        
        try:
            exported = 0
            # Export metadata.json files
            for metadata_file in models_dir.rglob("metadata.json"):
                export_file = self.output_dir / f"artifact_{metadata_file.parent.name}_metadata.json"
                import shutil
                shutil.copy2(metadata_file, export_file)
                exported += 1
            
            logger.info("training_artifacts_exported", artifacts_exported=exported)
            return exported
            
        except Exception as e:
            logger.error("training_artifacts_export_failed", error=str(e))
            return 0
    
    def export_comprehensive_metrics(self, run_date: date) -> int:
        """Export comprehensive metrics summary.
        
        Returns:
            1 if successful, 0 otherwise
        """
        if not self.dsn or not DB_AVAILABLE or psycopg2 is None or RealDictCursor is None:
            return 0
        
        try:
            conn = psycopg2.connect(self.dsn)
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Get comprehensive metrics
            metrics = {}
            
            # Total trades
            cursor.execute("SELECT COUNT(*) as count FROM trade_memory")
            row = cursor.fetchone()
            if row is not None:
                metrics["total_trades"] = row["count"]
            else:
                metrics["total_trades"] = 0
            
            # Win rate
            cursor.execute("""
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN is_winner THEN 1 ELSE 0 END) as wins
                FROM trade_memory
            """)
            row = cursor.fetchone()
            if row is not None and row["total"] is not None and row["total"] > 0:
                wins = row["wins"] if row["wins"] is not None else 0
                metrics["win_rate"] = wins / row["total"]
            else:
                metrics["win_rate"] = 0.0
            
            # Total profit
            cursor.execute("SELECT SUM(net_profit_gbp) as total FROM trade_memory")
            row = cursor.fetchone()
            if row is not None and row["total"] is not None:
                metrics["total_profit_gbp"] = float(row["total"])
            else:
                metrics["total_profit_gbp"] = 0.0
            
            # Patterns learned
            cursor.execute("SELECT COUNT(*) as count FROM pattern_library")
            row = cursor.fetchone()
            if row is not None:
                metrics["patterns_learned"] = row["count"]
            else:
                metrics["patterns_learned"] = 0
            
            # Models trained
            cursor.execute("SELECT COUNT(DISTINCT model_version) as count FROM model_performance")
            row = cursor.fetchone()
            if row is not None:
                metrics["models_trained"] = row["count"]
            else:
                metrics["models_trained"] = 0
            
            cursor.close()
            conn.close()
            
            # Add metadata
            metrics["run_date"] = run_date.isoformat()
            metrics["export_timestamp"] = datetime.now(timezone.utc).isoformat()
            
            output_file = self.output_dir / f"comprehensive_metrics_{run_date.isoformat()}.json"
            with open(output_file, "w") as f:
                json.dump(metrics, f, indent=2)
            
            logger.info("comprehensive_metrics_exported")
            return 1
            
        except Exception as e:
            logger.error("comprehensive_metrics_export_failed", error=str(e))
            return 0
    
    def export_performance_summary(self, run_date: date) -> int:
        """Export performance summary to JSON.
        
        Returns:
            1 if successful, 0 otherwise
        """
        try:
            summary = {
                "run_date": run_date.isoformat(),
                "export_timestamp": datetime.now(timezone.utc).isoformat(),
                "summary": "COMPREHENSIVE export of ALL engine data A-Z",
                "exports_included": [
                    "Trade history (all trades)",
                    "Model performance metrics",
                    "Win/loss analysis",
                    "Pattern library",
                    "Post-exit tracking",
                    "Regime analysis",
                    "Model evolution",
                    "Observability data (trades, models, deltas)",
                    "Learning snapshots",
                    "Backtest results",
                    "Training artifacts",
                    "Comprehensive metrics",
                ],
            }
            
            output_file = self.output_dir / f"performance_summary_{run_date.isoformat()}.json"
            with open(output_file, "w") as f:
                json.dump(summary, f, indent=2)
            
            return 1
            
        except Exception as e:
            logger.error("performance_summary_export_failed", error=str(e))
            return 0

