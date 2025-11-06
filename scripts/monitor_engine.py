#!/usr/bin/env python3
"""
Engine Monitoring CLI

Real-time monitoring of:
- Training progress (which coins, download status, completion)
- Shadow trades count and performance
- Engine status (all 23 alpha engines)
- System health
- Download progress

Usage:
    python scripts/monitor_engine.py [--watch] [--interval 5] [--json]
    
Commands:
    python scripts/monitor_engine.py status          # One-time status check
    python scripts/monitor_engine.py --watch         # Watch mode (updates every 5s)
    python scripts/monitor_engine.py shadow-trades  # Show shadow trades count
    python scripts/monitor_engine.py engines        # Show all engine status
    python scripts/monitor_engine.py training       # Show training progress
"""

import argparse
import json
import sqlite3
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.live import Live
    from rich.layout import Layout
    from rich.text import Text
    from rich.progress import Progress, BarColumn, TextColumn
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.cloud.training.config.settings import EngineSettings
from src.cloud.training.monitoring.system_status import SystemStatusReporter


class EngineMonitor:
    """Monitor engine training progress, shadow trades, and engine status."""
    
    def __init__(self, dsn: Optional[str] = None):
        """Initialize monitor."""
        self.console = Console() if RICH_AVAILABLE else None
        
        # Load settings
        try:
            settings = EngineSettings()
            self.dsn = dsn or settings.database.dsn
        except Exception:
            self.dsn = dsn or "postgresql://localhost/huracan"
        
        # Database paths
        self.journal_db = project_root / "observability" / "data" / "sqlite" / "journal.db"
        
        # System status reporter
        if PSYCOPG2_AVAILABLE:
            try:
                self.status_reporter = SystemStatusReporter(self.dsn)
                self.status_reporter.connect()
            except Exception:
                self.status_reporter = None
        else:
            self.status_reporter = None
    
    def get_shadow_trades_count(self) -> Dict[str, Any]:
        """Get shadow trades count and stats."""
        if not self.journal_db.exists():
            return {"error": "journal.db not found", "total": 0}
        
        try:
            conn = sqlite3.connect(str(self.journal_db))
            conn.row_factory = sqlite3.Row
            
            # Total count
            total = conn.execute("SELECT COUNT(*) as count FROM shadow_trades").fetchone()["count"]
            
            # Today's count
            today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
            today = conn.execute(
                "SELECT COUNT(*) as count FROM shadow_trades WHERE ts >= ?",
                (today_start,)
            ).fetchone()["count"]
            
            # By mode
            by_mode = conn.execute("""
                SELECT mode, COUNT(*) as count, 
                       AVG(CASE WHEN shadow_pnl_bps > 0 THEN 1.0 ELSE 0.0 END) as win_rate,
                       AVG(shadow_pnl_bps) as avg_pnl_bps
                FROM shadow_trades
                GROUP BY mode
            """).fetchall()
            
            # Recent trades (last 24h)
            yesterday = (datetime.now() - timedelta(days=1)).isoformat()
            recent = conn.execute(
                "SELECT COUNT(*) as count FROM shadow_trades WHERE ts >= ?",
                (yesterday,)
            ).fetchone()["count"]
            
            conn.close()
            
            return {
                "total": total,
                "today": today,
                "recent_24h": recent,
                "by_mode": {row["mode"]: {
                    "count": row["count"],
                    "win_rate": row["win_rate"],
                    "avg_pnl_bps": row["avg_pnl_bps"]
                } for row in by_mode}
            }
        except Exception as e:
            return {"error": str(e), "total": 0}
    
    def get_training_progress(self) -> Dict[str, Any]:
        """Get current training progress."""
        # Check for training log files
        log_dir = project_root / "logs"
        if log_dir.exists():
            log_files = list(log_dir.glob("*.log"))
            if log_files:
                latest_log = max(log_files, key=lambda p: p.stat().st_mtime)
                # Try to parse recent log entries
                try:
                    with open(latest_log, "r") as f:
                        lines = f.readlines()
                        # Get last 50 lines
                        recent_lines = lines[-50:] if len(lines) > 50 else lines
                        # Look for training progress indicators
                        training_indicators = [
                            line for line in recent_lines
                            if any(keyword in line.lower() for keyword in [
                                "training", "downloading", "completed", "symbol=", "batch"
                            ])
                        ]
                        return {
                            "log_file": str(latest_log),
                            "recent_activity": training_indicators[-10:] if training_indicators else [],
                            "status": "active" if training_indicators else "unknown"
                        }
                except Exception:
                    pass
        
        return {"status": "unknown", "message": "No training logs found"}
    
    def get_engine_status(self) -> Dict[str, Any]:
        """Get status of all 23 alpha engines."""
        # This would require access to the running engine instance
        # For now, return placeholder
        engines = [
            "Trend", "Range", "Breakout", "Tape", "Leader", "Sweep",
            "Scalper/Latency-Arb", "Volatility", "Correlation/Cluster",
            "Funding/Carry", "Arbitrage", "Adaptive/Meta-Learning",
            "Evolutionary/Auto-Discovery", "Risk", "Flow-Prediction",
            "Cross-Venue Latency", "Market-Maker/Inventory", "Anomaly-Detection",
            "Regime-Classifier", "Momentum Reversal", "Divergence",
            "Support/Resistance Bounce", "Meta-Label"
        ]
        
        return {
            "total_engines": len(engines),
            "engines": {name: {"status": "unknown"} for name in engines},
            "message": "Engine status requires running instance"
        }
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get system health status."""
        if not self.status_reporter:
            return {"status": "unknown", "message": "Database connection not available"}
        
        try:
            report = self.status_reporter.get_health_report()
            return {
                "overall_status": report.overall_status,
                "database": {
                    "healthy": report.database_status.healthy,
                    "issues": report.database_status.issues
                },
                "services": [
                    {
                        "name": s.name,
                        "enabled": s.enabled,
                        "running": s.running,
                        "healthy": s.healthy
                    }
                    for s in report.services
                ],
                "resource_usage": report.resource_usage,
                "active_features": report.active_features
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def print_status(self, json_output: bool = False):
        """Print comprehensive status."""
        shadow_trades = self.get_shadow_trades_count()
        training = self.get_training_progress()
        engines = self.get_engine_status()
        health = self.get_system_health()
        
        if json_output:
            print(json.dumps({
                "shadow_trades": shadow_trades,
                "training": training,
                "engines": engines,
                "health": health,
                "timestamp": datetime.now().isoformat()
            }, indent=2))
            return
        
        if not RICH_AVAILABLE:
            # Fallback to simple text output
            print("=" * 70)
            print("ENGINE MONITORING STATUS")
            print("=" * 70)
            print(f"\nShadow Trades: {shadow_trades.get('total', 0)} total, {shadow_trades.get('today', 0)} today")
            print(f"Training Status: {training.get('status', 'unknown')}")
            print(f"Engines: {engines.get('total_engines', 0)} total")
            print(f"System Health: {health.get('overall_status', 'unknown')}")
            return
        
        # Rich output
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=2)
        )
        layout["body"].split_row(
            Layout(name="left"),
            Layout(name="right")
        )
        
        # Header
        layout["header"].update(Panel(
            Text("🚀 ENGINE MONITORING DASHBOARD", style="bold cyan", justify="center"),
            border_style="cyan"
        ))
        
        # Shadow Trades
        shadow_table = Table(title="📊 Shadow Trades", show_header=True, header_style="bold magenta")
        shadow_table.add_column("Metric", style="cyan")
        shadow_table.add_column("Value", style="green")
        
        if "error" not in shadow_trades:
            shadow_table.add_row("Total", str(shadow_trades.get("total", 0)))
            shadow_table.add_row("Today", str(shadow_trades.get("today", 0)))
            shadow_table.add_row("Last 24h", str(shadow_trades.get("recent_24h", 0)))
            if "by_mode" in shadow_trades:
                for mode, stats in shadow_trades["by_mode"].items():
                    shadow_table.add_row(
                        f"{mode.title()} Mode",
                        f"{stats['count']} trades, {stats['win_rate']:.1%} win rate"
                    )
        else:
            shadow_table.add_row("Error", shadow_trades["error"])
        
        layout["left"].update(shadow_table)
        
        # Training Progress
        training_table = Table(title="🤖 Training Progress", show_header=True, header_style="bold yellow")
        training_table.add_column("Metric", style="cyan")
        training_table.add_column("Value", style="green")
        
        training_table.add_row("Status", training.get("status", "unknown"))
        if "log_file" in training:
            training_table.add_row("Log File", Path(training["log_file"]).name)
        if "recent_activity" in training and training["recent_activity"]:
            training_table.add_row("Recent Activity", f"{len(training['recent_activity'])} events")
        
        layout["right"].update(training_table)
        
        # Footer
        health_status = health.get("overall_status", "unknown")
        health_emoji = "✅" if health_status == "HEALTHY" else "⚠️" if health_status == "DEGRADED" else "❌"
        layout["footer"].update(Panel(
            f"{health_emoji} System Health: {health_status} | "
            f"Engines: {engines.get('total_engines', 0)} | "
            f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            border_style="blue"
        ))
        
        self.console.print(layout)
    
    def watch_mode(self, interval: int = 5):
        """Watch mode - continuously update status."""
        if not RICH_AVAILABLE:
            print("Rich library not available. Install with: pip install rich")
            return
        
        def generate_layout():
            self.print_status()
            return Panel("Press Ctrl+C to exit", border_style="dim")
        
        with Live(generate_layout(), refresh_per_second=1/interval, screen=True) as live:
            try:
                while True:
                    time.sleep(interval)
                    live.update(generate_layout())
            except KeyboardInterrupt:
                self.console.print("\n[bold yellow]Monitoring stopped[/bold yellow]")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Engine Monitoring CLI")
    parser.add_argument("command", nargs="?", default="status", choices=["status", "shadow-trades", "engines", "training"])
    parser.add_argument("--watch", action="store_true", help="Watch mode (continuous updates)")
    parser.add_argument("--interval", type=int, default=5, help="Update interval in seconds (watch mode)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--dsn", help="Database DSN (overrides config)")
    
    args = parser.parse_args()
    
    monitor = EngineMonitor(dsn=args.dsn)
    
    if args.watch:
        monitor.watch_mode(interval=args.interval)
    elif args.command == "status":
        monitor.print_status(json_output=args.json)
    elif args.command == "shadow-trades":
        shadow = monitor.get_shadow_trades_count()
        if args.json:
            print(json.dumps(shadow, indent=2))
        else:
            if RICH_AVAILABLE:
                table = Table(title="Shadow Trades", show_header=True)
                table.add_column("Metric", style="cyan")
                table.add_column("Value", style="green")
                if "error" not in shadow:
                    table.add_row("Total", str(shadow.get("total", 0)))
                    table.add_row("Today", str(shadow.get("today", 0)))
                    table.add_row("Last 24h", str(shadow.get("recent_24h", 0)))
                else:
                    table.add_row("Error", shadow["error"])
                Console().print(table)
            else:
                print(json.dumps(shadow, indent=2))
    elif args.command == "engines":
        engines = monitor.get_engine_status()
        if args.json:
            print(json.dumps(engines, indent=2))
        else:
            print(f"Total Engines: {engines.get('total_engines', 0)}")
            print("Engine status requires running instance")
    elif args.command == "training":
        training = monitor.get_training_progress()
        if args.json:
            print(json.dumps(training, indent=2))
        else:
            print(f"Status: {training.get('status', 'unknown')}")
            if "log_file" in training:
                print(f"Log File: {training['log_file']}")


if __name__ == "__main__":
    main()

