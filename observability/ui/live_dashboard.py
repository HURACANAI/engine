"""
Live Dashboard

Real-time terminal UI showing Engine activity.

Features:
- Live metrics (updates every 1s)
- Shadow trade feed
- Gate status
- Model metrics
- Learning progress
- Activity log

Usage:
    python -m observability.ui.live_dashboard
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import structlog

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.progress import Progress, BarColumn, TextColumn, TaskID
from rich import box

from observability.analytics.metrics_computer import MetricsComputer
from observability.analytics.trade_journal import TradeJournal
from observability.analytics.learning_tracker import LearningTracker
from observability.analytics.model_evolution import ModelEvolutionTracker

logger = structlog.get_logger(__name__)


@dataclass
class DashboardState:
    """Current dashboard state"""
    metrics: Dict[str, Any]
    recent_trades: List[Dict[str, Any]]
    recent_events: List[str]
    learning_progress: Optional[Dict[str, Any]]
    model_status: Optional[Dict[str, Any]]


class LiveDashboard:
    """
    Live terminal dashboard for Engine observability.

    Updates every 1 second with latest metrics.
    """

    def __init__(self, refresh_rate: float = 1.0):
        """
        Initialize dashboard.

        Args:
            refresh_rate: Update interval in seconds
        """
        self.refresh_rate = refresh_rate
        self.console = Console()

        # Initialize data sources
        self.metrics_computer = MetricsComputer()
        self.trade_journal = TradeJournal()
        self.learning_tracker = LearningTracker()
        self.model_tracker = ModelEvolutionTracker()

        # State
        self.running = True
        self.state = DashboardState(
            metrics={},
            recent_trades=[],
            recent_events=[],
            learning_progress=None,
            model_status=None
        )

        logger.info("live_dashboard_initialized", refresh_rate=refresh_rate)

    def build_layout(self) -> Layout:
        """Build dashboard layout"""
        layout = Layout()

        # Split into header, body, footer
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=3)
        )

        # Split body into left and right
        layout["body"].split_row(
            Layout(name="left"),
            Layout(name="right")
        )

        # Split left into metrics and trades
        layout["left"].split_column(
            Layout(name="metrics", ratio=2),
            Layout(name="trades", ratio=3)
        )

        # Split right into gates and learning
        layout["right"].split_column(
            Layout(name="gates", ratio=2),
            Layout(name="learning", ratio=2),
            Layout(name="activity", ratio=1)
        )

        return layout

    def render_header(self) -> Panel:
        """Render dashboard header"""
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

        header_text = Text()
        header_text.append("🚀 HURACAN ENGINE ", style="bold cyan")
        header_text.append("LIVE DASHBOARD ", style="bold white")
        header_text.append(f"• {now}", style="dim")

        return Panel(
            header_text,
            style="cyan",
            box=box.ROUNDED
        )

    def render_metrics(self) -> Panel:
        """Render key metrics"""
        metrics = self.state.metrics

        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="bold white")

        # Shadow trading metrics
        shadow = metrics.get('shadow_trading', {})
        table.add_row(
            "📊 Shadow Trades",
            str(shadow.get('total_trades', 0))
        )
        table.add_row(
            "✅ Win Rate",
            f"{shadow.get('win_rate', 0):.1%}"
        )
        table.add_row(
            "💰 Avg P&L",
            f"{shadow.get('avg_pnl_bps', 0):.1f} bps"
        )

        # Learning metrics
        learning = metrics.get('learning', {})
        table.add_row("", "")  # Spacer
        table.add_row(
            "🎓 Training Sessions",
            str(learning.get('num_sessions', 0))
        )
        table.add_row(
            "📈 Best AUC",
            f"{learning.get('best_auc', 0):.3f}"
        )

        # Model readiness
        models = metrics.get('models', {})
        ready = models.get('ready_for_hamilton', False)
        ready_icon = "✅" if ready else "⏳"
        table.add_row("", "")  # Spacer
        table.add_row(
            "🎯 Hamilton Ready",
            f"{ready_icon} {ready}"
        )

        return Panel(
            table,
            title="[bold cyan]📊 Metrics[/bold cyan]",
            border_style="cyan",
            box=box.ROUNDED
        )

    def render_trades(self) -> Panel:
        """Render recent trades"""
        trades = self.state.recent_trades[:10]  # Last 10

        if not trades:
            return Panel(
                Text("No trades yet", style="dim"),
                title="[bold cyan]💼 Recent Shadow Trades[/bold cyan]",
                border_style="cyan",
                box=box.ROUNDED
            )

        table = Table(box=None, padding=(0, 1), show_header=True)
        table.add_column("Time", style="dim", width=8)
        table.add_column("Symbol", style="cyan", width=8)
        table.add_column("Mode", width=6)
        table.add_column("P&L", justify="right")
        table.add_column("Status")

        for trade in trades:
            timestamp = trade.get('entry_ts', '')[:8]  # HH:MM:SS
            symbol = trade.get('symbol', '')
            mode = trade.get('mode', '')[:4].upper()  # SCAL/RUNN

            pnl_bps = trade.get('pnl_bps', 0)
            if pnl_bps > 0:
                pnl_text = f"+{pnl_bps:.1f}"
                pnl_style = "green"
            elif pnl_bps < 0:
                pnl_text = f"{pnl_bps:.1f}"
                pnl_style = "red"
            else:
                pnl_text = "0.0"
                pnl_style = "dim"

            status = trade.get('status', 'open')
            if status == 'closed':
                if pnl_bps > 0:
                    status_display = "✅ WIN"
                    status_style = "green"
                else:
                    status_display = "❌ LOSS"
                    status_style = "red"
            else:
                status_display = "⏳ OPEN"
                status_style = "yellow"

            table.add_row(
                timestamp,
                symbol,
                mode,
                Text(pnl_text, style=pnl_style),
                Text(status_display, style=status_style)
            )

        return Panel(
            table,
            title=f"[bold cyan]💼 Recent Shadow Trades ({len(trades)})[/bold cyan]",
            border_style="cyan",
            box=box.ROUNDED
        )

    def render_gates(self) -> Panel:
        """Render gate status"""
        gates = self.state.metrics.get('gates', [])

        if not gates:
            return Panel(
                Text("No gate data", style="dim"),
                title="[bold cyan]🚪 Gate Status[/bold cyan]",
                border_style="cyan",
                box=box.ROUNDED
            )

        table = Table(box=None, padding=(0, 1), show_header=True)
        table.add_column("Gate", style="cyan")
        table.add_column("Pass Rate", justify="right")
        table.add_column("Status")

        for gate in gates[:6]:  # Top 6 gates
            name = gate.get('name', '')
            pass_rate = gate.get('pass_rate', 0)

            # Status indicator
            if pass_rate < 0.10:
                status = "🔴 BLOCKING"
                status_style = "red"
            elif pass_rate < 0.30:
                status = "🟡 STRICT"
                status_style = "yellow"
            else:
                status = "🟢 GOOD"
                status_style = "green"

            table.add_row(
                name,
                f"{pass_rate:.1%}",
                Text(status, style=status_style)
            )

        return Panel(
            table,
            title="[bold cyan]🚪 Gate Status[/bold cyan]",
            border_style="cyan",
            box=box.ROUNDED
        )

    def render_learning(self) -> Panel:
        """Render learning progress"""
        learning = self.state.learning_progress

        if not learning:
            return Panel(
                Text("No learning data", style="dim"),
                title="[bold cyan]🎓 Learning Progress[/bold cyan]",
                border_style="cyan",
                box=box.ROUNDED
            )

        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")
        table.add_column("Change", justify="right")

        # AUC
        auc = learning.get('auc', 0)
        auc_delta = learning.get('auc_delta', 0)
        delta_text = f"+{auc_delta:.3f}" if auc_delta > 0 else f"{auc_delta:.3f}"
        delta_style = "green" if auc_delta > 0 else "red" if auc_delta < 0 else "dim"

        table.add_row(
            "AUC",
            f"{auc:.3f}",
            Text(delta_text, style=delta_style)
        )

        # ECE
        ece = learning.get('ece', 0)
        ece_delta = learning.get('ece_delta', 0)
        delta_text = f"{ece_delta:+.3f}"
        # For ECE, negative is good (lower calibration error)
        delta_style = "green" if ece_delta < 0 else "red" if ece_delta > 0 else "dim"

        table.add_row(
            "ECE",
            f"{ece:.3f}",
            Text(delta_text, style=delta_style)
        )

        # Samples
        samples = learning.get('samples_processed', 0)
        table.add_row(
            "Samples",
            f"{samples:,}",
            Text("", style="dim")
        )

        return Panel(
            table,
            title="[bold cyan]🎓 Learning Progress[/bold cyan]",
            border_style="cyan",
            box=box.ROUNDED
        )

    def render_activity(self) -> Panel:
        """Render recent activity"""
        events = self.state.recent_events[:5]  # Last 5

        if not events:
            text = Text("Waiting for activity...", style="dim")
        else:
            text = Text()
            for event in events:
                text.append(f"• {event}\n", style="white")

        return Panel(
            text,
            title="[bold cyan]📝 Activity[/bold cyan]",
            border_style="cyan",
            box=box.ROUNDED
        )

    def render_footer(self) -> Panel:
        """Render dashboard footer"""
        footer_text = Text()
        footer_text.append("Press ", style="dim")
        footer_text.append("Ctrl+C", style="bold red")
        footer_text.append(" to exit • Updates every ", style="dim")
        footer_text.append(f"{self.refresh_rate}s", style="bold white")

        return Panel(
            footer_text,
            style="dim",
            box=box.ROUNDED
        )

    async def update_state(self):
        """Update dashboard state from data sources"""
        try:
            # Get today's date
            today = datetime.utcnow().strftime("%Y-%m-%d")

            # Update metrics
            self.state.metrics = self.metrics_computer.compute_daily_metrics(today)

            # Update recent trades
            self.state.recent_trades = self.trade_journal.query_trades(limit=10)

            # Update learning progress
            summary = self.learning_tracker.get_daily_summary(today)
            if summary:
                self.state.learning_progress = summary

            # Update recent events (mock for now)
            now = datetime.utcnow().strftime("%H:%M:%S")
            if self.state.recent_trades:
                self.state.recent_events.insert(0, f"{now} Shadow trade executed")
                self.state.recent_events = self.state.recent_events[:20]

        except Exception as e:
            logger.error("state_update_failed", error=str(e))

    def render_dashboard(self) -> Layout:
        """Render complete dashboard"""
        layout = self.build_layout()

        layout["header"].update(self.render_header())
        layout["metrics"].update(self.render_metrics())
        layout["trades"].update(self.render_trades())
        layout["gates"].update(self.render_gates())
        layout["learning"].update(self.render_learning())
        layout["activity"].update(self.render_activity())
        layout["footer"].update(self.render_footer())

        return layout

    async def run(self):
        """Run live dashboard"""
        logger.info("starting_live_dashboard")

        with Live(
            self.render_dashboard(),
            console=self.console,
            screen=True,
            refresh_per_second=1
        ) as live:
            try:
                while self.running:
                    # Update state
                    await self.update_state()

                    # Re-render
                    live.update(self.render_dashboard())

                    # Wait
                    await asyncio.sleep(self.refresh_rate)

            except KeyboardInterrupt:
                logger.info("dashboard_stopped_by_user")
            except Exception as e:
                logger.error("dashboard_error", error=str(e))
                raise


async def main():
    """Main entry point"""
    dashboard = LiveDashboard(refresh_rate=1.0)
    await dashboard.run()


if __name__ == '__main__':
    asyncio.run(main())
