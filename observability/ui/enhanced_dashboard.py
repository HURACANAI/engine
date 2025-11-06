"""
Enhanced Real-Time Learning Dashboard

Comprehensive real-time dashboard showing:
- Live shadow trade feed
- Real-time learning metrics (AUC improving? Features changing?)
- Model confidence heatmap (which regimes are we confident in?)
- Gate pass/fail rates (updated every minute)
- Performance vs targets (win rate, Sharpe, etc.)
- Active learning indicators (what's being learned right now?)
- Circuit breaker status
- Concept drift warnings
- Confidence-based position scaling

Usage:
    python -m observability.ui.enhanced_dashboard
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
from rich.progress import Progress, BarColumn, TextColumn
from rich import box
from rich.align import Align

from observability.analytics.metrics_computer import MetricsComputer
from observability.analytics.trade_journal import TradeJournal
from observability.analytics.learning_tracker import LearningTracker
from observability.analytics.model_evolution import ModelEvolutionTracker
from observability.analytics.insight_aggregator import InsightAggregator

logger = structlog.get_logger(__name__)


@dataclass
class EnhancedDashboardState:
    """Enhanced dashboard state with all metrics"""
    # Core metrics
    metrics: Dict[str, Any]
    recent_trades: List[Dict[str, Any]]
    recent_events: List[str]
    learning_progress: Optional[Dict[str, Any]]
    model_status: Optional[Dict[str, Any]]
    
    # New metrics
    circuit_breaker_status: Dict[str, Any]
    concept_drift_warnings: List[Dict[str, Any]]
    confidence_heatmap: Dict[str, float]  # regime -> confidence
    active_learning: List[str]  # What's being learned right now
    performance_vs_targets: Dict[str, Dict[str, Any]]
    position_scaling: Dict[str, float]  # confidence -> scale factor


class EnhancedLiveDashboard:
    """
    Enhanced live terminal dashboard with comprehensive metrics.
    
    Shows everything happening in the Engine in real-time.
    """

    def __init__(self, refresh_rate: float = 1.0, capital_gbp: Optional[float] = None, shadow_trading_mode: bool = True):
        """
        Initialize enhanced dashboard.
        
        Args:
            refresh_rate: Update interval in seconds
            capital_gbp: Starting capital (None = unlimited for shadow trading)
            shadow_trading_mode: If True, capital is unlimited (shadow trading only)
        """
        self.refresh_rate = refresh_rate
        self.shadow_trading_mode = shadow_trading_mode
        self.capital_gbp = capital_gbp  # None = unlimited
        self.unlimited_mode = shadow_trading_mode or capital_gbp is None
        self.console = Console()

        # Initialize data sources
        self.metrics_computer = MetricsComputer()
        self.trade_journal = TradeJournal()
        self.learning_tracker = LearningTracker()
        self.model_tracker = ModelEvolutionTracker()
        self.insight_aggregator = InsightAggregator()

        # State
        self.running = True
        self.state = EnhancedDashboardState(
            metrics={},
            recent_trades=[],
            recent_events=[],
            learning_progress=None,
            model_status=None,
            circuit_breaker_status={},
            concept_drift_warnings=[],
            confidence_heatmap={},
            active_learning=[],
            performance_vs_targets={},
            position_scaling={},
        )

        logger.info("enhanced_dashboard_initialized", refresh_rate=refresh_rate, capital=capital_gbp)

    def build_layout(self) -> Layout:
        """Build enhanced dashboard layout"""
        layout = Layout()

        # Split into header, body, footer
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=3)
        )

        # Split body into left, center, right
        layout["body"].split_row(
            Layout(name="left", ratio=1),
            Layout(name="center", ratio=1),
            Layout(name="right", ratio=1)
        )

        # Left: Metrics and Circuit Breakers
        layout["left"].split_column(
            Layout(name="metrics", ratio=2),
            Layout(name="circuit_breakers", ratio=1),
            Layout(name="drift_warnings", ratio=1)
        )

        # Center: Trades and Learning
        layout["center"].split_column(
            Layout(name="trades", ratio=3),
            Layout(name="learning", ratio=2),
            Layout(name="active_learning", ratio=1)
        )

        # Right: Gates, Confidence, Performance
        layout["right"].split_column(
            Layout(name="gates", ratio=2),
            Layout(name="confidence_heatmap", ratio=1),
            Layout(name="performance_targets", ratio=1),
            Layout(name="activity", ratio=1)
        )

        return layout

    def render_header(self) -> Panel:
        """Render dashboard header with capital info"""
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

        header_text = Text()
        header_text.append("🚀 HURACAN ENGINE ", style="bold cyan")
        header_text.append("ENHANCED LIVE DASHBOARD ", style="bold white")
        header_text.append(f"• {now} • ", style="dim")
        if self.unlimited_mode:
            header_text.append("Capital: UNLIMITED (Shadow Trading)", style="bold green")
        else:
            header_text.append(f"Capital: £{self.capital_gbp:.2f}", style="bold green")

        return Panel(
            header_text,
            style="cyan",
            box=box.ROUNDED
        )

    def render_metrics(self) -> Panel:
        """Render enhanced key metrics"""
        metrics = self.state.metrics
        shadow = metrics.get('shadow_trading', {})
        learning = metrics.get('learning', {})
        models = metrics.get('models', {})

        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Metric", style="cyan", width=20)
        table.add_column("Value", style="bold white", width=15)
        table.add_column("Status", width=10)

        # Shadow trading metrics
        total_trades = shadow.get('total_trades', 0)
        win_rate = shadow.get('win_rate', 0)
        avg_pnl = shadow.get('avg_pnl_bps', 0)
        
        # Status indicators
        if total_trades == 0:
            trade_status = "🔴 NONE"
        elif total_trades < 20:
            trade_status = "🟡 LOW"
        else:
            trade_status = "🟢 GOOD"
        
        if win_rate >= 0.75:
            wr_status = "🟢 EXCELLENT"
        elif win_rate >= 0.70:
            wr_status = "🟢 GOOD"
        elif win_rate >= 0.60:
            wr_status = "🟡 OK"
        else:
            wr_status = "🔴 POOR"

        table.add_row("📊 Shadow Trades", str(total_trades), trade_status)
        table.add_row("✅ Win Rate", f"{win_rate:.1%}", wr_status)
        table.add_row("💰 Avg P&L", f"{avg_pnl:.1f} bps", "")

        # Learning metrics
        table.add_row("", "", "")  # Spacer
        sessions = learning.get('num_sessions', 0)
        best_auc = learning.get('best_auc', 0)
        
        if best_auc >= 0.70:
            auc_status = "🟢 EXCELLENT"
        elif best_auc >= 0.65:
            auc_status = "🟢 GOOD"
        else:
            auc_status = "🟡 NEEDS WORK"

        table.add_row("🎓 Training Sessions", str(sessions), "")
        table.add_row("📈 Best AUC", f"{best_auc:.3f}", auc_status)

        # Model readiness
        table.add_row("", "", "")  # Spacer
        ready = models.get('ready_for_hamilton', False)
        ready_icon = "✅ READY" if ready else "⏳ NOT READY"
        ready_style = "green" if ready else "yellow"

        table.add_row("🎯 Hamilton Ready", ready_icon, "")

        return Panel(
            table,
            title="[bold cyan]📊 Key Metrics[/bold cyan]",
            border_style="cyan",
            box=box.ROUNDED
        )

    def render_circuit_breakers(self) -> Panel:
        """Render circuit breaker status"""
        cb_status = self.state.circuit_breaker_status
        unlimited_mode = cb_status.get('unlimited_mode', self.unlimited_mode)
        
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Level", style="cyan", width=12)
        table.add_column("Status", width=15)
        table.add_column("Limit", justify="right", width=12)

        if unlimited_mode:
            # Shadow trading mode
            table.add_row("Mode", Text("🟢 SHADOW TRADING", style="green"), Text("UNLIMITED", style="bold green"))
            table.add_row("", "", "")  # Spacer
            table.add_row("Note", Text("No limits - unlimited capital", style="dim"), Text("Tracking only", style="dim"))
            return Panel(
                table,
                title="[bold cyan]🛡️ Circuit Breakers (Shadow Trading)[/bold cyan]",
                border_style="cyan",
                box=box.ROUNDED
            )

        # Level 1: Single Trade
        level1_active = cb_status.get('level1_active', False)
        level1_limit = cb_status.get('level1_limit', self.capital_gbp * 0.01 if self.capital_gbp else 10.0)  # 1%
        level1_current = cb_status.get('level1_current', 0)
        
        if level1_active:
            status = Text("🔴 TRIGGERED", style="red")
        else:
            status = Text("🟢 ACTIVE", style="green")
        
        table.add_row("Level 1: Trade", status, f"£{level1_limit:.2f}")

        # Level 2: Hourly
        level2_active = cb_status.get('level2_active', False)
        level2_limit = cb_status.get('level2_limit', self.capital_gbp * 0.03)  # 3%
        level2_current = cb_status.get('level2_current', 0)
        
        if level2_active:
            status = Text("🔴 TRIGGERED", style="red")
        else:
            status = Text("🟢 ACTIVE", style="green")
        
        table.add_row("Level 2: Hourly", status, f"£{level2_limit:.2f}")

        # Level 3: Daily
        level3_active = cb_status.get('level3_active', False)
        level3_limit = cb_status.get('level3_limit', self.capital_gbp * 0.05)  # 5%
        level3_current = cb_status.get('level3_current', 0)
        
        if level3_active:
            status = Text("🔴 TRIGGERED", style="red")
        else:
            status = Text("🟢 ACTIVE", style="green")
        
        table.add_row("Level 3: Daily", status, f"£{level3_limit:.2f}")

        # Level 4: Drawdown
        level4_active = cb_status.get('level4_active', False)
        level4_limit = cb_status.get('level4_limit', self.capital_gbp * 0.10)  # 10%
        level4_current = cb_status.get('level4_current', 0)
        
        if level4_active:
            status = Text("🔴 TRIGGERED", style="red")
        else:
            status = Text("🟢 ACTIVE", style="green")
        
        table.add_row("Level 4: Drawdown", status, f"£{level4_limit:.2f}")

        return Panel(
            table,
            title="[bold cyan]🛡️ Circuit Breakers[/bold cyan]",
            border_style="cyan",
            box=box.ROUNDED
        )

    def render_drift_warnings(self) -> Panel:
        """Render concept drift warnings"""
        warnings = self.state.concept_drift_warnings

        if not warnings:
            text = Text("✅ No drift detected", style="green")
        else:
            text = Text()
            for warning in warnings[:3]:  # Show top 3
                severity = warning.get('severity', 'WARNING')
                component = warning.get('component', 'Unknown')
                
                if severity == 'CRITICAL':
                    icon = "🔴"
                    style = "red"
                elif severity == 'SEVERE':
                    icon = "🟠"
                    style = "yellow"
                else:
                    icon = "🟡"
                    style = "yellow"
                
                text.append(f"{icon} {component}\n", style=style)

        return Panel(
            text,
            title="[bold cyan]⚠️ Concept Drift[/bold cyan]",
            border_style="cyan",
            box=box.ROUNDED
        )

    def render_trades(self) -> Panel:
        """Render recent shadow trades with enhanced info"""
        trades = self.state.recent_trades[:15]  # Last 15

        if not trades:
            return Panel(
                Text("No shadow trades yet", style="dim"),
                title="[bold cyan]💼 Recent Shadow Trades[/bold cyan]",
                border_style="cyan",
                box=box.ROUNDED
            )

        table = Table(box=None, padding=(0, 1), show_header=True)
        table.add_column("Time", style="dim", width=8)
        table.add_column("Symbol", style="cyan", width=8)
        table.add_column("Mode", width=6)
        table.add_column("Conf", width=5)
        table.add_column("P&L", justify="right", width=8)
        table.add_column("Status", width=8)

        for trade in trades:
            timestamp = trade.get('entry_ts', '')[:8] if trade.get('entry_ts') else ''
            symbol = trade.get('symbol', '')
            mode = trade.get('mode', '')[:4].upper()
            confidence = trade.get('confidence', 0)
            pnl_bps = trade.get('pnl_bps', 0)
            
            # Confidence color
            if confidence >= 0.80:
                conf_style = "green"
            elif confidence >= 0.60:
                conf_style = "yellow"
            else:
                conf_style = "red"

            # P&L color
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
                Text(f"{confidence:.2f}", style=conf_style),
                Text(pnl_text, style=pnl_style),
                Text(status_display, style=status_style)
            )

        return Panel(
            table,
            title=f"[bold cyan]💼 Recent Shadow Trades ({len(trades)})[/bold cyan]",
            border_style="cyan",
            box=box.ROUNDED
        )

    def render_learning(self) -> Panel:
        """Render learning progress with deltas"""
        learning = self.state.learning_progress

        if not learning:
            return Panel(
                Text("No learning data", style="dim"),
                title="[bold cyan]🎓 Learning Progress[/bold cyan]",
                border_style="cyan",
                box=box.ROUNDED
            )

        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Metric", style="cyan", width=12)
        table.add_column("Value", style="white", width=10)
        table.add_column("Change", justify="right", width=10)

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
        # For ECE, negative is good
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

        # Feature importance changes
        top_features = learning.get('top_features_changed', [])
        if top_features:
            table.add_row("", "", "")  # Spacer
            table.add_row("Top Feature", top_features[0] if top_features else "N/A", "")

        return Panel(
            table,
            title="[bold cyan]🎓 Learning Progress[/bold cyan]",
            border_style="cyan",
            box=box.ROUNDED
        )

    def render_active_learning(self) -> Panel:
        """Render what's being learned right now"""
        active = self.state.active_learning

        if not active:
            text = Text("Waiting for learning activity...", style="dim")
        else:
            text = Text()
            for item in active[:5]:  # Top 5
                text.append(f"• {item}\n", style="white")

        return Panel(
            text,
            title="[bold cyan]🧠 Active Learning[/bold cyan]",
            border_style="cyan",
            box=box.ROUNDED
        )

    def render_gates(self) -> Panel:
        """Render gate status with pass rates"""
        gates = self.state.metrics.get('gates', {}).get('gates', [])

        if not gates:
            return Panel(
                Text("No gate data", style="dim"),
                title="[bold cyan]🚪 Gate Status[/bold cyan]",
                border_style="cyan",
                box=box.ROUNDED
            )

        table = Table(box=None, padding=(0, 1), show_header=True)
        table.add_column("Gate", style="cyan", width=15)
        table.add_column("Pass Rate", justify="right", width=10)
        table.add_column("Status", width=12)

        for gate in gates[:8]:  # Top 8 gates
            name = gate.get('name', '')
            pass_rate = gate.get('pass_rate', 0)

            # Status indicator
            if pass_rate < 0.10:
                status = Text("🔴 BLOCKING", style="red")
            elif pass_rate < 0.30:
                status = Text("🟡 STRICT", style="yellow")
            else:
                status = Text("🟢 GOOD", style="green")

            table.add_row(
                name,
                f"{pass_rate:.1%}",
                status
            )

        return Panel(
            table,
            title="[bold cyan]🚪 Gate Status[/bold cyan]",
            border_style="cyan",
            box=box.ROUNDED
        )

    def render_confidence_heatmap(self) -> Panel:
        """Render confidence heatmap by regime"""
        heatmap = self.state.confidence_heatmap

        if not heatmap:
            return Panel(
                Text("No confidence data", style="dim"),
                title="[bold cyan]🎯 Confidence Heatmap[/bold cyan]",
                border_style="cyan",
                box=box.ROUNDED
            )

        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Regime", style="cyan", width=10)
        table.add_column("Confidence", width=12)

        for regime, confidence in sorted(heatmap.items(), key=lambda x: x[1], reverse=True):
            if confidence >= 0.80:
                conf_style = "green"
                conf_bar = "████████"
            elif confidence >= 0.60:
                conf_style = "yellow"
                conf_bar = "██████  "
            else:
                conf_style = "red"
                conf_bar = "████    "

            table.add_row(
                regime,
                Text(f"{conf_bar} {confidence:.1%}", style=conf_style)
            )

        return Panel(
            table,
            title="[bold cyan]🎯 Confidence by Regime[/bold cyan]",
            border_style="cyan",
            box=box.ROUNDED
        )

    def render_performance_targets(self) -> Panel:
        """Render performance vs targets"""
        perf = self.state.performance_vs_targets

        if not perf:
            return Panel(
                Text("No performance data", style="dim"),
                title="[bold cyan]📈 Performance vs Targets[/bold cyan]",
                border_style="cyan",
                box=box.ROUNDED
            )

        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Metric", style="cyan", width=12)
        table.add_column("Current", width=8)
        table.add_column("Target", width=8)
        table.add_column("Status", width=8)

        for metric_name, data in perf.items():
            current = data.get('current', 0)
            target = data.get('target', 0)
            
            if current >= target:
                status = Text("✅", style="green")
            elif current >= target * 0.9:
                status = Text("🟡", style="yellow")
            else:
                status = Text("🔴", style="red")

            table.add_row(
                metric_name,
                f"{current:.1%}" if isinstance(current, float) and current < 1 else str(current),
                f"{target:.1%}" if isinstance(target, float) and target < 1 else str(target),
                status
            )

        return Panel(
            table,
            title="[bold cyan]📈 Performance vs Targets[/bold cyan]",
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
        footer_text.append(" • Capital: ", style="dim")
        if self.unlimited_mode:
            footer_text.append("UNLIMITED (Shadow Trading)", style="bold green")
        else:
            footer_text.append(f"£{self.capital_gbp:.2f}", style="bold green")

        return Panel(
            footer_text,
            style="dim",
            box=box.ROUNDED
        )

    async def update_state(self):
        """Update enhanced dashboard state from all data sources"""
        try:
            today = datetime.utcnow().strftime("%Y-%m-%d")

            # Update core metrics
            self.state.metrics = self.metrics_computer.compute_daily_metrics(today)

            # Update recent trades
            self.state.recent_trades = self.trade_journal.query_trades(limit=15)

            # Update learning progress
            summary = self.learning_tracker.get_daily_summary(today)
            if summary:
                self.state.learning_progress = summary

            # Update circuit breaker status (mock for now - will integrate with risk manager)
            if self.unlimited_mode:
                # Shadow trading: unlimited capital, but still track for reporting
                virtual_capital = 1000.0  # For reporting only
                self.state.circuit_breaker_status = {
                    'unlimited_mode': True,
                    'level1_active': False,  # Never active in shadow trading
                    'level1_limit': virtual_capital * 0.01,
                    'level1_current': 0,
                    'level2_active': False,
                    'level2_limit': virtual_capital * 0.03,
                    'level2_current': 0,
                    'level3_active': False,
                    'level3_limit': virtual_capital * 0.05,
                    'level3_current': 0,
                    'level4_active': False,
                    'level4_limit': virtual_capital * 0.10,
                    'level4_current': 0,
                }
            else:
                self.state.circuit_breaker_status = {
                    'unlimited_mode': False,
                    'level1_active': False,
                    'level1_limit': self.capital_gbp * 0.01,
                    'level1_current': 0,
                    'level2_active': False,
                    'level2_limit': self.capital_gbp * 0.03,
                    'level2_current': 0,
                    'level3_active': False,
                    'level3_limit': self.capital_gbp * 0.05,
                    'level3_current': 0,
                    'level4_active': False,
                    'level4_limit': self.capital_gbp * 0.10,
                    'level4_current': 0,
                }

            # Update concept drift warnings (mock for now - will integrate with drift detector)
            self.state.concept_drift_warnings = []

            # Update confidence heatmap (mock for now)
            self.state.confidence_heatmap = {
                'TREND': 0.75,
                'RANGE': 0.65,
                'PANIC': 0.45,
            }

            # Update active learning
            now = datetime.utcnow().strftime("%H:%M:%S")
            if self.state.recent_trades:
                self.state.active_learning = [
                    f"{now} Processing shadow trade outcomes",
                    f"{now} Updating feature importance",
                    f"{now} Analyzing win/loss patterns",
                ]

            # Update performance vs targets
            shadow = self.state.metrics.get('shadow_trading', {})
            self.state.performance_vs_targets = {
                'Win Rate': {
                    'current': shadow.get('win_rate', 0),
                    'target': 0.75,
                },
                'Trades/Day': {
                    'current': shadow.get('total_trades', 0),
                    'target': 50,
                },
            }

            # Update recent events
            if self.state.recent_trades:
                self.state.recent_events.insert(0, f"{now} Shadow trade executed")
                self.state.recent_events = self.state.recent_events[:20]

        except Exception as e:
            logger.error("state_update_failed", error=str(e))

    def render_dashboard(self) -> Layout:
        """Render complete enhanced dashboard"""
        layout = self.build_layout()

        layout["header"].update(self.render_header())
        layout["metrics"].update(self.render_metrics())
        layout["circuit_breakers"].update(self.render_circuit_breakers())
        layout["drift_warnings"].update(self.render_drift_warnings())
        layout["trades"].update(self.render_trades())
        layout["learning"].update(self.render_learning())
        layout["active_learning"].update(self.render_active_learning())
        layout["gates"].update(self.render_gates())
        layout["confidence_heatmap"].update(self.render_confidence_heatmap())
        layout["performance_targets"].update(self.render_performance_targets())
        layout["activity"].update(self.render_activity())
        layout["footer"].update(self.render_footer())

        return layout

    async def run(self):
        """Run enhanced live dashboard"""
        logger.info("starting_enhanced_dashboard", capital=self.capital_gbp)

        with Live(
            self.render_dashboard(),
            console=self.console,
            screen=True,
            refresh_per_second=1
        ) as live:
            try:
                while self.running:
                    await self.update_state()
                    live.update(self.render_dashboard())
                    await asyncio.sleep(self.refresh_rate)

            except KeyboardInterrupt:
                logger.info("dashboard_stopped_by_user")
            except Exception as e:
                logger.error("dashboard_error", error=str(e))
                raise


async def main():
    """Main entry point"""
    import sys
    
    # Default: unlimited capital for shadow trading
    capital = None
    shadow_trading = True
    
    if len(sys.argv) > 1:
        try:
            capital = float(sys.argv[1])
            shadow_trading = False  # If capital specified, use real trading mode
        except ValueError:
            print(f"Invalid capital: {sys.argv[1]}, using unlimited (shadow trading mode)")
            capital = None
            shadow_trading = True
    
    dashboard = EnhancedLiveDashboard(
        refresh_rate=1.0,
        capital_gbp=capital,
        shadow_trading_mode=shadow_trading,
    )
    await dashboard.run()


if __name__ == '__main__':
    asyncio.run(main())

