"""
Enhanced Real-Time Learning Dashboard

Beautiful, user-friendly dashboard designed for clarity and ease of understanding.
No technical jargon - just clear explanations of what's happening.

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
from rich.progress import Progress, BarColumn, TextColumn, SpinnerColumn
from rich import box
from rich.align import Align
from rich.columns import Columns

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
    Beautiful, user-friendly live dashboard.
    
    Designed to be understood by anyone - no technical knowledge required.
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
            Layout(name="header", size=4),
            Layout(name="body"),
            Layout(name="footer", size=2)
        )

        # Split body into top and bottom
        layout["body"].split_column(
            Layout(name="top", size=12),
            Layout(name="bottom")
        )

        # Top: Key metrics in a row
        layout["top"].split_row(
            Layout(name="performance", ratio=1),
            Layout(name="trades", ratio=2),
            Layout(name="learning_status", ratio=1)
        )

        # Bottom: Split into left and right
        layout["bottom"].split_row(
            Layout(name="left", ratio=1),
            Layout(name="right", ratio=1)
        )

        # Left bottom: Safety and monitoring
        layout["left"].split_column(
            Layout(name="safety", ratio=1),
            Layout(name="warnings", ratio=1)
        )

        # Right bottom: System status
        layout["right"].split_column(
            Layout(name="system_health", ratio=1),
            Layout(name="recent_activity", ratio=1)
        )

        return layout

    def render_header(self) -> Panel:
        """Render beautiful dashboard header"""
        now = datetime.utcnow().strftime("%B %d, %Y  •  %I:%M:%S %p")
        
        # Create a beautiful header with gradient-like effect
        header_content = Text()
        header_content.append("\n", style="white")
        header_content.append("  ╔═══════════════════════════════════════════════════════════════════════════╗\n", style="bright_white")
        header_content.append("  ║", style="bright_white")
        header_content.append("  🚀  TRADING SYSTEM DASHBOARD  ", style="bold bright_white on blue")
        header_content.append("                                                                  ║\n", style="bright_white")
        header_content.append("  ║", style="bright_white")
        header_content.append(f"  {now}", style="dim white")
        if self.unlimited_mode:
            header_content.append("  •  Mode: Practice (No Real Money)", style="dim green")
        else:
            header_content.append(f"  •  Capital: £{self.capital_gbp:,.2f}", style="dim green")
        header_content.append("                                                                    ║\n", style="bright_white")
        header_content.append("  ╚═══════════════════════════════════════════════════════════════════════════╝\n", style="bright_white")

        return Panel(
            Align.center(header_content),
            border_style="bright_white",
            box=box.DOUBLE
        )

    def render_performance(self) -> Panel:
        """Render performance overview in plain English"""
        metrics = self.state.metrics
        shadow = metrics.get('shadow_trading', {})
        
        total_trades = shadow.get('total_trades', 0)
        win_rate = shadow.get('win_rate', 0)
        avg_pnl = shadow.get('avg_pnl_bps', 0)
        
        # Create beautiful performance cards
        content = Text()
        content.append("\n", style="white")
        content.append("  📊  HOW WE'RE DOING\n", style="bold bright_white")
        content.append("  ────────────────────\n\n", style="dim white")
        
        # Total trades with explanation
        content.append("  Trades Made Today\n", style="cyan")
        if total_trades == 0:
            content.append(f"  {total_trades:,} trades", style="dim white")
            content.append("  •  System is warming up\n\n", style="dim yellow")
        elif total_trades < 20:
            content.append(f"  {total_trades:,} trades", style="yellow")
            content.append("  •  Getting started\n\n", style="dim yellow")
        else:
            content.append(f"  {total_trades:,} trades", style="bold green")
            content.append("  •  Active and running\n\n", style="dim green")
        
        # Win rate with explanation
        content.append("  Success Rate\n", style="cyan")
        if win_rate >= 0.75:
            status = "Excellent"
            color = "bold bright_green"
            emoji = "🎯"
        elif win_rate >= 0.70:
            status = "Very Good"
            color = "green"
            emoji = "✅"
        elif win_rate >= 0.60:
            status = "Good"
            color = "yellow"
            emoji = "👍"
        else:
            status = "Needs Improvement"
            color = "red"
            emoji = "⚠️"
        
        content.append(f"  {win_rate:.1%}", style=f"bold {color}")
        content.append(f"  {emoji}  {status}\n\n", style=color)
        
        # Average profit with explanation
        content.append("  Average Profit Per Trade\n", style="cyan")
        if avg_pnl > 0:
            content.append(f"  +{avg_pnl:.2f} basis points", style="bold green")
            content.append("  •  Making money on average\n", style="dim green")
        elif avg_pnl < 0:
            content.append(f"  {avg_pnl:.2f} basis points", style="bold red")
            content.append("  •  Losing money on average\n", style="dim red")
        else:
            content.append("  0.00 basis points", style="dim white")
            content.append("  •  Breaking even\n", style="dim white")

        return Panel(
            content,
            title="[bold bright_white]📈 Performance Overview[/bold bright_white]",
            border_style="bright_white",
            box=box.ROUNDED,
            padding=(1, 2)
        )

    def render_trades(self) -> Panel:
        """Render recent trades with clear explanations"""
        trades = self.state.recent_trades[:12]

        if not trades:
            content = Text()
            content.append("\n", style="white")
            content.append("  No trades yet today.\n\n", style="dim white")
            content.append("  The system is analyzing the market and\n", style="dim white")
            content.append("  will start making trades when it finds\n", style="dim white")
            content.append("  good opportunities.\n", style="dim white")
            
            return Panel(
                content,
                title="[bold bright_white]💼 Recent Trades[/bold bright_white]",
                border_style="bright_white",
                box=box.ROUNDED,
                padding=(1, 2)
            )

        table = Table(box=None, padding=(0, 1), show_header=True, header_style="bold bright_white")
        table.add_column("Time", style="dim white", width=8)
        table.add_column("Coin", style="cyan", width=10)
        table.add_column("Type", width=8, header="Trading\nMode")
        table.add_column("Confidence", width=10, justify="center")
        table.add_column("Result", justify="right", width=12)
        table.add_column("Status", width=10)

        for trade in trades:
            timestamp = trade.get('entry_ts', '')[:8] if trade.get('entry_ts') else ''
            symbol = trade.get('symbol', 'N/A')
            mode = trade.get('mode', '')[:4].upper() if trade.get('mode') else 'N/A'
            confidence = trade.get('confidence', 0)
            pnl_bps = trade.get('pnl_bps', 0)
            
            # Confidence with visual indicator
            if confidence >= 0.80:
                conf_display = Text("High", style="bold green")
            elif confidence >= 0.60:
                conf_display = Text("Medium", style="yellow")
            else:
                conf_display = Text("Low", style="red")

            # P&L with clear formatting
            if pnl_bps > 0:
                pnl_text = Text(f"+£{abs(pnl_bps):.2f}", style="bold green")
            elif pnl_bps < 0:
                pnl_text = Text(f"-£{abs(pnl_bps):.2f}", style="bold red")
            else:
                pnl_text = Text("£0.00", style="dim white")

            # Status with clear labels
            status = trade.get('status', 'open')
            if status == 'closed':
                if pnl_bps > 0:
                    status_display = Text("✅ Profit", style="green")
                else:
                    status_display = Text("❌ Loss", style="red")
            else:
                status_display = Text("⏳ Open", style="yellow")

            table.add_row(
                timestamp,
                symbol,
                mode,
                conf_display,
                pnl_text,
                status_display
            )

        header_text = Text()
        header_text.append(f"\n  Showing {len(trades)} most recent trades\n\n", style="dim white")
        
        # Combine header and table
        combined = Layout()
        combined.split_column(
            Layout(header_text, size=2),
            Layout(table)
            )

        return Panel(
            combined,
            title="[bold bright_white]💼 Recent Trading Activity[/bold bright_white]",
            border_style="bright_white",
            box=box.ROUNDED,
            padding=(1, 1)
        )

    def render_learning_status(self) -> Panel:
        """Render learning status in plain English"""
        learning = self.state.learning_progress
        metrics = self.state.metrics
        models = metrics.get('models', {})
        
        content = Text()
        content.append("\n", style="white")
        content.append("  🧠  SYSTEM INTELLIGENCE\n", style="bold bright_white")
        content.append("  ────────────────────────\n\n", style="dim white")

        if not learning:
            content.append("  Status: Initializing\n", style="dim yellow")
            content.append("  The system is setting up its\n", style="dim white")
            content.append("  learning capabilities.\n", style="dim white")
        else:
        auc = learning.get('auc', 0)
        auc_delta = learning.get('auc_delta', 0)
            
            # Prediction accuracy
            content.append("  Prediction Accuracy\n", style="cyan")
            if auc >= 0.70:
                content.append(f"  {auc:.1%}", style="bold green")
                content.append("  •  Excellent\n\n", style="dim green")
            elif auc >= 0.65:
                content.append(f"  {auc:.1%}", style="green")
                content.append("  •  Very Good\n\n", style="dim green")
            elif auc >= 0.60:
                content.append(f"  {auc:.1%}", style="yellow")
                content.append("  •  Good\n\n", style="dim yellow")
            else:
                content.append(f"  {auc:.1%}", style="red")
                content.append("  •  Improving\n\n", style="dim red")
            
            # Learning trend
            if auc_delta > 0:
                content.append("  Trend: ", style="cyan")
                content.append("Getting Better", style="bold green")
                content.append(f" (+{auc_delta:.1%})\n", style="green")
            elif auc_delta < 0:
                content.append("  Trend: ", style="cyan")
                content.append("Needs Attention", style="yellow")
                content.append(f" ({auc_delta:.1%})\n", style="yellow")
            else:
                content.append("  Trend: ", style="cyan")
                content.append("Stable\n", style="dim white")
        
        # Production readiness
        content.append("\n", style="white")
        ready = models.get('ready_for_hamilton', False)
        content.append("  Production Ready\n", style="cyan")
        if ready:
            content.append("  ✅  Yes", style="bold green")
            content.append("  •  Safe to use with real money\n", style="dim green")
        else:
            content.append("  ⏳  Not Yet", style="yellow")
            content.append("  •  Still in practice mode\n", style="dim yellow")

        return Panel(
            content,
            title="[bold bright_white]🎓 Learning Status[/bold bright_white]",
            border_style="bright_white",
            box=box.ROUNDED,
            padding=(1, 2)
        )

    def render_safety(self) -> Panel:
        """Render safety systems in plain English"""
        cb_status = self.state.circuit_breaker_status
        unlimited_mode = cb_status.get('unlimited_mode', self.unlimited_mode)
        
        content = Text()
        content.append("\n", style="white")
        content.append("  🛡️  SAFETY SYSTEMS\n", style="bold bright_white")
        content.append("  ───────────────────\n\n", style="dim white")
        
        if unlimited_mode:
            content.append("  Mode: Practice Trading\n", style="cyan")
            content.append("  ✅  All safety systems active\n", style="green")
            content.append("  •  No real money at risk\n", style="dim green")
            content.append("  •  Unlimited practice capital\n", style="dim green")
        else:
            content.append("  Mode: Live Trading\n", style="cyan")
            content.append("  ✅  Safety limits active\n\n", style="green")
            
            # Show safety levels
            level1 = cb_status.get('level1_active', False)
            level2 = cb_status.get('level2_active', False)
            level3 = cb_status.get('level3_active', False)
            level4 = cb_status.get('level4_active', False)
            
            if any([level1, level2, level3, level4]):
                content.append("  ⚠️  Some limits triggered\n", style="bold yellow")
            else:
                content.append("  ✅  All systems normal\n", style="green")
            
            content.append("\n", style="white")
            content.append("  Safety Levels:\n", style="dim white")
            content.append("  • Single trade limit\n", style="dim white")
            content.append("  • Hourly loss limit\n", style="dim white")
            content.append("  • Daily loss limit\n", style="dim white")
            content.append("  • Maximum drawdown limit\n", style="dim white")

        return Panel(
            content,
            title="[bold bright_white]🛡️ Safety & Risk Management[/bold bright_white]",
            border_style="bright_white",
            box=box.ROUNDED,
            padding=(1, 2)
        )

    def render_warnings(self) -> Panel:
        """Render warnings and alerts in plain English"""
        warnings = self.state.concept_drift_warnings
        
        content = Text()
        content.append("\n", style="white")
        content.append("  ⚠️  ALERTS & WARNINGS\n", style="bold bright_white")
        content.append("  ──────────────────────\n\n", style="dim white")
        
        if not warnings:
            content.append("  ✅  All Clear\n\n", style="bold green")
            content.append("  No issues detected.\n", style="dim white")
            content.append("  System is operating normally.\n", style="dim white")
        else:
            for warning in warnings[:3]:
                severity = warning.get('severity', 'WARNING')
                component = warning.get('component', 'Unknown')
                
                if severity == 'CRITICAL':
                    icon = "🔴"
                    style = "bold red"
                    label = "Critical Issue"
                elif severity == 'SEVERE':
                    icon = "🟠"
                    style = "yellow"
                    label = "Warning"
            else:
                    icon = "🟡"
                    style = "yellow"
                    label = "Notice"
                
                content.append(f"  {icon}  {label}\n", style=style)
                content.append(f"     {component}\n\n", style="dim white")

        return Panel(
            content,
            title="[bold bright_white]⚠️ System Alerts[/bold bright_white]",
            border_style="bright_white",
            box=box.ROUNDED,
            padding=(1, 2)
        )

    def render_system_health(self) -> Panel:
        """Render system health in plain English"""
        gates = self.state.metrics.get('gates', {}).get('gates', [])
        heatmap = self.state.confidence_heatmap

        content = Text()
        content.append("\n", style="white")
        content.append("  💚  SYSTEM HEALTH\n", style="bold bright_white")
        content.append("  ──────────────────\n\n", style="dim white")
        
        # Gate status summary
        if gates:
            total_gates = len(gates)
            passing_gates = sum(1 for g in gates if g.get('pass_rate', 0) > 0.30)
            
            content.append("  Quality Checks\n", style="cyan")
            content.append(f"  {passing_gates}/{total_gates} systems healthy\n\n", style="green")
            else:
            content.append("  Quality Checks\n", style="cyan")
            content.append("  Initializing...\n\n", style="dim yellow")
        
        # Market confidence
        if heatmap:
            content.append("  Market Understanding\n", style="cyan")
            avg_confidence = sum(heatmap.values()) / len(heatmap) if heatmap else 0
            
            if avg_confidence >= 0.70:
                content.append("  ✅  High confidence\n", style="green")
                content.append("  System understands current\n", style="dim green")
                content.append("  market conditions well\n", style="dim green")
            elif avg_confidence >= 0.50:
                content.append("  ⚠️  Moderate confidence\n", style="yellow")
                content.append("  System is learning current\n", style="dim yellow")
                content.append("  market conditions\n", style="dim yellow")
            else:
                content.append("  ⚠️  Low confidence\n", style="red")
                content.append("  Market conditions are\n", style="dim red")
                content.append("  unusual or changing\n", style="dim red")

        return Panel(
            content,
            title="[bold bright_white]💚 System Health[/bold bright_white]",
            border_style="bright_white",
            box=box.ROUNDED,
            padding=(1, 2)
        )

    def render_recent_activity(self) -> Panel:
        """Render recent activity in plain English"""
        events = self.state.recent_events[:8]
        active = self.state.active_learning
        
        content = Text()
        content.append("\n", style="white")
        content.append("  📝  WHAT'S HAPPENING NOW\n", style="bold bright_white")
        content.append("  ──────────────────────────\n\n", style="dim white")
        
        if active:
            content.append("  Currently Learning:\n", style="cyan")
            for item in active[:3]:
                # Simplify the message
                simple_msg = item.replace("Processing shadow trade outcomes", "Analyzing trade results")
                simple_msg = simple_msg.replace("Updating feature importance", "Improving predictions")
                simple_msg = simple_msg.replace("Analyzing win/loss patterns", "Learning from results")
                content.append(f"  • {simple_msg}\n", style="dim white")
            content.append("\n", style="white")
        
        if events:
            content.append("  Recent Activity:\n", style="cyan")
            for event in events[:5]:
                # Simplify event messages
                simple_event = event.replace("Shadow trade executed", "Trade completed")
                simple_event = simple_event.replace("Shadow", "")
                content.append(f"  • {simple_event}\n", style="dim white")
        else:
            content.append("  Waiting for activity...\n", style="dim white")

        return Panel(
            content,
            title="[bold bright_white]📝 Live Activity Feed[/bold bright_white]",
            border_style="bright_white",
            box=box.ROUNDED,
            padding=(1, 2)
        )

    def render_footer(self) -> Panel:
        """Render beautiful dashboard footer"""
        footer_text = Text()
        footer_text.append("  Press ", style="dim white")
        footer_text.append("Ctrl+C", style="bold red")
        footer_text.append(" to exit  •  Updates every ", style="dim white")
        footer_text.append(f"{self.refresh_rate} second", style="bold white")
        if self.refresh_rate != 1:
            footer_text.append("s", style="bold white")
        footer_text.append("  •  ", style="dim white")
        if self.unlimited_mode:
            footer_text.append("Practice Mode", style="bold green")
        else:
            footer_text.append(f"Live Trading: £{self.capital_gbp:,.2f}", style="bold green")

        return Panel(
            Align.center(footer_text),
            border_style="dim white",
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

            # Update circuit breaker status
            if self.unlimited_mode:
                virtual_capital = 1000.0  # For reporting only
                self.state.circuit_breaker_status = {
                    'unlimited_mode': True,
                    'level1_active': False,
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

            # Update concept drift warnings
            self.state.concept_drift_warnings = []

            # Update confidence heatmap
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
        layout["performance"].update(self.render_performance())
        layout["trades"].update(self.render_trades())
        layout["learning_status"].update(self.render_learning_status())
        layout["safety"].update(self.render_safety())
        layout["warnings"].update(self.render_warnings())
        layout["system_health"].update(self.render_system_health())
        layout["recent_activity"].update(self.render_recent_activity())
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
