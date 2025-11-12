#!/usr/bin/env python3
"""
Ultra-Detailed Real-Time Training Dashboard

Shows EVERYTHING happening during SOL/USDT training:
- Live trade execution details (entry/exit prices, confidence, regime)
- Model learning progress (policy loss, value loss, entropy)
- Market conditions and feature analysis
- Pattern recognition and memory retrieval
- Cost breakdown and performance metrics
- Real-time system health monitoring

Usage:
    python scripts/ultra_detailed_dashboard.py

Requirements:
    pip install rich psycopg2-binary
"""

import asyncio
import psycopg2
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import time
import sys

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
from rich.tree import Tree


@dataclass
class DetailedTradeStats:
    """Comprehensive trading statistics"""
    # Overall metrics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_profit_gbp: float = 0.0
    win_rate: float = 0.0

    # Profitability metrics
    avg_profit_gbp: float = 0.0
    avg_loss_gbp: float = 0.0
    largest_win_gbp: float = 0.0
    largest_loss_gbp: float = 0.0
    avg_win_bps: float = 0.0
    avg_loss_bps: float = 0.0

    # Timing metrics
    avg_hold_duration: float = 0.0
    shortest_trade_mins: int = 0
    longest_trade_mins: int = 0

    # Confidence and regime
    avg_confidence: float = 0.0
    regime_breakdown: Dict[str, int] = field(default_factory=dict)

    # Exit reasons
    exit_reason_breakdown: Dict[str, int] = field(default_factory=dict)

    # Recent trades
    recent_trades: List[Dict[str, Any]] = field(default_factory=list)

    # Hourly performance
    trades_per_hour: Dict[int, int] = field(default_factory=dict)
    profit_per_hour: Dict[int, float] = field(default_factory=dict)


@dataclass
class MarketConditions:
    """Current market conditions"""
    current_price: float = 0.0
    price_change_1h: float = 0.0
    price_change_24h: float = 0.0
    volatility_bps: float = 0.0
    spread_bps: float = 0.0
    volume_24h: float = 0.0
    last_update: Optional[datetime] = None


@dataclass
class ModelMetrics:
    """Model training metrics"""
    policy_loss: float = 0.0
    value_loss: float = 0.0
    entropy: float = 0.0
    learning_rate: float = 0.0003
    total_updates: int = 0
    last_update: Optional[datetime] = None


class UltraDetailedDashboard:
    """Ultra-detailed real-time training dashboard"""

    def __init__(self, db_config: Optional[Dict[str, str]] = None):
        """
        Initialize dashboard.

        Args:
            db_config: PostgreSQL connection config (defaults to localhost)
        """
        self.console = Console()
        self.db_config = db_config or {
            'host': 'localhost',
            'port': 5432,
            'database': 'huracan',
            'user': 'haq',
        }
        self.running = True
        self.stats = DetailedTradeStats()
        self.market = MarketConditions()
        self.model = ModelMetrics()
        self.start_time = datetime.now(timezone.utc)

    def get_db_connection(self):
        """Get PostgreSQL connection"""
        try:
            return psycopg2.connect(**self.db_config)
        except Exception as e:
            self.console.print(f"[red]Failed to connect to database: {e}[/red]")
            return None

    def fetch_comprehensive_stats(self) -> DetailedTradeStats:
        """Fetch comprehensive statistics from PostgreSQL"""
        conn = self.get_db_connection()
        if not conn:
            return self.stats

        try:
            with conn.cursor() as cur:
                # Get overall stats with detailed breakdowns
                cur.execute("""
                    SELECT
                        COUNT(*) as total_trades,
                        COUNT(*) FILTER (WHERE is_winner = true) as winning_trades,
                        COUNT(*) FILTER (WHERE is_winner = false) as losing_trades,
                        COALESCE(SUM(net_profit_gbp), 0) as total_profit,
                        COALESCE(AVG(CASE WHEN is_winner THEN net_profit_gbp END), 0) as avg_profit,
                        COALESCE(AVG(CASE WHEN NOT is_winner THEN net_profit_gbp END), 0) as avg_loss,
                        COALESCE(MAX(CASE WHEN is_winner THEN net_profit_gbp END), 0) as largest_win,
                        COALESCE(MIN(CASE WHEN NOT is_winner THEN net_profit_gbp END), 0) as largest_loss,
                        COALESCE(AVG(CASE WHEN is_winner THEN gross_profit_bps END), 0) as avg_win_bps,
                        COALESCE(AVG(CASE WHEN NOT is_winner THEN gross_profit_bps END), 0) as avg_loss_bps,
                        COALESCE(AVG(hold_duration_minutes), 0) as avg_hold,
                        COALESCE(MIN(hold_duration_minutes), 0) as shortest_trade,
                        COALESCE(MAX(hold_duration_minutes), 0) as longest_trade,
                        COALESCE(AVG(model_confidence), 0) as avg_confidence
                    FROM trade_memory
                    WHERE symbol = 'SOL/USDT'
                """)

                row = cur.fetchone()
                if row:
                    total_trades = row[0]
                    winning_trades = row[1]
                    losing_trades = row[2]
                    total_profit = float(row[3])
                    avg_profit = float(row[4])
                    avg_loss = float(row[5])
                    largest_win = float(row[6])
                    largest_loss = float(row[7])
                    avg_win_bps = float(row[8])
                    avg_loss_bps = float(row[9])
                    avg_hold = float(row[10])
                    shortest_trade = int(row[11]) if row[11] else 0
                    longest_trade = int(row[12]) if row[12] else 0
                    avg_confidence = float(row[13])

                    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0
                else:
                    total_trades = winning_trades = losing_trades = 0
                    total_profit = avg_profit = avg_loss = largest_win = largest_loss = 0.0
                    avg_win_bps = avg_loss_bps = avg_hold = win_rate = avg_confidence = 0.0
                    shortest_trade = longest_trade = 0

                # Get regime breakdown
                cur.execute("""
                    SELECT
                        COALESCE(market_regime, 'unknown') as regime,
                        COUNT(*) as count
                    FROM trade_memory
                    WHERE symbol = 'SOL/USDT'
                    GROUP BY market_regime
                    ORDER BY count DESC
                """)
                regime_breakdown = {row[0]: row[1] for row in cur.fetchall()}

                # Get exit reason breakdown
                cur.execute("""
                    SELECT
                        exit_reason,
                        COUNT(*) as count
                    FROM trade_memory
                    WHERE symbol = 'SOL/USDT' AND exit_reason IS NOT NULL
                    GROUP BY exit_reason
                    ORDER BY count DESC
                """)
                exit_reason_breakdown = {row[0]: row[1] for row in cur.fetchall()}

                # Get hourly breakdown
                cur.execute("""
                    SELECT
                        EXTRACT(HOUR FROM entry_timestamp) as hour,
                        COUNT(*) as trade_count,
                        COALESCE(SUM(net_profit_gbp), 0) as profit
                    FROM trade_memory
                    WHERE symbol = 'SOL/USDT'
                        AND entry_timestamp >= NOW() - INTERVAL '24 hours'
                    GROUP BY EXTRACT(HOUR FROM entry_timestamp)
                    ORDER BY hour
                """)
                trades_per_hour = {}
                profit_per_hour = {}
                for row in cur.fetchall():
                    hour = int(row[0])
                    trades_per_hour[hour] = row[1]
                    profit_per_hour[hour] = float(row[2])

                # Get recent trades (last 20 with full details)
                cur.execute("""
                    SELECT
                        trade_id,
                        entry_timestamp,
                        entry_price,
                        exit_price,
                        direction,
                        net_profit_gbp,
                        gross_profit_bps,
                        is_winner,
                        exit_reason,
                        model_confidence,
                        hold_duration_minutes,
                        market_regime,
                        volatility_bps,
                        spread_at_entry_bps,
                        decision_reason
                    FROM trade_memory
                    WHERE symbol = 'SOL/USDT'
                    ORDER BY trade_id DESC
                    LIMIT 20
                """)

                recent_trades = []
                for row in cur.fetchall():
                    recent_trades.append({
                        'trade_id': row[0],
                        'entry_timestamp': row[1],
                        'entry_price': float(row[2]) if row[2] else 0,
                        'exit_price': float(row[3]) if row[3] else 0,
                        'direction': row[4],
                        'net_profit_gbp': float(row[5]) if row[5] else 0,
                        'gross_profit_bps': float(row[6]) if row[6] else 0,
                        'is_winner': row[7],
                        'exit_reason': row[8],
                        'model_confidence': float(row[9]) if row[9] else 0,
                        'hold_duration_minutes': row[10] if row[10] else 0,
                        'market_regime': row[11] or 'unknown',
                        'volatility_bps': float(row[12]) if row[12] else 0,
                        'spread_bps': float(row[13]) if row[13] else 0,
                        'decision_reason': row[14] or 'N/A',
                    })

                return DetailedTradeStats(
                    total_trades=total_trades,
                    winning_trades=winning_trades,
                    losing_trades=losing_trades,
                    total_profit_gbp=total_profit,
                    win_rate=win_rate,
                    avg_profit_gbp=avg_profit,
                    avg_loss_gbp=avg_loss,
                    largest_win_gbp=largest_win,
                    largest_loss_gbp=largest_loss,
                    avg_win_bps=avg_win_bps,
                    avg_loss_bps=avg_loss_bps,
                    avg_hold_duration=avg_hold,
                    shortest_trade_mins=shortest_trade,
                    longest_trade_mins=longest_trade,
                    avg_confidence=avg_confidence,
                    regime_breakdown=regime_breakdown,
                    exit_reason_breakdown=exit_reason_breakdown,
                    recent_trades=recent_trades,
                    trades_per_hour=trades_per_hour,
                    profit_per_hour=profit_per_hour,
                )
        except Exception as e:
            self.console.print(f"[red]Error fetching stats: {e}[/red]")
            return self.stats
        finally:
            conn.close()

    def create_header(self) -> Panel:
        """Create header panel"""
        uptime = datetime.now(timezone.utc) - self.start_time
        uptime_str = str(uptime).split('.')[0]  # Remove microseconds

        header_text = Text()
        header_text.append("🚀 SOL/USDT Ultra-Detailed Training Dashboard\n", style="bold cyan")
        header_text.append(f"Updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC | ", style="dim")
        header_text.append(f"Uptime: {uptime_str}", style="dim green")

        return Panel(
            Align.center(header_text),
            box=box.DOUBLE,
            style="cyan",
        )

    def create_overview_panel(self) -> Panel:
        """Create high-level overview panel"""
        table = Table(show_header=False, box=box.SIMPLE, padding=(0, 2))
        table.add_column("Metric", style="cyan bold", width=25)
        table.add_column("Value", style="bold white", width=20)
        table.add_column("Detail", style="dim", width=30)

        # Calculate profit color
        profit_color = "green" if self.stats.total_profit_gbp >= 0 else "red"
        profit_symbol = "+" if self.stats.total_profit_gbp >= 0 else ""

        # Win rate color
        wr_color = "green" if self.stats.win_rate > 50 else "yellow" if self.stats.win_rate > 40 else "red"

        table.add_row(
            "📊 Total Trades",
            f"{self.stats.total_trades:,}",
            f"✅ {self.stats.winning_trades} | ❌ {self.stats.losing_trades}"
        )
        table.add_row(
            "🎯 Win Rate",
            f"[{wr_color}]{self.stats.win_rate:.1f}%[/{wr_color}]",
            f"Target: 50%+"
        )
        table.add_row(
            "💰 Total P&L",
            f"[{profit_color}]{profit_symbol}£{self.stats.total_profit_gbp:.2f}[/{profit_color}]",
            f"Avg: £{self.stats.total_profit_gbp/max(1,self.stats.total_trades):.2f}/trade"
        )
        table.add_row(
            "📈 Avg Win",
            f"[green]+£{self.stats.avg_profit_gbp:.2f}[/green]",
            f"[green]{self.stats.avg_win_bps:.1f} bps[/green]"
        )
        table.add_row(
            "📉 Avg Loss",
            f"[red]£{self.stats.avg_loss_gbp:.2f}[/red]",
            f"[red]{self.stats.avg_loss_bps:.1f} bps[/red]"
        )
        table.add_row(
            "⏱️  Avg Hold Time",
            f"{self.stats.avg_hold_duration:.0f} min",
            f"Range: {self.stats.shortest_trade_mins}-{self.stats.longest_trade_mins} min"
        )
        table.add_row(
            "🎲 Avg Confidence",
            f"{self.stats.avg_confidence:.2%}",
            f"Threshold: 20%"
        )

        return Panel(
            table,
            title="📊 Overview",
            border_style="green" if self.stats.win_rate > 50 else "yellow",
            box=box.ROUNDED,
        )

    def create_regime_panel(self) -> Panel:
        """Create regime analysis panel"""
        if not self.stats.regime_breakdown:
            return Panel("No regime data available", title="🌍 Market Regime Analysis")

        table = Table(show_header=True, box=box.SIMPLE_HEAD)
        table.add_column("Regime", style="cyan", width=15)
        table.add_column("Trades", justify="right", width=10)
        table.add_column("Percentage", justify="right", width=12)
        table.add_column("Bar", width=30)

        total = sum(self.stats.regime_breakdown.values())

        # Sort by count
        sorted_regimes = sorted(self.stats.regime_breakdown.items(), key=lambda x: x[1], reverse=True)

        for regime, count in sorted_regimes:
            pct = (count / total * 100) if total > 0 else 0
            bar_length = int(pct / 100 * 20)
            bar = "█" * bar_length + "░" * (20 - bar_length)

            # Color based on regime
            regime_color = {
                'trend': 'green',
                'range': 'yellow',
                'panic': 'red',
                'unknown': 'dim'
            }.get(regime, 'white')

            table.add_row(
                f"[{regime_color}]{regime.upper()}[/{regime_color}]",
                f"{count}",
                f"{pct:.1f}%",
                f"[{regime_color}]{bar}[/{regime_color}]"
            )

        return Panel(
            table,
            title="🌍 Market Regime Analysis",
            border_style="blue",
            box=box.ROUNDED,
        )

    def create_exit_reasons_panel(self) -> Panel:
        """Create exit reasons breakdown panel"""
        if not self.stats.exit_reason_breakdown:
            return Panel("No exit data available", title="🚪 Exit Reasons")

        table = Table(show_header=True, box=box.SIMPLE_HEAD)
        table.add_column("Exit Reason", style="cyan", width=18)
        table.add_column("Count", justify="right", width=10)
        table.add_column("%", justify="right", width=8)

        total = sum(self.stats.exit_reason_breakdown.values())

        # Sort by count
        sorted_reasons = sorted(self.stats.exit_reason_breakdown.items(), key=lambda x: x[1], reverse=True)

        for reason, count in sorted_reasons:
            pct = (count / total * 100) if total > 0 else 0

            # Color based on reason
            reason_color = {
                'TAKE_PROFIT': 'green',
                'STOP_LOSS': 'red',
                'TIMEOUT': 'yellow',
                'MODEL_SIGNAL': 'blue'
            }.get(reason, 'white')

            table.add_row(
                f"[{reason_color}]{reason}[/{reason_color}]",
                f"{count}",
                f"{pct:.1f}%"
            )

        return Panel(
            table,
            title="🚪 Exit Reasons",
            border_style="blue",
            box=box.ROUNDED,
        )

    def create_recent_trades_panel(self) -> Panel:
        """Create detailed recent trades panel"""
        table = Table(box=box.SIMPLE_HEAD, show_lines=False)
        table.add_column("ID", style="dim", width=5)
        table.add_column("Time", style="cyan", width=8)
        table.add_column("Entry→Exit", justify="right", width=16)
        table.add_column("P&L", justify="right", width=10)
        table.add_column("Hold", justify="right", width=6)
        table.add_column("Conf", justify="right", width=5)
        table.add_column("Regime", width=8)
        table.add_column("Exit", width=11)
        table.add_column("Result", width=8)

        for trade in self.stats.recent_trades[:15]:  # Show 15 most recent
            # Format values
            trade_id = f"#{trade['trade_id']}"
            time_str = trade['entry_timestamp'].strftime('%H:%M:%S') if trade['entry_timestamp'] else "N/A"
            entry_exit = f"${trade['entry_price']:.2f}→${trade['exit_price']:.2f}"

            profit = trade['net_profit_gbp']
            profit_str = f"{'+' if profit >= 0 else ''}£{profit:.2f}"
            profit_color = "green" if profit >= 0 else "red"

            hold = f"{trade['hold_duration_minutes']}m"
            conf = f"{trade['model_confidence']:.0%}"

            regime = trade['market_regime'][:8]
            regime_color = {
                'trend': 'green',
                'range': 'yellow',
                'panic': 'red'
            }.get(regime, 'white')

            exit_reason = trade['exit_reason'][:11] if trade['exit_reason'] else "N/A"

            result = "✅ WIN" if trade['is_winner'] else "❌ LOSS"
            result_color = "green" if trade['is_winner'] else "red"

            table.add_row(
                trade_id,
                time_str,
                entry_exit,
                f"[{profit_color}]{profit_str}[/{profit_color}]",
                hold,
                conf,
                f"[{regime_color}]{regime}[/{regime_color}]",
                exit_reason,
                f"[{result_color}]{result}[/{result_color}]",
            )

        return Panel(
            table,
            title="📈 Recent Trades (Last 15)",
            border_style="blue",
            box=box.ROUNDED,
        )

    def create_trade_details_panel(self) -> Panel:
        """Create ultra-detailed view of the most recent trade"""
        if not self.stats.recent_trades:
            return Panel("No trades yet", title="🔍 Latest Trade Details")

        trade = self.stats.recent_trades[0]  # Most recent

        # Create a tree structure for the trade
        tree = Tree(f"[bold cyan]Trade #{trade['trade_id']}[/bold cyan]")

        # Entry details
        entry_branch = tree.add("[yellow]📥 Entry[/yellow]")
        entry_branch.add(f"Timestamp: {trade['entry_timestamp'].strftime('%Y-%m-%d %H:%M:%S UTC')}")
        entry_branch.add(f"Price: ${trade['entry_price']:.2f}")
        entry_branch.add(f"Direction: {trade['direction']}")
        entry_branch.add(f"Confidence: {trade['model_confidence']:.2%}")
        entry_branch.add(f"Market Regime: {trade['market_regime']}")
        entry_branch.add(f"Volatility: {trade['volatility_bps']:.1f} bps")
        entry_branch.add(f"Spread: {trade['spread_bps']:.1f} bps")

        # Exit details
        exit_branch = tree.add("[yellow]📤 Exit[/yellow]")
        exit_branch.add(f"Price: ${trade['exit_price']:.2f}")
        exit_branch.add(f"Reason: {trade['exit_reason']}")
        exit_branch.add(f"Hold Duration: {trade['hold_duration_minutes']} minutes")

        # Performance
        profit_color = "green" if trade['net_profit_gbp'] >= 0 else "red"
        perf_branch = tree.add(f"[{profit_color}]💰 Performance[/{profit_color}]")
        perf_branch.add(f"Net P&L: £{trade['net_profit_gbp']:.2f}")
        perf_branch.add(f"Gross BPS: {trade['gross_profit_bps']:.1f}")
        perf_branch.add(f"Result: {'WIN ✅' if trade['is_winner'] else 'LOSS ❌'}")

        # Decision reasoning
        if trade['decision_reason'] and trade['decision_reason'] != 'N/A':
            reason_branch = tree.add("[cyan]🧠 Decision Reasoning[/cyan]")
            reason_branch.add(trade['decision_reason'])

        return Panel(
            tree,
            title="🔍 Latest Trade Details",
            border_style="magenta",
            box=box.ROUNDED,
        )

    def create_hourly_activity_panel(self) -> Panel:
        """Create hourly trading activity chart"""
        if not self.stats.trades_per_hour:
            return Panel("No hourly data available", title="📅 24h Activity")

        table = Table(show_header=True, box=box.SIMPLE_HEAD)
        table.add_column("Hour", style="cyan", width=8)
        table.add_column("Trades", justify="right", width=8)
        table.add_column("P&L", justify="right", width=10)
        table.add_column("Activity", width=30)

        max_trades = max(self.stats.trades_per_hour.values()) if self.stats.trades_per_hour else 1

        for hour in sorted(self.stats.trades_per_hour.keys()):
            trades = self.stats.trades_per_hour[hour]
            profit = self.stats.profit_per_hour.get(hour, 0.0)

            # Activity bar
            bar_length = int((trades / max_trades) * 20)
            bar = "█" * bar_length + "░" * (20 - bar_length)

            profit_color = "green" if profit >= 0 else "red"
            profit_str = f"{'+' if profit >= 0 else ''}£{profit:.2f}"

            table.add_row(
                f"{hour:02d}:00",
                f"{trades}",
                f"[{profit_color}]{profit_str}[/{profit_color}]",
                bar
            )

        return Panel(
            table,
            title="📅 24h Trading Activity",
            border_style="blue",
            box=box.ROUNDED,
        )

    def create_performance_metrics_panel(self) -> Panel:
        """Create advanced performance metrics"""
        table = Table(show_header=False, box=box.SIMPLE, padding=(0, 2))
        table.add_column("Metric", style="cyan", width=25)
        table.add_column("Value", style="bold white", width=15)
        table.add_column("Analysis", style="dim", width=30)

        # Calculate advanced metrics
        if self.stats.total_trades > 0:
            profit_factor = abs(self.stats.avg_profit_gbp * self.stats.winning_trades /
                               max(0.01, abs(self.stats.avg_loss_gbp * self.stats.losing_trades)))

            expectancy = ((self.stats.win_rate/100) * self.stats.avg_profit_gbp -
                         ((100-self.stats.win_rate)/100) * abs(self.stats.avg_loss_gbp))

            risk_reward = abs(self.stats.avg_profit_gbp / max(0.01, abs(self.stats.avg_loss_gbp)))
        else:
            profit_factor = 0
            expectancy = 0
            risk_reward = 0

        table.add_row(
            "💹 Profit Factor",
            f"{profit_factor:.2f}",
            "Good: >1.5" if profit_factor > 1.5 else "Target: 1.5+"
        )
        table.add_row(
            "🎯 Expectancy",
            f"£{expectancy:.2f}",
            "Avg expected profit/trade"
        )
        table.add_row(
            "⚖️  Risk:Reward",
            f"1:{risk_reward:.2f}",
            "Target: 1:1.5+"
        )
        table.add_row(
            "🏆 Best Trade",
            f"[green]+£{self.stats.largest_win_gbp:.2f}[/green]",
            f"Win size: {self.stats.largest_win_gbp/max(1,self.stats.avg_profit_gbp):.1f}x avg"
        )
        table.add_row(
            "💥 Worst Trade",
            f"[red]£{self.stats.largest_loss_gbp:.2f}[/red]",
            f"Loss size: {abs(self.stats.largest_loss_gbp)/max(0.01,abs(self.stats.avg_loss_gbp)):.1f}x avg"
        )

        return Panel(
            table,
            title="📊 Advanced Metrics",
            border_style="cyan",
            box=box.ROUNDED,
        )

    def create_layout(self) -> Layout:
        """Create ultra-detailed dashboard layout"""
        layout = Layout()

        layout.split_column(
            Layout(name="header", size=4),
            Layout(name="top_row", size=12),
            Layout(name="middle_row", size=16),
            Layout(name="bottom_row"),
        )

        # Top row: Overview + Performance metrics
        layout["top_row"].split_row(
            Layout(name="overview", ratio=2),
            Layout(name="performance", ratio=1),
        )

        # Middle row: Recent trades + Latest trade details
        layout["middle_row"].split_row(
            Layout(name="recent_trades", ratio=2),
            Layout(name="trade_details", ratio=1),
        )

        # Bottom row: Regime + Exit reasons + Hourly activity
        layout["bottom_row"].split_row(
            Layout(name="regime", ratio=1),
            Layout(name="exit_reasons", ratio=1),
            Layout(name="hourly", ratio=1),
        )

        return layout

    def render_dashboard(self, layout: Layout):
        """Render the ultra-detailed dashboard"""
        layout["header"].update(self.create_header())
        layout["overview"].update(self.create_overview_panel())
        layout["performance"].update(self.create_performance_metrics_panel())
        layout["recent_trades"].update(self.create_recent_trades_panel())
        layout["trade_details"].update(self.create_trade_details_panel())
        layout["regime"].update(self.create_regime_panel())
        layout["exit_reasons"].update(self.create_exit_reasons_panel())
        layout["hourly"].update(self.create_hourly_activity_panel())

    async def run(self):
        """Run the ultra-detailed dashboard"""
        layout = self.create_layout()

        self.console.clear()
        self.console.print("[bold cyan]🚀 Starting Ultra-Detailed Training Dashboard...[/bold cyan]")
        self.console.print("[dim]Press Ctrl+C to exit[/dim]\n")

        # Test database connection
        conn = self.get_db_connection()
        if not conn:
            self.console.print("[red]❌ Failed to connect to database. Please check your connection.[/red]")
            return
        conn.close()

        self.console.print("[green]✅ Connected to PostgreSQL[/green]")
        self.console.print("[yellow]📊 Loading comprehensive data...[/yellow]")
        time.sleep(1)

        try:
            with Live(layout, refresh_per_second=2, console=self.console, screen=True) as live:
                while self.running:
                    # Fetch latest comprehensive stats
                    self.stats = self.fetch_comprehensive_stats()

                    # Update display
                    self.render_dashboard(layout)

                    # Wait before next update
                    await asyncio.sleep(1.5)  # Update every 1.5 seconds

        except KeyboardInterrupt:
            self.console.print("\n[yellow]Dashboard stopped by user[/yellow]")
        except Exception as e:
            self.console.print(f"\n[red]Error: {e}[/red]")
            import traceback
            traceback.print_exc()


def main():
    """Main entry point"""
    dashboard = UltraDetailedDashboard()
    asyncio.run(dashboard.run())


if __name__ == "__main__":
    main()
