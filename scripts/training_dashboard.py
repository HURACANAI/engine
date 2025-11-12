#!/usr/bin/env python3
"""
Real-Time Training Dashboard

Monitors SOL/USDT training progress with live updates from PostgreSQL.
Shows trades, performance metrics, and training progress in real-time.

Usage:
    python scripts/training_dashboard.py

Requirements:
    pip install rich psycopg2-binary
"""

import asyncio
import psycopg2
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import time

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.progress import Progress, BarColumn, TextColumn, SpinnerColumn
from rich import box
from rich.align import Align


@dataclass
class TrainingStats:
    """Training statistics"""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_profit_gbp: float = 0.0
    win_rate: float = 0.0
    avg_profit_gbp: float = 0.0
    avg_loss_gbp: float = 0.0
    largest_win_gbp: float = 0.0
    largest_loss_gbp: float = 0.0
    recent_trades: List[Dict[str, Any]] = None


class TrainingDashboard:
    """Real-time training dashboard using PostgreSQL data"""

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
        self.stats = TrainingStats(recent_trades=[])

    def get_db_connection(self):
        """Get PostgreSQL connection"""
        try:
            return psycopg2.connect(**self.db_config)
        except Exception as e:
            self.console.print(f"[red]Failed to connect to database: {e}[/red]")
            return None

    def fetch_stats(self) -> TrainingStats:
        """Fetch latest stats from PostgreSQL"""
        conn = self.get_db_connection()
        if not conn:
            return self.stats

        try:
            with conn.cursor() as cur:
                # Get overall stats
                cur.execute("""
                    SELECT
                        COUNT(*) as total_trades,
                        COUNT(*) FILTER (WHERE is_winner = true) as winning_trades,
                        COUNT(*) FILTER (WHERE is_winner = false) as losing_trades,
                        COALESCE(SUM(net_profit_gbp), 0) as total_profit,
                        COALESCE(AVG(CASE WHEN is_winner THEN net_profit_gbp END), 0) as avg_profit,
                        COALESCE(AVG(CASE WHEN NOT is_winner THEN net_profit_gbp END), 0) as avg_loss,
                        COALESCE(MAX(CASE WHEN is_winner THEN net_profit_gbp END), 0) as largest_win,
                        COALESCE(MIN(CASE WHEN NOT is_winner THEN net_profit_gbp END), 0) as largest_loss
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

                    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0
                else:
                    total_trades = winning_trades = losing_trades = 0
                    total_profit = avg_profit = avg_loss = largest_win = largest_loss = win_rate = 0.0

                # Get recent trades (last 10)
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
                        hold_duration_minutes
                    FROM trade_memory
                    WHERE symbol = 'SOL/USDT'
                    ORDER BY trade_id DESC
                    LIMIT 10
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
                    })

                return TrainingStats(
                    total_trades=total_trades,
                    winning_trades=winning_trades,
                    losing_trades=losing_trades,
                    total_profit_gbp=total_profit,
                    win_rate=win_rate,
                    avg_profit_gbp=avg_profit,
                    avg_loss_gbp=avg_loss,
                    largest_win_gbp=largest_win,
                    largest_loss_gbp=largest_loss,
                    recent_trades=recent_trades,
                )
        except Exception as e:
            self.console.print(f"[red]Error fetching stats: {e}[/red]")
            return self.stats
        finally:
            conn.close()

    def create_header(self) -> Panel:
        """Create header panel"""
        header_text = Text()
        header_text.append("🚀 SOL/USDT Training Dashboard\n", style="bold cyan")
        header_text.append(f"Updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC", style="dim")
        return Panel(
            Align.center(header_text),
            box=box.DOUBLE,
            style="cyan",
        )

    def create_stats_panel(self) -> Panel:
        """Create statistics panel"""
        table = Table(show_header=False, box=box.SIMPLE, padding=(0, 2))
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="bold white")

        # Calculate profit color
        profit_color = "green" if self.stats.total_profit_gbp >= 0 else "red"
        profit_symbol = "+" if self.stats.total_profit_gbp >= 0 else ""

        table.add_row("📊 Total Trades", f"{self.stats.total_trades:,}")
        table.add_row("✅ Winning Trades", f"[green]{self.stats.winning_trades:,}[/green]")
        table.add_row("❌ Losing Trades", f"[red]{self.stats.losing_trades:,}[/red]")
        table.add_row("🎯 Win Rate", f"[{'green' if self.stats.win_rate > 50 else 'red'}]{self.stats.win_rate:.1f}%[/{'green' if self.stats.win_rate > 50 else 'red'}]")
        table.add_row("💰 Total P&L", f"[{profit_color}]{profit_symbol}£{self.stats.total_profit_gbp:.2f}[/{profit_color}]")
        table.add_row("📈 Avg Win", f"[green]+£{self.stats.avg_profit_gbp:.2f}[/green]")
        table.add_row("📉 Avg Loss", f"[red]£{self.stats.avg_loss_gbp:.2f}[/red]")
        table.add_row("🏆 Largest Win", f"[green bold]+£{self.stats.largest_win_gbp:.2f}[/green bold]")
        table.add_row("💥 Largest Loss", f"[red bold]£{self.stats.largest_loss_gbp:.2f}[/red bold]")

        return Panel(
            table,
            title="📊 Performance Metrics",
            border_style="green" if self.stats.win_rate > 50 else "yellow",
            box=box.ROUNDED,
        )

    def create_recent_trades_panel(self) -> Panel:
        """Create recent trades panel"""
        table = Table(box=box.SIMPLE_HEAD, show_lines=False)
        table.add_column("ID", style="dim", width=6)
        table.add_column("Time", style="cyan", width=8)
        table.add_column("Entry", justify="right", width=8)
        table.add_column("Exit", justify="right", width=8)
        table.add_column("P&L (£)", justify="right", width=10)
        table.add_column("BPS", justify="right", width=8)
        table.add_column("Conf", justify="right", width=6)
        table.add_column("Hold", justify="right", width=7)
        table.add_column("Result", width=10)

        for trade in self.stats.recent_trades[:10]:
            # Format values
            trade_id = f"#{trade['trade_id']}"
            time_str = trade['entry_timestamp'].strftime('%H:%M:%S') if trade['entry_timestamp'] else "N/A"
            entry = f"${trade['entry_price']:.2f}"
            exit_val = f"${trade['exit_price']:.2f}"

            profit = trade['net_profit_gbp']
            profit_str = f"{'+' if profit >= 0 else ''}£{profit:.2f}"
            profit_color = "green" if profit >= 0 else "red"

            bps = trade['gross_profit_bps']
            bps_str = f"{'+' if bps >= 0 else ''}{bps:.1f}"
            bps_color = "green" if bps >= 0 else "red"

            conf = f"{trade['model_confidence']:.2f}"
            hold = f"{trade['hold_duration_minutes']}m"

            result = "✅ WIN" if trade['is_winner'] else "❌ LOSS"
            result_color = "green" if trade['is_winner'] else "red"

            table.add_row(
                trade_id,
                time_str,
                entry,
                exit_val,
                f"[{profit_color}]{profit_str}[/{profit_color}]",
                f"[{bps_color}]{bps_str}[/{bps_color}]",
                conf,
                hold,
                f"[{result_color}]{result}[/{result_color}]",
            )

        return Panel(
            table,
            title="📈 Recent Trades (Last 10)",
            border_style="blue",
            box=box.ROUNDED,
        )

    def create_layout(self) -> Layout:
        """Create dashboard layout"""
        layout = Layout()

        layout.split_column(
            Layout(name="header", size=4),
            Layout(name="body"),
        )

        layout["body"].split_row(
            Layout(name="stats", ratio=1),
            Layout(name="trades", ratio=2),
        )

        return layout

    def render_dashboard(self, layout: Layout):
        """Render the dashboard"""
        layout["header"].update(self.create_header())
        layout["stats"].update(self.create_stats_panel())
        layout["trades"].update(self.create_recent_trades_panel())

    async def run(self):
        """Run the dashboard"""
        layout = self.create_layout()

        self.console.clear()
        self.console.print("[bold cyan]🚀 Starting Training Dashboard...[/bold cyan]")
        self.console.print("[dim]Press Ctrl+C to exit[/dim]\n")

        # Test database connection
        conn = self.get_db_connection()
        if not conn:
            self.console.print("[red]❌ Failed to connect to database. Please check your connection.[/red]")
            return
        conn.close()

        self.console.print("[green]✅ Connected to PostgreSQL[/green]")
        time.sleep(1)

        try:
            with Live(layout, refresh_per_second=1, console=self.console) as live:
                while self.running:
                    # Fetch latest stats
                    self.stats = self.fetch_stats()

                    # Update display
                    self.render_dashboard(layout)

                    # Wait before next update
                    await asyncio.sleep(2.0)  # Update every 2 seconds

        except KeyboardInterrupt:
            self.console.print("\n[yellow]Dashboard stopped by user[/yellow]")
        except Exception as e:
            self.console.print(f"\n[red]Error: {e}[/red]")


def main():
    """Main entry point"""
    dashboard = TrainingDashboard()
    asyncio.run(dashboard.run())


if __name__ == "__main__":
    main()
