#!/usr/bin/env python3
"""
Live Training Monitor Dashboard

Real-time streaming dashboard that shows:
- Live log output
- Training progress
- Trade statistics
- System status

Usage:
    python3 scripts/live_training_monitor.py [--log-file /tmp/sol_final_run.log]
"""

import argparse
import asyncio
import psycopg2
import os
import re
from datetime import datetime
from typing import Optional, List
from collections import deque

try:
    from rich.console import Console
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich import box
    from rich.align import Align
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Installing rich library for beautiful dashboard...")
    os.system("pip3 install rich")
    from rich.console import Console
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich import box
    from rich.align import Align


class LiveTrainingMonitor:
    """Live streaming training monitor"""

    def __init__(self, log_file: str = "/tmp/sol_final_run.log", db_config: Optional[dict] = None):
        self.log_file = log_file
        self.console = Console()
        self.db_config = db_config or {
            'host': 'localhost',
            'port': 5432,
            'database': 'huracan',
            'user': 'haq',
        }

        # State
        self.log_lines = deque(maxlen=25)  # Last 25 log lines
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_profit = 0.0
        self.win_rate = 0.0
        self.current_idx = 0
        self.total_candles = 2160
        self.latest_trade_id = 0
        self.latest_price = 0.0
        self.latest_profit = 0.0
        self.confidence_threshold = 0.2
        self.running = True

        # Log file handle
        self.log_fp = None
        self.log_position = 0

    def get_db_stats(self):
        """Fetch latest stats from database"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()

            cur.execute("""
                SELECT
                    COUNT(*) as total,
                    COUNT(*) FILTER (WHERE is_winner = true) as wins,
                    COUNT(*) FILTER (WHERE is_winner = false) as losses,
                    COALESCE(SUM(net_profit_gbp), 0) as profit,
                    MAX(trade_id) as latest_id
                FROM trade_memory
                WHERE symbol = 'SOL/USDT'
            """)

            row = cur.fetchone()
            if row and row[0] > 0:
                self.total_trades = row[0]
                self.winning_trades = row[1]
                self.losing_trades = row[2]
                self.total_profit = float(row[3])
                self.latest_trade_id = row[4] or 0
                self.win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0.0

            conn.close()
        except Exception as e:
            pass  # Silently fail, we'll try again next time

    def read_new_logs(self):
        """Read new lines from log file"""
        try:
            if self.log_fp is None:
                if not os.path.exists(self.log_file):
                    return
                self.log_fp = open(self.log_file, 'r')
                # Skip to end on first open
                self.log_fp.seek(0, 2)
                self.log_position = self.log_fp.tell()

            # Read new lines
            self.log_fp.seek(self.log_position)
            new_lines = self.log_fp.readlines()
            self.log_position = self.log_fp.tell()

            for line in new_lines:
                line = line.rstrip()
                if not line:
                    continue

                # Add to deque
                self.log_lines.append(line)

                # Extract useful info from logs
                if 'shadow_entry' in line and 'idx=' in line:
                    match = re.search(r'idx=\[35m(\d+)\[0m', line)
                    if match:
                        self.current_idx = int(match.group(1))

                    match = re.search(r'price=\[35m([\d.]+)\[0m', line)
                    if match:
                        self.latest_price = float(match.group(1))

                elif 'trade_stored' in line and 'trade_id=' in line:
                    match = re.search(r'trade_id=\[35m(\d+)\[0m', line)
                    if match:
                        self.latest_trade_id = int(match.group(1))

                elif 'confidence_decision' in line and 'confidence=' in line:
                    match = re.search(r'confidence=.*?\(([0-9.]+)\)', line)
                    if match:
                        # Just note the confidence, don't store
                        pass

        except Exception as e:
            pass

    def create_header(self) -> Panel:
        """Create header"""
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        header = Text()
        header.append("🚀 ", style="bold cyan")
        header.append("LIVE TRAINING MONITOR", style="bold white")
        header.append(" - SOL/USDT\n", style="bold cyan")
        header.append(f"Updated: {now}", style="dim")

        return Panel(
            Align.center(header),
            box=box.DOUBLE,
            style="cyan",
        )

    def create_progress_panel(self) -> Panel:
        """Create progress panel"""
        progress_pct = (self.current_idx / self.total_candles * 100) if self.total_candles > 0 else 0

        # Create progress bar
        bar_width = 40
        filled = int(bar_width * progress_pct / 100)
        bar = '█' * filled + '░' * (bar_width - filled)

        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Label", style="cyan")
        table.add_column("Value", style="bold white")

        table.add_row("Progress", f"[cyan]{bar}[/cyan] {progress_pct:.1f}%")
        table.add_row("Candle", f"{self.current_idx:,} / {self.total_candles:,}")
        table.add_row("Latest Price", f"${self.latest_price:.2f}" if self.latest_price > 0 else "N/A")
        table.add_row("Latest Trade", f"#{self.latest_trade_id}" if self.latest_trade_id > 0 else "N/A")
        table.add_row("Threshold", f"{self.confidence_threshold:.2f}")

        return Panel(
            table,
            title="📊 Training Progress",
            border_style="blue",
            box=box.ROUNDED,
        )

    def create_stats_panel(self) -> Panel:
        """Create statistics panel"""
        profit_color = "green" if self.total_profit >= 0 else "red"
        win_rate_color = "green" if self.win_rate >= 50 else "yellow" if self.win_rate >= 30 else "red"

        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="bold white")

        table.add_row("Total Trades", f"{self.total_trades:,}")
        table.add_row("Wins", f"[green]{self.winning_trades:,}[/green]")
        table.add_row("Losses", f"[red]{self.losing_trades:,}[/red]")
        table.add_row("Win Rate", f"[{win_rate_color}]{self.win_rate:.1f}%[/{win_rate_color}]")
        table.add_row("Total P&L", f"[{profit_color}]£{self.total_profit:,.2f}[/{profit_color}]")

        return Panel(
            table,
            title="💰 Performance",
            border_style="green" if self.win_rate >= 50 else "yellow",
            box=box.ROUNDED,
        )

    def create_logs_panel(self) -> Panel:
        """Create live logs panel"""
        if not self.log_lines:
            log_text = Text("Waiting for logs...", style="dim")
        else:
            log_text = Text()
            for line in self.log_lines:
                # Strip ANSI codes for cleaner display
                clean_line = re.sub(r'\x1b\[[0-9;]*m', '', line)

                # Color code based on content
                if 'error' in clean_line.lower() or 'failed' in clean_line.lower():
                    style = "red"
                elif 'warning' in clean_line.lower():
                    style = "yellow"
                elif 'success' in clean_line.lower() or 'complete' in clean_line.lower():
                    style = "green"
                elif 'info' in clean_line.lower():
                    style = "cyan"
                elif 'trade_stored' in clean_line:
                    style = "bold green"
                elif 'shadow_entry' in clean_line:
                    style = "bold blue"
                else:
                    style = "white"

                # Truncate very long lines
                if len(clean_line) > 100:
                    clean_line = clean_line[:97] + "..."

                log_text.append(clean_line + "\n", style=style)

        return Panel(
            log_text,
            title="📋 Live Logs (Last 25 Lines)",
            border_style="white",
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
            Layout(name="left"),
            Layout(name="logs", ratio=2),
        )

        layout["left"].split_column(
            Layout(name="progress"),
            Layout(name="stats"),
        )

        return layout

    def render_dashboard(self, layout: Layout):
        """Render the dashboard"""
        layout["header"].update(self.create_header())
        layout["progress"].update(self.create_progress_panel())
        layout["stats"].update(self.create_stats_panel())
        layout["logs"].update(self.create_logs_panel())

    async def run(self):
        """Run the live monitor"""
        layout = self.create_layout()

        self.console.clear()
        self.console.print("[bold cyan]🚀 Starting Live Training Monitor...[/bold cyan]")
        self.console.print(f"[dim]Monitoring: {self.log_file}[/dim]")
        self.console.print("[dim]Press Ctrl+C to exit[/dim]\n")

        await asyncio.sleep(1)

        try:
            with Live(layout, refresh_per_second=2, console=self.console, screen=True) as live:
                while self.running:
                    # Read new logs
                    self.read_new_logs()

                    # Update database stats (every other iteration to reduce load)
                    self.get_db_stats()

                    # Render
                    self.render_dashboard(layout)

                    # Wait
                    await asyncio.sleep(0.5)

        except KeyboardInterrupt:
            self.console.clear()
            self.console.print("\n[yellow]Monitor stopped by user[/yellow]")
        except Exception as e:
            self.console.clear()
            self.console.print(f"\n[red]Error: {e}[/red]")
            import traceback
            traceback.print_exc()
        finally:
            if self.log_fp:
                self.log_fp.close()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Live Training Monitor')
    parser.add_argument('--log-file', default='/tmp/sol_final_run.log',
                        help='Path to training log file')
    args = parser.parse_args()

    monitor = LiveTrainingMonitor(log_file=args.log_file)
    asyncio.run(monitor.run())


if __name__ == "__main__":
    main()
