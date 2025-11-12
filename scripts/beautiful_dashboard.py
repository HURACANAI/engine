#!/usr/bin/env python3
"""
Beautiful Training Dashboard - Apple/Microsoft Style
No technical jargon - designed for everyone to understand!
"""

import asyncio
import psycopg2
import os
import re
from datetime import datetime
from typing import Optional
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
    from rich.columns import Columns
except ImportError:
    os.system("pip3 install rich")
    from rich.console import Console
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich import box
    from rich.align import Align
    from rich.columns import Columns


class BeautifulDashboard:
    """Apple/Microsoft style beautiful dashboard - no technical jargon!"""

    def __init__(self, log_file: str = "/tmp/sol_final_run.log"):
        self.log_file = log_file
        self.console = Console()
        self.db_config = {
            'host': 'localhost',
            'port': 5432,
            'database': 'huracan',
            'user': 'haq',
        }

        # Simple state anyone can understand
        self.status_message = "Starting up..."
        self.progress_percent = 0
        self.candles_processed = 0
        self.total_candles = 2160
        self.trades_made = 0
        self.successful_trades = 0
        self.money_earned = 0.0
        self.current_price = 0.0
        self.activity_log = deque(maxlen=15)

        self.log_fp = None
        self.running = True

    def get_latest_info(self):
        """Get latest information in simple terms"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()

            # Get simple stats
            cur.execute("""
                SELECT
                    COUNT(*) as trades,
                    COUNT(*) FILTER (WHERE is_winner = true) as wins,
                    COALESCE(SUM(net_profit_gbp), 0) as money
                FROM trade_memory
                WHERE symbol = 'SOL/USDT'
            """)

            row = cur.fetchone()
            if row:
                self.trades_made = row[0]
                self.successful_trades = row[1]
                self.money_earned = float(row[2])

            conn.close()
        except:
            pass

    def read_activity(self):
        """Read what's happening right now"""
        try:
            if self.log_fp is None:
                if not os.path.exists(self.log_file):
                    return
                self.log_fp = open(self.log_file, 'r')
                self.log_fp.seek(0, 2)  # Go to end

            # Read new activity
            new_lines = self.log_fp.readlines()

            for line in new_lines:
                line = line.rstrip()
                if not line:
                    continue

                # Translate technical stuff to simple English
                simple_message = None

                if 'shadow_entry' in line:
                    # Extract price
                    match = re.search(r'price=.*?(\d+\.\d+)', line)
                    if match:
                        price = float(match.group(1))
                        self.current_price = price
                        simple_message = f"💰 Considering buying at ${price:.2f}"

                elif 'trade_stored' in line:
                    match = re.search(r'trade_id=.*?(\d+)', line)
                    if match:
                        trade_num = match.group(1)
                        simple_message = f"✅ Trade #{trade_num} completed and saved"

                elif 'analyzing_loss' in line:
                    match = re.search(r'loss_gbp=.*?(-[\d.]+)', line)
                    if match:
                        loss = float(match.group(1))
                        simple_message = f"📉 Learning from mistake: Lost £{abs(loss):.2f}"

                elif 'trade_analyzed' in line and 'winner' in line.lower():
                    if 'True' in line:
                        simple_message = "🎉 Good trade! Made a profit"
                    else:
                        simple_message = "📚 Trade didn't work out, learning why"

                elif 'historical_data_loaded' in line:
                    match = re.search(r'rows=.*?(\d+)', line)
                    if match:
                        rows = match.group(1)
                        self.total_candles = int(rows)
                        simple_message = f"📊 Loaded {rows} hours of price history"

                elif 'shadow_trading_start' in line:
                    simple_message = "🤖 Starting to practice trading..."
                    self.status_message = "Training the AI bot"

                elif 'higher_order_features' in line:
                    simple_message = "🧠 Analyzing patterns in price movements"

                elif 'confidence_decision' in line:
                    if 'decision=[35mtrade' in line:
                        simple_message = "🎯 Bot is confident - making a trade!"
                    else:
                        simple_message = "⏸️  Not confident enough - waiting"

                # Update progress
                if 'idx=' in line:
                    match = re.search(r'idx=.*?(\d+)', line)
                    if match:
                        self.candles_processed = int(match.group(1))
                        self.progress_percent = (self.candles_processed / self.total_candles * 100) if self.total_candles > 0 else 0

                if simple_message:
                    timestamp = datetime.now().strftime('%H:%M:%S')
                    self.activity_log.append(f"[dim]{timestamp}[/dim] {simple_message}")

        except:
            pass

    def create_title(self) -> Panel:
        """Create beautiful title"""
        title = Text()
        title.append("  SOL Trading Bot\n", style="bold white on blue")
        title.append("\n")
        title.append("  Artificial Intelligence Training Session\n", style="white")
        title.append(f"  {datetime.now().strftime('%A, %B %d, %Y at %I:%M %p')}\n", style="dim")

        return Panel(
            Align.center(title),
            box=box.HEAVY,
            style="blue",
            padding=(1, 2),
        )

    def create_status_card(self) -> Panel:
        """Current status card"""
        # Progress bar
        bar_width = 30
        filled = int(bar_width * self.progress_percent / 100)
        progress_bar = '●' * filled + '○' * (bar_width - filled)

        status = Text()
        status.append("Status:\n", style="bold cyan")
        status.append(f"{self.status_message}\n\n", style="white")

        status.append("Training Progress:\n", style="bold cyan")
        status.append(f"{progress_bar} {self.progress_percent:.0f}%\n\n", style="green")

        status.append("Data Analyzed:\n", style="bold cyan")
        status.append(f"{self.candles_processed:,} out of {self.total_candles:,} hours\n", style="white")

        if self.current_price > 0:
            status.append("\nCurrent Price:\n", style="bold cyan")
            status.append(f"${self.current_price:.2f}\n", style="yellow")

        return Panel(
            status,
            title="📊 Current Status",
            border_style="cyan",
            box=box.ROUNDED,
            padding=(1, 2),
        )

    def create_performance_card(self) -> Panel:
        """Performance metrics card"""
        # Calculate success rate
        success_rate = (self.successful_trades / self.trades_made * 100) if self.trades_made > 0 else 0

        # Determine mood
        if success_rate >= 50:
            mood = "😊 Doing great!"
            color = "green"
        elif success_rate >= 30:
            mood = "🤔 Learning..."
            color = "yellow"
        else:
            mood = "📚 Early learning stage"
            color = "yellow"

        perf = Text()
        perf.append(f"{mood}\n\n", style=f"bold {color}")

        perf.append("Practice Trades Made:\n", style="bold cyan")
        perf.append(f"{self.trades_made:,} trades\n\n", style="white")

        perf.append("Successful Trades:\n", style="bold cyan")
        perf.append(f"{self.successful_trades:,} trades", style="green")
        perf.append(f" ({success_rate:.1f}%)\n\n", style="dim")

        # Money with clear explanation
        if self.money_earned >= 0:
            perf.append("Total Earnings:\n", style="bold cyan")
            perf.append(f"+£{self.money_earned:,.2f}\n", style="bold green")
        else:
            perf.append("Learning Costs:\n", style="bold cyan")
            perf.append(f"£{abs(self.money_earned):,.2f}\n", style="yellow")
            perf.append("(This is practice - no real money!)", style="dim italic")

        return Panel(
            perf,
            title="💰 Performance",
            border_style="green" if success_rate >= 50 else "yellow",
            box=box.ROUNDED,
            padding=(1, 2),
        )

    def create_activity_card(self) -> Panel:
        """What's happening right now"""
        if not self.activity_log:
            activity = Text("Waiting for activity...", style="dim italic")
        else:
            activity = Text()
            for msg in self.activity_log:
                activity.append(msg + "\n")

        return Panel(
            activity,
            title="🔴 Live Activity Feed",
            border_style="white",
            box=box.ROUNDED,
            padding=(1, 2),
        )

    def create_explanation_card(self) -> Panel:
        """Simple explanation of what's happening"""
        explanation = Text()
        explanation.append("What's This?\n\n", style="bold cyan")
        explanation.append(
            "This AI bot is learning to trade Solana (SOL) cryptocurrency. "
            "It's analyzing past price data and practicing trades to learn what works. "
            "\n\n"
            "Think of it like teaching a student:\n"
            "• It studies 2,160 hours of price history\n"
            "• Tries making practice trades\n"
            "• Learns from mistakes\n"
            "• Gets better over time\n"
            "\n"
            "No real money is used - this is pure learning!",
            style="white"
        )

        return Panel(
            explanation,
            title="❓ What You're Watching",
            border_style="blue",
            box=box.ROUNDED,
            padding=(1, 2),
        )

    def create_layout(self) -> Layout:
        """Create beautiful layout"""
        layout = Layout()

        layout.split_column(
            Layout(name="title", size=7),
            Layout(name="main"),
            Layout(name="bottom"),
        )

        layout["main"].split_row(
            Layout(name="left"),
            Layout(name="right"),
        )

        layout["left"].split_column(
            Layout(name="status"),
            Layout(name="performance"),
        )

        layout["right"].update(self.create_activity_card())
        layout["bottom"].update(self.create_explanation_card())

        return layout

    def render(self, layout: Layout):
        """Render everything"""
        layout["title"].update(self.create_title())
        layout["status"].update(self.create_status_card())
        layout["performance"].update(self.create_performance_card())
        layout["right"].update(self.create_activity_card())

    async def run(self):
        """Run the beautiful dashboard"""
        layout = self.create_layout()

        self.console.clear()
        self.console.print("\n[bold cyan]Loading Beautiful Dashboard...[/bold cyan]\n")

        # Show explanation first
        self.console.print(Panel(
            "Welcome! You're about to see an AI bot learning to trade.\n"
            "Everything is explained in simple terms.\n"
            "Press Ctrl+C anytime to exit.",
            title="👋 Welcome",
            style="cyan",
        ))

        await asyncio.sleep(2)

        try:
            with Live(layout, refresh_per_second=2, console=self.console, screen=True) as live:
                while self.running:
                    # Update everything
                    self.read_activity()
                    self.get_latest_info()
                    self.render(layout)

                    await asyncio.sleep(0.5)

        except KeyboardInterrupt:
            self.console.clear()
            self.console.print("\n[bold cyan]✨ Thanks for watching![/bold cyan]\n")
            if self.trades_made > 0:
                self.console.print(f"The bot practiced {self.trades_made:,} trades")
                self.console.print(f"and learned from each one!\n")
        finally:
            if self.log_fp:
                self.log_fp.close()


def main():
    """Start the beautiful dashboard"""
    dashboard = BeautifulDashboard()
    asyncio.run(dashboard.run())


if __name__ == "__main__":
    main()
