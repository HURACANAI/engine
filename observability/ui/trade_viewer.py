"""
Shadow Trade Viewer

Interactive UI for exploring shadow trades.

Features:
- Filter by mode, regime, date range
- Sort by P&L, duration, win rate
- Drill down into trade details
- View entry features

Usage:
    python -m observability.ui.trade_viewer
"""

from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box
from rich.prompt import Prompt, Confirm
import structlog

from observability.analytics.trade_journal import TradeJournal

logger = structlog.get_logger(__name__)


class TradeViewer:
    """Interactive shadow trade viewer"""

    def __init__(self):
        self.console = Console()
        self.journal = TradeJournal()
        logger.info("trade_viewer_initialized")

    def render_trade_list(
        self,
        trades: List[Dict[str, Any]],
        title: str = "Shadow Trades"
    ) -> Table:
        """Render list of trades"""
        table = Table(
            title=title,
            box=box.ROUNDED,
            show_header=True,
            header_style="bold cyan"
        )

        table.add_column("#", style="dim", width=4)
        table.add_column("Date", style="cyan", width=10)
        table.add_column("Symbol", width=8)
        table.add_column("Mode", width=6)
        table.add_column("Regime", width=8)
        table.add_column("Duration", justify="right", width=10)
        table.add_column("P&L (bps)", justify="right", width=10)
        table.add_column("Status", width=10)

        for i, trade in enumerate(trades, 1):
            trade_id = trade.get('trade_id', '')
            entry_ts = trade.get('entry_ts', '')[:10]  # YYYY-MM-DD
            symbol = trade.get('symbol', '')
            mode = trade.get('mode', '')[:4].upper()
            regime = trade.get('regime', '')

            duration_sec = trade.get('duration_sec', 0)
            if duration_sec < 60:
                duration = f"{duration_sec:.0f}s"
            elif duration_sec < 3600:
                duration = f"{duration_sec/60:.1f}m"
            else:
                duration = f"{duration_sec/3600:.1f}h"

            pnl_bps = trade.get('pnl_bps', 0)
            if pnl_bps > 0:
                pnl_text = f"+{pnl_bps:.2f}"
                pnl_style = "green"
            elif pnl_bps < 0:
                pnl_text = f"{pnl_bps:.2f}"
                pnl_style = "red"
            else:
                pnl_text = "0.00"
                pnl_style = "dim"

            status = trade.get('status', 'open')
            if status == 'closed':
                if pnl_bps > 0:
                    status_text = "✅ WIN"
                    status_style = "green"
                else:
                    status_text = "❌ LOSS"
                    status_style = "red"
            else:
                status_text = "⏳ OPEN"
                status_style = "yellow"

            table.add_row(
                str(i),
                entry_ts,
                symbol,
                mode,
                regime,
                duration,
                Text(pnl_text, style=pnl_style),
                Text(status_text, style=status_style)
            )

        return table

    def render_trade_details(self, trade: Dict[str, Any]) -> Panel:
        """Render detailed trade view"""
        text = Text()

        # Header
        text.append(f"Trade ID: {trade.get('trade_id', 'N/A')}\n", style="bold cyan")
        text.append("\n")

        # Basic info
        text.append("📊 TRADE INFO\n", style="bold white")
        text.append(f"  Symbol: {trade.get('symbol', 'N/A')}\n")
        text.append(f"  Mode: {trade.get('mode', 'N/A')}\n")
        text.append(f"  Regime: {trade.get('regime', 'N/A')}\n")
        text.append(f"  Entry: {trade.get('entry_ts', 'N/A')}\n")
        text.append(f"  Exit: {trade.get('exit_ts', 'N/A')}\n")
        text.append("\n")

        # P&L breakdown
        text.append("💰 P&L BREAKDOWN\n", style="bold white")
        pnl_bps = trade.get('pnl_bps', 0)
        pnl_style = "green" if pnl_bps > 0 else "red" if pnl_bps < 0 else "dim"
        text.append(f"  Total P&L: ", style="white")
        text.append(f"{pnl_bps:+.2f} bps\n", style=pnl_style)

        gross = trade.get('gross_pnl_bps', 0)
        text.append(f"  Gross: {gross:+.2f} bps\n")

        cost = trade.get('total_cost_bps', 0)
        text.append(f"  Cost: -{cost:.2f} bps\n", style="red")

        slippage = trade.get('slippage_bps', 0)
        text.append(f"  Slippage: -{slippage:.2f} bps\n")

        text.append("\n")

        # Entry features (if available)
        features = trade.get('features', {})
        if features:
            text.append("🎯 ENTRY FEATURES\n", style="bold white")
            for key, value in list(features.items())[:5]:  # Top 5
                text.append(f"  {key}: {value:.4f}\n")

        return Panel(
            text,
            title="[bold cyan]Trade Details[/bold cyan]",
            border_style="cyan",
            box=box.ROUNDED
        )

    def show_summary(self, days: int = 7):
        """Show summary statistics"""
        self.console.clear()
        self.console.print(f"\n[bold cyan]Shadow Trade Summary (Last {days} days)[/bold cyan]\n")

        stats = self.journal.get_stats(days=days)

        table = Table(box=box.ROUNDED, show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("All Modes", justify="right")
        table.add_column("Scalp", justify="right")
        table.add_column("Runner", justify="right")

        # Total trades
        table.add_row(
            "Total Trades",
            str(stats.get('total_trades', 0)),
            str(stats.get('scalp', {}).get('total', 0)),
            str(stats.get('runner', {}).get('total', 0))
        )

        # Win rate
        overall_wr = stats.get('win_rate', 0)
        scalp_wr = stats.get('scalp', {}).get('win_rate', 0)
        runner_wr = stats.get('runner', {}).get('win_rate', 0)

        table.add_row(
            "Win Rate",
            f"{overall_wr:.1%}",
            f"{scalp_wr:.1%}",
            f"{runner_wr:.1%}"
        )

        # Avg P&L
        overall_pnl = stats.get('avg_pnl_bps', 0)
        scalp_pnl = stats.get('scalp', {}).get('avg_pnl_bps', 0)
        runner_pnl = stats.get('runner', {}).get('avg_pnl_bps', 0)

        table.add_row(
            "Avg P&L (bps)",
            f"{overall_pnl:+.2f}",
            f"{scalp_pnl:+.2f}",
            f"{runner_pnl:+.2f}"
        )

        # Total P&L
        overall_total = stats.get('total_pnl_bps', 0)
        scalp_total = stats.get('scalp', {}).get('total_pnl_bps', 0)
        runner_total = stats.get('runner', {}).get('total_pnl_bps', 0)

        pnl_style = "green" if overall_total > 0 else "red"

        table.add_row(
            "Total P&L (bps)",
            Text(f"{overall_total:+.2f}", style=pnl_style),
            f"{scalp_total:+.2f}",
            f"{runner_total:+.2f}"
        )

        self.console.print(table)
        self.console.print()

    def interactive_menu(self):
        """Interactive menu"""
        while True:
            self.console.clear()
            self.console.print("\n[bold cyan]🔍 Shadow Trade Viewer[/bold cyan]\n")

            self.console.print("Options:")
            self.console.print("  [cyan]1[/cyan] - View recent trades")
            self.console.print("  [cyan]2[/cyan] - Filter by mode")
            self.console.print("  [cyan]3[/cyan] - Filter by regime")
            self.console.print("  [cyan]4[/cyan] - Show summary statistics")
            self.console.print("  [cyan]5[/cyan] - View best/worst trades")
            self.console.print("  [cyan]q[/cyan] - Quit")
            self.console.print()

            choice = Prompt.ask("Choose option", choices=["1", "2", "3", "4", "5", "q"])

            if choice == "q":
                break
            elif choice == "1":
                self.view_recent_trades()
            elif choice == "2":
                self.filter_by_mode()
            elif choice == "3":
                self.filter_by_regime()
            elif choice == "4":
                self.show_summary()
                Prompt.ask("\nPress Enter to continue")
            elif choice == "5":
                self.view_best_worst()

    def view_recent_trades(self, limit: int = 20):
        """View recent trades"""
        self.console.clear()
        self.console.print("\n[bold cyan]Recent Shadow Trades[/bold cyan]\n")

        trades = self.journal.query_trades(limit=limit)

        if not trades:
            self.console.print("[dim]No trades found[/dim]\n")
            Prompt.ask("Press Enter to continue")
            return

        table = self.render_trade_list(trades, f"Last {len(trades)} Trades")
        self.console.print(table)
        self.console.print()

        # Ask if user wants details
        if Confirm.ask("View trade details?"):
            trade_num = Prompt.ask(
                f"Enter trade number (1-{len(trades)})",
                default="1"
            )
            try:
                idx = int(trade_num) - 1
                if 0 <= idx < len(trades):
                    details = self.render_trade_details(trades[idx])
                    self.console.print(details)
                    Prompt.ask("\nPress Enter to continue")
            except ValueError:
                pass

    def filter_by_mode(self):
        """Filter trades by mode"""
        self.console.clear()
        mode = Prompt.ask(
            "Mode",
            choices=["scalp", "runner"],
            default="scalp"
        )

        trades = self.journal.query_trades(mode=mode, limit=50)

        if not trades:
            self.console.print(f"[dim]No {mode} trades found[/dim]\n")
            Prompt.ask("Press Enter to continue")
            return

        table = self.render_trade_list(trades, f"{mode.title()} Trades")
        self.console.print(table)
        Prompt.ask("\nPress Enter to continue")

    def filter_by_regime(self):
        """Filter trades by regime"""
        self.console.clear()
        regime = Prompt.ask(
            "Regime",
            choices=["TREND", "RANGE", "PANIC", "UNKNOWN"],
            default="TREND"
        )

        trades = self.journal.query_trades(regime=regime, limit=50)

        if not trades:
            self.console.print(f"[dim]No {regime} trades found[/dim]\n")
            Prompt.ask("Press Enter to continue")
            return

        table = self.render_trade_list(trades, f"{regime} Trades")
        self.console.print(table)
        Prompt.ask("\nPress Enter to continue")

    def view_best_worst(self):
        """View best and worst trades"""
        self.console.clear()
        self.console.print("\n[bold cyan]Best & Worst Trades[/bold cyan]\n")

        # Best trades
        best = self.journal.query_trades(
            min_return_bps=5.0,  # At least +5 bps
            limit=10
        )

        if best:
            self.console.print("[bold green]🏆 Top Performers[/bold green]\n")
            table = self.render_trade_list(best, "Best Trades")
            self.console.print(table)
            self.console.print()

        # Worst trades
        worst = self.journal.query_trades(
            max_return_bps=-5.0,  # At most -5 bps
            limit=10
        )

        if worst:
            self.console.print("[bold red]📉 Bottom Performers[/bold red]\n")
            table = self.render_trade_list(worst, "Worst Trades")
            self.console.print(table)

        Prompt.ask("\nPress Enter to continue")

    def run(self):
        """Run viewer"""
        try:
            self.interactive_menu()
        except KeyboardInterrupt:
            self.console.print("\n[dim]Goodbye![/dim]")


def main():
    """Main entry point"""
    viewer = TradeViewer()
    viewer.run()


if __name__ == '__main__':
    main()
