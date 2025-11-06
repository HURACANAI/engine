"""
Gate Inspector

Interactive UI for understanding gate decisions.

Features:
- View gate pass rates
- Understand why signals are blocked
- See counterfactual outcomes (good vs bad blocks)
- Adjust thresholds

Usage:
    python -m observability.ui.gate_inspector
"""

from typing import List, Dict, Any, Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box
from rich.prompt import Prompt, Confirm
from rich.progress import Progress, BarColumn, TextColumn
import structlog

from observability.analytics.gate_explainer import GateExplainer
from observability.analytics.metrics_computer import MetricsComputer

logger = structlog.get_logger(__name__)


class GateInspector:
    """Interactive gate decision inspector"""

    def __init__(self):
        self.console = Console()
        self.explainer = GateExplainer()
        self.metrics_computer = MetricsComputer()
        logger.info("gate_inspector_initialized")

    def render_gate_summary(self, gates: List[Dict[str, Any]]) -> Table:
        """Render summary of all gates"""
        table = Table(
            title="Gate Summary",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold cyan"
        )

        table.add_column("#", style="dim", width=3)
        table.add_column("Gate Name", style="cyan", width=20)
        table.add_column("Pass Rate", justify="right", width=12)
        table.add_column("Block Accuracy", justify="right", width=15)
        table.add_column("Status", width=15)

        for i, gate in enumerate(gates, 1):
            name = gate.get('name', 'Unknown')
            pass_rate = gate.get('pass_rate', 0)
            block_accuracy = gate.get('block_accuracy', 0)

            # Pass rate display
            pass_pct = f"{pass_rate:.1%}"

            # Block accuracy display
            if block_accuracy >= 0.70:
                acc_style = "green"
            elif block_accuracy >= 0.50:
                acc_style = "yellow"
            else:
                acc_style = "red"
            acc_text = Text(f"{block_accuracy:.1%}", style=acc_style)

            # Status
            if pass_rate < 0.10:
                status = Text("🔴 BLOCKING", style="red")
            elif pass_rate < 0.30:
                status = Text("🟡 STRICT", style="yellow")
            elif pass_rate > 0.70:
                status = Text("🟢 LENIENT", style="green")
            else:
                status = Text("🟢 GOOD", style="green")

            table.add_row(
                str(i),
                name,
                pass_pct,
                acc_text,
                status
            )

        return table

    def render_gate_details(self, gate_name: str) -> Panel:
        """Render detailed gate analysis"""
        # Get gate metrics
        from datetime import datetime
        today = datetime.utcnow().strftime("%Y-%m-%d")
        metrics = self.metrics_computer.compute_daily_metrics(today)
        gate_metrics = metrics.get('gates', [])

        gate = next((g for g in gate_metrics if g.get('name') == gate_name), None)

        if not gate:
            return Panel(
                Text("Gate not found", style="red"),
                title=f"[bold cyan]{gate_name}[/bold cyan]",
                border_style="cyan",
                box=box.ROUNDED
            )

        text = Text()

        # Header
        text.append(f"📊 {gate_name.upper()} ANALYSIS\n\n", style="bold cyan")

        # Pass rate
        pass_rate = gate.get('pass_rate', 0)
        text.append("Pass Rate: ", style="white")
        if pass_rate < 0.10:
            text.append(f"{pass_rate:.1%} ", style="red")
            text.append("🔴 BLOCKING MOST SIGNALS\n", style="red")
        elif pass_rate < 0.30:
            text.append(f"{pass_rate:.1%} ", style="yellow")
            text.append("🟡 STRICT\n", style="yellow")
        else:
            text.append(f"{pass_rate:.1%} ", style="green")
            text.append("🟢 GOOD\n", style="green")

        text.append("\n")

        # Block accuracy
        block_acc = gate.get('block_accuracy', 0)
        text.append("Block Accuracy: ", style="white")
        if block_acc >= 0.70:
            text.append(f"{block_acc:.1%} ", style="green")
            text.append("✅ GOOD BLOCKS\n", style="green")
        elif block_acc >= 0.50:
            text.append(f"{block_acc:.1%} ", style="yellow")
            text.append("⚠️ MIXED\n", style="yellow")
        else:
            text.append(f"{block_acc:.1%} ", style="red")
            text.append("❌ BAD BLOCKS\n", style="red")

        text.append("\n")

        # Thresholds
        thresholds = gate.get('thresholds', {})
        if thresholds:
            text.append("Current Thresholds:\n", style="bold white")
            for mode, threshold in thresholds.items():
                text.append(f"  {mode}: {threshold}\n")

        text.append("\n")

        # Recommendations
        text.append("💡 RECOMMENDATIONS:\n", style="bold white")

        if pass_rate < 0.10:
            text.append("  • Gate is too strict - consider lowering threshold\n", style="yellow")
            text.append("  • This is blocking 90%+ of signals\n", style="yellow")

        if block_acc < 0.50:
            text.append("  • Gate is blocking good trades - review threshold\n", style="red")

        if pass_rate > 0.70:
            text.append("  • Gate may be too lenient\n", style="yellow")

        if 0.20 <= pass_rate <= 0.40 and block_acc >= 0.70:
            text.append("  • Gate is working well! ✅\n", style="green")

        return Panel(
            text,
            title=f"[bold cyan]{gate_name}[/bold cyan]",
            border_style="cyan",
            box=box.ROUNDED
        )

    def show_blocked_signals(self, gate_name: str, limit: int = 10):
        """Show recently blocked signals"""
        self.console.clear()
        self.console.print(f"\n[bold cyan]Signals Blocked by {gate_name}[/bold cyan]\n")

        # Mock data for now (in real usage, query from database)
        self.console.print("[dim]Feature coming soon: Query blocked signals from database[/dim]\n")

        # Example of what this would show:
        table = Table(box=box.ROUNDED)
        table.add_column("Time", style="dim")
        table.add_column("Symbol")
        table.add_column("Mode")
        table.add_column("Input Value", justify="right")
        table.add_column("Threshold", justify="right")
        table.add_column("Counterfactual", justify="right")
        table.add_column("Good Block?")

        # Example rows
        table.add_row(
            "12:34:56",
            "ETH-USD",
            "scalp",
            "0.42",
            "0.45",
            "-8.5 bps",
            Text("✅ YES", style="green")
        )

        table.add_row(
            "12:35:12",
            "BTC-USD",
            "scalp",
            "0.43",
            "0.45",
            "+12.3 bps",
            Text("❌ NO", style="red")
        )

        self.console.print(table)
        Prompt.ask("\nPress Enter to continue")

    def suggest_threshold_adjustment(self, gate_name: str):
        """Suggest threshold adjustments"""
        self.console.clear()
        self.console.print(f"\n[bold cyan]Threshold Adjustment for {gate_name}[/bold cyan]\n")

        # Get gate metrics
        from datetime import datetime
        today = datetime.utcnow().strftime("%Y-%m-%d")
        metrics = self.metrics_computer.compute_daily_metrics(today)
        gate_metrics = metrics.get('gates', [])

        gate = next((g for g in gate_metrics if g.get('name') == gate_name), None)

        if not gate:
            self.console.print("[red]Gate not found[/red]")
            Prompt.ask("\nPress Enter to continue")
            return

        pass_rate = gate.get('pass_rate', 0)
        block_acc = gate.get('block_accuracy', 0)
        thresholds = gate.get('thresholds', {})

        # Analysis
        text = Text()
        text.append("Current State:\n", style="bold white")
        text.append(f"  Pass Rate: {pass_rate:.1%}\n")
        text.append(f"  Block Accuracy: {block_acc:.1%}\n")
        text.append("\n")

        text.append("Suggested Adjustments:\n", style="bold white")

        if pass_rate < 0.10 and block_acc < 0.60:
            # Too strict AND blocking good trades
            text.append("  🔴 URGENT: Lower threshold by 10-20%\n", style="red")
            text.append("  Reason: Blocking 90%+ of signals, many good ones\n", style="dim")

            for mode, threshold in thresholds.items():
                new_threshold = threshold * 0.85  # Lower by 15%
                text.append(f"\n  {mode}:\n")
                text.append(f"    Current: {threshold}\n", style="red")
                text.append(f"    Suggested: {new_threshold:.3f} (-15%)\n", style="green")

        elif pass_rate < 0.20:
            # Strict but maybe correct
            text.append("  🟡 Consider lowering threshold by 5-10%\n", style="yellow")
            text.append("  Reason: Blocking 80%+ of signals\n", style="dim")

            for mode, threshold in thresholds.items():
                new_threshold = threshold * 0.92  # Lower by 8%
                text.append(f"\n  {mode}:\n")
                text.append(f"    Current: {threshold}\n", style="yellow")
                text.append(f"    Suggested: {new_threshold:.3f} (-8%)\n", style="green")

        elif pass_rate > 0.70:
            # Too lenient
            text.append("  ⚠️ Consider raising threshold by 5-10%\n", style="yellow")
            text.append("  Reason: Passing 70%+ of signals, may be too lenient\n", style="dim")

            for mode, threshold in thresholds.items():
                new_threshold = threshold * 1.08  # Raise by 8%
                text.append(f"\n  {mode}:\n")
                text.append(f"    Current: {threshold}\n", style="yellow")
                text.append(f"    Suggested: {new_threshold:.3f} (+8%)\n", style="green")

        else:
            # Good range
            text.append("  ✅ Threshold looks good!\n", style="green")
            text.append("  Current pass rate is in healthy 20-70% range\n", style="dim")

        self.console.print(Panel(
            text,
            title=f"[bold cyan]Threshold Analysis: {gate_name}[/bold cyan]",
            border_style="cyan",
            box=box.ROUNDED
        ))

        Prompt.ask("\nPress Enter to continue")

    def interactive_menu(self):
        """Interactive menu"""
        while True:
            self.console.clear()
            self.console.print("\n[bold cyan]🚪 Gate Inspector[/bold cyan]\n")

            # Get current metrics
            from datetime import datetime
            today = datetime.utcnow().strftime("%Y-%m-%d")
            metrics = self.metrics_computer.compute_daily_metrics(today)
            gates = metrics.get('gates', [])

            if not gates:
                self.console.print("[dim]No gate data available yet[/dim]\n")
                self.console.print("Run the Engine first to generate gate metrics.\n")
                if Confirm.ask("Exit?"):
                    break
                continue

            # Show summary
            table = self.render_gate_summary(gates)
            self.console.print(table)
            self.console.print()

            self.console.print("Options:")
            self.console.print("  [cyan]1[/cyan] - Inspect specific gate")
            self.console.print("  [cyan]2[/cyan] - View blocked signals")
            self.console.print("  [cyan]3[/cyan] - Suggest threshold adjustments")
            self.console.print("  [cyan]q[/cyan] - Quit")
            self.console.print()

            choice = Prompt.ask("Choose option", choices=["1", "2", "3", "q"])

            if choice == "q":
                break
            elif choice == "1":
                gate_num = Prompt.ask(
                    f"Enter gate number (1-{len(gates)})",
                    default="1"
                )
                try:
                    idx = int(gate_num) - 1
                    if 0 <= idx < len(gates):
                        gate_name = gates[idx].get('name')
                        details = self.render_gate_details(gate_name)
                        self.console.print(details)
                        Prompt.ask("\nPress Enter to continue")
                except ValueError:
                    pass
            elif choice == "2":
                gate_num = Prompt.ask(
                    f"Enter gate number (1-{len(gates)})",
                    default="1"
                )
                try:
                    idx = int(gate_num) - 1
                    if 0 <= idx < len(gates):
                        gate_name = gates[idx].get('name')
                        self.show_blocked_signals(gate_name)
                except ValueError:
                    pass
            elif choice == "3":
                gate_num = Prompt.ask(
                    f"Enter gate number (1-{len(gates)})",
                    default="1"
                )
                try:
                    idx = int(gate_num) - 1
                    if 0 <= idx < len(gates):
                        gate_name = gates[idx].get('name')
                        self.suggest_threshold_adjustment(gate_name)
                except ValueError:
                    pass

    def run(self):
        """Run inspector"""
        try:
            self.interactive_menu()
        except KeyboardInterrupt:
            self.console.print("\n[dim]Goodbye![/dim]")


def main():
    """Main entry point"""
    inspector = GateInspector()
    inspector.run()


if __name__ == '__main__':
    main()
