"""
Model Evolution Tracker UI

Interactive UI for tracking model improvements.

Features:
- View model lineage
- Compare model versions
- Check Hamilton readiness
- View training history

Usage:
    python -m observability.ui.model_tracker_ui
"""

from typing import List, Dict, Any
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box
from rich.prompt import Prompt
from rich.tree import Tree
import structlog

from observability.analytics.model_evolution import ModelEvolutionTracker

logger = structlog.get_logger(__name__)


class ModelTrackerUI:
    """Interactive model evolution tracker"""

    def __init__(self):
        self.console = Console()
        self.tracker = ModelEvolutionTracker()
        logger.info("model_tracker_ui_initialized")

    def render_model_list(self, models: List[Dict[str, Any]]) -> Table:
        """Render list of models"""
        table = Table(
            title="Model History",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold cyan"
        )

        table.add_column("#", style="dim", width=3)
        table.add_column("Model ID", style="cyan", width=16)
        table.add_column("Created", width=19)
        table.add_column("AUC", justify="right", width=8)
        table.add_column("ECE", justify="right", width=8)
        table.add_column("Samples", justify="right", width=10)
        table.add_column("Hamilton", width=10)

        for i, model in enumerate(models, 1):
            model_id = model.get('model_id', '')[:16]  # Truncate
            created = model.get('created_at', '')
            auc = model.get('auc', 0)
            ece = model.get('ece', 0)
            samples = model.get('n_samples', 0)

            # Hamilton readiness
            ready = model.get('hamilton_ready', False)
            if ready:
                status = Text("✅ READY", style="green")
            else:
                status = Text("⏳ NOT YET", style="yellow")

            # AUC coloring
            auc_style = "green" if auc >= 0.65 else "yellow" if auc >= 0.55 else "red"
            auc_text = Text(f"{auc:.3f}", style=auc_style)

            # ECE coloring (lower is better)
            ece_style = "green" if ece <= 0.10 else "yellow" if ece <= 0.15 else "red"
            ece_text = Text(f"{ece:.3f}", style=ece_style)

            table.add_row(
                str(i),
                model_id,
                created,
                auc_text,
                ece_text,
                f"{samples:,}",
                status
            )

        return table

    def render_model_comparison(
        self,
        old_model: Dict[str, Any],
        new_model: Dict[str, Any]
    ) -> Panel:
        """Render comparison between two models"""
        text = Text()

        text.append("MODEL COMPARISON\n\n", style="bold cyan")

        # Model IDs
        text.append(f"Old: {old_model.get('model_id', 'N/A')[:16]}\n", style="dim")
        text.append(f"New: {new_model.get('model_id', 'N/A')[:16]}\n", style="dim")
        text.append("\n")

        # Metrics comparison
        old_auc = old_model.get('auc', 0)
        new_auc = new_model.get('auc', 0)
        delta_auc = new_auc - old_auc
        pct_auc = (delta_auc / old_auc * 100) if old_auc > 0 else 0

        text.append("AUC:\n", style="bold white")
        text.append(f"  Old: {old_auc:.3f}\n")
        text.append(f"  New: {new_auc:.3f}\n")
        text.append(f"  Change: ")
        if delta_auc > 0:
            text.append(f"+{delta_auc:.3f} (+{pct_auc:.1f}%)\n", style="green")
        elif delta_auc < 0:
            text.append(f"{delta_auc:.3f} ({pct_auc:.1f}%)\n", style="red")
        else:
            text.append("No change\n", style="dim")

        text.append("\n")

        # ECE
        old_ece = old_model.get('ece', 0)
        new_ece = new_model.get('ece', 0)
        delta_ece = new_ece - old_ece

        text.append("ECE (Calibration Error):\n", style="bold white")
        text.append(f"  Old: {old_ece:.3f}\n")
        text.append(f"  New: {new_ece:.3f}\n")
        text.append(f"  Change: ")
        # For ECE, negative is good (lower error)
        if delta_ece < 0:
            text.append(f"{delta_ece:.3f} (better!)\n", style="green")
        elif delta_ece > 0:
            text.append(f"+{delta_ece:.3f} (worse)\n", style="red")
        else:
            text.append("No change\n", style="dim")

        text.append("\n")

        # Samples
        old_samples = old_model.get('n_samples', 0)
        new_samples = new_model.get('n_samples', 0)

        text.append("Training Samples:\n", style="bold white")
        text.append(f"  Old: {old_samples:,}\n")
        text.append(f"  New: {new_samples:,}\n")
        text.append(f"  Increase: +{new_samples - old_samples:,}\n")

        text.append("\n")

        # Verdict
        text.append("VERDICT:\n", style="bold white")
        if delta_auc > 0.01 and delta_ece < 0:
            text.append("  ✅ SIGNIFICANT IMPROVEMENT\n", style="green")
            text.append("  Ready to export to Hamilton!\n", style="green")
        elif delta_auc > 0:
            text.append("  🟢 IMPROVEMENT\n", style="green")
        elif delta_auc < -0.01:
            text.append("  🔴 REGRESSION\n", style="red")
            text.append("  Consider reverting to previous model\n", style="red")
        else:
            text.append("  🟡 MINOR CHANGE\n", style="yellow")

        return Panel(
            text,
            title="[bold cyan]Model Comparison[/bold cyan]",
            border_style="cyan",
            box=box.ROUNDED
        )

    def render_hamilton_readiness(self, model: Dict[str, Any]) -> Panel:
        """Render Hamilton readiness assessment"""
        text = Text()

        model_id = model.get('model_id', 'N/A')[:16]
        text.append(f"Model: {model_id}\n\n", style="bold cyan")

        # Readiness checklist
        auc = model.get('auc', 0)
        ece = model.get('ece', 0)
        samples = model.get('n_samples', 0)

        text.append("READINESS CHECKLIST:\n\n", style="bold white")

        # AUC check
        auc_ok = auc >= 0.65
        text.append("  ")
        text.append("✅" if auc_ok else "❌", style="green" if auc_ok else "red")
        text.append(f" AUC ≥ 0.65: {auc:.3f}\n")

        # ECE check
        ece_ok = ece <= 0.10
        text.append("  ")
        text.append("✅" if ece_ok else "❌", style="green" if ece_ok else "red")
        text.append(f" ECE ≤ 0.10: {ece:.3f}\n")

        # Samples check
        samples_ok = samples >= 1000
        text.append("  ")
        text.append("✅" if samples_ok else "❌", style="green" if samples_ok else "red")
        text.append(f" Samples ≥ 1000: {samples:,}\n")

        text.append("\n")

        # Overall verdict
        ready = auc_ok and ece_ok and samples_ok

        text.append("HAMILTON READY: ")
        if ready:
            text.append("✅ YES\n", style="bold green")
            text.append("\nThis model is ready to export to Hamilton for live trading!\n", style="green")
        else:
            text.append("⏳ NOT YET\n", style="bold yellow")
            text.append("\nBlockers:\n", style="yellow")

            if not auc_ok:
                text.append(f"  • AUC too low (need {0.65 - auc:.3f} improvement)\n")
            if not ece_ok:
                text.append(f"  • ECE too high (need {ece - 0.10:.3f} reduction)\n")
            if not samples_ok:
                text.append(f"  • Not enough samples (need {1000 - samples:,} more)\n")

        return Panel(
            text,
            title="[bold cyan]Hamilton Readiness Assessment[/bold cyan]",
            border_style="cyan",
            box=box.ROUNDED
        )

    def interactive_menu(self):
        """Interactive menu"""
        while True:
            self.console.clear()
            self.console.print("\n[bold cyan]📈 Model Evolution Tracker[/bold cyan]\n")

            self.console.print("Options:")
            self.console.print("  [cyan]1[/cyan] - View model history")
            self.console.print("  [cyan]2[/cyan] - Compare two models")
            self.console.print("  [cyan]3[/cyan] - Check Hamilton readiness")
            self.console.print("  [cyan]4[/cyan] - View latest model")
            self.console.print("  [cyan]q[/cyan] - Quit")
            self.console.print()

            choice = Prompt.ask("Choose option", choices=["1", "2", "3", "4", "q"])

            if choice == "q":
                break
            elif choice == "1":
                self.view_model_history()
            elif choice == "2":
                self.compare_models()
            elif choice == "3":
                self.check_readiness()
            elif choice == "4":
                self.view_latest()

    def view_model_history(self):
        """View model history"""
        self.console.clear()
        self.console.print("\n[bold cyan]Model History[/bold cyan]\n")

        models = self.tracker.get_all_models(limit=20)

        if not models:
            self.console.print("[dim]No models found[/dim]\n")
            self.console.print("Train a model first to see it here.\n")
            Prompt.ask("Press Enter to continue")
            return

        table = self.render_model_list(models)
        self.console.print(table)
        Prompt.ask("\nPress Enter to continue")

    def compare_models(self):
        """Compare two models"""
        self.console.clear()
        self.console.print("\n[bold cyan]Compare Models[/bold cyan]\n")

        models = self.tracker.get_all_models(limit=10)

        if len(models) < 2:
            self.console.print("[dim]Need at least 2 models to compare[/dim]\n")
            Prompt.ask("Press Enter to continue")
            return

        # Show list
        for i, model in enumerate(models, 1):
            self.console.print(f"{i}. {model.get('model_id', '')[:16]} - AUC: {model.get('auc', 0):.3f}")

        self.console.print()

        # Get selections
        old_idx = Prompt.ask("Enter old model number", default="2")
        new_idx = Prompt.ask("Enter new model number", default="1")

        try:
            old_i = int(old_idx) - 1
            new_i = int(new_idx) - 1

            if 0 <= old_i < len(models) and 0 <= new_i < len(models):
                comparison = self.render_model_comparison(models[old_i], models[new_i])
                self.console.print(comparison)
                Prompt.ask("\nPress Enter to continue")
        except ValueError:
            pass

    def check_readiness(self):
        """Check Hamilton readiness"""
        self.console.clear()
        self.console.print("\n[bold cyan]Hamilton Readiness Check[/bold cyan]\n")

        models = self.tracker.get_all_models(limit=10)

        if not models:
            self.console.print("[dim]No models found[/dim]\n")
            Prompt.ask("Press Enter to continue")
            return

        # Show list
        for i, model in enumerate(models, 1):
            self.console.print(f"{i}. {model.get('model_id', '')[:16]} - AUC: {model.get('auc', 0):.3f}")

        self.console.print()

        model_idx = Prompt.ask("Enter model number", default="1")

        try:
            idx = int(model_idx) - 1
            if 0 <= idx < len(models):
                readiness = self.render_hamilton_readiness(models[idx])
                self.console.print(readiness)
                Prompt.ask("\nPress Enter to continue")
        except ValueError:
            pass

    def view_latest(self):
        """View latest model"""
        self.console.clear()
        self.console.print("\n[bold cyan]Latest Model[/bold cyan]\n")

        models = self.tracker.get_all_models(limit=1)

        if not models:
            self.console.print("[dim]No models found[/dim]\n")
            Prompt.ask("Press Enter to continue")
            return

        latest = models[0]

        readiness = self.render_hamilton_readiness(latest)
        self.console.print(readiness)
        Prompt.ask("\nPress Enter to continue")

    def run(self):
        """Run UI"""
        try:
            self.interactive_menu()
        except KeyboardInterrupt:
            self.console.print("\n[dim]Goodbye![/dim]")


def main():
    """Main entry point"""
    ui = ModelTrackerUI()
    ui.run()


if __name__ == '__main__':
    main()
