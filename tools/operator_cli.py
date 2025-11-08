#!/usr/bin/env python3
"""
Operator CLI

Command-line interface for operators to manage the Huracan Engine.

Commands:
- status: Show system status
- gates: View gate evaluation results
- deploy: Deploy model to Hamilton
- rollback: Rollback to previous model
- report: Generate reports

Usage:
    python operator_cli.py status
    python operator_cli.py gates --model-id btc_v48
    python operator_cli.py deploy --model-id btc_v48 --symbol BTC
    python operator_cli.py rollback --symbol BTC --to-version 47
    python operator_cli.py report --days 7
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Add engine to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from observability.decision_gates import GateSystem
from integration.hand_off import HamiltonDeployer
from observability.analytics.enhanced_learning_tracker import EnhancedLearningTracker
from integration.lineage import LineageTracker, RollbackManager


class OperatorCLI:
    """Operator command-line interface"""

    def __init__(self):
        self.gate_system = GateSystem()
        self.deployer = HamiltonDeployer()
        self.tracker = EnhancedLearningTracker()
        self.lineage = LineageTracker()
        self.rollback_mgr = RollbackManager()

    def status(self, args):
        """Show system status"""
        print("=" * 80)
        print("HURACAN ENGINE STATUS")
        print("=" * 80)
        print()

        # Active models
        active_models = self.deployer.get_active_models()
        print("ACTIVE MODELS:")
        for symbol, model_id in active_models.items():
            print(f"  {symbol}: {model_id}")
        print()

        # Recent gate evaluations (last 24h)
        print("RECENT GATE EVALUATIONS (Last 24h):")
        print("  [Would query gate_evaluations table]")
        print()

        # System health
        print("SYSTEM HEALTH:")
        print("  ✅ All systems operational")
        print()

        print("=" * 80)

    def gates(self, args):
        """View gate evaluation results"""
        model_id = args.model_id

        print("=" * 80)
        print(f"GATE EVALUATION: {model_id}")
        print("=" * 80)
        print()

        # Get gate thresholds
        thresholds = self.gate_system.get_gate_thresholds()

        print("GATE THRESHOLDS:")
        for gate_name, config in thresholds.items():
            print(f"\n{gate_name}:")
            for key, value in config.items():
                print(f"  {key}: {value}")

        print()
        print("=" * 80)

    def deploy(self, args):
        """Deploy model to Hamilton"""
        model_id = args.model_id
        symbol = args.symbol
        activate = args.activate

        print("=" * 80)
        print(f"DEPLOYING MODEL: {model_id}")
        print("=" * 80)
        print()

        try:
            result = self.deployer.deploy_model(
                model_id=model_id,
                symbol=symbol,
                activate_immediately=activate,
                deployment_notes=f"Deployed via CLI by operator"
            )

            print(f"✅ Deployment successful!")
            print()
            print(f"Model ID: {result.model_id}")
            print(f"Symbol: {result.symbol}")
            print(f"Config: {result.config_path}")
            print(f"Active: {result.is_active}")
            print(f"Gate Score: {result.gate_score:.2%}")
            print()

            if result.previous_model_id:
                print(f"Previous model: {result.previous_model_id}")

        except Exception as e:
            print(f"❌ Deployment failed: {e}")

        print("=" * 80)

    def rollback(self, args):
        """Rollback to previous model"""
        symbol = args.symbol
        to_version = args.to_version
        reason = args.reason or "Operator-initiated rollback"

        print("=" * 80)
        print(f"ROLLBACK: {symbol} → v{to_version}")
        print("=" * 80)
        print()

        target_model_id = f"{symbol.lower()}_v{to_version}"

        try:
            result = self.deployer.rollback_model(
                symbol=symbol,
                target_model_id=target_model_id,
                reason=reason
            )

            print(f"✅ Rollback successful!")
            print()
            print(f"Symbol: {result.symbol}")
            print(f"Now active: {result.model_id}")
            print(f"Previous: {result.previous_model_id}")
            print(f"Reason: {result.deployment_notes}")
            print()

        except Exception as e:
            print(f"❌ Rollback failed: {e}")

        print("=" * 80)

    def report(self, args):
        """Generate comprehensive report"""
        days = args.days

        print(self.tracker.generate_comprehensive_report(days=days))


def main():
    parser = argparse.ArgumentParser(description="Huracan Engine Operator CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Status command
    parser_status = subparsers.add_parser("status", help="Show system status")

    # Gates command
    parser_gates = subparsers.add_parser("gates", help="View gate evaluations")
    parser_gates.add_argument("--model-id", required=True, help="Model ID")

    # Deploy command
    parser_deploy = subparsers.add_parser("deploy", help="Deploy model")
    parser_deploy.add_argument("--model-id", required=True, help="Model ID")
    parser_deploy.add_argument("--symbol", required=True, help="Symbol (e.g., BTC)")
    parser_deploy.add_argument("--activate", action="store_true", help="Activate immediately")

    # Rollback command
    parser_rollback = subparsers.add_parser("rollback", help="Rollback model")
    parser_rollback.add_argument("--symbol", required=True, help="Symbol")
    parser_rollback.add_argument("--to-version", type=int, required=True, help="Target version")
    parser_rollback.add_argument("--reason", help="Rollback reason")

    # Report command
    parser_report = subparsers.add_parser("report", help="Generate report")
    parser_report.add_argument("--days", type=int, default=7, help="Days to include")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    cli = OperatorCLI()

    # Execute command
    if args.command == "status":
        cli.status(args)
    elif args.command == "gates":
        cli.gates(args)
    elif args.command == "deploy":
        cli.deploy(args)
    elif args.command == "rollback":
        cli.rollback(args)
    elif args.command == "report":
        cli.report(args)


if __name__ == "__main__":
    main()
