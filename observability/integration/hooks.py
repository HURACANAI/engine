"""
Integration Hooks

Easy-to-use hooks for integrating observability into simulations.

Usage:
    from observability.integration import ObservabilityHooks

    # Initialize
    hooks = ObservabilityHooks()

    # Hook into your simulation
    hooks.on_signal(signal_id, symbol, features, regime)
    hooks.on_gate_decision(signal_id, gate_name, decision, inputs)
    hooks.on_trade(trade_id, symbol, mode, entry_price, features)
    hooks.on_trade_exit(trade_id, exit_price, pnl_bps)
    hooks.on_training(model_id, metrics, samples)

    # Get daily summary
    summary = hooks.get_daily_summary()
"""

import asyncio
from datetime import datetime
from typing import Dict, Any, Optional
import structlog

from observability.core.event_logger import EventLogger
from observability.core.schemas import (
    create_signal_event,
    create_gate_event,
    create_trade_exec_event
)
from observability.analytics.trade_journal import TradeJournal
from observability.analytics.learning_tracker import LearningTracker
from observability.analytics.insight_aggregator import InsightAggregator

logger = structlog.get_logger(__name__)


class ObservabilityHooks:
    """
    Easy integration hooks for observability.

    Add 3 lines to your simulation:
    1. Initialize hooks
    2. Call hooks.on_*() at key points
    3. Get daily summary

    That's it!
    """

    def __init__(self, enable_logging: bool = True):
        """
        Initialize hooks.

        Args:
            enable_logging: Enable event logging (set False for testing)
        """
        self.enable_logging = enable_logging

        if enable_logging:
            self.event_logger = EventLogger()
            self.event_logger.start()
        else:
            self.event_logger = None

        self.trade_journal = TradeJournal()
        self.learning_tracker = LearningTracker()
        self.insight_aggregator = InsightAggregator()

        # State tracking
        self.active_trades: Dict[str, Dict[str, Any]] = {}

        logger.info("observability_hooks_initialized", logging_enabled=enable_logging)

    def on_signal(
        self,
        signal_id: str,
        symbol: str,
        features: Dict[str, float],
        regime: str = "UNKNOWN",
        timestamp: Optional[str] = None
    ):
        """
        Hook: Signal received

        Args:
            signal_id: Unique signal identifier
            symbol: Trading symbol
            features: Feature vector
            regime: Market regime
            timestamp: ISO timestamp (defaults to now)
        """
        if timestamp is None:
            timestamp = datetime.utcnow().isoformat()

        if self.event_logger:
            event = create_signal_event(
                ts=timestamp,
                signal_id=signal_id,
                symbol=symbol,
                regime=regime,
                features=features
            )
            self.event_logger.log_event(event)

        logger.debug(
            "signal_received",
            signal_id=signal_id,
            symbol=symbol,
            regime=regime
        )

    def on_gate_decision(
        self,
        signal_id: str,
        gate_name: str,
        decision: str,
        inputs: Dict[str, float],
        context: Optional[Dict[str, Any]] = None,
        timestamp: Optional[str] = None
    ):
        """
        Hook: Gate decision

        Args:
            signal_id: Signal being evaluated
            gate_name: Name of gate
            decision: "PASS" or "FAIL"
            inputs: Gate inputs
            context: Additional context (thresholds, etc.)
            timestamp: ISO timestamp (defaults to now)
        """
        if timestamp is None:
            timestamp = datetime.utcnow().isoformat()

        if self.event_logger:
            event = create_gate_event(
                ts=timestamp,
                signal_id=signal_id,
                gate_name=gate_name,
                decision=decision,
                inputs=inputs,
                context=context or {}
            )
            self.event_logger.log_event(event)

        logger.debug(
            "gate_decision",
            signal_id=signal_id,
            gate=gate_name,
            decision=decision
        )

    def on_trade(
        self,
        trade_id: str,
        symbol: str,
        mode: str,
        entry_price: float,
        side: str = "long",
        size_gbp: float = 100.0,
        features: Optional[Dict[str, float]] = None,
        regime: str = "UNKNOWN",
        timestamp: Optional[str] = None
    ):
        """
        Hook: Trade executed (shadow trade)

        Args:
            trade_id: Unique trade identifier
            symbol: Trading symbol
            mode: "scalp" or "runner"
            entry_price: Entry price
            side: "long" or "short"
            size_gbp: Position size in GBP
            features: Entry features
            regime: Market regime
            timestamp: ISO timestamp (defaults to now)
        """
        if timestamp is None:
            timestamp = datetime.utcnow().isoformat()

        # Calculate size in asset terms (simplified)
        size_asset = size_gbp / entry_price

        # Estimate entry fee (0.1% maker fee)
        fee_entry_gbp = size_gbp * 0.001

        # Record in journal
        self.trade_journal.record_trade(
            trade_id=trade_id,
            ts_open=timestamp,
            symbol=symbol,
            mode=mode,
            regime=regime,
            side=side,
            entry_price=entry_price,
            size_gbp=size_gbp,
            size_asset=size_asset,
            fee_entry_gbp=fee_entry_gbp
        )

        # Track active trades
        self.active_trades[trade_id] = {
            'entry_ts': timestamp,
            'entry_price': entry_price,
            'symbol': symbol,
            'mode': mode,
            'features': features or {}
        }

        # Log event
        if self.event_logger:
            event = create_trade_exec_event(
                ts=timestamp,
                trade_id=trade_id,
                symbol=symbol,
                mode=mode,
                entry_price=entry_price,
                regime=regime
            )
            self.event_logger.log_event(event)

        logger.info(
            "trade_executed",
            trade_id=trade_id,
            symbol=symbol,
            mode=mode,
            price=entry_price
        )

    def on_trade_exit(
        self,
        trade_id: str,
        exit_price: float,
        exit_reason: str = "manual",
        timestamp: Optional[str] = None
    ):
        """
        Hook: Trade closed

        Args:
            trade_id: Trade identifier
            exit_price: Exit price
            exit_reason: "TP", "SL", "timeout", or "manual"
            timestamp: ISO timestamp (defaults to now)
        """
        if timestamp is None:
            timestamp = datetime.utcnow().isoformat()

        # Estimate exit fee (0.1% maker fee)
        # Get trade info to calculate fee
        trade_info = self.active_trades.get(trade_id)
        if trade_info:
            entry_price = trade_info.get('entry_price', exit_price)
            # Simplified: assume same GBP size
            fee_exit_gbp = 100.0 * 0.001
            # Estimate slippage
            slippage_bps = abs((exit_price - entry_price) / entry_price) * 10000 * 0.1  # 10% of move
        else:
            fee_exit_gbp = 0.1
            slippage_bps = 1.0

        # Close in journal
        self.trade_journal.close_trade(
            trade_id=trade_id,
            ts_close=timestamp,
            exit_price=exit_price,
            exit_reason=exit_reason,
            fee_exit_gbp=fee_exit_gbp,
            slippage_bps=slippage_bps
        )

        # Remove from active
        if trade_id in self.active_trades:
            del self.active_trades[trade_id]

        logger.info(
            "trade_closed",
            trade_id=trade_id,
            exit_reason=exit_reason
        )

    def on_training(
        self,
        model_id: str,
        samples_processed: int,
        metrics: Dict[str, float],
        feature_importance: Optional[Dict[str, float]] = None,
        timestamp: Optional[str] = None
    ):
        """
        Hook: Model training completed

        Args:
            model_id: Model identifier
            samples_processed: Number of samples
            metrics: Training metrics (auc, ece, etc.)
            feature_importance: Feature importance scores
            timestamp: ISO timestamp (defaults to now)
        """
        if timestamp is None:
            timestamp = datetime.utcnow().isoformat()

        # Record training session
        self.learning_tracker.record_training(
            model_id=model_id,
            samples_processed=samples_processed,
            metrics=metrics,
            feature_importance=feature_importance
        )

        logger.info(
            "training_completed",
            model_id=model_id[:16],
            samples=samples_processed,
            auc=metrics.get('auc', 0)
        )

    def get_daily_summary(self, date: Optional[str] = None) -> Dict[str, Any]:
        """
        Get daily summary of Engine activity.

        Args:
            date: Date string (YYYY-MM-DD), defaults to today

        Returns:
            Dict with summary, recommendations, and metrics
        """
        if date is None:
            date = datetime.utcnow().strftime("%Y-%m-%d")

        summary = self.insight_aggregator.get_daily_insights(date)

        logger.info("daily_summary_generated", date=date)

        return summary

    def get_stats(self, days: int = 7) -> Dict[str, Any]:
        """
        Get statistics for last N days.

        Args:
            days: Number of days

        Returns:
            Dict with trade statistics
        """
        return self.trade_journal.get_stats(days=days)

    def shutdown(self):
        """Shutdown hooks (flush logs)"""
        if self.event_logger:
            self.event_logger.stop()

        logger.info("observability_hooks_shutdown")

    def __enter__(self):
        """Context manager support"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup"""
        self.shutdown()


# Convenience: Global hooks instance
_global_hooks: Optional[ObservabilityHooks] = None


def get_hooks(enable_logging: bool = True) -> ObservabilityHooks:
    """Get or create global hooks instance"""
    global _global_hooks
    if _global_hooks is None:
        _global_hooks = ObservabilityHooks(enable_logging=enable_logging)
    return _global_hooks


def reset_hooks():
    """Reset global hooks (for testing)"""
    global _global_hooks
    if _global_hooks:
        _global_hooks.shutdown()
    _global_hooks = None


# Quick start example
if __name__ == '__main__':
    print("Observability Hooks - Quick Start Example")
    print("=" * 80)

    # Initialize
    hooks = ObservabilityHooks(enable_logging=False)  # Disable for demo

    # Simulate some activity
    print("\n1. Signal received")
    hooks.on_signal(
        signal_id="sig_001",
        symbol="ETH-USD",
        features={"momentum": 0.5, "volatility": 0.3},
        regime="TREND"
    )

    print("2. Gate decision")
    hooks.on_gate_decision(
        signal_id="sig_001",
        gate_name="meta_label",
        decision="PASS",
        inputs={"probability": 0.72},
        context={"threshold": 0.65}
    )

    print("3. Trade executed")
    hooks.on_trade(
        trade_id="trade_001",
        symbol="ETH-USD",
        mode="scalp",
        entry_price=2045.50,
        features={"momentum": 0.5},
        regime="TREND"
    )

    print("4. Trade closed")
    hooks.on_trade_exit(
        trade_id="trade_001",
        exit_price=2048.20,
        exit_reason="TP"
    )

    print("5. Model trained")
    hooks.on_training(
        model_id="model_v1",
        samples_processed=1000,
        metrics={"auc": 0.72, "ece": 0.08}
    )

    # Get stats
    print("\n6. Get statistics")
    stats = hooks.get_stats(days=7)
    print(f"   Total trades: {stats.get('total_trades', 0)}")
    print(f"   Win rate: {stats.get('win_rate', 0):.1%}")

    # Cleanup
    hooks.shutdown()

    print("\n✓ Observability hooks demo complete!")
    print("\nIntegration is as simple as:")
    print("  1. hooks = ObservabilityHooks()")
    print("  2. hooks.on_trade(...)")
    print("  3. summary = hooks.get_daily_summary()")
