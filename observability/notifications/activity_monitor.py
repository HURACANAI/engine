"""
Real-Time Engine Activity Monitor

Monitors and reports EVERYTHING the Engine does:
- Every trade (entry/exit)
- Every learning update
- Every error
- Every gate decision
- Every model update
- Performance metrics
- System health

Integrates with Telegram for real-time notifications.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import structlog  # type: ignore
import asyncio

from observability.notifications.telegram_monitor import ComprehensiveTelegramMonitor, NotificationLevel
from observability.analytics.learning_tracker import LearningTracker
from observability.analytics.trade_journal import TradeJournal
from observability.analytics.metrics_computer import MetricsComputer
from observability.analytics.model_evolution import ModelEvolutionTracker
from observability.core.event_logger import EventLogger

logger = structlog.get_logger(__name__)


@dataclass
class ActivitySummary:
    """Activity summary for a time period."""
    period: str  # "last_hour", "last_day"
    trades_executed: int
    trades_closed: int
    learning_updates: int
    errors: int
    gate_decisions: int
    model_updates: int
    performance: Dict[str, float]


class RealTimeActivityMonitor:
    """
    Real-time activity monitor that tracks EVERYTHING the Engine does.
    
    Features:
    - Real-time event monitoring
    - Automatic Telegram notifications
    - Activity summaries
    - Error tracking
    - Performance monitoring
    """

    def __init__(
        self,
        telegram_monitor: ComprehensiveTelegramMonitor,
        event_logger: EventLogger,
        learning_tracker: LearningTracker,
        trade_journal: TradeJournal,
        metrics_computer: MetricsComputer,
        model_tracker: ModelEvolutionTracker,
    ):
        """
        Initialize activity monitor.
        
        Args:
            telegram_monitor: Telegram notification bot
            event_logger: Event logger for capturing events
            learning_tracker: Learning progress tracker
            trade_journal: Trade history journal
            metrics_computer: Metrics calculator
            model_tracker: Model evolution tracker
        """
        self.telegram_monitor = telegram_monitor
        self.event_logger = event_logger
        self.learning_tracker = learning_tracker
        self.trade_journal = trade_journal
        self.metrics_computer = metrics_computer
        self.model_tracker = model_tracker
        
        # Activity tracking
        self.recent_trades: List[Dict[str, Any]] = []
        self.recent_errors: List[Dict[str, Any]] = []
        self.recent_learning: List[Dict[str, Any]] = []
        
        # Performance tracking
        self.last_hourly_summary = datetime.now()
        self.last_daily_summary = datetime.now()
        
        # Event-driven monitoring
        self.pending_notifications: List[Dict[str, Any]] = []
        self.notification_batch_size = 5
        self.last_batch_send = datetime.now()
        self.batch_interval = timedelta(seconds=30)  # Batch every 30 seconds
        
        # Event queue for event-driven updates
        self.event_queue: asyncio.Queue = asyncio.Queue()
        self.monitoring_active = False
        
        logger.info("real_time_activity_monitor_initialized")

    async def monitor_continuously(self):
        """Continuously monitor Engine activity (event-driven)."""
        logger.info("starting_continuous_monitoring")
        self.monitoring_active = True
        
        # Start event processor
        event_processor = asyncio.create_task(self._process_events())
        
        while self.monitoring_active:
            try:
                # Check for new events (less frequent now - event-driven)
                await self._check_trades()
                await self._check_learning()
                await self._check_errors()
                await self._check_performance()
                await self._check_model_updates()
                
                # Send batched notifications if due
                await self._send_batched_notifications_if_due()
                
                # Send summaries
                await self._send_hourly_summary_if_due()
                await self._send_daily_summary_if_due()
                
                # Wait before next check (longer interval - event-driven)
                await asyncio.sleep(30)  # Check every 30 seconds (was 10)
                
            except Exception as e:
                logger.error("monitoring_error", error=str(e))
                self.telegram_monitor.notify_error(
                    error_type="Monitoring Error",
                    error_message=str(e),
                    context="Real-time activity monitoring",
                    severity="high",
                )
                await asyncio.sleep(60)  # Wait longer on error
        
        # Cancel event processor
        event_processor.cancel()

    async def _process_events(self):
        """Process events from queue (event-driven)."""
        while self.monitoring_active:
            try:
                # Wait for event with timeout
                event = await asyncio.wait_for(self.event_queue.get(), timeout=1.0)
                
                # Process event immediately
                if event.get('type') == 'trade_executed':
                    await self._handle_trade_executed(event)
                elif event.get('type') == 'trade_exited':
                    await self._handle_trade_exited(event)
                elif event.get('type') == 'learning_update':
                    await self._handle_learning_update(event)
                elif event.get('type') == 'error':
                    await self._handle_error(event)
                
            except asyncio.TimeoutError:
                continue  # No events, continue
            except Exception as e:
                logger.error("event_processing_error", error=str(e))

    async def _handle_trade_executed(self, event: Dict[str, Any]):
        """Handle trade executed event."""
        trade = event.get('data', {})
        self.pending_notifications.append({
            'type': 'trade_executed',
            'data': trade,
        })

    async def _handle_trade_exited(self, event: Dict[str, Any]):
        """Handle trade exited event."""
        trade = event.get('data', {})
        self.pending_notifications.append({
            'type': 'trade_exited',
            'data': trade,
        })

    async def _handle_learning_update(self, event: Dict[str, Any]):
        """Handle learning update event."""
        update = event.get('data', {})
        self.pending_notifications.append({
            'type': 'learning_update',
            'data': update,
        })

    async def _handle_error(self, event: Dict[str, Any]):
        """Handle error event (send immediately)."""
        error = event.get('data', {})
        self.telegram_monitor.notify_error(
            error_type=error.get('error_type', 'Error'),
            error_message=error.get('error_message', ''),
            context=error.get('context', ''),
            severity=error.get('severity', 'medium'),
        )

    async def _send_batched_notifications_if_due(self):
        """Send batched notifications if due."""
        now = datetime.now()
        
        # Check if batch interval has passed or batch is full
        if (now - self.last_batch_send) >= self.batch_interval or len(self.pending_notifications) >= self.notification_batch_size:
            if self.pending_notifications:
                # Send batched notifications
                self.telegram_monitor.batch_notify(
                    notifications=self.pending_notifications,
                    max_batch_size=self.notification_batch_size,
                )
                
                # Clear pending
                self.pending_notifications = []
                self.last_batch_send = now

    def trigger_event(self, event_type: str, data: Dict[str, Any]):
        """Trigger an event (non-blocking)."""
        if self.monitoring_active:
            try:
                self.event_queue.put_nowait({
                    'type': event_type,
                    'data': data,
                    'timestamp': datetime.now(),
                })
            except asyncio.QueueFull:
                logger.warning("event_queue_full", event_type=event_type)

    async def _check_trades(self):
        """Check for new trades."""
        # Get recent trades from journal
        recent_trades = self.trade_journal.get_recent_trades(limit=10)
        
        for trade in recent_trades:
            # Check if we've already notified about this trade
            trade_id = trade.get('trade_id')
            if trade_id and trade_id not in [t.get('trade_id') for t in self.recent_trades]:
                # New trade - notify
                if trade.get('status') == 'open':
                    self.telegram_monitor.notify_trade_executed(
                        symbol=trade.get('symbol', 'N/A'),
                        direction=trade.get('direction', 'buy'),
                        entry_price=trade.get('entry_price', 0.0),
                        size_gbp=trade.get('size_gbp', 0.0),
                        confidence=trade.get('confidence', 0.0),
                        technique=trade.get('technique', 'N/A'),
                        regime=trade.get('regime', 'N/A'),
                    )
                elif trade.get('status') == 'closed':
                    self.telegram_monitor.notify_trade_exited(
                        symbol=trade.get('symbol', 'N/A'),
                        direction=trade.get('direction', 'buy'),
                        entry_price=trade.get('entry_price', 0.0),
                        exit_price=trade.get('exit_price', 0.0),
                        pnl_gbp=trade.get('pnl_gbp', 0.0),
                        pnl_bps=trade.get('pnl_bps', 0.0),
                        hold_duration_minutes=trade.get('hold_duration_minutes', 0),
                        reason=trade.get('exit_reason', 'N/A'),
                    )
                
                self.recent_trades.append(trade)
                
                # Keep only last 100
                if len(self.recent_trades) > 100:
                    self.recent_trades.pop(0)

    async def _check_learning(self):
        """Check for learning updates."""
        # Get recent learning updates
        recent_learning = self.learning_tracker.get_recent_updates(limit=5)
        
        for update in recent_learning:
            update_id = update.get('update_id')
            if update_id and update_id not in [u.get('update_id') for u in self.recent_learning]:
                # New learning update - notify
                self.telegram_monitor.notify_learning_update(
                    what_learned=update.get('what_learned', 'N/A'),
                    impact=update.get('impact', 'N/A'),
                    confidence_change=update.get('confidence_change', 0.0),
                    feature_importance_changes=update.get('feature_changes', {}),
                )
                
                self.recent_learning.append(update)
                
                # Keep only last 50
                if len(self.recent_learning) > 50:
                    self.recent_learning.pop(0)

    async def _check_errors(self):
        """Check for errors."""
        # Get recent errors from event logger
        # This would need to be implemented in event_logger
        # For now, errors are caught and notified directly

    async def _check_performance(self):
        """Check performance metrics."""
        # Calculate current performance
        metrics = self.metrics_computer.compute_all_metrics()
        
        # Check for performance degradation
        if metrics.get('win_rate', 0.0) < 0.50:
            self.telegram_monitor.notify_health_alert(
                alert_type="Performance Degradation",
                severity="high",
                message=f"Win rate dropped to {metrics.get('win_rate', 0):.1%}",
                action_required=True,
            )

    async def _check_model_updates(self):
        """Check for model updates."""
        # Get recent model updates
        recent_updates = self.model_tracker.get_recent_updates(limit=5)
        
        for update in recent_updates:
            update_id = update.get('update_id')
            if update_id:
                # Notify model update
                self.telegram_monitor.notify_model_update(
                    symbol=update.get('symbol', 'N/A'),
                    old_metrics=update.get('old_metrics', {}),
                    new_metrics=update.get('new_metrics', {}),
                    improvement=update.get('improvement', {}),
                )

    async def _send_hourly_summary_if_due(self):
        """Send hourly summary if due."""
        now = datetime.now()
        if (now - self.last_hourly_summary).total_seconds() >= 3600:  # 1 hour
            # Calculate hourly performance
            trades = self.trade_journal.get_trades_in_period(
                start_time=now - timedelta(hours=1),
                end_time=now,
            )
            
            if trades:
                wins = sum(1 for t in trades if t.get('pnl_gbp', 0) > 0)
                losses = len(trades) - wins
                win_rate = wins / len(trades) if trades else 0.0
                total_pnl = sum(t.get('pnl_gbp', 0) for t in trades)
                total_pnl_bps = sum(t.get('pnl_bps', 0) for t in trades)
                
                # Calculate Sharpe (simplified)
                sharpe = 0.0  # Would calculate from returns
                
                best_trade = max(trades, key=lambda x: x.get('pnl_gbp', 0)) if trades else None
                worst_trade = min(trades, key=lambda x: x.get('pnl_gbp', 0)) if trades else None
                
                self.telegram_monitor.notify_performance_summary(
                    period="hourly",
                    total_trades=len(trades),
                    wins=wins,
                    losses=losses,
                    win_rate=win_rate,
                    total_pnl_gbp=total_pnl,
                    total_pnl_bps=total_pnl_bps,
                    sharpe_ratio=sharpe,
                    best_trade=best_trade,
                    worst_trade=worst_trade,
                )
            
            self.last_hourly_summary = now

    async def _send_daily_summary_if_due(self):
        """Send daily summary if due."""
        now = datetime.now()
        if (now - self.last_daily_summary).total_seconds() >= 86400:  # 24 hours
            # Calculate daily performance
            trades = self.trade_journal.get_trades_in_period(
                start_time=now - timedelta(days=1),
                end_time=now,
            )
            
            wins = sum(1 for t in trades if t.get('pnl_gbp', 0) > 0)
            win_rate = wins / len(trades) if trades else 0.0
            total_pnl = sum(t.get('pnl_gbp', 0) for t in trades)
            
            # Get learning updates
            learning_updates = self.learning_tracker.get_recent_updates(limit=10)
            what_learned = [u.get('what_learned', 'N/A') for u in learning_updates]
            
            # Get model updates
            model_updates = self.model_tracker.get_recent_updates(limit=10)
            num_model_updates = len(model_updates)
            
            # Get errors
            num_errors = len(self.recent_errors)
            
            # Top performers
            top_performers = sorted(trades, key=lambda x: x.get('pnl_gbp', 0), reverse=True)[:3]
            
            # Issues
            issues = []
            if win_rate < 0.50:
                issues.append(f"Low win rate: {win_rate:.1%}")
            if num_errors > 10:
                issues.append(f"High error count: {num_errors}")
            
            # Calculate Sharpe
            sharpe = 0.0  # Would calculate from returns
            
            self.telegram_monitor.notify_daily_summary(
                date=now.strftime('%Y-%m-%d'),
                total_trades=len(trades),
                win_rate=win_rate,
                total_pnl_gbp=total_pnl,
                sharpe_ratio=sharpe,
                what_learned=what_learned,
                model_updates=num_model_updates,
                errors=num_errors,
                top_performers=top_performers,
                issues=issues,
            )
            
            self.last_daily_summary = now

    def record_error(
        self,
        error_type: str,
        error_message: str,
        context: str,
        severity: str = "medium",
    ):
        """Record an error and notify."""
        error_record = {
            'error_type': error_type,
            'error_message': error_message,
            'context': context,
            'severity': severity,
            'timestamp': datetime.now(),
        }
        
        self.recent_errors.append(error_record)
        
        # Notify via Telegram
        self.telegram_monitor.notify_error(
            error_type=error_type,
            error_message=error_message,
            context=context,
            severity=severity,
            action_required=(severity == "critical"),
        )
        
        # Keep only last 100 errors
        if len(self.recent_errors) > 100:
            self.recent_errors.pop(0)

    def record_gate_decision(
        self,
        symbol: str,
        direction: str,
        confidence: float,
        gates_passed: List[str],
        gates_blocked: List[str],
        reason: str,
        counterfactual: Optional[Dict[str, Any]] = None,
    ):
        """Record gate decision and notify."""
        self.telegram_monitor.notify_gate_decision(
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            gates_passed=gates_passed,
            gates_blocked=gates_blocked,
            reason=reason,
            counterfactual=counterfactual,
        )

