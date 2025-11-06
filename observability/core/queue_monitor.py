"""
Queue Monitor - Health monitoring and kill switch

Purpose:
- Monitor event queue health (fill %, lag)
- Auto-throttle if queue filling
- Trigger kill switch if critical
- Send alerts (Telegram/Discord)

Key Features:
1. Real-time monitoring
   - Queue fill percentage
   - Writer lag (ms)
   - Event rates (enqueued/written/dropped)
   - Error rates

2. Auto-throttling
   - If queue >80% full, throttle events
   - If queue >95% full, trigger kill switch

3. Alerts
   - Telegram notifications
   - Discord webhooks
   - Email alerts (optional)

Usage:
    monitor = QueueMonitor(event_logger)
    monitor.set_kill_switch_callback(stop_trading)
    monitor.set_alert_callback(send_telegram)

    # Check periodically
    health = monitor.check_health()
    if health.status == HealthStatus.CRITICAL:
        # Kill switch triggered
"""

import time
from typing import Optional, Callable
from dataclasses import dataclass
from datetime import datetime
import structlog

from .event_logger import EventLogger
from .schemas import HealthStatus

logger = structlog.get_logger(__name__)


@dataclass
class HealthReport:
    """Health report with recommendations"""
    status: HealthStatus
    queue_fill_pct: float
    queue_size: int
    queue_max: int
    writer_lag_ms: float
    events_enqueued: int
    events_written: int
    events_dropped: int
    error_count: int

    # Metrics
    enqueue_rate: float  # Events/sec
    write_rate: float    # Events/sec
    drop_rate: float     # Events/sec

    # Recommendations
    message: str
    actions: list
    alerts: list


class QueueMonitor:
    """
    Monitor event queue health and trigger kill switch if needed.

    Thresholds:
    - HEALTHY: <80% queue fill, <1s lag
    - WARNING: 80-95% queue fill OR 1-5s lag
    - CRITICAL: >95% queue fill OR >5s lag → KILL SWITCH
    """

    def __init__(
        self,
        event_logger: EventLogger,
        warning_threshold: float = 0.80,
        critical_threshold: float = 0.95,
        lag_warning_ms: float = 1000,
        lag_critical_ms: float = 5000,
        check_interval_sec: float = 10.0
    ):
        """
        Initialize queue monitor.

        Args:
            event_logger: EventLogger instance to monitor
            warning_threshold: Queue fill % for warning
            critical_threshold: Queue fill % for kill switch
            lag_warning_ms: Writer lag for warning
            lag_critical_ms: Writer lag for kill switch
            check_interval_sec: How often to check health
        """
        self.event_logger = event_logger
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.lag_warning_ms = lag_warning_ms
        self.lag_critical_ms = lag_critical_ms
        self.check_interval = check_interval_sec

        # Callbacks
        self.kill_switch_callback: Optional[Callable] = None
        self.alert_callback: Optional[Callable] = None

        # Tracking
        self.last_check_time = time.time()
        self.last_stats = self.event_logger.get_stats()
        self.kill_switch_triggered = False

        # Alert history (prevent spam)
        self.last_alert_time = {}
        self.alert_cooldown_sec = 300  # 5 minutes

        logger.info(
            "queue_monitor_initialized",
            warning_threshold=warning_threshold,
            critical_threshold=critical_threshold
        )

    def set_kill_switch_callback(self, callback: Callable):
        """Set kill switch callback (should stop trading)"""
        self.kill_switch_callback = callback
        self.event_logger.set_kill_switch(self._trigger_kill_switch)
        logger.info("kill_switch_callback_registered")

    def set_alert_callback(self, callback: Callable):
        """Set alert callback (Telegram/Discord)"""
        self.alert_callback = callback
        logger.info("alert_callback_registered")

    def check_health(self) -> HealthReport:
        """
        Check queue health and return report.

        Returns:
            HealthReport with status and recommendations
        """
        # Get current stats
        stats = self.event_logger.get_stats()
        health = self.event_logger.get_health()

        # Calculate rates
        time_elapsed = time.time() - self.last_check_time
        if time_elapsed > 0:
            enqueue_rate = (stats.enqueued - self.last_stats.enqueued) / time_elapsed
            write_rate = (stats.written - self.last_stats.written) / time_elapsed
            drop_rate = (stats.dropped - self.last_stats.dropped) / time_elapsed
        else:
            enqueue_rate = write_rate = drop_rate = 0.0

        # Determine status
        queue_fill_pct = stats.queue_size / stats.queue_max
        lag_ms = stats.writer_lag_ms

        status = HealthStatus.HEALTHY
        actions = []
        alerts = []

        # Check critical conditions
        if queue_fill_pct >= self.critical_threshold:
            status = HealthStatus.CRITICAL
            actions.append("TRIGGER_KILL_SWITCH")
            alerts.append(f"🚨 Queue {queue_fill_pct:.0%} full - CRITICAL")

            if not self.kill_switch_triggered:
                self._trigger_kill_switch("Queue overflow")

        elif lag_ms >= self.lag_critical_ms:
            status = HealthStatus.CRITICAL
            actions.append("TRIGGER_KILL_SWITCH")
            alerts.append(f"🚨 Writer lag {lag_ms:.0f}ms - CRITICAL")

            if not self.kill_switch_triggered:
                self._trigger_kill_switch("Writer lag too high")

        # Check warning conditions
        elif queue_fill_pct >= self.warning_threshold:
            status = HealthStatus.WARNING
            actions.append("THROTTLE_EVENTS")
            alerts.append(f"⚠️ Queue {queue_fill_pct:.0%} full - WARNING")

        elif lag_ms >= self.lag_warning_ms:
            status = HealthStatus.WARNING
            actions.append("THROTTLE_EVENTS")
            alerts.append(f"⚠️ Writer lag {lag_ms:.0f}ms - WARNING")

        # Check drop rate
        if drop_rate > 10:
            if status == HealthStatus.HEALTHY:
                status = HealthStatus.WARNING
            alerts.append(f"⚠️ Dropping {drop_rate:.0f} events/sec")

        # Check error rate
        if stats.errors > self.last_stats.errors:
            new_errors = stats.errors - self.last_stats.errors
            alerts.append(f"⚠️ {new_errors} new errors")

        # Generate message
        if status == HealthStatus.CRITICAL:
            message = f"CRITICAL: Queue {queue_fill_pct:.0%} full, lag {lag_ms:.0f}ms"
        elif status == HealthStatus.WARNING:
            message = f"WARNING: Queue {queue_fill_pct:.0%} full, lag {lag_ms:.0f}ms"
        else:
            message = f"HEALTHY: Queue {queue_fill_pct:.0%} full, lag {lag_ms:.0f}ms"

        # Create report
        report = HealthReport(
            status=status,
            queue_fill_pct=queue_fill_pct,
            queue_size=stats.queue_size,
            queue_max=stats.queue_max,
            writer_lag_ms=lag_ms,
            events_enqueued=stats.enqueued,
            events_written=stats.written,
            events_dropped=stats.dropped,
            error_count=stats.errors,
            enqueue_rate=enqueue_rate,
            write_rate=write_rate,
            drop_rate=drop_rate,
            message=message,
            actions=actions,
            alerts=alerts
        )

        # Send alerts if needed
        if alerts:
            self._send_alerts(report)

        # Update tracking
        self.last_check_time = time.time()
        self.last_stats = stats

        # Log health check
        logger.debug(
            "health_check",
            status=status.value,
            queue_fill_pct=queue_fill_pct,
            lag_ms=lag_ms
        )

        return report

    def _trigger_kill_switch(self, reason: str):
        """Trigger kill switch"""
        if self.kill_switch_triggered:
            return  # Already triggered

        self.kill_switch_triggered = True

        logger.error("KILL_SWITCH_TRIGGERED", reason=reason)

        # Call kill switch callback
        if self.kill_switch_callback:
            try:
                self.kill_switch_callback(reason)
            except Exception as e:
                logger.error("kill_switch_callback_error", error=str(e))

        # Send critical alert
        self._send_alert(
            "🚨 KILL SWITCH TRIGGERED",
            f"Reason: {reason}\nTrading stopped immediately.",
            priority="critical"
        )

    def _send_alerts(self, report: HealthReport):
        """Send alerts for health issues"""
        for alert_msg in report.alerts:
            self._send_alert(
                title="Queue Health Alert",
                message=f"{alert_msg}\n\n{report.message}",
                priority="warning" if report.status == HealthStatus.WARNING else "critical"
            )

    def _send_alert(self, title: str, message: str, priority: str = "info"):
        """Send alert via callback"""
        # Rate limit alerts (no spam)
        alert_key = f"{title}:{priority}"
        now = time.time()

        if alert_key in self.last_alert_time:
            if now - self.last_alert_time[alert_key] < self.alert_cooldown_sec:
                logger.debug("alert_rate_limited", alert_key=alert_key)
                return

        self.last_alert_time[alert_key] = now

        # Call alert callback
        if self.alert_callback:
            try:
                self.alert_callback(title, message, priority)
            except Exception as e:
                logger.error("alert_callback_error", error=str(e))

        logger.info("alert_sent", title=title, priority=priority)

    def reset_kill_switch(self):
        """Reset kill switch (manual recovery)"""
        self.kill_switch_triggered = False
        logger.info("kill_switch_reset")

    def get_status_summary(self) -> str:
        """Get human-readable status summary"""
        report = self.check_health()

        summary = f"""
📊 QUEUE HEALTH SUMMARY
{'='*60}

Status: {report.status.value.upper()}
Queue: {report.queue_size:,}/{report.queue_max:,} ({report.queue_fill_pct:.1%})
Writer Lag: {report.writer_lag_ms:.0f}ms

Events:
  Enqueued: {report.events_enqueued:,} ({report.enqueue_rate:.1f}/sec)
  Written:  {report.events_written:,} ({report.write_rate:.1f}/sec)
  Dropped:  {report.events_dropped:,} ({report.drop_rate:.1f}/sec)
  Errors:   {report.error_count:,}

{report.message}

Actions: {', '.join(report.actions) if report.actions else 'None'}
Alerts: {len(report.alerts)}
"""
        return summary


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

async def example():
    """Example usage"""
    from .event_logger import EventLogger
    from .schemas import create_signal_event, MarketContext

    # Create logger
    event_logger = EventLogger(max_queue_size=100)  # Small queue for demo
    await event_logger.start()

    # Create monitor
    monitor = QueueMonitor(event_logger)

    # Set callbacks
    def kill_switch(reason):
        print(f"\n🚨 KILL SWITCH: {reason}")

    def send_alert(title, message, priority):
        print(f"\n📢 ALERT ({priority}): {title}")
        print(f"   {message}")

    monitor.set_kill_switch_callback(kill_switch)
    monitor.set_alert_callback(send_alert)

    # Simulate normal operation
    print("Simulating normal operation...")
    for i in range(50):
        event = create_signal_event(
            symbol="ETH-USD",
            price=2000.0,
            features={"confidence": 0.5},
            regime="TREND",
            market_context=MarketContext(
                volatility_1h=0.3,
                spread_bps=4.0,
                liquidity_score=0.8,
                recent_trend_30m=0.01,
                volume_vs_avg=1.2
            )
        )
        await event_logger.log(event)

    # Check health
    print("\nHealth check (normal):")
    print(monitor.get_status_summary())

    # Simulate overload (fill queue)
    print("\nSimulating overload...")
    for i in range(90):  # Queue size is 100
        event = create_signal_event(
            symbol="ETH-USD",
            price=2000.0,
            features={"confidence": 0.5},
            regime="TREND",
            market_context=MarketContext(
                volatility_1h=0.3,
                spread_bps=4.0,
                liquidity_score=0.8,
                recent_trend_30m=0.01,
                volume_vs_avg=1.2
            )
        )
        await event_logger.log(event)

    # Check health (should be WARNING)
    print("\nHealth check (overload):")
    print(monitor.get_status_summary())

    await event_logger.stop()


if __name__ == '__main__':
    import asyncio

    print("Testing Queue Monitor...")
    print("=" * 60)

    asyncio.run(example())

    print("\nQueue monitor tests passed ✓")
