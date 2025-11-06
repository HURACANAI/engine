"""
Comprehensive Telegram Monitoring for Engine

Real-time monitoring and notifications for:
- Every trade executed (with details)
- Every learning update (what it learned)
- Every error (with explanations)
- Performance summaries (hourly/daily)
- Model updates (when models change)
- Gate decisions (why trades were blocked)
- System health (alerts and warnings)
- Validation failures (OOS, overfitting, data quality)
- Health check results

Features:
- Simple, easy-to-understand messages
- Real-time updates
- Rich formatting with emojis
- Actionable insights
- Error explanations
- File logging integration
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional
from pathlib import Path

import requests
import structlog

logger = structlog.get_logger(__name__)


class NotificationLevel(Enum):
    """Notification priority levels."""
    CRITICAL = "critical"  # Immediate alert
    HIGH = "high"  # Important update
    MEDIUM = "medium"  # Regular update
    LOW = "low"  # Info only


@dataclass
class TelegramNotification:
    """Telegram notification message."""
    level: NotificationLevel
    title: str
    message: str
    emoji: str
    action_required: bool = False
    details: Optional[Dict[str, Any]] = None


class ComprehensiveTelegramMonitor:
    """
    Comprehensive Telegram monitoring bot for Engine.
    
    Sends real-time updates about:
    - Every trade (entry, exit, P&L)
    - Learning updates (what it learned)
    - Errors (with explanations)
    - Performance (hourly/daily summaries)
    - Model changes (when models update)
    - Gate decisions (why trades blocked)
    - System health (alerts)
    - Validation failures (OOS, overfitting, data quality)
    """

    def __init__(
        self,
        bot_token: str,
        chat_id: str,
        log_file: Optional[Path] = None,
        enable_trade_notifications: bool = True,
        enable_learning_notifications: bool = True,
        enable_error_notifications: bool = True,
        enable_performance_summaries: bool = True,
        enable_model_updates: bool = True,
        enable_gate_decisions: bool = True,
        enable_health_alerts: bool = True,
        enable_validation_alerts: bool = True,
        min_notification_level: NotificationLevel = NotificationLevel.LOW,
    ):
        """
        Initialize Telegram monitor.
        
        Args:
            bot_token: Telegram bot token
            chat_id: Telegram chat ID
            log_file: Optional file path to log all notifications
            enable_*: Enable/disable specific notification types
            min_notification_level: Minimum level to send (filters out low-priority)
        """
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.api_url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        self.log_file = log_file
        
        self.enable_trade_notifications = enable_trade_notifications
        self.enable_learning_notifications = enable_learning_notifications
        self.enable_error_notifications = enable_error_notifications
        self.enable_performance_summaries = enable_performance_summaries
        self.enable_model_updates = enable_model_updates
        self.enable_gate_decisions = enable_gate_decisions
        self.enable_health_alerts = enable_health_alerts
        self.enable_validation_alerts = enable_validation_alerts
        self.min_notification_level = min_notification_level
        
        # Rate limiting (max 20 messages per minute)
        self.message_times: List[datetime] = []
        self.max_messages_per_minute = 20
        
        # Notification history for logging
        self.notification_history: List[TelegramNotification] = []
        
        # Initialize log file
        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            self._log_to_file("=" * 80)
            self._log_to_file(f"ENGINE MONITORING STARTED - {datetime.now().isoformat()}")
            self._log_to_file("=" * 80)
        
        logger.info(
            "comprehensive_telegram_monitor_initialized",
            chat_id=chat_id,
            log_file=str(log_file) if log_file else None,
        )

    def notify_trade_executed(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        size_gbp: float,
        confidence: float,
        technique: str,
        regime: str,
    ):
        """Notify about trade execution."""
        if not self.enable_trade_notifications:
            return
        
        emoji = "🟢" if direction == "buy" else "🔴"
        message = (
            f"*Trade Executed*\n\n"
            f"Symbol: {symbol}\n"
            f"Direction: {direction.upper()}\n"
            f"Entry Price: ${entry_price:,.2f}\n"
            f"Size: £{size_gbp:,.2f}\n"
            f"Confidence: {confidence:.1%}\n"
            f"Technique: {technique}\n"
            f"Regime: {regime}"
        )
        
        self._send_notification(
            NotificationLevel.MEDIUM,
            "Trade Executed",
            message,
            emoji,
            action_required=False,
        )

    def notify_trade_closed(
        self,
        symbol: str,
        entry_price: float,
        exit_price: float,
        size_gbp: float,
        profit_gbp: float,
        profit_bps: float,
        duration_minutes: int,
        won: bool,
    ):
        """Notify about trade closure."""
        if not self.enable_trade_notifications:
            return
        
        emoji = "✅" if won else "❌"
        profit_pct = (profit_gbp / size_gbp) * 100 if size_gbp > 0 else 0
        
        message = (
            f"*Trade Closed*\n\n"
            f"Symbol: {symbol}\n"
            f"Entry: ${entry_price:,.2f}\n"
            f"Exit: ${exit_price:,.2f}\n"
            f"Size: £{size_gbp:,.2f}\n"
            f"Profit: £{profit_gbp:,.2f} ({profit_pct:.2f}%)\n"
            f"Profit: {profit_bps:.1f} bps\n"
            f"Duration: {duration_minutes} min\n"
            f"Result: {'WIN' if won else 'LOSS'}"
        )
        
        self._send_notification(
            NotificationLevel.MEDIUM,
            "Trade Closed",
            message,
            emoji,
            action_required=False,
        )

    def notify_learning_update(
        self,
        symbol: str,
        what_learned: str,
        feature_importance_changes: Optional[Dict[str, float]] = None,
        model_improvements: Optional[Dict[str, float]] = None,
    ):
        """Notify about learning updates."""
        if not self.enable_learning_notifications:
            return
        
        message = f"*Learning Update - {symbol}*\n\n{what_learned}\n"
        
        if feature_importance_changes:
            message += "\n*Feature Changes:*\n"
            for feature, change in list(feature_importance_changes.items())[:5]:
                direction = "📈" if change > 0 else "📉"
                message += f"{direction} {feature}: {change:+.2f}\n"
        
        if model_improvements:
            message += "\n*Model Improvements:*\n"
            for metric, improvement in model_improvements.items():
                message += f"✅ {metric}: {improvement:+.4f}\n"
        
        self._send_notification(
            NotificationLevel.LOW,
            "Learning Update",
            message,
            "🧠",
            action_required=False,
        )

    def notify_error(
        self,
        error_type: str,
        error_message: str,
        symbol: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        """Notify about errors."""
        if not self.enable_error_notifications:
            return
        
        message = f"*Error Detected*\n\n"
        message += f"Type: {error_type}\n"
        message += f"Message: {error_message}\n"
        
        if symbol:
            message += f"Symbol: {symbol}\n"
        
        if context:
            message += "\n*Context:*\n"
            for key, value in list(context.items())[:5]:
                message += f"{key}: {value}\n"
        
        self._send_notification(
            NotificationLevel.HIGH,
            "Error Detected",
            message,
            "🚨",
            action_required=True,
        )

    def notify_validation_failure(
        self,
        validation_type: str,
        reason: str,
        symbol: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        """Notify about validation failures."""
        if not self.enable_validation_alerts:
            return
        
        message = f"*Validation Failed*\n\n"
        message += f"Type: {validation_type}\n"
        message += f"Reason: {reason}\n"
        
        if symbol:
            message += f"Symbol: {symbol}\n"
        
        if details:
            message += "\n*Details:*\n"
            for key, value in list(details.items())[:5]:
                message += f"{key}: {value}\n"
        
        message += "\n⚠️ *Action Required:* Fix before deploying!"
        
        self._send_notification(
            NotificationLevel.CRITICAL,
            "Validation Failed",
            message,
            "🚨",
            action_required=True,
        )

    def notify_health_check(
        self,
        status: str,
        alerts: List[Dict[str, Any]],
        warnings: List[Dict[str, Any]],
        healthy_services: int,
        total_services: int,
    ):
        """Notify about health check results."""
        if not self.enable_health_alerts:
            return
        
        emoji = "✅" if status == "HEALTHY" else "⚠️" if status == "DEGRADED" else "🚨"
        
        message = f"*Health Check*\n\n"
        message += f"Status: {status}\n"
        message += f"Services: {healthy_services}/{total_services} healthy\n"
        
        if alerts:
            message += f"\n🚨 *Alerts ({len(alerts)}):*\n"
            for alert in alerts[:3]:
                message += f"• {alert.get('message', 'Unknown')}\n"
        
        if warnings:
            message += f"\n⚠️ *Warnings ({len(warnings)}):*\n"
            for warning in warnings[:3]:
                message += f"• {warning.get('message', 'Unknown')}\n"
        
        level = NotificationLevel.CRITICAL if alerts else NotificationLevel.HIGH if warnings else NotificationLevel.MEDIUM
        
        self._send_notification(
            level,
            "Health Check",
            message,
            emoji,
            action_required=len(alerts) > 0,
        )

    def notify_performance_summary(
        self,
        period: str,
        total_trades: int,
        win_rate: float,
        sharpe: float,
        profit_gbp: float,
        best_symbol: Optional[str] = None,
        worst_symbol: Optional[str] = None,
    ):
        """Notify about performance summary."""
        if not self.enable_performance_summaries:
            return
        
        emoji = "📊"
        message = f"*Performance Summary - {period}*\n\n"
        message += f"Total Trades: {total_trades}\n"
        message += f"Win Rate: {win_rate:.1%}\n"
        message += f"Sharpe Ratio: {sharpe:.2f}\n"
        message += f"Profit: £{profit_gbp:,.2f}\n"
        
        if best_symbol:
            message += f"\nBest: {best_symbol}\n"
        if worst_symbol:
            message += f"Worst: {worst_symbol}\n"
        
        self._send_notification(
            NotificationLevel.MEDIUM,
            "Performance Summary",
            message,
            emoji,
            action_required=False,
        )

    def notify_model_update(
        self,
        symbol: str,
        model_id: str,
        improvements: Dict[str, float],
        metrics: Dict[str, float],
    ):
        """Notify about model updates."""
        if not self.enable_model_updates:
            return
        
        message = f"*Model Updated - {symbol}*\n\n"
        message += f"Model ID: {model_id}\n"
        message += "\n*Improvements:*\n"
        for metric, improvement in improvements.items():
            message += f"✅ {metric}: {improvement:+.4f}\n"
        message += "\n*Current Metrics:*\n"
        for metric, value in metrics.items():
            message += f"{metric}: {value:.4f}\n"
        
        self._send_notification(
            NotificationLevel.MEDIUM,
            "Model Updated",
            message,
            "🔄",
            action_required=False,
        )

    def notify_gate_decision(
        self,
        symbol: str,
        passed: bool,
        reason: str,
        blocked_gates: Optional[List[str]] = None,
    ):
        """Notify about gate decisions."""
        if not self.enable_gate_decisions:
            return
        
        emoji = "✅" if passed else "🚫"
        message = f"*Gate Decision - {symbol}*\n\n"
        message += f"Result: {'PASSED' if passed else 'BLOCKED'}\n"
        message += f"Reason: {reason}\n"
        
        if blocked_gates:
            message += f"\nBlocked Gates: {', '.join(blocked_gates)}\n"
        
        self._send_notification(
            NotificationLevel.LOW,
            "Gate Decision",
            message,
            emoji,
            action_required=False,
        )

    def notify_system_startup(
        self,
        symbols: List[str],
        total_coins: int,
    ):
        """Notify about system startup."""
        message = f"*🚀 Engine Started*\n\n"
        message += f"Training on {total_coins} coins\n"
        message += f"Symbols: {', '.join(symbols[:10])}"
        if len(symbols) > 10:
            message += f" ... and {len(symbols) - 10} more"
        
        self._send_notification(
            NotificationLevel.MEDIUM,
            "System Startup",
            message,
            "🚀",
            action_required=False,
        )

    def notify_system_shutdown(
        self,
        total_trades: int,
        total_profit_gbp: float,
        duration_minutes: int,
    ):
        """Notify about system shutdown."""
        message = f"*🏁 Engine Completed*\n\n"
        message += f"Total Trades: {total_trades}\n"
        message += f"Total Profit: £{total_profit_gbp:,.2f}\n"
        message += f"Duration: {duration_minutes} minutes\n"
        
        self._send_notification(
            NotificationLevel.MEDIUM,
            "System Shutdown",
            message,
            "🏁",
            action_required=False,
        )

    def _send_notification(
        self,
        level: NotificationLevel,
        title: str,
        message: str,
        emoji: str,
        action_required: bool = False,
    ):
        """Send notification to Telegram."""
        # Check minimum level
        level_priority = {
            NotificationLevel.CRITICAL: 4,
            NotificationLevel.HIGH: 3,
            NotificationLevel.MEDIUM: 2,
            NotificationLevel.LOW: 1,
        }
        
        if level_priority[level] < level_priority[self.min_notification_level]:
            return  # Filtered out
        
        # Rate limiting
        now = datetime.now()
        self.message_times = [t for t in self.message_times if (now - t).total_seconds() < 60]
        
        if len(self.message_times) >= self.max_messages_per_minute:
            logger.warning("telegram_rate_limit_reached", skipped_message=title)
            return
        
        # Format message
        full_message = f"{emoji} *{title}*\n\n{message}"
        if action_required:
            full_message += "\n\n⚠️ *Action Required*"
        
        # Log to file
        notification = TelegramNotification(
            level=level,
            title=title,
            message=message,
            emoji=emoji,
            action_required=action_required,
        )
        self.notification_history.append(notification)
        self._log_to_file(f"[{level.value.upper()}] {title}: {message}")
        
        # Send to Telegram
        payload = {
            "chat_id": self.chat_id,
            "text": full_message,
            "parse_mode": "Markdown",
            "disable_web_page_preview": True,
        }
        
        try:
            response = requests.post(self.api_url, json=payload, timeout=10)
            if response.ok:
                self.message_times.append(now)
                logger.debug("telegram_notification_sent", title=title, level=level.value)
            else:
                logger.error("telegram_send_failed", status=response.status_code, text=response.text)
                self._log_to_file(f"ERROR: Failed to send Telegram notification: {response.status_code} - {response.text}")
        except Exception as e:
            logger.error("telegram_send_error", error=str(e), title=title)
            self._log_to_file(f"ERROR: Exception sending Telegram notification: {str(e)}")

    def _log_to_file(self, message: str):
        """Log message to file."""
        if not self.log_file:
            return
        
        try:
            timestamp = datetime.now().isoformat()
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(f"[{timestamp}] {message}\n")
        except Exception as e:
            logger.error("log_file_write_error", error=str(e))

    def get_notification_history(self) -> List[TelegramNotification]:
        """Get notification history."""
        return self.notification_history.copy()

    def export_log(self, output_path: Path):
        """Export notification log to file."""
        if not self.log_file or not self.log_file.exists():
            return
        
        try:
            import shutil
            shutil.copy(self.log_file, output_path)
            logger.info("log_exported", output_path=str(output_path))
        except Exception as e:
            logger.error("log_export_error", error=str(e))

