"""
Comprehensive Telegram Monitoring Bot

Real-time monitoring and notifications for the Engine:
- Every trade executed (with details)
- Every learning update (what it learned)
- Every error (with explanations)
- Performance summaries (hourly/daily)
- Model updates (when models change)
- Gate decisions (why trades were blocked)
- System health (alerts and warnings)

Source: Verified Telegram bot best practices
Expected Impact: Complete visibility into Engine operations

Features:
- Simple, easy-to-understand messages
- Real-time updates
- Rich formatting with emojis
- Actionable insights
- Error explanations
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import structlog  # type: ignore
import requests
from enum import Enum

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
    timestamp: datetime
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
    """

    def __init__(
        self,
        bot_token: str,
        chat_id: str,
        enable_trade_notifications: bool = True,
        enable_learning_notifications: bool = True,
        enable_error_notifications: bool = True,
        enable_performance_summaries: bool = True,
        enable_model_updates: bool = True,
        enable_gate_decisions: bool = True,
        enable_health_alerts: bool = True,
        min_notification_level: NotificationLevel = NotificationLevel.LOW,
    ):
        """
        Initialize Telegram monitor.
        
        Args:
            bot_token: Telegram bot token
            chat_id: Telegram chat ID
            enable_*: Enable/disable specific notification types
            min_notification_level: Minimum level to send (filters out low-priority)
        """
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.api_url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        
        self.enable_trade_notifications = enable_trade_notifications
        self.enable_learning_notifications = enable_learning_notifications
        self.enable_error_notifications = enable_error_notifications
        self.enable_performance_summaries = enable_performance_summaries
        self.enable_model_updates = enable_model_updates
        self.enable_gate_decisions = enable_gate_decisions
        self.enable_health_alerts = enable_health_alerts
        self.min_notification_level = min_notification_level
        
        # Rate limiting (max 20 messages per minute)
        self.message_times: List[datetime] = []
        self.max_messages_per_minute = 20
        
        logger.info("comprehensive_telegram_monitor_initialized")

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
        """Notify when a trade is executed."""
        if not self.enable_trade_notifications:
            return
        
        emoji = "🟢" if direction == "buy" else "🔴"
        message = f"""
{emoji} **TRADE EXECUTED**

**Symbol**: {symbol}
**Direction**: {direction.upper()}
**Entry Price**: ${entry_price:,.2f}
**Size**: £{size_gbp:,.2f}
**Confidence**: {confidence:.1%}
**Technique**: {technique}
**Regime**: {regime}

_Time: {datetime.now().strftime('%H:%M:%S')}_
        """.strip()
        
        self._send_notification(
            NotificationLevel.MEDIUM,
            "Trade Executed",
            message,
            emoji,
        )

    def notify_trade_exited(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        exit_price: float,
        pnl_gbp: float,
        pnl_bps: float,
        hold_duration_minutes: int,
        reason: str,
    ):
        """Notify when a trade is exited."""
        if not self.enable_trade_notifications:
            return
        
        emoji = "✅" if pnl_gbp > 0 else "❌"
        win_loss = "WIN" if pnl_gbp > 0 else "LOSS"
        
        message = f"""
{emoji} **TRADE CLOSED - {win_loss}**

**Symbol**: {symbol}
**Direction**: {direction.upper()}
**Entry**: ${entry_price:,.2f}
**Exit**: ${exit_price:,.2f}
**P&L**: £{pnl_gbp:+.2f} ({pnl_bps:+.1f} bps)
**Hold Time**: {hold_duration_minutes} min
**Reason**: {reason}

_Time: {datetime.now().strftime('%H:%M:%S')}_
        """.strip()
        
        self._send_notification(
            NotificationLevel.MEDIUM,
            f"Trade Closed - {win_loss}",
            message,
            emoji,
        )

    def notify_learning_update(
        self,
        what_learned: str,
        impact: str,
        confidence_change: float,
        feature_importance_changes: Dict[str, float],
    ):
        """Notify when the Engine learns something new."""
        if not self.enable_learning_notifications:
            return
        
        emoji = "🧠"
        message = f"""
{emoji} **ENGINE LEARNING UPDATE**

**What It Learned**:
{what_learned}

**Impact**:
{impact}

**Confidence Change**: {confidence_change:+.1%}

**Top Feature Changes**:
{self._format_feature_changes(feature_importance_changes)}

_Time: {datetime.now().strftime('%H:%M:%S')}_
        """.strip()
        
        self._send_notification(
            NotificationLevel.MEDIUM,
            "Learning Update",
            message,
            emoji,
        )

    def notify_error(
        self,
        error_type: str,
        error_message: str,
        context: str,
        severity: str = "medium",
        action_required: bool = False,
    ):
        """Notify when an error occurs."""
        if not self.enable_error_notifications:
            return
        
        emoji = "🔴" if severity == "critical" else "⚠️"
        level = NotificationLevel.CRITICAL if severity == "critical" else NotificationLevel.HIGH
        
        message = f"""
{emoji} **ERROR DETECTED**

**Type**: {error_type}
**Message**: {error_message}
**Context**: {context}
**Severity**: {severity.upper()}
{"**⚠️ ACTION REQUIRED**" if action_required else ""}

_Time: {datetime.now().strftime('%H:%M:%S')}_
        """.strip()
        
        self._send_notification(
            level,
            "Error Detected",
            message,
            emoji,
            action_required=action_required,
        )

    def notify_gate_decision(
        self,
        symbol: str,
        direction: str,
        confidence: float,
        gates_passed: List[str],
        gates_blocked: List[str],
        reason: str,
        counterfactual: Optional[Dict[str, Any]] = None,
    ):
        """Notify when a gate blocks/allows a trade."""
        if not self.enable_gate_decisions:
            return
        
        if gates_blocked:
            emoji = "🚫"
            title = "Trade Blocked by Gates"
            message = f"""
{emoji} **TRADE BLOCKED**

**Symbol**: {symbol}
**Direction**: {direction.upper()}
**Confidence**: {confidence:.1%}

**Gates Passed**: {', '.join(gates_passed) if gates_passed else 'None'}
**Gates Blocked**: {', '.join(gates_blocked)}

**Reason**: {reason}

{f"**Counterfactual**: Would have made £{counterfactual.get('potential_profit_gbp', 0):.2f}" if counterfactual else ""}

_Time: {datetime.now().strftime('%H:%M:%S')}_
            """.strip()
        else:
            emoji = "✅"
            title = "Trade Approved by Gates"
            message = f"""
{emoji} **TRADE APPROVED**

**Symbol**: {symbol}
**Direction**: {direction.upper()}
**Confidence**: {confidence:.1%}

**All Gates Passed**: {', '.join(gates_passed)}

_Time: {datetime.now().strftime('%H:%M:%S')}_
            """.strip()
        
        self._send_notification(
            NotificationLevel.LOW,
            title,
            message,
            emoji,
        )

    def notify_performance_summary(
        self,
        period: str,  # "hourly" or "daily"
        total_trades: int,
        wins: int,
        losses: int,
        win_rate: float,
        total_pnl_gbp: float,
        total_pnl_bps: float,
        sharpe_ratio: float,
        best_trade: Optional[Dict[str, Any]] = None,
        worst_trade: Optional[Dict[str, Any]] = None,
    ):
        """Notify performance summary (hourly/daily)."""
        if not self.enable_performance_summaries:
            return
        
        emoji = "📊"
        period_emoji = "⏰" if period == "hourly" else "📅"
        
        message = f"""
{emoji} **{period.upper()} PERFORMANCE SUMMARY**

{period_emoji} **Period**: {period.capitalize()}

**Trades**: {total_trades} ({wins} wins, {losses} losses)
**Win Rate**: {win_rate:.1%}
**Total P&L**: £{total_pnl_gbp:+.2f} ({total_pnl_bps:+.1f} bps)
**Sharpe Ratio**: {sharpe_ratio:.2f}

{f"**Best Trade**: {best_trade.get('symbol', 'N/A')} - £{best_trade.get('pnl_gbp', 0):+.2f}" if best_trade else ""}
{f"**Worst Trade**: {worst_trade.get('symbol', 'N/A')} - £{worst_trade.get('pnl_gbp', 0):+.2f}" if worst_trade else ""}

_Time: {datetime.now().strftime('%H:%M:%S')}_
        """.strip()
        
        self._send_notification(
            NotificationLevel.MEDIUM,
            f"{period.capitalize()} Performance Summary",
            message,
            emoji,
        )

    def notify_model_update(
        self,
        symbol: str,
        old_metrics: Dict[str, float],
        new_metrics: Dict[str, float],
        improvement: Dict[str, float],
    ):
        """Notify when a model is updated."""
        if not self.enable_model_updates:
            return
        
        emoji = "🔄"
        message = f"""
{emoji} **MODEL UPDATED**

**Symbol**: {symbol}

**Old Metrics**:
- Sharpe: {old_metrics.get('sharpe', 0):.2f}
- Win Rate: {old_metrics.get('win_rate', 0):.1%}
- P&L: {old_metrics.get('pnl_bps', 0):+.1f} bps

**New Metrics**:
- Sharpe: {new_metrics.get('sharpe', 0):.2f} ({improvement.get('sharpe', 0):+.2f})
- Win Rate: {new_metrics.get('win_rate', 0):.1%} ({improvement.get('win_rate', 0):+.1%})
- P&L: {new_metrics.get('pnl_bps', 0):+.1f} bps ({improvement.get('pnl_bps', 0):+.1f} bps)

**Improvement**: {improvement.get('overall', 0):+.1%}

_Time: {datetime.now().strftime('%H:%M:%S')}_
        """.strip()
        
        self._send_notification(
            NotificationLevel.HIGH,
            "Model Updated",
            message,
            emoji,
        )

    def notify_health_alert(
        self,
        alert_type: str,
        severity: str,
        message: str,
        action_required: bool = False,
    ):
        """Notify health alerts."""
        if not self.enable_health_alerts:
            return
        
        emoji = "🔴" if severity == "critical" else "⚠️" if severity == "high" else "ℹ️"
        level = NotificationLevel.CRITICAL if severity == "critical" else NotificationLevel.HIGH
        
        alert_message = f"""
{emoji} **HEALTH ALERT - {alert_type.upper()}**

**Severity**: {severity.upper()}
**Message**: {message}
{"**⚠️ ACTION REQUIRED**" if action_required else ""}

_Time: {datetime.now().strftime('%H:%M:%S')}_
        """.strip()
        
        self._send_notification(
            level,
            f"Health Alert - {alert_type}",
            alert_message,
            emoji,
            action_required=action_required,
        )

    def notify_daily_summary(
        self,
        date: str,
        total_trades: int,
        win_rate: float,
        total_pnl_gbp: float,
        sharpe_ratio: float,
        what_learned: List[str],
        model_updates: int,
        errors: int,
        top_performers: List[Dict[str, Any]],
        issues: List[str],
    ):
        """Send comprehensive daily summary."""
        emoji = "📅"
        
        message = f"""
{emoji} **DAILY SUMMARY - {date}**

**Performance**:
- Trades: {total_trades}
- Win Rate: {win_rate:.1%}
- Total P&L: £{total_pnl_gbp:+.2f}
- Sharpe: {sharpe_ratio:.2f}

**What It Learned**:
{chr(10).join(f"• {item}" for item in what_learned[:5])}

**Updates**:
- Models Updated: {model_updates}
- Errors: {errors}

**Top Performers**:
{chr(10).join(f"• {p.get('symbol', 'N/A')}: £{p.get('pnl_gbp', 0):+.2f}" for p in top_performers[:3])}

{"**Issues**: " + chr(10).join(f"• {issue}" for issue in issues[:3]) if issues else ""}

_Time: {datetime.now().strftime('%H:%M:%S')}_
        """.strip()
        
        self._send_notification(
            NotificationLevel.HIGH,
            f"Daily Summary - {date}",
            message,
            emoji,
        )

    def _format_feature_changes(self, changes: Dict[str, float]) -> str:
        """Format feature importance changes."""
        if not changes:
            return "No significant changes"
        
        sorted_changes = sorted(changes.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        lines = []
        for feature, change in sorted_changes:
            direction = "📈" if change > 0 else "📉"
            lines.append(f"{direction} {feature}: {change:+.2%}")
        
        return "\n".join(lines)

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
        except Exception as e:
            logger.error("telegram_send_error", error=str(e), title=title)

    def send_simple_message(self, message: str, emoji: str = "ℹ️"):
        """Send a simple text message."""
        full_message = f"{emoji} {message}"
        payload = {
            "chat_id": self.chat_id,
            "text": full_message,
            "parse_mode": "Markdown",
        }
        
        try:
            response = requests.post(self.api_url, json=payload, timeout=10)
            if not response.ok:
                logger.error("telegram_send_failed", status=response.status_code)
        except Exception as e:
            logger.error("telegram_send_error", error=str(e))

    def batch_notify(
        self,
        notifications: List[Dict[str, Any]],
        max_batch_size: int = 5,
    ):
        """
        Batch multiple notifications into single messages.
        
        Args:
            notifications: List of notification dicts with 'type', 'data', etc.
            max_batch_size: Maximum notifications per batch
        """
        if not notifications:
            return
        
        # Group notifications by type
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for notif in notifications:
            notif_type = notif.get('type', 'unknown')
            if notif_type not in grouped:
                grouped[notif_type] = []
            grouped[notif_type].append(notif)
        
        # Send batched notifications
        for notif_type, notif_list in grouped.items():
            # Split into batches
            for i in range(0, len(notif_list), max_batch_size):
                batch = notif_list[i:i + max_batch_size]
                
                if notif_type == 'trade_executed':
                    self._batch_trade_executed(batch)
                elif notif_type == 'trade_exited':
                    self._batch_trade_exited(batch)
                elif notif_type == 'learning_update':
                    self._batch_learning_update(batch)
                elif notif_type == 'gate_decision':
                    self._batch_gate_decision(batch)
                else:
                    # Send individually for unknown types
                    for notif in batch:
                        self._send_notification(
                            NotificationLevel.MEDIUM,
                            notif.get('title', 'Notification'),
                            notif.get('message', ''),
                            notif.get('emoji', 'ℹ️'),
                        )

    def _batch_trade_executed(self, trades: List[Dict[str, Any]]):
        """Batch trade executed notifications."""
        if not trades:
            return
        
        emoji = "🟢"
        message = f"""
{emoji} **{len(trades)} TRADES EXECUTED**

{chr(10).join(self._format_trade_executed(t) for t in trades)}

_Time: {datetime.now().strftime('%H:%M:%S')}_
        """.strip()
        
        self._send_notification(
            NotificationLevel.MEDIUM,
            f"{len(trades)} Trades Executed",
            message,
            emoji,
        )

    def _batch_trade_exited(self, trades: List[Dict[str, Any]]):
        """Batch trade exited notifications."""
        if not trades:
            return
        
        wins = sum(1 for t in trades if t.get('pnl_gbp', 0) > 0)
        losses = len(trades) - wins
        total_pnl = sum(t.get('pnl_gbp', 0) for t in trades)
        
        emoji = "✅" if total_pnl > 0 else "❌"
        message = f"""
{emoji} **{len(trades)} TRADES CLOSED**

**Summary**: {wins} wins, {losses} losses
**Total P&L**: £{total_pnl:+.2f}

{chr(10).join(self._format_trade_exited(t) for t in trades[:5])}
{"..." if len(trades) > 5 else ""}

_Time: {datetime.now().strftime('%H:%M:%S')}_
        """.strip()
        
        self._send_notification(
            NotificationLevel.MEDIUM,
            f"{len(trades)} Trades Closed",
            message,
            emoji,
        )

    def _batch_learning_update(self, updates: List[Dict[str, Any]]):
        """Batch learning update notifications."""
        if not updates:
            return
        
        emoji = "🧠"
        message = f"""
{emoji} **{len(updates)} LEARNING UPDATES**

{chr(10).join(f"• {u.get('what_learned', 'N/A')}" for u in updates[:3])}
{"..." if len(updates) > 3 else ""}

_Time: {datetime.now().strftime('%H:%M:%S')}_
        """.strip()
        
        self._send_notification(
            NotificationLevel.MEDIUM,
            f"{len(updates)} Learning Updates",
            message,
            emoji,
        )

    def _batch_gate_decision(self, decisions: List[Dict[str, Any]]):
        """Batch gate decision notifications."""
        if not decisions:
            return
        
        blocked = sum(1 for d in decisions if d.get('gates_blocked'))
        approved = len(decisions) - blocked
        
        emoji = "🚫" if blocked > approved else "✅"
        message = f"""
{emoji} **{len(decisions)} GATE DECISIONS**

**Summary**: {approved} approved, {blocked} blocked

{chr(10).join(f"• {d.get('symbol', 'N/A')}: {'BLOCKED' if d.get('gates_blocked') else 'APPROVED'}" for d in decisions[:5])}
{"..." if len(decisions) > 5 else ""}

_Time: {datetime.now().strftime('%H:%M:%S')}_
        """.strip()
        
        self._send_notification(
            NotificationLevel.LOW,
            f"{len(decisions)} Gate Decisions",
            message,
            emoji,
        )

    def _format_trade_executed(self, trade: Dict[str, Any]) -> str:
        """Format single trade executed for batch."""
        return f"**{trade.get('symbol', 'N/A')}** {trade.get('direction', 'buy').upper()} @ ${trade.get('entry_price', 0):,.2f} ({trade.get('confidence', 0):.1%})"

    def _format_trade_exited(self, trade: Dict[str, Any]) -> str:
        """Format single trade exited for batch."""
        pnl = trade.get('pnl_gbp', 0)
        emoji = "✅" if pnl > 0 else "❌"
        return f"{emoji} **{trade.get('symbol', 'N/A')}**: £{pnl:+.2f}"

