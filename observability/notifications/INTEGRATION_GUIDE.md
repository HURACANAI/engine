"""
Telegram Monitoring Integration Guide

Complete guide for setting up real-time Telegram monitoring for the Engine.
"""

from typing import Dict, Any
import os

# Example usage
"""
# 1. Create Telegram Bot
# - Message @BotFather on Telegram
# - Send /newbot
# - Follow instructions to create bot
# - Save the bot token (e.g., "123456789:ABCdefGHIjklMNOpqrsTUVwxyz")

# 2. Get Your Chat ID
# - Message your bot
# - Visit: https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates
# - Find "chat":{"id":123456789} - that's your chat_id

# 3. Initialize Telegram Monitor
from observability.notifications.telegram_monitor import ComprehensiveTelegramMonitor, NotificationLevel
from observability.notifications.activity_monitor import RealTimeActivityMonitor

# Initialize monitor
telegram_monitor = ComprehensiveTelegramMonitor(
    bot_token="YOUR_BOT_TOKEN",
    chat_id="YOUR_CHAT_ID",
    enable_trade_notifications=True,
    enable_learning_notifications=True,
    enable_error_notifications=True,
    enable_performance_summaries=True,
    enable_model_updates=True,
    enable_gate_decisions=True,
    enable_health_alerts=True,
    min_notification_level=NotificationLevel.LOW,  # Get all notifications
)

# 4. Initialize Activity Monitor
from observability.core.event_logger import EventLogger
from observability.analytics.learning_tracker import LearningTracker
from observability.analytics.trade_journal import TradeJournal
from observability.analytics.metrics_computer import MetricsComputer
from observability.analytics.model_evolution import ModelEvolutionTracker

event_logger = EventLogger()
learning_tracker = LearningTracker()
trade_journal = TradeJournal()
metrics_computer = MetricsComputer()
model_tracker = ModelEvolutionTracker()

activity_monitor = RealTimeActivityMonitor(
    telegram_monitor=telegram_monitor,
    event_logger=event_logger,
    learning_tracker=learning_tracker,
    trade_journal=trade_journal,
    metrics_computer=metrics_computer,
    model_tracker=model_tracker,
)

# 5. Start Monitoring
import asyncio

async def main():
    await activity_monitor.monitor_continuously()

if __name__ == "__main__":
    asyncio.run(main())

# 6. Integrate with Engine
# In your Engine code, use activity_monitor to record events:

# Record trade
activity_monitor.telegram_monitor.notify_trade_executed(
    symbol="BTCUSDT",
    direction="buy",
    entry_price=50000.0,
    size_gbp=100.0,
    confidence=0.75,
    technique="trend",
    regime="trend",
)

# Record error
activity_monitor.record_error(
    error_type="Connection Error",
    error_message="Failed to connect to exchange",
    context="Order execution",
    severity="high",
)

# Record gate decision
activity_monitor.record_gate_decision(
    symbol="ETHUSDT",
    direction="buy",
    confidence=0.60,
    gates_passed=["cost_gate", "meta_label_gate"],
    gates_blocked=["regime_gate"],
    reason="Not in optimal regime",
)
"""

