"""
Telegram Bot Command Handler

Handles interactive commands from Telegram:
- /health - Get comprehensive health check
- /status - Get system status
- /help - Show available commands

Uses long polling to receive commands and respond.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import requests
import structlog

from .enhanced_health_check import EnhancedHealthChecker, ComprehensiveHealthReport
from ..config.settings import EngineSettings
from ..integrations.dropbox_sync import DropboxSync

logger = structlog.get_logger(__name__)


class TelegramCommandHandler:
    """Handle Telegram bot commands with interactive responses."""

    def __init__(
        self,
        bot_token: str,
        chat_id: str,
        settings: Optional[EngineSettings] = None,
        dropbox_sync: Optional[DropboxSync] = None,
    ):
        """
        Initialize Telegram command handler.

        Args:
            bot_token: Telegram bot token
            chat_id: Telegram chat ID (allowed user)
            settings: EngineSettings instance
            dropbox_sync: DropboxSync instance (optional)
        """
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.settings = settings or EngineSettings.load()
        self.dropbox_sync = dropbox_sync

        self.api_url = f"https://api.telegram.org/bot{bot_token}"
        self.last_update_id = 0
        self.running = False
        self.thread: Optional[threading.Thread] = None

        # Initialize health checker
        dsn = self.settings.postgres.dsn if self.settings.postgres else None
        self.health_checker = EnhancedHealthChecker(
            dsn=dsn,
            settings=self.settings,
            dropbox_sync=dropbox_sync,
        )

        logger.info("telegram_command_handler_initialized", chat_id=chat_id)

    def send_message(self, text: str, parse_mode: str = "Markdown") -> bool:
        """
        Send a message to the chat.

        Args:
            text: Message text
            parse_mode: Markdown or HTML

        Returns:
            True if sent successfully
        """
        try:
            url = f"{self.api_url}/sendMessage"
            payload = {
                "chat_id": self.chat_id,
                "text": text,
                "parse_mode": parse_mode,
            }
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            return True
        except Exception as e:
            logger.exception("telegram_send_message_failed", error=str(e))
            return False

    def get_updates(self) -> list:
        """
        Get pending updates from Telegram.

        Returns:
            List of updates
        """
        try:
            url = f"{self.api_url}/getUpdates"
            params = {
                "offset": self.last_update_id + 1,
                "timeout": 30,  # Long polling
            }
            response = requests.get(url, params=params, timeout=35)
            response.raise_for_status()
            data = response.json()
            if data.get("ok"):
                return data.get("result", [])
            return []
        except Exception as e:
            logger.warning("telegram_get_updates_failed", error=str(e))
            return []

    def handle_command(self, command: str, message_text: str) -> Optional[str]:
        """
        Handle a command.

        Args:
            command: Command name (e.g., "/health")
            message_text: Full message text

        Returns:
            Response message or None
        """
        command = command.lower().strip()

        if command == "/health" or command == "/health@your_bot_name":
            return self.handle_health_command()

        elif command == "/status":
            return self.handle_status_command()

        elif command == "/help":
            return self.handle_help_command()

        else:
            return f"❓ Unknown command: {command}\n\nUse /help to see available commands."

    def handle_health_command(self) -> str:
        """Handle /health command - run comprehensive health check."""
        logger.info("handling_health_command")

        # Send "Running health check..." message
        self.send_message("🔍 Running comprehensive health check...\n\nThis may take a few seconds...")

        try:
            # Run comprehensive health check
            report = self.health_checker.run_comprehensive_check()

            # Format response
            response = self.format_health_report(report)

            return response

        except Exception as e:
            logger.exception("health_check_command_failed", error=str(e))
            return f"❌ Health check failed:\n\n`{str(e)}`"

    def handle_status_command(self) -> str:
        """Handle /status command - get quick status."""
        try:
            report = self.health_checker.run_comprehensive_check()
            status_icon = {
                "HEALTHY": "✅",
                "DEGRADED": "⚠️",
                "CRITICAL": "🚨",
            }.get(report.overall_status, "❓")

            summary = report.summary
            response = f"{status_icon} *System Status: {report.overall_status}*\n\n"
            response += f"✅ Healthy: {summary.get('healthy', 0)}\n"
            response += f"⚠️ Warnings: {summary.get('warnings', 0)}\n"
            response += f"🚨 Critical: {summary.get('critical', 0)}\n"

            if report.resource_usage:
                resources = report.resource_usage
                response += f"\n💻 *Resources:*\n"
                response += f"CPU: {resources.get('cpu_percent', 0):.1f}%\n"
                response += f"Memory: {resources.get('memory_percent', 0):.1f}%\n"
                response += f"Disk: {resources.get('disk_percent', 0):.1f}%\n"

            return response

        except Exception as e:
            logger.exception("status_command_failed", error=str(e))
            return f"❌ Status check failed:\n\n`{str(e)}`"

    def handle_help_command(self) -> str:
        """Handle /help command - show available commands."""
        response = "🤖 *Huracan Engine Bot Commands*\n\n"
        response += "*/health* - Run comprehensive health check\n"
        response += "   Shows detailed status of all 18 system components\n"
        response += "   Includes issues, recommendations, and resource usage\n\n"
        response += "*/status* - Quick system status\n"
        response += "   Shows overall status and resource usage\n\n"
        response += "*/help* - Show this help message\n\n"
        response += "The bot monitors your engine and sends automatic\n"
        response += "notifications about training, errors, and health checks."
        return response

    def format_health_report(self, report: ComprehensiveHealthReport) -> str:
        """
        Format comprehensive health report for Telegram.

        Args:
            report: ComprehensiveHealthReport instance

        Returns:
            Formatted message string
        """
        status_icon = {
            "HEALTHY": "✅",
            "DEGRADED": "⚠️",
            "CRITICAL": "🚨",
        }.get(report.overall_status, "❓")

        message = f"🏥 *Comprehensive Health Check*\n\n"
        message += f"Status: {status_icon} *{report.overall_status}*\n"
        message += f"Time: {report.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}\n\n"

        # Summary
        summary = report.summary
        message += f"📊 *Summary:*\n"
        message += f"• Total Checks: {len(report.checks)}\n"
        message += f"• ✅ Healthy: {summary.get('healthy', 0)}\n"
        message += f"• ⚠️ Warnings: {summary.get('warnings', 0)}\n"
        message += f"• 🚨 Critical: {summary.get('critical', 0)}\n"
        message += f"• ⏸️ Disabled: {summary.get('disabled', 0)}\n\n"

        # Resource usage
        if report.resource_usage:
            resources = report.resource_usage
            message += f"💻 *Resources:*\n"
            message += f"• CPU: {resources.get('cpu_percent', 0):.1f}%\n"
            message += f"• Memory: {resources.get('memory_percent', 0):.1f}%\n"
            message += f"• Disk: {resources.get('disk_percent', 0):.1f}%\n\n"

        # Critical issues
        critical_checks = [c for c in report.checks if c.status == "CRITICAL"]
        if critical_checks:
            message += f"🚨 *Critical Issues ({len(critical_checks)}):*\n"
            for check in critical_checks[:5]:  # Limit to 5 for Telegram message size
                message += f"• *{check.name}:* {check.message}\n"
                if check.issues:
                    for issue in check.issues[:2]:  # Limit issues per check
                        message += f"  └ {issue}\n"
            if len(critical_checks) > 5:
                message += f"  ... and {len(critical_checks) - 5} more\n"
            message += "\n"

        # Warnings
        warning_checks = [c for c in report.checks if c.status == "WARNING"]
        if warning_checks:
            message += f"⚠️ *Warnings ({len(warning_checks)}):*\n"
            for check in warning_checks[:3]:  # Limit to 3
                message += f"• *{check.name}:* {check.message}\n"
            if len(warning_checks) > 3:
                message += f"  ... and {len(warning_checks) - 3} more\n"
            message += "\n"

        # All checks status (compact)
        message += f"📋 *All Checks:*\n"
        for check in report.checks:
            status_emoji = {
                "HEALTHY": "✅",
                "WARNING": "⚠️",
                "CRITICAL": "🚨",
                "DISABLED": "⏸️",
            }.get(check.status, "❓")
            message += f"{status_emoji} {check.name}\n"

        # Recommendations
        if report.recommendations:
            message += f"\n💡 *Recommendations:*\n"
            for rec in report.recommendations[:5]:  # Limit to 5
                message += f"• {rec}\n"

        # Add footer
        message += f"\n_Use /help for more commands_"

        return message

    def process_update(self, update: Dict[str, Any]) -> None:
        """
        Process a single update from Telegram.

        Args:
            update: Update dictionary from Telegram API
        """
        try:
            message = update.get("message", {})
            if not message:
                return

            # Check if from allowed chat
            chat = message.get("chat", {})
            chat_id = str(chat.get("id", ""))
            if chat_id != self.chat_id:
                logger.warning("message_from_unauthorized_chat", chat_id=chat_id)
                return

            # Get command
            text = message.get("text", "").strip()
            if not text or not text.startswith("/"):
                return

            # Extract command
            parts = text.split(maxsplit=1)
            command = parts[0]

            logger.info("telegram_command_received", command=command, chat_id=chat_id)

            # Handle command
            response = self.handle_command(command, text)
            if response:
                # Split long messages (Telegram limit is 4096 characters)
                max_length = 4000
                if len(response) > max_length:
                    # Send in chunks
                    chunks = [response[i:i+max_length] for i in range(0, len(response), max_length)]
                    for chunk in chunks:
                        self.send_message(chunk)
                        time.sleep(0.5)  # Rate limiting
                else:
                    self.send_message(response)

        except Exception as e:
            logger.exception("process_update_failed", error=str(e))

    def start_polling(self) -> None:
        """Start polling for commands in a separate thread."""
        if self.running:
            logger.warning("polling_already_running")
            return

        try:
            self.running = True
            self.thread = threading.Thread(target=self._poll_loop, daemon=True)
            self.thread.start()
            logger.info("telegram_command_polling_started", chat_id=self.chat_id)
            print(f"✅ Telegram command handler started - listening for /health, /status, /help")
        except Exception as e:
            logger.exception("telegram_command_polling_start_failed", error=str(e))
            self.running = False
            raise

    def stop_polling(self) -> None:
        """Stop polling for commands."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        logger.info("telegram_command_polling_stopped")

    def _poll_loop(self) -> None:
        """Main polling loop (runs in thread)."""
        logger.info("telegram_poll_loop_started")
        while self.running:
            try:
                updates = self.get_updates()
                for update in updates:
                    update_id = update.get("update_id", 0)
                    if update_id > self.last_update_id:
                        self.last_update_id = update_id
                        self.process_update(update)

                # Small delay to avoid hitting rate limits
                time.sleep(1)

            except Exception as e:
                logger.exception("poll_loop_error", error=str(e))
                time.sleep(5)  # Wait longer on error

        logger.info("telegram_poll_loop_stopped")

