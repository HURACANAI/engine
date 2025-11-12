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
from .download_progress_tracker import DownloadStatus
from ..config.settings import EngineSettings
from ..integrations.dropbox_sync import DropboxSync
import time

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

        elif command == "/download":
            return self.handle_download_command()

        elif command.startswith("/grok"):
            # Extract question from message
            question = message_text[len(command):].strip() if len(message_text) > len(command) else ""
            return self.handle_grok_command(question)

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
        response += "*/download* - Show download progress\n"
        response += "   Shows progress of downloading historical data for all coins\n"
        response += "   Includes current symbol, progress percentage, and time estimates\n\n"
        response += "*/grok <question>* - Ask Grok AI about the codebase\n"
        response += "   Ask any question about the engine codebase\n"
        response += "   Example: `/grok How does the training pipeline work?`\n"
        response += "   ⚠️ Note: Requires valid Grok API key with API access\n\n"
        response += "*/help* - Show this help message\n\n"
        response += "The bot monitors your engine and sends automatic\n"
        response += "notifications about training, errors, and health checks."
        return response

    def handle_grok_command(self, question: str) -> str:
        """Handle /grok command - ask Grok AI about the codebase."""
        if not question:
            return (
                "🤖 *Grok AI Assistant*\n\n"
                "Ask me anything about the codebase!\n\n"
                "Usage: `/grok <your question>`\n\n"
                "Examples:\n"
                "• `/grok How does the training pipeline work?`\n"
                "• `/grok What are the main components?`\n"
                "• `/grok How does walk-forward validation work?`\n"
                "• `/grok Where is the cost calculator?`"
            )
        
        logger.info("grok_command_received", question=question[:100])
        
        # Send "Thinking..." message
        self.send_message("🤔 Asking Grok AI...\n\nThis may take a few seconds...")
        
        try:
            # Get Grok API key from settings
            grok_api_key = self.settings.notifications.grok_api_key
            if not grok_api_key:
                return (
                    "❌ Grok API key not configured.\n\n"
                    "Please set `grok_api_key` in your config file or `GROK_API_KEY` environment variable."
                )
            
            # Prepare codebase context
            codebase_context = self._get_codebase_context()
            
            # Query Grok API
            response = self._query_grok_api(grok_api_key, question, codebase_context)
            
            if response:
                # Format response for Telegram
                formatted_response = f"🤖 *Grok AI Response*\n\n{response}"
                return formatted_response
            else:
                # Provide helpful error message with troubleshooting
                return (
                    "❌ *Grok API Error*\n\n"
                    "The API key is being rejected. Possible reasons:\n\n"
                    "1. *Free Plan Limitation*: The free plan may not include API access.\n"
                    "   Check: https://console.x.ai for plan details\n\n"
                    "2. *Key Not Activated*: The key might need activation in the console\n\n"
                    "3. *Wrong Key Type*: Ensure you're using an API key, not a web access token\n\n"
                    "4. *Account Verification*: Your account might need additional verification\n\n"
                    "💡 *To Fix:*\n"
                    "• Visit https://console.x.ai\n"
                    "• Verify your API key is active\n"
                    "• Check if your plan includes API access\n"
                    "• Try generating a new API key\n\n"
                    "The command handler is working correctly - this is an API access issue."
                )
                
        except Exception as e:
            logger.exception("grok_command_failed", error=str(e))
            return f"❌ Grok command failed:\n\n`{str(e)}`"
    
    def _get_codebase_context(self) -> str:
        """Get codebase context for Grok API."""
        context = """
CODEBASE CONTEXT - Huracan Engine Trading System

MAIN PURPOSE:
- Cloud Training Box that builds daily baseline models
- Trains on 3-6 months of historical market data
- Performs walk-forward validation
- Exports models for other components (Archive/Mechanic/Pilot)

KEY COMPONENTS:

1. TRAINING PIPELINE:
   - Entry point: `src/cloud/training/pipelines/daily_retrain.py`
   - Orchestration: `src/cloud/training/services/orchestration.py`
   - Walk-forward validation with multiple splits
   - Model training with LightGBM, RL agents

2. DATA QUALITY:
   - Data sanity pipeline: `src/cloud/engine/data_quality/sanity_pipeline.py`
   - Gap handler: `src/cloud/engine/data_quality/gap_handler.py`
   - Outlier removal, duplicate detection, gap filling

3. LABELING:
   - Triple barrier labeling: `src/cloud/engine/labeling/triple_barrier.py`
   - Meta labeling for trade selection
   - Labeled trades for model training

4. COST CALCULATION:
   - Cost breakdown: `src/cloud/training/services/costs.py`
   - Includes fees, spread, slippage
   - Per-symbol cost calculation

5. NOTIFICATIONS:
   - Telegram notifications: `src/cloud/training/services/notifications.py`
   - Grok AI explanations for metrics
   - Training progress updates

6. RL TRAINING:
   - RL agent: `src/cloud/training/agents/rl_agent.py`
   - PPO-based trading agent
   - Memory store for experience replay

7. VALIDATION:
   - Walk-forward validation across multiple time periods
   - Metrics: Sharpe ratio, Profit Factor, Hit Rate, Max Drawdown
   - Stability analysis across splits

8. CONFIGURATION:
   - Settings: `src/cloud/training/config/settings.py`
   - YAML configs: `config/base.yaml`
   - Environment variable overrides

KEY METRICS:
- Sharpe Ratio: Risk-adjusted return
- Profit Factor: Gross profits / Gross losses
- Hit Rate: Percentage of winning trades
- Max Drawdown: Largest peak-to-trough decline
- Costs (bps): Total trading costs in basis points

MAIN FLOW:
1. Download historical data (150 days)
2. Data quality pipeline (clean, remove outliers, fill gaps)
3. Feature engineering
4. Triple barrier labeling
5. Walk-forward validation (multiple splits)
6. Model training (LightGBM, RL)
7. Calculate metrics (Sharpe, PF, Hit Rate, Max DD)
8. Validate against gates
9. Export model if passed
10. Send Telegram notification with Grok AI explanation
"""
        return context
    
    def _query_grok_api(self, api_key: str, question: str, context: str) -> Optional[str]:
        """Query Grok API with question and codebase context."""
        try:
            
            # Verify API key format
            api_key = api_key.strip()
            if not api_key.startswith("gsk_"):
                logger.warning("grok_api_key_format_invalid", key_prefix=api_key[:10])
                return None
            
            # Log exact key details for debugging
            logger.info(
                "grok_api_request_details",
                key_length=len(api_key),
                key_prefix=api_key[:15],
                key_suffix=api_key[-5:],
                key_starts_with_gsk=api_key.startswith("gsk_")
            )
            
            # Prepare prompt
            prompt = f"""You are an expert codebase analyst for the Huracan Engine trading system.

{context}

USER QUESTION: {question}

Please provide a clear, helpful answer about the codebase. Focus on:
- How things work in this codebase
- Where to find relevant code
- How components interact
- Best practices for this system

Keep your response concise and actionable."""
            
            # Prepare request
            url = "https://api.x.ai/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "grok-2-latest",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert codebase analyst. Answer questions about code structure, functionality, and implementation. Be clear, concise, and actionable."
                    },
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.7,
                "max_tokens": 1000
            }
            
            # Log request details (without full key)
            logger.debug(
                "grok_api_request",
                url=url,
                model=payload["model"],
                message_count=len(payload["messages"]),
                prompt_length=len(prompt)
            )
            
            # Query Grok API
            response = requests.post(url, headers=headers, json=payload, timeout=20)
            
            if response.ok:
                result = response.json()
                if 'choices' in result and len(result['choices']) > 0:
                    explanation = result['choices'][0]['message']['content'].strip()
                    logger.info("grok_api_response_received", question_length=len(question), response_length=len(explanation))
                    return explanation
                else:
                    logger.warning("grok_api_invalid_response", response=result)
                    return None
            else:
                error_text = response.text[:500] if response.text else "No error message"
                error_code = "unknown"
                error_detail = error_text
                
                try:
                    error_json = response.json()
                    if isinstance(error_json, dict):
                        # Handle nested error structure
                        if "error" in error_json:
                            if isinstance(error_json["error"], dict):
                                error_detail = error_json["error"].get("message", error_text)
                                error_code = error_json["error"].get("code", error_json.get("code", "unknown"))
                            else:
                                error_detail = str(error_json["error"])
                        else:
                            error_detail = error_json.get("message", error_text)
                            error_code = error_json.get("code", "unknown")
                except Exception as parse_error:
                    # If JSON parsing fails, use raw text
                    logger.debug("grok_error_parse_failed", parse_error=str(parse_error), raw_text=error_text[:200])
                
                # Log full error details for debugging
                logger.error(
                    "grok_api_failed",
                    status=response.status_code,
                    error_code=error_code,
                    error_message=error_detail,
                    response_headers=dict(response.headers),
                    api_key_length=len(api_key),
                    api_key_prefix=api_key[:15],
                    api_key_suffix=api_key[-5:],
                    full_response=response.text[:1000] if response.text else None
                )
                
                # If it's specifically an API key error, provide more detailed troubleshooting
                if "Incorrect API key" in error_detail or "invalid" in error_detail.lower():
                    logger.error(
                        "grok_api_key_issue",
                        message="API key is being rejected by xAI",
                        suggestion="Verify key at https://console.x.ai and ensure account has API access enabled"
                    )
                
                return None
                
        except Exception as e:
            logger.exception("grok_api_query_failed", error=str(e))
            return None

    def handle_download_command(self) -> str:
        """Handle /download command - show download progress."""
        try:
            from .download_progress_tracker import get_progress_tracker
            progress_tracker = get_progress_tracker()
            progress = progress_tracker.get_progress()
            
            if not progress:
                return "📊 *Download Progress*\n\nNo download session active.\n\nTraining may not have started yet, or all downloads are complete."
            
            # Format progress message
            progress_percent = progress.get_progress_percent()
            elapsed_time = progress.get_elapsed_time()
            estimated_remaining = progress.get_estimated_remaining_time()
            
            response = "📊 *Download Progress*\n\n"
            response += f"Progress: *{progress_percent:.1f}%*\n\n"
            
            # Summary
            response += f"📈 *Summary:*\n"
            response += f"• Total Symbols: {progress.total_symbols}\n"
            response += f"• ✅ Completed: {progress.completed}\n"
            response += f"• ⏳ Downloading: {progress.downloading}\n"
            response += f"• ❌ Failed: {progress.failed}\n"
            response += f"• ⏸️ Pending: {progress.pending}\n\n"
            
            # Time information
            if elapsed_time:
                minutes = int(elapsed_time // 60)
                seconds = int(elapsed_time % 60)
                response += f"⏱️ *Time:*\n"
                response += f"• Elapsed: {minutes}m {seconds}s\n"
                
                if estimated_remaining and progress.completed > 0:
                    rem_minutes = int(estimated_remaining // 60)
                    rem_seconds = int(estimated_remaining % 60)
                    response += f"• Estimated Remaining: {rem_minutes}m {rem_seconds}s\n"
                response += "\n"
            
            # Currently downloading symbols
            downloading_symbols = [
                s for s in progress.symbols.values()
                if s.status == DownloadStatus.DOWNLOADING
            ]
            if downloading_symbols:
                response += f"⏳ *Currently Downloading ({len(downloading_symbols)}):*\n"
                for sym_progress in downloading_symbols[:5]:  # Limit to 5
                    duration = ""
                    if sym_progress.start_time:
                        elapsed = time.time() - sym_progress.start_time
                        duration = f" ({int(elapsed)}s)"
                    response += f"• {sym_progress.symbol}{duration}\n"
                    if sym_progress.batch_num and sym_progress.task_num:
                        response += f"  └ Batch {sym_progress.batch_num}/{progress.total_batches}, Task {sym_progress.task_num}/{sym_progress.total_tasks}\n"
                if len(downloading_symbols) > 5:
                    response += f"  ... and {len(downloading_symbols) - 5} more\n"
                response += "\n"
            
            # Recently completed symbols
            completed_symbols = [
                s for s in progress.symbols.values()
                if s.status == DownloadStatus.COMPLETED
            ]
            if completed_symbols:
                # Get last 5 completed
                recent_completed = sorted(
                    completed_symbols,
                    key=lambda x: x.end_time or 0,
                    reverse=True
                )[:5]
                response += f"✅ *Recently Completed ({len(recent_completed)}):*\n"
                for sym_progress in recent_completed:
                    rows_info = ""
                    if sym_progress.rows_downloaded > 0:
                        rows_info = f" ({sym_progress.rows_downloaded:,} rows)"
                    response += f"• {sym_progress.symbol}{rows_info}\n"
                response += "\n"
            
            # Failed symbols
            failed_symbols = [
                s for s in progress.symbols.values()
                if s.status == DownloadStatus.FAILED
            ]
            if failed_symbols:
                response += f"❌ *Failed ({len(failed_symbols)}):*\n"
                for sym_progress in failed_symbols[:3]:  # Limit to 3
                    error_msg = sym_progress.error or "Unknown error"
                    response += f"• {sym_progress.symbol}: {error_msg}\n"
                if len(failed_symbols) > 3:
                    response += f"  ... and {len(failed_symbols) - 3} more\n"
                response += "\n"
            
            # Batch information
            if progress.batch_num and progress.total_batches:
                response += f"📦 *Batch:* {progress.batch_num}/{progress.total_batches}\n\n"
            
            response += "_Use /download to refresh progress_"
            
            return response
            
        except Exception as e:
            logger.exception("download_command_failed", error=str(e))
            return f"❌ Download progress check failed:\n\n`{str(e)}`"

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
            print(f"✅ Telegram command handler started - listening for /health, /status, /help, /grok, /download")
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

