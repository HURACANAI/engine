"""
Telegram Service for Notifications

Sends training summaries and alerts to Telegram.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import structlog  # type: ignore[import-untyped]

logger = structlog.get_logger(__name__)

try:
    import requests  # type: ignore[import-untyped]
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    requests = None  # type: ignore[assignment]


class TelegramService:
    """Telegram service for notifications."""
    
    def __init__(
        self,
        token: Optional[str] = None,
        chat_id: Optional[str] = None,
    ):
        """Initialize Telegram service.
        
        Args:
            token: Telegram bot token (defaults to TELEGRAM_TOKEN env var)
            chat_id: Telegram chat ID (defaults to TELEGRAM_CHAT_ID env var)
        """
        self.token = token or os.getenv("TELEGRAM_TOKEN")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID")
        
        if not self.token or not self.chat_id:
            logger.warning("telegram_not_configured", message="Telegram token or chat_id not set")
            self.enabled = False
        else:
            self.enabled = True
            self.api_url = f"https://api.telegram.org/bot{self.token}/sendMessage"
            logger.info("telegram_service_initialized", chat_id=self.chat_id)
    
    def send_message(self, message: str, parse_mode: str = "HTML") -> bool:
        """Send a message to Telegram.
        
        Args:
            message: Message text
            parse_mode: Parse mode ("HTML", "Markdown", etc.)
            
        Returns:
            True if successful
        """
        if not self.enabled:
            return False
        
        if not REQUESTS_AVAILABLE:
            logger.warning("requests_not_available", message="Cannot send Telegram message without requests package")
            return False
        
        try:
            response = requests.post(  # type: ignore[union-attr]
                self.api_url,
                json={
                    "chat_id": self.chat_id,
                    "text": message,
                    "parse_mode": parse_mode,
                },
                timeout=10,
            )
            response.raise_for_status()
            
            logger.info("telegram_message_sent", chat_id=self.chat_id)
            return True
            
        except Exception as e:
            logger.error("telegram_message_failed", error=str(e))
            return False
    
    def send_start_summary(self, total_symbols: int, mode: str) -> bool:
        """Send training start summary.
        
        Args:
            total_symbols: Total number of symbols
            mode: Training mode (sequential, parallel, hybrid)
            
        Returns:
            True if successful
        """
        message = f"🚀 <b>Engine Training Started</b>\n\n"
        message += f"Mode: {mode}\n"
        message += f"Symbols: {total_symbols}\n"
        message += f"Status: Running"
        
        return self.send_message(message)
    
    def send_completion_summary(
        self,
        total_symbols: int,
        succeeded: int,
        failed: int,
        skipped: int,
        total_wall_minutes: float,
    ) -> bool:
        """Send training completion summary.
        
        Args:
            total_symbols: Total number of symbols
            succeeded: Number of successful trainings
            failed: Number of failed trainings
            skipped: Number of skipped trainings
            total_wall_minutes: Total wall time in minutes
            
        Returns:
            True if successful
        """
        message = f"✅ <b>Engine Training Completed</b>\n\n"
        message += f"Total: {total_symbols}\n"
        message += f"✅ Success: {succeeded}\n"
        message += f"❌ Failed: {failed}\n"
        message += f"⚠️ Skipped: {skipped}\n"
        message += f"⏱️ Duration: {total_wall_minutes:.1f} minutes"
        
        return self.send_message(message)
    
    def send_coin_failure_alert(self, symbol: str, error: str, error_type: Optional[str] = None) -> bool:
        """Send coin failure alert.
        
        Args:
            symbol: Trading symbol
            error: Error message
            error_type: Error type (optional)
            
        Returns:
            True if successful
        """
        message = f"❌ <b>Training Failed</b>\n\n"
        message += f"Symbol: {symbol}\n"
        if error_type:
            message += f"Type: {error_type}\n"
        message += f"Error: {error[:100]}"  # Truncate to 100 chars
        
        return self.send_message(message)

