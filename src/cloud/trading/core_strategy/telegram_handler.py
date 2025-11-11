"""
Telegram Command Handler for CoreBook Strategy

Handles Telegram bot commands for CoreBook management.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import structlog

from .core_strategy import CoreBookStrategy, CoreBookConfig

logger = structlog.get_logger(__name__)


class CoreBookTelegramHandler:
    """Telegram command handler for CoreBook strategy."""
    
    def __init__(self, core_strategy: CoreBookStrategy):
        """Initialize Telegram handler.
        
        Args:
            core_strategy: CoreBook strategy instance
        """
        self.core_strategy = core_strategy
        logger.info("core_book_telegram_handler_initialized")
    
    def handle_command(self, command: str, args: List[str]) -> str:
        """Handle Telegram command.
        
        Args:
            command: Command name
            args: Command arguments
            
        Returns:
            Response message
        """
        try:
            if command == "status":
                return self._handle_status(args)
            elif command == "cap":
                return self._handle_cap(args)
            elif command == "add":
                return self._handle_add(args)
            elif command == "trim":
                return self._handle_trim(args)
            elif command == "auto":
                return self._handle_auto(args)
            else:
                return f"Unknown command: {command}. Available commands: status, cap, add, trim, auto"
        except Exception as e:
            logger.error("command_handler_error", command=command, error=str(e))
            return f"Error processing command: {str(e)}"
    
    def _handle_status(self, args: List[str]) -> str:
        """Handle /core status command.
        
        Args:
            args: Command arguments (optional coin symbol)
            
        Returns:
            Status message
        """
        if args:
            # Status for specific coin
            symbol = args[0].upper()
            coin_status = self.core_strategy.get_coin_status(symbol)
            
            if not coin_status:
                return f"Coin {symbol} not found in CoreBook."
            
            message = f"📊 CoreBook Status: {symbol}\n\n"
            message += f"Units: {coin_status['units_held']:.6f}\n"
            message += f"Avg Cost: ${coin_status['average_cost_price']:.2f}\n"
            message += f"Cost Basis: ${coin_status['total_cost_basis']:.2f}\n"
            message += f"Exposure Limit: {coin_status['total_exposure_limit_pct']:.1f}%\n"
            message += f"DCA Count: {coin_status['dca_count']}/{coin_status['max_dca_buys']}\n"
            
            if coin_status['next_dca_trigger_price']:
                message += f"DCA Trigger: ${coin_status['next_dca_trigger_price']:.2f}\n"
            if coin_status['partial_sell_target_price']:
                message += f"Sell Target: ${coin_status['partial_sell_target_price']:.2f}\n"
            
            if coin_status['last_action_timestamp']:
                message += f"Last Action: {coin_status['last_action_type']} at {coin_status['last_action_timestamp']}\n"
            
            return message
        else:
            # Status for all coins
            status = self.core_strategy.get_status()
            
            message = "📊 CoreBook Status\n\n"
            message += f"Auto Trading: {'✅ Enabled' if status['auto_trading_enabled'] else '❌ Disabled'}\n\n"
            
            for symbol, coin_status in status['coins'].items():
                message += f"**{symbol}**\n"
                message += f"  Units: {coin_status['units_held']:.6f}\n"
                message += f"  Avg Cost: ${coin_status['average_cost_price']:.2f}\n"
                message += f"  Exposure Limit: {coin_status['total_exposure_limit_pct']:.1f}%\n"
                message += f"  DCA Count: {coin_status['dca_count']}/{coin_status['max_dca_buys']}\n"
                
                if coin_status['next_dca_trigger_price']:
                    message += f"  DCA Trigger: ${coin_status['next_dca_trigger_price']:.2f}\n"
                if coin_status['partial_sell_target_price']:
                    message += f"  Sell Target: ${coin_status['partial_sell_target_price']:.2f}\n"
                
                message += "\n"
            
            return message
    
    def _handle_cap(self, args: List[str]) -> str:
        """Handle /core cap <coin> <percent> command.
        
        Args:
            args: Command arguments [coin, percent]
            
        Returns:
            Response message
        """
        if len(args) < 2:
            return "Usage: /core cap <coin> <percent>"
        
        symbol = args[0].upper()
        try:
            exposure_pct = float(args[1])
        except ValueError:
            return f"Invalid percent: {args[1]}"
        
        if exposure_pct < 0 or exposure_pct > 100:
            return "Exposure percent must be between 0 and 100"
        
        success = self.core_strategy.set_exposure_cap(symbol, exposure_pct)
        
        if success:
            return f"✅ Exposure cap set for {symbol}: {exposure_pct:.1f}%"
        else:
            return f"❌ Failed to set exposure cap for {symbol}"
    
    def _handle_add(self, args: List[str]) -> str:
        """Handle /core add <coin> <percent> command.
        
        Args:
            args: Command arguments [coin, percent]
            
        Returns:
            Response message
        """
        if len(args) < 2:
            return "Usage: /core add <coin> <percent>"
        
        symbol = args[0].upper()
        try:
            exposure_pct = float(args[1])
        except ValueError:
            return f"Invalid percent: {args[1]}"
        
        if exposure_pct < 0 or exposure_pct > 100:
            return "Exposure percent must be between 0 and 100"
        
        success = self.core_strategy.add_coin(symbol, exposure_pct)
        
        if success:
            return f"✅ Added {symbol} to CoreBook with exposure cap: {exposure_pct:.1f}%"
        else:
            return f"❌ Failed to add {symbol} to CoreBook"
    
    def _handle_trim(self, args: List[str]) -> str:
        """Handle /core trim <coin> [percent] command.
        
        Args:
            args: Command arguments [coin, percent (optional)]
            
        Returns:
            Response message
        """
        if len(args) < 1:
            return "Usage: /core trim <coin> [percent]"
        
        symbol = args[0].upper()
        trim_pct = float(args[1]) if len(args) > 1 else 25.0
        
        if trim_pct < 0 or trim_pct > 100:
            return "Trim percent must be between 0 and 100"
        
        success = self.core_strategy.trim_position(symbol, trim_pct)
        
        if success:
            return f"✅ Trimmed {symbol} position by {trim_pct:.1f}%"
        else:
            return f"❌ Failed to trim {symbol} position"
    
    def _handle_auto(self, args: List[str]) -> str:
        """Handle /core auto on/off command.
        
        Args:
            args: Command arguments [on/off]
            
        Returns:
            Response message
        """
        if len(args) < 1:
            return "Usage: /core auto on/off"
        
        enabled_str = args[0].lower()
        
        if enabled_str == "on":
            enabled = True
        elif enabled_str == "off":
            enabled = False
        else:
            return "Usage: /core auto on/off"
        
        success = self.core_strategy.set_auto_trading(enabled)
        
        if success:
            status = "enabled" if enabled else "disabled"
            return f"✅ Auto trading {status}"
        else:
            return "❌ Failed to set auto trading"


class TelegramBotIntegration:
    """Integration with Telegram bot."""
    
    def __init__(
        self,
        core_strategy: CoreBookStrategy,
        telegram_token: Optional[str] = None,
        chat_id: Optional[str] = None,
    ):
        """Initialize Telegram bot integration.
        
        Args:
            core_strategy: CoreBook strategy instance
            telegram_token: Telegram bot token
            chat_id: Telegram chat ID
        """
        self.core_strategy = core_strategy
        self.handler = CoreBookTelegramHandler(core_strategy)
        self.telegram_token = telegram_token
        self.chat_id = chat_id
        
        # TODO: Initialize Telegram bot
        logger.info("telegram_bot_integration_initialized")
    
    def process_message(self, message: str) -> str:
        """Process Telegram message.
        
        Args:
            message: Telegram message text
            
        Returns:
            Response message
        """
        # Parse command
        parts = message.strip().split()
        
        if not parts:
            return "Invalid command"
        
        command = parts[0].lower()
        args = parts[1:] if len(parts) > 1 else []
        
        # Handle /core commands
        if command == "/core" and args:
            subcommand = args[0].lower()
            subargs = args[1:] if len(args) > 1 else []
            return self.handler.handle_command(subcommand, subargs)
        elif command == "/core":
            return self.handler.handle_command("status", [])
        
        return "Unknown command"
    
    def send_message(self, message: str) -> bool:
        """Send message to Telegram.
        
        Args:
            message: Message text
            
        Returns:
            True if successful
        """
        # TODO: Implement Telegram message sending
        logger.info("telegram_message_sent", message=message[:100])
        return True

