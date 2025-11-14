#!/usr/bin/env python3
"""Start Telegram bot command handler for interactive commands."""

import sys
import signal
from pathlib import Path

# Add src to path (scripts/ is one level down from root)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from cloud.training.config.settings import EngineSettings
from cloud.training.monitoring.telegram_command_handler import TelegramCommandHandler


def main():
    """Start Telegram command handler."""
    print("=" * 60)
    print("🤖 Starting Telegram Bot Command Handler")
    print("=" * 60)
    print()
    
    # Load settings
    print("📋 Loading settings...")
    try:
        settings = EngineSettings.load()
        notification_settings = settings.notifications
        
        # Extract bot token and chat ID from webhook URL
        webhook_url = notification_settings.telegram_webhook_url
        if not webhook_url:
            print("❌ ERROR: Telegram webhook URL not configured")
            print("   Please set telegram_webhook_url in config/base.yaml")
            sys.exit(1)
        
        # Extract bot token from webhook URL
        # Format: https://api.telegram.org/bot<TOKEN>/sendMessage
        if "/bot" in webhook_url:
            bot_token = webhook_url.split("/bot")[-1].split("/")[0]
        else:
            print("❌ ERROR: Could not extract bot token from webhook URL")
            sys.exit(1)
        
        chat_id = notification_settings.telegram_chat_id
        if not chat_id:
            print("❌ ERROR: Telegram chat ID not configured")
            print("   Please set telegram_chat_id in config/base.yaml")
            sys.exit(1)
        
        print(f"✅ Settings loaded")
        print(f"   📱 Bot token: {bot_token[:20]}...")
        print(f"   💬 Chat ID: {chat_id}")
        print(f"   🔑 Grok API key: {'✅ Set' if notification_settings.grok_api_key else '❌ Not set'}")
        print()
        
    except Exception as e:
        print(f"❌ ERROR: Failed to load settings: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Create command handler
    print("🔧 Creating command handler...")
    try:
        handler = TelegramCommandHandler(
            bot_token=bot_token,
            chat_id=chat_id,
            settings=settings
        )
        print("✅ Command handler created")
        print()
    except Exception as e:
        print(f"❌ ERROR: Failed to create handler: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Setup signal handler for graceful shutdown
    def signal_handler(sig, frame):
        print("\n\n🛑 Shutting down...")
        handler.stop_polling()
        print("✅ Bot stopped gracefully")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start polling
    print("🚀 Starting bot...")
    print()
    print("=" * 60)
    print("📱 Bot is now listening for commands!")
    print("=" * 60)
    print()
    print("Available commands:")
    print("  /health  - Comprehensive health check")
    print("  /status  - Quick system status")
    print("  /download - Download progress")
    print("  /grok <question> - Ask Grok AI about the codebase")
    print("  /help    - Show help message")
    print()
    print("💡 Send commands to your Telegram bot to test them!")
    print()
    print("Press Ctrl+C to stop the bot")
    print("=" * 60)
    print()
    
    try:
        handler.start_polling()
        
        # Keep main thread alive
        while handler.running:
            import time
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down...")
        handler.stop_polling()
        print("✅ Bot stopped gracefully")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        handler.stop_polling()
        sys.exit(1)


if __name__ == "__main__":
    main()

