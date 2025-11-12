#!/usr/bin/env python3
"""Test the /grok Telegram command handler."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from cloud.training.config.settings import EngineSettings
from cloud.training.monitoring.telegram_command_handler import TelegramCommandHandler


def main():
    """Test Grok command handler."""
    print("🧪 Testing /grok Telegram Command Handler...")
    print("=" * 60)
    
    # Load settings
    print("\n1️⃣  Loading settings...")
    try:
        settings = EngineSettings.load()
        notification_settings = settings.notifications
        
        # Extract bot token and chat ID from webhook URL
        webhook_url = notification_settings.telegram_webhook_url
        if not webhook_url:
            print("   ❌ Telegram webhook URL not configured")
            return
        
        # Extract bot token from webhook URL
        # Format: https://api.telegram.org/bot<TOKEN>/sendMessage
        if "/bot" in webhook_url:
            bot_token = webhook_url.split("/bot")[-1].split("/")[0]
        else:
            print("   ❌ Could not extract bot token from webhook URL")
            return
        
        chat_id = notification_settings.telegram_chat_id
        if not chat_id:
            print("   ❌ Telegram chat ID not configured")
            return
        
        print(f"   ✅ Settings loaded")
        print(f"   📱 Bot token: {bot_token[:20]}...")
        print(f"   💬 Chat ID: {chat_id}")
        print(f"   🔑 Grok API key: {'✅ Set' if notification_settings.grok_api_key else '❌ Not set'}")
        
    except Exception as e:
        print(f"   ❌ Failed to load settings: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Create command handler
    print("\n2️⃣  Creating Telegram command handler...")
    try:
        handler = TelegramCommandHandler(
            bot_token=bot_token,
            chat_id=chat_id,
            settings=settings
        )
        print("   ✅ Command handler created")
    except Exception as e:
        print(f"   ❌ Failed to create handler: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test Grok command without question
    print("\n3️⃣  Testing /grok command (no question)...")
    try:
        response = handler.handle_grok_command("")
        print("   ✅ Response received:")
        print(f"   {response[:200]}...")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test Grok command with question
    print("\n4️⃣  Testing /grok command (with question)...")
    try:
        question = "How does the training pipeline work?"
        response = handler.handle_grok_command(question)
        print("   ✅ Response received:")
        print(f"   {response[:300]}...")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test command handler
    print("\n5️⃣  Testing command handler...")
    try:
        response = handler.handle_command("/grok", "/grok How does walk-forward validation work?")
        print("   ✅ Response received:")
        print(f"   {response[:300]}...")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("✅ Test complete!")
    print("=" * 60)
    print("\n💡 To test in Telegram:")
    print("   1. Start the command handler (if not already running)")
    print("   2. Send `/grok <your question>` to your Telegram bot")
    print("   3. Wait for the response")


if __name__ == "__main__":
    main()

