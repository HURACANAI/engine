"""
Get Telegram Chat ID

Run this script and send a message to your bot to get your chat_id.
The bot token is: 8229109041:AAFIcLRx3V50khoaEIG7WXeI1ITzy4s6hf0

Usage:
1. Start this script
2. Send a message to your bot on Telegram
3. The script will print your chat_id
"""

import requests
import time

BOT_TOKEN = "8229109041:AAFIcLRx3V50khoaEIG7WXeI1ITzy4s6hf0"
API_URL = f"https://api.telegram.org/bot{BOT_TOKEN}"

def get_updates():
    """Get updates from Telegram bot."""
    url = f"{API_URL}/getUpdates"
    response = requests.get(url, timeout=10)
    if response.ok:
        return response.json()
    return None

def get_chat_id():
    """Get chat ID from latest message."""
    print("=" * 60)
    print("TELEGRAM CHAT ID GETTER")
    print("=" * 60)
    print()
    print(f"Bot Token: {BOT_TOKEN[:20]}...")
    print()
    print("Instructions:")
    print("1. Open Telegram and find your bot")
    print("2. Send any message to the bot (e.g., '/start' or 'Hello')")
    print("3. Wait a few seconds...")
    print()
    print("Checking for messages...")
    print()
    
    last_update_id = None
    
    while True:
        try:
            updates = get_updates()
            if updates and updates.get("ok"):
                results = updates.get("result", [])
                if results:
                    latest = results[-1]
                    update_id = latest.get("update_id")
                    
                    if update_id != last_update_id:
                        message = latest.get("message")
                        if message:
                            chat = message.get("chat")
                            if chat:
                                chat_id = chat.get("id")
                                chat_type = chat.get("type")
                                first_name = chat.get("first_name", "")
                                username = chat.get("username", "")
                                
                                print("=" * 60)
                                print("✅ MESSAGE RECEIVED!")
                                print("=" * 60)
                                print()
                                print(f"Chat ID: {chat_id}")
                                print(f"Chat Type: {chat_type}")
                                print(f"Name: {first_name}")
                                if username:
                                    print(f"Username: @{username}")
                                print()
                                print("=" * 60)
                                print("ADD THIS TO YOUR CONFIG:")
                                print("=" * 60)
                                print()
                                print(f"telegram_chat_id: {chat_id}")
                                print()
                                print("=" * 60)
                                return chat_id
                    
                    last_update_id = update_id
            
            time.sleep(2)
            print(".", end="", flush=True)
            
        except KeyboardInterrupt:
            print("\n\nStopped by user.")
            break
        except Exception as e:
            print(f"\nError: {e}")
            time.sleep(5)

if __name__ == "__main__":
    get_chat_id()

