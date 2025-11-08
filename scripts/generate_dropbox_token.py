#!/usr/bin/env python3
"""Helper script to generate Dropbox access token using App Key and App Secret."""

import sys
import webbrowser
from urllib.parse import urlencode

# App credentials
APP_KEY = "yxnputg7g9kijch"
APP_SECRET = "8llmdzmxj5hw6i8"

def generate_token_instructions():
    """Print instructions for generating a Dropbox access token."""
    print("=" * 70)
    print("📦 Dropbox Access Token Generator")
    print("=" * 70)
    print()
    print("You have two options to generate an access token:")
    print()
    print("OPTION 1: Generate token from Dropbox App Console (EASIEST)")
    print("-" * 70)
    print("1. Go to: https://www.dropbox.com/developers/apps")
    print("2. Find your app (or create a new one)")
    print("3. Go to the 'Settings' tab")
    print("4. Scroll down to 'OAuth 2' section")
    print("5. Click 'Generate' button under 'Generated access token'")
    print("6. Copy the token (it should start with 'sl.')")
    print("7. The token will be long (1000+ characters)")
    print()
    print("OPTION 2: Use OAuth2 Flow (More complex)")
    print("-" * 70)
    print("This requires opening a browser and authorizing the app.")
    print()
    
    # Build OAuth URL
    oauth_url = (
        f"https://www.dropbox.com/oauth2/authorize?"
        f"client_id={APP_KEY}&"
        f"response_type=code&"
        f"token_access_type=offline"
    )
    
    print(f"OAuth URL: {oauth_url}")
    print()
    print("After authorization, you'll get a code that needs to be exchanged")
    print("for an access token. This is more complex and requires a web server.")
    print()
    print("RECOMMENDATION: Use Option 1 (App Console) - it's much easier!")
    print()
    print("=" * 70)
    
    # Ask if user wants to open the browser
    response = input("Open Dropbox App Console in browser? (y/n): ").strip().lower()
    if response == 'y':
        webbrowser.open("https://www.dropbox.com/developers/apps")
        print("✅ Browser opened!")
        print()
        print("After generating the token, update it in:")
        print("  - src/cloud/training/config/settings.py (DropboxSettings.access_token)")
        print("  - src/cloud/training/pipelines/daily_retrain.py (dropbox_token_raw)")
        print()

if __name__ == "__main__":
    generate_token_instructions()




