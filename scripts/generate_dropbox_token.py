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
    print("⚠️  IMPORTANT: Enable required scopes FIRST!")
    print()
    print("STEP 1: Enable Required Scopes")
    print("  1. Go to: https://www.dropbox.com/developers/apps")
    print("  2. Find your app (ID: 7988481) or create a new one")
    print("  3. Click on your app → Go to 'Permissions' tab")
    print("  4. Under 'Scopes', enable:")
    print("     ✅ files.content.write (REQUIRED for uploads)")
    print("     ✅ files.metadata.read (REQUIRED for listing)")
    print("     ✅ files.content.read (recommended)")
    print()
    print("STEP 2: Generate Access Token")
    print("  1. Go to the 'Settings' tab")
    print("  2. Scroll down to 'OAuth 2' section")
    print("  3. Click 'Generate' button under 'Generated access token'")
    print("  4. Copy the ENTIRE token (it should start with 'sl.')")
    print("  5. The token will be long (1000+ characters)")
    print("  6. Update it in src/cloud/training/config/settings.py")
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





