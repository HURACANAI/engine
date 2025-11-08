#!/usr/bin/env python3
"""
Quick script to update Dropbox token in settings.

Usage:
    python scripts/update_dropbox_token.py YOUR_NEW_TOKEN
    python scripts/update_dropbox_token.py  # Will prompt for token
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def update_token(new_token: str):
    """Update Dropbox token in settings.py"""
    settings_file = project_root / "src" / "cloud" / "training" / "config" / "settings.py"
    
    if not settings_file.exists():
        print(f"❌ Settings file not found: {settings_file}")
        return False
    
    # Read current file
    content = settings_file.read_text()
    
    # Find the token line and replace it
    lines = content.split('\n')
    updated = False
    
    for i, line in enumerate(lines):
        if 'access_token: Optional[str] = os.getenv("DROPBOX_ACCESS_TOKEN") or' in line:
            # Extract the part before the token
            prefix = 'access_token: Optional[str] = os.getenv("DROPBOX_ACCESS_TOKEN") or "'
            if line.startswith(prefix):
                # Replace the token
                lines[i] = prefix + new_token + '"'
                updated = True
                break
    
    if not updated:
        print("❌ Could not find token line in settings.py")
        return False
    
    # Write back
    settings_file.write_text('\n'.join(lines))
    print(f"✅ Token updated in {settings_file}")
    return True

def main():
    if len(sys.argv) > 1:
        token = sys.argv[1]
    else:
        print("Enter your new Dropbox access token:")
        print("(Paste the entire token - it should be 1000+ characters)")
        token = input("Token: ").strip().strip('"').strip("'")
    
    if not token:
        print("❌ No token provided")
        sys.exit(1)
    
    if not token.startswith("sl."):
        print("⚠️  Warning: Token should start with 'sl.'")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    if len(token) < 500:
        print("⚠️  Warning: Token seems short. Valid tokens are typically 1000+ characters.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    if update_token(token):
        print("\n✅ Token updated successfully!")
        print("   Test it with: python -c 'from src.cloud.training.config.settings import EngineSettings; print(\"OK\")'")
    else:
        print("❌ Failed to update token")
        sys.exit(1)

if __name__ == "__main__":
    main()

