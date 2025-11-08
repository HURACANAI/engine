#!/usr/bin/env python3
"""Quick diagnostic script to check common issues."""

import sys
import os

print("=" * 70)
print("QUICK DIAGNOSTIC")
print("=" * 70)
print()

# Check Python version
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")
print()

# Check Dropbox token
print("Checking Dropbox token...")
token = os.getenv("DROPBOX_ACCESS_TOKEN")
if token:
    print(f"✅ Token found in environment (length: {len(token)})")
else:
    print("⚠️  No token in environment variable")
    # Try reading from settings
    try:
        settings_file = "src/cloud/training/config/settings.py"
        if os.path.exists(settings_file):
            with open(settings_file, "r") as f:
                content = f.read()
                import re
                match = re.search(r'access_token.*?"(sl\.[^"]+)"', content)
                if match:
                    token = match.group(1)
                    print(f"✅ Token found in settings file (length: {len(token)})")
                else:
                    print("❌ No token found in settings file")
        else:
            print("❌ Settings file not found")
    except Exception as e:
        print(f"❌ Error reading settings: {e}")

if token:
    print(f"Token prefix: {token[:30]}...")
    if token.startswith("sl."):
        print("✅ Token format looks valid")
    else:
        print("❌ Token format invalid (should start with 'sl.')")
print()

# Check Dropbox package
print("Checking Dropbox package...")
try:
    import dropbox
    print("✅ dropbox package installed")
    
    # Test token if available
    if token:
        try:
            dbx = dropbox.Dropbox(token)
            account = dbx.users_get_current_account()
            print(f"✅ Token is valid! Connected to: {account.email}")
        except Exception as e:
            print(f"❌ Token authentication failed: {e}")
            if "expired" in str(e).lower():
                print("   → Token is EXPIRED. Generate a new one!")
except ImportError:
    print("❌ dropbox package not installed")
    print("   Install with: pip install dropbox")
print()

# Check other dependencies
print("Checking other dependencies...")
deps = ["pydantic", "pydantic_settings", "structlog", "yaml"]
missing = []
for dep in deps:
    try:
        __import__(dep.replace("_", "."))
        print(f"✅ {dep}")
    except ImportError:
        print(f"❌ {dep} - MISSING")
        missing.append(dep)

if missing:
    print()
    print("⚠️  Missing dependencies. Install with:")
    print(f"   pip install {' '.join(missing)}")

print()
print("=" * 70)


