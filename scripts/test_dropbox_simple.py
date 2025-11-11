#!/usr/bin/env python3
"""
Simple standalone Dropbox test - doesn't require engine dependencies.
Tests Dropbox token and file operations directly.
"""

import sys
import os

try:
    import dropbox
    from dropbox.exceptions import AuthError, ApiError
except ImportError:
    print("❌ dropbox package not installed. Install with: pip install dropbox")
    sys.exit(1)


def print_header(text: str):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70 + "\n")


def get_token():
    """Get Dropbox token from environment or user input."""
    print_header("🔑 Getting Dropbox Token")
    
    # Try environment variable first
    token = os.getenv("DROPBOX_ACCESS_TOKEN")
    if token:
        print("✅ Found token in environment variable DROPBOX_ACCESS_TOKEN")
    else:
        # Try reading from settings file (simple approach - just look for the token pattern)
        try:
            settings_file = "src/cloud/training/config/settings.py"
            if os.path.exists(settings_file):
                with open(settings_file, "r") as f:
                    content = f.read()
                    # Look for token pattern (sl. followed by long string)
                    import re
                    match = re.search(r'access_token.*?"(sl\.[^"]+)"', content)
                    if match:
                        token = match.group(1)
                        print(f"✅ Found token in {settings_file}")
                    else:
                        print("⚠️  Could not find token in settings file")
            else:
                print("⚠️  Settings file not found")
        except Exception as e:
            print(f"⚠️  Could not read settings file: {e}")
    
    if not token:
        print("\n❌ No token found!")
        print("\nPlease provide your Dropbox token:")
        print("  Option 1: Set environment variable:")
        print("    export DROPBOX_ACCESS_TOKEN='sl.your_token_here'")
        print("  Option 2: Enter token manually:")
        token = input("\nEnter Dropbox token (or press Enter to exit): ").strip()
        if not token:
            print("❌ No token provided. Exiting.")
            sys.exit(1)
    
    # Clean token
    token = token.strip().strip('"').strip("'").replace("\n", "").replace("\r", "")
    
    print(f"\nToken info:")
    print(f"  Length: {len(token)} characters")
    print(f"  Prefix: {token[:30]}...")
    
    if not token.startswith("sl."):
        print(f"\n❌ Token format invalid - should start with 'sl.'")
        print(f"   Got: {token[:20]}...")
        return None
    
    if len(token) < 50:
        print(f"\n⚠️  Token seems unusually short (may be invalid)")
    
    print("✅ Token format looks valid")
    return token


def test_authentication(token: str):
    """Test Dropbox authentication."""
    print_header("🔐 Testing Authentication")
    
    try:
        print("🔌 Initializing Dropbox client...")
        dbx = dropbox.Dropbox(token)
        print("✅ Dropbox client initialized")
    except Exception as e:
        print(f"❌ Failed to initialize Dropbox client: {e}")
        return False
    
    try:
        print("\n🔐 Testing authentication...")
        account_info = dbx.users_get_current_account()
        print(f"✅ Authentication successful!")
        print(f"   Account: {account_info.email}")
        print(f"   Name: {account_info.name.display_name}")
        print(f"   Account ID: {account_info.account_id}")
        return True, dbx
    except AuthError as e:
        print(f"❌ Authentication failed: {e}")
        print("\n💡 Possible issues:")
        print("   1. Token is expired - generate a new token")
        print("   2. Token was revoked - check Dropbox app settings")
        print("   3. Token doesn't have required permissions")
        print("   4. Token is for a different app")
        print("\n🔧 How to fix:")
        print("   1. Go to https://www.dropbox.com/developers/apps")
        print("   2. Select your app (or create a new one)")
        print("   3. Go to 'Permissions' tab")
        print("   4. Ensure 'files.content.write' and 'files.metadata.read' are enabled")
        print("   5. Generate a new access token")
        print("   6. Copy the full token (it should start with 'sl.')")
        return False, None
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_file_operations(dbx):
    """Test file operations."""
    print_header("📁 Testing File Operations")
    
    test_path = "/test_engine_connection.txt"
    test_content = b"Engine connection test - you can delete this file"
    
    try:
        # Upload test file
        print("📤 Uploading test file...")
        dbx.files_upload(
            test_content,
            test_path,
            mode=dropbox.files.WriteMode.overwrite,
        )
        print(f"✅ File upload successful: {test_path}")
        
        # Download test file
        print("\n📥 Downloading test file...")
        metadata, response = dbx.files_download(test_path)
        print(f"✅ File download successful: {metadata.name}")
        print(f"   File size: {len(response.content)} bytes")
        
        # Verify content
        if response.content == test_content:
            print("✅ File content matches")
        else:
            print("⚠️  File content doesn't match (but download worked)")
        
        # Delete test file
        print("\n🗑️  Deleting test file...")
        dbx.files_delete_v2(test_path)
        print(f"✅ File delete successful")
        
        return True
        
    except ApiError as e:
        print(f"❌ File operation failed: {e}")
        print("\n💡 Possible issues:")
        print("   1. Token doesn't have 'files.content.write' permission")
        print("   2. Token doesn't have 'files.content.read' permission")
        print("   3. App folder access is restricted")
        print("\n🔧 How to fix:")
        print("   1. Go to https://www.dropbox.com/developers/apps")
        print("   2. Select your app")
        print("   3. Go to 'Permissions' tab")
        print("   4. Enable 'files.content.write' and 'files.metadata.read'")
        print("   5. Generate a new access token")
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_folder_operations(dbx, app_folder: str = "Runpodhuracan"):
    """Test folder operations."""
    print_header("📂 Testing Folder Operations")
    
    try:
        # Test creating app folder
        print(f"Testing folder operations in: /{app_folder}")
        
        # Try to list app folder
        try:
            result = dbx.files_list_folder(f"/{app_folder}")
            print(f"✅ Can list folder: /{app_folder}")
            print(f"   Found {len(result.entries)} items")
        except ApiError as e:
            if "not_found" in str(e.error):
                print(f"⚠️  Folder doesn't exist yet, trying to create...")
                try:
                    dbx.files_create_folder_v2(f"/{app_folder}")
                    print(f"✅ Created folder: /{app_folder}")
                except ApiError as e2:
                    print(f"❌ Failed to create folder: {e2}")
                    print("   This might be a permissions issue")
                    return False
            else:
                print(f"❌ Failed to list folder: {e}")
                return False
        
        # Test creating dated folder
        from datetime import datetime, timezone
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        dated_path = f"/{app_folder}/{today}"
        
        print(f"\nTesting dated folder creation: {dated_path}")
        try:
            dbx.files_create_folder_v2(dated_path)
            print(f"✅ Created dated folder: {dated_path}")
        except ApiError as e:
            if "path/conflict/folder" in str(e.error):
                print(f"✅ Dated folder already exists: {dated_path}")
            else:
                print(f"❌ Failed to create dated folder: {e}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Folder operations test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print_header("🧪 DROPBOX SIMPLE TEST")
    print("This script tests Dropbox integration without requiring engine dependencies.")
    print("It will help diagnose authentication and file storage issues.\n")
    
    # Get token
    token = get_token()
    if not token:
        print("\n❌ Cannot proceed without a valid token.")
        sys.exit(1)
    
    # Test authentication
    auth_ok, dbx = test_authentication(token)
    if not auth_ok:
        print("\n❌ Authentication failed. Cannot proceed with other tests.")
        sys.exit(1)
    
    # Test file operations
    file_ops_ok = test_file_operations(dbx)
    
    # Test folder operations
    folder_ops_ok = test_folder_operations(dbx)
    
    # Summary
    print_header("📊 TEST SUMMARY")
    print(f"Token loading:        {'✅ PASS' if token else '❌ FAIL'}")
    print(f"Authentication:       {'✅ PASS' if auth_ok else '❌ FAIL'}")
    print(f"File operations:      {'✅ PASS' if file_ops_ok else '❌ FAIL'}")
    print(f"Folder operations:    {'✅ PASS' if folder_ops_ok else '❌ FAIL'}")
    
    if all([token, auth_ok, file_ops_ok, folder_ops_ok]):
        print("\n🎉 All tests passed! Dropbox integration should work.")
        print("\n💡 Next steps:")
        print("   1. Make sure your token is set in environment variable or settings.py")
        print("   2. Run the engine and check if Dropbox sync works")
        print("   3. Check Dropbox app folder to see if files are being uploaded")
        return 0
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")
        print("\n💡 Common fixes:")
        print("   1. Generate a new token from Dropbox App Console")
        print("   2. Ensure permissions are enabled (files.content.write, files.metadata.read)")
        print("   3. Make sure token is not expired or revoked")
        return 1


if __name__ == "__main__":
    sys.exit(main())






