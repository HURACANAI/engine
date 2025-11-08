#!/usr/bin/env python3
"""
Comprehensive local Dropbox test script.
Tests Dropbox integration exactly as the engine code does.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import dropbox
    from dropbox.exceptions import AuthError, ApiError
except ImportError:
    print("❌ dropbox package not installed. Install with: pip install dropbox")
    sys.exit(1)

# Import the actual engine code
try:
    from src.cloud.training.config.settings import EngineSettings
    from src.cloud.training.integrations.dropbox_sync import DropboxSync
except ImportError as e:
    print(f"❌ Failed to import engine code: {e}")
    print("Make sure you're running from the project root and dependencies are installed.")
    sys.exit(1)


def print_header(text: str):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70 + "\n")


def test_token_from_settings():
    """Test token loading from settings (same as engine code)."""
    print_header("📋 TEST 1: Loading Token from Settings")
    
    try:
        settings = EngineSettings.load()
        print(f"✅ Settings loaded successfully")
        print(f"   Dropbox enabled: {settings.dropbox.enabled}")
        print(f"   App folder: {settings.dropbox.app_folder}")
        
        # Get token exactly as the engine code does
        dropbox_token_raw = os.getenv("DROPBOX_ACCESS_TOKEN") or settings.dropbox.access_token
        print(f"\n   Token source: {'Environment variable' if os.getenv('DROPBOX_ACCESS_TOKEN') else 'Settings file'}")
        
        if not dropbox_token_raw:
            print("❌ No Dropbox token found!")
            print("   Set DROPBOX_ACCESS_TOKEN environment variable or update settings.py")
            return None
        
        # Clean token exactly as the engine code does
        dropbox_token = dropbox_token_raw.strip().strip('"').strip("'").strip()
        
        print(f"   Token length: {len(dropbox_token)}")
        print(f"   Token prefix: {dropbox_token[:30]}...")
        
        if not dropbox_token.startswith("sl."):
            print(f"❌ Token format invalid - should start with 'sl.'")
            print(f"   Got: {dropbox_token[:20]}...")
            return None
        
        print("✅ Token format valid")
        return dropbox_token
        
    except Exception as e:
        print(f"❌ Failed to load settings: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_direct_authentication(token: str):
    """Test direct Dropbox authentication."""
    print_header("🔐 TEST 2: Direct Dropbox Authentication")
    
    try:
        print("🔌 Initializing Dropbox client...")
        dbx = dropbox.Dropbox(token)
        print("✅ Dropbox client initialized")
    except Exception as e:
        print(f"❌ Failed to initialize Dropbox client: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    try:
        print("\n🔐 Testing authentication...")
        account_info = dbx.users_get_current_account()
        print(f"✅ Authentication successful!")
        print(f"   Account: {account_info.email}")
        print(f"   Name: {account_info.name.display_name}")
        print(f"   Account ID: {account_info.account_id}")
        return True
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
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_file_operations(token: str):
    """Test file operations (upload, download, delete)."""
    print_header("📁 TEST 3: File Operations")
    
    try:
        dbx = dropbox.Dropbox(token)
        test_path = "/test_engine_connection.txt"
        test_content = b"Engine connection test - you can delete this file"
        
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


def test_dropbox_sync_class(token: str):
    """Test DropboxSync class (exactly as engine uses it)."""
    print_header("🔄 TEST 4: DropboxSync Class (Engine Integration)")
    
    try:
        settings = EngineSettings.load()
        app_folder = settings.dropbox.app_folder or "Runpodhuracan"
        
        print(f"Initializing DropboxSync with:")
        print(f"   App folder: {app_folder}")
        print(f"   Create dated folder: True")
        
        # Initialize exactly as engine code does
        dropbox_sync = DropboxSync(
            access_token=token,
            app_folder=app_folder,
            enabled=True,
            create_dated_folder=True,
        )
        
        print("✅ DropboxSync initialized successfully")
        
        # Check dated folder
        if hasattr(dropbox_sync, "_dated_folder") and dropbox_sync._dated_folder:
            print(f"✅ Dated folder created: {dropbox_sync._dated_folder}")
        else:
            print("⚠️  Dated folder not created (may already exist)")
        
        # Test file upload
        print("\n📤 Testing file upload through DropboxSync...")
        test_file = Path("test_dropbox_sync.txt")
        test_file.write_text("Test content from DropboxSync class")
        
        remote_path = "/test/test_dropbox_sync.txt"
        success = dropbox_sync.upload_file(test_file, remote_path)
        
        if success:
            print(f"✅ File upload successful: {remote_path}")
            
            # Clean up
            try:
                dbx = dropbox.Dropbox(token)
                dbx.files_delete_v2(remote_path)
                print(f"✅ Test file deleted")
            except:
                pass
        else:
            print(f"❌ File upload failed")
        
        # Clean up local test file
        if test_file.exists():
            test_file.unlink()
        
        return True
        
    except Exception as e:
        print(f"❌ DropboxSync test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_folder_operations(token: str):
    """Test folder creation operations."""
    print_header("📂 TEST 5: Folder Operations")
    
    try:
        settings = EngineSettings.load()
        app_folder = settings.dropbox.app_folder or "Runpodhuracan"
        
        dbx = dropbox.Dropbox(token)
        
        # Test creating app folder
        print(f"Testing folder creation in: /{app_folder}")
        
        # Try to list app folder (this tests metadata.read permission)
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
    print_header("🧪 DROPBOX LOCAL TEST SUITE")
    print("This script tests Dropbox integration exactly as the engine code does.")
    print("It will help diagnose authentication and file storage issues.\n")
    
    # Test 1: Load token from settings
    token = test_token_from_settings()
    if not token:
        print("\n❌ Cannot proceed without a valid token.")
        print("   Please set DROPBOX_ACCESS_TOKEN environment variable or update settings.py")
        sys.exit(1)
    
    # Test 2: Direct authentication
    auth_ok = test_direct_authentication(token)
    if not auth_ok:
        print("\n❌ Authentication failed. Cannot proceed with other tests.")
        sys.exit(1)
    
    # Test 3: File operations
    file_ops_ok = test_file_operations(token)
    
    # Test 4: Folder operations
    folder_ops_ok = test_folder_operations(token)
    
    # Test 5: DropboxSync class
    sync_class_ok = test_dropbox_sync_class(token)
    
    # Summary
    print_header("📊 TEST SUMMARY")
    print(f"Token loading:        {'✅ PASS' if token else '❌ FAIL'}")
    print(f"Authentication:       {'✅ PASS' if auth_ok else '❌ FAIL'}")
    print(f"File operations:      {'✅ PASS' if file_ops_ok else '❌ FAIL'}")
    print(f"Folder operations:    {'✅ PASS' if folder_ops_ok else '❌ FAIL'}")
    print(f"DropboxSync class:    {'✅ PASS' if sync_class_ok else '❌ FAIL'}")
    
    if all([token, auth_ok, file_ops_ok, folder_ops_ok, sync_class_ok]):
        print("\n🎉 All tests passed! Dropbox integration is working correctly.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

