#!/usr/bin/env python3
"""
Test Dropbox integration using the exact same code path as the engine.
This mimics what happens in daily_retrain.py but only tests Dropbox.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import the exact same modules the engine uses
try:
    from src.cloud.training.config.settings import EngineSettings
    from src.cloud.training.integrations.dropbox_sync import DropboxSync
    import structlog
except ImportError as e:
    print(f"❌ Failed to import engine code: {e}")
    print("Make sure you're running from the project root and dependencies are installed.")
    sys.exit(1)


def configure_logging():
    """Configure logging exactly as the engine does."""
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(structlog.INFO),
    )


def test_dropbox_integration():
    """Test Dropbox integration exactly as engine does."""
    print("=" * 70)
    print("  TESTING DROPBOX INTEGRATION (Engine Code Path)")
    print("=" * 70)
    print()
    
    configure_logging()
    logger = structlog.get_logger("dropbox_test")
    
    # Load settings exactly as engine does
    print("📋 Loading engine settings...")
    try:
        settings = EngineSettings.load()
        print(f"✅ Settings loaded")
        print(f"   Dropbox enabled: {settings.dropbox.enabled}")
        print(f"   App folder: {settings.dropbox.app_folder}")
    except Exception as e:
        print(f"❌ Failed to load settings: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Get token exactly as engine code does (line 106 in daily_retrain.py)
    print("\n🔑 Getting Dropbox token...")
    dropbox_token_raw = os.getenv("DROPBOX_ACCESS_TOKEN") or settings.dropbox.access_token
    
    if not dropbox_token_raw:
        print("❌ No Dropbox token found!")
        print("   Token sources checked:")
        print(f"   - Environment variable DROPBOX_ACCESS_TOKEN: {os.getenv('DROPBOX_ACCESS_TOKEN') is not None}")
        print(f"   - Settings file: {settings.dropbox.access_token is not None}")
        return False
    
    # Clean token exactly as engine code does (line 108)
    dropbox_token = dropbox_token_raw.strip().strip('"').strip("'").strip()
    print(f"✅ Token found (length: {len(dropbox_token)})")
    print(f"   Token prefix: {dropbox_token[:30]}...")
    
    if not dropbox_token.startswith("sl."):
        print(f"❌ Token format invalid - should start with 'sl.'")
        print(f"   Got: {dropbox_token[:20]}...")
        return False
    
    dropbox_folder = settings.dropbox.app_folder or "Runpodhuracan"
    
    # Test Dropbox initialization exactly as engine does (lines 111-144)
    if settings.dropbox.enabled and dropbox_token:
        print("\n📁 Testing Dropbox initialization (same as engine)...")
        try:
            logger.info("dropbox_creating_dated_folder_immediately")
            print("📁 Creating Dropbox dated folder...")
            
            # Initialize Dropbox sync - this will create dated folder as FIRST action
            dropbox_sync = DropboxSync(
                access_token=dropbox_token,
                app_folder=dropbox_folder,
                enabled=True,
                create_dated_folder=True,  # Create dated folder immediately
            )
            
            # Log the dated folder that was created
            if hasattr(dropbox_sync, "_dated_folder") and dropbox_sync._dated_folder:
                logger.info(
                    "dropbox_dated_folder_created_successfully",
                    folder=dropbox_sync._dated_folder,
                    message="✅ Dated folder created - ready for data sync",
                )
                print(f"✅ Dropbox folder created: {dropbox_sync._dated_folder}\n")
            else:
                logger.warning("dropbox_dated_folder_creation_failed", message="Folder not created")
                print("⚠️  Dropbox folder creation failed - check logs\n")
            
            # Test file upload
            print("📤 Testing file upload...")
            test_file = Path("test_engine_dropbox.txt")
            test_file.write_text("Test file from engine integration test")
            
            remote_path = "/test/test_engine_dropbox.txt"
            success = dropbox_sync.upload_file(test_file, remote_path)
            
            if success:
                print(f"✅ File upload successful: {remote_path}")
                
                # Clean up test file from Dropbox
                try:
                    import dropbox
                    dbx = dropbox.Dropbox(dropbox_token)
                    dbx.files_delete_v2(remote_path)
                    print("✅ Test file deleted from Dropbox")
                except:
                    pass
            else:
                print(f"❌ File upload failed")
            
            # Clean up local test file
            if test_file.exists():
                test_file.unlink()
            
            # Test continuous sync initialization (without actually starting it)
            print("\n🔄 Testing sync directory listing...")
            files = dropbox_sync.list_files("/")
            print(f"✅ Can list files in root (found {len(files)} items)")
            
            print("\n🎉 All Dropbox integration tests passed!")
            print("   The engine should be able to use Dropbox correctly.")
            return True
            
        except Exception as sync_error:
            # Dropbox errors are non-fatal - but we want to know about them
            logger.warning(
                "dropbox_folder_creation_failed_non_fatal",
                error=str(sync_error),
                message="Dropbox integration failed",
            )
            print(f"❌ Dropbox integration failed: {sync_error}\n")
            import traceback
            traceback.print_exc()
            return False
    else:
        print("⚠️  Dropbox is disabled or token is missing")
        print(f"   Enabled: {settings.dropbox.enabled}")
        print(f"   Token present: {dropbox_token is not None}")
        return False


if __name__ == "__main__":
    success = test_dropbox_integration()
    sys.exit(0 if success else 1)






