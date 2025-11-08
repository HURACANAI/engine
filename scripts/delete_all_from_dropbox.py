#!/usr/bin/env python3
"""
Delete all files and folders from Dropbox candles directory.

This will clean everything so we can start fresh.

Usage:
    python scripts/delete_all_from_dropbox.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.cloud.training.config.settings import EngineSettings
from src.cloud.training.integrations.dropbox_sync import DropboxSync
import dropbox
import os

os.environ['HURACAN_ENV'] = 'local'

def delete_recursive(dbx, path):
    """Recursively delete all files and folders."""
    try:
        result = dbx.files_list_folder(path)
        for entry in result.entries:
            if isinstance(entry, dropbox.files.FolderMetadata):
                # Recursively delete folder contents first
                delete_recursive(dbx, entry.path_lower)
                # Then delete the folder
                try:
                    dbx.files_delete_v2(entry.path_lower)
                    print(f"   🗑️  Deleted folder: {entry.path_lower}")
                except Exception as e:
                    print(f"   ⚠️  Could not delete folder {entry.path_lower}: {e}")
            elif isinstance(entry, dropbox.files.FileMetadata):
                try:
                    dbx.files_delete_v2(entry.path_lower)
                    print(f"   🗑️  Deleted file: {entry.name}")
                except Exception as e:
                    print(f"   ⚠️  Could not delete file {entry.name}: {e}")
    except Exception as e:
        print(f"   ⚠️  Error listing {path}: {e}")

def main():
    settings = EngineSettings.load()
    sync = DropboxSync(
        access_token=settings.dropbox.access_token,
        app_folder=settings.dropbox.app_folder,
        enabled=True,
        create_dated_folder=False,
    )
    
    print("=" * 60)
    print("🗑️  Delete All from Dropbox")
    print("=" * 60)
    print()
    print("⚠️  WARNING: This will delete ALL files and folders in:")
    print(f"   /{settings.dropbox.app_folder}/data/candles/")
    print()
    print("Deleting all files and folders...")
    print()
    
    candles_path = f"/{settings.dropbox.app_folder}/data/candles"
    
    try:
        # Check if path exists
        try:
            sync._dbx.files_get_metadata(candles_path)
        except dropbox.exceptions.ApiError:
            print(f"✅ Path {candles_path} doesn't exist (already clean)")
            return
        
        # Delete everything recursively
        delete_recursive(sync._dbx, candles_path)
        
        # Try to delete the candles folder itself (might fail if not empty, that's ok)
        try:
            sync._dbx.files_delete_v2(candles_path)
            print(f"   🗑️  Deleted folder: {candles_path}")
        except:
            pass
        
        print()
        print("=" * 60)
        print("✅ Cleanup Complete!")
        print("=" * 60)
        print()
        print("All files and folders have been deleted.")
        print("You can now run the download script to start fresh.")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

