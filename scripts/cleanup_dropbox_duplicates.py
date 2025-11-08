#!/usr/bin/env python3
"""
Clean up duplicate files in Dropbox.

Removes files from root folder if they exist in coin folders.
Ensures all files are in coin folders only.

Usage:
    python scripts/cleanup_dropbox_duplicates.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.cloud.training.config.settings import EngineSettings
from src.cloud.training.integrations.dropbox_sync import DropboxSync
import dropbox  # type: ignore[reportMissingImports]
import os

os.environ['HURACAN_ENV'] = 'local'

def cleanup_duplicates():
    """Remove duplicate files from Dropbox root folder."""
    settings = EngineSettings.load()
    sync = DropboxSync(
        access_token=settings.dropbox.access_token,
        app_folder=settings.dropbox.app_folder,
        enabled=True,
        create_dated_folder=False,
    )
    
    print("=" * 60)
    print("🧹 Cleaning Up Dropbox Duplicates")
    print("=" * 60)
    print()
    
    try:
        result = sync._dbx.files_list_folder('/Runpodhuracan/data/candles')
        
        root_files = []
        coin_folders = {}
        
        # Collect files
        for entry in result.entries:
            if isinstance(entry, dropbox.files.FileMetadata):
                root_files.append((entry.name, entry.path_lower))
            elif isinstance(entry, dropbox.files.FolderMetadata):
                coin = entry.name
                try:
                    coin_result = sync._dbx.files_list_folder(entry.path_lower)
                    files = [
                        (e.name, e.path_lower) 
                        for e in coin_result.entries 
                        if isinstance(e, dropbox.files.FileMetadata)
                    ]
                    coin_folders[coin] = files
                except:
                    pass
        
        print(f"Found {len(root_files)} files in root folder")
        print(f"Found {len(coin_folders)} coin folders")
        print()
        
        # Find duplicates (same filename in root and coin folder)
        duplicates_to_remove = []
        coin_file_names = set()
        for coin, files in coin_folders.items():
            for filename, path in files:
                coin_file_names.add(filename)
        
        for filename, path in root_files:
            if filename in coin_file_names:
                duplicates_to_remove.append((filename, path))
        
        if not duplicates_to_remove:
            print("✅ No duplicates found - all files are in coin folders")
            return
        
        print(f"⚠️  Found {len(duplicates_to_remove)} duplicate files in root folder")
        print("   These will be removed (keeping coin folder versions)")
        print()
        
        # Remove duplicates
        removed_count = 0
        failed_count = 0
        
        for filename, path in duplicates_to_remove:
            try:
                sync._dbx.files_delete_v2(path)
                removed_count += 1
                print(f"   ✅ Removed {filename}")
            except Exception as e:
                failed_count += 1
                print(f"   ❌ Failed to remove {filename}: {e}")
        
        print()
        print("=" * 60)
        print("📊 Cleanup Summary")
        print("=" * 60)
        print(f"   ✅ Removed: {removed_count}/{len(duplicates_to_remove)}")
        print(f"   ❌ Failed: {failed_count}/{len(duplicates_to_remove)}")
        print("=" * 60)
        print()
        
        if removed_count > 0:
            print("✅ Cleanup complete! All files are now in coin folders only.")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    cleanup_duplicates()

