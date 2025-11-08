#!/usr/bin/env python3
"""
Remove duplicate files from Dropbox.

Removes files with futures suffix (e.g., BTC-USDT:USDT) if normal version exists (BTC-USDT).

Usage:
    python scripts/remove_duplicate_files.py
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

def remove_duplicate_futures_files():
    """Remove files with futures suffix if normal version exists."""
    settings = EngineSettings.load()
    sync = DropboxSync(
        access_token=settings.dropbox.access_token,
        app_folder=settings.dropbox.app_folder,
        enabled=True,
        create_dated_folder=False,
    )
    
    print("=" * 60)
    print("🧹 Removing Duplicate Futures Files")
    print("=" * 60)
    print()
    
    try:
        result = sync._dbx.files_list_folder('/Runpodhuracan/data/candles')
        
        files_to_remove = []
        
        # Check each coin folder
        for entry in result.entries:
            if isinstance(entry, dropbox.files.FolderMetadata):
                coin = entry.name
                try:
                    coin_result = sync._dbx.files_list_folder(entry.path_lower)
                    files = {}
                    for file_entry in coin_result.entries:
                        if isinstance(file_entry, dropbox.files.FileMetadata):
                            filename = file_entry.name
                            files[filename] = file_entry.path_lower
                    
                    # Find duplicates (futures vs normal)
                    for filename, path in files.items():
                        # Check if this is a futures file (has :USDT or :USDC)
                        if ':USDT' in filename or ':USDC' in filename:
                            # Create normal version name
                            normal_name = filename.replace(':USDT', '').replace(':USDC', '')
                            # Check if normal version exists
                            if normal_name in files:
                                files_to_remove.append((filename, path, normal_name))
                
                except Exception as e:
                    print(f"   ⚠️  Error checking {coin}: {e}")
        
        if not files_to_remove:
            print("✅ No duplicate futures files found")
            return
        
        print(f"⚠️  Found {len(files_to_remove)} duplicate futures files to remove:")
        for futures_file, path, normal_file in files_to_remove[:10]:
            print(f"   {futures_file} (keeping {normal_file})")
        if len(files_to_remove) > 10:
            print(f"   ... and {len(files_to_remove) - 10} more")
        print()
        
        # Remove duplicates
        removed_count = 0
        failed_count = 0
        
        for futures_file, path, normal_file in files_to_remove:
            try:
                sync._dbx.files_delete_v2(path)
                removed_count += 1
                print(f"   ✅ Removed {futures_file}")
            except Exception as e:
                failed_count += 1
                print(f"   ❌ Failed to remove {futures_file}: {e}")
        
        print()
        print("=" * 60)
        print("📊 Cleanup Summary")
        print("=" * 60)
        print(f"   ✅ Removed: {removed_count}/{len(files_to_remove)}")
        print(f"   ❌ Failed: {failed_count}/{len(files_to_remove)}")
        print("=" * 60)
        print()
        
        if removed_count > 0:
            print("✅ Cleanup complete! Only normal files remain.")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    remove_duplicate_futures_files()

