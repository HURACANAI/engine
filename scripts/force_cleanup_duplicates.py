#!/usr/bin/env python3
"""
Force cleanup of duplicate folders - handles edge cases.

This script will:
1. List ALL folders in Dropbox (including potential duplicates)
2. Manually check and remove duplicates
3. Handle case sensitivity, spaces, and other edge cases

Usage:
    python scripts/force_cleanup_duplicates.py
"""

import sys
from pathlib import Path
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.cloud.training.config.settings import EngineSettings
from src.cloud.training.integrations.dropbox_sync import DropboxSync
import dropbox
import os

os.environ['HURACAN_ENV'] = 'local'

def get_all_folders(dbx, path='/Runpodhuracan'):
    """Get all folders recursively."""
    folders = []
    try:
        result = dbx.files_list_folder(path)
        for entry in result.entries:
            if isinstance(entry, dropbox.files.FolderMetadata):
                folders.append(entry)
                subfolders = get_all_folders(dbx, entry.path_lower)
                folders.extend(subfolders)
    except:
        pass
    return folders

def get_folder_files(dbx, folder_path):
    """Get all files in a folder."""
    files = []
    try:
        result = dbx.files_list_folder(folder_path)
        for entry in result.entries:
            if isinstance(entry, dropbox.files.FileMetadata):
                files.append(entry)
    except:
        pass
    return files

def main():
    settings = EngineSettings.load()
    sync = DropboxSync(
        access_token=settings.dropbox.access_token,
        app_folder=settings.dropbox.app_folder,
        enabled=True,
        create_dated_folder=False,
    )
    
    print("=" * 60)
    print("🔍 Force Cleanup - Finding ALL Duplicates")
    print("=" * 60)
    print()
    
    # Get all folders
    all_folders = get_all_folders(sync._dbx, '/Runpodhuracan')
    
    # Group by name (case-insensitive) - but keep original names
    folder_groups = defaultdict(list)
    for folder in all_folders:
        # Only check coin folders (in candles path, short names)
        if '/candles/' in folder.path_lower.lower() and len(folder.name) <= 5:
            name_key = folder.name.upper()
            folder_groups[name_key].append(folder)
    
    # Find duplicates
    duplicates = {name: folders for name, folders in folder_groups.items() if len(folders) > 1}
    
    if not duplicates:
        print("✅ No duplicates found via API")
        print()
        print("This might be a Dropbox UI caching issue.")
        print("Try refreshing your Dropbox web interface (F5 or Cmd+R)")
        print()
        print("If duplicates still appear, they might be:")
        print("  1. In a different Dropbox location")
        print("  2. In a shared folder")
        print("  3. Created by a different app/script")
        return
    
    print(f"⚠️  Found {len(duplicates)} duplicate groups:")
    print()
    
    removed_count = 0
    
    for coin_name, folders in sorted(duplicates.items()):
        print(f"📁 {coin_name}: {len(folders)} folders")
        
        # Analyze each folder
        folder_info = []
        for folder in folders:
            files = get_folder_files(sync._dbx, folder.path_lower)
            folder_info.append({
                'folder': folder,
                'files': files,
                'file_count': len(files),
                'total_size': sum(f.size for f in files)
            })
            print(f"   - {folder.path_lower}: {len(files)} files")
        
        # Keep the folder with most files/largest size
        keep_info = max(folder_info, key=lambda x: (x['file_count'], x['total_size']))
        keep_path = keep_info['folder'].path_lower
        keep_files = {f.name: f for f in keep_info['files']}
        
        print(f"   ✅ Keeping: {keep_path}")
        print()
        
        # Remove other folders (after moving files if needed)
        for info in folder_info:
            if info['folder'].path_lower == keep_path:
                continue
            
            folder_path = info['folder'].path_lower
            files = info['files']
            
            if files:
                print(f"   📦 Moving {len(files)} files from {folder_path}...")
                for file_entry in files:
                    source = file_entry.path_lower
                    dest = f"{keep_path}/{file_entry.name}"
                    
                    # Check if file exists in destination
                    if file_entry.name in keep_files:
                        # Same size = duplicate, just delete
                        if keep_files[file_entry.name].size == file_entry.size:
                            try:
                                sync._dbx.files_delete_v2(source)
                                print(f"      🗑️  Deleted duplicate {file_entry.name}")
                            except Exception as e:
                                print(f"      ⚠️  Could not delete {file_entry.name}: {e}")
                            continue
                    
                    # Move file
                    try:
                        sync._dbx.files_move_v2(source, dest)
                        print(f"      ✅ Moved {file_entry.name}")
                    except Exception as e:
                        print(f"      ❌ Failed to move {file_entry.name}: {e}")
            
            # Delete folder
            try:
                remaining = get_folder_files(sync._dbx, folder_path)
                if len(remaining) == 0:
                    sync._dbx.files_delete_v2(folder_path)
                    removed_count += 1
                    print(f"   🗑️  Removed folder: {folder_path}")
                else:
                    print(f"   ⚠️  Folder still has {len(remaining)} files")
            except Exception as e:
                print(f"   ⚠️  Could not remove {folder_path}: {e}")
        
        print()
    
    print("=" * 60)
    print("📊 Summary")
    print("=" * 60)
    print(f"   🗑️  Removed: {removed_count} duplicate folders")
    print("=" * 60)
    
    if removed_count > 0:
        print()
        print("✅ Cleanup complete! Please refresh your Dropbox view.")
    else:
        print()
        print("⚠️  No duplicates found to remove.")
        print("If you still see duplicates in Dropbox:")
        print("  1. Refresh your browser/app (F5 or Cmd+R)")
        print("  2. Check if they're in a different location")
        print("  3. Check if they're in a shared folder")

if __name__ == "__main__":
    main()

