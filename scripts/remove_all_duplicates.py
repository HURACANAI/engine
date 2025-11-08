#!/usr/bin/env python3
"""
Aggressively remove duplicate folders - handles any edge cases.

This script will:
1. List ALL folders in Dropbox
2. Find duplicates by exact name match (case-insensitive)
3. Keep the folder with most files, delete others
4. Handle files in duplicate folders

Usage:
    python scripts/remove_all_duplicates.py
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
    except Exception as e:
        print(f"Error scanning {path}: {e}")
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
    print("🧹 Aggressive Duplicate Removal")
    print("=" * 60)
    print()
    
    # Get ALL folders
    print("Scanning all folders...")
    all_folders = get_all_folders(sync._dbx, '/Runpodhuracan')
    print(f"Found {len(all_folders)} total folders")
    print()
    
    # Group by normalized name
    folder_groups = defaultdict(list)
    for folder in all_folders:
        # Normalize: uppercase, strip spaces
        normalized = folder.name.strip().upper()
        folder_groups[normalized].append(folder)
    
    # Find duplicates
    duplicates = {name: folders for name, folders in folder_groups.items() if len(folders) > 1}
    
    # Filter to coin folders (short names, likely in candles)
    coin_duplicates = {}
    for name, folders in duplicates.items():
        # Coin folders: 2-5 chars, letters only, in candles path
        if 2 <= len(name) <= 5 and name.isalpha():
            coin_folders = [f for f in folders if '/candles/' in f.path_lower.lower()]
            if len(coin_folders) > 1:
                coin_duplicates[name] = coin_folders
    
    if not coin_duplicates:
        print("✅ No duplicate coin folders found")
        print()
        print("If you still see duplicates, they might be:")
        print("  - In a different location")
        print("  - Created by a different process")
        print("  - UI caching (try refreshing)")
        return
    
    print(f"⚠️  Found {len(coin_duplicates)} coins with duplicate folders:")
    print()
    
    removed_count = 0
    merged_count = 0
    
    for coin_name, folders in sorted(coin_duplicates.items()):
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
            print(f"   - {folder.path_lower}: {len(files)} files ({sum(f.size for f in files) / 1024 / 1024:.2f} MB)")
        
        # Keep folder with most files/largest size
        keep_info = max(folder_info, key=lambda x: (x['file_count'], x['total_size']))
        keep_path = keep_info['folder'].path_lower
        keep_files = {f.name: f for f in keep_info['files']}
        
        print(f"   ✅ Keeping: {keep_path}")
        print()
        
        # Process other folders
        for info in folder_info:
            if info['folder'].path_lower == keep_path:
                continue
            
            folder_path = info['folder'].path_lower
            files = info['files']
            
            if files:
                print(f"   📦 Processing {len(files)} files from {folder_path}...")
                moved = 0
                deleted = 0
                
                for file_entry in files:
                    source = file_entry.path_lower
                    dest = f"{keep_path}/{file_entry.name}"
                    
                    # Check if file exists in destination
                    if file_entry.name in keep_files:
                        existing = keep_files[file_entry.name]
                        # Same size = duplicate, delete source
                        if existing.size == file_entry.size:
                            try:
                                sync._dbx.files_delete_v2(source)
                                deleted += 1
                                print(f"      🗑️  Deleted duplicate {file_entry.name}")
                            except Exception as e:
                                print(f"      ⚠️  Could not delete {file_entry.name}: {e}")
                            continue
                        else:
                            # Different size, rename
                            base = Path(file_entry.name).stem
                            ext = Path(file_entry.name).suffix
                            counter = 1
                            while f"{base}_dup{counter}{ext}" in keep_files:
                                counter += 1
                            dest = f"{keep_path}/{base}_dup{counter}{ext}"
                    
                    # Move file
                    try:
                        sync._dbx.files_move_v2(source, dest)
                        moved += 1
                        print(f"      ✅ Moved {file_entry.name}")
                    except Exception as e:
                        print(f"      ❌ Failed to move {file_entry.name}: {e}")
                
                print(f"      Summary: {moved} moved, {deleted} deleted")
            
            # Delete folder
            try:
                remaining = get_folder_files(sync._dbx, folder_path)
                if len(remaining) == 0:
                    sync._dbx.files_delete_v2(folder_path)
                    removed_count += 1
                    print(f"   🗑️  Removed folder: {folder_path}")
                    merged_count += 1
                else:
                    print(f"   ⚠️  Folder still has {len(remaining)} files, not removing")
            except Exception as e:
                print(f"   ⚠️  Could not remove {folder_path}: {e}")
        
        print()
    
    print("=" * 60)
    print("📊 Summary")
    print("=" * 60)
    print(f"   ✅ Merged: {merged_count} folders")
    print(f"   🗑️  Removed: {removed_count} duplicate folders")
    print("=" * 60)
    
    if removed_count > 0:
        print()
        print("✅ Cleanup complete! Refresh your Dropbox view to see changes.")

if __name__ == "__main__":
    main()

