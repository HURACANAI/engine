#!/usr/bin/env python3
"""
Manual duplicate cleanup - lists all folders and allows selective deletion.

This script will show you ALL folders and let you manually identify and remove duplicates.

Usage:
    python scripts/manual_cleanup_duplicates.py
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
    print("🔍 Manual Duplicate Cleanup")
    print("=" * 60)
    print()
    
    # Get all folders
    all_folders = get_all_folders(sync._dbx, '/Runpodhuracan')
    
    # Group by name (case-insensitive)
    folder_groups = defaultdict(list)
    for folder in all_folders:
        name_upper = folder.name.upper().strip()
        folder_groups[name_upper].append(folder)
    
    # Show all coin folders
    print("📁 All Coin Folders Found:")
    print()
    
    coin_folders = {}
    for folder in all_folders:
        if '/candles/' in folder.path_lower.lower() and 2 <= len(folder.name) <= 5 and folder.name.isalpha():
            name_upper = folder.name.upper()
            if name_upper not in coin_folders:
                coin_folders[name_upper] = []
            coin_folders[name_upper].append(folder)
    
    duplicates_found = False
    for name, folders in sorted(coin_folders.items()):
        if len(folders) > 1:
            duplicates_found = True
            print(f"⚠️  {name}: {len(folders)} folders (DUPLICATE!)")
            for i, folder in enumerate(folders, 1):
                files = get_folder_files(sync._dbx, folder.path_lower)
                print(f"   {i}. {folder.path_lower} ({len(files)} files)")
            print()
        else:
            folder = folders[0]
            files = get_folder_files(sync._dbx, folder.path_lower)
            print(f"✅ {name}: {folder.path_lower} ({len(files)} files)")
    
    if not duplicates_found:
        print("✅ No duplicates found!")
        print()
        print("If you still see duplicates in Dropbox:")
        print("  1. Refresh your Dropbox view (F5 or Cmd+R)")
        print("  2. Check if they're in a different location")
        print("  3. They might be in a shared folder")
        return
    
    print()
    print("=" * 60)
    print("🧹 Cleaning up duplicates...")
    print("=" * 60)
    print()
    
    removed_count = 0
    
    for name, folders in sorted(coin_folders.items()):
        if len(folders) <= 1:
            continue
        
        print(f"📁 {name}: {len(folders)} folders")
        
        # Get file info for each folder
        folder_info = []
        for folder in folders:
            files = get_folder_files(sync._dbx, folder.path_lower)
            folder_info.append({
                'folder': folder,
                'files': files,
                'file_count': len(files),
                'total_size': sum(f.size for f in files)
            })
        
        # Keep the folder with most files
        keep_info = max(folder_info, key=lambda x: (x['file_count'], x['total_size']))
        keep_path = keep_info['folder'].path_lower
        keep_files = {f.name: f for f in keep_info['files']}
        
        print(f"   ✅ Keeping: {keep_path} ({len(keep_files)} files)")
        
        # Process other folders
        for info in folder_info:
            if info['folder'].path_lower == keep_path:
                continue
            
            folder_path = info['folder'].path_lower
            files = info['files']
            
            print(f"   📦 Processing: {folder_path} ({len(files)} files)")
            
            # Move or delete files
            for file_entry in files:
                source = file_entry.path_lower
                dest = f"{keep_path}/{file_entry.name}"
                
                if file_entry.name in keep_files:
                    # Duplicate file, delete it
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
            
            # Delete empty folder
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
        print("✅ Cleanup complete! Refresh your Dropbox view.")

if __name__ == "__main__":
    main()

