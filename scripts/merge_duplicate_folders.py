#!/usr/bin/env python3
"""
Merge duplicate coin folders in Dropbox.

Finds folders with the same name (case-insensitive) and merges their contents,
keeping the folder with the most files or the first one found.

Usage:
    python scripts/merge_duplicate_folders.py
"""

import sys
from pathlib import Path
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.cloud.training.config.settings import EngineSettings
from src.cloud.training.integrations.dropbox_sync import DropboxSync
import dropbox  # type: ignore[reportMissingImports]
import os

os.environ['HURACAN_ENV'] = 'local'

def find_all_folders(dbx, path='/Runpodhuracan'):
    """Recursively find all folders starting from path."""
    folders = []
    try:
        result = dbx.files_list_folder(path)
        for entry in result.entries:
            if isinstance(entry, dropbox.files.FolderMetadata):
                folders.append(entry)
                # Recursively check subfolders
                subfolders = find_all_folders(dbx, entry.path_lower)
                folders.extend(subfolders)
    except Exception as e:
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
    except Exception as e:
        pass
    return files

def merge_duplicate_folders():
    """Find and merge duplicate coin folders."""
    settings = EngineSettings.load()
    sync = DropboxSync(
        access_token=settings.dropbox.access_token,
        app_folder=settings.dropbox.app_folder,
        enabled=True,
        create_dated_folder=False,
    )
    
    print("=" * 60)
    print("🔍 Finding Duplicate Folders")
    print("=" * 60)
    print()
    
    try:
        # Find all folders recursively
        all_folders = find_all_folders(sync._dbx, '/Runpodhuracan')
        
        # Group by case-insensitive name
        folder_groups = defaultdict(list)
        for folder in all_folders:
            name_lower = folder.name.upper()  # Use uppercase for consistency
            folder_groups[name_lower].append(folder)
        
        # Find duplicates (only check coin folders in candles directory)
        duplicates = {}
        for name, folders in folder_groups.items():
            # Only process folders that are coin folders (short names, in candles path)
            coin_folders = [f for f in folders if '/candles/' in f.path_lower and len(f.name) <= 5]
            if len(coin_folders) > 1:
                duplicates[name] = coin_folders
        
        if not duplicates:
            print("✅ No duplicate coin folders found")
            return
        
        print(f"⚠️  Found {len(duplicates)} coins with duplicate folders:")
        print()
        
        merged_count = 0
        removed_count = 0
        
        for coin_name, folders in sorted(duplicates.items()):
            print(f"📁 {coin_name}: {len(folders)} folders")
            
            # Get files from each folder
            folder_contents = {}
            for folder in folders:
                files = get_folder_files(sync._dbx, folder.path_lower)
                folder_contents[folder.path_lower] = {
                    'folder': folder,
                    'files': files,
                    'file_count': len(files)
                }
                print(f"   - {folder.path_lower}: {len(files)} files")
            
            # Choose the folder to keep (one with most files, or first one)
            keep_folder = max(folder_contents.items(), key=lambda x: x[1]['file_count'])
            keep_path = keep_folder[0]
            keep_files = {f.name: f for f in keep_folder[1]['files']}
            
            print(f"   ✅ Keeping: {keep_path} ({len(keep_files)} files)")
            print()
            
            # Merge files from other folders
            for folder_path, contents in folder_contents.items():
                if folder_path == keep_path:
                    continue
                
                folder = contents['folder']
                files = contents['files']
                
                print(f"   📦 Merging {len(files)} files from {folder_path}...")
                
                moved_count = 0
                skipped_count = 0
                
                for file_entry in files:
                    source_path = file_entry.path_lower
                    dest_path = f"{keep_path}/{file_entry.name}"
                    
                    # Check if file already exists in destination
                    if file_entry.name in keep_files:
                        # File exists, check if same size
                        existing_file = keep_files[file_entry.name]
                        if existing_file.size == file_entry.size:
                            print(f"      ⏭️  Skipping {file_entry.name} (already exists)")
                            skipped_count += 1
                            # Delete duplicate
                            try:
                                sync._dbx.files_delete_v2(source_path)
                                print(f"      🗑️  Deleted duplicate {file_entry.name}")
                            except Exception as e:
                                print(f"      ⚠️  Could not delete {source_path}: {e}")
                            continue
                        else:
                            # Different sizes, rename the new one
                            base_name = Path(file_entry.name).stem
                            ext = Path(file_entry.name).suffix
                            dest_path = f"{keep_path}/{base_name}_duplicate{ext}"
                    
                    # Move file to keep folder
                    try:
                        sync._dbx.files_move_v2(source_path, dest_path)
                        moved_count += 1
                        print(f"      ✅ Moved {file_entry.name}")
                    except Exception as e:
                        print(f"      ❌ Failed to move {file_entry.name}: {e}")
                
                # Delete empty folder
                try:
                    remaining_files = get_folder_files(sync._dbx, folder_path)
                    if len(remaining_files) == 0:
                        sync._dbx.files_delete_v2(folder_path)
                        removed_count += 1
                        print(f"   🗑️  Removed empty folder: {folder_path}")
                    else:
                        print(f"   ⚠️  Folder still has {len(remaining_files)} files, not removing")
                except Exception as e:
                    print(f"   ⚠️  Could not remove folder {folder_path}: {e}")
                
                print()
        
        print("=" * 60)
        print("📊 Merge Summary")
        print("=" * 60)
        print(f"   ✅ Merged: {merged_count} folders")
        print(f"   🗑️  Removed: {removed_count} duplicate folders")
        print("=" * 60)
        print()
        
        if removed_count > 0:
            print("✅ Cleanup complete! Duplicate folders merged and removed.")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    merge_duplicate_folders()

