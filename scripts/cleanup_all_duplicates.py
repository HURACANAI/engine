#!/usr/bin/env python3
"""
Comprehensive duplicate folder cleanup for Dropbox.

This script will:
1. List ALL folders found in Dropbox
2. Find duplicates by name (case-insensitive, ignoring spaces)
3. Merge contents and remove duplicates
4. Handle edge cases like trailing spaces, case differences, etc.

Usage:
    python scripts/cleanup_all_duplicates.py [--dry-run]
"""

import sys
import argparse
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

def normalize_name(name):
    """Normalize folder name for comparison (case-insensitive, strip spaces)."""
    return name.strip().upper()

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

def cleanup_duplicates(dry_run=False):
    """Find and merge duplicate folders."""
    settings = EngineSettings.load()
    sync = DropboxSync(
        access_token=settings.dropbox.access_token,
        app_folder=settings.dropbox.app_folder,
        enabled=True,
        create_dated_folder=False,
    )
    
    print("=" * 60)
    print("🧹 Comprehensive Duplicate Folder Cleanup")
    print("=" * 60)
    if dry_run:
        print("🔍 DRY RUN MODE - No changes will be made")
    print()
    
    try:
        # Find all folders recursively
        all_folders = find_all_folders(sync._dbx, '/Runpodhuracan')
        
        print(f"📁 Found {len(all_folders)} total folders")
        print()
        
        # Group by normalized name
        folder_groups = defaultdict(list)
        for folder in all_folders:
            normalized = normalize_name(folder.name)
            folder_groups[normalized].append(folder)
        
        # Find duplicates
        duplicates = {name: folders for name, folders in folder_groups.items() if len(folders) > 1}
        
        # Filter to only coin folders (short names, likely in candles path)
        coin_duplicates = {}
        for name, folders in duplicates.items():
            # Check if this looks like a coin folder (3-5 chars, uppercase)
            if len(name) >= 2 and len(name) <= 5 and name.isalpha():
                coin_folders = [f for f in folders if '/candles/' in f.path_lower.lower()]
                if len(coin_folders) > 1:
                    coin_duplicates[name] = coin_folders
        
        if not coin_duplicates:
            print("✅ No duplicate coin folders found")
            print()
            print("All folders found:")
            for folder in sorted(all_folders, key=lambda x: normalize_name(x.name)):
                print(f"  - {folder.name} ({folder.path_lower})")
            return
        
        print(f"⚠️  Found {len(coin_duplicates)} coins with duplicate folders:")
        print()
        
        merged_count = 0
        removed_count = 0
        
        for coin_name, folders in sorted(coin_duplicates.items()):
            print(f"📁 {coin_name}: {len(folders)} folders")
            
            # Get files from each folder
            folder_contents = {}
            for folder in folders:
                files = get_folder_files(sync._dbx, folder.path_lower)
                folder_contents[folder.path_lower] = {
                    'folder': folder,
                    'files': files,
                    'file_count': len(files),
                    'total_size': sum(f.size for f in files)
                }
                print(f"   - {folder.path_lower}: {len(files)} files ({sum(f.size for f in files) / 1024 / 1024:.2f} MB)")
            
            # Choose the folder to keep (one with most files, or largest)
            keep_folder = max(folder_contents.items(), key=lambda x: (x[1]['file_count'], x[1]['total_size']))
            keep_path = keep_folder[0]
            keep_files = {f.name: f for f in keep_folder[1]['files']}
            
            print(f"   ✅ Keeping: {keep_path} ({len(keep_files)} files)")
            print()
            
            if dry_run:
                print("   [DRY RUN] Would merge and remove duplicates")
                print()
                continue
            
            # Merge files from other folders
            for folder_path, contents in folder_contents.items():
                if folder_path == keep_path:
                    continue
                
                folder = contents['folder']
                files = contents['files']
                
                print(f"   📦 Merging {len(files)} files from {folder_path}...")
                
                moved_count = 0
                skipped_count = 0
                deleted_count = 0
                
                for file_entry in files:
                    source_path = file_entry.path_lower
                    dest_path = f"{keep_path}/{file_entry.name}"
                    
                    # Check if file already exists in destination
                    if file_entry.name in keep_files:
                        existing_file = keep_files[file_entry.name]
                        # Same size = duplicate, delete it
                        if existing_file.size == file_entry.size:
                            print(f"      ⏭️  Skipping {file_entry.name} (duplicate)")
                            skipped_count += 1
                            try:
                                sync._dbx.files_delete_v2(source_path)
                                deleted_count += 1
                            except Exception as e:
                                print(f"      ⚠️  Could not delete {source_path}: {e}")
                            continue
                        else:
                            # Different sizes, rename the new one
                            base_name = Path(file_entry.name).stem
                            ext = Path(file_entry.name).suffix
                            counter = 1
                            while f"{base_name}_dup{counter}{ext}" in keep_files:
                                counter += 1
                            dest_path = f"{keep_path}/{base_name}_dup{counter}{ext}"
                    
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
                        merged_count += 1
                    else:
                        print(f"   ⚠️  Folder still has {len(remaining_files)} files, not removing")
                except Exception as e:
                    print(f"   ⚠️  Could not remove folder {folder_path}: {e}")
                
                print()
        
        if not dry_run:
            print("=" * 60)
            print("📊 Cleanup Summary")
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
    parser = argparse.ArgumentParser(description='Cleanup duplicate folders in Dropbox')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done without making changes')
    args = parser.parse_args()
    
    cleanup_duplicates(dry_run=args.dry_run)

