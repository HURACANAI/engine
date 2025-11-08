#!/usr/bin/env python3
"""
Check download status and upload all candles to Dropbox.

Usage:
    python scripts/check_and_upload_candles.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.upload_local_candles_to_dropbox import upload_local_candles

def main():
    """Check status and upload."""
    candles_dir = Path("data/candles")
    
    if not candles_dir.exists():
        print("❌ data/candles directory not found")
        return
    
    # Count files by coin
    coin_files = {}
    total_files = 0
    
    for coin_dir in candles_dir.iterdir():
        if coin_dir.is_dir():
            files = list(coin_dir.glob("*.parquet"))
            if files:
                coin_files[coin_dir.name] = len(files)
                total_files += len(files)
    
    print("=" * 60)
    print("📊 Current Download Status")
    print("=" * 60)
    print(f"Total files: {total_files}")
    print(f"Coins with data: {len(coin_files)}")
    print()
    
    if coin_files:
        print("Files per coin:")
        for coin, count in sorted(coin_files.items()):
            print(f"  {coin:8}: {count:2} files")
        print()
    
    # Upload to Dropbox
    print("=" * 60)
    print("📤 Uploading to Dropbox...")
    print("=" * 60)
    print()
    
    upload_local_candles()

if __name__ == "__main__":
    main()

