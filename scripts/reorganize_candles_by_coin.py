#!/usr/bin/env python3
"""
Reorganize candle files into coin-specific folders.

Moves files from data/candles/*.parquet to data/candles/{COIN}/*.parquet
"""

import sys
from pathlib import Path
from shutil import move

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def reorganize_candles():
    """Reorganize existing candle files into coin folders."""
    candles_dir = Path("data/candles")
    
    if not candles_dir.exists():
        print("❌ data/candles directory not found")
        return
    
    # Find all parquet files
    parquet_files = list(candles_dir.glob("*.parquet"))
    
    if not parquet_files:
        print("⚠️  No parquet files found in data/candles/")
        return
    
    print(f"📦 Found {len(parquet_files)} files to reorganize\n")
    
    moved_count = 0
    skipped_count = 0
    
    for file_path in parquet_files:
        try:
            # Extract coin from filename (e.g., "BTC-USDT_1m_..." -> "BTC")
            filename = file_path.name
            coin = filename.split("-")[0]
            
            # Skip if already in a coin folder (shouldn't happen, but check)
            if file_path.parent.name == coin:
                skipped_count += 1
                continue
            
            # Create coin folder
            coin_dir = candles_dir / coin
            coin_dir.mkdir(parents=True, exist_ok=True)
            
            # Move file
            dest_path = coin_dir / filename
            if dest_path.exists():
                print(f"⚠️  Skipping {filename} (already exists in {coin}/)")
                skipped_count += 1
                continue
            
            move(str(file_path), str(dest_path))
            moved_count += 1
            print(f"✅ Moved {filename} → {coin}/{filename}")
            
        except Exception as e:
            print(f"❌ Error moving {file_path.name}: {e}")
    
    print(f"\n{'='*60}")
    print(f"📊 Summary")
    print(f"{'='*60}")
    print(f"   ✅ Moved: {moved_count}")
    print(f"   ⏭️  Skipped: {skipped_count}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    reorganize_candles()

