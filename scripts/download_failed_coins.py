#!/usr/bin/env python3
"""
Download the 4 failed coins with shorter time period.

These coins are new and don't have 150 days of history, so we'll try
with 30 days instead.

Usage:
    python scripts/download_failed_coins.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.simple_download_candles import download_and_upload

# The 4 coins that failed with 150 days
FAILED_COINS = [
    "GIGGLE/USDT",
    "XPL/USDT",
    "MMT/USDT",
    "ASTER/USDT",
]

def main():
    print("=" * 60)
    print("📥 Downloading Failed Coins (30 days)")
    print("=" * 60)
    print()
    print("These coins failed with 150 days because they're new.")
    print("Trying with 30 days of data instead...")
    print()
    
    download_and_upload(
        symbols=FAILED_COINS,
        days=30,  # Shorter period for new coins
        exchange_id="binance",
        timeframe="1m",
    )

if __name__ == "__main__":
    main()

