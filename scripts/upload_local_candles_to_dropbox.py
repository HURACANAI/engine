#!/usr/bin/env python3
"""
Upload existing local candle data to Dropbox.

This script uploads all parquet files from data/candles/ to Dropbox.
Useful when you've downloaded data locally and want to upload it later.

Usage:
    python scripts/upload_local_candles_to_dropbox.py
    python scripts/upload_local_candles_to_dropbox.py --dropbox-token YOUR_TOKEN
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.cloud.training.config.settings import EngineSettings
from src.cloud.training.integrations.dropbox_sync import DropboxSync
import structlog  # type: ignore[reportMissingImports]

logger = structlog.get_logger(__name__)


def upload_local_candles(
    dropbox_token: Optional[str] = None,
    app_folder: str = "Runpodhuracan",
    data_dir: str = "data/candles",
) -> None:
    """Upload all local candle files to Dropbox.
    
    Args:
        dropbox_token: Dropbox access token (if None, uses settings)
        app_folder: Dropbox app folder name
        data_dir: Local directory containing candle files
    """
    # Load settings
    settings = EngineSettings.load(environment=os.getenv("HURACAN_ENV", "local"))
    
    # Initialize Dropbox
    dropbox_access_token = dropbox_token or settings.dropbox.access_token
    if not dropbox_access_token:
        logger.error("dropbox_token_missing", message="Dropbox token is required")
        print("❌ Error: Dropbox token is required")
        print("   Set DROPBOX_ACCESS_TOKEN environment variable or configure in settings")
        print("   Or use: python scripts/upload_local_candles_to_dropbox.py --dropbox-token YOUR_TOKEN")
        sys.exit(1)
    
    try:
        dropbox_sync = DropboxSync(
            access_token=dropbox_access_token,
            app_folder=app_folder,
            enabled=True,
            create_dated_folder=False,
        )
        print("✅ Dropbox connection initialized\n")
    except Exception as e:
        logger.error("dropbox_init_failed", error=str(e))
        print(f"❌ Error: Failed to initialize Dropbox: {e}")
        print("   The token may be expired. Generate a new token at:")
        print("   https://www.dropbox.com/developers/apps")
        sys.exit(1)
    
    # Find all parquet files (including in coin subdirectories)
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"❌ Error: Directory not found: {data_path}")
        sys.exit(1)
    
    # Support both flat structure and coin folder structure
    parquet_files = list(data_path.glob("*.parquet")) + list(data_path.glob("*/*.parquet"))
    
    if not parquet_files:
        print(f"⚠️  No parquet files found in {data_path}")
        print("   Run download script first: python scripts/download_and_upload_candles.py --all-symbols")
        sys.exit(0)
    
    print(f"📦 Found {len(parquet_files)} candle file(s) to upload")
    print(f"   Source: {data_path}")
    print(f"   Destination: /{app_folder}/data/candles/\n")
    
    uploaded_count = 0
    failed_count = 0
    
    for parquet_file in sorted(parquet_files):
        try:
            # Show relative path if in subdirectory
            display_name = str(parquet_file.relative_to(data_path)) if parquet_file.parent != data_path else parquet_file.name
            print(f"📤 Uploading {display_name}...")
            
            # Get relative path (preserves coin folder structure)
            rel_path = parquet_file.relative_to(data_path)
            # Ensure path uses forward slashes for Dropbox
            remote_path = f"/{app_folder}/data/candles/{rel_path.as_posix()}"
            
            # Check if file already exists in Dropbox (prevent duplicates)
            try:
                import dropbox  # type: ignore[reportMissingImports]
                existing = dropbox_sync._dbx.files_get_metadata(remote_path)
                if isinstance(existing, dropbox.files.FileMetadata):  # type: ignore[misc]
                    print(f"   ⏭️  Already exists in Dropbox (skipping duplicate)")
                    uploaded_count += 1  # Count as uploaded since it's already there
                    continue
            except:
                # File doesn't exist, proceed with upload
                pass
            
            # Upload to shared location
            success = dropbox_sync.upload_file(
                local_path=parquet_file,
                remote_path=remote_path,
                use_dated_folder=False,  # Use shared location
                overwrite=True,
            )
            
            if success:
                uploaded_count += 1
                file_size = parquet_file.stat().st_size
                print(f"   ✅ Uploaded ({file_size / 1024 / 1024:.2f} MB) → {remote_path}")
            else:
                failed_count += 1
                print(f"   ❌ Failed to upload")
                
        except Exception as e:
            failed_count += 1
            print(f"   ❌ Error: {e}")
            logger.warning("upload_failed", file=str(parquet_file), error=str(e))
    
    # Summary
    print(f"\n{'='*60}")
    print(f"📊 Summary")
    print(f"{'='*60}")
    print(f"   ✅ Uploaded: {uploaded_count}/{len(parquet_files)}")
    print(f"   ❌ Failed: {failed_count}/{len(parquet_files)}")
    print(f"{'='*60}\n")
    
    if uploaded_count > 0:
        print(f"✅ Successfully uploaded {uploaded_count} file(s) to Dropbox")
        print(f"   Location: /{app_folder}/data/candles/")
        print(f"   RunPod engine will restore this data on startup\n")
    
    if failed_count > 0:
        print(f"⚠️  {failed_count} file(s) failed to upload")
        sys.exit(1)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Upload local candle data to Dropbox",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upload using token from settings
  python scripts/upload_local_candles_to_dropbox.py
  
  # Upload with custom token
  python scripts/upload_local_candles_to_dropbox.py --dropbox-token YOUR_TOKEN
  
  # Upload from custom directory
  python scripts/upload_local_candles_to_dropbox.py --data-dir custom/path/to/candles
        """,
    )
    
    parser.add_argument(
        "--dropbox-token",
        type=str,
        help="Dropbox access token (default: from settings or environment)",
    )
    parser.add_argument(
        "--app-folder",
        type=str,
        default="Runpodhuracan",
        help="Dropbox app folder name (default: Runpodhuracan)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/candles",
        help="Local directory containing candle files (default: data/candles)",
    )
    
    args = parser.parse_args()
    
    upload_local_candles(
        dropbox_token=args.dropbox_token,
        app_folder=args.app_folder,
        data_dir=args.data_dir,
    )


if __name__ == "__main__":
    main()

