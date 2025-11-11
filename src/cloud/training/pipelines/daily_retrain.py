"""
Daily Retrain Entry Point

Main entry point for daily retraining with hybrid scheduler.
This is the new simplified entry point that uses the scheduler.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.cloud.training.pipelines.daily_retrain_scheduler import main


def run_daily_retrain() -> int:
    """Run daily retrain - wrapper function for main().
    
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    return main()


if __name__ == "__main__":
    sys.exit(main())
