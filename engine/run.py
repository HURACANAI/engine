"""
Engine - Simple Training Module

Trains models nightly from historical data.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.shared.config_loader import load_config
from src.shared.simple_logger import logger
from src.cloud.training.pipelines.daily_retrain import run_daily_retrain


def main():
    """Run Engine training."""
    logger.info("Engine starting", version="2.0")
    
    # Load configuration
    config = load_config()
    engine_config = config.get("engine", {})
    
    logger.info("Configuration loaded", 
                lookback_days=engine_config.get("lookback_days"),
                symbols=config.get("general", {}).get("symbols", []))
    
    try:
        # Run training
        run_daily_retrain()
        
        logger.success("Engine finished successfully")
        return 0
    except Exception as e:
        logger.fail(f"Engine failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

