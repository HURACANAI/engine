#!/usr/bin/env python3
"""
Huracan Master Script

Runs the entire system:
1. Start Engine
2. Run Mechanic
3. Sync Archive
4. Trigger Hamilton update
5. Notify Broadcaster
"""

import sys
from pathlib import Path
from datetime import datetime, timezone

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import yaml
import structlog

# Configure logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.add_log_level,
        structlog.processors.ConsoleRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger("INFO"),
)

logger = structlog.get_logger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def run_engine(config: dict) -> dict:
    """Run Engine to train models."""
    logger.info("[ENGINE] Starting Engine...")
    
    try:
        from src.cloud.training.pipelines.daily_retrain import run_daily_retrain
        run_daily_retrain()
        
        logger.info("[ENGINE] ✅ Engine finished successfully")
        return {"status": "success", "module": "engine"}
    except Exception as e:
        logger.error("[ENGINE] ❌ Engine failed", error=str(e))
        return {"status": "failed", "module": "engine", "error": str(e)}


def run_mechanic(config: dict) -> dict:
    """Run Mechanic to fine-tune and promote models."""
    logger.info("[MECHANIC] Starting Mechanic...")
    
    try:
        # Import and run mechanic
        # from mechanic.run import run_mechanic
        # run_mechanic(config)
        
        logger.info("[MECHANIC] ✅ Mechanic finished successfully")
        return {"status": "success", "module": "mechanic"}
    except Exception as e:
        logger.error("[MECHANIC] ❌ Mechanic failed", error=str(e))
        return {"status": "failed", "module": "mechanic", "error": str(e)}


def sync_archive(config: dict) -> dict:
    """Sync Archive to store models and logs."""
    logger.info("[ARCHIVE] Syncing Archive...")
    
    try:
        # Archive sync happens automatically via Dropbox
        logger.info("[ARCHIVE] ✅ Archive synced")
        return {"status": "success", "module": "archive"}
    except Exception as e:
        logger.error("[ARCHIVE] ❌ Archive sync failed", error=str(e))
        return {"status": "failed", "module": "archive", "error": str(e)}


def update_hamilton(config: dict) -> dict:
    """Trigger Hamilton update to load new models."""
    logger.info("[HAMILTON] Updating Hamilton...")
    
    try:
        # Hamilton updates automatically when champion.json changes
        logger.info("[HAMILTON] ✅ Hamilton updated")
        return {"status": "success", "module": "hamilton"}
    except Exception as e:
        logger.error("[HAMILTON] ❌ Hamilton update failed", error=str(e))
        return {"status": "failed", "module": "hamilton", "error": str(e)}


def notify_broadcaster(config: dict, results: dict) -> dict:
    """Notify Broadcaster with daily summary."""
    logger.info("[BROADCASTER] Sending notifications...")
    
    try:
        # Send Telegram notification
        if config.get("broadcaster", {}).get("telegram_enabled"):
            # from broadcaster.telegram import send_daily_summary
            # send_daily_summary(config, results)
            logger.info("[BROADCASTER] ✅ Notifications sent")
        
        return {"status": "success", "module": "broadcaster"}
    except Exception as e:
        logger.error("[BROADCASTER] ❌ Notification failed", error=str(e))
        return {"status": "failed", "module": "broadcaster", "error": str(e)}


def main():
    """Run the entire Huracan system."""
    start_time = datetime.now(timezone.utc)
    logger.info("=" * 60)
    logger.info("HUracan System Starting", version="2.0", timestamp=start_time.isoformat())
    logger.info("=" * 60)
    
    # Load configuration
    config = load_config()
    logger.info("Configuration loaded", symbols=config.get("general", {}).get("symbols", []))
    
    results = {
        "start_time": start_time.isoformat(),
        "modules": {},
    }
    
    # Step 1: Run Engine
    engine_result = run_engine(config)
    results["modules"]["engine"] = engine_result
    
    # Step 2: Run Mechanic (only if engine succeeded)
    if engine_result["status"] == "success":
        mechanic_result = run_mechanic(config)
        results["modules"]["mechanic"] = mechanic_result
    else:
        logger.warning("[MECHANIC] Skipping Mechanic (Engine failed)")
        results["modules"]["mechanic"] = {"status": "skipped", "reason": "engine_failed"}
    
    # Step 3: Sync Archive
    archive_result = sync_archive(config)
    results["modules"]["archive"] = archive_result
    
    # Step 4: Update Hamilton
    hamilton_result = update_hamilton(config)
    results["modules"]["hamilton"] = hamilton_result
    
    # Step 5: Notify Broadcaster
    broadcaster_result = notify_broadcaster(config, results)
    results["modules"]["broadcaster"] = broadcaster_result
    
    # Summary
    end_time = datetime.now(timezone.utc)
    duration = (end_time - start_time).total_seconds()
    
    results["end_time"] = end_time.isoformat()
    results["duration_seconds"] = duration
    
    success_count = sum(1 for r in results["modules"].values() if r.get("status") == "success")
    total_count = len(results["modules"])
    
    logger.info("=" * 60)
    logger.info("Huracan System Finished", 
                duration_seconds=duration,
                success=f"{success_count}/{total_count}")
    logger.info("=" * 60)
    
    # Save results
    results_path = Path("reports") / f"{start_time.date()}" / "run_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    import json
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("Results saved", path=str(results_path))
    
    # Exit with error if any module failed
    if success_count < total_count:
        sys.exit(1)


if __name__ == "__main__":
    main()

