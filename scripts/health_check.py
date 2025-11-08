#!/usr/bin/env python3
"""
Standalone Health Check Script

Run this script to test all engine components before starting the bot.

Usage:
    python scripts/health_check.py
    
Exit codes:
    0 - All checks passed
    1 - Critical failures detected
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from cloud.training.services.health_check import validate_health_and_exit
from cloud.training.config.settings import EngineSettings

if __name__ == "__main__":
    print("🏥 Huracan Engine Health Check")
    print("=" * 80)
    print()
    
    # Load settings
    try:
        settings = EngineSettings.load()
    except Exception as e:
        print(f"❌ Failed to load settings: {e}")
        sys.exit(1)
    
    # Run health check
    is_healthy = validate_health_and_exit(settings=settings, exit_on_failure=True)
    
    if is_healthy:
        print("✅ Health check passed - Engine is ready to start")
        sys.exit(0)
    else:
        print("❌ Health check failed - Please fix issues before starting")
        sys.exit(1)

