"""
Dropbox Path Helpers for Per-Coin Training

Path utilities for Dropbox-compatible directory structure.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional


def get_dropbox_base_path(base_folder: str = "huracan") -> str:
    """Get base Dropbox path."""
    return f"/{base_folder}"


def get_manifest_path(date_str: str, base_folder: str = "huracan") -> str:
    """Get manifest path for a date.
    
    Args:
        date_str: Date in YYYYMMDD format
        base_folder: Base folder name (default: "huracan")
        
    Returns:
        Dropbox path to manifest.json
    """
    return f"/{base_folder}/{date_str}/manifest.json"


def get_champion_pointer_path(base_folder: str = "huracan") -> str:
    """Get champion pointer path.
    
    Args:
        base_folder: Base folder name (default: "huracan")
        
    Returns:
        Dropbox path to latest.json
    """
    return f"/{base_folder}/champion/latest.json"


def get_model_path(date_str: str, symbol: str, base_folder: str = "huracan") -> str:
    """Get model path for a symbol on a date.
    
    Args:
        date_str: Date in YYYYMMDD format
        symbol: Trading symbol (e.g., "BTCUSDT")
        base_folder: Base folder name (default: "huracan")
        
    Returns:
        Dropbox path to model.bin
    """
    return f"/{base_folder}/models/baselines/{date_str}/{symbol}/model.bin"


def get_config_path(date_str: str, symbol: str, base_folder: str = "huracan") -> str:
    """Get config path for a symbol on a date.
    
    Args:
        date_str: Date in YYYYMMDD format
        symbol: Trading symbol (e.g., "BTCUSDT")
        base_folder: Base folder name (default: "huracan")
        
    Returns:
        Dropbox path to config.json
    """
    return f"/{base_folder}/models/baselines/{date_str}/{symbol}/config.json"


def get_metrics_path(date_str: str, symbol: str, base_folder: str = "huracan") -> str:
    """Get metrics path for a symbol on a date.
    
    Args:
        date_str: Date in YYYYMMDD format
        symbol: Trading symbol (e.g., "BTCUSDT")
        base_folder: Base folder name (default: "huracan")
        
    Returns:
        Dropbox path to metrics.json
    """
    return f"/{base_folder}/models/baselines/{date_str}/{symbol}/metrics.json"


def get_feature_recipe_path(date_str: str, symbol: str, base_folder: str = "huracan") -> str:
    """Get feature recipe path for a symbol on a date.
    
    Args:
        date_str: Date in YYYYMMDD format
        symbol: Trading symbol (e.g., "BTCUSDT")
        base_folder: Base folder name (default: "huracan")
        
    Returns:
        Dropbox path to feature_recipe.json
    """
    return f"/{base_folder}/models/baselines/{date_str}/{symbol}/feature_recipe.json"


def get_heartbeat_path(base_folder: str = "huracan") -> str:
    """Get heartbeat path.
    
    Args:
        base_folder: Base folder name (default: "huracan")
        
    Returns:
        Dropbox path to engine.json
    """
    return f"/{base_folder}/heartbeats/engine.json"


def get_failure_report_path(date_str: str, base_folder: str = "huracan") -> str:
    """Get failure report path for a date.
    
    Args:
        date_str: Date in YYYYMMDD format
        base_folder: Base folder name (default: "huracan")
        
    Returns:
        Dropbox path to failure_report.json
    """
    return f"/{base_folder}/{date_str}/logs/failure_report.json"


def get_daily_summary_path(date_str: str, base_folder: str = "huracan") -> str:
    """Get daily summary report path.
    
    Args:
        date_str: Date in YYYYMMDD format
        base_folder: Base folder name (default: "huracan")
        
    Returns:
        Dropbox path to summary.json
    """
    return f"/{base_folder}/reports/daily/{date_str}/summary.json"


def format_date_str(date: Optional[datetime] = None) -> str:
    """Format date as YYYYMMDD string.
    
    Args:
        date: Date to format (defaults to today)
        
    Returns:
        Date string in YYYYMMDD format
    """
    if date is None:
        date = datetime.now()
    return date.strftime("%Y%m%d")


def format_date_iso(date: Optional[datetime] = None) -> str:
    """Format date as YYYY-MM-DD string.
    
    Args:
        date: Date to format (defaults to today)
        
    Returns:
        Date string in YYYY-MM-DD format
    """
    if date is None:
        date = datetime.now()
    return date.strftime("%Y-%m-%d")


def get_symbol_directory(date_str: str, symbol: str, base_folder: str = "huracan") -> str:
    """Get directory path for a symbol on a date.
    
    Args:
        date_str: Date in YYYYMMDD format
        symbol: Trading symbol (e.g., "BTCUSDT")
        base_folder: Base folder name (default: "huracan")
        
    Returns:
        Dropbox directory path for symbol
    """
    return f"/{base_folder}/models/baselines/{date_str}/{symbol}"


def get_promotions_log_path(base_folder: str = "huracan") -> str:
    """Get promotions log path.
    
    Args:
        base_folder: Base folder name (default: "huracan")
        
    Returns:
        Dropbox path to promotions log
    """
    return f"/{base_folder}/champion/promotions.json"


def make_absolute_path(dropbox_path: str, base_path: str = "/huracan") -> str:
    """Make an absolute Dropbox path.
    
    Args:
        dropbox_path: Relative Dropbox path
        base_path: Base path prefix
        
    Returns:
        Absolute Dropbox path
    """
    if dropbox_path.startswith("/"):
        return dropbox_path
    return f"{base_path}/{dropbox_path}"

