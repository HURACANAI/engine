"""
Daily Report Generator

Generates a simple daily_report.json with system summary.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import structlog

logger = structlog.get_logger(__name__)


class DailyReport:
    """Generates daily reports."""
    
    def __init__(self, output_dir: str = "reports"):
        """Initialize daily report generator.
        
        Args:
            output_dir: Output directory for reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("daily_report_initialized", output_dir=str(self.output_dir))
    
    def generate_report(
        self,
        date: Optional[datetime] = None,
        models_trained: int = 0,
        promotions: int = 0,
        total_trades: int = 0,
        avg_pnl_pct: float = 0.0,
        hit_rate_pct: float = 0.0,
        symbols_trained: Optional[list] = None,
        symbols_failed: Optional[list] = None,
        errors: Optional[list] = None,
    ) -> Dict[str, Any]:
        """Generate daily report.
        
        Args:
            date: Report date (defaults to today)
            models_trained: Number of models trained
            promotions: Number of promotions
            total_trades: Total number of trades
            avg_pnl_pct: Average P&L percentage
            hit_rate_pct: Hit rate percentage
            symbols_trained: List of symbols trained
            symbols_failed: List of symbols that failed
            errors: List of errors
            
        Returns:
            Daily report dictionary
        """
        if date is None:
            date = datetime.now(timezone.utc)
        
        report = {
            "date": date.strftime("%Y-%m-%d"),
            "models_trained": models_trained,
            "promotions": promotions,
            "total_trades": total_trades,
            "avg_pnl_pct": round(avg_pnl_pct, 2),
            "hit_rate_pct": round(hit_rate_pct, 1),
            "symbols_trained": symbols_trained or [],
            "symbols_failed": symbols_failed or [],
            "errors": errors or [],
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        
        return report
    
    def save_report(self, report: Dict[str, Any]) -> Path:
        """Save daily report to file.
        
        Args:
            report: Daily report dictionary
            
        Returns:
            Path to saved report
        """
        date_str = report["date"]
        report_dir = self.output_dir / date_str
        report_dir.mkdir(parents=True, exist_ok=True)
        
        report_path = report_dir / "daily_report.json"
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info("daily_report_saved", path=str(report_path), date=date_str)
        return report_path
    
    def load_report(self, date: Optional[datetime] = None) -> Optional[Dict[str, Any]]:
        """Load daily report from file.
        
        Args:
            date: Report date (defaults to today)
            
        Returns:
            Daily report dictionary, or None if not found
        """
        if date is None:
            date = datetime.now(timezone.utc)
        
        date_str = date.strftime("%Y-%m-%d")
        report_path = self.output_dir / date_str / "daily_report.json"
        
        if not report_path.exists():
            return None
        
        with open(report_path, 'r') as f:
            report = json.load(f)
        
        return report

