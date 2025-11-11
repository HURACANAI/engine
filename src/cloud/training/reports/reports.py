"""
Reports System for Training Architecture

Generates comprehensive reports:
- Metrics bundle (Sharpe, Sortino, Max drawdown, Hit rate, Profit factor, Turnover, Capacity)
- Cost report (Fees, Spread, Slippage, Funding, Net edge)
- Decision logs (Consensus score, Votes, Confidence, Actions)
- Regime map (Trend, Range, Panic, Illiquid)
- Data integrity report (Gaps, Outliers, Vendor mismatches)
- Model manifest (Version, Training window, Features hash, Code hash, Timestamp)

Author: Huracan Engine Team
Date: 2025-01-27
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class MetricsReport:
    """Metrics report."""
    coin: str
    horizon: str
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    hit_rate: float
    profit_factor: float
    turnover: float
    capacity_estimate: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class CostReport:
    """Cost report."""
    coin: str
    horizon: str
    fees_bps: float
    spread_bps: float
    slippage_bps: float
    funding_bps: float
    net_edge_after_cost_bps: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class DecisionLog:
    """Decision log entry."""
    timestamp: str
    coin: str
    horizon: str
    consensus_score: float
    votes: List[Dict[str, Any]]
    confidence: float
    action_taken: str
    edge_bps: float
    net_edge_bps: float


@dataclass
class RegimeMapReport:
    """Regime map report."""
    coin: str
    horizons: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class DataIntegrityReport:
    """Data integrity report."""
    coin: str
    horizon: str
    gaps: List[Dict[str, Any]] = field(default_factory=list)
    outliers: List[Dict[str, Any]] = field(default_factory=list)
    vendor_mismatches: List[Dict[str, Any]] = field(default_factory=list)
    data_quality_score: float = 1.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class ReportsGenerator:
    """
    Reports generator for training pipeline.
    
    Generates all required reports for Dropbox export.
    """
    
    def __init__(self, output_dir: Path = Path("/tmp/reports")):
        """
        Initialize reports generator.
        
        Args:
            output_dir: Output directory for reports
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("reports_generator_initialized", output_dir=str(output_dir))
    
    def generate_metrics_report(
        self,
        coin: str,
        horizon: str,
        sharpe_ratio: float,
        sortino_ratio: float,
        max_drawdown: float,
        hit_rate: float,
        profit_factor: float,
        turnover: float,
        capacity_estimate: float,
    ) -> Path:
        """Generate metrics report."""
        report = MetricsReport(
            coin=coin,
            horizon=horizon,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            hit_rate=hit_rate,
            profit_factor=profit_factor,
            turnover=turnover,
            capacity_estimate=capacity_estimate,
        )
        
        report_path = self.output_dir / f"{coin}_{horizon}_metrics.json"
        with open(report_path, "w") as f:
            json.dump(asdict(report), f, indent=2)
        
        logger.info("metrics_report_generated", coin=coin, horizon=horizon, path=str(report_path))
        return report_path
    
    def generate_cost_report(
        self,
        coin: str,
        horizon: str,
        fees_bps: float,
        spread_bps: float,
        slippage_bps: float,
        funding_bps: float,
        net_edge_after_cost_bps: float,
    ) -> Path:
        """Generate cost report."""
        report = CostReport(
            coin=coin,
            horizon=horizon,
            fees_bps=fees_bps,
            spread_bps=spread_bps,
            slippage_bps=slippage_bps,
            funding_bps=funding_bps,
            net_edge_after_cost_bps=net_edge_after_cost_bps,
        )
        
        report_path = self.output_dir / f"{coin}_{horizon}_cost.json"
        with open(report_path, "w") as f:
            json.dump(asdict(report), f, indent=2)
        
        logger.info("cost_report_generated", coin=coin, horizon=horizon, path=str(report_path))
        return report_path
    
    def generate_decision_logs(
        self,
        coin: str,
        horizon: str,
        logs: List[DecisionLog],
    ) -> Path:
        """Generate decision logs."""
        logs_data = [asdict(log) for log in logs]
        
        report_path = self.output_dir / f"{coin}_{horizon}_decision_logs.json"
        with open(report_path, "w") as f:
            json.dump(logs_data, f, indent=2)
        
        logger.info("decision_logs_generated", coin=coin, horizon=horizon, path=str(report_path), log_count=len(logs))
        return report_path
    
    def generate_regime_map(
        self,
        coin: str,
        regime_data: Dict[str, Dict[str, Any]],
    ) -> Path:
        """Generate regime map."""
        report = RegimeMapReport(
            coin=coin,
            horizons=regime_data,
        )
        
        report_path = self.output_dir / f"{coin}_regime_map.json"
        with open(report_path, "w") as f:
            json.dump(asdict(report), f, indent=2)
        
        logger.info("regime_map_generated", coin=coin, path=str(report_path))
        return report_path
    
    def generate_data_integrity_report(
        self,
        coin: str,
        horizon: str,
        gaps: List[Dict[str, Any]],
        outliers: List[Dict[str, Any]],
        vendor_mismatches: List[Dict[str, Any]],
        data_quality_score: float = 1.0,
    ) -> Path:
        """Generate data integrity report."""
        report = DataIntegrityReport(
            coin=coin,
            horizon=horizon,
            gaps=gaps,
            outliers=outliers,
            vendor_mismatches=vendor_mismatches,
            data_quality_score=data_quality_score,
        )
        
        report_path = self.output_dir / f"{coin}_{horizon}_data_integrity.json"
        with open(report_path, "w") as f:
            json.dump(asdict(report), f, indent=2)
        
        logger.info("data_integrity_report_generated", coin=coin, horizon=horizon, path=str(report_path))
        return report_path
    
    def generate_daily_summary(
        self,
        coins_processed: int,
        champions_exported: int,
        skipped_coins: Dict[str, str],
        errors: List[str],
    ) -> Path:
        """Generate daily summary."""
        summary = {
            "timestamp": datetime.now().isoformat(),
            "coins_processed": coins_processed,
            "champions_exported": champions_exported,
            "skipped_coins": skipped_coins,
            "errors": errors,
            "success_rate": champions_exported / coins_processed if coins_processed > 0 else 0.0,
        }
        
        summary_path = self.output_dir / "daily_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        
        logger.info("daily_summary_generated", path=str(summary_path))
        return summary_path

