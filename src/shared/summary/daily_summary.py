"""
Daily Summary Generator

Daily summary JSON. Top contributors. Hit rate. Net edge. Trades counted.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class DailySummary:
    """Daily summary data."""
    date: str  # YYYY-MM-DD
    total_symbols: int
    succeeded: int
    failed: int
    skipped: int
    top_contributors: List[Dict[str, Any]]  # Top symbols by net edge
    hit_rate: float  # Overall hit rate
    net_edge_bps: float  # Overall net edge in basis points
    trades_counted: int  # Number of trades
    avg_train_minutes: float  # Average training time per symbol
    total_wall_minutes: float  # Total wall time
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "date": self.date,
            "total_symbols": self.total_symbols,
            "succeeded": self.succeeded,
            "failed": self.failed,
            "skipped": self.skipped,
            "top_contributors": self.top_contributors,
            "hit_rate": self.hit_rate,
            "net_edge_bps": self.net_edge_bps,
            "trades_counted": self.trades_counted,
            "avg_train_minutes": self.avg_train_minutes,
            "total_wall_minutes": self.total_wall_minutes,
            "metadata": self.metadata or {},
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DailySummary:
        """Create from dictionary."""
        return cls(
            date=data["date"],
            total_symbols=data["total_symbols"],
            succeeded=data["succeeded"],
            failed=data["failed"],
            skipped=data["skipped"],
            top_contributors=data.get("top_contributors", []),
            hit_rate=data.get("hit_rate", 0.0),
            net_edge_bps=data.get("net_edge_bps", 0.0),
            trades_counted=data.get("trades_counted", 0),
            avg_train_minutes=data.get("avg_train_minutes", 0.0),
            total_wall_minutes=data.get("total_wall_minutes", 0.0),
            metadata=data.get("metadata"),
        )


class DailySummaryGenerator:
    """Daily summary generator."""
    
    def __init__(
        self,
        base_path: str = "summaries/daily",
    ):
        """Initialize daily summary generator.
        
        Args:
            base_path: Base path for summaries
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        logger.info("daily_summary_generator_initialized", base_path=str(self.base_path))
    
    def generate_summary(
        self,
        date: str,
        results: List[Dict[str, Any]],
        meta_weights: Optional[Dict[str, float]] = None,
    ) -> DailySummary:
        """Generate daily summary.
        
        Args:
            date: Date in YYYY-MM-DD format
            results: List of training results
            meta_weights: Optional meta weights for top contributors
            
        Returns:
            Daily summary
        """
        total_symbols = len(results)
        succeeded = sum(1 for r in results if r.get("status") == "success")
        failed = sum(1 for r in results if r.get("status") == "failed")
        skipped = sum(1 for r in results if r.get("status") == "skipped")
        
        # Calculate metrics
        hit_rates = [r.get("hit_rate", 0.0) for r in results if r.get("hit_rate")]
        net_edges = [r.get("net_edge_bps", 0.0) for r in results if r.get("net_edge_bps")]
        train_times = [r.get("wall_minutes", 0.0) for r in results if r.get("wall_minutes")]
        trades_counted = sum(r.get("trades_counted", 0) for r in results)
        
        avg_hit_rate = sum(hit_rates) / len(hit_rates) if hit_rates else 0.0
        avg_net_edge = sum(net_edges) / len(net_edges) if net_edges else 0.0
        avg_train_minutes = sum(train_times) / len(train_times) if train_times else 0.0
        total_wall_minutes = sum(train_times)
        
        # Get top contributors
        top_contributors = self._get_top_contributors(results, meta_weights or {})
        
        summary = DailySummary(
            date=date,
            total_symbols=total_symbols,
            succeeded=succeeded,
            failed=failed,
            skipped=skipped,
            top_contributors=top_contributors,
            hit_rate=avg_hit_rate,
            net_edge_bps=avg_net_edge,
            trades_counted=trades_counted,
            avg_train_minutes=avg_train_minutes,
            total_wall_minutes=total_wall_minutes,
            metadata={
                "generated_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        
        return summary
    
    def _get_top_contributors(
        self,
        results: List[Dict[str, Any]],
        meta_weights: Dict[str, float],
        n: int = 10,
    ) -> List[Dict[str, Any]]:
        """Get top contributors by net edge.
        
        Args:
            results: List of training results
            meta_weights: Meta weights for symbols
            n: Number of top contributors to return
            
        Returns:
            List of top contributors
        """
        contributors = []
        
        for result in results:
            symbol = result.get("symbol")
            if not symbol:
                continue
            
            net_edge = result.get("net_edge_bps", 0.0)
            hit_rate = result.get("hit_rate", 0.0)
            meta_weight = meta_weights.get(symbol, 0.0)
            
            contributors.append({
                "symbol": symbol,
                "net_edge_bps": net_edge,
                "hit_rate": hit_rate,
                "meta_weight": meta_weight,
            })
        
        # Sort by net edge
        sorted_contributors = sorted(contributors, key=lambda x: x["net_edge_bps"], reverse=True)
        
        return sorted_contributors[:n]
    
    def save_summary(self, summary: DailySummary) -> Path:
        """Save daily summary to file.
        
        Args:
            summary: Daily summary
            
        Returns:
            Path to saved summary file
        """
        summary_path = self.base_path / f"{summary.date}.json"
        
        try:
            with open(summary_path, 'w') as f:
                json.dump(summary.to_dict(), f, indent=2)
            
            logger.info("summary_saved", date=summary.date, path=str(summary_path))
            return summary_path
        except Exception as e:
            logger.error("summary_save_failed", date=summary.date, error=str(e))
            raise
    
    def load_summary(self, date: str) -> Optional[DailySummary]:
        """Load daily summary from file.
        
        Args:
            date: Date in YYYY-MM-DD format
            
        Returns:
            Daily summary, or None if not found
        """
        summary_path = self.base_path / f"{date}.json"
        
        if not summary_path.exists():
            return None
        
        try:
            with open(summary_path, 'r') as f:
                data = json.load(f)
            
            return DailySummary.from_dict(data)
        except Exception as e:
            logger.error("summary_load_failed", date=date, error=str(e))
            return None

