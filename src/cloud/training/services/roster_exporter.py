"""
Roster Exporter for Hamilton

Writes champions/roster.json with ranked symbols by liquidity, cost, and recent net edge.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog

from src.shared.contracts.per_coin import PerCoinMetrics, CostModel

logger = structlog.get_logger(__name__)


class RosterExporter:
    """Exports roster.json for Hamilton."""
    
    def __init__(
        self,
        output_dir: str = "champions",
        base_folder: str = "huracan",
    ):
        """Initialize roster exporter.
        
        Args:
            output_dir: Output directory for roster file
            base_folder: Base folder name in Dropbox
        """
        self.output_dir = Path(output_dir)
        self.base_folder = base_folder
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("roster_exporter_initialized", output_dir=str(self.output_dir))
    
    def create_roster_entry(
        self,
        symbol: str,
        model_path: str,
        metrics: PerCoinMetrics,
        cost_model: CostModel,
        rank: int,
        last_7d_net_bps: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Create roster entry for a symbol.
        
        Args:
            symbol: Trading symbol
            model_path: Path to model file
            metrics: Per-coin metrics
            cost_model: Cost model
            rank: Rank of symbol
            last_7d_net_bps: Last 7 days net bps (optional)
            
        Returns:
            Roster entry dictionary
        """
        # Calculate total cost
        total_cost_bps = cost_model.total_cost_bps(order_type="taker")
        
        # Determine trade_ok
        # Trade OK if: net_pnl_pct > 0, sample_size > 100, sharpe > 0.5
        trade_ok = (
            metrics.net_pnl_pct > 0 and
            metrics.sample_size > 100 and
            metrics.sharpe > 0.5 and
            metrics.hit_rate > 0.45 and
            metrics.max_drawdown_pct < 20.0
        )
        
        entry = {
            "symbol": symbol,
            "model_path": model_path,
            "rank": rank,
            "spread_bps": cost_model.median_spread_bps,
            "fee_bps": cost_model.taker_fee_bps,
            "avg_slip_bps": cost_model.slippage_bps_per_sigma,
            "total_cost_bps": total_cost_bps,
            "last_7d_net_bps": last_7d_net_bps or metrics.net_pnl_pct * 100,  # Convert to bps
            "trade_ok": trade_ok,
            "metrics": {
                "sharpe": metrics.sharpe,
                "hit_rate": metrics.hit_rate,
                "net_pnl_pct": metrics.net_pnl_pct,
                "max_drawdown_pct": metrics.max_drawdown_pct,
                "sample_size": metrics.sample_size,
            },
        }
        
        return entry
    
    def rank_symbols(
        self,
        symbols_data: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Rank symbols by liquidity, cost, and recent net edge.
        
        Args:
            symbols_data: List of symbol data dictionaries
            
        Returns:
            Ranked list of symbols
        """
        # Sort by: net_pnl_pct (desc), then by total_cost_bps (asc), then by sample_size (desc)
        ranked = sorted(
            symbols_data,
            key=lambda x: (
                -x.get("last_7d_net_bps", 0),  # Higher net bps is better
                x.get("total_cost_bps", 100),  # Lower cost is better
                -x.get("metrics", {}).get("sample_size", 0),  # Larger sample is better
            ),
            reverse=False,  # We want best first
        )
        
        # Assign ranks
        for i, entry in enumerate(ranked):
            entry["rank"] = i + 1
        
        return ranked
    
    def export_roster(
        self,
        symbols_data: List[Dict[str, Any]],
        date_str: Optional[str] = None,
    ) -> Path:
        """Export roster.json file.
        
        Args:
            symbols_data: List of symbol data dictionaries
            date_str: Date string in YYYYMMDD format (defaults to today)
            
        Returns:
            Path to saved roster file
        """
        if date_str is None:
            date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
        
        # Rank symbols
        ranked_symbols = self.rank_symbols(symbols_data)
        
        # Create roster
        roster = {
            "date": date_str,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_symbols": len(ranked_symbols),
            "trade_ok_count": sum(1 for s in ranked_symbols if s.get("trade_ok", False)),
            "symbols": ranked_symbols,
        }
        
        # Save to file
        roster_path = self.output_dir / "roster.json"
        with open(roster_path, 'w') as f:
            json.dump(roster, f, indent=2)
        
        logger.info("roster_exported", 
                   path=str(roster_path),
                   total_symbols=len(ranked_symbols),
                   trade_ok_count=roster["trade_ok_count"])
        
        return roster_path
    
    def load_roster(self, roster_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
        """Load roster.json file.
        
        Args:
            roster_path: Path to roster file (defaults to champions/roster.json)
            
        Returns:
            Roster dictionary, or None if not found
        """
        if roster_path is None:
            roster_path = self.output_dir / "roster.json"
        
        if not roster_path.exists():
            return None
        
        with open(roster_path, 'r') as f:
            roster = json.load(f)
        
        return roster
    
    def get_trade_ok_symbols(self, limit: Optional[int] = None) -> List[str]:
        """Get symbols with trade_ok=true, ranked.
        
        Args:
            limit: Maximum number of symbols to return (None for all)
            
        Returns:
            List of symbols
        """
        roster = self.load_roster()
        if not roster:
            return []
        
        trade_ok_symbols = [
            s["symbol"] for s in roster["symbols"]
            if s.get("trade_ok", False)
        ]
        
        if limit:
            trade_ok_symbols = trade_ok_symbols[:limit]
        
        return trade_ok_symbols

