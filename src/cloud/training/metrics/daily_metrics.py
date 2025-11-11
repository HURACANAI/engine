"""
Daily Metrics System

Tracks and logs all KPIs daily for comprehensive monitoring.

Key Metrics:
- OOS Sharpe and its standard deviation across windows
- Brier score of confidence per regime
- Diversity score in consensus
- Simulator slippage error vs realized costs
- Exploration budget used vs cap
- Early warning precision and recall
- Drawdown days saved by risk advisories

Author: Huracan Engine Team
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json

import numpy as np
import structlog

from .enhanced_metrics import EnhancedMetricsCalculator, PerformanceMetrics

logger = structlog.get_logger(__name__)


@dataclass
class DailyMetrics:
    """Daily metrics snapshot"""
    date: datetime
    symbol: str
    
    # Performance metrics
    oos_sharpe: float
    oos_sharpe_std: float  # Standard deviation across windows
    sharpe_windows: List[float] = field(default_factory=list)
    
    # Confidence metrics
    brier_score: float = 0.0
    brier_score_by_regime: Dict[str, float] = field(default_factory=dict)
    
    # Consensus metrics
    diversity_score: float
    consensus_agreement: float
    
    # Slippage metrics
    simulator_slippage_error: float  # Error vs realized costs
    realized_slippage_bps: float
    expected_slippage_bps: float
    
    # Exploration metrics
    exploration_budget_used: float
    exploration_budget_cap: float
    exploration_budget_ratio: float
    
    # Early warning metrics
    early_warning_precision: float
    early_warning_recall: float
    early_warning_f1: float
    
    # Risk advisory metrics
    drawdown_days_saved: int
    risk_advisories_issued: int
    risk_advisories_effective: int
    
    # Additional metrics
    total_trades: int
    total_pnl_usd: float
    max_drawdown: float
    current_drawdown: float
    
    # Metadata
    metadata: Dict[str, any] = field(default_factory=dict)


class DailyMetricsCollector:
    """
    Collects and logs daily metrics for comprehensive monitoring.
    
    Tracks:
    - OOS Sharpe across windows
    - Brier score per regime
    - Diversity score
    - Slippage error
    - Exploration budget
    - Early warning metrics
    - Risk advisory effectiveness
    
    Usage:
        collector = DailyMetricsCollector(storage_path="metrics/")
        
        collector.record_daily_metrics(
            symbol="BTCUSDT",
            oos_sharpes=[1.5, 1.6, 1.4, ...],
            brier_scores={"trend": 0.1, "range": 0.15, ...},
            diversity_score=0.7,
            ...
        )
        
        # Get metrics for a date
        metrics = collector.get_daily_metrics("BTCUSDT", date)
    """
    
    def __init__(
        self,
        storage_path: Optional[Path] = None,
        metrics_calculator: Optional[EnhancedMetricsCalculator] = None
    ):
        """
        Initialize daily metrics collector.
        
        Args:
            storage_path: Path to store metrics (default: metrics/daily/)
            metrics_calculator: Optional metrics calculator
        """
        self.storage_path = storage_path or Path("metrics/daily")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.metrics_calculator = metrics_calculator or EnhancedMetricsCalculator()
        
        # Store daily metrics
        self.daily_metrics: Dict[str, Dict[str, DailyMetrics]] = {}  # symbol -> date -> metrics
        
        logger.info(
            "daily_metrics_collector_initialized",
            storage_path=str(self.storage_path)
        )
    
    def record_daily_metrics(
        self,
        symbol: str,
        date: Optional[datetime] = None,
        oos_sharpes: Optional[List[float]] = None,
        brier_scores: Optional[Dict[str, float]] = None,
        diversity_score: Optional[float] = None,
        consensus_agreement: Optional[float] = None,
        realized_slippage_bps: Optional[float] = None,
        expected_slippage_bps: Optional[float] = None,
        exploration_budget_used: Optional[float] = None,
        exploration_budget_cap: Optional[float] = None,
        early_warning_precision: Optional[float] = None,
        early_warning_recall: Optional[float] = None,
        drawdown_days_saved: Optional[int] = None,
        risk_advisories_issued: Optional[int] = None,
        risk_advisories_effective: Optional[int] = None,
        trades: Optional[List[Dict[str, any]]] = None,
        returns: Optional[List[float]] = None,
        metadata: Optional[Dict[str, any]] = None
    ) -> DailyMetrics:
        """
        Record daily metrics.
        
        Args:
            symbol: Trading symbol
            date: Date (default: today)
            oos_sharpes: List of OOS Sharpe ratios across windows
            brier_scores: Brier scores by regime
            diversity_score: Diversity score in consensus
            consensus_agreement: Consensus agreement ratio
            realized_slippage_bps: Realized slippage in bps
            expected_slippage_bps: Expected slippage in bps
            exploration_budget_used: Exploration budget used
            exploration_budget_cap: Exploration budget cap
            early_warning_precision: Early warning precision
            early_warning_recall: Early warning recall
            drawdown_days_saved: Drawdown days saved by risk advisories
            risk_advisories_issued: Number of risk advisories issued
            risk_advisories_effective: Number of effective risk advisories
            trades: List of trades
            returns: List of returns
            metadata: Optional metadata
        
        Returns:
            DailyMetrics
        """
        if date is None:
            date = datetime.now(timezone.utc).date()
        else:
            date = date.date() if isinstance(date, datetime) else date
        
        # Calculate OOS Sharpe metrics
        oos_sharpe = 0.0
        oos_sharpe_std = 0.0
        sharpe_windows = oos_sharpes or []
        
        if sharpe_windows:
            oos_sharpe = float(np.mean(sharpe_windows))
            oos_sharpe_std = float(np.std(sharpe_windows))
        
        # Calculate Brier score
        brier_score = 0.0
        brier_score_by_regime = brier_scores or {}
        
        if brier_score_by_regime:
            brier_score = float(np.mean(list(brier_score_by_regime.values())))
        
        # Calculate slippage error
        simulator_slippage_error = 0.0
        if realized_slippage_bps is not None and expected_slippage_bps is not None:
            simulator_slippage_error = abs(realized_slippage_bps - expected_slippage_bps)
        
        # Calculate exploration budget ratio
        exploration_budget_ratio = 0.0
        if exploration_budget_used is not None and exploration_budget_cap is not None:
            if exploration_budget_cap > 0:
                exploration_budget_ratio = exploration_budget_used / exploration_budget_cap
        
        # Calculate early warning F1
        early_warning_f1 = 0.0
        if early_warning_precision is not None and early_warning_recall is not None:
            if early_warning_precision + early_warning_recall > 0:
                early_warning_f1 = 2 * (early_warning_precision * early_warning_recall) / (
                    early_warning_precision + early_warning_recall
                )
        
        # Calculate performance metrics
        total_trades = len(trades) if trades else 0
        total_pnl_usd = sum(t.get("pnl_usd", 0.0) for t in trades) if trades else 0.0
        
        # Calculate drawdown from returns
        max_drawdown = 0.0
        current_drawdown = 0.0
        if returns:
            drawdown_metrics = self.metrics_calculator._calculate_drawdown_metrics(
                np.array(returns),
                None
            )
            max_drawdown = drawdown_metrics["max_drawdown"]
            current_drawdown = drawdown_metrics["current_drawdown"]
        
        # Create daily metrics
        metrics = DailyMetrics(
            date=datetime.combine(date, datetime.min.time()).replace(tzinfo=timezone.utc),
            symbol=symbol,
            oos_sharpe=oos_sharpe,
            oos_sharpe_std=oos_sharpe_std,
            sharpe_windows=sharpe_windows,
            brier_score=brier_score,
            brier_score_by_regime=brier_score_by_regime,
            diversity_score=diversity_score or 0.0,
            consensus_agreement=consensus_agreement or 0.0,
            simulator_slippage_error=simulator_slippage_error,
            realized_slippage_bps=realized_slippage_bps or 0.0,
            expected_slippage_bps=expected_slippage_bps or 0.0,
            exploration_budget_used=exploration_budget_used or 0.0,
            exploration_budget_cap=exploration_budget_cap or 0.0,
            exploration_budget_ratio=exploration_budget_ratio,
            early_warning_precision=early_warning_precision or 0.0,
            early_warning_recall=early_warning_recall or 0.0,
            early_warning_f1=early_warning_f1,
            drawdown_days_saved=drawdown_days_saved or 0,
            risk_advisories_issued=risk_advisories_issued or 0,
            risk_advisories_effective=risk_advisories_effective or 0,
            total_trades=total_trades,
            total_pnl_usd=total_pnl_usd,
            max_drawdown=max_drawdown,
            current_drawdown=current_drawdown,
            metadata=metadata or {}
        )
        
        # Store metrics
        if symbol not in self.daily_metrics:
            self.daily_metrics[symbol] = {}
        
        date_str = date.isoformat()
        self.daily_metrics[symbol][date_str] = metrics
        
        # Save to disk
        self._save_metrics(symbol, date_str, metrics)
        
        logger.info(
            "daily_metrics_recorded",
            symbol=symbol,
            date=date_str,
            oos_sharpe=oos_sharpe,
            diversity_score=diversity_score,
            total_trades=total_trades
        )
        
        return metrics
    
    def get_daily_metrics(
        self,
        symbol: str,
        date: datetime
    ) -> Optional[DailyMetrics]:
        """Get daily metrics for a symbol and date"""
        date_str = date.date().isoformat() if isinstance(date, datetime) else date.isoformat()
        
        if symbol in self.daily_metrics:
            return self.daily_metrics[symbol].get(date_str)
        
        # Try to load from disk
        return self._load_metrics(symbol, date_str)
    
    def get_metrics_summary(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, float]:
        """Get summary metrics for a date range"""
        metrics_list = []
        
        current_date = start_date.date() if isinstance(start_date, datetime) else start_date
        end_date_obj = end_date.date() if isinstance(end_date, datetime) else end_date
        
        while current_date <= end_date_obj:
            metrics = self.get_daily_metrics(symbol, datetime.combine(current_date, datetime.min.time()))
            if metrics:
                metrics_list.append(metrics)
            current_date += timedelta(days=1)
        
        if not metrics_list:
            return {}
        
        # Aggregate metrics
        return {
            "avg_oos_sharpe": float(np.mean([m.oos_sharpe for m in metrics_list])),
            "avg_oos_sharpe_std": float(np.mean([m.oos_sharpe_std for m in metrics_list])),
            "avg_diversity_score": float(np.mean([m.diversity_score for m in metrics_list])),
            "avg_slippage_error": float(np.mean([m.simulator_slippage_error for m in metrics_list])),
            "avg_exploration_ratio": float(np.mean([m.exploration_budget_ratio for m in metrics_list])),
            "avg_early_warning_f1": float(np.mean([m.early_warning_f1 for m in metrics_list])),
            "total_drawdown_days_saved": sum(m.drawdown_days_saved for m in metrics_list),
            "total_trades": sum(m.total_trades for m in metrics_list),
            "total_pnl_usd": sum(m.total_pnl_usd for m in metrics_list)
        }
    
    def _save_metrics(self, symbol: str, date_str: str, metrics: DailyMetrics) -> None:
        """Save metrics to disk"""
        symbol_dir = self.storage_path / symbol
        symbol_dir.mkdir(parents=True, exist_ok=True)
        
        metrics_file = symbol_dir / f"{date_str}.json"
        
        # Convert to dict
        metrics_dict = {
            "date": metrics.date.isoformat(),
            "symbol": metrics.symbol,
            "oos_sharpe": metrics.oos_sharpe,
            "oos_sharpe_std": metrics.oos_sharpe_std,
            "sharpe_windows": metrics.sharpe_windows,
            "brier_score": metrics.brier_score,
            "brier_score_by_regime": metrics.brier_score_by_regime,
            "diversity_score": metrics.diversity_score,
            "consensus_agreement": metrics.consensus_agreement,
            "simulator_slippage_error": metrics.simulator_slippage_error,
            "realized_slippage_bps": metrics.realized_slippage_bps,
            "expected_slippage_bps": metrics.expected_slippage_bps,
            "exploration_budget_used": metrics.exploration_budget_used,
            "exploration_budget_cap": metrics.exploration_budget_cap,
            "exploration_budget_ratio": metrics.exploration_budget_ratio,
            "early_warning_precision": metrics.early_warning_precision,
            "early_warning_recall": metrics.early_warning_recall,
            "early_warning_f1": metrics.early_warning_f1,
            "drawdown_days_saved": metrics.drawdown_days_saved,
            "risk_advisories_issued": metrics.risk_advisories_issued,
            "risk_advisories_effective": metrics.risk_advisories_effective,
            "total_trades": metrics.total_trades,
            "total_pnl_usd": metrics.total_pnl_usd,
            "max_drawdown": metrics.max_drawdown,
            "current_drawdown": metrics.current_drawdown,
            "metadata": metrics.metadata
        }
        
        metrics_file.write_text(json.dumps(metrics_dict, indent=2))
    
    def _load_metrics(self, symbol: str, date_str: str) -> Optional[DailyMetrics]:
        """Load metrics from disk"""
        symbol_dir = self.storage_path / symbol
        metrics_file = symbol_dir / f"{date_str}.json"
        
        if not metrics_file.exists():
            return None
        
        try:
            metrics_dict = json.loads(metrics_file.read_text())
            
            return DailyMetrics(
                date=datetime.fromisoformat(metrics_dict["date"]),
                symbol=metrics_dict["symbol"],
                oos_sharpe=metrics_dict["oos_sharpe"],
                oos_sharpe_std=metrics_dict["oos_sharpe_std"],
                sharpe_windows=metrics_dict.get("sharpe_windows", []),
                brier_score=metrics_dict["brier_score"],
                brier_score_by_regime=metrics_dict.get("brier_score_by_regime", {}),
                diversity_score=metrics_dict["diversity_score"],
                consensus_agreement=metrics_dict["consensus_agreement"],
                simulator_slippage_error=metrics_dict["simulator_slippage_error"],
                realized_slippage_bps=metrics_dict["realized_slippage_bps"],
                expected_slippage_bps=metrics_dict["expected_slippage_bps"],
                exploration_budget_used=metrics_dict["exploration_budget_used"],
                exploration_budget_cap=metrics_dict["exploration_budget_cap"],
                exploration_budget_ratio=metrics_dict["exploration_budget_ratio"],
                early_warning_precision=metrics_dict["early_warning_precision"],
                early_warning_recall=metrics_dict["early_warning_recall"],
                early_warning_f1=metrics_dict["early_warning_f1"],
                drawdown_days_saved=metrics_dict["drawdown_days_saved"],
                risk_advisories_issued=metrics_dict["risk_advisories_issued"],
                risk_advisories_effective=metrics_dict["risk_advisories_effective"],
                total_trades=metrics_dict["total_trades"],
                total_pnl_usd=metrics_dict["total_pnl_usd"],
                max_drawdown=metrics_dict["max_drawdown"],
                current_drawdown=metrics_dict["current_drawdown"],
                metadata=metrics_dict.get("metadata", {})
            )
        except Exception as e:
            logger.warning("metrics_load_failed", symbol=symbol, date=date_str, error=str(e))
            return None

