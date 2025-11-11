"""
Enhanced Metrics Layer

Calculates and logs Sharpe, Sortino, and drawdown metrics.
Provides comprehensive performance tracking.

Key Features:
- Sharpe ratio calculation
- Sortino ratio calculation
- Maximum drawdown tracking
- Calmar ratio calculation
- Per-regime metrics
- Daily logging

Author: Huracan Engine Team
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from enum import Enum

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics"""
    # Returns
    total_return: float
    mean_return: float
    annualized_return: float
    
    # Volatility
    volatility: float
    annualized_volatility: float
    downside_volatility: float
    
    # Risk-adjusted returns
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    
    # Drawdown
    max_drawdown: float
    max_drawdown_duration_days: int
    current_drawdown: float
    drawdown_recovery_slope: float
    
    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    
    # Regime-specific
    regime_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Timestamps
    calculation_date: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class EnhancedMetricsCalculator:
    """
    Enhanced metrics calculator for comprehensive performance tracking.
    
    Calculates:
    - Sharpe ratio (annualized)
    - Sortino ratio (downside deviation)
    - Maximum drawdown
    - Calmar ratio
    - Per-regime metrics
    
    Usage:
        calculator = EnhancedMetricsCalculator()
        metrics = calculator.calculate_metrics(
            returns=[0.01, -0.02, 0.03, ...],
            trades=[...],
            regime_labels=[...]
        )
        
        print(f"Sharpe: {metrics.sharpe_ratio:.2f}")
        print(f"Sortino: {metrics.sortino_ratio:.2f}")
        print(f"Max Drawdown: {metrics.max_drawdown:.2%}")
    """
    
    def __init__(
        self,
        risk_free_rate: float = 0.0,
        trading_days_per_year: int = 252
    ):
        """
        Initialize enhanced metrics calculator.
        
        Args:
            risk_free_rate: Risk-free rate (default: 0.0)
            trading_days_per_year: Trading days per year (default: 252)
        """
        self.risk_free_rate = risk_free_rate
        self.trading_days_per_year = trading_days_per_year
        
        logger.info(
            "enhanced_metrics_calculator_initialized",
            risk_free_rate=risk_free_rate,
            trading_days_per_year=trading_days_per_year
        )
    
    def calculate_metrics(
        self,
        returns: List[float],
        trades: Optional[List[Dict[str, any]]] = None,
        regime_labels: Optional[List[str]] = None,
        timestamps: Optional[List[datetime]] = None
    ) -> PerformanceMetrics:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            returns: List of returns
            trades: List of trade dictionaries (optional)
            regime_labels: List of regime labels (optional)
            timestamps: List of timestamps (optional)
        
        Returns:
            PerformanceMetrics with all calculated metrics
        """
        returns_array = np.array(returns)
        
        if len(returns_array) == 0:
            return self._empty_metrics()
        
        # Basic return metrics
        total_return = float(np.sum(returns_array))
        mean_return = float(np.mean(returns_array))
        annualized_return = float(mean_return * self.trading_days_per_year)
        
        # Volatility metrics
        volatility = float(np.std(returns_array))
        annualized_volatility = float(volatility * np.sqrt(self.trading_days_per_year))
        
        # Downside volatility (for Sortino)
        downside_returns = returns_array[returns_array < 0]
        downside_volatility = float(np.std(downside_returns)) if len(downside_returns) > 0 else 0.0
        annualized_downside_volatility = downside_volatility * np.sqrt(self.trading_days_per_year)
        
        # Sharpe ratio
        sharpe_ratio = self._calculate_sharpe_ratio(
            returns_array,
            annualized_return,
            annualized_volatility
        )
        
        # Sortino ratio
        sortino_ratio = self._calculate_sortino_ratio(
            returns_array,
            annualized_return,
            annualized_downside_volatility
        )
        
        # Drawdown metrics
        drawdown_metrics = self._calculate_drawdown_metrics(
            returns_array,
            timestamps
        )
        
        # Trade statistics
        trade_stats = self._calculate_trade_statistics(trades) if trades else {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0
        }
        
        # Regime-specific metrics
        regime_metrics = self._calculate_regime_metrics(
            returns_array,
            regime_labels
        ) if regime_labels else {}
        
        # Calmar ratio
        calmar_ratio = self._calculate_calmar_ratio(
            annualized_return,
            drawdown_metrics["max_drawdown"]
        )
        
        metrics = PerformanceMetrics(
            total_return=total_return,
            mean_return=mean_return,
            annualized_return=annualized_return,
            volatility=volatility,
            annualized_volatility=annualized_volatility,
            downside_volatility=downside_volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            max_drawdown=drawdown_metrics["max_drawdown"],
            max_drawdown_duration_days=drawdown_metrics["max_drawdown_duration_days"],
            current_drawdown=drawdown_metrics["current_drawdown"],
            drawdown_recovery_slope=drawdown_metrics["recovery_slope"],
            total_trades=trade_stats["total_trades"],
            winning_trades=trade_stats["winning_trades"],
            losing_trades=trade_stats["losing_trades"],
            win_rate=trade_stats["win_rate"],
            profit_factor=trade_stats["profit_factor"],
            regime_metrics=regime_metrics
        )
        
        logger.info(
            "metrics_calculated",
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=drawdown_metrics["max_drawdown"],
            total_trades=trade_stats["total_trades"],
            win_rate=trade_stats["win_rate"]
        )
        
        return metrics
    
    def _calculate_sharpe_ratio(
        self,
        returns: np.ndarray,
        annualized_return: float,
        annualized_volatility: float
    ) -> float:
        """Calculate Sharpe ratio"""
        if annualized_volatility == 0:
            return 0.0
        
        sharpe = (annualized_return - self.risk_free_rate) / annualized_volatility
        return float(sharpe)
    
    def _calculate_sortino_ratio(
        self,
        returns: np.ndarray,
        annualized_return: float,
        annualized_downside_volatility: float
    ) -> float:
        """Calculate Sortino ratio (downside deviation only)"""
        if annualized_downside_volatility == 0:
            return 0.0
        
        sortino = (annualized_return - self.risk_free_rate) / annualized_downside_volatility
        return float(sortino)
    
    def _calculate_drawdown_metrics(
        self,
        returns: np.ndarray,
        timestamps: Optional[List[datetime]]
    ) -> Dict[str, float]:
        """Calculate drawdown metrics"""
        # Calculate cumulative returns
        cumulative = np.cumsum(returns)
        
        # Calculate running maximum
        running_max = np.maximum.accumulate(cumulative)
        
        # Calculate drawdown
        drawdown = cumulative - running_max
        
        # Maximum drawdown
        max_drawdown = float(abs(np.min(drawdown))) if len(drawdown) > 0 else 0.0
        
        # Current drawdown
        current_drawdown = float(abs(drawdown[-1])) if len(drawdown) > 0 else 0.0
        
        # Drawdown duration (simplified - would need timestamps for accurate days)
        if timestamps is not None and len(timestamps) > 1:
            # Find max drawdown period
            max_dd_idx = np.argmin(drawdown)
            # Find recovery point (when drawdown returns to 0)
            recovery_idx = max_dd_idx
            for i in range(max_dd_idx, len(drawdown)):
                if drawdown[i] >= 0:
                    recovery_idx = i
                    break
            
            if recovery_idx > max_dd_idx:
                duration = (timestamps[recovery_idx] - timestamps[max_dd_idx]).days
            else:
                duration = 0
        else:
            duration = 0
        
        # Recovery slope (positive = recovering)
        if len(drawdown) >= 2:
            recent_drawdowns = drawdown[-min(10, len(drawdown)):]
            if len(recent_drawdowns) > 1:
                recovery_slope = float(np.polyfit(range(len(recent_drawdowns)), recent_drawdowns, 1)[0])
            else:
                recovery_slope = 0.0
        else:
            recovery_slope = 0.0
        
        return {
            "max_drawdown": max_drawdown,
            "max_drawdown_duration_days": duration,
            "current_drawdown": current_drawdown,
            "recovery_slope": recovery_slope
        }
    
    def _calculate_trade_statistics(
        self,
        trades: List[Dict[str, any]]
    ) -> Dict[str, any]:
        """Calculate trade statistics"""
        if not trades:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0
            }
        
        total_trades = len(trades)
        
        # Extract P&L
        pnls = [t.get("pnl_usd", 0.0) or t.get("pnl_bps", 0.0) for t in trades]
        
        winning_trades = sum(1 for pnl in pnls if pnl > 0)
        losing_trades = total_trades - winning_trades
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
        
        # Profit factor
        gross_profit = sum(pnl for pnl in pnls if pnl > 0)
        gross_loss = abs(sum(pnl for pnl in pnls if pnl < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0
        
        return {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
            "profit_factor": profit_factor
        }
    
    def _calculate_regime_metrics(
        self,
        returns: np.ndarray,
        regime_labels: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate metrics per regime"""
        if len(returns) != len(regime_labels):
            return {}
        
        regime_metrics = {}
        unique_regimes = set(regime_labels)
        
        for regime in unique_regimes:
            regime_returns = returns[np.array(regime_labels) == regime]
            
            if len(regime_returns) > 0:
                regime_mean = float(np.mean(regime_returns))
                regime_std = float(np.std(regime_returns))
                regime_sharpe = regime_mean / regime_std * np.sqrt(self.trading_days_per_year) if regime_std > 0 else 0.0
                
                regime_metrics[regime] = {
                    "mean_return": regime_mean,
                    "volatility": regime_std,
                    "sharpe_ratio": regime_sharpe,
                    "trade_count": len(regime_returns)
                }
        
        return regime_metrics
    
    def _calculate_calmar_ratio(
        self,
        annualized_return: float,
        max_drawdown: float
    ) -> float:
        """Calculate Calmar ratio (annual return / max drawdown)"""
        if max_drawdown == 0:
            return 0.0
        
        calmar = annualized_return / max_drawdown
        return float(calmar)
    
    def evaluate_sharpe(
        self,
        returns: List[float] | np.ndarray,
        periods_per_year: Optional[int] = None
    ) -> float:
        """
        Evaluate Sharpe ratio for returns.
        
        This is a convenience method that can be called from Engine.
        Formula: sharpe = annualized_return / annualized_volatility
        
        Args:
            returns: Array or list of returns
            periods_per_year: Number of periods per year (default: uses trading_days_per_year)
        
        Returns:
            Sharpe ratio
        
        Usage:
            sharpe = calculator.evaluate_sharpe(returns)
            # Council can use Sharpe for voting (e.g., only models with Sharpe > 1 qualify)
        """
        if isinstance(returns, list):
            returns_array = np.array(returns)
        else:
            returns_array = returns
        
        if len(returns_array) == 0:
            return 0.0
        
        # Calculate annualized return and volatility
        if periods_per_year is None:
            periods_per_year = self.trading_days_per_year
        
        mean_return = float(np.mean(returns_array))
        annualized_return = mean_return * periods_per_year
        
        volatility = float(np.std(returns_array))
        annualized_volatility = volatility * np.sqrt(periods_per_year)
        
        # Calculate Sharpe ratio
        if annualized_volatility == 0:
            return 0.0
        
        sharpe = (annualized_return - self.risk_free_rate) / annualized_volatility
        
        logger.info(
            "sharpe_evaluated",
            sharpe_ratio=float(sharpe),
            annualized_return=annualized_return,
            annualized_volatility=annualized_volatility
        )
        
        return float(sharpe)
    
    def _empty_metrics(self) -> PerformanceMetrics:
        """Return empty metrics"""
        return PerformanceMetrics(
            total_return=0.0,
            mean_return=0.0,
            annualized_return=0.0,
            volatility=0.0,
            annualized_volatility=0.0,
            downside_volatility=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            calmar_ratio=0.0,
            max_drawdown=0.0,
            max_drawdown_duration_days=0,
            current_drawdown=0.0,
            drawdown_recovery_slope=0.0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            profit_factor=0.0
        )
    
    def calculate_cost_aware_metrics(
        self,
        returns: List[float],
        trades: Optional[List[Dict[str, any]]] = None,
        taker_fee_bps: float = 10.0,
        maker_fee_bps: float = 5.0,
        avg_spread_bps: Optional[Dict[str, float]] = None,
        slippage_bps_per_sigma: float = 2.0,
    ) -> Dict[str, float]:
        """Calculate cost-aware net metrics after transaction costs.
        
        Args:
            returns: List of gross returns (before costs)
            trades: List of trade dictionaries (optional)
            taker_fee_bps: Taker fee in basis points (default: 10.0)
            maker_fee_bps: Maker fee in basis points (default: 5.0)
            avg_spread_bps: Map of symbol to average spread in bps (optional)
            slippage_bps_per_sigma: Slippage in bps per sigma of volatility (default: 2.0)
            
        Returns:
            Dict with cost-aware metrics ready for manifest.json:
            - sharpe: Sharpe ratio after costs
            - hit_rate: Win rate after costs
            - dd: Maximum drawdown after costs
            - cost_bps: Average cost per trade in bps
            - net_return: Net return after costs
            - taker_fee_bps: Taker fee used
            - maker_fee_bps: Maker fee used
            - avg_spread_bps: Average spread (weighted by symbol)
            - slippage_bps_per_sigma: Slippage parameter used
        """
        returns_array = np.array(returns)
        
        if len(returns_array) == 0:
            return {
                "sharpe": 0.0,
                "hit_rate": 0.0,
                "dd": 0.0,
                "cost_bps": 0.0,
                "net_return": 0.0,
                "taker_fee_bps": taker_fee_bps,
                "maker_fee_bps": maker_fee_bps,
                "avg_spread_bps": 0.0,
                "slippage_bps_per_sigma": slippage_bps_per_sigma,
            }
        
        # Calculate average spread (weighted by symbol if provided)
        if avg_spread_bps and trades:
            # Calculate weighted average spread
            symbol_spreads = {}
            symbol_counts = {}
            for trade in trades:
                symbol = trade.get("symbol", "unknown")
                if symbol in avg_spread_bps:
                    symbol_spreads[symbol] = avg_spread_bps[symbol]
                    symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1
            
            if symbol_counts:
                total_count = sum(symbol_counts.values())
                weighted_spread = sum(
                    symbol_spreads.get(symbol, 0.0) * count
                    for symbol, count in symbol_counts.items()
                ) / total_count
            else:
                weighted_spread = np.mean(list(avg_spread_bps.values())) if avg_spread_bps else 5.0
        else:
            weighted_spread = 5.0  # Default spread
        
        # Estimate costs per trade
        # Cost = taker_fee + maker_fee (round trip) + spread/2 (half spread on entry) + slippage
        # For simplicity, assume all trades are taker (worst case)
        cost_per_trade_bps = taker_fee_bps * 2  # Round trip
        cost_per_trade_bps += weighted_spread / 2  # Half spread on entry
        cost_per_trade_bps += slippage_bps_per_sigma * np.std(returns_array) if len(returns_array) > 0 else 0.0
        
        # Calculate net returns (gross returns minus costs)
        net_returns = returns_array - (cost_per_trade_bps / 10000.0)  # Convert bps to decimal
        
        # Calculate net metrics
        net_mean = float(np.mean(net_returns))
        net_std = float(np.std(net_returns))
        net_sharpe = (net_mean * self.trading_days_per_year) / (net_std * np.sqrt(self.trading_days_per_year)) if net_std > 0 else 0.0
        
        # Calculate net drawdown
        net_cumulative = np.cumsum(net_returns)
        net_running_max = np.maximum.accumulate(net_cumulative)
        net_drawdown = net_cumulative - net_running_max
        net_max_dd = float(abs(np.min(net_drawdown))) if len(net_drawdown) > 0 else 0.0
        
        # Calculate hit rate (win rate after costs)
        net_wins = sum(1 for r in net_returns if r > 0)
        net_hit_rate = net_wins / len(net_returns) if len(net_returns) > 0 else 0.0
        
        # Total net return
        net_total_return = float(np.sum(net_returns))
        
        cost_aware_metrics = {
            "sharpe": float(net_sharpe),
            "hit_rate": float(net_hit_rate),
            "dd": float(net_max_dd),
            "cost_bps": float(cost_per_trade_bps),
            "net_return": float(net_total_return),
            "taker_fee_bps": taker_fee_bps,
            "maker_fee_bps": maker_fee_bps,
            "avg_spread_bps": float(weighted_spread),
            "slippage_bps_per_sigma": slippage_bps_per_sigma,
        }
        
        logger.info(
            "cost_aware_metrics_calculated",
            sharpe=cost_aware_metrics["sharpe"],
            hit_rate=cost_aware_metrics["hit_rate"],
            cost_bps=cost_aware_metrics["cost_bps"],
        )
        
        return cost_aware_metrics

