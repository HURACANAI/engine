"""
Enhanced Backtest Reports - Comprehensive metrics.

Provides all required metrics for backtest evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class BacktestMetrics:
    """Comprehensive backtest metrics."""
    # Returns
    net_sharpe: float
    gross_sharpe: float
    sortino_ratio: float
    calmar_ratio: float
    
    # Trade statistics
    hit_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    total_trades: int
    
    # Drawdown
    max_drawdown: float
    avg_drawdown: float
    time_to_recover_days: float
    
    # Turnover
    turnover: float
    trades_per_day: float
    avg_holding_time_hours: float
    
    # Costs
    slippage_share_of_gross_alpha: float
    fee_share_of_gross_alpha: float
    funding_share_of_gross_alpha: float
    total_cost_share: float
    
    # Capacity
    capacity_estimate_usd: float
    pnl_drop_at_scale_pct: float
    
    # Stability
    sharpe_by_year: Dict[str, float]
    sharpe_by_coin: Dict[str, float]
    sharpe_std: float  # Dispersion across periods
    sharpe_min: float
    sharpe_max: float


class EnhancedBacktestReporter:
    """
    Enhanced backtest reporter with comprehensive metrics.
    
    Calculates all required metrics including:
    - Risk-adjusted returns (Sharpe, Sortino, Calmar)
    - Trade statistics (hit rate, profit factor)
    - Drawdown metrics
    - Turnover and holding time
    - Cost breakdown
    - Capacity estimates
    - Stability metrics
    """
    
    def __init__(self, risk_free_rate: float = 0.0) -> None:
        """
        Initialize enhanced backtest reporter.
        
        Args:
            risk_free_rate: Risk-free rate for Sharpe calculation (default: 0.0)
        """
        self.risk_free_rate = risk_free_rate
        logger.info("enhanced_backtest_reporter_initialized")
    
    def calculate_metrics(
        self,
        returns: pd.Series | np.ndarray,
        trades: List[Dict[str, any]],
        equity_curve: pd.Series | np.ndarray,
        costs_breakdown: Dict[str, float],
        timestamps: Optional[pd.Series] = None,
    ) -> BacktestMetrics:
        """
        Calculate comprehensive backtest metrics.
        
        Args:
            returns: Series of returns
            trades: List of trade dictionaries
            equity_curve: Equity curve over time
            costs_breakdown: Cost breakdown (slippage, fees, funding)
            timestamps: Optional timestamps for time-based calculations
        
        Returns:
            BacktestMetrics with all calculated metrics
        """
        returns_array = np.array(returns) if not isinstance(returns, np.ndarray) else returns
        
        # Risk-adjusted returns
        net_sharpe = self._calculate_sharpe(returns_array)
        gross_sharpe = self._calculate_gross_sharpe(returns_array, costs_breakdown)
        sortino = self._calculate_sortino(returns_array)
        calmar = self._calculate_calmar(returns_array, equity_curve)
        
        # Trade statistics
        hit_rate, avg_win, avg_loss, profit_factor = self._calculate_trade_stats(trades)
        
        # Drawdown
        max_dd, avg_dd, time_to_recover = self._calculate_drawdown_metrics(
            equity_curve, timestamps
        )
        
        # Turnover
        turnover, trades_per_day, avg_holding_time = self._calculate_turnover_metrics(
            trades, timestamps
        )
        
        # Costs
        cost_shares = self._calculate_cost_shares(returns_array, costs_breakdown)
        
        # Capacity (simplified)
        capacity, pnl_drop = self._estimate_capacity(returns_array, trades)
        
        # Stability
        sharpe_by_year, sharpe_by_coin, sharpe_std, sharpe_min, sharpe_max = \
            self._calculate_stability(returns, timestamps, trades)
        
        return BacktestMetrics(
            net_sharpe=net_sharpe,
            gross_sharpe=gross_sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            hit_rate=hit_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            total_trades=len(trades),
            max_drawdown=max_dd,
            avg_drawdown=avg_dd,
            time_to_recover_days=time_to_recover,
            turnover=turnover,
            trades_per_day=trades_per_day,
            avg_holding_time_hours=avg_holding_time,
            slippage_share_of_gross_alpha=cost_shares['slippage'],
            fee_share_of_gross_alpha=cost_shares['fees'],
            funding_share_of_gross_alpha=cost_shares['funding'],
            total_cost_share=cost_shares['total'],
            capacity_estimate_usd=capacity,
            pnl_drop_at_scale_pct=pnl_drop,
            sharpe_by_year=sharpe_by_year,
            sharpe_by_coin=sharpe_by_coin,
            sharpe_std=sharpe_std,
            sharpe_min=sharpe_min,
            sharpe_max=sharpe_max,
        )
    
    def _calculate_sharpe(self, returns: np.ndarray) -> float:
        """Calculate Sharpe ratio (annualized)."""
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        return float((np.mean(returns) - self.risk_free_rate / 252) / np.std(returns) * np.sqrt(252))
    
    def _calculate_gross_sharpe(
        self,
        returns: np.ndarray,
        costs: Dict[str, float]
    ) -> float:
        """Calculate gross Sharpe (before costs)."""
        # Add costs back to returns
        total_costs = sum(costs.values())
        gross_returns = returns + (total_costs / len(returns))
        return self._calculate_sharpe(gross_returns)
    
    def _calculate_sortino(self, returns: np.ndarray) -> float:
        """Calculate Sortino ratio (downside deviation only)."""
        if len(returns) == 0:
            return 0.0
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0 or np.std(downside_returns) == 0:
            return 0.0
        return float(np.mean(returns) / np.std(downside_returns) * np.sqrt(252))
    
    def _calculate_calmar(
        self,
        returns: np.ndarray,
        equity_curve: np.ndarray | pd.Series
    ) -> float:
        """Calculate Calmar ratio (return / max drawdown)."""
        equity_array = np.array(equity_curve) if not isinstance(equity_curve, np.ndarray) else equity_curve
        if len(equity_array) == 0:
            return 0.0
        
        # Calculate max drawdown
        cumulative = np.cumprod(1 + returns) if len(returns) > 0 else equity_array
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_dd = abs(np.min(drawdown)) if len(drawdown) > 0 else 0.0
        
        if max_dd == 0:
            return 0.0
        
        annual_return = np.mean(returns) * 252 if len(returns) > 0 else 0.0
        return float(annual_return / max_dd)
    
    def _calculate_trade_stats(
        self,
        trades: List[Dict[str, any]]
    ) -> tuple[float, float, float, float]:
        """Calculate trade statistics."""
        if not trades:
            return 0.0, 0.0, 0.0, 0.0
        
        pnls = [t.get('pnl', 0.0) for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        
        hit_rate = len(wins) / len(pnls) if pnls else 0.0
        avg_win = np.mean(wins) if wins else 0.0
        avg_loss = abs(np.mean(losses)) if losses else 0.0
        profit_factor = (sum(wins) / abs(sum(losses))) if losses and sum(losses) != 0 else 0.0
        
        return hit_rate, avg_win, avg_loss, profit_factor
    
    def _calculate_drawdown_metrics(
        self,
        equity_curve: np.ndarray | pd.Series,
        timestamps: Optional[pd.Series]
    ) -> tuple[float, float, float]:
        """Calculate drawdown metrics."""
        equity_array = np.array(equity_curve) if not isinstance(equity_curve, np.ndarray) else equity_curve
        if len(equity_array) == 0:
            return 0.0, 0.0, 0.0
        
        cumulative = equity_array / equity_array[0] if equity_array[0] != 0 else equity_array
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        
        max_dd = abs(float(np.min(drawdown))) if len(drawdown) > 0 else 0.0
        avg_dd = abs(float(np.mean(drawdown[drawdown < 0]))) if np.any(drawdown < 0) else 0.0
        
        # Time to recover (simplified)
        time_to_recover = 0.0  # Would need more complex calculation
        
        return max_dd, avg_dd, time_to_recover
    
    def _calculate_turnover_metrics(
        self,
        trades: List[Dict[str, any]],
        timestamps: Optional[pd.Series]
    ) -> tuple[float, float, float]:
        """Calculate turnover metrics."""
        if not trades:
            return 0.0, 0.0, 0.0
        
        total_volume = sum(abs(t.get('size_usd', 0.0)) for t in trades)
        equity = 100000.0  # Default, should be passed in
        turnover = total_volume / equity if equity > 0 else 0.0
        
        # Trades per day
        if timestamps is not None and len(timestamps) > 1:
            days = (timestamps.iloc[-1] - timestamps.iloc[0]).days
            trades_per_day = len(trades) / days if days > 0 else 0.0
        else:
            trades_per_day = len(trades) / 30.0  # Assume 30 days
        
        # Average holding time
        holding_times = [
            (t.get('exit_time', 0) - t.get('entry_time', 0)) / 3600
            for t in trades
            if 'exit_time' in t and 'entry_time' in t
        ]
        avg_holding_time = np.mean(holding_times) if holding_times else 0.0
        
        return turnover, trades_per_day, avg_holding_time
    
    def _calculate_cost_shares(
        self,
        returns: np.ndarray,
        costs: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate cost shares of gross alpha."""
        gross_alpha = np.sum(returns) + sum(costs.values())
        if gross_alpha == 0:
            return {'slippage': 0.0, 'fees': 0.0, 'funding': 0.0, 'total': 0.0}
        
        slippage_share = costs.get('slippage', 0.0) / gross_alpha
        fee_share = costs.get('fees', 0.0) / gross_alpha
        funding_share = costs.get('funding', 0.0) / gross_alpha
        total_share = sum(costs.values()) / gross_alpha
        
        return {
            'slippage': slippage_share,
            'fees': fee_share,
            'funding': funding_share,
            'total': total_share,
        }
    
    def _estimate_capacity(
        self,
        returns: np.ndarray,
        trades: List[Dict[str, any]]
    ) -> tuple[float, float]:
        """Estimate capacity and PnL drop at scale."""
        # Simplified capacity estimation
        # In practice, would use more sophisticated models
        avg_trade_size = np.mean([abs(t.get('size_usd', 0.0)) for t in trades]) if trades else 0.0
        capacity = avg_trade_size * 100  # Placeholder
        
        # PnL drop at scale (simplified)
        pnl_drop = 0.1  # Assume 10% drop
        
        return capacity, pnl_drop
    
    def _calculate_stability(
        self,
        returns: pd.Series | np.ndarray,
        timestamps: Optional[pd.Series],
        trades: List[Dict[str, any]]
    ) -> tuple[Dict[str, float], Dict[str, float], float, float, float]:
        """Calculate stability metrics (dispersion)."""
        sharpe_by_year = {}
        sharpe_by_coin = {}
        
        # Calculate Sharpe by year if timestamps available
        if timestamps is not None and len(timestamps) > 0:
            if isinstance(returns, pd.Series):
                returns_df = pd.DataFrame({'returns': returns, 'year': timestamps.dt.year})
                for year, year_returns in returns_df.groupby('year')['returns']:
                    sharpe_by_year[str(year)] = self._calculate_sharpe(year_returns.values)
        
        # Calculate Sharpe by coin
        if trades:
            coin_returns = {}
            for trade in trades:
                coin = trade.get('symbol', 'unknown')
                pnl = trade.get('pnl', 0.0)
                if coin not in coin_returns:
                    coin_returns[coin] = []
                coin_returns[coin].append(pnl)
            
            for coin, coin_pnls in coin_returns.items():
                if len(coin_pnls) > 10:  # Minimum trades
                    sharpe_by_coin[coin] = self._calculate_sharpe(np.array(coin_pnls))
        
        # Calculate dispersion
        all_sharpes = list(sharpe_by_year.values()) + list(sharpe_by_coin.values())
        if all_sharpes:
            sharpe_std = float(np.std(all_sharpes))
            sharpe_min = float(np.min(all_sharpes))
            sharpe_max = float(np.max(all_sharpes))
        else:
            sharpe_std = 0.0
            sharpe_min = 0.0
            sharpe_max = 0.0
        
        return sharpe_by_year, sharpe_by_coin, sharpe_std, sharpe_min, sharpe_max
    
    def generate_report(self, metrics: BacktestMetrics) -> str:
        """
        Generate formatted backtest report.
        
        Args:
            metrics: BacktestMetrics
        
        Returns:
            Formatted report string
        """
        report = f"""
=== BACKTEST REPORT ===

RETURNS:
  Net Sharpe:        {metrics.net_sharpe:.2f}
  Gross Sharpe:      {metrics.gross_sharpe:.2f}
  Sortino Ratio:     {metrics.sortino_ratio:.2f}
  Calmar Ratio:      {metrics.calmar_ratio:.2f}

TRADE STATISTICS:
  Total Trades:      {metrics.total_trades}
  Hit Rate:          {metrics.hit_rate:.2%}
  Avg Win:           ${metrics.avg_win:.2f}
  Avg Loss:          ${metrics.avg_loss:.2f}
  Profit Factor:     {metrics.profit_factor:.2f}

DRAWDOWN:
  Max Drawdown:      {metrics.max_drawdown:.2%}
  Avg Drawdown:      {metrics.avg_drawdown:.2%}
  Time to Recover:   {metrics.time_to_recover_days:.1f} days

TURNOVER:
  Turnover:          {metrics.turnover:.2f}
  Trades/Day:        {metrics.trades_per_day:.2f}
  Avg Holding Time:  {metrics.avg_holding_time_hours:.1f} hours

COSTS:
  Slippage Share:    {metrics.slippage_share_of_gross_alpha:.2%}
  Fee Share:         {metrics.fee_share_of_gross_alpha:.2%}
  Funding Share:     {metrics.funding_share_of_gross_alpha:.2%}
  Total Cost Share:  {metrics.total_cost_share:.2%}

CAPACITY:
  Estimate:          ${metrics.capacity_estimate_usd:,.0f}
  PnL Drop at Scale: {metrics.pnl_drop_at_scale_pct:.1%}

STABILITY:
  Sharpe Std:        {metrics.sharpe_std:.2f}
  Sharpe Min:        {metrics.sharpe_min:.2f}
  Sharpe Max:        {metrics.sharpe_max:.2f}
"""
        return report

