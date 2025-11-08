"""
Daily Win/Loss Analytics

Analyzes wins and losses to auto-update risk presets and cooldowns.
Generates daily reports with actionable insights.

Key Features:
- Win/loss analysis by regime, feature bins, error types
- Calibration analysis (bucket predictions by confidence)
- Slippage mapping by pair and hour of day
- Auto-update risk presets based on performance
- Cooldown logic after streak losses
- Daily analytics reports

Author: Huracan Engine Team
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone, timedelta
from enum import Enum

import numpy as np
import polars as pl
import structlog

logger = structlog.get_logger(__name__)


class ErrorType(Enum):
    """Error type"""
    DIRECTION_WRONG = "direction_wrong"
    TIMING_LATE = "timing_late"
    STOP_TOO_TIGHT = "stop_too_tight"
    SPREAD_BLOWOUT = "spread_blowout"
    NEWS_SPIKE = "news_spike"
    LOW_LIQUIDITY = "low_liquidity"
    REGIME_CHANGE = "regime_change"
    NONE = "none"


@dataclass
class WinLossAnalysis:
    """Win/loss analysis result"""
    date: datetime.date
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    avg_winner: float
    avg_loser: float
    profit_factor: float
    payoff_ratio: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    hit_rate_by_regime: Dict[str, float] = field(default_factory=dict)
    hit_rate_by_error_type: Dict[str, float] = field(default_factory=dict)
    slippage_by_pair: Dict[str, float] = field(default_factory=dict)
    slippage_by_hour: Dict[int, float] = field(default_factory=dict)


@dataclass
class CalibrationAnalysis:
    """Calibration analysis result"""
    confidence_buckets: List[Tuple[float, float]]  # (min, max) confidence ranges
    avg_pnl_per_bucket: Dict[Tuple[float, float], float]
    predicted_vs_actual: Dict[Tuple[float, float], float]  # Predicted win rate vs actual
    calibration_error: float


@dataclass
class RiskPresetUpdate:
    """Risk preset update recommendation"""
    preset_name: str
    current_size: float
    recommended_size: float
    reason: str
    confidence: float


class DailyWinLossAnalytics:
    """
    Daily Win/Loss Analytics.
    
    Analyzes wins and losses to generate insights and update risk presets.
    
    Usage:
        analytics = DailyWinLossAnalytics()
        
        # Analyze trades
        analysis = analytics.analyze_trades(trades, predictions, date)
        
        # Generate calibration analysis
        calibration = analytics.analyze_calibration(predictions, trades)
        
        # Generate risk preset updates
        updates = analytics.generate_risk_preset_updates(analysis, calibration)
    """
    
    def __init__(
        self,
        confidence_buckets: List[Tuple[float, float]] = None,
        min_trades_for_analysis: int = 10
    ):
        """
        Initialize daily win/loss analytics.
        
        Args:
            confidence_buckets: Confidence buckets for calibration (default: 0-0.5, 0.5-0.7, 0.7-0.9, 0.9-1.0)
            min_trades_for_analysis: Minimum trades for analysis
        """
        self.confidence_buckets = confidence_buckets or [
            (0.0, 0.5),
            (0.5, 0.7),
            (0.7, 0.9),
            (0.9, 1.0)
        ]
        self.min_trades_for_analysis = min_trades_for_analysis
        
        logger.info("daily_win_loss_analytics_initialized")
    
    def analyze_trades(
        self,
        trades: List[Dict[str, Any]],
        predictions: List[Dict[str, Any]],
        date: datetime.date
    ) -> WinLossAnalysis:
        """
        Analyze trades for a given date.
        
        Args:
            trades: List of trade records
            predictions: List of prediction records
            date: Analysis date
        
        Returns:
            WinLossAnalysis
        """
        if len(trades) < self.min_trades_for_analysis:
            logger.warning(
                "insufficient_trades_for_analysis",
                date=date,
                trades=len(trades),
                min_required=self.min_trades_for_analysis
            )
            return WinLossAnalysis(
                date=date,
                total_trades=len(trades),
                winning_trades=0,
                losing_trades=0,
                win_rate=0.0,
                total_pnl=0.0,
                avg_winner=0.0,
                avg_loser=0.0,
                profit_factor=0.0,
                payoff_ratio=0.0,
                sharpe_ratio=0.0,
                sortino_ratio=0.0,
                max_drawdown=0.0
            )
        
        # Filter trades for date
        date_trades = [t for t in trades if self._get_trade_date(t) == date]
        
        if not date_trades:
            return WinLossAnalysis(
                date=date,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0.0,
                total_pnl=0.0,
                avg_winner=0.0,
                avg_loser=0.0,
                profit_factor=0.0,
                payoff_ratio=0.0,
                sharpe_ratio=0.0,
                sortino_ratio=0.0,
                max_drawdown=0.0
            )
        
        # Calculate basic metrics
        winning_trades = [t for t in date_trades if t.get("pnl_after_costs", 0) > 0]
        losing_trades = [t for t in date_trades if t.get("pnl_after_costs", 0) <= 0]
        
        total_pnl = sum(t.get("pnl_after_costs", 0) for t in date_trades)
        avg_winner = np.mean([t.get("pnl_after_costs", 0) for t in winning_trades]) if winning_trades else 0.0
        avg_loser = np.mean([t.get("pnl_after_costs", 0) for t in losing_trades]) if losing_trades else 0.0
        
        profit_factor = abs(avg_winner / avg_loser) if avg_loser != 0 else 0.0
        payoff_ratio = avg_winner / abs(avg_loser) if avg_loser != 0 else 0.0
        
        # Calculate Sharpe and Sortino
        returns = [t.get("pnl_after_costs", 0) for t in date_trades]
        sharpe_ratio = self._calculate_sharpe(returns)
        sortino_ratio = self._calculate_sortino(returns)
        max_drawdown = self._calculate_max_drawdown(returns)
        
        # Analyze by regime
        hit_rate_by_regime = self._analyze_by_regime(date_trades, predictions)
        
        # Analyze by error type
        hit_rate_by_error_type = self._analyze_by_error_type(date_trades)
        
        # Analyze slippage
        slippage_by_pair = self._analyze_slippage_by_pair(date_trades)
        slippage_by_hour = self._analyze_slippage_by_hour(date_trades)
        
        return WinLossAnalysis(
            date=date,
            total_trades=len(date_trades),
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            win_rate=len(winning_trades) / len(date_trades),
            total_pnl=total_pnl,
            avg_winner=avg_winner,
            avg_loser=avg_loser,
            profit_factor=profit_factor,
            payoff_ratio=payoff_ratio,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            hit_rate_by_regime=hit_rate_by_regime,
            hit_rate_by_error_type=hit_rate_by_error_type,
            slippage_by_pair=slippage_by_pair,
            slippage_by_hour=slippage_by_hour
        )
    
    def analyze_calibration(
        self,
        predictions: List[Dict[str, Any]],
        trades: List[Dict[str, Any]]
    ) -> CalibrationAnalysis:
        """
        Analyze prediction calibration.
        
        Args:
            predictions: List of prediction records
            trades: List of trade records
        
        Returns:
            CalibrationAnalysis
        """
        # Create prediction -> trade mapping
        pred_to_trade = {t.get("pred_id"): t for t in trades if "pred_id" in t}
        
        # Group predictions by confidence bucket
        bucket_trades: Dict[Tuple[float, float], List[float]] = {}
        bucket_predicted_rates: Dict[Tuple[float, float], List[float]] = {}
        
        for pred in predictions:
            confidence = pred.get("predicted_confidence", 0.0)
            pred_id = pred.get("pred_id")
            
            # Find confidence bucket
            bucket = self._get_confidence_bucket(confidence)
            if bucket is None:
                continue
            
            # Get corresponding trade
            trade = pred_to_trade.get(pred_id)
            if not trade:
                continue
            
            # Record PnL and predicted win rate
            pnl = trade.get("pnl_after_costs", 0.0)
            if bucket not in bucket_trades:
                bucket_trades[bucket] = []
                bucket_predicted_rates[bucket] = []
            
            bucket_trades[bucket].append(pnl)
            bucket_predicted_rates[bucket].append(confidence)
        
        # Calculate average PnL per bucket
        avg_pnl_per_bucket = {
            bucket: np.mean(pnls) if pnls else 0.0
            for bucket, pnls in bucket_trades.items()
        }
        
        # Calculate predicted vs actual win rate
        predicted_vs_actual = {}
        for bucket, pnls in bucket_trades.items():
            predicted_rate = np.mean(bucket_predicted_rates.get(bucket, [0.0]))
            actual_rate = len([p for p in pnls if p > 0]) / len(pnls) if pnls else 0.0
            predicted_vs_actual[bucket] = actual_rate - predicted_rate
        
        # Calculate calibration error (MSE)
        calibration_error = np.mean([
            (predicted_vs_actual[bucket]) ** 2
            for bucket in predicted_vs_actual.keys()
        ])
        
        return CalibrationAnalysis(
            confidence_buckets=self.confidence_buckets,
            avg_pnl_per_bucket=avg_pnl_per_bucket,
            predicted_vs_actual=predicted_vs_actual,
            calibration_error=calibration_error
        )
    
    def generate_risk_preset_updates(
        self,
        analysis: WinLossAnalysis,
        calibration: CalibrationAnalysis,
        current_presets: Dict[str, float]
    ) -> List[RiskPresetUpdate]:
        """
        Generate risk preset update recommendations.
        
        Args:
            analysis: Win/loss analysis
            calibration: Calibration analysis
            current_presets: Current risk presets
        
        Returns:
            List of RiskPresetUpdate recommendations
        """
        updates = []
        
        # Reduce size if win rate is low
        if analysis.win_rate < 0.4:
            for preset_name, current_size in current_presets.items():
                recommended_size = current_size * 0.8  # Reduce by 20%
                updates.append(RiskPresetUpdate(
                    preset_name=preset_name,
                    current_size=current_size,
                    recommended_size=recommended_size,
                    reason=f"Low win rate: {analysis.win_rate:.2%}",
                    confidence=0.8
                ))
        
        # Reduce size if profit factor is low
        if analysis.profit_factor < 1.0:
            for preset_name, current_size in current_presets.items():
                recommended_size = current_size * 0.9  # Reduce by 10%
                updates.append(RiskPresetUpdate(
                    preset_name=preset_name,
                    current_size=current_size,
                    recommended_size=recommended_size,
                    reason=f"Low profit factor: {analysis.profit_factor:.2f}",
                    confidence=0.7
                ))
        
        # Increase size if win rate and profit factor are high
        if analysis.win_rate > 0.6 and analysis.profit_factor > 1.5:
            for preset_name, current_size in current_presets.items():
                recommended_size = current_size * 1.1  # Increase by 10%
                updates.append(RiskPresetUpdate(
                    preset_name=preset_name,
                    current_size=current_size,
                    recommended_size=recommended_size,
                    reason=f"High win rate: {analysis.win_rate:.2%}, Profit factor: {analysis.profit_factor:.2f}",
                    confidence=0.6
                ))
        
        return updates
    
    def _get_trade_date(self, trade: Dict[str, Any]) -> datetime.date:
        """Get trade date from trade record"""
        timestamp = trade.get("timestamp_open")
        if isinstance(timestamp, datetime):
            return timestamp.date()
        elif isinstance(timestamp, str):
            return datetime.fromisoformat(timestamp).date()
        else:
            return datetime.now(timezone.utc).date()
    
    def _calculate_sharpe(self, returns: List[float]) -> float:
        """Calculate Sharpe ratio"""
        if not returns or len(returns) < 2:
            return 0.0
        returns_array = np.array(returns)
        mean_return = np.mean(returns_array)
        std_return = np.std(returns_array)
        return mean_return / (std_return + 1e-9) * np.sqrt(252)  # Annualized
    
    def _calculate_sortino(self, returns: List[float]) -> float:
        """Calculate Sortino ratio"""
        if not returns or len(returns) < 2:
            return 0.0
        returns_array = np.array(returns)
        mean_return = np.mean(returns_array)
        downside_returns = returns_array[returns_array < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 1e-9
        return mean_return / downside_std * np.sqrt(252)  # Annualized
    
    def _calculate_max_drawdown(self, returns: List[float]) -> float:
        """Calculate maximum drawdown"""
        if not returns:
            return 0.0
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = cumulative - running_max
        return abs(np.min(drawdown))
    
    def _analyze_by_regime(
        self,
        trades: List[Dict[str, Any]],
        predictions: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Analyze hit rate by regime"""
        # Create prediction -> trade mapping
        pred_to_trade = {t.get("pred_id"): t for t in trades if "pred_id" in t}
        pred_dict = {p.get("pred_id"): p for p in predictions}
        
        regime_trades: Dict[str, List[float]] = {}
        
        for trade in trades:
            pred_id = trade.get("pred_id")
            pred = pred_dict.get(pred_id)
            if not pred:
                continue
            
            regime = pred.get("regime", "unknown")
            pnl = trade.get("pnl_after_costs", 0.0)
            
            if regime not in regime_trades:
                regime_trades[regime] = []
            regime_trades[regime].append(pnl)
        
        # Calculate hit rate per regime
        hit_rate_by_regime = {}
        for regime, pnls in regime_trades.items():
            wins = len([p for p in pnls if p > 0])
            hit_rate_by_regime[regime] = wins / len(pnls) if pnls else 0.0
        
        return hit_rate_by_regime
    
    def _analyze_by_error_type(self, trades: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analyze hit rate by error type"""
        error_trades: Dict[str, List[float]] = {}
        
        for trade in trades:
            error_type = trade.get("error_type", "none")
            pnl = trade.get("pnl_after_costs", 0.0)
            
            if error_type not in error_trades:
                error_trades[error_type] = []
            error_trades[error_type].append(pnl)
        
        # Calculate hit rate per error type
        hit_rate_by_error_type = {}
        for error_type, pnls in error_trades.items():
            wins = len([p for p in pnls if p > 0])
            hit_rate_by_error_type[error_type] = wins / len(pnls) if pnls else 0.0
        
        return hit_rate_by_error_type
    
    def _analyze_slippage_by_pair(self, trades: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analyze slippage by pair"""
        pair_slippage: Dict[str, List[float]] = {}
        
        for trade in trades:
            symbol = trade.get("symbol", "unknown")
            slippage = trade.get("slippage", 0.0)
            
            if symbol not in pair_slippage:
                pair_slippage[symbol] = []
            pair_slippage[symbol].append(slippage)
        
        # Calculate average slippage per pair
        avg_slippage_by_pair = {
            pair: np.mean(slippages) if slippages else 0.0
            for pair, slippages in pair_slippage.items()
        }
        
        return avg_slippage_by_pair
    
    def _analyze_slippage_by_hour(self, trades: List[Dict[str, Any]]) -> Dict[int, float]:
        """Analyze slippage by hour of day"""
        hour_slippage: Dict[int, List[float]] = {}
        
        for trade in trades:
            timestamp = trade.get("timestamp_open")
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp)
            elif not isinstance(timestamp, datetime):
                continue
            
            hour = timestamp.hour
            slippage = trade.get("slippage", 0.0)
            
            if hour not in hour_slippage:
                hour_slippage[hour] = []
            hour_slippage[hour].append(slippage)
        
        # Calculate average slippage per hour
        avg_slippage_by_hour = {
            hour: np.mean(slippages) if slippages else 0.0
            for hour, slippages in hour_slippage.items()
        }
        
        return avg_slippage_by_hour
    
    def _get_confidence_bucket(self, confidence: float) -> Optional[Tuple[float, float]]:
        """Get confidence bucket for a given confidence value"""
        for bucket_min, bucket_max in self.confidence_buckets:
            if bucket_min <= confidence < bucket_max:
                return (bucket_min, bucket_max)
        return None

