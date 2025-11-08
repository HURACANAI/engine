"""
Counterfactual Evaluator

Re-runs exits and sizes nightly. Updates exit rules from regret analysis.

Key Features:
- Counterfactual analysis
- Regret calculation
- Exit rule optimization
- Size optimization
- Nightly re-evaluation
- Historical what-if analysis

Author: Huracan Engine Team
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
from datetime import datetime, timezone

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


class RegretType(Enum):
    """Regret type"""
    EXIT_TIMING = "exit_timing"
    POSITION_SIZE = "position_size"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"


@dataclass
class CounterfactualTrade:
    """Counterfactual trade analysis"""
    trade_id: str
    actual_exit_price: float
    actual_exit_time: datetime
    actual_pnl: float
    optimal_exit_price: float
    optimal_exit_time: datetime
    optimal_pnl: float
    regret: float
    regret_type: RegretType
    actual_size: float
    optimal_size: float
    size_regret: float


@dataclass
class ExitRuleRecommendation:
    """Exit rule recommendation"""
    rule_type: str  # "stop_loss", "take_profit", "time_based"
    current_value: float
    recommended_value: float
    expected_improvement: float
    confidence: float
    metadata: Dict[str, any] = field(default_factory=dict)


@dataclass
class CounterfactualReport:
    """Counterfactual evaluation report"""
    trade_analyses: List[CounterfactualTrade]
    exit_recommendations: List[ExitRuleRecommendation]
    total_regret: float
    avg_regret_per_trade: float
    improvement_potential: float
    metadata: Dict[str, any] = field(default_factory=dict)


class CounterfactualEvaluator:
    """
    Counterfactual Evaluator.
    
    Re-runs exits and sizes nightly to update exit rules from regret analysis.
    
    Usage:
        evaluator = CounterfactualEvaluator()
        
        # Analyze trades
        report = evaluator.analyze_trades(
            trades=[...],
            price_history={...}
        )
        
        # Get exit rule recommendations
        recommendations = report.exit_recommendations
        
        # Update exit rules based on recommendations
        for rec in recommendations:
            if rec.confidence > 0.7:
                update_exit_rule(rec.rule_type, rec.recommended_value)
    """
    
    def __init__(
        self,
        lookback_days: int = 30,  # Lookback period for analysis
        min_trades_for_analysis: int = 10,  # Minimum trades for analysis
        confidence_threshold: float = 0.7  # Confidence threshold for recommendations
    ):
        """
        Initialize counterfactual evaluator.
        
        Args:
            lookback_days: Lookback period for analysis
            min_trades_for_analysis: Minimum trades for analysis
            confidence_threshold: Confidence threshold for recommendations
        """
        self.lookback_days = lookback_days
        self.min_trades_for_analysis = min_trades_for_analysis
        self.confidence_threshold = confidence_threshold
        
        # Store trade history
        self.trade_history: List[Dict[str, any]] = []
        
        logger.info(
            "counterfactual_evaluator_initialized",
            lookback_days=lookback_days,
            min_trades_for_analysis=min_trades_for_analysis
        )
    
    def analyze_trades(
        self,
        trades: List[Dict[str, any]],
        price_history: Dict[str, List[Tuple[datetime, float]]]  # symbol -> [(timestamp, price), ...]
    ) -> CounterfactualReport:
        """
        Analyze trades for counterfactual scenarios.
        
        Args:
            trades: List of trade dictionaries
            price_history: Price history for symbols
        
        Returns:
            CounterfactualReport
        """
        trade_analyses = []
        exit_timing_regrets = []
        size_regrets = []
        
        for trade in trades:
            analysis = self._analyze_trade(trade, price_history)
            if analysis:
                trade_analyses.append(analysis)
                exit_timing_regrets.append(analysis.regret)
                size_regrets.append(analysis.size_regret)
        
        # Calculate total regret
        total_regret = sum(analysis.regret for analysis in trade_analyses)
        avg_regret_per_trade = total_regret / len(trade_analyses) if trade_analyses else 0.0
        
        # Generate exit rule recommendations
        exit_recommendations = self._generate_exit_recommendations(trade_analyses)
        
        # Calculate improvement potential
        improvement_potential = self._calculate_improvement_potential(trade_analyses)
        
        report = CounterfactualReport(
            trade_analyses=trade_analyses,
            exit_recommendations=exit_recommendations,
            total_regret=total_regret,
            avg_regret_per_trade=avg_regret_per_trade,
            improvement_potential=improvement_potential,
            metadata={
                "num_trades_analyzed": len(trade_analyses),
                "lookback_days": self.lookback_days
            }
        )
        
        logger.info(
            "counterfactual_analysis_complete",
            num_trades=len(trade_analyses),
            total_regret=total_regret,
            avg_regret=avg_regret_per_trade,
            improvement_potential=improvement_potential
        )
        
        return report
    
    def _analyze_trade(
        self,
        trade: Dict[str, any],
        price_history: Dict[str, List[Tuple[datetime, float]]]
    ) -> Optional[CounterfactualTrade]:
        """Analyze a single trade for counterfactual scenarios"""
        symbol = trade.get("symbol")
        if symbol not in price_history:
            return None
        
        # Get trade details
        entry_price = trade.get("entry_price")
        entry_time = trade.get("entry_time")
        exit_price = trade.get("exit_price")
        exit_time = trade.get("exit_time")
        direction = trade.get("direction", "buy")
        size = trade.get("size", 1.0)
        
        if not all([entry_price, entry_time, exit_price, exit_time]):
            return None
        
        # Get price history for symbol
        prices = price_history[symbol]
        if not prices:
            return None
        
        # Find optimal exit
        optimal_exit_price, optimal_exit_time = self._find_optimal_exit(
            entry_price=entry_price,
            entry_time=entry_time,
            exit_time=exit_time,
            direction=direction,
            prices=prices
        )
        
        # Calculate actual and optimal P&L
        if direction == "buy":
            actual_pnl = (exit_price - entry_price) / entry_price
            optimal_pnl = (optimal_exit_price - entry_price) / entry_price
        else:
            actual_pnl = (entry_price - exit_price) / entry_price
            optimal_pnl = (entry_price - optimal_exit_price) / entry_price
        
        # Calculate regret
        regret = optimal_pnl - actual_pnl
        
        # Calculate optimal size (simplified)
        optimal_size = self._calculate_optimal_size(trade, prices)
        size_regret = abs(optimal_size - size) / size if size > 0 else 0.0
        
        # Determine regret type
        regret_type = self._determine_regret_type(regret, trade)
        
        return CounterfactualTrade(
            trade_id=trade.get("trade_id", "unknown"),
            actual_exit_price=exit_price,
            actual_exit_time=exit_time,
            actual_pnl=actual_pnl,
            optimal_exit_price=optimal_exit_price,
            optimal_exit_time=optimal_exit_time,
            optimal_pnl=optimal_pnl,
            regret=regret,
            regret_type=regret_type,
            actual_size=size,
            optimal_size=optimal_size,
            size_regret=size_regret
        )
    
    def _find_optimal_exit(
        self,
        entry_price: float,
        entry_time: datetime,
        exit_time: datetime,
        direction: str,
        prices: List[Tuple[datetime, float]]
    ) -> Tuple[float, datetime]:
        """Find optimal exit price and time"""
        # Filter prices between entry and exit
        relevant_prices = [
            (ts, price) for ts, price in prices
            if entry_time <= ts <= exit_time
        ]
        
        if not relevant_prices:
            return entry_price, entry_time
        
        # Find best exit
        if direction == "buy":
            # Find highest price
            best_price, best_time = max(relevant_prices, key=lambda x: x[1])
        else:
            # Find lowest price
            best_price, best_time = min(relevant_prices, key=lambda x: x[1])
        
        return best_price, best_time
    
    def _calculate_optimal_size(
        self,
        trade: Dict[str, any],
        prices: List[Tuple[datetime, float]]
    ) -> float:
        """Calculate optimal position size (simplified)"""
        # Simplified: use Kelly criterion or volatility-based sizing
        # In production, would use more sophisticated methods
        
        current_size = trade.get("size", 1.0)
        volatility = trade.get("volatility", 0.02)
        
        # Kelly fraction (simplified)
        win_rate = trade.get("win_rate", 0.5)
        avg_win = trade.get("avg_win", 0.02)
        avg_loss = trade.get("avg_loss", -0.01)
        
        if abs(avg_loss) > 0:
            kelly_fraction = (win_rate * avg_win - (1 - win_rate) * abs(avg_loss)) / avg_win
            kelly_fraction = max(0.0, min(1.0, kelly_fraction))
        else:
            kelly_fraction = 0.5
        
        # Volatility-based sizing
        vol_scaling = 1.0 / (1.0 + volatility * 10)
        
        optimal_size = current_size * kelly_fraction * vol_scaling
        
        return float(optimal_size)
    
    def _determine_regret_type(
        self,
        regret: float,
        trade: Dict[str, any]
    ) -> RegretType:
        """Determine type of regret"""
        if regret > 0.02:  # Significant regret
            # Check if it's exit timing or size
            if trade.get("exit_reason") == "stop_loss":
                return RegretType.STOP_LOSS
            elif trade.get("exit_reason") == "take_profit":
                return RegretType.TAKE_PROFIT
            else:
                return RegretType.EXIT_TIMING
        else:
            return RegretType.POSITION_SIZE
    
    def _generate_exit_recommendations(
        self,
        trade_analyses: List[CounterfactualTrade]
    ) -> List[ExitRuleRecommendation]:
        """Generate exit rule recommendations"""
        if len(trade_analyses) < self.min_trades_for_analysis:
            return []
        
        recommendations = []
        
        # Analyze stop loss regrets
        stop_loss_regrets = [t for t in trade_analyses if t.regret_type == RegretType.STOP_LOSS]
        if stop_loss_regrets:
            rec = self._recommend_stop_loss(stop_loss_regrets)
            if rec:
                recommendations.append(rec)
        
        # Analyze take profit regrets
        take_profit_regrets = [t for t in trade_analyses if t.regret_type == RegretType.TAKE_PROFIT]
        if take_profit_regrets:
            rec = self._recommend_take_profit(take_profit_regrets)
            if rec:
                recommendations.append(rec)
        
        # Analyze exit timing regrets
        exit_timing_regrets = [t for t in trade_analyses if t.regret_type == RegretType.EXIT_TIMING]
        if exit_timing_regrets:
            rec = self._recommend_exit_timing(exit_timing_regrets)
            if rec:
                recommendations.append(rec)
        
        return recommendations
    
    def _recommend_stop_loss(self, regrets: List[CounterfactualTrade]) -> Optional[ExitRuleRecommendation]:
        """Recommend stop loss adjustment"""
        if not regrets:
            return None
        
        # Calculate average improvement from optimal exits
        avg_improvement = np.mean([t.optimal_pnl - t.actual_pnl for t in regrets])
        
        # Recommend wider stop loss if regrets are high
        current_stop_loss = 0.02  # 2% default (would come from config)
        recommended_stop_loss = current_stop_loss * (1 + avg_improvement * 0.5)
        
        confidence = min(1.0, len(regrets) / 20.0)
        
        return ExitRuleRecommendation(
            rule_type="stop_loss",
            current_value=current_stop_loss,
            recommended_value=recommended_stop_loss,
            expected_improvement=avg_improvement,
            confidence=confidence,
            metadata={"num_trades": len(regrets)}
        )
    
    def _recommend_take_profit(self, regrets: List[CounterfactualTrade]) -> Optional[ExitRuleRecommendation]:
        """Recommend take profit adjustment"""
        if not regrets:
            return None
        
        # Calculate average improvement
        avg_improvement = np.mean([t.optimal_pnl - t.actual_pnl for t in regrets])
        
        # Recommend higher take profit if optimal exits were better
        current_take_profit = 0.05  # 5% default
        recommended_take_profit = current_take_profit * (1 + avg_improvement * 0.3)
        
        confidence = min(1.0, len(regrets) / 20.0)
        
        return ExitRuleRecommendation(
            rule_type="take_profit",
            current_value=current_take_profit,
            recommended_value=recommended_take_profit,
            expected_improvement=avg_improvement,
            confidence=confidence,
            metadata={"num_trades": len(regrets)}
        )
    
    def _recommend_exit_timing(self, regrets: List[CounterfactualTrade]) -> Optional[ExitRuleRecommendation]:
        """Recommend exit timing adjustment"""
        if not regrets:
            return None
        
        # Calculate average time difference
        time_diffs = [
            (t.optimal_exit_time - t.actual_exit_time).total_seconds() / 3600
            for t in regrets
        ]
        
        avg_time_diff = np.mean(time_diffs)
        
        # Recommend holding longer if optimal exits were later
        current_hold_time = 24.0  # 24 hours default
        recommended_hold_time = current_hold_time + avg_time_diff
        
        confidence = min(1.0, len(regrets) / 20.0)
        
        return ExitRuleRecommendation(
            rule_type="time_based",
            current_value=current_hold_time,
            recommended_value=recommended_hold_time,
            expected_improvement=np.mean([t.regret for t in regrets]),
            confidence=confidence,
            metadata={"num_trades": len(regrets), "avg_time_diff_hours": avg_time_diff}
        )
    
    def _calculate_improvement_potential(self, trade_analyses: List[CounterfactualTrade]) -> float:
        """Calculate improvement potential"""
        if not trade_analyses:
            return 0.0
        
        total_actual_pnl = sum(t.actual_pnl for t in trade_analyses)
        total_optimal_pnl = sum(t.optimal_pnl for t in trade_analyses)
        
        if total_actual_pnl == 0:
            return 0.0
        
        improvement_potential = (total_optimal_pnl - total_actual_pnl) / abs(total_actual_pnl)
        
        return float(improvement_potential)

