"""
Shadow trading engine that backtests EVERY possible trade on historical data
with strict no-lookahead enforcement.

This is the key component that makes the bot learn from all historical patterns.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import polars as pl
import structlog

from ..agents.rl_agent import RLTradingAgent, TradingAction, TradingState
from ..analyzers.loss_analyzer import LossAnalyzer
from ..analyzers.win_analyzer import WinAnalyzer
from ..memory.store import MemoryStore, TradeMemory
from ..models.confidence_scorer import ConfidenceScorer
from ..models.feature_importance_learner import FeatureImportanceLearner
from ..models.regime_detector import RegimeDetector
from ..services.costs import CostBreakdown, CostModel
from shared.features.recipe import FeatureRecipe

logger = structlog.get_logger(__name__)


@dataclass
class ShadowTradeResult:
    """Result of a single shadow trade."""
    entry_idx: int
    entry_timestamp: datetime
    entry_price: float
    exit_idx: int
    exit_timestamp: datetime
    exit_price: float
    exit_reason: str
    hold_duration_minutes: int
    gross_profit_bps: float
    net_profit_gbp: float
    is_winner: bool

    # Post-exit tracking
    best_exit_price: float  # Optimal price in future
    best_exit_idx: int
    missed_profit_gbp: float

    # Context
    entry_features: Dict[str, Any]
    entry_embedding: np.ndarray
    market_regime: str
    costs: CostBreakdown

    # NEW: Confidence scoring
    regime_confidence: float = 0.5  # Confidence in regime detection
    trade_confidence: float = 0.5  # Overall confidence in trade decision
    decision_reason: str = ""  # Human-readable explanation


@dataclass
class BacktestConfig:
    """Configuration for shadow trading backtest."""
    position_size_gbp: float = 1000.0
    max_hold_minutes: int = 120  # Max 2 hours
    stop_loss_bps: float = 15.0
    take_profit_bps: float = 20.0
    lookback_for_optimal_exit: int = 60  # Minutes to look ahead for best exit
    min_confidence_threshold: float = 0.52  # Minimum model confidence to enter


class ShadowTrader:
    """
    Executes shadow trades on historical data with no lookahead bias.

    Key principle: At each candle, the agent only knows data UP TO that point.
    It makes a decision, then we reveal what happens next to evaluate the decision.
    """

    def __init__(
        self,
        agent: RLTradingAgent,
        memory_store: MemoryStore,
        cost_model: CostModel,
        win_analyzer: WinAnalyzer,
        loss_analyzer: LossAnalyzer,
        feature_recipe: FeatureRecipe,
        config: Optional[BacktestConfig] = None,
    ):
        self.agent = agent
        self.memory = memory_store
        self.cost_model = cost_model
        self.win_analyzer = win_analyzer
        self.loss_analyzer = loss_analyzer
        self.feature_recipe = feature_recipe
        self.config = config or BacktestConfig()

        # Initialize regime detection and confidence scoring
        self.regime_detector = RegimeDetector()
        self.confidence_scorer = ConfidenceScorer(
            min_confidence_threshold=self.config.min_confidence_threshold,
        )

        # Initialize feature importance learning
        self.feature_importance_learner = FeatureImportanceLearner(
            ema_alpha=0.05,  # ~20-trade memory
            min_samples_for_confidence=30,
            top_k_features=10,
        )

        self.current_position: Optional[Dict[str, Any]] = None
        self.trades_today = 0
        self.wins_today = 0

    def backtest_symbol(
        self,
        symbol: str,
        historical_data: pl.DataFrame,
        training_mode: bool = True,
    ) -> List[ShadowTradeResult]:
        """
        Run shadow trading on historical data for a symbol.

        Args:
            symbol: Trading pair symbol
            historical_data: DataFrame with columns: ts, open, high, low, close, volume
            training_mode: If True, agent learns from trades; if False, just evaluates

        Returns:
            List of all trades executed
        """
        logger.info("shadow_trading_start", symbol=symbol, rows=historical_data.height)

        if historical_data.height < 100:
            logger.warning("insufficient_data", symbol=symbol)
            return []

        # Ensure sorted by time
        historical_data = historical_data.sort("ts")

        # Build features for entire dataset
        features_df = self.feature_recipe.build(historical_data)

        all_trades: List[ShadowTradeResult] = []

        # Walk forward through time - critical: no lookahead!
        for current_idx in range(100, features_df.height - self.config.lookback_for_optimal_exit):
            # Agent only sees data UP TO current_idx
            visible_data = features_df[:current_idx + 1]

            # Make decision based on visible data only
            if self.current_position is None:
                # No position - consider entry
                self._consider_entry(
                    symbol=symbol,
                    current_idx=current_idx,
                    visible_data=visible_data,
                    future_data=features_df[current_idx + 1:],  # Hidden from agent!
                    training_mode=training_mode,
                )

            else:
                # Have position - consider exit
                exit_result = self._consider_exit(
                    current_idx=current_idx,
                    visible_data=visible_data,
                    future_data=features_df[current_idx + 1:],  # Hidden from agent!
                    training_mode=training_mode,
                )

                if exit_result:
                    all_trades.append(exit_result)

                    # Update feature importance learning
                    self.feature_importance_learner.update(
                        features=exit_result.entry_features,
                        is_winner=exit_result.is_winner,
                        profit_bps=exit_result.gross_profit_bps,
                        timestamp=str(exit_result.entry_timestamp),
                    )

        # Get feature importance results
        importance_result = self.feature_importance_learner.get_feature_importance()

        logger.info(
            "shadow_trading_complete",
            symbol=symbol,
            total_trades=len(all_trades),
            wins=sum(1 for t in all_trades if t.is_winner),
            feature_importance_samples=importance_result.total_samples,
            top_win_feature=importance_result.top_win_features[0][0] if importance_result.top_win_features else None,
        )

        return all_trades

    def _consider_entry(
        self,
        symbol: str,
        current_idx: int,
        visible_data: pl.DataFrame,
        future_data: pl.DataFrame,
        training_mode: bool,
    ) -> Optional[ShadowTradeResult]:
        """Consider entering a new position."""

        # Extract current state (only from visible data!)
        current_row = visible_data.row(current_idx, named=True)

        # Build state for agent
        state = self._build_state(
            current_row=current_row,
            visible_data=visible_data,
            current_idx=current_idx,
            has_position=False,
            symbol=symbol,
        )

        # Agent decides: enter or not?
        action, _ = self.agent.select_action(state, deterministic=not training_mode)

        # Only enter if agent chooses entry action and we have confidence
        if action not in [TradingAction.ENTER_LONG_SMALL, TradingAction.ENTER_LONG_NORMAL, TradingAction.ENTER_LONG_LARGE]:
            self.agent.store_reward(reward=0.0, done=False)  # Neutral reward for skipping
            return None

        # Calculate confidence score for this trade
        # Get regime info (already calculated in _build_state)
        regime_str, regime_conf = self._classify_regime(visible_data, current_idx)

        # Check if regime matches historical best (simplified - would query pattern_library)
        regime_match = regime_str == "trend"  # Simplified assumption

        # Calculate confidence using our confidence scorer
        confidence_result = self.confidence_scorer.calculate_confidence(
            sample_count=max(1, int(state.similar_pattern_reliability * 100)),  # Estimate sample count
            best_score=state.similar_pattern_win_rate,
            runner_up_score=0.5,  # Simplified - would need actual runner-up from agent
            pattern_similarity=state.similar_pattern_reliability,
            pattern_reliability=state.similar_pattern_reliability,
            regime_match=regime_match,
            regime_confidence=regime_conf,
        )

        # Skip trade if confidence is too low
        if confidence_result.decision == "skip":
            logger.debug(
                "trade_skipped_low_confidence",
                confidence=confidence_result.confidence,
                reason=confidence_result.reason,
            )
            self.agent.store_reward(reward=-0.1, done=False)  # Penalty for ignoring low confidence
            return None

        # Execute entry
        entry_timestamp = current_row["ts"]
        entry_price = float(current_row["close"])

        # Position size based on action
        size_multiplier = {
            TradingAction.ENTER_LONG_SMALL: 0.5,
            TradingAction.ENTER_LONG_NORMAL: 1.0,
            TradingAction.ENTER_LONG_LARGE: 1.5,
        }[action]

        position_size_gbp = self.config.position_size_gbp * size_multiplier

        # Store position
        self.current_position = {
            "symbol": symbol,
            "entry_idx": current_idx,
            "entry_timestamp": entry_timestamp,
            "entry_price": entry_price,
            "entry_features": dict(current_row),
            "position_size_gbp": position_size_gbp,
            "entry_volatility_bps": float(current_row.get("realized_sigma_30", 0.0)) * 10000,
            "entry_spread_bps": state.spread_bps,
            "visible_data": visible_data,  # Store for regime detection
            "entry_state": state,
            "entry_action": action,
            "confidence_result": confidence_result,  # Store confidence info for trade result
        }

        logger.debug("shadow_entry", symbol=symbol, price=entry_price, idx=current_idx)

        # Store neutral reward until trade outcome is known
        self.agent.store_reward(reward=0.0, done=False)

        return None

    def _consider_exit(
        self,
        current_idx: int,
        visible_data: pl.DataFrame,
        future_data: pl.DataFrame,
        training_mode: bool,
    ) -> Optional[ShadowTradeResult]:
        """Consider exiting current position."""
        if not self.current_position:
            return None

        current_row = visible_data.row(current_idx, named=True)
        current_price = float(current_row["close"])
        entry_price = self.current_position["entry_price"]

        # Calculate current P&L
        pnl_bps = ((current_price - entry_price) / entry_price) * 10000

        hold_minutes = (current_row["ts"] - self.current_position["entry_timestamp"]).total_seconds() / 60

        # Agent decision (always gather feedback so experience stays aligned)
        symbol = self.current_position["symbol"]
        state = self._build_state(
            current_row=current_row,
            visible_data=visible_data,
            current_idx=current_idx,
            has_position=True,
            symbol=symbol,
            unrealized_pnl_bps=pnl_bps,
            hold_duration_minutes=int(hold_minutes),
        )

        action, _ = self.agent.select_action(state, deterministic=False)

        exit_reason: Optional[str] = None

        # Forced exits override agent decision
        if pnl_bps <= -self.config.stop_loss_bps:
            exit_reason = "STOP_LOSS"
        elif pnl_bps >= self.config.take_profit_bps:
            exit_reason = "TAKE_PROFIT"
        elif hold_minutes >= self.config.max_hold_minutes:
            exit_reason = "TIMEOUT"
        elif action == TradingAction.EXIT_POSITION:
            exit_reason = "MODEL_SIGNAL"

        if exit_reason is None:
            # Continue holding, reward is neutral this step
            self.agent.store_reward(reward=0.0, done=False)
            return None

        trade_result = self._execute_exit(current_idx, current_row, exit_reason, future_data)

        reward = self.agent.calculate_reward(
            action=self.current_position["entry_action"],
            profit_gbp=trade_result.net_profit_gbp,
            missed_profit_gbp=trade_result.missed_profit_gbp,
            hold_duration_minutes=trade_result.hold_duration_minutes,
            position_size_gbp=self.current_position["position_size_gbp"],
            costs=trade_result.costs,
            state=state,
        )

        self.agent.store_reward(reward=reward, done=True)

        # Update stats
        self.trades_today += 1
        if trade_result.is_winner:
            self.wins_today += 1

        # Persist analysis while position context still available
        if training_mode:
            self._store_and_analyze_trade(symbol, trade_result)

        # Clear position
        self.current_position = None

        return trade_result

    def _execute_exit(
        self,
        exit_idx: int,
        exit_row: Dict[str, Any],
        exit_reason: str,
        future_data: pl.DataFrame,
    ) -> ShadowTradeResult:
        """Execute exit and create trade result."""
        position = self.current_position
        if position is None:
            raise RuntimeError("_execute_exit called without active position")

        return self._simulate_trade_outcome(
            position=position,
            future_data=future_data,
            entry_idx=position["entry_idx"],
            exit_idx=exit_idx,
            exit_row=exit_row,
            visible_data=position["visible_data"],
            exit_reason=exit_reason,
        )

    def _simulate_trade_outcome(
        self,
        position: Dict[str, Any],
        future_data: pl.DataFrame,
        entry_idx: int,
        exit_idx: int,
        exit_row: Dict[str, Any],
        visible_data: pl.DataFrame,
        exit_reason: str,
    ) -> ShadowTradeResult:
        """
        Simulate trade outcome using future data (hidden from agent during decision).

        Also tracks what WOULD have been the optimal exit (for learning).
        """
        entry_price = position["entry_price"]
        exit_price = float(exit_row["close"])
        exit_timestamp = exit_row["ts"]

        # Calculate P&L
        gross_profit_bps = ((exit_price - entry_price) / entry_price) * 10000

        # Calculate costs
        costs = self.cost_model.estimate(
            taker_fee_bps=5.0,  # Assuming taker
            spread_bps=position["entry_spread_bps"],
            volatility_bps=position["entry_volatility_bps"],
            adv_quote=None,
        )

        net_profit_bps = gross_profit_bps - costs.total_costs_bps
        net_profit_gbp = (net_profit_bps / 10000) * position["position_size_gbp"]

        is_winner = net_profit_gbp > 0

        # Find optimal exit in future (for learning)
        best_exit_price, best_exit_idx = self._find_optimal_exit(
            future_data=future_data,
            entry_price=entry_price,
            max_candles=self.config.lookback_for_optimal_exit,
        )

        optimal_profit_bps = ((best_exit_price - entry_price) / entry_price) * 10000
        optimal_profit_gbp = (optimal_profit_bps / 10000) * position["position_size_gbp"]
        missed_profit_gbp = optimal_profit_gbp - net_profit_gbp

        # Create embedding (simplified - would use actual model embedding)
        entry_embedding = self._create_embedding(position["entry_features"])

        # Classify regime at entry
        regime_str, regime_conf = self._classify_regime(visible_data, entry_idx)

        # Get confidence info from position
        confidence_result = position.get("confidence_result")
        if confidence_result:
            trade_confidence = confidence_result.confidence
            decision_reason = confidence_result.reason
        else:
            # Fallback for legacy positions without confidence
            trade_confidence = 0.5
            decision_reason = "No confidence calculation available"

        hold_duration_minutes = int((exit_timestamp - position["entry_timestamp"]).total_seconds() / 60)

        return ShadowTradeResult(
            entry_idx=position["entry_idx"],
            entry_timestamp=position["entry_timestamp"],
            entry_price=entry_price,
            exit_idx=exit_idx,
            exit_timestamp=exit_timestamp,
            exit_price=exit_price,
            exit_reason=exit_reason,
            hold_duration_minutes=hold_duration_minutes,
            gross_profit_bps=gross_profit_bps,
            net_profit_gbp=net_profit_gbp,
            is_winner=is_winner,
            best_exit_price=best_exit_price,
            best_exit_idx=best_exit_idx,
            missed_profit_gbp=max(0, missed_profit_gbp),
            entry_features=entry["entry_features"],
            entry_embedding=entry_embedding,
            market_regime=regime_str,
            regime_confidence=regime_conf,
            trade_confidence=trade_confidence,
            decision_reason=decision_reason,
            costs=costs,
        )

    def _find_optimal_exit(
        self,
        future_data: pl.DataFrame,
        entry_price: float,
        max_candles: int,
    ) -> Tuple[float, int]:
        """Find the best exit price in future data (for learning, not trading)."""
        max_profit = -float('inf')
        best_price = entry_price
        best_idx = 0

        for idx in range(min(max_candles, future_data.height)):
            row = future_data.row(idx, named=True)
            high_price = float(row["high"])
            profit_bps = ((high_price - entry_price) / entry_price) * 10000

            if profit_bps > max_profit:
                max_profit = profit_bps
                best_price = high_price
                best_idx = idx

        return best_price, best_idx

    def _build_state(
        self,
        current_row: Dict[str, Any],
        visible_data: pl.DataFrame,
        current_idx: int,
        has_position: bool,
        symbol: str,
        unrealized_pnl_bps: float = 0.0,
        hold_duration_minutes: int = 0,
    ) -> TradingState:
        """Build TradingState from current market data and memory."""

        # Extract market features
        feature_names = [c for c in visible_data.columns if c not in ["ts", "open", "high", "low", "close", "volume"]]
        market_features = np.array([float(current_row.get(f, 0.0)) for f in feature_names], dtype=np.float32)

        # Create embedding and query memory
        embedding = self._create_embedding(current_row)
        similar_patterns = self.memory.find_similar_patterns(
            embedding=embedding,
            symbol=symbol,
            top_k=20,
            min_similarity=0.6,
        )

        pattern_stats = self.memory.get_pattern_stats(similar_patterns)

        # Classify regime using sophisticated regime detector
        volatility_bps = float(current_row.get("realized_sigma_30", 0.0)) * 10000
        regime_str, regime_confidence = self._classify_regime(visible_data, current_idx)

        # Convert to code for RL agent state (backward compatibility)
        regime_code = {
            "trend": 3,
            "range": 4,
            "panic": 2,
            "unknown": 1,
        }.get(regime_str, 1)

        return TradingState(
            market_features=market_features,
            similar_pattern_win_rate=pattern_stats.win_rate,
            similar_pattern_avg_profit=pattern_stats.avg_profit_gbp,
            similar_pattern_reliability=pattern_stats.reliability_score,
            has_position=has_position,
            position_size_multiplier=1.0 if has_position else 0.0,
            unrealized_pnl_bps=unrealized_pnl_bps,
            hold_duration_minutes=hold_duration_minutes,
            volatility_bps=volatility_bps,
            spread_bps=5.0,  # Simplified - would fetch real spread
            regime_code=regime_code,
            current_drawdown_gbp=0.0,  # Would track real drawdown
            trades_today=self.trades_today,
            win_rate_today=self.wins_today / self.trades_today if self.trades_today > 0 else 0.5,
        )

    def _create_embedding(self, features: Dict[str, Any]) -> np.ndarray:
        """Create vector embedding from features (simplified)."""
        # In production, use actual model's embedding layer
        feature_values = []
        for k, v in sorted(features.items()):
            if isinstance(v, (int, float)) and not np.isnan(v):
                feature_values.append(v)

        # Pad/truncate to 128 dimensions
        if len(feature_values) < 128:
            feature_values.extend([0.0] * (128 - len(feature_values)))
        else:
            feature_values = feature_values[:128]

        return np.array(feature_values, dtype=np.float32)

    def _classify_regime(self, features_df: pl.DataFrame, current_idx: int) -> tuple[str, float]:
        """
        Classify market regime using our sophisticated regime detector.

        Returns:
            tuple: (regime_name, confidence)
        """
        result = self.regime_detector.detect_regime(features_df, current_idx)
        return (result.regime.value, result.confidence)

    def _classify_regime_code(self, features: Dict[str, Any]) -> int:
        """Classify regime as integer code (kept for compatibility)."""
        # This is used by RL agent state - keep simple mapping for now
        regime_str = features.get("regime", "unknown")
        return {
            "trend": 3,
            "range": 4,
            "panic": 2,
            "unknown": 1,
        }.get(regime_str, 1)

    def _store_and_analyze_trade(self, symbol: str, trade: ShadowTradeResult) -> None:
        """Store trade in memory and run win/loss analysis."""

        # Store in memory
        trade_memory = TradeMemory(
            trade_id=None,
            symbol=symbol,
            entry_timestamp=trade.entry_timestamp,
            entry_price=trade.entry_price,
            entry_features=trade.entry_features,
            entry_embedding=trade.entry_embedding,
            position_size_gbp=self.config.position_size_gbp,
            direction="LONG",
            exit_timestamp=trade.exit_timestamp,
            exit_price=trade.exit_price,
            exit_reason=trade.exit_reason,
            hold_duration_minutes=trade.hold_duration_minutes,
            gross_profit_bps=trade.gross_profit_bps,
            net_profit_gbp=trade.net_profit_gbp,
            fees_gbp=trade.costs.fee_bps / 10000 * self.config.position_size_gbp,
            slippage_bps=trade.costs.slippage_bps,
            market_regime=trade.market_regime,
            volatility_bps=float(trade.entry_features.get("realized_sigma_30", 0.0)) * 10000,
            spread_at_entry_bps=trade.costs.spread_bps,
            is_winner=trade.is_winner,
            win_quality="OPTIMAL" if trade.missed_profit_gbp < 0.5 else "EARLY",
            model_version="shadow_v1",
            model_confidence=0.6,
        )

        trade_id = self.memory.store_trade(trade_memory)

        # Run analysis
        if trade.is_winner:
            self.win_analyzer.analyze_win(
                trade_id=trade_id,
                entry_features=trade.entry_features,
                entry_embedding=trade.entry_embedding,
                profit_gbp=trade.net_profit_gbp,
                profit_bps=trade.gross_profit_bps,
                missed_profit_gbp=trade.missed_profit_gbp,
                symbol=symbol,
                market_regime=trade.market_regime,
            )
        else:
            self.loss_analyzer.analyze_loss(
                trade_id=trade_id,
                entry_features=trade.entry_features,
                entry_embedding=trade.entry_embedding,
                loss_gbp=trade.net_profit_gbp,
                loss_bps=trade.gross_profit_bps,
                entry_spread_bps=trade.costs.spread_bps,
                exit_spread_bps=trade.costs.spread_bps,
                entry_volatility_bps=float(trade.entry_features.get("realized_sigma_30", 0.0)) * 10000,
                exit_volatility_bps=float(trade.entry_features.get("realized_sigma_30", 0.0)) * 10000,
                symbol=symbol,
                market_regime_entry=trade.market_regime,
                market_regime_exit=trade.market_regime,
                hold_duration_minutes=trade.hold_duration_minutes,
                stop_loss_bps=self.config.stop_loss_bps,
            )

        logger.info(
            "trade_analyzed",
            trade_id=trade_id,
            is_winner=trade.is_winner,
            profit_gbp=trade.net_profit_gbp,
        )
