"""
Main RL training pipeline that ties everything together.

This is the powerhouse that:
1. Loads ALL historical data
2. Runs shadow trading (every possible trade)
3. Learns from wins/losses
4. Trains the RL agent
5. Updates memory with patterns
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import List

import polars as pl
import structlog

from ..agents.rl_agent import PPOConfig, RLTradingAgent
from ..analyzers.loss_analyzer import LossAnalyzer
from ..analyzers.pattern_matcher import PatternMatcher
from ..analyzers.post_exit_tracker import PostExitTracker
from ..analyzers.win_analyzer import WinAnalyzer
from ..backtesting.shadow_trader import BacktestConfig, ShadowTrader
from ..config.settings import EngineSettings
from ..datasets.data_loader import CandleDataLoader, CandleQuery
from ..datasets.quality_checks import DataQualitySuite
from ..memory.store import MemoryStore
from ..services.costs import CostModel
from ..services.exchange import ExchangeClient
from shared.features.recipe import FeatureRecipe

logger = structlog.get_logger(__name__)


class RLTrainingPipeline:
    """
    Complete reinforcement learning training pipeline.

    This implements your vision:
    - Trains on ALL historical data
    - Every possible shadow trade
    - Learns why wins/losses happened
    - Watches what happens after exit
    - Updates strategy based on learnings
    """

    def __init__(self, settings: EngineSettings, dsn: str):
        self.settings = settings
        self.dsn = dsn

        # Initialize components
        self.memory_store = MemoryStore(dsn=dsn, embedding_dim=128)
        self.feature_recipe = FeatureRecipe()

        # Analyzers
        self.win_analyzer = WinAnalyzer(dsn=dsn, memory_store=self.memory_store)
        self.loss_analyzer = LossAnalyzer(dsn=dsn, memory_store=self.memory_store)
        self.post_exit_tracker = PostExitTracker(dsn=dsn)
        self.pattern_matcher = PatternMatcher(dsn=dsn, memory_store=self.memory_store)

        # Cost model
        self.cost_model = CostModel(settings.costs)

        # RL Agent
        ppo_config = PPOConfig(
            learning_rate=settings.training.rl_agent.learning_rate,
            gamma=settings.training.rl_agent.gamma,
            clip_epsilon=settings.training.rl_agent.clip_epsilon,
            entropy_coef=settings.training.rl_agent.entropy_coef,
            n_epochs=settings.training.rl_agent.n_epochs,
            batch_size=settings.training.rl_agent.batch_size,
        )

        self.agent = RLTradingAgent(
            state_dim=settings.training.rl_agent.state_dim,
            memory_store=self.memory_store,
            config=ppo_config,
            device=settings.training.rl_agent.device,
        )

        # Shadow trader
        backtest_config = BacktestConfig(
            position_size_gbp=settings.training.shadow_trading.position_size_gbp,
            max_hold_minutes=settings.training.shadow_trading.max_hold_minutes,
            stop_loss_bps=settings.training.shadow_trading.stop_loss_bps,
            take_profit_bps=settings.training.shadow_trading.take_profit_bps,
            min_confidence_threshold=settings.training.shadow_trading.min_confidence_threshold,
        )

        self.shadow_trader = ShadowTrader(
            agent=self.agent,
            memory_store=self.memory_store,
            cost_model=self.cost_model,
            win_analyzer=self.win_analyzer,
            loss_analyzer=self.loss_analyzer,
            feature_recipe=self.feature_recipe,
            config=backtest_config,
        )

        logger.info("rl_training_pipeline_initialized")

    def train_on_symbol(
        self,
        symbol: str,
        exchange_client: ExchangeClient,
        lookback_days: int = 365,
    ) -> dict:
        """
        Train the RL agent on all historical data for a symbol.

        This is where the magic happens - the bot learns from EVERY historical pattern.

        Args:
            symbol: Trading pair symbol
            exchange_client: Exchange client for data fetching
            lookback_days: How far back to train (365 = 1 year of learning)

        Returns:
            Training metrics and statistics
        """
        logger.info("training_start", symbol=symbol, lookback_days=lookback_days)

        # 1. Load ALL historical data
        historical_data = self._load_historical_data(
            symbol=symbol,
            exchange_client=exchange_client,
            lookback_days=lookback_days,
        )

        if historical_data is None or historical_data.height < 1000:
            logger.warning("insufficient_historical_data", symbol=symbol)
            return {"error": "insufficient_data", "symbol": symbol}

        logger.info("historical_data_loaded", symbol=symbol, rows=historical_data.height)

        # 2. Run shadow trading - every possible trade
        logger.info("starting_shadow_trading", symbol=symbol)
        trades = self.shadow_trader.backtest_symbol(
            symbol=symbol,
            historical_data=historical_data,
            training_mode=True,  # Agent learns from these trades
        )

        logger.info("shadow_trading_complete", symbol=symbol, total_trades=len(trades))

        # 3. Update agent with accumulated experience
        if len(trades) > 0:
            logger.info("updating_rl_agent", trades=len(trades))
            update_metrics = self.agent.update()
            logger.info("agent_updated", **update_metrics)
        else:
            update_metrics = {}

        # 4. Analyze aggregate performance
        wins = sum(1 for t in trades if t.is_winner)
        total_profit = sum(t.net_profit_gbp for t in trades)
        avg_profit = total_profit / len(trades) if trades else 0.0
        win_rate = wins / len(trades) if trades else 0.0

        # 5. Learn optimal parameters for discovered patterns
        logger.info("learning_pattern_parameters", symbol=symbol)
        top_patterns = self.pattern_matcher.get_top_patterns(min_win_rate=0.55, min_sample_size=10)
        for pattern in top_patterns[:10]:  # Top 10 patterns
            if pattern.pattern_id:
                self.pattern_matcher.learn_optimal_parameters(pattern.pattern_id)

        # 6. Get exit timing stats
        exit_stats = self.post_exit_tracker.get_exit_learning_stats(symbol=symbol, days=lookback_days)

        metrics = {
            "symbol": symbol,
            "total_trades": len(trades),
            "wins": wins,
            "losses": len(trades) - wins,
            "win_rate": win_rate,
            "total_profit_gbp": total_profit,
            "avg_profit_per_trade_gbp": avg_profit,
            "patterns_learned": len(top_patterns),
            "exit_timing_accuracy": exit_stats.get("exit_timing_accuracy", 0.0),
            "avg_missed_profit_bps": exit_stats.get("avg_missed_profit_bps", 0.0),
            **update_metrics,
        }

        logger.info("training_complete", **metrics)
        return metrics

    def train_on_universe(
        self,
        symbols: List[str],
        exchange_client: ExchangeClient,
        lookback_days: int = 365,
    ) -> List[dict]:
        """
        Train on entire universe of symbols.

        This builds a comprehensive memory of ALL patterns across ALL coins.
        """
        logger.info("universe_training_start", symbols=len(symbols))

        results = []

        for idx, symbol in enumerate(symbols):
            logger.info("training_symbol", symbol=symbol, progress=f"{idx+1}/{len(symbols)}")

            try:
                metrics = self.train_on_symbol(
                    symbol=symbol,
                    exchange_client=exchange_client,
                    lookback_days=lookback_days,
                )
                results.append(metrics)

            except Exception as exc:
                logger.exception("training_failed", symbol=symbol, error=str(exc))
                results.append({"symbol": symbol, "error": str(exc)})

        # Aggregate stats
        total_trades = sum(r.get("total_trades", 0) for r in results)
        total_wins = sum(r.get("wins", 0) for r in results)
        total_profit = sum(r.get("total_profit_gbp", 0.0) for r in results)

        logger.info(
            "universe_training_complete",
            symbols=len(symbols),
            total_trades=total_trades,
            total_wins=total_wins,
            overall_win_rate=total_wins / total_trades if total_trades > 0 else 0.0,
            total_profit_gbp=total_profit,
        )

        return results

    def save_agent(self, path: str) -> None:
        """Save trained RL agent."""
        self.agent.save(path)
        logger.info("agent_saved", path=path)

    def load_agent(self, path: str) -> None:
        """Load pre-trained RL agent."""
        self.agent.load(path)
        logger.info("agent_loaded", path=path)

    def _load_historical_data(
        self,
        symbol: str,
        exchange_client: ExchangeClient,
        lookback_days: int,
    ) -> pl.DataFrame | None:
        """Load historical candle data."""
        try:
            quality_suite = DataQualitySuite()
            loader = CandleDataLoader(exchange_client=exchange_client, quality_suite=quality_suite)

            # Calculate start/end dates
            end_date = datetime.now(tz=timezone.utc)
            start_date = end_date - timedelta(days=lookback_days)

            query = CandleQuery(
                symbol=symbol,
                timeframe="15m",  # 15-minute candles for intraday learning
                start_ts=start_date,
                end_ts=end_date,
            )

            data = loader.load(query)

            logger.info(
                "data_loaded",
                symbol=symbol,
                rows=data.height,
                start=start_date.isoformat(),
                end=end_date.isoformat(),
            )

            return data

        except Exception as exc:
            logger.exception("data_load_failed", symbol=symbol, error=str(exc))
            return None

    def get_memory_stats(self) -> dict:
        """Get statistics about learned memory."""
        # Would query database for stats
        return {
            "total_trades_in_memory": 0,  # TODO: Implement
            "total_patterns": 0,
            "top_patterns_count": 0,
        }
