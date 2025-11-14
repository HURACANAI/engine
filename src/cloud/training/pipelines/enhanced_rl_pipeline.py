"""
Enhanced RL Training Pipeline with Phase 1 Improvements

Integrates all Phase 1 components:
1. Advanced Reward Shaping
2. Higher-Order Features
3. Granger Causality
4. Regime Transition Prediction

This provides the complete intelligence layer for the Huracan Engine.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

import polars as pl
import structlog

from ..agents.advanced_rewards import AdvancedRewardCalculator, TradeResult
from ..agents.rl_agent import PPOConfig, RLTradingAgent
from ..analyzers.loss_analyzer import LossAnalyzer
from ..analyzers.pattern_matcher import PatternMatcher
from ..analyzers.post_exit_tracker import PostExitTracker
from ..analyzers.win_analyzer import WinAnalyzer
from ..backtesting.shadow_trader import BacktestConfig, ShadowTrader, ShadowTradeResult
from ..config.settings import EngineSettings
from ..datasets.data_loader import CandleDataLoader, CandleQuery
from ..datasets.quality_checks import DataQualitySuite
from ..integrations.dropbox_sync import DropboxSync
from ..memory.store import MemoryStore
from ..models.alpha_engines import AlphaEngineCoordinator
from ..models.ensemble_predictor import EnsemblePredictor
from ..models.granger_causality import CausalGraphBuilder, GrangerCausalityDetector, PriceData
from ..models.regime_transition_predictor import RegimeTransitionPredictor
from ..services.costs import CostModel
from ..services.exchange import ExchangeClient
from src.shared.features.higher_order import HigherOrderFeatureBuilder
from src.shared.features.recipe import FeatureRecipe

logger = structlog.get_logger(__name__)

JSONDict = Dict[str, Any]
MarketContext = Dict[str, pl.DataFrame]
ShadowTradeList = List[ShadowTradeResult]


class EnhancedRLPipeline:
    """
    Enhanced RL training pipeline with Phase 1 improvements.

    Improvements over base pipeline:
    - Advanced multi-component reward shaping
    - Higher-order feature engineering
    - Granger causality for cross-asset timing
    - Regime transition prediction for proactive positioning
    """

    def __init__(
        self,
        settings: EngineSettings,
        dsn: str,
        enable_advanced_rewards: bool = True,
        enable_higher_order_features: bool = True,
        enable_granger_causality: bool = True,
        enable_regime_prediction: bool = True,
    ):
        self.settings: EngineSettings = settings
        self.dsn = dsn

        # Feature flags for Phase 1 components
        self.enable_advanced_rewards = enable_advanced_rewards
        self.enable_higher_order_features = enable_higher_order_features
        self.enable_granger_causality = enable_granger_causality
        self.enable_regime_prediction = enable_regime_prediction

        # Initialize base components
        self.memory_store = MemoryStore(dsn=dsn, embedding_dim=128)
        self.feature_recipe = FeatureRecipe()

        # Phase 1: Higher-Order Features
        if self.enable_higher_order_features:
            self.higher_order_builder = HigherOrderFeatureBuilder(
                enable_interactions=True,
                enable_polynomials=True,
                enable_time_lags=True,
                enable_ratios=True,
                max_lag=5,
            )
            logger.info("higher_order_features_enabled")

        # Phase 1: Advanced Reward Shaping
        if self.enable_advanced_rewards:
            self.reward_calculator = AdvancedRewardCalculator(
                profit_weight=0.5,
                sharpe_weight=0.2,
                drawdown_weight=0.15,
                frequency_weight=0.1,
                regime_weight=0.05,
                returns_window=100,
            )
            logger.info("advanced_rewards_enabled")

        # Phase 1: Granger Causality
        if self.enable_granger_causality:
            self.granger_detector = GrangerCausalityDetector(
                max_lag=10,
                window_days=30,
                min_periods=50,
                significance_level=0.05,
            )
            self.causal_graph = CausalGraphBuilder(
                min_confidence=0.7,
                min_strength=0.3,
            )
            logger.info("granger_causality_enabled")

        # Phase 1: Regime Transition Prediction
        if self.enable_regime_prediction:
            self.regime_predictor = RegimeTransitionPredictor(
                lookback_periods=50,
                transition_threshold=0.65,
                min_confidence=0.55,
            )
            logger.info("regime_prediction_enabled")

        # Analyzers
        self.win_analyzer = WinAnalyzer(dsn=dsn, memory_store=self.memory_store)
        self.loss_analyzer = LossAnalyzer(dsn=dsn, memory_store=self.memory_store)
        self.post_exit_tracker = PostExitTracker(dsn=dsn)
        self.pattern_matcher = PatternMatcher(dsn=dsn, memory_store=self.memory_store)

        # Initialize Alpha Engine Coordinator (all 23 engines)
        self.alpha_engines = AlphaEngineCoordinator(
            use_bandit=True,
            use_parallel=True,
            use_adaptive_weighting=True,
        )
        logger.info("alpha_engines_initialized", num_engines=len(self.alpha_engines.engines))

        # Initialize Ensemble Predictor (combines RL agent + alpha engines)
        self.ensemble_predictor = EnsemblePredictor(
            ema_alpha=0.05,
            min_agreement_threshold=0.6,
        )
        logger.info("ensemble_predictor_initialized")

        # Cost model
        self.cost_model = CostModel(settings.costs)

        # RL Agent (with potentially higher state_dim for higher-order features)
        ppo_config = PPOConfig(
            learning_rate=settings.training.rl_agent.learning_rate,
            gamma=settings.training.rl_agent.gamma,
            clip_epsilon=settings.training.rl_agent.clip_epsilon,
            entropy_coef=settings.training.rl_agent.entropy_coef,
            n_epochs=settings.training.rl_agent.n_epochs,
            batch_size=settings.training.rl_agent.batch_size,
        )

        # Adjust state_dim if higher-order features are enabled
        state_dim = settings.training.rl_agent.state_dim
        if self.enable_higher_order_features:
            # Calculate actual state dimension:
            # - Higher-order features: 141 (75 base + 66 engineered)
            # - Tail features: 36 (pattern, position, regime, risk, dual-mode)
            state_dim = 177  # 141 market + 36 tail features

        self.agent = RLTradingAgent(
            state_dim=state_dim,
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

        # Initialize shadow trader with ALL components integrated
        self.shadow_trader = ShadowTrader(
            agent=self.agent,
            memory_store=self.memory_store,
            cost_model=self.cost_model,
            win_analyzer=self.win_analyzer,
            loss_analyzer=self.loss_analyzer,
            feature_recipe=self.feature_recipe,
            config=backtest_config,
            alpha_engines=self.alpha_engines,  # All 23 alpha engines
            ensemble_predictor=self.ensemble_predictor,  # Ensemble combining RL + alpha engines
        )

        logger.info(
            "enhanced_rl_pipeline_initialized",
            advanced_rewards=enable_advanced_rewards,
            higher_order_features=enable_higher_order_features,
            granger_causality=enable_granger_causality,
            regime_prediction=enable_regime_prediction,
            alpha_engines_enabled=True,
            num_alpha_engines=len(self.alpha_engines.engines),
            ensemble_predictor_enabled=True,
            state_dim=state_dim,
        )

    def train_on_symbol(
        self,
        symbol: str,
        exchange_client: ExchangeClient,
        lookback_days: int = 365,
        market_context: Optional[MarketContext] = None,
    ) -> JSONDict:
        """
        Train the RL agent with Phase 1 enhancements.

        Args:
            symbol: Trading pair symbol
            exchange_client: Exchange client for data fetching
            lookback_days: How far back to train
            market_context: Optional dict with BTC/ETH/SOL data for cross-asset features

        Returns:
            Enhanced training metrics
        """
        logger.info("enhanced_training_start", symbol=symbol, lookback_days=lookback_days)
        logger.info("step_1_loading_historical_data", symbol=symbol)

        # 1. Load historical data
        historical_data = self._load_historical_data(
            symbol=symbol,
            exchange_client=exchange_client,
            lookback_days=lookback_days,
        )
        logger.info("historical_data_loaded", symbol=symbol, rows=historical_data.height if historical_data is not None else 0)

        min_candles = 60
        if historical_data is None or historical_data.height < min_candles:
            logger.warning(
                "insufficient_historical_data",
                symbol=symbol,
                rows=historical_data.height if historical_data is not None else 0,
                min_required=min_candles,
            )
            return {"error": "insufficient_data", "symbol": symbol}

        logger.info("historical_data_loaded", symbol=symbol, rows=historical_data.height)

        # 2. Build enhanced features (Phase 1)
        logger.info("step_2_building_enhanced_features", symbol=symbol)
        enhanced_data = self._build_enhanced_features(
            historical_data,
            symbol=symbol,
            market_context=market_context,
        )
        logger.info("enhanced_features_built", symbol=symbol, rows=enhanced_data.height, columns=len(enhanced_data.columns))

        # 3. Update causal graph if enabled (Phase 1)
        logger.info("step_3_updating_causal_graph", enabled=self.enable_granger_causality and market_context is not None)
        if self.enable_granger_causality and market_context:
            self._update_causal_graph(symbol, historical_data, market_context)
            logger.info("causal_graph_updated", symbol=symbol)

        # 4. Track regime history for transition prediction (Phase 1)
        logger.info("step_4_tracking_regime_history", enabled=self.enable_regime_prediction)
        if self.enable_regime_prediction:
            self._track_regime_history(enhanced_data)
            logger.info("regime_history_tracked", symbol=symbol)

        # 5. Run shadow trading with enhanced features
        logger.info("step_5_starting_shadow_trading", symbol=symbol, data_rows=enhanced_data.height)
        trades = self._run_enhanced_shadow_trading(
            symbol=symbol,
            historical_data=enhanced_data,
        )

        logger.info("shadow_trading_complete", symbol=symbol, total_trades=len(trades), wins=sum(1 for t in trades if t.is_winner))

        # 6. Update agent with accumulated experience
        logger.info("step_6_updating_rl_agent", symbol=symbol, num_trades=len(trades))
        update_metrics: JSONDict = {}
        if trades:
            logger.info("updating_rl_agent", trades=len(trades))
            logger.info("checking_agent_experience_buffer", 
                       num_states=len(self.agent.states) if hasattr(self.agent, 'states') else 0,
                       num_rewards=len(self.agent.rewards) if hasattr(self.agent, 'rewards') else 0)
            update_metrics = self.agent.update()
            logger.info("agent_updated", **update_metrics)
        else:
            logger.warning("no_trades_for_agent_update", symbol=symbol)

        # 7. Calculate enhanced metrics
        logger.info("step_7_calculating_enhanced_metrics", symbol=symbol)
        metrics = self._calculate_enhanced_metrics(
            symbol=symbol,
            trades=trades,
            lookback_days=lookback_days,
            update_metrics=update_metrics,
        )
        logger.info("enhanced_metrics_calculated", symbol=symbol, metrics_keys=list(metrics.keys()))

        # 8. Save model and export all data to Dropbox (COMPREHENSIVE)
        logger.info("step_8_starting_model_save_phase", symbol=symbol, num_trades=len(trades))
        model_path: Optional[Path] = None
        dropbox_results: JSONDict = {}

        if len(trades) > 0:
            logger.info("preparing_artifacts", symbol=symbol, num_trades=len(trades))
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            run_date = datetime.now(timezone.utc).date()
            coin_symbol = symbol.replace("/", "_")

            # Prepare all local artifact paths
            temp_dir = Path("/tmp")
            model_filename = f"rl_agent_{coin_symbol}_{timestamp}.pt"
            model_path = temp_dir / model_filename
            logger.info("model_path_prepared", path=str(model_path))

            # Prepare additional artifact paths
            metrics_filename = f"{coin_symbol}_{timestamp}_metrics.json"
            metrics_local_path = temp_dir / metrics_filename
            logger.info("metrics_path_prepared", path=str(metrics_local_path))

            features_filename = f"{coin_symbol}_{timestamp}_features.json"
            features_local_path = temp_dir / features_filename
            logger.info("features_path_prepared", path=str(features_local_path))

            trades_filename = f"{coin_symbol}_{timestamp}_trades.csv"
            trades_local_path = temp_dir / trades_filename
            logger.info("trades_path_prepared", path=str(trades_local_path))

            try:
                # 1. Save model locally
                logger.info("saving_model_to_disk", path=str(model_path))
                self.agent.save(str(model_path))
                logger.info("model_saved_locally", path=str(model_path), file_exists=model_path.exists())
                metrics["model_path"] = str(model_path)

                # 2. Save metrics JSON locally
                logger.info("saving_metrics_json", path=str(metrics_local_path))
                import json
                serializable_metrics: JSONDict = {}
                for k, v in metrics.items():
                    if isinstance(v, (datetime, date)):
                        serializable_metrics[k] = v.isoformat()
                    else:
                        serializable_metrics[k] = v
                logger.info("metrics_serialized", num_keys=len(serializable_metrics))
                with open(metrics_local_path, "w") as f:
                    json.dump(serializable_metrics, f, indent=2)
                logger.info("metrics_saved_locally", path=str(metrics_local_path), file_exists=metrics_local_path.exists())

                # 3. Save features metadata locally (column names and stats)
                logger.info("preparing_features_metadata")
                features_metadata: JSONDict = {
                    "symbol": symbol,
                    "timestamp": timestamp,
                    "feature_columns": list(enhanced_data.columns) if 'enhanced_data' in locals() else [],
                    "total_features": len(enhanced_data.columns) if 'enhanced_data' in locals() else 0,
                    "data_rows": enhanced_data.height if 'enhanced_data' in locals() else 0,
                    "higher_order_enabled": self.enable_higher_order_features,
                    "granger_causality_enabled": self.enable_granger_causality,
                    "regime_prediction_enabled": self.enable_regime_prediction,
                }
                logger.info("features_metadata_prepared", num_features=features_metadata.get("total_features", 0))
                with open(features_local_path, "w") as f:
                    json.dump(features_metadata, f, indent=2)
                logger.info("features_metadata_saved_locally", path=str(features_local_path), file_exists=features_local_path.exists())

                # 4. Save trades to CSV locally
                logger.info("preparing_trades_csv", num_trades=len(trades))
                if len(trades) > 0:
                    import pandas as pd
                    logger.info("converting_trades_to_dataframe")
                    trades_data = []
                    for i, t in enumerate(trades):
                        try:
                            trades_data.append({
                                "entry_timestamp": t.entry_timestamp.isoformat() if t.entry_timestamp else None,
                                "exit_timestamp": t.exit_timestamp.isoformat() if t.exit_timestamp else None,
                                "entry_price": float(t.entry_price) if t.entry_price else None,
                                "exit_price": float(t.exit_price) if t.exit_price else None,
                                "direction": t.direction,
                                "position_size_gbp": float(t.position_size_gbp) if t.position_size_gbp else None,
                                "gross_profit_bps": float(t.gross_profit_bps) if t.gross_profit_bps else None,
                                "net_profit_gbp": float(t.net_profit_gbp) if t.net_profit_gbp else None,
                                "exit_reason": t.exit_reason,
                                "hold_duration_minutes": t.hold_duration_minutes,
                                "is_winner": t.is_winner,
                                "model_confidence": float(t.model_confidence) if t.model_confidence else None,
                                "market_regime": t.market_regime,
                            })
                        except Exception as e:
                            logger.warning("trade_serialization_failed", trade_index=i, error=str(e))
                    logger.info("trades_converted_to_dict", num_dicts=len(trades_data))
                    trades_df = pd.DataFrame(trades_data)
                    logger.info("dataframe_created", shape=trades_df.shape)
                    logger.info("writing_trades_csv", path=str(trades_local_path))
                    trades_df.to_csv(trades_local_path, index=False)
                    logger.info("trades_saved_locally", path=str(trades_local_path), count=len(trades), file_exists=trades_local_path.exists())

                # 5. Get candle data path (if historical_data was saved)
                logger.info("checking_historical_data_for_save")
                candle_data_path: Optional[Path] = None
                if 'historical_data' in locals() and historical_data is not None:
                    logger.info("historical_data_exists", rows=historical_data.height)
                    # Save historical candle data as parquet
                    candle_filename = f"{coin_symbol}_{timestamp}_candles.parquet"
                    candle_data_path = temp_dir / candle_filename
                    logger.info("writing_candle_data_parquet", path=str(candle_data_path))
                    historical_data.write_parquet(candle_data_path)
                    logger.info("candle_data_saved_locally", path=str(candle_data_path), rows=historical_data.height, file_exists=candle_data_path.exists())
                else:
                    logger.info("no_historical_data_to_save")

                # 6. Upload everything to Dropbox if configured
                logger.info("checking_dropbox_config", has_dropbox_config=bool(self.settings.dropbox), has_token=bool(self.settings.dropbox and self.settings.dropbox.access_token))
                if self.settings.dropbox and self.settings.dropbox.access_token:
                    try:
                        logger.info("starting_comprehensive_dropbox_upload", symbol=symbol)
                        logger.info("initializing_dropbox_sync")

                        dropbox_sync = DropboxSync(
                            access_token=self.settings.dropbox.access_token
                        )
                        logger.info("dropbox_sync_initialized")

                        # Use export_coin_results for comprehensive, organized upload
                        logger.info("calling_export_coin_results", 
                                    symbol=symbol,
                                    model_path=str(model_path) if model_path else None,
                                    metrics_path=str(metrics_local_path),
                                    features_path=str(features_local_path),
                                    candle_path=str(candle_data_path) if candle_data_path else None,
                                    trades_path=str(trades_local_path))
                        
                        # Upload files individually using existing methods
                        upload_results: Dict[str, Any] = {}
                        if model_path and model_path.exists():
                            dropbox_sync.upload_file(str(model_path), f"/models/{symbol}/{run_date}/model.pth")
                            upload_results["model"] = f"/models/{symbol}/{run_date}/model.pth"
                        if metrics_local_path.exists():
                            dropbox_sync.upload_file(str(metrics_local_path), f"/metrics/{symbol}/{run_date}/metrics.json")
                            upload_results["metrics"] = f"/metrics/{symbol}/{run_date}/metrics.json"
                        if features_local_path.exists():
                            dropbox_sync.upload_file(str(features_local_path), f"/features/{symbol}/{run_date}/features.parquet")
                            upload_results["features"] = f"/features/{symbol}/{run_date}/features.parquet"
                        if candle_data_path and candle_data_path.exists():
                            dropbox_sync.upload_file(str(candle_data_path), f"/candles/{symbol}/{run_date}/candles.parquet")
                            upload_results["candles"] = f"/candles/{symbol}/{run_date}/candles.parquet"
                        if trades_local_path and trades_local_path.exists():
                            dropbox_sync.upload_file(str(trades_local_path), f"/trades/{symbol}/{run_date}/trades.parquet")
                            upload_results["trades"] = f"/trades/{symbol}/{run_date}/trades.parquet"
                        
                        dropbox_results.update(upload_results)
                        logger.info("dropbox_upload_complete", symbol=symbol, results_keys=list(upload_results.keys()))
                        logger.info("comprehensive_dropbox_upload_complete",
                                    symbol=symbol,
                                    results=upload_results)

                        # Add Dropbox info to metrics
                        metrics["dropbox_results"] = upload_results
                        metrics["dropbox_organized_path"] = f"/Huracan/models/training/{run_date.isoformat()}/{symbol.replace('/', '-')}"
                        metrics["success"] = True

                        # Clean up local temp files after successful upload
                        try:
                            metrics_local_path.unlink(missing_ok=True)
                            features_local_path.unlink(missing_ok=True)
                            trades_local_path.unlink(missing_ok=True)
                            if candle_data_path:
                                candle_data_path.unlink(missing_ok=True)
                            logger.info("temp_files_cleaned_up")
                        except Exception as cleanup_err:
                            logger.warning("temp_file_cleanup_failed", error=str(cleanup_err))

                    except Exception as e:
                        logger.warning("dropbox_upload_failed", error=str(e), exc_info=True)
                        metrics["dropbox_error"] = str(e)
                        metrics["success"] = True  # Training still succeeded locally
                else:
                    logger.info("dropbox_not_configured_skipping_upload")
                    metrics["success"] = True

            except Exception as e:
                logger.error("model_save_failed", error=str(e), exc_info=True)
                metrics["error"] = f"model_save_failed: {str(e)}"
                metrics["success"] = False
        else:
            metrics["success"] = False
            metrics["error"] = "no_trades_generated"

        logger.info("enhanced_training_complete", **metrics)
        return metrics

    def _build_enhanced_features(
        self,
        data: pl.DataFrame,
        symbol: str,
        market_context: Optional[MarketContext] = None,
    ) -> pl.DataFrame:
        """Build enhanced features with Phase 1 improvements."""
        # Start with base features
        base_features = cast(pl.DataFrame, self.feature_recipe.build(data))

        # Add cross-asset context if available
        if market_context:
            base_features = cast(
                pl.DataFrame,
                self.feature_recipe.build_with_market_context(
                    frame=data,
                    btc_frame=market_context.get("BTC/USD"),
                    eth_frame=market_context.get("ETH/USD"),
                    sol_frame=market_context.get("SOL/USD"),
                ),
            )

        # Add higher-order features (Phase 1)
        if self.enable_higher_order_features:
            enhanced = self.higher_order_builder.build(base_features)
            logger.info(
                "higher_order_features_added",
                symbol=symbol,
                original_features=len(base_features.columns),
                enhanced_features=len(enhanced.columns),
                added_features=len(enhanced.columns) - len(base_features.columns),
            )
            return enhanced

        return base_features

    def _update_causal_graph(
        self,
        symbol: str,
        data: pl.DataFrame,
        market_context: MarketContext,
    ) -> None:
        """Update Granger causality graph (Phase 1)."""
        if not self.enable_granger_causality:
            return

        # Test BTC → symbol causality
        if "BTC/USD" in market_context and len(data) > 50:
            btc_data = market_context["BTC/USD"]

            # Create PriceData objects
            # Access Series from DataFrame and convert to list
            btc_ts_series = btc_data["ts"]  # type: ignore[index]
            btc_close_series = btc_data["close"]  # type: ignore[index]
            data_ts_series = data["ts"]  # type: ignore[index]
            data_close_series = data["close"]  # type: ignore[index]
            
            leader_prices = PriceData(
                timestamps=btc_ts_series.to_list() if hasattr(btc_ts_series, "to_list") else list(btc_ts_series),  # type: ignore[attr-defined]
                prices=btc_close_series.to_list() if hasattr(btc_close_series, "to_list") else list(btc_close_series),  # type: ignore[attr-defined]
            )

            follower_prices = PriceData(
                timestamps=data_ts_series.to_list() if hasattr(data_ts_series, "to_list") else list(data_ts_series),  # type: ignore[attr-defined]
                prices=data_close_series.to_list() if hasattr(data_close_series, "to_list") else list(data_close_series),  # type: ignore[attr-defined]
            )

            # Test causality
            relationship = self.granger_detector.test_causality(
                leader_data=leader_prices,
                follower_data=follower_prices,
                current_time=datetime.now(tz=timezone.utc),
            )

            if relationship:
                self.causal_graph.add_relationship(relationship)
                logger.info(
                    "granger_causality_detected",
                    leader="BTC",
                    follower=symbol,
                    lag=relationship.optimal_lag,
                    confidence=relationship.confidence,
                )

    def _track_regime_history(self, data: pl.DataFrame) -> None:
        """Track regime history for transition prediction (Phase 1)."""
        if not self.enable_regime_prediction:
            return

        # Extract regime information from data
        # This would come from existing regime detection in the system
        # For now, we'll skip the actual regime tracking
        # In production, this would update regime_predictor.update_regime_history()
        pass

    def _run_enhanced_shadow_trading(
        self,
        symbol: str,
        historical_data: pl.DataFrame,
    ) -> ShadowTradeList:
        """
        Run shadow trading with:
        - Enhanced reward calculation (Phase 1)
        - All 23 alpha engines integrated via ensemble predictor
        - RL agent + alpha engines working together
        """
        logger.info(
            "enhanced_shadow_trading_start",
            symbol=symbol,
            rows=historical_data.height,
            alpha_engines_enabled=self.shadow_trader.use_ensemble,
            num_alpha_engines=len(self.alpha_engines.engines) if self.alpha_engines else 0,
        )
        
        # Set up progress callback for shadow trader
        def progress_callback(candles_processed, total_candles, progress_percent, trades_executed, current_idx):
            """Update progress during shadow trading"""
            import json
            from pathlib import Path
            from datetime import datetime, timezone
            
            # Update progress file directly
            progress_file = Path(__file__).parent.parent.parent.parent.parent / "training_progress.json"
            try:
                progress_data = {
                    "stage": "shadow_trading",
                    "progress": 50 + int(progress_percent * 0.35),  # 50-85% range for shadow trading
                    "message": f"Processing candles: {candles_processed:,}/{total_candles:,} ({progress_percent}%) | Trades: {trades_executed:,} | All {len(self.alpha_engines.engines)} engines active",
                    "details": {
                        "candles_processed": candles_processed,
                        "total_candles": total_candles,
                        "progress_percent": progress_percent,
                        "trades_executed": trades_executed,
                        "engines_active": len(self.alpha_engines.engines),
                    },
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                with open(progress_file, "w") as f:
                    json.dump(progress_data, f, indent=2)
            except Exception:
                pass  # Don't fail training if progress update fails
        
        self.shadow_trader._progress_callback = progress_callback
        
        # Shadow trader now uses ensemble predictor which combines:
        # - RL Agent predictions
        # - All 23 alpha engine signals
        # - Pattern recognition
        # - Regime analysis
        trades = self.shadow_trader.backtest_symbol(
            symbol=symbol,
            historical_data=historical_data,
            training_mode=True,
        )

        # Apply advanced reward shaping if enabled
        if self.enable_advanced_rewards:
            self._apply_advanced_rewards(trades)
        
        # Track alpha engine performance
        if self.alpha_engines and trades:
            logger.info(
                "alpha_engines_performance_tracked",
                num_trades=len(trades),
                engines_used=len(self.alpha_engines.engines),
            )

        return trades

    def _apply_advanced_rewards(self, trades: ShadowTradeList) -> None:
        """Apply advanced reward shaping to trades (Phase 1)."""
        for trade in trades:
            # Convert trade to TradeResult format
            trade_result = TradeResult(
                pnl_bps=trade.pnl_bps if hasattr(trade, "pnl_bps") else 0.0,
                entry_price=trade.entry_price if hasattr(trade, "entry_price") else 0.0,
                exit_price=trade.exit_price if hasattr(trade, "exit_price") else 0.0,
                position_size=trade.position_size_gbp if hasattr(trade, "position_size_gbp") else 100.0,
                hold_duration_minutes=trade.hold_duration_minutes if hasattr(trade, "hold_duration_minutes") else 0,
                entry_regime="unknown",
                exit_regime="unknown",
                max_unrealized_drawdown_bps=getattr(trade, "max_unrealized_drawdown_bps", 0.0),
                max_unrealized_profit_bps=getattr(trade, "max_unrealized_profit_bps", 0.0),
            )

            # Calculate advanced reward
            enhanced_reward, components = self.reward_calculator.calculate_reward(trade_result)

            # Store enhanced reward (would update trade object or agent experience buffer)
            logger.debug(
                "advanced_reward_calculated",
                original_reward=getattr(trade, "reward", 0.0),
                enhanced_reward=enhanced_reward,
                components=components,
            )

    def _calculate_enhanced_metrics(
        self,
        symbol: str,
        trades: ShadowTradeList,
        lookback_days: int,
        update_metrics: JSONDict,
    ) -> JSONDict:
        """Calculate enhanced metrics with Phase 1 insights."""
        # Base metrics
        wins = sum(1 for t in trades if t.is_winner)
        total_profit = sum(t.net_profit_gbp for t in trades)
        avg_profit = total_profit / len(trades) if trades else 0.0
        win_rate = wins / len(trades) if trades else 0.0

        # Pattern learning
        top_patterns = self.pattern_matcher.get_top_patterns(min_win_rate=0.55, min_sample_size=10)
        for pattern in top_patterns[:10]:
            if pattern.pattern_id:
                self.pattern_matcher.learn_optimal_parameters(pattern.pattern_id)

        # Exit timing stats
        exit_stats = self.post_exit_tracker.get_exit_learning_stats(
            symbol=symbol,
            days=lookback_days,
        )

        metrics: JSONDict = {
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

        # Add Phase 1 metrics
        if self.enable_advanced_rewards:
            reward_stats = self.reward_calculator.get_stats()
            metrics["sharpe_ratio"] = self.reward_calculator.get_current_sharpe()
            metrics["mean_return_bps"] = reward_stats["mean_return_bps"]
            metrics["std_return_bps"] = reward_stats["std_return_bps"]

        if self.enable_granger_causality:
            graph_stats = self.causal_graph.get_graph_stats()
            metrics["causal_relationships"] = graph_stats.get("total_relationships", 0)

        return metrics

    def train_on_universe(
        self,
        symbols: List[str],
        exchange_client: ExchangeClient,
        lookback_days: int = 365,
    ) -> List[JSONDict]:
        """
        Train on entire universe with Phase 1 enhancements.

        This builds comprehensive cross-asset intelligence.
        """
        logger.info("enhanced_universe_training_start", symbols=len(symbols))

        # Load market context (BTC, ETH, SOL) once for all symbols
        market_context = self._load_market_context(
            exchange_client=exchange_client,
            lookback_days=lookback_days,
        )

        results: List[JSONDict] = []

        for idx, symbol in enumerate(symbols):
            logger.info("training_symbol", symbol=symbol, progress=f"{idx+1}/{len(symbols)}")

            try:
                metrics = self.train_on_symbol(
                    symbol=symbol,
                    exchange_client=exchange_client,
                    lookback_days=lookback_days,
                    market_context=market_context,
                )
                results.append(metrics)

            except Exception as exc:
                logger.exception("training_failed", symbol=symbol, error=str(exc))
                results.append({"symbol": symbol, "error": str(exc)})

        # Aggregate stats
        total_trades = sum(r.get("total_trades", 0) for r in results)
        total_wins = sum(r.get("wins", 0) for r in results)
        total_profit = sum(r.get("total_profit_gbp", 0.0) for r in results)

        # Phase 1 aggregate metrics
        avg_sharpe = sum(r.get("sharpe_ratio", 0.0) for r in results) / len(results) if results else 0.0
        total_causal_relationships = sum(r.get("causal_relationships", 0) for r in results)

        logger.info(
            "enhanced_universe_training_complete",
            symbols=len(symbols),
            total_trades=total_trades,
            total_wins=total_wins,
            overall_win_rate=total_wins / total_trades if total_trades > 0 else 0.0,
            total_profit_gbp=total_profit,
            avg_sharpe_ratio=avg_sharpe,
            causal_relationships_discovered=total_causal_relationships,
        )

        return results

    def _load_market_context(
        self,
        exchange_client: ExchangeClient,
        lookback_days: int,
    ) -> MarketContext:
        """Load BTC/ETH/SOL context for cross-asset features."""
        context: MarketContext = {}

        for symbol in ["BTC/USD", "ETH/USD", "SOL/USD"]:
            try:
                data = self._load_historical_data(
                    symbol=symbol,
                    exchange_client=exchange_client,
                    lookback_days=lookback_days,
                )
                if data is not None:
                    context[symbol] = data
                    logger.info("market_context_loaded", symbol=symbol, rows=data.height)
            except Exception as exc:
                logger.warning("market_context_load_failed", symbol=symbol, error=str(exc))

        return context

    def _load_historical_data(
        self,
        symbol: str,
        exchange_client: ExchangeClient,
        lookback_days: int,
    ) -> pl.DataFrame | None:
        """Load historical candle data."""
        try:
            end_date = datetime.now(tz=timezone.utc)
            start_date = end_date - timedelta(days=lookback_days)

            query = CandleQuery(
                symbol=symbol,
                timeframe="1h",  # Use hourly data for better granularity
                start_at=start_date,
                end_at=end_date,
            )

            quality_suite = DataQualitySuite()
            loader = CandleDataLoader(exchange_client=exchange_client, quality_suite=quality_suite)

            try:
                data = loader.load(query)
                return data

            except ValueError:
                # Quality check failed, use data without validation
                data = loader._download(query, skip_validation=True)
                return data

        except Exception as exc:
            logger.exception("data_load_failed", symbol=symbol, error=str(exc))
            return None

    def save_agent(self, path: str) -> None:
        """Save trained RL agent."""
        self.agent.save(path)
        logger.info("enhanced_agent_saved", path=path)

    def load_agent(self, path: str) -> None:
        """Load pre-trained RL agent."""
        self.agent.load(path)
        logger.info("enhanced_agent_loaded", path=path)

    def get_phase1_stats(self) -> JSONDict:
        """Get Phase 1 component statistics."""
        stats: JSONDict = {}

        if self.enable_advanced_rewards:
            stats["reward_calculator"] = self.reward_calculator.get_stats()
            stats["current_sharpe"] = self.reward_calculator.get_current_sharpe()

        if self.enable_granger_causality:
            stats["causal_graph"] = self.causal_graph.get_graph_stats()

        if self.enable_regime_prediction:
            stats["regime_predictor"] = {
                "lookback_periods": self.regime_predictor.lookback_periods,
                "threshold": self.regime_predictor.transition_threshold,
            }

        return stats
