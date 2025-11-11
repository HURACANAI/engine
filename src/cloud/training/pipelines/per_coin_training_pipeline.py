"""
Per-Coin Training Pipeline

Trains one tailored model per coin with shared encoder for cross-coin learning.
"""

from __future__ import annotations

import hashlib
import os
import pickle
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog  # type: ignore[import-untyped]

from ..services.job_queue import JobQueue, TrainingJob, JobStatus
from ..services.shared_encoder import SharedEncoder
from ..services.data_gates import DataGates
from ..services.slippage_calibration import SlippageCalibrator
from ..services.per_symbol_champion import PerSymbolChampion
from ..services.roster_exporter import RosterExporter
from ..services.feature_bank import FeatureBank
from ..services.costs import CostBreakdown
from ..services.per_coin_training import PerCoinTrainingService
from src.shared.config_loader import load_config
from src.shared.contracts.per_coin import PerCoinMetrics, CostModel, FeatureRecipe
from src.shared.contracts.paths import format_date_str

logger = structlog.get_logger(__name__)


class PerCoinTrainingPipeline:
    """Pipeline for per-coin training with shared encoder."""
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        dropbox_sync: Optional[Any] = None,
    ):
        """Initialize per-coin training pipeline.
        
        Args:
            config: Configuration dictionary (loads from config.yaml if not provided)
            dropbox_sync: DropboxSync instance for uploading files
        """
        if config is None:
            config = load_config()
        
        self.config = config
        self.dropbox_sync = dropbox_sync
        
        # Initialize services
        self.data_gates = DataGates(
            min_volume_usd=config.get("engine", {}).get("min_volume_usd", 1_000_000.0),
            max_gap_pct=config.get("engine", {}).get("max_gap_pct", 5.0),
            max_spread_bps=config.get("engine", {}).get("max_spread_bps", 50.0),
        )
        
        self.slippage_calibrator = SlippageCalibrator(lookback_days=30)
        self.per_symbol_champion = PerSymbolChampion(champions_dir="champions")
        self.roster_exporter = RosterExporter(output_dir="champions")
        self.feature_bank = FeatureBank(output_dir="meta")
        self.per_coin_service = PerCoinTrainingService(
            dropbox_sync=dropbox_sync,
            base_folder=config.get("general", {}).get("dropbox_root", "/Huracan/").strip("/"),
        )
        
        # Shared encoder (will be trained on all coins)
        encoder_config = config.get("engine", {}).get("shared_encoder", {})
        self.shared_encoder = SharedEncoder(
            encoder_type=encoder_config.get("type", "pca"),
            n_components=encoder_config.get("n_components", 50),
        )
        self.shared_encoder_trained = False
        
        # Training results
        self.training_results: Dict[str, Dict[str, Any]] = {}
        self.all_features: Dict[str, Any] = {}  # For shared encoder training
        
        logger.info("per_coin_training_pipeline_initialized")
    
    def train_symbol(
        self,
        symbol: str,
    ) -> Dict[str, Any]:
        """Train a single symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Training result dictionary
        """
        logger.info("training_symbol", symbol=symbol)
        
        try:
            # Step 1: Load data
            candles_df = self._load_candles(symbol)
            if candles_df.empty:
                return {
                    "symbol": symbol,
                    "status": "skipped",
                    "reason": "no_data",
                }
            
            # Step 2: Data gates
            gate_result = self.data_gates.check_symbol(symbol, candles_df)
            if not gate_result.passed:
                return {
                    "symbol": symbol,
                    "status": "skipped",
                    "reason": "data_gates_failed",
                    "skip_reasons": gate_result.skip_reasons,
                }
            
            # Step 3: Build features
            features_df = self._build_features(symbol, candles_df)
            if features_df.empty:
                return {
                    "symbol": symbol,
                    "status": "skipped",
                    "reason": "feature_building_failed",
                }
            
            # Step 4: Calibrate slippage
            slippage_bps, fit_date = self.slippage_calibrator.calibrate_slippage(symbol, candles_df)
            
            # Step 5: Train model (with shared encoder if trained)
            if self.shared_encoder_trained:
                # Use shared encoder features
                shared_features = self.shared_encoder.transform(features_df)
                # Combine with coin-specific features
                # TODO: Combine shared and coin-specific features
                model_features = features_df
            else:
                model_features = features_df
            
            model, metrics = self._train_model(symbol, model_features, candles_df)
            
            # Step 6: Calculate costs
            cost_model = self._create_cost_model(symbol, slippage_bps, fit_date)
            
            # Step 7: Cost and liquidity gates
            trade_ok = self._check_trade_ok(metrics, cost_model)
            if not trade_ok:
                logger.info("symbol_failed_trade_ok", symbol=symbol, metrics=metrics)
            
            # Step 8: Export artifacts
            date_str = format_date_str()
            artifacts = self._export_artifacts(symbol, model, metrics, cost_model, date_str)
            
            # Step 9: Update champion
            cost_model_dict: Dict[str, Any]
            if isinstance(cost_model, CostModel):
                cost_model_dict = cost_model.to_dict()
            elif isinstance(cost_model, dict):
                cost_model_dict = cost_model
            else:
                # Fallback: try to convert to dict
                cost_model_dict = cost_model.to_dict() if hasattr(cost_model, 'to_dict') else {}  # type: ignore
            
            self.per_symbol_champion.update_champion(
                symbol=symbol,
                model_path=artifacts["model_path"],
                metrics=metrics,
                cost_model=cost_model_dict,
                feature_recipe_hash=artifacts.get("feature_recipe_hash"),
            )
            
            # Step 10: Update feature bank
            feature_importance = self._get_feature_importance(model, features_df)
            self.feature_bank.update_feature_importance(symbol, feature_importance)
            
            # Store features for shared encoder training
            self.all_features[symbol] = features_df
            
            # Extract data info for reporting
            data_info = {
                "rows": len(candles_df),
                "days": 0,
                "start_date": None,
                "end_date": None,
            }
            
            if "ts" in candles_df.columns and not candles_df.empty:
                data_info["start_date"] = candles_df["ts"].min().isoformat() if hasattr(candles_df["ts"].min(), "isoformat") else str(candles_df["ts"].min())
                data_info["end_date"] = candles_df["ts"].max().isoformat() if hasattr(candles_df["ts"].max(), "isoformat") else str(candles_df["ts"].max())
                if len(candles_df) > 1:
                    data_info["days"] = (candles_df["ts"].max() - candles_df["ts"].min()).days
                else:
                    data_info["days"] = 1
            
            return {
                "symbol": symbol,
                "success": True,
                "status": "completed",
                "sample_size": metrics.get("sample_size", len(candles_df)),
                "num_features": len(features_df.columns) if hasattr(features_df, "columns") else 0,
                "metrics": metrics,
                "cost_model": cost_model.to_dict() if hasattr(cost_model, 'to_dict') else cost_model,
                "artifacts": artifacts,
                "trade_ok": trade_ok,
                "data_info": data_info,  # Include data info about days used
            }
            
        except Exception as e:
            logger.error("symbol_training_failed", symbol=symbol, error=str(e))
            return {
                "symbol": symbol,
                "status": "failed",
                "error": str(e),
            }
    
    def train_all_symbols(
        self,
        symbols: List[str],
        max_workers: Optional[int] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Train all symbols using job queue.
        
        Args:
            symbols: List of symbols to train
            max_workers: Maximum number of parallel workers (defaults to config)
            
        Returns:
            Dictionary mapping symbol to training result
        """
        if max_workers is None:
            max_workers = self.config.get("engine", {}).get("parallel_tasks", 8)
        
        # Ensure max_workers is an int
        max_workers_int = int(max_workers) if max_workers is not None else 8
        
        # Create job queue
        job_queue = JobQueue(
            symbols=symbols,
            train_func=self.train_symbol,
            max_workers=max_workers_int,
        )
        
        # Run training
        logger.info("starting_training", total_symbols=len(symbols), max_workers=max_workers)
        results = job_queue.run()
        
        # Train shared encoder on all features
        if len(self.all_features) > 0:
            logger.info("training_shared_encoder", symbols=len(self.all_features))
            self.shared_encoder.fit(self.all_features)
            self.shared_encoder_trained = True
            
            # Save shared encoder
            encoder_path = Path("meta/shared_encoder.pkl")
            encoder_path.parent.mkdir(parents=True, exist_ok=True)
            self.shared_encoder.save(str(encoder_path))
        
        # Export roster
        self._export_roster(results)
        
        return results
    
    def _load_candles(self, symbol: str) -> Any:
        """Load candle data for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            DataFrame with candle data
        """
        from datetime import datetime, timedelta, timezone
        from ..datasets.data_loader import CandleDataLoader, CandleQuery
        from ..datasets.quality_checks import DataQualitySuite
        from ..services.exchange import ExchangeClient
        from ...config.settings import EngineSettings
        import pandas as pd
        
        # Get lookback days from config
        lookback_days = self.config.get("engine", {}).get("lookback_days", 180)
        
        # Initialize exchange client
        settings = EngineSettings.load(environment=os.getenv("HURACAN_ENV", "local"))
        exchange = ExchangeClient(
            exchange_id="binance",
            credentials={},
            sandbox=settings.exchange.sandbox,
        )
        
        # Initialize data loader
        quality_suite = DataQualitySuite(coverage_threshold=0.01)
        loader = CandleDataLoader(
            exchange_client=exchange,
            quality_suite=quality_suite,
            cache_dir=Path("data/candles"),
            fallback_exchanges=[],
        )
        
        # Calculate date range
        end_at = datetime.now(tz=timezone.utc)
        start_at = end_at - timedelta(days=lookback_days)
        
        # Create query
        query = CandleQuery(
            symbol=symbol,
            timeframe="1d",  # Use daily candles for training
            start_at=start_at,
            end_at=end_at,
        )
        
        # Load data
        try:
            frame = loader.load(query, use_cache=True)
            
            if frame.is_empty():
                logger.warning("no_data_loaded", symbol=symbol, lookback_days=lookback_days)
                return pd.DataFrame()
            
            # Convert to pandas DataFrame
            df = frame.to_pandas()
            
            # Calculate actual days used
            if "ts" in df.columns:
                actual_days = (df["ts"].max() - df["ts"].min()).days
                logger.info(
                    "data_loaded",
                    symbol=symbol,
                    rows=len(df),
                    lookback_days=lookback_days,
                    actual_days=actual_days,
                    start_date=df["ts"].min(),
                    end_date=df["ts"].max(),
                )
            
            return df
            
        except Exception as e:
            logger.error("data_load_failed", symbol=symbol, error=str(e))
            return pd.DataFrame()
    
    def _build_features(self, symbol: str, candles_df: Any) -> Any:
        """Build features for a symbol.
        
        Args:
            symbol: Trading symbol
            candles_df: DataFrame with candle data
            
        Returns:
            DataFrame with features
        """
        # TODO: Implement actual feature building
        # For now, return empty DataFrame
        import pandas as pd
        return pd.DataFrame()
    
    def _train_model(self, symbol: str, features_df: Any, candles_df: Any) -> tuple[Any, Dict[str, Any]]:
        """Train model for a symbol.
        
        Args:
            symbol: Trading symbol
            features_df: DataFrame with features
            candles_df: DataFrame with candle data
            
        Returns:
            Tuple of (model, metrics)
        """
        import pandas as pd
        import numpy as np
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_squared_error, r2_score
        
        if features_df.empty:
            return None, {
                "sharpe": 0.0,
                "hit_rate": 0.0,
                "net_pnl_pct": 0.0,
                "max_drawdown_pct": 0.0,
                "sample_size": 0,
            }
        
        # Get model type from config
        model_type = self.config.get("engine", {}).get("model_type", "xgboost")
        
        # Create target (next period return)
        if "close" in candles_df.columns:
            # Align target with features
            target = candles_df['close'].pct_change().shift(-1)
            # Align indices
            aligned_target = target.loc[features_df.index]
            aligned_target = aligned_target.dropna()
            aligned_features = features_df.loc[aligned_target.index]
        else:
            # Fallback: use first feature as target
            aligned_features = features_df.iloc[:-1]
            aligned_target = features_df.iloc[1:, 0] if len(features_df.columns) > 0 else pd.Series([0] * (len(features_df) - 1))
        
        if aligned_features.empty or len(aligned_target) == 0:
            return None, {
                "sharpe": 0.0,
                "hit_rate": 0.0,
                "net_pnl_pct": 0.0,
                "max_drawdown_pct": 0.0,
                "sample_size": 0,
            }
        
        # Remove non-numeric columns
        numeric_features = aligned_features.select_dtypes(include=[np.number])
        if numeric_features.empty:
            return None, {
                "sharpe": 0.0,
                "hit_rate": 0.0,
                "net_pnl_pct": 0.0,
                "max_drawdown_pct": 0.0,
                "sample_size": 0,
            }
        
        # Split into train/test (temporal split)
        split_idx = int(len(numeric_features) * 0.8)
        X_train = numeric_features.iloc[:split_idx]
        X_test = numeric_features.iloc[split_idx:]
        y_train = aligned_target.iloc[:split_idx]
        y_test = aligned_target.iloc[split_idx:]
        
        # Train model
        model = None
        try:
            if model_type == "xgboost":
                try:
                    from xgboost import XGBRegressor  # type: ignore[reportMissingImports]
                    model = XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42, n_jobs=-1)
                    model.fit(X_train.values, y_train.values)
                except ImportError:
                    logger.warning("xgboost_not_available", using_fallback=True)
                    model_type = "linear"
            
            if model is None or model_type == "linear":
                from sklearn.linear_model import LinearRegression
                model = LinearRegression()
                model.fit(X_train.values, y_train.values)
        except Exception as e:
            logger.error("model_training_failed", symbol=symbol, error=str(e))
            return None, {
                "sharpe": 0.0,
                "hit_rate": 0.0,
                "net_pnl_pct": 0.0,
                "max_drawdown_pct": 0.0,
                "sample_size": len(X_train),
            }
        
        # Make predictions
        y_pred_train = pd.Series(model.predict(X_train.values), index=X_train.index)
        y_pred_test = pd.Series(model.predict(X_test.values), index=X_test.index)
        
        # Calculate metrics
        train_mse = mean_squared_error(y_train, y_pred_train)
        test_mse = mean_squared_error(y_test, y_pred_test)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        
        # Hit rate: % of times prediction direction matches actual return direction
        correct_direction = ((y_pred_test > 0) == (y_test > 0)).sum()
        hit_rate = correct_direction / len(y_test) if len(y_test) > 0 else 0.0
        
        # Sharpe Ratio: mean return / std dev of return
        mean_return = y_test.mean()
        std_return = y_test.std()
        sharpe = mean_return / std_return if std_return > 0 else 0.0
        
        # Max drawdown (simplified)
        cumulative_returns = (1 + y_test).cumprod()
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown_pct = abs(drawdown.min()) * 100 if len(drawdown) > 0 else 0.0
        
        metrics = {
            "sharpe": float(sharpe),
            "hit_rate": float(hit_rate),
            "net_pnl_pct": float(mean_return * 100),
            "max_drawdown_pct": float(max_drawdown_pct),
            "sample_size": len(X_train),
            "test_r2": float(test_r2),
            "test_mse": float(test_mse),
        }
        
        return model, metrics
    
    def _create_cost_model(self, symbol: str, slippage_bps: float, fit_date: datetime) -> CostModel:
        """Create cost model for a symbol.
        
        Args:
            symbol: Trading symbol
            slippage_bps: Slippage in basis points per sigma
            fit_date: Fit date
            
        Returns:
            CostModel instance
        """
        costs_config = self.config.get("costs", {})
        
        return CostModel(
            symbol=symbol,
            taker_fee_bps=costs_config.get("taker_fee_bps", 4.0),
            maker_fee_bps=costs_config.get("maker_fee_bps", 2.0),
            median_spread_bps=costs_config.get("median_spread_bps", 5.0),
            slippage_bps_per_sigma=slippage_bps,
            min_notional=10.0,
            step_size=0.001,
            last_updated_utc=fit_date,
        )
    
    def _check_trade_ok(self, metrics: Dict[str, Any], cost_model: CostModel) -> bool:
        """Check if symbol passes trade_ok gates.
        
        Args:
            metrics: Metrics dictionary
            cost_model: Cost model
            
        Returns:
            True if trade_ok
        """
        # Trade OK if: net_pnl_pct > 0, sample_size > 100, sharpe > 0.5
        net_pnl_pct = metrics.get("net_pnl_pct", 0.0)
        sample_size = metrics.get("sample_size", 0)
        sharpe = metrics.get("sharpe", 0.0)
        hit_rate = metrics.get("hit_rate", 0.0)
        max_drawdown_pct = metrics.get("max_drawdown_pct", 0.0)
        
        trade_ok = (
            net_pnl_pct > 0 and
            sample_size > 100 and
            sharpe > 0.5 and
            hit_rate > 0.45 and
            max_drawdown_pct < 20.0
        )
        
        return trade_ok
    
    def _export_artifacts(
        self,
        symbol: str,
        model: Any,
        metrics: Dict[str, Any],
        cost_model: CostModel,
        date_str: str,
    ) -> Dict[str, str]:
        """Export artifacts for a symbol.
        
        Args:
            symbol: Trading symbol
            model: Trained model
            metrics: Metrics dictionary
            cost_model: Cost model
            date_str: Date string in YYYYMMDD format
            
        Returns:
            Dictionary with artifact paths
        """
        # Create output directory
        output_dir = Path("models") / symbol / f"baseline_{date_str}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = output_dir / "model.bin"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Save metrics
        metrics_path = output_dir / "metrics.json"
        import json
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Save cost model
        costs_path = output_dir / "costs.json"
        with open(costs_path, 'w') as f:
            json.dump(cost_model.to_dict(), f, indent=2)
        
        # Save feature recipe
        feature_recipe = FeatureRecipe(
            symbol=symbol,
            timeframes=["1h"],
            indicators={},
            fill_rules={"strategy": "forward_fill"},
            normalization={"type": "standard", "scaler": "StandardScaler"},
        )
        feature_recipe.hash = feature_recipe.compute_hash()
        
        features_path = output_dir / "features.json"
        with open(features_path, 'w') as f:
            json.dump(feature_recipe.to_dict(), f, indent=2)
        
        # Upload to Dropbox if available
        if self.dropbox_sync:
            # Upload model
            dropbox_model_path = f"/{self.config.get('general', {}).get('dropbox_root', 'Huracan/').strip('/')}/models/{symbol}/baseline_{date_str}/model.bin"
            self.dropbox_sync.upload_file(str(model_path), dropbox_model_path, overwrite=True)
            
            # Upload metrics
            dropbox_metrics_path = dropbox_model_path.replace("model.bin", "metrics.json")
            self.dropbox_sync.upload_file(str(metrics_path), dropbox_metrics_path, overwrite=True)
            
            # Upload costs
            dropbox_costs_path = dropbox_model_path.replace("model.bin", "costs.json")
            self.dropbox_sync.upload_file(str(costs_path), dropbox_costs_path, overwrite=True)
            
            # Upload features
            dropbox_features_path = dropbox_model_path.replace("model.bin", "features.json")
            self.dropbox_sync.upload_file(str(features_path), dropbox_features_path, overwrite=True)
        
        return {
            "model_path": str(model_path),
            "metrics_path": str(metrics_path),
            "costs_path": str(costs_path),
            "features_path": str(features_path),
            "feature_recipe_hash": feature_recipe.hash,
        }
    
    def _get_feature_importance(self, model: Any, features_df: Any) -> Dict[str, float]:
        """Get feature importance from model.
        
        Args:
            model: Trained model
            features_df: DataFrame with features
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        # TODO: Implement actual feature importance extraction
        # For now, return dummy importance
        return {f"feature_{i}": 1.0 / len(features_df.columns) for i in range(len(features_df.columns))}
    
    def _export_roster(self, results: Dict[str, Dict[str, Any]]) -> None:
        """Export roster.json for Hamilton.
        
        Args:
            results: Training results dictionary
        """
        symbols_data = []
        
        for symbol, result in results.items():
            if result.get("status") != "completed":
                continue
            
            metrics = result.get("metrics", {})
            cost_model_dict = result.get("cost_model", {})
            
            # Create PerCoinMetrics
            per_coin_metrics = PerCoinMetrics(
                symbol=symbol,
                sample_size=metrics.get("sample_size", 0),
                gross_pnl_pct=metrics.get("gross_pnl_pct", 0.0),
                net_pnl_pct=metrics.get("net_pnl_pct", 0.0),
                sharpe=metrics.get("sharpe", 0.0),
                hit_rate=metrics.get("hit_rate", 0.0),
                max_drawdown_pct=metrics.get("max_drawdown_pct", 0.0),
                avg_trade_bps=metrics.get("avg_trade_bps", 0.0),
                costs_bps_used=cost_model_dict,
            )
            
            # Create CostModel
            cost_model = CostModel(
                symbol=symbol,
                taker_fee_bps=cost_model_dict.get("taker_fee_bps", 4.0),
                maker_fee_bps=cost_model_dict.get("maker_fee_bps", 2.0),
                median_spread_bps=cost_model_dict.get("median_spread_bps", 5.0),
                slippage_bps_per_sigma=cost_model_dict.get("slippage_bps_per_sigma", 2.0),
                min_notional=cost_model_dict.get("min_notional", 10.0),
                step_size=cost_model_dict.get("step_size", 0.001),
                last_updated_utc=datetime.fromisoformat(cost_model_dict.get("last_updated_utc", datetime.now(timezone.utc).isoformat())),
            )
            
            # Create roster entry
            model_path = result.get("artifacts", {}).get("model_path", f"models/{symbol}/model.bin")
            entry = self.roster_exporter.create_roster_entry(
                symbol=symbol,
                model_path=model_path,
                metrics=per_coin_metrics,
                cost_model=cost_model,
                rank=0,  # Will be set by rank_symbols
            )
            
            symbols_data.append(entry)
        
        # Export roster
        self.roster_exporter.export_roster(symbols_data)

