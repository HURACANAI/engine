"""
Training Pipeline for Scalable Architecture

Implements the 9-step training flow:
1. Build daily coin universe
2. Ingest and validate data
3. Generate features
4. Label with forward returns
5. Train per engine and per regime
6. Score with edge after costs
7. Run consensus
8. Shadow test challengers
9. Export champions

Author: Huracan Engine Team
Date: 2025-01-27
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable

import structlog

from ..consensus.consensus_service import ConsensusService, EngineVote
from ..regime.regime_gate import RegimeGate, RegimeType, RegimeGateConfig
from ..costs.real_time_cost_model import RealTimeCostModel
from ..export.dropbox_publisher import DropboxPublisher, ExportBundle, ModelManifest
from ..models.model_registry import ModelRegistry
from .orchestrator import TrainingOrchestrator, TrainingJob, TrainingConfig, TrainingBackend

logger = structlog.get_logger(__name__)


@dataclass
class TrainingPipelineConfig:
    """Training pipeline configuration."""
    lookback_days: int = 150
    horizons: List[str] = field(default_factory=lambda: ["1h", "4h", "1d"])
    risk_preset: str = "balanced"
    dry_run: bool = False
    min_liquidity_gbp: float = 10000000.0
    max_spread_bps: float = 8.0
    min_edge_after_cost_bps: float = 5.0
    training_backend: str = "asyncio"
    max_concurrent_jobs: int = 10
    dropbox_access_token: Optional[str] = None
    dropbox_base_path: str = "/HuracanEngine"


class TrainingPipeline:
    """
    Training pipeline for scalable training.
    
    Implements 9-step flow:
    1. Build daily coin universe
    2. Ingest and validate data
    3. Generate features
    4. Label with forward returns
    5. Train per engine and per regime
    6. Score with edge after costs
    7. Run consensus
    8. Shadow test challengers
    9. Export champions
    """
    
    def __init__(
        self,
        config: TrainingPipelineConfig,
        data_loader: Optional[Callable] = None,
        feature_builder: Optional[Callable] = None,
        model_trainer: Optional[Callable] = None,
        validator: Optional[Callable] = None,
    ):
        """
        Initialize training pipeline.
        
        Args:
            config: Pipeline configuration
            data_loader: Data loader function
            feature_builder: Feature builder function
            model_trainer: Model trainer function
            validator: Validator function
        """
        self.config = config
        
        # Dependencies
        self.data_loader = data_loader
        self.feature_builder = feature_builder
        self.model_trainer = model_trainer
        self.validator = validator
        
        # Services
        self.consensus_service = ConsensusService()
        self.regime_gate = RegimeGate(RegimeGateConfig())
        self.cost_model = RealTimeCostModel(
            min_edge_after_cost_bps=config.min_edge_after_cost_bps,
        )
        self.model_registry = ModelRegistry()
        
        # Dropbox publisher
        if config.dropbox_access_token:
            self.dropbox_publisher = DropboxPublisher(
                access_token=config.dropbox_access_token,
                base_path=config.dropbox_base_path,
                dry_run=config.dry_run,
            )
        else:
            self.dropbox_publisher = None
        
        # Training orchestrator
        training_config = TrainingConfig(
            backend=TrainingBackend.ASYNCIO if config.training_backend == "asyncio" else TrainingBackend.RAY,
            max_concurrent_jobs=config.max_concurrent_jobs,
            lookback_days=config.lookback_days,
            horizons=config.horizons,
            risk_preset=config.risk_preset,
            dry_run=config.dry_run,
        )
        self.orchestrator = TrainingOrchestrator(training_config)
        
        # State
        self.coin_universe: List[str] = []
        self.training_results: Dict[str, Any] = {}
        self.champions: Dict[str, Dict[str, Any]] = {}  # coin -> horizon -> champion
        
        logger.info(
            "training_pipeline_initialized",
            lookback_days=config.lookback_days,
            horizons=config.horizons,
            dry_run=config.dry_run,
        )
    
    async def run(self) -> Dict[str, Any]:
        """
        Run the training pipeline.
        
        Returns:
            Dictionary with pipeline results
        """
        logger.info("training_pipeline_started")
        start_time = time.time()
        
        try:
            # Step 1: Build daily coin universe
            self.coin_universe = await self.step1_build_coin_universe()
            logger.info("step1_completed", coin_count=len(self.coin_universe))
            
            # Step 2: Ingest and validate data
            data_results = await self.step2_ingest_and_validate_data()
            logger.info("step2_completed", coins_processed=len(data_results))
            
            # Step 3: Generate features
            feature_results = await self.step3_generate_features(data_results)
            logger.info("step3_completed", coins_processed=len(feature_results))
            
            # Step 4: Label with forward returns
            labeled_data = await self.step4_label_forward_returns(feature_results)
            logger.info("step4_completed", coins_processed=len(labeled_data))
            
            # Step 5: Train per engine and per regime
            training_results = await self.step5_train_models(labeled_data)
            logger.info("step5_completed", models_trained=len(training_results))
            
            # Step 6: Score with edge after costs
            scored_models = await self.step6_score_edge_after_costs(training_results)
            logger.info("step6_completed", models_scored=len(scored_models))
            
            # Step 7: Run consensus
            consensus_results = await self.step7_run_consensus(scored_models)
            logger.info("step7_completed", consensus_results=len(consensus_results))
            
            # Step 8: Shadow test challengers
            champions = await self.step8_shadow_test_challengers(consensus_results)
            logger.info("step8_completed", champions=len(champions))
            
            # Step 9: Export champions
            export_results = await self.step9_export_champions(champions)
            logger.info("step9_completed", exports=len(export_results))
            
            # Generate summary
            summary = self._generate_summary(
                coin_universe=self.coin_universe,
                data_results=data_results,
                training_results=training_results,
                champions=champions,
                export_results=export_results,
            )
            
            # Publish summary
            if self.dropbox_publisher:
                self.dropbox_publisher.publish_summary(summary)
            
            duration = time.time() - start_time
            logger.info(
                "training_pipeline_completed",
                duration_seconds=duration,
                coins_processed=len(self.coin_universe),
                champions=len(champions),
            )
            
            return {
                "success": True,
                "duration_seconds": duration,
                "coin_universe": self.coin_universe,
                "champions": champions,
                "export_results": export_results,
                "summary": summary,
            }
        
        except Exception as e:
            logger.error("training_pipeline_failed", error=str(e))
            return {
                "success": False,
                "error": str(e),
            }
    
    async def step1_build_coin_universe(self) -> List[str]:
        """Step 1: Build daily coin universe from liquidity and spread filters."""
        # Placeholder implementation
        # In production, fetch from exchange and filter by liquidity/spread
        logger.info("building_coin_universe")
        
        # Mock implementation
        coins = ["BTC", "ETH", "SOL", "BNB", "XRP"]  # Would fetch from Binance
        return coins
    
    async def step2_ingest_and_validate_data(self) -> Dict[str, Any]:
        """Step 2: Ingest and validate data. Repair small gaps. Drop large gaps. Tag vendors."""
        if not self.data_loader:
            raise ValueError("Data loader not provided")
        
        results = {}
        for coin in self.coin_universe:
            try:
                data = await self.data_loader(coin, self.config.lookback_days)
                # Validate data
                validation_result = self._validate_data(data, coin)
                if validation_result["valid"]:
                    results[coin] = data
                else:
                    logger.warning("data_validation_failed", coin=coin, reason=validation_result["reason"])
            except Exception as e:
                logger.error("data_ingestion_failed", coin=coin, error=str(e))
        
        return results
    
    async def step3_generate_features(self, data_results: Dict[str, Any]) -> Dict[str, Any]:
        """Step 3: Generate features. Store in versioned feature store."""
        if not self.feature_builder:
            raise ValueError("Feature builder not provided")
        
        results = {}
        for coin, data in data_results.items():
            try:
                features = await self.feature_builder(coin, data)
                results[coin] = {
                    "data": data,
                    "features": features,
                }
            except Exception as e:
                logger.error("feature_generation_failed", coin=coin, error=str(e))
        
        return results
    
    async def step4_label_forward_returns(self, feature_results: Dict[str, Any]) -> Dict[str, Any]:
        """Step 4: Label with forward returns that match execution latency."""
        results = {}
        for coin, result in feature_results.items():
            try:
                # Label with forward returns for each horizon
                labeled_data = {}
                for horizon in self.config.horizons:
                    # Convert horizon to seconds (e.g., "1h" -> 3600)
                    horizon_seconds = self._horizon_to_seconds(horizon)
                    labels = self._compute_forward_returns(result["data"], horizon_seconds)
                    labeled_data[horizon] = {
                        "features": result["features"],
                        "labels": labels,
                    }
                results[coin] = labeled_data
            except Exception as e:
                logger.error("labeling_failed", coin=coin, error=str(e))
        
        return results
    
    async def step5_train_models(self, labeled_data: Dict[str, Any]) -> Dict[str, Any]:
        """Step 5: Train per engine and per regime with purged walk forward splits."""
        if not self.model_trainer:
            raise ValueError("Model trainer not provided")
        
        # Create training jobs
        jobs = []
        for coin, horizons_data in labeled_data.items():
            for horizon, data in horizons_data.items():
                # Train for each regime
                for regime in RegimeType:
                    job = TrainingJob(
                        coin=coin,
                        horizon=horizon,
                        regime=regime.value,
                        priority=1,
                    )
                    jobs.append(job)
        
        # Add jobs to orchestrator
        self.orchestrator.add_jobs(jobs)
        
        # Register training function
        self.orchestrator.register_training_function(self.model_trainer)
        
        # Train all jobs
        await self.orchestrator.initialize()
        training_results = await self.orchestrator.train_all()
        await self.orchestrator.shutdown()
        
        return training_results
    
    async def step6_score_edge_after_costs(self, training_results: Dict[str, Any]) -> Dict[str, Any]:
        """Step 6: Score with edge after costs. Costs include spread, fees, slippage, funding."""
        scored_models = {}
        
        for job_id, result in training_results.get("results", {}).items():
            if not result.get("success"):
                continue
            
            job = self.orchestrator.get_job_status(job_id)
            if not job:
                continue
            
            coin = job.coin
            horizon = job.horizon
            
            # Get edge before costs
            edge_bps = result.get("edge_bps", 0.0)
            
            # Calculate edge after costs
            edge_after_cost = self.cost_model.calculate_edge_after_cost(
                symbol=coin,
                edge_bps=edge_bps,
                use_maker=True,
                include_funding=True,
            )
            
            # Check if meets threshold
            if edge_after_cost >= self.config.min_edge_after_cost_bps:
                scored_models[job_id] = {
                    "coin": coin,
                    "horizon": horizon,
                    "regime": job.regime,
                    "edge_bps": edge_bps,
                    "edge_after_cost_bps": edge_after_cost,
                    "model": result.get("model"),
                    "metrics": result.get("metrics", {}),
                }
        
        return scored_models
    
    async def step7_run_consensus(self, scored_models: Dict[str, Any]) -> Dict[str, Any]:
        """Step 7: Run consensus. Weight by recent reliability. Penalize correlation. Produce score S."""
        consensus_results = {}
        
        # Group by coin and horizon
        grouped_models = {}
        for job_id, model_data in scored_models.items():
            key = (model_data["coin"], model_data["horizon"])
            if key not in grouped_models:
                grouped_models[key] = []
            grouped_models[key].append(model_data)
        
        # Run consensus for each coin/horizon
        for (coin, horizon), models in grouped_models.items():
            # Create engine votes
            votes = []
            for model_data in models:
                vote = EngineVote(
                    engine_type=model_data.get("regime", "unknown"),
                    direction="long" if model_data["edge_after_cost_bps"] > 0 else "short",
                    confidence=min(1.0, abs(model_data["edge_after_cost_bps"]) / 100.0),
                    edge_bps=model_data["edge_after_cost_bps"],
                )
                votes.append(vote)
            
            # Compute consensus
            consensus = self.consensus_service.compute_consensus(votes)
            
            consensus_results[f"{coin}_{horizon}"] = {
                "coin": coin,
                "horizon": horizon,
                "consensus": consensus,
                "models": models,
            }
        
        return consensus_results
    
    async def step8_shadow_test_challengers(self, consensus_results: Dict[str, Any]) -> Dict[str, Any]:
        """Step 8: Shadow test challengers against last champion. Promote only if statistically better."""
        champions = {}
        
        for key, result in consensus_results.items():
            coin = result["coin"]
            horizon = result["horizon"]
            consensus = result["consensus"]
            
            # Get current champion
            current_champion = self.model_registry.get_model_metadata(coin)
            
            # Select best model from consensus
            best_model = None
            best_score = -float('inf')
            
            for model_data in result["models"]:
                score = model_data["edge_after_cost_bps"] * consensus.consensus_score
                if score > best_score:
                    best_score = score
                    best_model = model_data
            
            if best_model:
                # Compare with current champion
                if current_champion:
                    # Shadow test (simplified - would run actual backtest)
                    if best_score > current_champion.sharpe_ratio * 10:  # Simplified comparison
                        champions[f"{coin}_{horizon}"] = best_model
                else:
                    # No current champion, promote
                    champions[f"{coin}_{horizon}"] = best_model
        
        return champions
    
    async def step9_export_champions(self, champions: Dict[str, Any]) -> Dict[str, Any]:
        """Step 9: Export champions and full artifacts to Dropbox with manifest."""
        if not self.dropbox_publisher:
            logger.warning("dropbox_publisher_not_available")
            return {}
        
        export_results = {}
        
        for key, champion in champions.items():
            coin = champion["coin"]
            horizon = champion["horizon"]
            
            try:
                # Create export bundle
                bundle = await self._create_export_bundle(coin, horizon, champion)
                
                # Publish to Dropbox
                success = self.dropbox_publisher.publish_bundle(bundle)
                
                export_results[key] = {
                    "success": success,
                    "coin": coin,
                    "horizon": horizon,
                }
            
            except Exception as e:
                logger.error("export_failed", coin=coin, horizon=horizon, error=str(e))
                export_results[key] = {
                    "success": False,
                    "error": str(e),
                }
        
        return export_results
    
    def _validate_data(self, data: Any, coin: str) -> Dict[str, Any]:
        """Validate data for a coin."""
        # Placeholder implementation
        return {"valid": True, "reason": ""}
    
    def _horizon_to_seconds(self, horizon: str) -> int:
        """Convert horizon string to seconds."""
        if horizon.endswith("h"):
            return int(horizon[:-1]) * 3600
        elif horizon.endswith("d"):
            return int(horizon[:-1]) * 86400
        else:
            return 3600  # Default to 1 hour
    
    def _compute_forward_returns(self, data: Any, horizon_seconds: int) -> Any:
        """Compute forward returns for labeling."""
        # Placeholder implementation
        return []
    
    async def _create_export_bundle(self, coin: str, horizon: str, champion: Dict[str, Any]) -> ExportBundle:
        """Create export bundle for a champion."""
        # Create model manifest
        manifest = ModelManifest(
            coin=coin,
            horizon=horizon,
            version=1,  # Would get from model registry
            training_window_start=datetime.now().isoformat(),
            training_window_end=datetime.now().isoformat(),
            features_hash="",  # Would compute from features
            code_hash="",  # Would compute from code
            timestamp=datetime.now().isoformat(),
            model_path=f"/models/{coin}/{horizon}/model.pkl",
            metrics=champion.get("metrics", {}),
            regime=champion.get("regime"),
        )
        
        # Create export bundle
        bundle = ExportBundle(
            coin=coin,
            horizon=horizon,
            champion_model=Path(f"/tmp/{coin}_{horizon}_model.pkl"),  # Would be actual model path
            metrics_report=Path(f"/tmp/{coin}_{horizon}_metrics.json"),
            cost_report=Path(f"/tmp/{coin}_{horizon}_cost.json"),
            decision_logs=Path(f"/tmp/{coin}_{horizon}_logs.json"),
            regime_map=Path(f"/tmp/{coin}_{horizon}_regime.json"),
            data_integrity_report=Path(f"/tmp/{coin}_{horizon}_integrity.json"),
            model_manifest=manifest,
        )
        
        return bundle
    
    def _generate_summary(self, **kwargs: Any) -> Dict[str, Any]:
        """Generate pipeline summary."""
        return {
            "timestamp": datetime.now().isoformat(),
            "coin_universe_size": len(kwargs.get("coin_universe", [])),
            "champions_count": len(kwargs.get("champions", {})),
            "export_results": kwargs.get("export_results", {}),
        }

