"""
Hamilton Hand-off Deployer

Deploys approved models from Engine to Hamilton for live trading.

Flow:
1. Engine trains model
2. Model passes all gates
3. Model registered in Unified Registry
4. Deployer exports model + config to Hamilton
5. Hamilton picks up new model
6. Live trades → Feedback collector → Back to Engine

Usage:
    from integration.hand_off import HamiltonDeployer

    deployer = HamiltonDeployer(
        registry_dsn="postgresql://...",
        hamilton_config_dir="/path/to/hamilton/configs",
        hamilton_models_dir="/path/to/hamilton/models"
    )

    # Deploy approved model
    deployment = deployer.deploy_model(
        model_id="btc_trend_v48",
        symbol="BTC",
        activate_immediately=True
    )

    print(f"Deployed to: {deployment.config_path}")
    print(f"Active: {deployment.is_active}")
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional
import json
import shutil

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class DeploymentResult:
    """Model deployment result"""
    model_id: str
    symbol: str
    deployment_time: datetime

    # Paths
    model_artifact_path: str
    config_path: str
    manifest_path: str

    # Status
    is_active: bool
    previous_model_id: Optional[str] = None

    # Metadata
    gate_score: Optional[float] = None
    deployment_notes: str = ""


class HamiltonDeployer:
    """
    Hamilton Hand-off Deployer

    Deploys models from Engine to Hamilton for live trading.

    Example:
        deployer = HamiltonDeployer(
            hamilton_config_dir="/hamilton/configs",
            hamilton_models_dir="/hamilton/models"
        )

        # Deploy model
        result = deployer.deploy_model(
            model_id="btc_v48",
            symbol="BTC",
            activate_immediately=True
        )

        # List active models
        active = deployer.get_active_models()
    """

    def __init__(
        self,
        registry_dsn: Optional[str] = None,
        hamilton_config_dir: str = "/hamilton/configs",
        hamilton_models_dir: str = "/hamilton/models",
        engine_models_dir: str = "/engine/models"
    ):
        """
        Initialize Hamilton deployer

        Args:
            registry_dsn: PostgreSQL DSN for model registry
            hamilton_config_dir: Hamilton config directory
            hamilton_models_dir: Hamilton models directory
            engine_models_dir: Engine models directory
        """
        self.registry_dsn = registry_dsn
        self.hamilton_config_dir = Path(hamilton_config_dir)
        self.hamilton_models_dir = Path(hamilton_models_dir)
        self.engine_models_dir = Path(engine_models_dir)

        # Create directories if they don't exist
        self.hamilton_config_dir.mkdir(parents=True, exist_ok=True)
        self.hamilton_models_dir.mkdir(parents=True, exist_ok=True)

    def deploy_model(
        self,
        model_id: str,
        symbol: str,
        activate_immediately: bool = False,
        deployment_notes: str = ""
    ) -> DeploymentResult:
        """
        Deploy model to Hamilton

        Args:
            model_id: Model identifier
            symbol: Trading symbol
            activate_immediately: Activate for live trading immediately
            deployment_notes: Optional deployment notes

        Returns:
            DeploymentResult
        """
        logger.info(
            "deploying_model_to_hamilton",
            model_id=model_id,
            symbol=symbol,
            activate=activate_immediately
        )

        # 1. Get model from registry
        model_info = self._get_model_from_registry(model_id)

        if not model_info:
            raise ValueError(f"Model {model_id} not found in registry")

        # Check gate approval
        if model_info.get('gate_verdict') != 'APPROVED':
            raise ValueError(
                f"Model {model_id} not approved by gates: "
                f"{model_info.get('gate_verdict')}"
            )

        # 2. Copy model artifact
        model_artifact_path = self._copy_model_artifact(model_id, symbol)

        # 3. Generate Hamilton config
        config_path = self._generate_hamilton_config(
            model_id, symbol, model_info
        )

        # 4. Generate run manifest
        manifest_path = self._copy_run_manifest(model_id, symbol)

        # 5. Activate if requested
        previous_model_id = None
        is_active = False

        if activate_immediately:
            previous_model_id = self._activate_model(symbol, model_id)
            is_active = True

        deployment_time = datetime.utcnow()

        result = DeploymentResult(
            model_id=model_id,
            symbol=symbol,
            deployment_time=deployment_time,
            model_artifact_path=str(model_artifact_path),
            config_path=str(config_path),
            manifest_path=str(manifest_path),
            is_active=is_active,
            previous_model_id=previous_model_id,
            gate_score=model_info.get('gate_score'),
            deployment_notes=deployment_notes
        )

        logger.info(
            "model_deployed_to_hamilton",
            model_id=model_id,
            symbol=symbol,
            active=is_active,
            config_path=str(config_path)
        )

        return result

    def _get_model_from_registry(self, model_id: str) -> Optional[dict]:
        """Get model info from registry"""
        # TODO: Query PostgreSQL registry
        # For now, return mock data
        return {
            "model_id": model_id,
            "gate_verdict": "APPROVED",
            "gate_score": 0.85,
            "sharpe": 1.5,
            "win_rate": 0.55
        }

    def _copy_model_artifact(self, model_id: str, symbol: str) -> Path:
        """
        Copy model artifact to Hamilton directory

        Args:
            model_id: Model ID
            symbol: Symbol

        Returns:
            Path to copied model
        """
        # Source path in Engine
        source_path = self.engine_models_dir / f"{model_id}.pkl"

        # Destination path in Hamilton
        dest_dir = self.hamilton_models_dir / symbol.lower()
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / f"{model_id}.pkl"

        # Copy if source exists
        if source_path.exists():
            shutil.copy2(source_path, dest_path)
            logger.info("model_artifact_copied", source=str(source_path), dest=str(dest_path))
        else:
            logger.warning("model_artifact_not_found", path=str(source_path))

        return dest_path

    def _generate_hamilton_config(
        self,
        model_id: str,
        symbol: str,
        model_info: dict
    ) -> Path:
        """
        Generate Hamilton configuration file

        Args:
            model_id: Model ID
            symbol: Symbol
            model_info: Model metadata

        Returns:
            Path to config file
        """
        config = {
            "model_id": model_id,
            "symbol": symbol,
            "model_path": f"models/{symbol.lower()}/{model_id}.pkl",
            "gate_score": model_info.get("gate_score"),
            "performance": {
                "sharpe": model_info.get("sharpe"),
                "win_rate": model_info.get("win_rate")
            },
            "risk_limits": {
                "max_position_pct": 10.0,
                "max_leverage": 2.0
            },
            "deployed_at": datetime.utcnow().isoformat(),
            "feedback_enabled": True
        }

        # Write config
        config_path = self.hamilton_config_dir / f"{symbol.lower()}_{model_id}.json"

        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        logger.info("hamilton_config_generated", path=str(config_path))

        return config_path

    def _copy_run_manifest(self, model_id: str, symbol: str) -> Path:
        """Copy run manifest to Hamilton"""
        # Source manifest
        source_path = Path(f"observability/run_manifest/manifests/{model_id}.json")

        # Destination
        dest_dir = self.hamilton_config_dir / "manifests"
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / f"{symbol.lower()}_{model_id}_manifest.json"

        if source_path.exists():
            shutil.copy2(source_path, dest_path)
            logger.info("manifest_copied", dest=str(dest_path))
        else:
            logger.warning("manifest_not_found", path=str(source_path))

        return dest_path

    def _activate_model(self, symbol: str, model_id: str) -> Optional[str]:
        """
        Activate model for live trading

        Args:
            symbol: Symbol
            model_id: Model to activate

        Returns:
            Previous active model ID (if any)
        """
        # Active model file
        active_file = self.hamilton_config_dir / f"{symbol.lower()}_active.txt"

        # Get previous active model
        previous_model_id = None
        if active_file.exists():
            previous_model_id = active_file.read_text().strip()

        # Write new active model
        active_file.write_text(model_id)

        logger.info(
            "model_activated",
            symbol=symbol,
            model_id=model_id,
            previous=previous_model_id
        )

        return previous_model_id

    def get_active_models(self) -> dict:
        """
        Get currently active models

        Returns:
            Dict of {symbol: model_id}
        """
        active_models = {}

        for active_file in self.hamilton_config_dir.glob("*_active.txt"):
            symbol = active_file.stem.replace("_active", "").upper()
            model_id = active_file.read_text().strip()
            active_models[symbol] = model_id

        return active_models

    def rollback_model(
        self,
        symbol: str,
        target_model_id: str,
        reason: str
    ) -> DeploymentResult:
        """
        Rollback to previous model

        Args:
            symbol: Symbol
            target_model_id: Model to rollback to
            reason: Rollback reason

        Returns:
            DeploymentResult
        """
        logger.warning(
            "rolling_back_model",
            symbol=symbol,
            target_model_id=target_model_id,
            reason=reason
        )

        # Activate target model
        previous = self._activate_model(symbol, target_model_id)

        return DeploymentResult(
            model_id=target_model_id,
            symbol=symbol,
            deployment_time=datetime.utcnow(),
            model_artifact_path="",
            config_path="",
            manifest_path="",
            is_active=True,
            previous_model_id=previous,
            deployment_notes=f"ROLLBACK: {reason}"
        )
