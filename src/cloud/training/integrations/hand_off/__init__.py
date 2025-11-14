"""
Hamilton Hand-off Integration

Deploys approved models from Engine to Hamilton for live trading.

Usage:
    from src.cloud.training.integrations.hand_off import HamiltonDeployer

    deployer = HamiltonDeployer()
    result = deployer.deploy_model("btc_v48", "BTC", activate_immediately=True)
"""

from .hamilton_deployer import HamiltonDeployer, DeploymentResult

__all__ = ["HamiltonDeployer", "DeploymentResult"]
