"""
Component-Specific Training Configurations

Each trading component needs different historical data:
- Fast components (scalp) → short windows (microstructure changes fast)
- Slow components (regime) → long windows (need full market cycles)

This prevents "one size fits all" training that dilutes signal.
"""

from dataclasses import dataclass
from typing import Literal

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class ComponentConfig:
    """Configuration for a single training component."""

    name: str
    lookback_days: int
    timeframe: str  # '1m', '5m', '1h', '1d'
    recency_halflife_days: float
    description: str

    # Training parameters
    min_samples: int = 1000
    max_samples: int = 100000

    # Validation parameters
    walk_forward_train_days: int = 30
    walk_forward_test_days: int = 1
    embargo_minutes: int = 30

    def __post_init__(self):
        """Validate configuration."""
        if self.lookback_days < self.walk_forward_train_days:
            logger.warning(
                "lookback_shorter_than_walk_forward",
                component=self.name,
                lookback_days=self.lookback_days,
                wf_train_days=self.walk_forward_train_days
            )

        logger.info(
            "component_config_created",
            component=self.name,
            lookback_days=self.lookback_days,
            timeframe=self.timeframe,
            halflife_days=self.recency_halflife_days
        )


class ScalpCoreConfig(ComponentConfig):
    """
    Scalp Core - The main entry signal.

    Philosophy:
    - Shortest window (60 days) because microstructure changes FAST
    - New algos, fee changes, venue updates → old patterns fade quickly
    - Aggressive recency weighting (10-day halflife)
    - 1-minute granularity for precise entries
    """

    def __init__(self):
        super().__init__(
            name='scalp_core',
            lookback_days=60,
            timeframe='1m',
            recency_halflife_days=10.0,
            description='Primary entry signal for scalp trades',
            min_samples=2000,  # At least 2000 labeled trades
            max_samples=50000,
            walk_forward_train_days=30,
            walk_forward_test_days=1,
            embargo_minutes=30
        )


class ConfirmFilterConfig(ComponentConfig):
    """
    Confirm Filter - Multi-timeframe confirmation.

    Philosophy:
    - Medium window (120 days) for more regime variety
    - Needs to see different volatility regimes, trending vs ranging
    - 5-minute timeframe (reduces noise vs 1m)
    - Medium recency weighting (20-day halflife)
    """

    def __init__(self):
        super().__init__(
            name='confirm_filter',
            lookback_days=120,
            timeframe='5m',
            recency_halflife_days=20.0,
            description='Multi-timeframe confirmation filter',
            min_samples=3000,
            max_samples=75000,
            walk_forward_train_days=45,
            walk_forward_test_days=1,
            embargo_minutes=60
        )


class RegimeClassifierConfig(ComponentConfig):
    """
    Regime Classifier - Market state detection.

    Philosophy:
    - Long window (365 days) to capture FULL market cycles
    - Need bull/bear/sideways/volatile/quiet regimes
    - 1-minute data (regime can shift quickly)
    - Slow recency weighting (60-day halflife) - regimes have memory
    """

    def __init__(self):
        super().__init__(
            name='regime_classifier',
            lookback_days=365,
            timeframe='1m',
            recency_halflife_days=60.0,
            description='Market regime and state detection',
            min_samples=5000,
            max_samples=100000,
            walk_forward_train_days=90,
            walk_forward_test_days=7,
            embargo_minutes=120
        )


class RiskContextConfig(ComponentConfig):
    """
    Risk Context - Long-term correlations and risk factors.

    Philosophy:
    - Very long window (730 days = 2 years) for macro correlations
    - Daily timeframe (we care about sustained moves, not intraday noise)
    - Very slow recency weighting (120-day halflife) - correlations are sticky
    - Used for position sizing, not entries
    """

    def __init__(self):
        super().__init__(
            name='risk_context',
            lookback_days=730,
            timeframe='1d',
            recency_halflife_days=120.0,
            description='Long-term risk and correlation context',
            min_samples=500,  # Fewer samples (daily data)
            max_samples=1500,
            walk_forward_train_days=180,
            walk_forward_test_days=30,
            embargo_minutes=1440  # 1 day embargo
        )


def create_all_component_configs() -> dict[str, ComponentConfig]:
    """
    Create all component configurations.

    Returns:
        Dictionary mapping component names to configs

    Usage:
        configs = create_all_component_configs()
        scalp_config = configs['scalp_core']
        print(f"Scalp lookback: {scalp_config.lookback_days} days")
    """
    configs = {
        'scalp_core': ScalpCoreConfig(),
        'confirm_filter': ConfirmFilterConfig(),
        'regime_classifier': RegimeClassifierConfig(),
        'risk_context': RiskContextConfig()
    }

    logger.info(
        "all_component_configs_created",
        components=list(configs.keys()),
        total_configs=len(configs)
    )

    return configs


def get_component_config(component_name: str) -> ComponentConfig:
    """
    Get configuration for a specific component.

    Args:
        component_name: One of 'scalp_core', 'confirm_filter',
                       'regime_classifier', 'risk_context'

    Returns:
        ComponentConfig instance

    Raises:
        ValueError: If component name not recognized
    """
    configs = create_all_component_configs()

    if component_name not in configs:
        raise ValueError(
            f"Unknown component '{component_name}'. "
            f"Valid options: {list(configs.keys())}"
        )

    return configs[component_name]


def print_config_summary():
    """Print summary table of all component configs."""
    configs = create_all_component_configs()

    print("\n" + "="*80)
    print("MULTI-WINDOW TRAINING CONFIGURATION SUMMARY")
    print("="*80 + "\n")

    print(f"{'Component':<20} {'Lookback':<12} {'TF':<6} {'Halflife':<12} {'WF Train':<12}")
    print("-"*80)

    for name, config in configs.items():
        print(
            f"{config.name:<20} "
            f"{config.lookback_days} days{'':<4} "
            f"{config.timeframe:<6} "
            f"{config.recency_halflife_days} days{'':<4} "
            f"{config.walk_forward_train_days} days"
        )

    print("\n" + "="*80)
    print("\nPhilosophy:")
    print("  • Fast components (scalp) → short windows (microstructure evolves)")
    print("  • Slow components (regime) → long windows (need full cycles)")
    print("  • Each component weighted by recency appropriate to its horizon")
    print("="*80 + "\n")


if __name__ == "__main__":
    # Demo
    import structlog
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer()
        ]
    )

    print_config_summary()

    # Show specific config details
    scalp = ScalpCoreConfig()
    print(f"\n{scalp.name.upper()} Details:")
    print(f"  {scalp.description}")
    print(f"  Training window: {scalp.lookback_days} days")
    print(f"  Recency halflife: {scalp.recency_halflife_days} days")
    print(f"  → Sample from 60 days ago has weight = 0.01 (1%)")
    print(f"  → Sample from 10 days ago has weight = 0.50 (50%)")
    print(f"  → Sample from today has weight = 1.00 (100%)")
