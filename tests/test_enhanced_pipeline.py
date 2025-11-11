"""Targeted tests for the enhanced RL pipeline."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict

import numpy as np
import pytest

pl = pytest.importorskip("polars")

from src.cloud.config.settings import EngineSettings
from src.cloud.training.pipelines.enhanced_rl_pipeline import EnhancedRLPipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "config"


def _load_settings() -> EngineSettings:
    """Load the local profile to hydrate mandatory fields."""
    return EngineSettings.load(environment="local", config_dir=CONFIG_DIR)


def create_synthetic_market_data(symbol: str, days: int = 180) -> pl.DataFrame:
    """Create moderately long synthetic series so rolling features have coverage."""
    np.random.seed(hash(symbol) % (2**32))

    end_time = datetime.now(tz=timezone.utc)
    timestamps = [end_time - timedelta(days=days - i) for i in range(days)]

    initial_price = 45_000.0 if symbol == "BTC/USD" else 100.0
    returns = np.random.normal(0.001, 0.02, days)
    prices = initial_price * np.exp(np.cumsum(returns))

    frame = pl.DataFrame(
        {
            "ts": timestamps,
            "open": prices * (1 + np.random.uniform(-0.01, 0.01, days)),
            "high": prices * (1 + np.random.uniform(0, 0.02, days)),
            "low": prices * (1 + np.random.uniform(-0.02, 0, days)),
            "close": prices,
            "volume": np.random.uniform(1_000, 10_000, days),
        }
    )
    return frame


@pytest.fixture(scope="module")
def engine_settings() -> EngineSettings:
    return _load_settings()


@pytest.fixture(scope="module", autouse=True)
def patch_memory_store() -> None:
    """Replace the real DatabaseConnectionPool to avoid Postgres dependencies in tests."""
    from _pytest.monkeypatch import MonkeyPatch

    monkeypatch = MonkeyPatch()

    class DummyPool:
        def __init__(self, *args, **kwargs):
            self.dsn = kwargs.get("dsn") or (args[0] if args else "mock://")

        @contextmanager
        def get_connection(self):
            raise RuntimeError("Database access not available inside unit tests")

        def close_all(self) -> None:
            pass

    monkeypatch.setattr(
        "src.cloud.training.memory.store.DatabaseConnectionPool",
        DummyPool,
    )
    yield
    monkeypatch.undo()


@pytest.fixture(scope="module")
def enhanced_pipeline(engine_settings: EngineSettings) -> EnhancedRLPipeline:
    return EnhancedRLPipeline(
        settings=engine_settings,
        dsn="postgresql://localhost/test_db",
        enable_advanced_rewards=True,
        enable_higher_order_features=True,
        enable_granger_causality=True,
        enable_regime_prediction=True,
    )


def test_pipeline_feature_flags_control_components(engine_settings: EngineSettings) -> None:
    """Ensure feature flags toggle major components and state dimensionality."""
    full_pipeline = EnhancedRLPipeline(
        settings=engine_settings,
        dsn="postgresql://localhost/test_db",
        enable_advanced_rewards=True,
        enable_higher_order_features=True,
        enable_granger_causality=True,
        enable_regime_prediction=True,
    )

    assert hasattr(full_pipeline, "reward_calculator")
    assert hasattr(full_pipeline, "higher_order_builder")
    assert hasattr(full_pipeline, "granger_detector")
    assert hasattr(full_pipeline, "regime_predictor")
    assert full_pipeline.agent.state_dim == 148  # Up-sized for higher-order features

    minimal_pipeline = EnhancedRLPipeline(
        settings=engine_settings,
        dsn="postgresql://localhost/test_db",
        enable_advanced_rewards=False,
        enable_higher_order_features=False,
        enable_granger_causality=False,
        enable_regime_prediction=False,
    )

    assert not hasattr(minimal_pipeline, "reward_calculator")
    assert not hasattr(minimal_pipeline, "higher_order_builder")
    assert not hasattr(minimal_pipeline, "granger_detector")
    assert not hasattr(minimal_pipeline, "regime_predictor")
    assert minimal_pipeline.agent.state_dim == engine_settings.training.rl_agent.state_dim


def test_build_enhanced_features_adds_polynomials_and_lags(
    enhanced_pipeline: EnhancedRLPipeline,
) -> None:
    """Building features should add higher-order columns, not just print banners."""
    symbol = "SOL/USD"
    base_data = create_synthetic_market_data(symbol, days=180)
    market_context: Dict[str, pl.DataFrame] = {
        "BTC/USD": create_synthetic_market_data("BTC/USD", days=180),
        "ETH/USD": create_synthetic_market_data("ETH/USD", days=180),
        "SOL/USD": base_data,
    }

    base_features = enhanced_pipeline.feature_recipe.build(base_data)
    enhanced_features = enhanced_pipeline._build_enhanced_features(
        data=base_data,
        symbol=symbol,
        market_context=market_context,
    )

    assert enhanced_features.height == base_features.height
    assert len(enhanced_features.columns) - len(base_features.columns) >= 50

    expected_columns = {"rsi_14_squared", "ret_3_cubed", "rsi_7_lag_1"}
    missing = expected_columns.difference(set(enhanced_features.columns))
    assert not missing, f"Missing higher-order columns: {missing}"


def test_phase1_stats_surface_component_details(
    enhanced_pipeline: EnhancedRLPipeline,
) -> None:
    """Stats endpoint should expose telemetry for each Phase 1 component."""
    stats = enhanced_pipeline.get_phase1_stats()

    assert "reward_calculator" in stats
    assert "current_sharpe" in stats
    assert "causal_graph" in stats
    assert "regime_predictor" in stats

    reward_stats = stats["reward_calculator"]
    assert reward_stats["num_recent_returns"] == 0
    assert reward_stats["sharpe_ratio"] == 0

    causal_stats = stats["causal_graph"]
    assert causal_stats["num_relationships"] == 0

    regime_stats = stats["regime_predictor"]
    assert regime_stats["lookback_periods"] == enhanced_pipeline.regime_predictor.lookback_periods
