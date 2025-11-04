import polars as pl
import torch

from cloud.training.agents.rl_agent import ExperienceReplayBuffer, TradingState
from cloud.training.services.costs import CostModel
from cloud.training.config.settings import CostSettings
from cloud.training.models.regime_detector import RegimeDetector
from cloud.training.models.confidence_scorer import ConfidenceScorer
from cloud.training.backtesting.shadow_trader import ShadowTrader
from shared.features.recipe import FeatureRecipe


def test_replay_buffer_sampling_returns_expected_shapes():
    buffer = ExperienceReplayBuffer(16)
    states = torch.randn(4, 8)
    actions = torch.randint(0, 3, (4,))
    log_probs = torch.randn(4)
    advantages = torch.randn(4)
    returns = torch.randn(4)
    dummy_state = TradingState(
        market_features=torch.zeros(1),
        similar_pattern_win_rate=0.5,
        similar_pattern_avg_profit=0.0,
        similar_pattern_reliability=0.5,
        has_position=False,
        position_size_multiplier=0.0,
        unrealized_pnl_bps=0.0,
        hold_duration_minutes=0,
        volatility_bps=0.0,
        spread_bps=5.0,
        regime_code=1,
        current_drawdown_gbp=0.0,
        trades_today=0,
        win_rate_today=0.5,
        recent_return_1m=0.0,
        recent_return_5m=0.0,
        recent_return_30m=0.0,
        volume_zscore=0.0,
        volatility_zscore=0.0,
        estimated_transaction_cost_bps=5.0,
        trend_flag_5m=0.0,
        trend_flag_1h=0.0,
        orderbook_imbalance=0.0,
        flow_trend_score=0.0,
        symbol="TEST",
    )
    contexts = [dummy_state for _ in range(4)]
    buffer.add_batch(states, actions, log_probs, advantages, returns, contexts)
    batch = buffer.sample(3)
    assert batch is not None
    assert batch["states"].shape[0] == 3
    assert len(batch["contexts"]) == 3


def test_cost_model_dynamic_penalty_increases_with_low_liquidity():
    settings = CostSettings()
    model = CostModel(settings)
    rich_liquidity = model.estimate(
        taker_fee_bps=5,
        spread_bps=2,
        volatility_bps=10,
        adv_quote=settings.notional_per_trade * 5000,
    )
    thin_liquidity = model.estimate(
        taker_fee_bps=5,
        spread_bps=2,
        volatility_bps=10,
        adv_quote=settings.notional_per_trade * 2,
    )
    assert thin_liquidity.slippage_bps > rich_liquidity.slippage_bps


def test_regime_detector_meta_scores_present():
    df = pl.DataFrame({
        "ts": pl.datetime_range(start=0, end=99, interval="1m"),
        "close": [100 + i * 0.1 for i in range(100)],
        "high": [100 + i * 0.1 + 0.5 for i in range(100)],
        "low": [100 + i * 0.1 - 0.5 for i in range(100)],
        "volume": [1_000 + (i % 10) * 10 for i in range(100)],
    })
    recipe = FeatureRecipe()
    features = recipe.build(df)
    detector = RegimeDetector()
    result = detector.detect_regime(features)
    for key in ("trend_meta", "range_meta", "panic_meta"):
        assert key in result.regime_scores


def test_confidence_scorer_meta_features_adjust_confidence():
    scorer = ConfidenceScorer()
    baseline = scorer.calculate_confidence(
        sample_count=50,
        best_score=0.6,
        runner_up_score=0.55,
        pattern_similarity=0.6,
        pattern_reliability=0.6,
        regime_match=True,
        regime_confidence=0.7,
    )
    boosted = scorer.calculate_confidence(
        sample_count=50,
        best_score=0.6,
        runner_up_score=0.55,
        pattern_similarity=0.6,
        pattern_reliability=0.6,
        regime_match=True,
        regime_confidence=0.7,
        meta_features={"meta_signal": 0.9, "orderbook_bias": 0.4},
    )
    assert boosted.confidence > baseline.confidence


def test_apply_scenario_modifiers_spread_shock():
    frame = pl.DataFrame({
        "ts": pl.datetime_range(start=0, end=4, interval="1m"),
        "close": [100, 101, 102, 103, 104],
        "open": [99, 100, 101, 102, 103],
        "high": [101, 102, 103, 104, 105],
        "low": [98, 99, 100, 101, 102],
        "volume": [1000, 1100, 1050, 1150, 1200],
        "spread_bps": [5, 5, 5, 5, 5],
    })
    trader = ShadowTrader.__new__(ShadowTrader)
    modified = ShadowTrader._apply_scenario_modifiers(trader, frame, {"spread_multiplier": 2.0})
    assert modified["spread_bps"].to_list()[0] == 10


def test_apply_scenario_modifiers_latency_shift():
    frame = pl.DataFrame({
        "ts": pl.datetime_range(start=0, end=4, interval="1m"),
        "close": [100, 101, 102, 103, 104],
        "open": [99, 100, 101, 102, 103],
        "high": [101, 102, 103, 104, 105],
        "low": [98, 99, 100, 101, 102],
        "volume": [1000, 1100, 1050, 1150, 1200],
    })
    trader = ShadowTrader.__new__(ShadowTrader)
    modified = ShadowTrader._apply_scenario_modifiers(trader, frame, {"latency_minutes": 1})
    assert modified["close"].null_count() == 0


def test_feature_recipe_handles_missing_values():
    frame = pl.DataFrame({
        "ts": pl.datetime_range(start=0, end=50, interval="1m"),
        "open": [100.0] * 51,
        "high": [101.0] * 51,
        "low": [99.0] * 51,
        "close": [100.5 if i % 10 else None for i in range(51)],
        "volume": [1000 + i for i in range(51)],
    })
    recipe = FeatureRecipe()
    features = recipe.build(frame)
    assert features.null_count().sum_horizontal()[0] == 0
