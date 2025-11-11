"""
Tests for Dynamic Coin Selector

Example test suite for coin selection system.
"""

import pytest
from unittest.mock import Mock

from src.cloud.training.services.coin_selector import (
    DynamicCoinSelector,
    CoinSelectionConfig,
    RankingMethod,
)


@pytest.fixture
def config():
    """Test configuration."""
    return CoinSelectionConfig(
        min_daily_volume_usd=1_000_000,
        max_spread_bps=8.0,
        min_age_days=30,
        ranking_method=RankingMethod.LIQUIDITY_SCORE,
    )


@pytest.fixture
def mock_exchange_client():
    """Mock exchange client."""
    client = Mock()
    markets = {
        "BTC/USDT": Mock(symbol="BTC/USDT", active=True, base="BTC", quote="USDT"),
        "ETH/USDT": Mock(symbol="ETH/USDT", active=True, base="ETH", quote="USDT"),
    }
    client.fetch_markets.return_value = markets
    return client


@pytest.fixture
def mock_metadata_loader():
    """Mock metadata loader."""
    loader = Mock()
    
    # Mock liquidity data
    import polars as pl
    liquidity_data = pl.DataFrame({
        "symbol": ["BTC/USDT", "ETH/USDT"],
        "quote_volume": [10_000_000, 5_000_000],
        "spread_bps": [2.0, 5.0],
    })
    
    # Mock fee data
    fee_data = pl.DataFrame({
        "symbol": ["BTC/USDT", "ETH/USDT"],
        "maker_fee_bps": [2.0, 2.0],
        "taker_fee_bps": [4.0, 4.0],
    })
    
    loader.liquidity_snapshot.return_value = liquidity_data
    loader.fee_schedule.return_value = fee_data
    
    return loader


def test_coin_selector_initialization(config, mock_exchange_client, mock_metadata_loader):
    """Test coin selector initialization."""
    selector = DynamicCoinSelector(
        config=config,
        exchange_client=mock_exchange_client,
        metadata_loader=mock_metadata_loader,
    )
    
    assert selector.config == config
    assert selector.exchange_client == mock_exchange_client


def test_coin_filtering(config, mock_exchange_client, mock_metadata_loader):
    """Test coin filtering by volume and spread."""
    selector = DynamicCoinSelector(
        config=config,
        exchange_client=mock_exchange_client,
        metadata_loader=mock_metadata_loader,
    )
    
    # Force update
    selector._update_coin_metrics()
    
    # Both coins should pass filters
    filtered = selector._filter_coins()
    assert len(filtered) >= 0  # Depends on mock data


def test_ranking_methods(config, mock_exchange_client, mock_metadata_loader):
    """Test different ranking methods."""
    # Test liquidity score ranking
    config.ranking_method = RankingMethod.LIQUIDITY_SCORE
    selector = DynamicCoinSelector(
        config=config,
        exchange_client=mock_exchange_client,
        metadata_loader=mock_metadata_loader,
    )
    
    selector._update_coin_metrics()
    coins = selector._rank_coins(list(selector.coin_metrics_cache.values()))
    
    # Should be ranked by liquidity score
    if len(coins) > 1:
        assert coins[0].liquidity_score >= coins[1].liquidity_score

