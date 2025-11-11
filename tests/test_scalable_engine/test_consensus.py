"""
Tests for Enhanced Consensus Engine

Example test suite for consensus voting system.
"""

import pytest
import numpy as np

from src.cloud.training.consensus.enhanced_consensus import (
    EnhancedConsensusEngine,
    EngineVote,
    MarketRegime,
)


@pytest.fixture
def consensus_engine():
    """Test consensus engine."""
    return EnhancedConsensusEngine(
        num_engines=23,
        min_agreement_threshold=0.6,
    )


def test_consensus_unanimous_buy(consensus_engine):
    """Test consensus with unanimous buy votes."""
    votes = [
        EngineVote(
            engine_id=f"engine_{i}",
            signal=1,
            confidence=0.8,
            raw_score=0.75,
        )
        for i in range(23)
    ]
    
    result = consensus_engine.generate_consensus(
        votes=votes,
        current_regime=MarketRegime.TREND,
    )
    
    assert result.consensus_signal == 1
    assert result.consensus_confidence > 0.8
    assert result.agreement_ratio == 1.0


def test_consensus_mixed_votes(consensus_engine):
    """Test consensus with mixed votes."""
    votes = [
        EngineVote(
            engine_id=f"engine_{i}",
            signal=1 if i < 15 else -1,
            confidence=0.7,
            raw_score=0.5,
        )
        for i in range(23)
    ]
    
    result = consensus_engine.generate_consensus(
        votes=votes,
        current_regime=MarketRegime.TREND,
    )
    
    assert result.consensus_signal == 1  # Majority buy
    assert result.num_agree == 15
    assert result.agreement_ratio == 15 / 23


def test_consensus_empty_votes(consensus_engine):
    """Test consensus with empty votes."""
    result = consensus_engine.generate_consensus(
        votes=[],
        current_regime=MarketRegime.TREND,
    )
    
    assert result.consensus_signal == 0
    assert result.consensus_confidence == 0.0
    assert result.num_engines == 0


def test_reliability_update(consensus_engine):
    """Test reliability update."""
    engine_id = "test_engine"
    
    # Initial reliability should be default
    initial_reliability = consensus_engine.get_engine_reliability(engine_id)
    assert initial_reliability == 0.5
    
    # Update with good performance
    consensus_engine.update_reliability(
        engine_id=engine_id,
        pnl=100.0,
        confidence=0.8,
        regime=MarketRegime.TREND,
    )
    
    # Reliability should increase
    new_reliability = consensus_engine.get_engine_reliability(engine_id)
    assert new_reliability > initial_reliability


def test_regime_thresholds(consensus_engine):
    """Test regime-specific thresholds."""
    votes = [
        EngineVote(
            engine_id=f"engine_{i}",
            signal=1,
            confidence=0.55,  # Just above TREND threshold (0.5)
            raw_score=0.5,
        )
        for i in range(23)
    ]
    
    # TREND regime - should pass
    result_trend = consensus_engine.generate_consensus(
        votes=votes,
        current_regime=MarketRegime.TREND,
    )
    assert result_trend.consensus_signal == 1
    
    # PANIC regime - should fail (threshold 0.65)
    result_panic = consensus_engine.generate_consensus(
        votes=votes,
        current_regime=MarketRegime.PANIC,
    )
    assert result_panic.consensus_signal == 0  # Below threshold

