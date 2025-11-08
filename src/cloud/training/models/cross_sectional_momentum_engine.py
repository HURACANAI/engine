"""
Cross Sectional Momentum Engine

Separate from Leader Engine to allow pure factor tests.
Ranks coins by 1d, 3d, 7d risk-adjusted returns.
Goes long top decile, avoids bottom decile.

Key Features:
- Risk-adjusted return ranking
- Decile-based selection
- Cross-sectional momentum
- Separate from Leader Engine
- Factor testing support

Author: Huracan Engine Team
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum

import numpy as np
import structlog

from .alpha_engines import AlphaSignal, TradingTechnique

logger = structlog.get_logger(__name__)


@dataclass
class CrossSectionalRanking:
    """Cross-sectional ranking result"""
    symbol: str
    rank: int  # 1 = best, N = worst
    percentile: float  # 0-100, higher is better
    decile: int  # 1-10, 1 = top decile, 10 = bottom decile
    risk_adj_return_1d: float
    risk_adj_return_3d: float
    risk_adj_return_7d: float
    composite_score: float


@dataclass
class CrossSectionalSignal:
    """Cross-sectional momentum signal"""
    symbol: str
    direction: str  # "buy" or "sell"
    confidence: float
    ranking: CrossSectionalRanking
    reasoning: str


class CrossSectionalMomentumEngine:
    """
    Cross Sectional Momentum Engine.
    
    Ranks coins by risk-adjusted returns and goes long top decile,
    avoids bottom decile.
    
    Separate from Leader Engine to allow pure factor tests.
    
    Usage:
        engine = CrossSectionalMomentumEngine()
        
        # Rank all symbols
        rankings = engine.rank_symbols(
            symbols_data={
                "BTCUSDT": {"returns_1d": 0.02, "volatility": 0.03, ...},
                "ETHUSDT": {"returns_1d": 0.01, "volatility": 0.02, ...},
                ...
            }
        )
        
        # Generate signal for a symbol
        signal = engine.generate_signal(
            symbol="BTCUSDT",
            ranking=rankings["BTCUSDT"],
            features={...}
        )
    """
    
    def __init__(
        self,
        top_decile_threshold: float = 0.9,  # Top 10% (decile 1)
        bottom_decile_threshold: float = 0.1,  # Bottom 10% (decile 10)
        min_confidence: float = 0.6,  # Minimum confidence to trade
        use_risk_adjustment: bool = True,  # Use risk-adjusted returns
        return_weights: Dict[str, float] = None  # Weights for 1d, 3d, 7d returns
    ):
        """
        Initialize cross-sectional momentum engine.
        
        Args:
            top_decile_threshold: Threshold for top decile (0.9 = top 10%)
            bottom_decile_threshold: Threshold for bottom decile (0.1 = bottom 10%)
            min_confidence: Minimum confidence to trade
            use_risk_adjustment: Use risk-adjusted returns
            return_weights: Weights for 1d, 3d, 7d returns
        """
        self.top_decile_threshold = top_decile_threshold
        self.bottom_decile_threshold = bottom_decile_threshold
        self.min_confidence = min_confidence
        self.use_risk_adjustment = use_risk_adjustment
        
        # Default weights: 1d (0.3), 3d (0.4), 7d (0.3)
        self.return_weights = return_weights or {
            "1d": 0.3,
            "3d": 0.4,
            "7d": 0.3
        }
        
        logger.info(
            "cross_sectional_momentum_engine_initialized",
            top_decile_threshold=top_decile_threshold,
            bottom_decile_threshold=bottom_decile_threshold,
            use_risk_adjustment=use_risk_adjustment
        )
    
    def rank_symbols(
        self,
        symbols_data: Dict[str, Dict[str, float]]
    ) -> Dict[str, CrossSectionalRanking]:
        """
        Rank symbols by risk-adjusted returns.
        
        Args:
            symbols_data: Dictionary of symbol -> data
                Expected keys: returns_1d, returns_3d, returns_7d, volatility
        
        Returns:
            Dictionary of symbol -> CrossSectionalRanking
        """
        rankings = {}
        
        # Extract returns and volatilities
        returns_1d = []
        returns_3d = []
        returns_7d = []
        volatilities = []
        symbols = []
        
        for symbol, data in symbols_data.items():
            returns_1d.append(data.get("returns_1d", 0.0))
            returns_3d.append(data.get("returns_3d", 0.0))
            returns_7d.append(data.get("returns_7d", 0.0))
            volatilities.append(data.get("volatility", 0.01))
            symbols.append(symbol)
        
        if not symbols:
            return {}
        
        # Calculate risk-adjusted returns
        risk_adj_returns_1d = []
        risk_adj_returns_3d = []
        risk_adj_returns_7d = []
        
        for i, symbol in enumerate(symbols):
            vol = volatilities[i] if volatilities[i] > 0 else 0.01
            
            if self.use_risk_adjustment:
                risk_adj_1d = returns_1d[i] / vol if vol > 0 else 0.0
                risk_adj_3d = returns_3d[i] / vol if vol > 0 else 0.0
                risk_adj_7d = returns_7d[i] / vol if vol > 0 else 0.0
            else:
                risk_adj_1d = returns_1d[i]
                risk_adj_3d = returns_3d[i]
                risk_adj_7d = returns_7d[i]
            
            risk_adj_returns_1d.append(risk_adj_1d)
            risk_adj_returns_3d.append(risk_adj_3d)
            risk_adj_returns_7d.append(risk_adj_7d)
        
        # Calculate composite score
        composite_scores = []
        for i in range(len(symbols)):
            composite = (
                risk_adj_returns_1d[i] * self.return_weights["1d"] +
                risk_adj_returns_3d[i] * self.return_weights["3d"] +
                risk_adj_returns_7d[i] * self.return_weights["7d"]
            )
            composite_scores.append(composite)
        
        # Rank by composite score
        sorted_indices = np.argsort(composite_scores)[::-1]  # Descending
        
        # Create rankings
        for rank, idx in enumerate(sorted_indices, 1):
            symbol = symbols[idx]
            percentile = (len(symbols) - rank + 1) / len(symbols) * 100
            decile = self._percentile_to_decile(percentile)
            
            ranking = CrossSectionalRanking(
                symbol=symbol,
                rank=rank,
                percentile=percentile,
                decile=decile,
                risk_adj_return_1d=risk_adj_returns_1d[idx],
                risk_adj_return_3d=risk_adj_returns_3d[idx],
                risk_adj_return_7d=risk_adj_returns_7d[idx],
                composite_score=composite_scores[idx]
            )
            
            rankings[symbol] = ranking
        
        logger.info(
            "symbols_ranked",
            num_symbols=len(rankings),
            top_symbol=rankings[symbols[sorted_indices[0]]].symbol if sorted_indices else None
        )
        
        return rankings
    
    def _percentile_to_decile(self, percentile: float) -> int:
        """Convert percentile to decile (1-10)"""
        # Percentile: 0-100, Decile: 1-10
        # Top 10% = decile 1, bottom 10% = decile 10
        if percentile >= 90:
            return 1
        elif percentile >= 80:
            return 2
        elif percentile >= 70:
            return 3
        elif percentile >= 60:
            return 4
        elif percentile >= 50:
            return 5
        elif percentile >= 40:
            return 6
        elif percentile >= 30:
            return 7
        elif percentile >= 20:
            return 8
        elif percentile >= 10:
            return 9
        else:
            return 10
    
    def generate_signal(
        self,
        symbol: str,
        ranking: CrossSectionalRanking,
        features: Dict[str, float],
        current_regime: str = "unknown"
    ) -> Optional[AlphaSignal]:
        """
        Generate signal based on cross-sectional ranking.
        
        Args:
            symbol: Trading symbol
            ranking: Cross-sectional ranking
            features: Feature dictionary
            current_regime: Current market regime
        
        Returns:
            AlphaSignal or None
        """
        # Check if in top decile (long)
        if ranking.percentile >= self.top_decile_threshold * 100:
            direction = "buy"
            confidence = min(1.0, ranking.percentile / 100)
            reasoning = f"Top decile rank: {ranking.rank}, percentile: {ranking.percentile:.1f}%"
        
        # Check if in bottom decile (avoid/short)
        elif ranking.percentile <= self.bottom_decile_threshold * 100:
            direction = "sell"
            confidence = min(1.0, (100 - ranking.percentile) / 100)
            reasoning = f"Bottom decile rank: {ranking.rank}, percentile: {ranking.percentile:.1f}%"
        
        # Middle deciles: no signal
        else:
            return None
        
        # Check minimum confidence
        if confidence < self.min_confidence:
            return None
        
        return AlphaSignal(
            technique=TradingTechnique.LEADER,  # Use LEADER technique
            direction=direction,
            confidence=confidence,
            reasoning=reasoning,
            key_features={
                "rank": ranking.rank,
                "percentile": ranking.percentile,
                "decile": ranking.decile,
                "risk_adj_return_1d": ranking.risk_adj_return_1d,
                "risk_adj_return_3d": ranking.risk_adj_return_3d,
                "risk_adj_return_7d": ranking.risk_adj_return_7d,
                "composite_score": ranking.composite_score
            },
            regime_affinity=0.8 if current_regime == "trend" else 0.5
        )
    
    def generate_signals_batch(
        self,
        symbols_data: Dict[str, Dict[str, float]],
        features_by_symbol: Dict[str, Dict[str, float]],
        current_regime: str = "unknown"
    ) -> Dict[str, AlphaSignal]:
        """
        Generate signals for all symbols.
        
        Args:
            symbols_data: Symbol data for ranking
            features_by_symbol: Features for each symbol
            current_regime: Current market regime
        
        Returns:
            Dictionary of symbol -> AlphaSignal
        """
        # Rank all symbols
        rankings = self.rank_symbols(symbols_data)
        
        # Generate signals
        signals = {}
        for symbol, ranking in rankings.items():
            features = features_by_symbol.get(symbol, {})
            signal = self.generate_signal(
                symbol=symbol,
                ranking=ranking,
                features=features,
                current_regime=current_regime
            )
            
            if signal:
                signals[symbol] = signal
        
        logger.info(
            "cross_sectional_signals_generated",
            num_signals=len(signals),
            num_ranked=len(rankings)
        )
        
        return signals

