"""
Funding / Carry Engine

Exploits perpetual futures funding rates and basis differentials.
Trades the carry premium/discount between spot and futures.

Key Features:
1. Funding rate analysis (8-hour funding rates)
2. Basis trading (spot-futures spread)
3. Carry trade detection (positive/negative carry)
4. Funding rate prediction
5. Cross-exchange funding arbitrage

Best in: All regimes (funding rates independent of price action)
Strategy: Trade funding rate differentials and basis spreads
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import numpy as np
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class FundingSignal:
    """Signal from funding/carry engine."""
    direction: str  # "buy", "sell", "hold"
    confidence: float  # 0-1
    expected_funding_bps: float  # Expected funding rate in bps
    basis_bps: float  # Spot-futures basis in bps
    carry_annualized_pct: float  # Annualized carry return
    reasoning: str
    key_features: Dict[str, float]


class FundingCarryEngine:
    """
    Funding / Carry Engine.
    
    Exploits perpetual futures funding rates and basis differentials.
    Trades the carry premium/discount between spot and futures.
    
    Key Features:
    - Funding rate analysis (8-hour funding rates)
    - Basis trading (spot-futures spread)
    - Carry trade detection
    - Funding rate prediction
    - Cross-exchange funding arbitrage
    """
    
    def __init__(
        self,
        min_funding_bps: float = 5.0,  # Minimum funding rate to trade (5 bps)
        min_carry_annualized: float = 0.10,  # Minimum annualized carry (10%)
        max_funding_bps: float = 50.0,  # Maximum funding rate (50 bps = extreme)
        use_basis: bool = True,  # Use basis trading
    ):
        """
        Initialize funding/carry engine.
        
        Args:
            min_funding_bps: Minimum funding rate to trade (in bps)
            min_carry_annualized: Minimum annualized carry return
            max_funding_bps: Maximum funding rate (safety limit)
            use_basis: Whether to use basis trading
        """
        self.min_funding_bps = min_funding_bps
        self.min_carry_annualized = min_carry_annualized
        self.max_funding_bps = max_funding_bps
        self.use_basis = use_basis
        
        # Feature weights for this technique
        self.feature_weights = {
            "funding_rate": 0.40,
            "funding_rate_trend": 0.25,
            "basis_bps": 0.20,
            "funding_sentiment": 0.15,
        }
        
        logger.info(
            "funding_carry_engine_initialized",
            min_funding_bps=min_funding_bps,
            min_carry_annualized=min_carry_annualized,
        )
    
    def generate_signal(
        self,
        features: Dict[str, float],
        current_regime: str,
    ) -> FundingSignal:
        """
        Generate funding/carry signal.
        
        Args:
            features: Feature dictionary (should contain funding_rate, basis_bps, etc.)
            current_regime: Current market regime
        
        Returns:
            FundingSignal
        """
        # Extract key features
        funding_rate = features.get("funding_rate", 0.0)  # 8-hour funding rate (annualized)
        funding_rate_trend = features.get("funding_rate_trend", 0.0)
        basis_bps = features.get("basis_bps", 0.0)  # Spot-futures basis in bps
        funding_sentiment = features.get("funding_sentiment", 0.0)  # -1 to +1
        
        # Convert funding rate to bps (if in decimal, multiply by 10000)
        if abs(funding_rate) < 1.0:
            funding_bps = funding_rate * 10000.0  # Convert to bps
        else:
            funding_bps = funding_rate
        
        # Calculate annualized carry
        # Funding rate is 8-hour, so annualized = funding_rate * (365 * 24 / 8) = funding_rate * 1095
        carry_annualized_pct = funding_bps * 1095.0 / 10000.0  # Convert bps to percentage
        
        # Check if opportunity is viable
        if abs(funding_bps) < self.min_funding_bps:
            return FundingSignal(
                direction="hold",
                confidence=0.0,
                expected_funding_bps=funding_bps,
                basis_bps=basis_bps,
                carry_annualized_pct=carry_annualized_pct,
                reasoning=f"Funding rate too small ({funding_bps:.1f} bps < {self.min_funding_bps:.1f} bps)",
                key_features={
                    "funding_rate": funding_rate,
                    "funding_bps": funding_bps,
                    "basis_bps": basis_bps,
                },
            )
        
        if abs(funding_bps) > self.max_funding_bps:
            return FundingSignal(
                direction="hold",
                confidence=0.0,
                expected_funding_bps=funding_bps,
                basis_bps=basis_bps,
                carry_annualized_pct=carry_annualized_pct,
                reasoning=f"Funding rate too extreme ({funding_bps:.1f} bps > {self.max_funding_bps:.1f} bps)",
                key_features={
                    "funding_rate": funding_rate,
                    "funding_bps": funding_bps,
                },
            )
        
        if abs(carry_annualized_pct) < self.min_carry_annualized:
            return FundingSignal(
                direction="hold",
                confidence=0.0,
                expected_funding_bps=funding_bps,
                basis_bps=basis_bps,
                carry_annualized_pct=carry_annualized_pct,
                reasoning=f"Carry too small ({carry_annualized_pct:.2%} < {self.min_carry_annualized:.2%})",
                key_features={
                    "funding_rate": funding_rate,
                    "carry_annualized_pct": carry_annualized_pct,
                },
            )
        
        # Determine direction based on funding rate
        # Positive funding = longs pay shorts (bearish) → SELL
        # Negative funding = shorts pay longs (bullish) → BUY
        if funding_bps > 0:
            # Positive funding: longs pay shorts → SELL (receive funding)
            direction = "sell"
            confidence = min(0.9, 0.5 + (funding_bps / self.max_funding_bps) * 0.4)
            reasoning = f"Positive funding ({funding_bps:.1f} bps): SELL to receive funding"
        else:
            # Negative funding: shorts pay longs → BUY (receive funding)
            direction = "buy"
            confidence = min(0.9, 0.5 + (abs(funding_bps) / self.max_funding_bps) * 0.4)
            reasoning = f"Negative funding ({funding_bps:.1f} bps): BUY to receive funding"
        
        # Adjust confidence based on funding trend
        if funding_rate_trend > 0 and funding_bps > 0:
            # Funding increasing (more positive) → higher confidence for SELL
            confidence = min(0.95, confidence + 0.1)
        elif funding_rate_trend < 0 and funding_bps < 0:
            # Funding decreasing (more negative) → higher confidence for BUY
            confidence = min(0.95, confidence + 0.1)
        
        # Adjust confidence based on basis (if using basis trading)
        if self.use_basis:
            if basis_bps > 0 and direction == "buy":
                # Positive basis (futures premium) → higher confidence for BUY
                confidence = min(0.95, confidence + 0.05)
            elif basis_bps < 0 and direction == "sell":
                # Negative basis (futures discount) → higher confidence for SELL
                confidence = min(0.95, confidence + 0.05)
        
        return FundingSignal(
            direction=direction,
            confidence=confidence,
            expected_funding_bps=funding_bps,
            basis_bps=basis_bps,
            carry_annualized_pct=carry_annualized_pct,
            reasoning=reasoning,
            key_features={
                "funding_rate": funding_rate,
                "funding_bps": funding_bps,
                "funding_rate_trend": funding_rate_trend,
                "basis_bps": basis_bps,
                "funding_sentiment": funding_sentiment,
                "carry_annualized_pct": carry_annualized_pct,
            },
        )

