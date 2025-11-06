"""
Cross-Asset Correlation Analyzer

Tracks real-time correlations between assets to detect market-wide risk and
prevent over-concentrated portfolio exposure.

Key Problems Solved:
1. **Correlation Risk**: BTC dumps → ETH/SOL dump together (90% correlated)
2. **Hidden Concentration**: 3 positions that look diversified but move together = 3x risk
3. **Market-Wide Events**: All assets dumping = systemic risk, exit everything
4. **Lead-Lag Relationships**: BTC often leads alts by 5-15 minutes

Example:
    Portfolio has 3 positions:
    - Long BTC at $47k
    - Long ETH at $2.5k
    - Long SOL at $100

    Correlation Analysis:
    - BTC-ETH correlation: 0.85 (high)
    - BTC-SOL correlation: 0.82 (high)
    - ETH-SOL correlation: 0.88 (very high)

    Effective Diversification: Only 30% (should be 100% with 3 uncorrelated assets)
    → This is really 1.5 positions, not 3!
    → If BTC dumps, ALL positions dump together
    → Recommendation: Exit 1-2 positions to reduce correlation risk
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import structlog

logger = structlog.get_logger(__name__)

# Optional: Import reliable pattern detector for enhanced pattern detection
try:
    from .reliable_pattern_detector import ReliablePatternDetector
    HAS_RELIABLE_PATTERNS = True
except ImportError:
    HAS_RELIABLE_PATTERNS = False


@dataclass
class CorrelationMetrics:
    """Correlation metrics between two assets."""

    asset1: str
    asset2: str
    correlation: float  # -1 to +1
    rolling_correlation: float  # Recent correlation (more reactive)
    correlation_strength: str  # VERY_HIGH, HIGH, MODERATE, LOW, NONE
    lead_lag_minutes: Optional[float] = None  # Which asset leads (positive = asset1 leads)


@dataclass
class PortfolioCorrelationRisk:
    """Portfolio-wide correlation risk assessment."""

    total_positions: int
    effective_positions: float  # Adjusted for correlations (3 correlated = 1.5 effective)
    diversification_ratio: float  # 0-1, 1 = perfectly diversified
    max_pairwise_correlation: float
    avg_correlation: float
    correlated_pairs: List[Tuple[str, str, float]]  # Pairs with >0.7 correlation
    systemic_risk_score: float  # 0-1, higher = more systemic risk
    recommendation: str  # 'REDUCE_EXPOSURE', 'REBALANCE', 'ACCEPTABLE'
    warning_message: Optional[str] = None


@dataclass
class MarketWideEvent:
    """Detection of market-wide risk events."""

    event_type: str  # 'RISK_OFF', 'RISK_ON', 'VOLATILITY_SPIKE', 'LIQUIDITY_CRISIS'
    severity: float  # 0-1
    affected_assets: List[str]
    correlation_surge: float  # How much correlations increased
    recommended_action: str  # 'EXIT_ALL', 'REDUCE_50PCT', 'TIGHTEN_STOPS', 'MONITOR'
    description: str


class CorrelationAnalyzer:
    """
    Analyzes cross-asset correlations to manage portfolio risk.

    Three main functions:
    1. **Pairwise Correlation**: Track correlation between each pair of assets
    2. **Portfolio Diversification**: Calculate true diversification accounting for correlations
    3. **Market-Wide Events**: Detect systemic risk (all assets moving together)

    Correlation Strength Classification:
    - 0.90-1.00: VERY_HIGH (basically same asset)
    - 0.70-0.90: HIGH (move together most of the time)
    - 0.50-0.70: MODERATE (some relationship)
    - 0.30-0.50: LOW (weak relationship)
    - 0.00-0.30: NONE (independent)

    Usage:
        analyzer = CorrelationAnalyzer()

        # Update with new returns
        analyzer.update_returns(
            symbol='BTC',
            returns=[0.002, -0.001, 0.003, ...],
            timestamps=[...],
        )
        analyzer.update_returns('ETH', ...)
        analyzer.update_returns('SOL', ...)

        # Check correlation between two assets
        btc_eth = analyzer.get_correlation('BTC', 'ETH')
        print(f"BTC-ETH correlation: {btc_eth.correlation:.2f}")

        # Analyze portfolio risk
        portfolio_symbols = ['BTC', 'ETH', 'SOL']
        risk = analyzer.analyze_portfolio_risk(portfolio_symbols)

        if risk.recommendation == 'REDUCE_EXPOSURE':
            # Too correlated, reduce positions
            logger.warning(risk.warning_message)

        # Detect market-wide events
        event = analyzer.detect_market_event()
        if event and event.event_type == 'RISK_OFF':
            # Market-wide dump, exit everything
            exit_all_positions()
    """

    def __init__(
        self,
        lookback_periods: int = 100,
        rolling_window: int = 20,
        high_correlation_threshold: float = 0.70,
        very_high_correlation_threshold: float = 0.90,
        min_periods: int = 30,
        systemic_event_threshold: float = 0.80,
        use_reliable_patterns: bool = True,  # Use reliable pattern detector
    ):
        """
        Initialize correlation analyzer.

        Args:
            lookback_periods: Historical periods to store for correlation calculation
            rolling_window: Recent periods for rolling correlation
            high_correlation_threshold: Threshold for HIGH correlation classification
            very_high_correlation_threshold: Threshold for VERY_HIGH correlation
            min_periods: Minimum periods needed for valid correlation
            systemic_event_threshold: Threshold for detecting market-wide events
            use_reliable_patterns: Whether to use reliable pattern detector (>90% reliability)
        """
        self.lookback_periods = lookback_periods
        self.rolling_window = rolling_window
        self.high_corr_threshold = high_correlation_threshold
        self.very_high_corr_threshold = very_high_correlation_threshold
        self.min_periods = min_periods
        self.systemic_threshold = systemic_event_threshold
        self.use_reliable_patterns = use_reliable_patterns and HAS_RELIABLE_PATTERNS

        # Store returns history per symbol
        self.returns_history: Dict[str, List[float]] = {}
        self.timestamps_history: Dict[str, List[float]] = {}

        # Cached correlations (updated on each new return)
        self.correlation_matrix: Dict[Tuple[str, str], CorrelationMetrics] = {}
        
        # Reliable pattern detector (optional)
        self.pattern_detector: Optional[ReliablePatternDetector] = None
        if self.use_reliable_patterns:
            self.pattern_detector = ReliablePatternDetector(
                min_reliability=0.90,
                min_observations=100,
            )
            logger.info("reliable_pattern_detector_enabled")

        logger.info(
            "correlation_analyzer_initialized",
            lookback_periods=lookback_periods,
            high_corr_threshold=high_correlation_threshold,
            use_reliable_patterns=self.use_reliable_patterns,
        )

    def update_returns(
        self,
        symbol: str,
        returns: List[float],
        timestamps: List[float],
    ) -> None:
        """
        Update returns history for a symbol.

        Args:
            symbol: Asset symbol (e.g., 'BTC', 'ETH')
            returns: List of returns (e.g., [0.002, -0.001, ...])
            timestamps: Corresponding timestamps
        """
        if symbol not in self.returns_history:
            self.returns_history[symbol] = []
            self.timestamps_history[symbol] = []

        # Append new returns
        self.returns_history[symbol].extend(returns)
        self.timestamps_history[symbol].extend(timestamps)

        # Keep only lookback_periods
        if len(self.returns_history[symbol]) > self.lookback_periods:
            self.returns_history[symbol] = self.returns_history[symbol][-self.lookback_periods:]
            self.timestamps_history[symbol] = self.timestamps_history[symbol][-self.lookback_periods:]

        # Incrementally update correlations involving this symbol (more efficient)
        self._incremental_update_correlations(symbol)

        logger.debug(
            "returns_updated",
            symbol=symbol,
            total_periods=len(self.returns_history[symbol]),
        )

    def add_single_return(
        self,
        symbol: str,
        return_value: float,
        timestamp: float,
    ) -> None:
        """Add a single new return (convenience method)."""
        self.update_returns(symbol, [return_value], [timestamp])

    def get_correlation(
        self,
        symbol1: str,
        symbol2: str,
    ) -> Optional[CorrelationMetrics]:
        """
        Get correlation between two symbols.

        Returns:
            CorrelationMetrics if both symbols have sufficient data, None otherwise
        """
        # Check cache
        key = tuple(sorted([symbol1, symbol2]))
        return self.correlation_matrix.get(key)

    def analyze_portfolio_risk(
        self,
        portfolio_symbols: List[str],
        position_sizes: Optional[Dict[str, float]] = None,
    ) -> PortfolioCorrelationRisk:
        """
        Analyze portfolio-wide correlation risk.

        Args:
            portfolio_symbols: List of symbols in portfolio
            position_sizes: Optional dict of {symbol: size_usd} for weighting

        Returns:
            PortfolioCorrelationRisk with diversification assessment
        """
        if len(portfolio_symbols) < 2:
            # Single position = no correlation risk
            return PortfolioCorrelationRisk(
                total_positions=len(portfolio_symbols),
                effective_positions=float(len(portfolio_symbols)),
                diversification_ratio=1.0,
                max_pairwise_correlation=0.0,
                avg_correlation=0.0,
                correlated_pairs=[],
                systemic_risk_score=0.0,
                recommendation='ACCEPTABLE',
            )

        # Get all pairwise correlations
        correlations = []
        correlated_pairs = []

        for i, sym1 in enumerate(portfolio_symbols):
            for sym2 in portfolio_symbols[i + 1:]:
                corr_metrics = self.get_correlation(sym1, sym2)
                if corr_metrics:
                    correlations.append(corr_metrics.correlation)

                    if abs(corr_metrics.correlation) >= self.high_corr_threshold:
                        correlated_pairs.append((sym1, sym2, corr_metrics.correlation))

        if not correlations:
            # Insufficient data
            return PortfolioCorrelationRisk(
                total_positions=len(portfolio_symbols),
                effective_positions=float(len(portfolio_symbols)),
                diversification_ratio=1.0,
                max_pairwise_correlation=0.0,
                avg_correlation=0.0,
                correlated_pairs=[],
                systemic_risk_score=0.0,
                recommendation='ACCEPTABLE',
                warning_message="Insufficient correlation data",
            )

        max_corr = max(abs(c) for c in correlations)
        avg_corr = np.mean([abs(c) for c in correlations])

        # Calculate effective number of positions
        # Formula: N_eff = N / (1 + (N-1) * avg_corr)
        # Perfect diversification (corr=0) → N_eff = N
        # Perfect correlation (corr=1) → N_eff = 1
        n = len(portfolio_symbols)
        effective_positions = n / (1 + (n - 1) * avg_corr)

        # Diversification ratio: how much diversification benefit we get
        diversification_ratio = effective_positions / n

        # Systemic risk score: how correlated everything is
        systemic_risk_score = avg_corr

        # Determine recommendation
        if diversification_ratio < 0.4:
            # Very poor diversification
            recommendation = 'REDUCE_EXPOSURE'
            warning_message = (
                f"Portfolio correlation too high! "
                f"{n} positions but only {effective_positions:.1f} effective positions. "
                f"Average correlation: {avg_corr:.0%}. "
                f"Consider exiting highly correlated positions."
            )
        elif diversification_ratio < 0.6:
            # Moderate diversification
            recommendation = 'REBALANCE'
            warning_message = (
                f"Portfolio diversification suboptimal. "
                f"{n} positions = {effective_positions:.1f} effective positions. "
                f"Consider rebalancing to reduce correlation."
            )
        else:
            # Good diversification
            recommendation = 'ACCEPTABLE'
            warning_message = None

        logger.info(
            "portfolio_correlation_analyzed",
            total_positions=n,
            effective_positions=effective_positions,
            diversification_ratio=diversification_ratio,
            recommendation=recommendation,
        )

        return PortfolioCorrelationRisk(
            total_positions=n,
            effective_positions=effective_positions,
            diversification_ratio=diversification_ratio,
            max_pairwise_correlation=max_corr,
            avg_correlation=avg_corr,
            correlated_pairs=correlated_pairs,
            systemic_risk_score=systemic_risk_score,
            recommendation=recommendation,
            warning_message=warning_message,
        )

    def detect_market_event(
        self,
        all_symbols: Optional[List[str]] = None,
    ) -> Optional[MarketWideEvent]:
        """
        Detect market-wide risk events.

        Looks for:
        - All assets moving together (correlation surge)
        - All assets dumping (risk-off event)
        - All assets pumping (risk-on event)
        - Volatility spike across all assets

        Args:
            all_symbols: List of all tradeable symbols (if None, uses all tracked symbols)

        Returns:
            MarketWideEvent if detected, None otherwise
        """
        if all_symbols is None:
            all_symbols = list(self.returns_history.keys())

        if len(all_symbols) < 3:
            return None  # Need at least 3 assets for market-wide event

        # Get recent returns for all symbols
        recent_returns = []
        valid_symbols = []

        for symbol in all_symbols:
            if symbol in self.returns_history and len(self.returns_history[symbol]) >= self.rolling_window:
                recent = self.returns_history[symbol][-self.rolling_window:]
                recent_returns.append(recent)
                valid_symbols.append(symbol)

        if len(valid_symbols) < 3:
            return None

        # Calculate correlation matrix for recent period
        returns_matrix = np.array(recent_returns)  # Shape: (n_symbols, rolling_window)
        recent_corr_matrix = np.corrcoef(returns_matrix)

        # Get average pairwise correlation (excluding diagonal)
        n = len(recent_corr_matrix)
        avg_recent_corr = (np.sum(np.abs(recent_corr_matrix)) - n) / (n * (n - 1))

        # Check for correlation surge (market-wide event)
        if avg_recent_corr > self.systemic_threshold:
            # High correlation across all assets

            # Check direction: all dumping or all pumping?
            recent_period_returns = returns_matrix[:, -5:]  # Last 5 periods
            avg_returns = np.mean(recent_period_returns, axis=1)
            pct_negative = np.sum(avg_returns < 0) / len(avg_returns)

            if pct_negative > 0.75:
                # Most assets dumping
                event_type = 'RISK_OFF'
                severity = min(avg_recent_corr, 1.0)
                recommended_action = 'EXIT_ALL' if severity > 0.90 else 'REDUCE_50PCT'
                description = (
                    f"RISK-OFF EVENT: {pct_negative:.0%} of assets dumping. "
                    f"Correlation surge to {avg_recent_corr:.0%}. "
                    f"Market-wide sell-off detected."
                )

            elif pct_negative < 0.25:
                # Most assets pumping
                event_type = 'RISK_ON'
                severity = avg_recent_corr * 0.5  # Less severe than risk-off
                recommended_action = 'MONITOR'
                description = (
                    f"RISK-ON EVENT: {1-pct_negative:.0%} of assets rallying. "
                    f"Correlation surge to {avg_recent_corr:.0%}. "
                    f"Market-wide rally detected."
                )

            else:
                # High correlation but mixed direction = volatility spike
                event_type = 'VOLATILITY_SPIKE'
                severity = avg_recent_corr * 0.7
                recommended_action = 'TIGHTEN_STOPS'
                description = (
                    f"VOLATILITY SPIKE: Correlation {avg_recent_corr:.0%}. "
                    f"High correlation with mixed directions = choppy market."
                )

            logger.warning(
                "market_event_detected",
                event_type=event_type,
                severity=severity,
                avg_correlation=avg_recent_corr,
                affected_assets=len(valid_symbols),
            )

            return MarketWideEvent(
                event_type=event_type,
                severity=severity,
                affected_assets=valid_symbols,
                correlation_surge=avg_recent_corr,
                recommended_action=recommended_action,
                description=description,
            )

        return None

    def _recalculate_correlations(self, symbol: str) -> None:
        """Recalculate correlations involving the given symbol."""
        if len(self.returns_history[symbol]) < self.min_periods:
            return  # Not enough data yet

        # Calculate correlations with all other symbols
        for other_symbol in self.returns_history:
            if other_symbol == symbol:
                continue

            if len(self.returns_history[other_symbol]) < self.min_periods:
                continue

            # Align returns (use common timestamps)
            corr_metrics = self._calculate_correlation(symbol, other_symbol)

            if corr_metrics:
                # Store in cache (use sorted tuple as key for consistency)
                key = tuple(sorted([symbol, other_symbol]))
                self.correlation_matrix[key] = corr_metrics

    def _incremental_update_correlations(self, symbol: str) -> None:
        """
        Incrementally update correlations (more efficient than full recalculation).
        
        Only updates correlations for the symbol that changed, not all pairs.
        """
        if len(self.returns_history[symbol]) < self.min_periods:
            return  # Not enough data yet

        # Only update correlations involving this symbol
        for other_symbol in self.returns_history:
            if other_symbol == symbol:
                continue

            if len(self.returns_history[other_symbol]) < self.min_periods:
                continue

            # Calculate correlation (incremental - only for changed symbol)
            corr_metrics = self._calculate_correlation(symbol, other_symbol)

            if corr_metrics:
                # Store in cache (use sorted tuple as key for consistency)
                key = tuple(sorted([symbol, other_symbol]))
                self.correlation_matrix[key] = corr_metrics

    def _calculate_correlation(
        self,
        symbol1: str,
        symbol2: str,
    ) -> Optional[CorrelationMetrics]:
        """Calculate correlation metrics between two symbols."""
        returns1 = self.returns_history[symbol1]
        returns2 = self.returns_history[symbol2]

        # Use the shorter length
        min_len = min(len(returns1), len(returns2))
        if min_len < self.min_periods:
            return None

        # Take last min_len periods
        returns1 = np.array(returns1[-min_len:])
        returns2 = np.array(returns2[-min_len:])

        # Calculate full correlation
        full_corr = np.corrcoef(returns1, returns2)[0, 1]

        # Calculate rolling correlation (recent)
        if min_len >= self.rolling_window:
            rolling_returns1 = returns1[-self.rolling_window:]
            rolling_returns2 = returns2[-self.rolling_window:]
            rolling_corr = np.corrcoef(rolling_returns1, rolling_returns2)[0, 1]
        else:
            rolling_corr = full_corr

        # Classify correlation strength
        abs_corr = abs(full_corr)
        if abs_corr >= self.very_high_corr_threshold:
            strength = 'VERY_HIGH'
        elif abs_corr >= self.high_corr_threshold:
            strength = 'HIGH'
        elif abs_corr >= 0.50:
            strength = 'MODERATE'
        elif abs_corr >= 0.30:
            strength = 'LOW'
        else:
            strength = 'NONE'

        # Calculate lead-lag relationship using cross-correlation
        lead_lag_minutes = self._calculate_lead_lag(symbol1, symbol2)
        
        # If using reliable patterns, validate lead-lag relationship
        if self.use_reliable_patterns and self.pattern_detector and lead_lag_minutes is not None:
            # Check if this lead-lag pattern is reliable (>90%)
            pattern_id = self.pattern_detector._generate_pattern_id(symbol1, symbol2, "lead_lag")
            active_patterns = self.pattern_detector.get_active_patterns(
                coin1=symbol1,
                coin2=symbol2,
                pattern_type="lead_lag",
            )
            
            # If pattern exists and is active, use it; otherwise, validate it
            if active_patterns:
                # Pattern is already validated and reliable
                pattern = active_patterns[0]
                if pattern.is_active:
                    # Use validated lead-lag
                    lead_lag_minutes = pattern.pattern_details.get('lead_lag_minutes', lead_lag_minutes)
                    logger.debug("using_reliable_lead_lag", pattern_id=pattern_id, lead_lag_minutes=lead_lag_minutes)
            else:
                # Validate pattern on full history
                if len(returns1) >= self.pattern_detector.min_observations and len(returns2) >= self.pattern_detector.min_observations:
                    timestamps = np.array(self.timestamps_history.get(symbol1, []))
                    if len(timestamps) == 0:
                        timestamps = np.arange(len(returns1))
                    
                    historical_data = {
                        f"{symbol1}_returns": np.array(returns1),
                        f"{symbol2}_returns": np.array(returns2),
                        "timestamps": timestamps,
                    }
                    
                    pattern = self.pattern_detector.validate_pattern_on_full_history(
                        coin1=symbol1,
                        coin2=symbol2,
                        pattern_type="lead_lag",
                        historical_data=historical_data,
                    )
                    
                    if pattern and pattern.is_active:
                        # Pattern is reliable, use it
                        lead_lag_minutes = pattern.pattern_details.get('lead_lag_minutes', lead_lag_minutes)
                        logger.info("validated_reliable_lead_lag", pattern_id=pattern_id, lead_lag_minutes=lead_lag_minutes)
                    else:
                        # Pattern not reliable, don't use lead-lag
                        lead_lag_minutes = None
                        logger.debug("lead_lag_not_reliable", pattern_id=pattern_id)

        return CorrelationMetrics(
            asset1=symbol1,
            asset2=symbol2,
            correlation=full_corr,
            rolling_correlation=rolling_corr,
            correlation_strength=strength,
            lead_lag_minutes=lead_lag_minutes,
        )

    def get_correlation_matrix(
        self,
        symbols: List[str],
    ) -> Dict[Tuple[str, str], float]:
        """Get correlation matrix for given symbols."""
        matrix = {}

        for i, sym1 in enumerate(symbols):
            for sym2 in symbols[i + 1:]:
                corr_metrics = self.get_correlation(sym1, sym2)
                if corr_metrics:
                    matrix[(sym1, sym2)] = corr_metrics.correlation

        return matrix

    def get_diversification_score(
        self,
        portfolio_symbols: List[str],
    ) -> float:
        """Get simple diversification score (0-1) for portfolio."""
        risk = self.analyze_portfolio_risk(portfolio_symbols)
        return risk.diversification_ratio

    def _calculate_lead_lag(
        self,
        symbol1: str,
        symbol2: str,
        max_lag_minutes: int = 30,
    ) -> Optional[float]:
        """
        Calculate lead-lag relationship using cross-correlation.
        
        Returns:
            Lead-lag in minutes. Positive = symbol1 leads symbol2.
            Negative = symbol2 leads symbol1. None if insufficient data.
        """
        if symbol1 not in self.returns_history or symbol2 not in self.returns_history:
            return None
        
        returns1 = self.returns_history[symbol1]
        returns2 = self.returns_history[symbol2]
        timestamps1 = self.timestamps_history[symbol1]
        timestamps2 = self.timestamps_history[symbol2]
        
        # Need at least min_periods
        if len(returns1) < self.min_periods or len(returns2) < self.min_periods:
            return None
        
        # Align by timestamps
        min_len = min(len(returns1), len(returns2))
        returns1 = np.array(returns1[-min_len:])
        returns2 = np.array(returns2[-min_len:])
        timestamps1 = np.array(timestamps1[-min_len:])
        timestamps2 = np.array(timestamps2[-min_len:])
        
        # Calculate average time difference between samples
        if len(timestamps1) < 2 or len(timestamps2) < 2:
            return None
        
        avg_interval1 = np.mean(np.diff(timestamps1)) / 60.0  # Convert to minutes
        avg_interval2 = np.mean(np.diff(timestamps2)) / 60.0
        
        # Use the average interval
        avg_interval = (avg_interval1 + avg_interval2) / 2.0
        
        # Calculate cross-correlation with lags
        max_lag_samples = int(max_lag_minutes / avg_interval) if avg_interval > 0 else 10
        max_lag_samples = min(max_lag_samples, min_len // 4)  # Don't use more than 25% of data
        
        if max_lag_samples < 1:
            return None
        
        best_corr = -1.0
        best_lag = 0
        
        # Try different lags
        for lag in range(-max_lag_samples, max_lag_samples + 1):
            if lag == 0:
                # No lag
                corr = np.corrcoef(returns1, returns2)[0, 1]
            elif lag > 0:
                # symbol1 leads (shift symbol2 forward)
                if lag >= len(returns2):
                    continue
                corr = np.corrcoef(returns1[:-lag], returns2[lag:])[0, 1]
            else:
                # symbol2 leads (shift symbol1 forward)
                lag_abs = abs(lag)
                if lag_abs >= len(returns1):
                    continue
                corr = np.corrcoef(returns1[lag_abs:], returns2[:-lag_abs])[0, 1]
            
            if not np.isnan(corr) and abs(corr) > abs(best_corr):
                best_corr = corr
                best_lag = lag
        
        # Convert lag to minutes
        if abs(best_corr) < 0.3:  # Weak correlation, don't trust the lag
            return None
        
        lead_lag_minutes = best_lag * avg_interval
        
        return lead_lag_minutes
