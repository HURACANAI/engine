"""
Reliable Cross-Coin Pattern Detector

Only uses patterns that are consistently true (>90% reliability).
Tracks pattern stability over time and auto-disables unreliable patterns.

Key Features:
1. Track pattern reliability over time
2. Only use patterns with >90% reliability
3. Auto-disable patterns that become unreliable
4. Validate patterns on full historical data
5. Detect lead-lag relationships that are always true

Usage:
    detector = ReliablePatternDetector(min_reliability=0.90)
    
    # Validate pattern on full history
    pattern = detector.validate_pattern(
        coin1="BTC",
        coin2="ETH",
        pattern_type="lead_lag",
        historical_data=full_history,
    )
    
    if pattern.is_active:
        # Pattern is reliable, use it
        use_pattern(pattern)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone, timedelta
import numpy as np
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class PatternReliability:
    """Reliability metrics for a pattern."""
    pattern_id: str
    coin1: str
    coin2: str
    pattern_type: str  # 'lead_lag', 'correlation', 'volatility_spillover'
    reliability: float  # 0-1, percentage of time pattern holds
    confidence: float  # 0-1, confidence in reliability estimate
    n_observations: int
    n_verified: int  # Number of times pattern was verified
    last_verified: datetime
    is_active: bool  # True if pattern is reliable enough to use
    min_reliability_threshold: float = 0.90  # Must be >90% reliable
    pattern_details: Optional[Dict] = None  # Pattern-specific details (e.g., lag_minutes)


@dataclass
class PatternObservation:
    """Single observation of a pattern."""
    timestamp: datetime
    pattern_id: str
    pattern_held: bool  # True if pattern was true at this time
    context: Dict  # Additional context (regime, volatility, etc.)


class ReliablePatternDetector:
    """
    Detects cross-coin patterns and only uses those that are consistently true.
    
    Key Features:
    1. Tracks pattern reliability over time
    2. Only uses patterns with >90% reliability
    3. Auto-disables patterns that become unreliable
    4. Validates patterns on full historical data
    5. Detects lead-lag relationships that are always true
    
    Pattern Types:
    - lead_lag: Coin1 leads Coin2 by X minutes (always)
    - correlation: Coin1 and Coin2 move together (always)
    - volatility_spillover: Coin1 volatility spills to Coin2 (always)
    """
    
    def __init__(
        self,
        min_reliability: float = 0.90,  # Must be >90% reliable
        min_observations: int = 100,  # Minimum observations to trust reliability
        confidence_threshold: float = 0.80,  # Minimum confidence to use pattern
        lookback_days: int = 365,  # Days to look back for validation
    ):
        """
        Initialize reliable pattern detector.
        
        Args:
            min_reliability: Minimum reliability (0-1) to use pattern
            min_observations: Minimum observations to trust reliability
            confidence_threshold: Minimum confidence to use pattern
            lookback_days: Days to look back for validation
        """
        self.min_reliability = min_reliability
        self.min_observations = min_observations
        self.confidence_threshold = confidence_threshold
        self.lookback_days = lookback_days
        
        # Store pattern reliability data
        self.patterns: Dict[str, PatternReliability] = {}
        self.observations: Dict[str, List[PatternObservation]] = {}
        
        logger.info(
            "reliable_pattern_detector_initialized",
            min_reliability=min_reliability,
            min_observations=min_observations,
            confidence_threshold=confidence_threshold,
        )
    
    def _generate_pattern_id(self, coin1: str, coin2: str, pattern_type: str) -> str:
        """Generate unique pattern ID."""
        return f"{pattern_type}_{coin1}_{coin2}"
    
    def validate_lead_lag_pattern(
        self,
        coin1: str,
        coin2: str,
        coin1_returns: np.ndarray,
        coin2_returns: np.ndarray,
        timestamps: np.ndarray,
        max_lag_minutes: int = 30,
    ) -> Optional[PatternReliability]:
        """
        Validate lead-lag pattern on historical data.
        
        Checks if coin1 consistently leads coin2 by a fixed amount.
        Only returns pattern if it's >90% reliable.
        
        Args:
            coin1: Leading coin symbol
            coin2: Following coin symbol
            coin1_returns: Returns for coin1
            coin2_returns: Returns for coin2
            timestamps: Timestamps for returns
            max_lag_minutes: Maximum lag to check (minutes)
        
        Returns:
            PatternReliability if pattern is reliable, None otherwise
        """
        pattern_id = self._generate_pattern_id(coin1, coin2, "lead_lag")
        
        if len(coin1_returns) < self.min_observations or len(coin2_returns) < self.min_observations:
            logger.warning("insufficient_data_for_validation", pattern_id=pattern_id)
            return None
        
        # Calculate average time interval
        if len(timestamps) < 2:
            return None
        
        time_diffs = np.diff(timestamps)
        avg_interval_minutes = np.mean(time_diffs) / 60.0
        
        # Calculate cross-correlation with different lags
        max_lag_samples = int(max_lag_minutes / avg_interval_minutes) if avg_interval_minutes > 0 else 10
        max_lag_samples = min(max_lag_samples, len(coin1_returns) // 4)
        
        if max_lag_samples < 1:
            return None
        
        best_corr = -1.0
        best_lag = 0
        lag_correlations = []
        
        # Try different lags
        for lag in range(-max_lag_samples, max_lag_samples + 1):
            if lag == 0:
                corr = np.corrcoef(coin1_returns, coin2_returns)[0, 1]
            elif lag > 0:
                # coin1 leads (shift coin2 forward)
                if lag >= len(coin2_returns):
                    continue
                corr = np.corrcoef(coin1_returns[:-lag], coin2_returns[lag:])[0, 1]
            else:
                # coin2 leads (shift coin1 forward)
                lag_abs = abs(lag)
                if lag_abs >= len(coin1_returns):
                    continue
                corr = np.corrcoef(coin1_returns[lag_abs:], coin2_returns[:-lag_abs])[0, 1]
            
            if not np.isnan(corr):
                lag_correlations.append((lag, abs(corr)))
                if abs(corr) > abs(best_corr):
                    best_corr = corr
                    best_lag = lag
        
        if abs(best_corr) < 0.3:
            # Weak correlation, no reliable pattern
            return None
        
        # Convert lag to minutes
        lead_lag_minutes = best_lag * avg_interval_minutes
        
        # Now validate: does this lag hold consistently?
        # Check if correlation at this lag is consistently high
        n_windows = min(20, len(coin1_returns) // 100)  # Check 20 windows
        window_size = len(coin1_returns) // n_windows if n_windows > 0 else len(coin1_returns)
        
        reliable_count = 0
        total_windows = 0
        
        for i in range(0, len(coin1_returns) - window_size, window_size):
            window_coin1 = coin1_returns[i:i+window_size]
            window_coin2 = coin2_returns[i:i+window_size]
            
            if len(window_coin1) < 10:
                continue
            
            # Check correlation at best_lag
            if best_lag == 0:
                window_corr = np.corrcoef(window_coin1, window_coin2)[0, 1]
            elif best_lag > 0:
                if best_lag >= len(window_coin2):
                    continue
                window_corr = np.corrcoef(window_coin1[:-best_lag], window_coin2[best_lag:])[0, 1]
            else:
                lag_abs = abs(best_lag)
                if lag_abs >= len(window_coin1):
                    continue
                window_corr = np.corrcoef(window_coin1[lag_abs:], window_coin2[:-lag_abs])[0, 1]
            
            if not np.isnan(window_corr):
                total_windows += 1
                # Pattern holds if correlation is >0.5 and same direction
                if abs(window_corr) > 0.5 and np.sign(window_corr) == np.sign(best_corr):
                    reliable_count += 1
        
        if total_windows == 0:
            return None
        
        reliability = reliable_count / total_windows
        confidence = min(1.0, total_windows / 20.0)  # More windows = higher confidence
        
        is_active = (
            reliability >= self.min_reliability and
            confidence >= self.confidence_threshold and
            total_windows >= 5  # At least 5 windows
        )
        
        pattern = PatternReliability(
            pattern_id=pattern_id,
            coin1=coin1,
            coin2=coin2,
            pattern_type="lead_lag",
            reliability=reliability,
            confidence=confidence,
            n_observations=len(coin1_returns),
            n_verified=reliable_count,
            last_verified=datetime.now(timezone.utc),
            is_active=is_active,
            min_reliability_threshold=self.min_reliability,
            pattern_details={
                'lead_lag_minutes': lead_lag_minutes,
                'correlation': best_corr,
                'best_lag_samples': best_lag,
            },
        )
        
        if is_active:
            self.patterns[pattern_id] = pattern
            logger.info(
                "reliable_pattern_detected",
                pattern_id=pattern_id,
                reliability=reliability,
                confidence=confidence,
                lead_lag_minutes=lead_lag_minutes,
            )
        
        return pattern
    
    def validate_correlation_pattern(
        self,
        coin1: str,
        coin2: str,
        coin1_returns: np.ndarray,
        coin2_returns: np.ndarray,
        min_correlation: float = 0.70,
    ) -> Optional[PatternReliability]:
        """
        Validate correlation pattern on historical data.
        
        Checks if coin1 and coin2 consistently move together.
        Only returns pattern if it's >90% reliable.
        
        Args:
            coin1: First coin symbol
            coin2: Second coin symbol
            coin1_returns: Returns for coin1
            coin2_returns: Returns for coin2
            min_correlation: Minimum correlation to consider
        
        Returns:
            PatternReliability if pattern is reliable, None otherwise
        """
        pattern_id = self._generate_pattern_id(coin1, coin2, "correlation")
        
        if len(coin1_returns) < self.min_observations or len(coin2_returns) < self.min_observations:
            logger.warning("insufficient_data_for_validation", pattern_id=pattern_id)
            return None
        
        # Calculate full correlation
        full_corr = np.corrcoef(coin1_returns, coin2_returns)[0, 1]
        
        if abs(full_corr) < min_correlation:
            # Not correlated enough
            return None
        
        # Validate: does this correlation hold consistently?
        n_windows = min(20, len(coin1_returns) // 100)
        window_size = len(coin1_returns) // n_windows if n_windows > 0 else len(coin1_returns)
        
        reliable_count = 0
        total_windows = 0
        
        for i in range(0, len(coin1_returns) - window_size, window_size):
            window_coin1 = coin1_returns[i:i+window_size]
            window_coin2 = coin2_returns[i:i+window_size]
            
            if len(window_coin1) < 10:
                continue
            
            window_corr = np.corrcoef(window_coin1, window_coin2)[0, 1]
            
            if not np.isnan(window_corr):
                total_windows += 1
                # Pattern holds if correlation is >min_correlation and same direction
                if abs(window_corr) >= min_correlation and np.sign(window_corr) == np.sign(full_corr):
                    reliable_count += 1
        
        if total_windows == 0:
            return None
        
        reliability = reliable_count / total_windows
        confidence = min(1.0, total_windows / 20.0)
        
        is_active = (
            reliability >= self.min_reliability and
            confidence >= self.confidence_threshold and
            total_windows >= 5
        )
        
        pattern = PatternReliability(
            pattern_id=pattern_id,
            coin1=coin1,
            coin2=coin2,
            pattern_type="correlation",
            reliability=reliability,
            confidence=confidence,
            n_observations=len(coin1_returns),
            n_verified=reliable_count,
            last_verified=datetime.now(timezone.utc),
            is_active=is_active,
            min_reliability_threshold=self.min_reliability,
            pattern_details={
                'correlation': full_corr,
                'min_correlation': min_correlation,
            },
        )
        
        if is_active:
            self.patterns[pattern_id] = pattern
            logger.info(
                "reliable_pattern_detected",
                pattern_id=pattern_id,
                reliability=reliability,
                confidence=confidence,
                correlation=full_corr,
            )
        
        return pattern
    
    def get_active_patterns(
        self,
        coin1: Optional[str] = None,
        coin2: Optional[str] = None,
        pattern_type: Optional[str] = None,
    ) -> List[PatternReliability]:
        """
        Get active (reliable) patterns.
        
        Args:
            coin1: Filter by coin1 (optional)
            coin2: Filter by coin2 (optional)
            pattern_type: Filter by pattern type (optional)
        
        Returns:
            List of active patterns
        """
        active_patterns = [
            p for p in self.patterns.values()
            if p.is_active
        ]
        
        if coin1:
            active_patterns = [p for p in active_patterns if p.coin1 == coin1 or p.coin2 == coin1]
        
        if coin2:
            active_patterns = [p for p in active_patterns if p.coin1 == coin2 or p.coin2 == coin2]
        
        if pattern_type:
            active_patterns = [p for p in active_patterns if p.pattern_type == pattern_type]
        
        return active_patterns
    
    def update_pattern_reliability(
        self,
        pattern_id: str,
        pattern_held: bool,
        context: Optional[Dict] = None,
    ) -> None:
        """
        Update pattern reliability with new observation.
        
        Args:
            pattern_id: Pattern ID
            pattern_held: Whether pattern held true
            context: Additional context
        """
        if pattern_id not in self.patterns:
            return
        
        pattern = self.patterns[pattern_id]
        
        # Add observation
        observation = PatternObservation(
            timestamp=datetime.now(timezone.utc),
            pattern_id=pattern_id,
            pattern_held=pattern_held,
            context=context or {},
        )
        
        if pattern_id not in self.observations:
            self.observations[pattern_id] = []
        
        self.observations[pattern_id].append(observation)
        
        # Keep only recent observations (last N days)
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=self.lookback_days)
        self.observations[pattern_id] = [
            obs for obs in self.observations[pattern_id]
            if obs.timestamp >= cutoff_date
        ]
        
        # Recalculate reliability
        if len(self.observations[pattern_id]) >= self.min_observations:
            verified_count = sum(1 for obs in self.observations[pattern_id] if obs.pattern_held)
            reliability = verified_count / len(self.observations[pattern_id])
            confidence = min(1.0, len(self.observations[pattern_id]) / self.min_observations)
            
            pattern.reliability = reliability
            pattern.confidence = confidence
            pattern.n_observations = len(self.observations[pattern_id])
            pattern.n_verified = verified_count
            pattern.last_verified = datetime.now(timezone.utc)
            
            # Auto-disable if reliability drops below threshold
            if reliability < self.min_reliability or confidence < self.confidence_threshold:
                pattern.is_active = False
                logger.warning(
                    "pattern_disabled",
                    pattern_id=pattern_id,
                    reliability=reliability,
                    confidence=confidence,
                )
            else:
                pattern.is_active = True
    
    def validate_pattern_on_full_history(
        self,
        coin1: str,
        coin2: str,
        pattern_type: str,
        historical_data: Dict[str, np.ndarray],
    ) -> Optional[PatternReliability]:
        """
        Validate pattern on full historical data.
        
        Uses full history to validate pattern consistency.
        Only returns pattern if it's >90% reliable across full history.
        
        Args:
            coin1: First coin symbol
            coin2: Second coin symbol
            pattern_type: Type of pattern ('lead_lag', 'correlation')
            historical_data: Dict with coin1 and coin2 returns and timestamps
        
        Returns:
            PatternReliability if pattern is reliable, None otherwise
        """
        coin1_returns = historical_data.get(f"{coin1}_returns")
        coin2_returns = historical_data.get(f"{coin2}_returns")
        timestamps = historical_data.get("timestamps")
        
        if coin1_returns is None or coin2_returns is None:
            logger.warning("missing_data_for_validation", coin1=coin1, coin2=coin2)
            return None
        
        if pattern_type == "lead_lag":
            return self.validate_lead_lag_pattern(
                coin1=coin1,
                coin2=coin2,
                coin1_returns=coin1_returns,
                coin2_returns=coin2_returns,
                timestamps=timestamps if timestamps is not None else np.arange(len(coin1_returns)),
            )
        elif pattern_type == "correlation":
            return self.validate_correlation_pattern(
                coin1=coin1,
                coin2=coin2,
                coin1_returns=coin1_returns,
                coin2_returns=coin2_returns,
            )
        else:
            logger.warning("unknown_pattern_type", pattern_type=pattern_type)
            return None

