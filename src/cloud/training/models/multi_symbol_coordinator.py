"""
Multi-Symbol Coordination Module

Coordinates trading across multiple symbols to:
1. Detect and avoid correlated asset overexposure
2. Manage portfolio-level risk
3. Coordinate entries/exits across symbols
4. Share learning across correlated pairs

Based on advanced portfolio management principles.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import structlog

from .market_structure import MarketStructureCoordinator, PriceData, MarketRegime

logger = structlog.get_logger()


@dataclass
class SymbolPosition:
    """Represents a position in a single symbol."""

    symbol: str
    entry_price: float
    entry_timestamp: datetime
    position_size_gbp: float
    unrealized_pnl_gbp: float = 0.0
    regime: str = "unknown"


@dataclass
class CorrelationMetrics:
    """Correlation metrics between two symbols."""

    symbol_a: str
    symbol_b: str
    price_correlation: float  # -1 to 1
    return_correlation: float  # -1 to 1
    sample_size: int
    last_updated: datetime


@dataclass
class PortfolioState:
    """Current state of the entire portfolio."""

    total_exposure_gbp: float  # Sum of all position sizes
    total_unrealized_pnl_gbp: float
    num_positions: int
    positions_by_symbol: Dict[str, SymbolPosition]
    heat_score: float  # 0-1, where 1 = maximum risk
    correlated_exposure: Dict[str, float]  # Exposure to correlated groups


class MultiSymbolCoordinator:
    """
    Coordinates trading across multiple symbols.

    Key responsibilities:
    1. Track correlations between symbols
    2. Prevent overexposure to correlated assets
    3. Manage portfolio-level risk
    4. Coordinate entry/exit timing
    """

    def __init__(
        self,
        max_portfolio_heat: float = 0.7,  # Max 70% of capital at risk
        max_correlated_exposure: float = 0.4,  # Max 40% in correlated assets
        correlation_threshold: float = 0.7,  # Consider correlated if > 0.7
        correlation_lookback_days: int = 30,
    ):
        """
        Initialize multi-symbol coordinator.

        Args:
            max_portfolio_heat: Maximum fraction of capital at risk
            max_correlated_exposure: Maximum exposure to correlated group
            correlation_threshold: Threshold for considering symbols correlated
            correlation_lookback_days: Days of history for correlation calculation
        """
        self.max_portfolio_heat = max_portfolio_heat
        self.max_correlated_exposure = max_correlated_exposure
        self.correlation_threshold = correlation_threshold
        self.correlation_lookback_days = correlation_lookback_days

        # State tracking
        self.positions: Dict[str, SymbolPosition] = {}
        self.correlations: Dict[Tuple[str, str], CorrelationMetrics] = {}
        self.correlation_groups: List[Set[str]] = []  # Groups of correlated symbols

        # Price history for correlation calculation
        self.price_history: Dict[str, List[Tuple[datetime, float]]] = {}

        # Market structure intelligence (cross-asset modeling)
        self.market_structure = MarketStructureCoordinator(
            beta_window_days=30,
            leadlag_window_days=30,
            spillover_window_days=30,
        )

        # Current market regime
        self.current_regime: MarketRegime = MarketRegime.UNKNOWN

        logger.info(
            "multi_symbol_coordinator_initialized",
            max_portfolio_heat=max_portfolio_heat,
            max_correlated_exposure=max_correlated_exposure,
            correlation_threshold=correlation_threshold,
            market_structure_enabled=True,
        )

    def update_price(self, symbol: str, timestamp: datetime, price: float) -> None:
        """
        Update price history for a symbol.

        Args:
            symbol: Symbol name
            timestamp: Price timestamp
            price: Current price
        """
        if symbol not in self.price_history:
            self.price_history[symbol] = []

        self.price_history[symbol].append((timestamp, price))

        # Keep only recent history
        cutoff = timestamp - timedelta(days=self.correlation_lookback_days)
        self.price_history[symbol] = [
            (ts, p) for ts, p in self.price_history[symbol] if ts >= cutoff
        ]

    def calculate_correlation(
        self, symbol_a: str, symbol_b: str, timestamp: datetime
    ) -> Optional[CorrelationMetrics]:
        """
        Calculate correlation between two symbols.

        Args:
            symbol_a: First symbol
            symbol_b: Second symbol
            timestamp: Current timestamp

        Returns:
            CorrelationMetrics or None if insufficient data
        """
        if symbol_a not in self.price_history or symbol_b not in self.price_history:
            return None

        history_a = self.price_history[symbol_a]
        history_b = self.price_history[symbol_b]

        if len(history_a) < 10 or len(history_b) < 10:
            return None

        # Align timestamps (find common timestamps)
        timestamps_a = {ts: price for ts, price in history_a}
        timestamps_b = {ts: price for ts, price in history_b}

        common_timestamps = sorted(set(timestamps_a.keys()) & set(timestamps_b.keys()))

        if len(common_timestamps) < 10:
            return None

        # Extract aligned prices
        prices_a = np.array([timestamps_a[ts] for ts in common_timestamps])
        prices_b = np.array([timestamps_b[ts] for ts in common_timestamps])

        # Calculate price correlation
        price_corr = float(np.corrcoef(prices_a, prices_b)[0, 1])

        # Calculate return correlation
        returns_a = np.diff(prices_a) / prices_a[:-1]
        returns_b = np.diff(prices_b) / prices_b[:-1]

        if len(returns_a) > 0:
            return_corr = float(np.corrcoef(returns_a, returns_b)[0, 1])
        else:
            return_corr = 0.0

        return CorrelationMetrics(
            symbol_a=symbol_a,
            symbol_b=symbol_b,
            price_correlation=price_corr,
            return_correlation=return_corr,
            sample_size=len(common_timestamps),
            last_updated=timestamp,
        )

    def update_correlations(self, timestamp: datetime) -> None:
        """
        Update all pairwise correlations.

        Args:
            timestamp: Current timestamp
        """
        symbols = list(self.price_history.keys())

        for i, symbol_a in enumerate(symbols):
            for symbol_b in symbols[i + 1 :]:
                corr = self.calculate_correlation(symbol_a, symbol_b, timestamp)

                if corr is not None:
                    key = (symbol_a, symbol_b) if symbol_a < symbol_b else (symbol_b, symbol_a)
                    self.correlations[key] = corr

        # Update correlation groups
        self._update_correlation_groups()

    def _update_correlation_groups(self) -> None:
        """Update groups of correlated symbols."""
        # Build adjacency list of correlated symbols
        correlated: Dict[str, Set[str]] = {}

        for (symbol_a, symbol_b), metrics in self.correlations.items():
            if abs(metrics.return_correlation) >= self.correlation_threshold:
                if symbol_a not in correlated:
                    correlated[symbol_a] = set()
                if symbol_b not in correlated:
                    correlated[symbol_b] = set()

                correlated[symbol_a].add(symbol_b)
                correlated[symbol_b].add(symbol_a)

        # Find connected components (correlation groups)
        visited = set()
        groups = []

        def dfs(symbol: str, group: Set[str]):
            if symbol in visited:
                return
            visited.add(symbol)
            group.add(symbol)

            for neighbor in correlated.get(symbol, []):
                dfs(neighbor, group)

        for symbol in correlated.keys():
            if symbol not in visited:
                group: Set[str] = set()
                dfs(symbol, group)
                groups.append(group)

        self.correlation_groups = groups

        if groups:
            logger.debug(
                "correlation_groups_updated",
                num_groups=len(groups),
                groups=[list(g) for g in groups],
            )

    def get_correlated_symbols(self, symbol: str) -> Set[str]:
        """Get all symbols correlated with the given symbol."""
        for group in self.correlation_groups:
            if symbol in group:
                return group - {symbol}  # Exclude self

        return set()

    def can_enter_position(
        self,
        symbol: str,
        position_size_gbp: float,
        total_capital_gbp: float,
    ) -> Tuple[bool, str]:
        """
        Check if we can enter a position without exceeding risk limits.

        Args:
            symbol: Symbol to trade
            position_size_gbp: Proposed position size
            total_capital_gbp: Total available capital

        Returns:
            (can_enter, reason) tuple
        """
        # Check portfolio heat
        current_exposure = sum(pos.position_size_gbp for pos in self.positions.values())
        new_exposure = current_exposure + position_size_gbp
        new_heat = new_exposure / total_capital_gbp

        if new_heat > self.max_portfolio_heat:
            return (
                False,
                f"Portfolio heat {new_heat:.1%} exceeds max {self.max_portfolio_heat:.1%}",
            )

        # Check correlated exposure
        correlated_symbols = self.get_correlated_symbols(symbol)
        correlated_exposure = sum(
            pos.position_size_gbp
            for sym, pos in self.positions.items()
            if sym in correlated_symbols
        )
        new_correlated_exposure = correlated_exposure + position_size_gbp
        correlated_heat = new_correlated_exposure / total_capital_gbp

        if correlated_heat > self.max_correlated_exposure:
            return (
                False,
                f"Correlated exposure {correlated_heat:.1%} exceeds max {self.max_correlated_exposure:.1%}. "
                f"Correlated with: {', '.join(correlated_symbols)}",
            )

        # Check if already have position in this symbol
        if symbol in self.positions:
            return (False, f"Already have position in {symbol}")

        return (True, "OK")

    def add_position(self, position: SymbolPosition) -> None:
        """Add a new position to portfolio."""
        self.positions[position.symbol] = position

        logger.info(
            "position_added",
            symbol=position.symbol,
            size_gbp=position.position_size_gbp,
            total_positions=len(self.positions),
        )

    def remove_position(self, symbol: str) -> Optional[SymbolPosition]:
        """Remove a position from portfolio."""
        position = self.positions.pop(symbol, None)

        if position:
            logger.info(
                "position_removed",
                symbol=symbol,
                pnl_gbp=position.unrealized_pnl_gbp,
                total_positions=len(self.positions),
            )

        return position

    def update_position_pnl(
        self, symbol: str, current_price: float
    ) -> Optional[float]:
        """
        Update position PnL.

        Args:
            symbol: Symbol name
            current_price: Current market price

        Returns:
            Updated unrealized PnL or None if position doesn't exist
        """
        position = self.positions.get(symbol)
        if not position:
            return None

        # Calculate PnL
        price_change = current_price - position.entry_price
        pnl_gbp = (price_change / position.entry_price) * position.position_size_gbp

        position.unrealized_pnl_gbp = pnl_gbp

        return pnl_gbp

    def get_portfolio_state(self, total_capital_gbp: float) -> PortfolioState:
        """
        Get current portfolio state.

        Args:
            total_capital_gbp: Total available capital

        Returns:
            PortfolioState snapshot
        """
        total_exposure = sum(pos.position_size_gbp for pos in self.positions.values())
        total_pnl = sum(pos.unrealized_pnl_gbp for pos in self.positions.values())
        heat_score = total_exposure / total_capital_gbp if total_capital_gbp > 0 else 0.0

        # Calculate correlated exposure
        correlated_exposure = {}
        for i, group in enumerate(self.correlation_groups):
            group_exposure = sum(
                pos.position_size_gbp
                for sym, pos in self.positions.items()
                if sym in group
            )
            if group_exposure > 0:
                correlated_exposure[f"group_{i}"] = group_exposure

        return PortfolioState(
            total_exposure_gbp=total_exposure,
            total_unrealized_pnl_gbp=total_pnl,
            num_positions=len(self.positions),
            positions_by_symbol=self.positions.copy(),
            heat_score=heat_score,
            correlated_exposure=correlated_exposure,
        )

    def should_reduce_exposure(
        self, total_capital_gbp: float, current_drawdown_pct: float
    ) -> Tuple[bool, str]:
        """
        Determine if portfolio exposure should be reduced.

        Args:
            total_capital_gbp: Total available capital
            current_drawdown_pct: Current portfolio drawdown as percentage

        Returns:
            (should_reduce, reason) tuple
        """
        state = self.get_portfolio_state(total_capital_gbp)

        # Reduce if heat score too high
        if state.heat_score > self.max_portfolio_heat:
            return (True, f"Portfolio heat {state.heat_score:.1%} exceeds limit")

        # Reduce if in significant drawdown
        if current_drawdown_pct > 15.0:  # More than 15% drawdown
            return (True, f"Portfolio in {current_drawdown_pct:.1%} drawdown")

        return (False, "OK")

    # ========================================
    # MARKET STRUCTURE INTEGRATION
    # (Cross-Asset Intelligence)
    # ========================================

    def update_leader_price(self, leader: str, timestamp: datetime, price: float) -> None:
        """
        Update market leader price data.

        Args:
            leader: Leader symbol ("BTC", "ETH", "SOL")
            timestamp: Price timestamp
            price: Current price
        """
        # Also update in regular price history
        self.update_price(leader, timestamp, price)

        # Convert to PriceData format for market structure
        if leader in self.price_history:
            timestamps = [ts for ts, _ in self.price_history[leader]]
            prices = [p for _, p in self.price_history[leader]]

            price_data = PriceData(timestamps=timestamps, prices=prices)
            self.market_structure.update_leader_data(leader, price_data)

    def get_asset_beta(self, symbol: str, benchmark: str = "BTC") -> Optional[float]:
        """
        Get beta for an asset vs benchmark.

        Args:
            symbol: Asset symbol
            benchmark: Benchmark symbol (default: BTC)

        Returns:
            Beta value or None
        """
        if symbol not in self.price_history or not self.price_history[symbol]:
            return None

        timestamps = [ts for ts, _ in self.price_history[symbol]]
        prices = [p for _, p in self.price_history[symbol]]
        asset_data = PriceData(timestamps=timestamps, prices=prices)

        current_time = timestamps[-1] if timestamps else datetime.now()

        beta_metrics = self.market_structure.get_beta(
            symbol, asset_data, benchmark, current_time
        )

        return beta_metrics.beta if beta_metrics else None

    def detect_market_regime_for_symbol(self, symbol: str) -> Tuple[MarketRegime, float]:
        """
        Detect market regime for a specific symbol.

        Args:
            symbol: Symbol name

        Returns:
            (regime, confidence) tuple
        """
        if symbol not in self.price_history or not self.price_history[symbol]:
            return (MarketRegime.UNKNOWN, 0.0)

        timestamps = [ts for ts, _ in self.price_history[symbol]]
        prices = [p for _, p in self.price_history[symbol]]
        asset_data = PriceData(timestamps=timestamps, prices=prices)

        current_time = timestamps[-1] if timestamps else datetime.now()

        snapshot = self.market_structure.detect_market_regime(
            symbol, asset_data, current_time
        )

        self.current_regime = snapshot.regime

        return (snapshot.regime, snapshot.regime_confidence)

    def should_trade_based_on_leaders(
        self, symbol: str, direction: str
    ) -> Tuple[bool, str]:
        """
        Determine if trade should be taken based on leader behavior.

        Args:
            symbol: Symbol to trade
            direction: "buy" or "sell"

        Returns:
            (should_trade, reason) tuple
        """
        # Get current regime
        regime, confidence = self.detect_market_regime_for_symbol(symbol)

        # Get beta vs BTC
        beta_btc = self.get_asset_beta(symbol, "BTC")

        # Risk-off regime: avoid longs in high-beta assets
        if regime == MarketRegime.RISK_OFF and direction == "buy":
            if beta_btc and beta_btc > 1.2:
                return (
                    False,
                    f"RISK_OFF regime: avoid high-beta ({beta_btc:.2f}) longs",
                )

        # BTC leader data available?
        if "BTC" in self.price_history and len(self.price_history["BTC"]) > 20:
            btc_prices = [p for _, p in self.price_history["BTC"][-20:]]
            btc_recent_change = (btc_prices[-1] - btc_prices[-10]) / btc_prices[-10]

            # BTC dumping: be cautious on alt longs
            if btc_recent_change < -0.05 and direction == "buy":  # BTC down 5%+
                if beta_btc and beta_btc > 0.8:
                    return (
                        False,
                        f"BTC down {btc_recent_change:.1%}: avoid correlated longs",
                    )

        # Divergence opportunity
        if regime == MarketRegime.DIVERGENCE:
            return (
                True,
                f"DIVERGENCE regime: good opportunity (conf={confidence:.1%})",
            )

        # Risk-on: good for longs
        if regime == MarketRegime.RISK_ON and direction == "buy":
            return (
                True,
                f"RISK_ON regime: favorable for longs (conf={confidence:.1%})",
            )

        # Default: allow trade
        return (True, f"Regime: {regime.value} (conf={confidence:.1%})")
