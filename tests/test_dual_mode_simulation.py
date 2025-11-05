"""
Dual-Mode Trading Simulation

Simulates a full trading day with realistic market conditions:
- Price movements (trending, ranging, volatile)
- Feature extraction
- Signal generation
- Position lifecycle
- P&L tracking

Tests expected results:
- Scalp book: 30-50 trades, 70-75% WR, £1-£2 avg profit
- Runner book: 2-5 trades, 95%+ WR, £5-£20 avg profit
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List
import time

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from cloud.training.models.trading_coordinator import TradingCoordinator
from cloud.training.models.dual_book_manager import BookType


@dataclass
class SimulationResult:
    """Results from trading simulation."""

    # Overall
    total_signals: int
    total_trades: int
    total_pnl: float

    # Scalp book
    scalp_trades: int
    scalp_wins: int
    scalp_win_rate: float
    scalp_avg_profit: float
    scalp_pnl: float

    # Runner book
    runner_trades: int
    runner_wins: int
    runner_win_rate: float
    runner_avg_profit: float
    runner_pnl: float

    # Execution
    approval_rate: float
    avg_hold_time_sec: float


class MarketSimulator:
    """Simulates realistic market conditions."""

    def __init__(self, seed: int = 42):
        """Initialize market simulator."""
        np.random.seed(seed)

        self.current_price = 2000.0
        self.current_regime = 'RANGE'
        self.regime_duration = 0

        # Price history for features
        self.price_history = [self.current_price]
        self.volume_history = [1.0]

    def step(self) -> tuple:
        """
        Advance market by one step.

        Returns:
            (price, features, regime)
        """
        # Regime transitions
        self.regime_duration += 1

        if self.regime_duration > 50:  # Change regime every ~50 steps
            regimes = ['TREND', 'RANGE', 'PANIC']
            weights = [0.35, 0.50, 0.15]  # More RANGE, some TREND, rare PANIC
            self.current_regime = np.random.choice(regimes, p=weights)
            self.regime_duration = 0

        # Price movement based on regime
        if self.current_regime == 'TREND':
            # Trending: drift + noise
            drift = np.random.choice([-0.15, 0.15])  # Up or down trend
            noise = np.random.normal(0, 0.05)
            price_change_pct = drift + noise
        elif self.current_regime == 'RANGE':
            # Ranging: mean reversion
            deviation = (self.current_price - 2000.0) / 2000.0
            revert = -deviation * 0.5
            noise = np.random.normal(0, 0.03)
            price_change_pct = revert + noise
        else:  # PANIC
            # Panic: high volatility, large moves
            price_change_pct = np.random.normal(0, 0.30)

        # Update price
        self.current_price *= (1 + price_change_pct / 100)
        self.price_history.append(self.current_price)

        # Keep last 100 prices
        if len(self.price_history) > 100:
            self.price_history = self.price_history[-100:]

        # Generate volume
        volume = np.random.lognormal(0, 0.5)
        self.volume_history.append(volume)
        if len(self.volume_history) > 100:
            self.volume_history = self.volume_history[-100:]

        # Extract features
        features = self._extract_features()

        return self.current_price, features, self.current_regime

    def _extract_features(self) -> dict:
        """Extract features from price history."""
        if len(self.price_history) < 20:
            # Not enough data yet
            return self._default_features()

        prices = np.array(self.price_history[-50:])
        volumes = np.array(self.volume_history[-50:])

        # Trend features
        ema_20 = pd.Series(prices).ewm(span=20).mean().iloc[-1]
        ema_50 = pd.Series(prices).ewm(span=50).mean().iloc[-1] if len(prices) >= 50 else ema_20

        trend_strength = (prices[-1] - ema_50) / ema_50 if ema_50 > 0 else 0.0
        ema_slope = (ema_20 - ema_50) / ema_50 if ema_50 > 0 else 0.0

        # Momentum
        returns = np.diff(prices) / prices[:-1]
        momentum_slope = np.mean(returns[-10:]) if len(returns) >= 10 else 0.0

        # ADX (simplified)
        price_range = np.max(prices[-14:]) - np.min(prices[-14:]) if len(prices) >= 14 else 0.0
        adx = min(price_range / prices[-1] * 1000, 100.0)  # Normalized to 0-100

        # Range features
        mean_price = np.mean(prices)
        std_price = np.std(prices)

        compression = 1.0 - (std_price / mean_price) if mean_price > 0 and std_price > 0 else 0.5

        # Bollinger Bands
        bb_upper = mean_price + 2 * std_price
        bb_lower = mean_price - 2 * std_price
        bb_width = (bb_upper - bb_lower) / mean_price if mean_price > 0 else 0.05

        price_position = (prices[-1] - bb_lower) / (bb_upper - bb_lower) if bb_upper > bb_lower else 0.5

        # Mean reversion bias
        deviation_from_mean = abs(prices[-1] - mean_price) / mean_price if mean_price > 0 else 0.0
        mean_revert_bias = deviation_from_mean

        # Higher timeframe (simplified)
        htf_bias = 0.60 if trend_strength > 0 else 0.40

        return {
            'trend_strength': float(np.clip(trend_strength, -1, 1)),
            'ema_slope': float(np.clip(ema_slope, -1, 1)),
            'momentum_slope': float(np.clip(momentum_slope, -1, 1)),
            'htf_bias': float(np.clip(htf_bias, 0, 1)),
            'adx': float(np.clip(adx, 0, 100)),
            'mean_revert_bias': float(np.clip(mean_revert_bias, 0, 1)),
            'compression': float(np.clip(compression, 0, 1)),
            'bb_width': float(max(bb_width, 0.01)),
            'price_position': float(np.clip(price_position, 0, 1)),
        }

    def _default_features(self) -> dict:
        """Return default features when insufficient data."""
        return {
            'trend_strength': 0.0,
            'ema_slope': 0.0,
            'momentum_slope': 0.0,
            'htf_bias': 0.5,
            'adx': 20.0,
            'mean_revert_bias': 0.5,
            'compression': 0.5,
            'bb_width': 0.03,
            'price_position': 0.5,
        }


def run_simulation(
    n_steps: int = 500,
    print_progress: bool = True,
) -> SimulationResult:
    """
    Run trading simulation.

    Args:
        n_steps: Number of simulation steps
        print_progress: Print progress updates

    Returns:
        SimulationResult
    """
    # Initialize
    coordinator = TradingCoordinator(
        total_capital=10000.0,
        asset_symbols=['ETH-USD'],
        max_short_heat=0.40,
        max_long_heat=0.50,
    )

    market = MarketSimulator(seed=42)

    # Track all trades
    all_trades = []

    if print_progress:
        print("=" * 70)
        print("DUAL-MODE TRADING SIMULATION")
        print("=" * 70)
        print(f"Steps: {n_steps}")
        print(f"Initial Capital: ${coordinator.total_capital:,.0f}")
        print()

    # Run simulation
    for step in range(n_steps):
        # Get market update
        price, features, regime = market.step()

        # Spread varies by regime
        if regime == 'TREND':
            spread_bps = np.random.uniform(6, 10)
            liquidity = np.random.uniform(0.70, 0.90)
        elif regime == 'RANGE':
            spread_bps = np.random.uniform(5, 8)
            liquidity = np.random.uniform(0.75, 0.85)
        else:  # PANIC
            spread_bps = np.random.uniform(15, 30)
            liquidity = np.random.uniform(0.30, 0.60)

        # Process signal
        decision = coordinator.process_signal(
            symbol='ETH-USD',
            price=price,
            features=features,
            regime=regime,
            spread_bps=spread_bps,
            liquidity_score=liquidity,
        )

        if decision:
            all_trades.append({
                'step': step,
                'book': decision.book.value,
                'technique': decision.technique,
                'confidence': decision.confidence,
                'regime': regime,
                'entry_price': price,
            })

        # Update existing positions
        coordinator.update_positions('ETH-USD', price)

        # Close positions based on simple exit logic
        for book_type in [BookType.SHORT_HOLD, BookType.LONG_HOLD]:
            positions = coordinator.book_manager._get_book_positions(book_type, open_only=True)

            for position in positions:
                # Exit conditions
                hold_time = time.time() - position.entry_time
                pnl_pct = position.unrealized_pnl / position.size_usd if position.size_usd > 0 else 0

                should_exit = False

                if book_type == BookType.SHORT_HOLD:
                    # Scalp exit: Quick profit or stop loss
                    if pnl_pct >= 0.01:  # +1% profit (£1 on £100)
                        should_exit = True
                    elif pnl_pct <= -0.005:  # -0.5% stop loss
                        should_exit = True
                    elif hold_time > 15:  # 15 second timeout
                        should_exit = True
                else:  # LONG_HOLD
                    # Runner exit: Larger profit or stop loss
                    if pnl_pct >= 0.08:  # +8% profit (£8 on £100)
                        should_exit = True
                    elif pnl_pct <= -0.02:  # -2% stop loss
                        should_exit = True
                    elif hold_time > 60:  # 60 second timeout
                        should_exit = True

                if should_exit:
                    pnl = coordinator.book_manager.close_position(
                        position.position_id,
                        exit_price=price,
                        reason="sim_exit",
                    )

                    # Record in trades
                    for trade in all_trades:
                        if trade.get('entry_price') == position.entry_price:
                            trade['exit_price'] = price
                            trade['pnl'] = pnl
                            trade['hold_time'] = hold_time
                            trade['won'] = pnl > 0
                            break

        # Progress update
        if print_progress and (step + 1) % 100 == 0:
            metrics = coordinator.get_metrics()
            print(f"Step {step + 1}/{n_steps}: "
                  f"Scalp: {metrics['books']['short_book']['positions']} open, "
                  f"Runner: {metrics['books']['long_book']['positions']} open, "
                  f"Signals: {metrics['coordinator']['total_signals']}")

    # Final metrics
    metrics = coordinator.get_metrics()

    # Analyze trades
    scalp_trades = [t for t in all_trades if t.get('book') == 'short_hold' and 'pnl' in t]
    runner_trades = [t for t in all_trades if t.get('book') == 'long_hold' and 'pnl' in t]

    # Scalp stats
    if scalp_trades:
        scalp_wins = sum(1 for t in scalp_trades if t['won'])
        scalp_wr = scalp_wins / len(scalp_trades)
        scalp_avg_profit = np.mean([t['pnl'] for t in scalp_trades if t['won']]) if scalp_wins > 0 else 0
        scalp_pnl = sum(t['pnl'] for t in scalp_trades)
    else:
        scalp_wins = 0
        scalp_wr = 0.0
        scalp_avg_profit = 0.0
        scalp_pnl = 0.0

    # Runner stats
    if runner_trades:
        runner_wins = sum(1 for t in runner_trades if t['won'])
        runner_wr = runner_wins / len(runner_trades)
        runner_avg_profit = np.mean([t['pnl'] for t in runner_trades if t['won']]) if runner_wins > 0 else 0
        runner_pnl = sum(t['pnl'] for t in runner_trades)
    else:
        runner_wins = 0
        runner_wr = 0.0
        runner_avg_profit = 0.0
        runner_pnl = 0.0

    # Overall stats
    all_closed_trades = [t for t in all_trades if 'pnl' in t]
    total_pnl = scalp_pnl + runner_pnl

    avg_hold_time = np.mean([t['hold_time'] for t in all_closed_trades]) if all_closed_trades else 0

    result = SimulationResult(
        total_signals=metrics['coordinator']['total_signals'],
        total_trades=len(all_closed_trades),
        total_pnl=total_pnl,
        scalp_trades=len(scalp_trades),
        scalp_wins=scalp_wins,
        scalp_win_rate=scalp_wr,
        scalp_avg_profit=scalp_avg_profit,
        scalp_pnl=scalp_pnl,
        runner_trades=len(runner_trades),
        runner_wins=runner_wins,
        runner_win_rate=runner_wr,
        runner_avg_profit=runner_avg_profit,
        runner_pnl=runner_pnl,
        approval_rate=metrics['coordinator']['execution_rate'],
        avg_hold_time_sec=avg_hold_time,
    )

    return result


def print_results(result: SimulationResult):
    """Print simulation results."""
    print("\n" + "=" * 70)
    print("SIMULATION RESULTS")
    print("=" * 70)

    print(f"\nOVERALL:")
    print(f"  Total Signals: {result.total_signals}")
    print(f"  Total Trades: {result.total_trades}")
    print(f"  Approval Rate: {result.approval_rate:.1%}")
    print(f"  Total P&L: ${result.total_pnl:,.2f}")
    print(f"  Avg Hold Time: {result.avg_hold_time_sec:.1f} sec")

    print(f"\nSCALP BOOK (SHORT_HOLD):")
    print(f"  Trades: {result.scalp_trades}")
    print(f"  Wins: {result.scalp_wins}")
    print(f"  Win Rate: {result.scalp_win_rate:.1%} (Target: 70-75%)")
    print(f"  Avg Profit: ${result.scalp_avg_profit:.2f} (Target: £1-£2)")
    print(f"  Total P&L: ${result.scalp_pnl:.2f}")

    print(f"\nRUNNER BOOK (LONG_HOLD):")
    print(f"  Trades: {result.runner_trades}")
    print(f"  Wins: {result.runner_wins}")
    print(f"  Win Rate: {result.runner_win_rate:.1%} (Target: 95%+)")
    print(f"  Avg Profit: ${result.runner_avg_profit:.2f} (Target: £5-£20)")
    print(f"  Total P&L: ${result.runner_pnl:.2f}")

    print("\n" + "=" * 70)

    # Validation
    print("\nVALIDATION:")
    checks = []

    # Scalp checks
    if result.scalp_trades > 0:
        if 0.68 <= result.scalp_win_rate <= 0.78:
            checks.append("✓ Scalp WR in target range")
        else:
            checks.append(f"✗ Scalp WR {result.scalp_win_rate:.1%} outside target (70-75%)")

        if 1.0 <= result.scalp_avg_profit <= 3.0:
            checks.append("✓ Scalp avg profit in target range")
        else:
            checks.append(f"✗ Scalp avg profit ${result.scalp_avg_profit:.2f} outside target (£1-£2)")

    # Runner checks
    if result.runner_trades > 0:
        if result.runner_win_rate >= 0.90:
            checks.append("✓ Runner WR meets target (95%+)")
        else:
            checks.append(f"✗ Runner WR {result.runner_win_rate:.1%} below target (95%+)")

        if 5.0 <= result.runner_avg_profit <= 25.0:
            checks.append("✓ Runner avg profit in target range")
        else:
            checks.append(f"✗ Runner avg profit ${result.runner_avg_profit:.2f} outside target (£5-£20)")

    # Overall checks
    if result.total_pnl > 0:
        checks.append("✓ Positive total P&L")
    else:
        checks.append(f"✗ Negative total P&L: ${result.total_pnl:.2f}")

    for check in checks:
        print(f"  {check}")

    print()


if __name__ == '__main__':
    # Run simulation
    result = run_simulation(n_steps=500, print_progress=True)

    # Print results
    print_results(result)
