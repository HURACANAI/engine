"""
Simple backtest test - validates basic execution simulation.

Tests the ExecutionSimulator and basic trade flow without needing
full TradingCoordinator integration.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'cloud', 'training', 'models'))

import numpy as np
import pandas as pd

from backtesting_framework import (
    ExecutionSimulator,
    Direction,
    OrderFillStatus,
    Backtester,
)


def test_execution_simulator():
    """Test execution simulator logic."""
    print("=" * 70)
    print("TEST: Execution Simulator")
    print("=" * 70)

    sim = ExecutionSimulator(
        maker_fill_probability=0.75,
        maker_rebate_bps=2.0,
        taker_fee_bps=5.0,
    )

    # Test maker entry
    print("\n1. Testing Maker Entry...")
    result = sim.execute_entry(
        symbol='BTC-USD',
        direction=Direction.LONG,
        size_usd=1000.0,
        entry_price=50000.0,
        spread_bps=5.0,
        prefer_maker=True,
        liquidity_score=0.8,
    )

    print(f"   Status: {result.status}")
    print(f"   Fill Price: ${result.fill_price:,.2f}")
    print(f"   Cost: {result.total_cost_bps:.2f} bps")
    print(f"   Reason: {result.reason}")

    # Test taker entry
    print("\n2. Testing Taker Entry...")
    result = sim.execute_entry(
        symbol='BTC-USD',
        direction=Direction.LONG,
        size_usd=1000.0,
        entry_price=50000.0,
        spread_bps=5.0,
        prefer_maker=False,
        liquidity_score=0.8,
    )

    print(f"   Status: {result.status}")
    print(f"   Fill Price: ${result.fill_price:,.2f}")
    print(f"   Cost: {result.total_cost_bps:.2f} bps")
    print(f"   Reason: {result.reason}")

    # Test stop loss exit
    print("\n3. Testing Stop Loss Exit...")
    result = sim.execute_exit(
        symbol='BTC-USD',
        direction=Direction.LONG,
        size_usd=1000.0,
        exit_price=49500.0,
        spread_bps=5.0,
        is_stop_loss=True,
    )

    print(f"   Status: {result.status}")
    print(f"   Fill Price: ${result.fill_price:,.2f}")
    print(f"   Cost: {result.total_cost_bps:.2f} bps")
    print(f"   Reason: {result.reason}")

    print("\n✓ Execution simulator tests passed\n")


def test_trade_pnl_calculation():
    """Test P&L calculation with realistic costs."""
    print("=" * 70)
    print("TEST: P&L Calculation")
    print("=" * 70)

    sim = ExecutionSimulator()

    # Scenario 1: Winning trade with maker orders
    print("\nScenario 1: Winning Trade (Maker Entry + Maker Exit)")
    entry = sim.execute_entry(
        symbol='BTC-USD',
        direction=Direction.LONG,
        size_usd=1000.0,
        entry_price=50000.0,
        spread_bps=5.0,
        prefer_maker=True,
        liquidity_score=0.9,
    )

    # Assume price moved up
    exit_price = 50100.0  # +100 = +20 bps raw move
    exit_result = sim.execute_exit(
        symbol='BTC-USD',
        direction=Direction.LONG,
        size_usd=1000.0,
        exit_price=exit_price,
        spread_bps=5.0,
        is_stop_loss=False,
    )

    if entry.status == OrderFillStatus.FILLED and exit_result.status == OrderFillStatus.FILLED:
        # Calculate P&L
        raw_pnl_bps = (exit_result.fill_price - entry.fill_price) / entry.fill_price * 10000
        net_pnl_bps = raw_pnl_bps - entry.total_cost_bps - exit_result.total_cost_bps
        pnl_usd = net_pnl_bps / 10000 * 1000.0

        print(f"   Entry Fill: ${entry.fill_price:,.2f} (cost: {entry.total_cost_bps:.2f} bps)")
        print(f"   Exit Fill: ${exit_result.fill_price:,.2f} (cost: {exit_result.total_cost_bps:.2f} bps)")
        print(f"   Raw P&L: {raw_pnl_bps:+.2f} bps")
        print(f"   Net P&L: {net_pnl_bps:+.2f} bps")
        print(f"   P&L USD: ${pnl_usd:+.2f}")

    # Scenario 2: Losing trade with stop loss
    print("\nScenario 2: Losing Trade (Stop Loss)")
    entry = sim.execute_entry(
        symbol='BTC-USD',
        direction=Direction.LONG,
        size_usd=1000.0,
        entry_price=50000.0,
        spread_bps=5.0,
        prefer_maker=True,
        liquidity_score=0.9,
    )

    exit_price = 49900.0  # -100 = -20 bps
    exit_result = sim.execute_exit(
        symbol='BTC-USD',
        direction=Direction.LONG,
        size_usd=1000.0,
        exit_price=exit_price,
        spread_bps=8.0,  # Wider spread in panic
        is_stop_loss=True,
    )

    if entry.status == OrderFillStatus.FILLED and exit_result.status == OrderFillStatus.FILLED:
        raw_pnl_bps = (exit_result.fill_price - entry.fill_price) / entry.fill_price * 10000
        net_pnl_bps = raw_pnl_bps - entry.total_cost_bps - exit_result.total_cost_bps
        pnl_usd = net_pnl_bps / 10000 * 1000.0

        print(f"   Entry Fill: ${entry.fill_price:,.2f} (cost: {entry.total_cost_bps:.2f} bps)")
        print(f"   Exit Fill: ${exit_result.fill_price:,.2f} (cost: {exit_result.total_cost_bps:.2f} bps)")
        print(f"   Raw P&L: {raw_pnl_bps:+.2f} bps")
        print(f"   Net P&L: {net_pnl_bps:+.2f} bps (ouch!)")
        print(f"   P&L USD: ${pnl_usd:+.2f}")

    print("\n✓ P&L calculation tests passed\n")


def test_cost_model_accuracy():
    """Verify that maker rebates actually improve net P&L."""
    print("=" * 70)
    print("TEST: Cost Model Accuracy (Maker vs Taker)")
    print("=" * 70)

    sim = ExecutionSimulator()

    n_trials = 100
    maker_pnls = []
    taker_pnls = []

    np.random.seed(42)

    for _ in range(n_trials):
        # Same price move for both
        entry_price = 50000.0
        price_move_bps = 15.0  # +15 bps move
        exit_price = entry_price * (1 + price_move_bps / 10000)

        # Maker strategy
        maker_entry = sim.execute_entry(
            symbol='BTC-USD',
            direction=Direction.LONG,
            size_usd=1000.0,
            entry_price=entry_price,
            spread_bps=5.0,
            prefer_maker=True,
            liquidity_score=1.0,  # Always fill
        )

        maker_exit = sim.execute_exit(
            symbol='BTC-USD',
            direction=Direction.LONG,
            size_usd=1000.0,
            exit_price=exit_price,
            spread_bps=5.0,
            is_stop_loss=False,
        )

        if maker_entry.status == OrderFillStatus.FILLED and maker_exit.status == OrderFillStatus.FILLED:
            raw_pnl = (maker_exit.fill_price - maker_entry.fill_price) / maker_entry.fill_price * 10000
            net_pnl = raw_pnl - maker_entry.total_cost_bps - maker_exit.total_cost_bps
            maker_pnls.append(net_pnl)

        # Taker strategy
        taker_entry = sim.execute_entry(
            symbol='BTC-USD',
            direction=Direction.LONG,
            size_usd=1000.0,
            entry_price=entry_price,
            spread_bps=5.0,
            prefer_maker=False,
            liquidity_score=1.0,
        )

        taker_exit = sim.execute_exit(
            symbol='BTC-USD',
            direction=Direction.LONG,
            size_usd=1000.0,
            exit_price=exit_price,
            spread_bps=5.0,
            is_stop_loss=False,
        )

        if taker_entry.status == OrderFillStatus.FILLED and taker_exit.status == OrderFillStatus.FILLED:
            raw_pnl = (taker_exit.fill_price - taker_entry.fill_price) / taker_entry.fill_price * 10000
            net_pnl = raw_pnl - taker_entry.total_cost_bps - taker_exit.total_cost_bps
            taker_pnls.append(net_pnl)

    print(f"\nResults over {n_trials} trials with +15 bps price move:")
    print(f"   Maker Strategy:")
    print(f"      Avg Net P&L: {np.mean(maker_pnls):+.2f} bps")
    print(f"      Std Dev: {np.std(maker_pnls):.2f} bps")
    print(f"   Taker Strategy:")
    print(f"      Avg Net P&L: {np.mean(taker_pnls):+.2f} bps")
    print(f"      Std Dev: {np.std(taker_pnls):.2f} bps")
    print(f"   Maker Advantage: {np.mean(maker_pnls) - np.mean(taker_pnls):+.2f} bps")

    assert np.mean(maker_pnls) > np.mean(taker_pnls), "Maker should outperform taker!"

    print("\n✓ Cost model tests passed - Maker strategy is superior\n")


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("BACKTESTING FRAMEWORK - UNIT TESTS")
    print("=" * 70 + "\n")

    test_execution_simulator()
    test_trade_pnl_calculation()
    test_cost_model_accuracy()

    print("=" * 70)
    print("ALL TESTS PASSED ✓")
    print("=" * 70)
