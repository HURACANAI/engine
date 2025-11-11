"""
[ENGINE - USED FOR VALIDATION]

Backtesting Framework - Historical Simulation of Dual-Mode Trading

This module is USED by the Engine for validating models on historical data.
However, it uses trading_coordinator.py which is marked as FUTURE/PILOT.

NOTE: This framework validates models for Engine, but uses Pilot components
(trading_coordinator) for signal processing. This is acceptable for validation.

Purpose:
- Simulate the full dual-mode trading system on historical data
- Generate realistic P&L with proper execution costs
- Validate gate calibration and model performance
- Produce comprehensive performance reports

Key Features:
1. Realistic Execution Simulation
   - Market impact and slippage
   - Maker vs taker fee differentiation
   - Order fill probability based on liquidity
   - Partial fills in low liquidity

2. Comprehensive Metrics
   - P&L and Sharpe ratio
   - Win rate by mode (scalp vs runner)
   - Maximum drawdown
   - Risk-adjusted returns
   - Trade distribution analysis

3. Out-of-Sample Testing
   - Train/test split to avoid overfitting
   - Walk-forward analysis
   - Regime-specific performance

4. Integration with All Components
   - trading_coordinator.py for signal processing (FUTURE/PILOT component)
   - gate_profiles.py for tiered filtering
   - dual_book_manager.py for position management
   - conformal_gating.py for uncertainty quantification
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)

from .dual_book_manager import DualBookManager, BookType
from .trading_coordinator import TradingCoordinator


class Direction(Enum):
    """Trade direction."""
    LONG = "long"
    SHORT = "short"


class OrderFillStatus(Enum):
    """Order fill outcomes."""
    FILLED = "filled"
    PARTIAL = "partial"
    TIMEOUT = "timeout"
    REJECTED = "rejected"


@dataclass
class ExecutionResult:
    """Result of order execution simulation."""

    status: OrderFillStatus
    fill_price: Optional[float] = None
    fill_size: float = 0.0
    slippage_bps: float = 0.0
    total_cost_bps: float = 0.0
    fill_time_sec: float = 0.0
    reason: str = ""


@dataclass
class BacktestTrade:
    """A single trade in the backtest."""

    trade_id: int
    timestamp: float
    symbol: str
    book: Optional[BookType]
    direction: Direction
    entry_price: float
    exit_price: float
    size: float
    entry_cost_bps: float
    exit_cost_bps: float
    hold_time_sec: float

    pnl_bps: float
    pnl_usd: float
    won: bool

    # Features at entry
    confidence: float
    technique: str
    regime: str
    edge_hat_bps: float
    gates_passed: int
    gates_blocked: int
    edge_net_bps: float
    win_probability: float


@dataclass
class BacktestResults:
    """Complete backtest results."""

    # Performance metrics
    total_pnl_usd: float
    total_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float

    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    overall_win_rate: float

    # Mode-specific
    scalp_trades: int
    scalp_win_rate: float
    scalp_pnl_usd: float

    runner_trades: int
    runner_win_rate: float
    runner_pnl_usd: float

    # Risk metrics
    avg_trade_pnl_usd: float
    avg_winner_usd: float
    avg_loser_usd: float
    win_loss_ratio: float

    profit_factor: float

    # Time-based
    total_days: float
    trades_per_day: float
    pnl_per_day_usd: float

    # Detailed trades
    trades: List[BacktestTrade] = field(default_factory=list)

    # Equity curve
    equity_curve: List[float] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)


class ExecutionSimulator:
    """
    Realistic execution simulation with market impact and slippage.

    Simulates:
    - Maker orders: Better price, negative fees, but fill probability < 100%
    - Taker orders: Immediate fill, pay fees, slippage
    - Market impact: Price moves against you based on order size
    - Liquidity: Fill rate depends on market depth
    """

    def __init__(
        self,
        maker_fill_probability: float = 0.75,
        maker_rebate_bps: float = 2.0,
        taker_fee_bps: float = 5.0,
        base_slippage_bps: float = 1.0,
        impact_coefficient: float = 0.0001,  # Impact per $1000 size
    ):
        self.maker_fill_prob = maker_fill_probability
        self.maker_rebate = maker_rebate_bps
        self.taker_fee = taker_fee_bps
        self.base_slippage = base_slippage_bps
        self.impact_coef = impact_coefficient

    def execute_entry(
        self,
        symbol: str,
        direction: Direction,
        size_usd: float,
        entry_price: float,
        spread_bps: float,
        prefer_maker: bool = True,
        liquidity_score: float = 0.5,
    ) -> ExecutionResult:
        """
        Simulate order execution for entry.

        Args:
            symbol: Asset symbol
            direction: LONG or SHORT
            size_usd: Size in USD
            entry_price: Intended entry price
            spread_bps: Current spread in bps
            prefer_maker: Try maker order first
            liquidity_score: 0-1, affects fill probability

        Returns:
            ExecutionResult with fill details
        """
        # Try maker first if preferred
        if prefer_maker:
            # Maker order fill probability depends on liquidity
            adjusted_fill_prob = self.maker_fill_prob * (0.5 + 0.5 * liquidity_score)

            if np.random.random() < adjusted_fill_prob:
                # Maker filled - get better price
                if direction == Direction.LONG:
                    fill_price = entry_price * (1 - spread_bps / 20000)  # Buy at bid
                else:
                    fill_price = entry_price * (1 + spread_bps / 20000)  # Sell at ask

                # Negative cost = rebate
                cost_bps = -self.maker_rebate

                # Fill time for maker (waiting in book)
                fill_time_sec = np.random.exponential(2.0)  # Avg 2 seconds

                return ExecutionResult(
                    status=OrderFillStatus.FILLED,
                    fill_price=fill_price,
                    fill_size=size_usd,
                    slippage_bps=abs((fill_price - entry_price) / entry_price * 10000),
                    total_cost_bps=cost_bps,
                    fill_time_sec=fill_time_sec,
                    reason="Maker order filled with rebate",
                )
            else:
                # Maker timeout, fall back to taker
                pass

        # Taker order (crosses spread)
        if direction == Direction.LONG:
            fill_price = entry_price * (1 + spread_bps / 20000)  # Buy at ask
        else:
            fill_price = entry_price * (1 - spread_bps / 20000)  # Sell at bid

        # Market impact
        impact_bps = self.impact_coef * (size_usd / 1000)
        if direction == Direction.LONG:
            fill_price *= (1 + impact_bps / 10000)
        else:
            fill_price *= (1 - impact_bps / 10000)

        # Slippage
        slippage_bps = self.base_slippage + impact_bps

        # Total cost
        cost_bps = self.taker_fee + slippage_bps

        # Instant fill
        fill_time_sec = 0.1

        return ExecutionResult(
            status=OrderFillStatus.FILLED,
            fill_price=fill_price,
            fill_size=size_usd,
            slippage_bps=slippage_bps,
            total_cost_bps=cost_bps,
            fill_time_sec=fill_time_sec,
            reason="Taker order filled immediately",
        )

    def execute_exit(
        self,
        symbol: str,
        direction: Direction,
        size_usd: float,
        exit_price: float,
        spread_bps: float,
        is_stop_loss: bool = False,
    ) -> ExecutionResult:
        """
        Simulate order execution for exit.

        Args:
            symbol: Asset symbol
            direction: LONG or SHORT (of original position)
            size_usd: Size in USD
            exit_price: Intended exit price
            spread_bps: Current spread
            is_stop_loss: If True, uses taker (urgent exit)

        Returns:
            ExecutionResult with fill details
        """
        # Stop losses always use taker
        if is_stop_loss:
            # Worse slippage on stop loss
            slippage_bps = self.base_slippage * 2.0
            impact_bps = self.impact_coef * (size_usd / 1000)

            if direction == Direction.LONG:
                # Selling to exit long
                fill_price = exit_price * (1 - (spread_bps + slippage_bps) / 10000)
            else:
                # Buying to exit short
                fill_price = exit_price * (1 + (spread_bps + slippage_bps) / 10000)

            cost_bps = self.taker_fee + slippage_bps

            return ExecutionResult(
                status=OrderFillStatus.FILLED,
                fill_price=fill_price,
                fill_size=size_usd,
                slippage_bps=slippage_bps,
                total_cost_bps=cost_bps,
                fill_time_sec=0.1,
                reason="Stop loss executed as taker",
            )

        # Normal exit - try maker first
        if np.random.random() < self.maker_fill_prob:
            if direction == Direction.LONG:
                # Selling to exit long - sell at ask
                fill_price = exit_price * (1 + spread_bps / 20000)
            else:
                # Buying to exit short - buy at bid
                fill_price = exit_price * (1 - spread_bps / 20000)

            cost_bps = -self.maker_rebate
            fill_time_sec = np.random.exponential(2.0)

            return ExecutionResult(
                status=OrderFillStatus.FILLED,
                fill_price=fill_price,
                fill_size=size_usd,
                slippage_bps=abs((fill_price - exit_price) / exit_price * 10000),
                total_cost_bps=cost_bps,
                fill_time_sec=fill_time_sec,
                reason="Exit maker order filled",
            )

        # Fallback to taker
        if direction == Direction.LONG:
            fill_price = exit_price * (1 - spread_bps / 10000)
        else:
            fill_price = exit_price * (1 + spread_bps / 10000)

        cost_bps = self.taker_fee + self.base_slippage

        return ExecutionResult(
            status=OrderFillStatus.FILLED,
            fill_price=fill_price,
            fill_size=size_usd,
            slippage_bps=self.base_slippage,
            total_cost_bps=cost_bps,
            fill_time_sec=0.1,
            reason="Exit taker order filled",
        )


class Backtester:
    """
    Main backtesting engine.

    Architecture:
        Historical Data → TradingCoordinator → ExecutionSimulator → BacktestResults
                              ↓
        Signal Processing, Gate Filtering, Mode Selection
    """

    def __init__(
        self,
        initial_capital: float = 100_000,
        train_ratio: float = 0.70,
        position_size_pct: float = 0.02,  # 2% of capital per trade
        max_positions: int = 5,
        enable_conformal: bool = True,
        enable_governor: bool = False,  # Disabled for backtest (fixed thresholds)
    ):
        """
        Initialize backtester.

        Args:
            initial_capital: Starting capital in USD
            train_ratio: Fraction of data for training (rest for testing)
            position_size_pct: Position size as % of capital
            max_positions: Max concurrent positions
            enable_conformal: Use conformal prediction gates
            enable_governor: Use win-rate governor (not recommended for backtest)
        """
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.train_ratio = train_ratio
        self.position_size_pct = position_size_pct
        self.max_positions = max_positions
        self.enable_conformal = enable_conformal
        self.enable_governor = enable_governor

        # Initialize components
        self.coordinator = TradingCoordinator()
        self.execution_sim = ExecutionSimulator()

        # Backtest state
        self.trades: List[BacktestTrade] = []
        self.equity_curve: List[float] = [initial_capital]
        self.timestamps: List[float] = []

        logger.info(
            "backtester_initialized",
            initial_capital=initial_capital,
            train_ratio=train_ratio,
            position_size_pct=position_size_pct,
        )

    def run(
        self,
        historical_data: pd.DataFrame,
        verbose: bool = True,
    ) -> BacktestResults:
        """
        Run backtest on historical data.

        Args:
            historical_data: DataFrame with columns:
                - timestamp (float)
                - symbol (str)
                - price (float)
                - features (dict or JSON)
                - regime (str)
                - actual_outcome_bps (float) - realized P&L
                - hold_time_sec (float)
                - spread_bps (float)
                - liquidity_score (float)
            verbose: Print progress

        Returns:
            BacktestResults
        """
        if verbose:
            logger.info("=" * 70)
            logger.info("DUAL-MODE BACKTEST")
            logger.info("=" * 70)
            logger.info("backtest_parameters",
                       initial_capital=self.initial_capital,
                       total_samples=len(historical_data),
                       train_samples=int(len(historical_data) * self.train_ratio),
                       test_samples=len(historical_data) - int(len(historical_data) * self.train_ratio))

        # Split into train/test
        split_idx = int(len(historical_data) * self.train_ratio)
        train_data = historical_data.iloc[:split_idx]
        test_data = historical_data.iloc[split_idx:]

        if verbose:
            logger.info("backtest_phase", phase=1, description="Training (calibrating gates)")

        # Calibrate on training data (if conformal enabled)
        if self.enable_conformal:
            self._calibrate_gates(train_data)

        if verbose:
            logger.info("backtest_phase", phase=2, description="Testing", sample_count=len(test_data))

        # Run backtest on test data
        for idx, row in test_data.iterrows():
            self._process_signal(row, verbose=False)

        # Generate results
        results = self._generate_results()

        if verbose:
            self._print_results(results)

        return results

    def _calibrate_gates(self, train_data: pd.DataFrame) -> None:
        """Calibrate conformal prediction on training data."""
        # TODO: Implement proper calibration
        # For now, using pre-calibrated thresholds from gate_calibration.py
        logger.info("gate_calibration_skipped", reason="Using pre-calibrated thresholds")

    def _process_signal(self, row: pd.Series, verbose: bool = False) -> None:
        """Process a single signal through the full pipeline."""
        # Extract signal data
        timestamp = row['timestamp']
        symbol = row['symbol']
        price = row['price']
        features = row['features'] if isinstance(row['features'], dict) else {}
        regime = row['regime']
        actual_outcome_bps = row.get('actual_outcome_bps', 0.0)
        hold_time_sec = row.get('hold_time_sec', 30.0)
        spread_bps = row.get('spread_bps', 5.0)
        liquidity_score = row.get('liquidity_score', 0.5)

        # Get technique and confidence from features
        technique = features.get('technique', 'TREND')
        confidence = features.get('confidence', 0.5)

        # Process through coordinator
        decision = self.coordinator.process_signal(
            symbol=symbol,
            price=price,
            features=features,
            regime=regime,
        )

        if decision is None or not decision.approved or decision.book is None:
            return

        # Determine direction (for simulation, alternate or use feature)
        direction = Direction.LONG if features.get('signal', 1.0) > 0 else Direction.SHORT

        # Calculate position size
        size_usd = self.capital * self.position_size_pct

        # Simulate entry execution
        entry_result = self.execution_sim.execute_entry(
            symbol=symbol,
            direction=direction,
            size_usd=size_usd,
            entry_price=price,
            spread_bps=spread_bps,
            prefer_maker=True,
            liquidity_score=liquidity_score,
        )

        if entry_result.status != OrderFillStatus.FILLED:
            return

        if entry_result.fill_price is None:
            return

        # Simulate exit (use actual outcome)
        exit_price = entry_result.fill_price * (1 + actual_outcome_bps / 10000)
        if direction == Direction.SHORT:
            exit_price = entry_result.fill_price * (1 - actual_outcome_bps / 10000)

        exit_result = self.execution_sim.execute_exit(
            symbol=symbol,
            direction=direction,
            size_usd=size_usd,
            exit_price=exit_price,
            spread_bps=spread_bps,
            is_stop_loss=(actual_outcome_bps < -20),
        )

        if exit_result.fill_price is None:
            return

        # Calculate P&L
        if direction == Direction.LONG:
            pnl_bps = (exit_result.fill_price - entry_result.fill_price) / entry_result.fill_price * 10000
        else:
            pnl_bps = (entry_result.fill_price - exit_result.fill_price) / entry_result.fill_price * 10000

        # Subtract costs
        pnl_bps -= (entry_result.total_cost_bps + exit_result.total_cost_bps)

        pnl_usd = pnl_bps / 10000 * size_usd

        # Update capital
        self.capital += pnl_usd

        # Record trade
        trade = BacktestTrade(
            trade_id=len(self.trades) + 1,
            timestamp=timestamp,
            symbol=symbol,
            book=decision.book,
            direction=direction,
            entry_price=entry_result.fill_price,
            exit_price=exit_result.fill_price,
            size=size_usd,
            entry_cost_bps=entry_result.total_cost_bps,
            exit_cost_bps=exit_result.total_cost_bps,
            hold_time_sec=hold_time_sec,
            pnl_bps=pnl_bps,
            pnl_usd=pnl_usd,
            won=(pnl_bps > 0),
            confidence=confidence,
            technique=technique,
            regime=regime,
            edge_hat_bps=decision.edge_hat_bps,
            gates_passed=decision.gates_passed,
            gates_blocked=decision.gates_blocked,
            edge_net_bps=decision.edge_net_bps,
            win_probability=decision.win_probability,
        )

        self.trades.append(trade)
        self.equity_curve.append(self.capital)
        self.timestamps.append(timestamp)

    def _generate_results(self) -> BacktestResults:
        """Generate comprehensive backtest results."""
        if not self.trades:
            return BacktestResults(
                total_pnl_usd=0.0,
                total_return_pct=0.0,
                sharpe_ratio=0.0,
                max_drawdown_pct=0.0,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                overall_win_rate=0.0,
                scalp_trades=0,
                scalp_win_rate=0.0,
                scalp_pnl_usd=0.0,
                runner_trades=0,
                runner_win_rate=0.0,
                runner_pnl_usd=0.0,
                avg_trade_pnl_usd=0.0,
                avg_winner_usd=0.0,
                avg_loser_usd=0.0,
                win_loss_ratio=0.0,
                profit_factor=0.0,
                total_days=0.0,
                trades_per_day=0.0,
                pnl_per_day_usd=0.0,
            )

        # Overall metrics
        total_pnl = float(sum(t.pnl_usd for t in self.trades))
        total_return = float((self.capital - self.initial_capital) / self.initial_capital * 100)

        winning_trades = [t for t in self.trades if t.won]
        losing_trades = [t for t in self.trades if not t.won]

        win_rate = len(winning_trades) / len(self.trades)

        # Mode-specific
        scalp_trades = [t for t in self.trades if t.book == BookType.SHORT_HOLD]
        runner_trades = [t for t in self.trades if t.book == BookType.LONG_HOLD]

        scalp_wr = len([t for t in scalp_trades if t.won]) / len(scalp_trades) if scalp_trades else 0.0
        runner_wr = len([t for t in runner_trades if t.won]) / len(runner_trades) if runner_trades else 0.0

        scalp_pnl = float(sum(t.pnl_usd for t in scalp_trades))
        runner_pnl = float(sum(t.pnl_usd for t in runner_trades))

        # Risk metrics
        avg_winner = float(np.mean([t.pnl_usd for t in winning_trades])) if winning_trades else 0.0
        avg_loser = float(np.mean([t.pnl_usd for t in losing_trades])) if losing_trades else 0.0
        win_loss_ratio = abs(avg_winner / avg_loser) if avg_loser != 0 else 0.0

        gross_profit = float(sum(t.pnl_usd for t in winning_trades))
        gross_loss = float(abs(sum(t.pnl_usd for t in losing_trades)))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

        # Sharpe ratio
        returns = [t.pnl_usd / self.initial_capital for t in self.trades]
        sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(252)) if len(returns) > 1 else 0.0

        # Max drawdown
        equity = np.array(self.equity_curve)
        running_max = np.maximum.accumulate(equity)
        drawdown = (equity - running_max) / running_max * 100
        max_dd = float(abs(np.min(drawdown)))

        # Time-based
        if self.timestamps:
            total_days = (self.timestamps[-1] - self.timestamps[0]) / 86400
            if total_days <= 0:
                total_days = 1.0
        else:
            total_days = 1.0

        trades_per_day = len(self.trades) / total_days
        pnl_per_day = total_pnl / total_days

        return BacktestResults(
            total_pnl_usd=total_pnl,
            total_return_pct=total_return,
            sharpe_ratio=sharpe,
            max_drawdown_pct=max_dd,
            total_trades=len(self.trades),
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            overall_win_rate=win_rate,
            scalp_trades=len(scalp_trades),
            scalp_win_rate=scalp_wr,
            scalp_pnl_usd=scalp_pnl,
            runner_trades=len(runner_trades),
            runner_win_rate=runner_wr,
            runner_pnl_usd=runner_pnl,
            avg_trade_pnl_usd=float(total_pnl / len(self.trades)),
            avg_winner_usd=avg_winner,
            avg_loser_usd=avg_loser,
            win_loss_ratio=win_loss_ratio,
            profit_factor=profit_factor,
            total_days=total_days,
            trades_per_day=trades_per_day,
            pnl_per_day_usd=pnl_per_day,
            trades=self.trades,
            equity_curve=self.equity_curve,
            timestamps=self.timestamps,
        )

    def _print_results(self, results: BacktestResults) -> None:
        """Print comprehensive results using structured logging."""
        logger.info("=" * 70)
        logger.info("BACKTEST RESULTS")
        logger.info("=" * 70)

        logger.info("overall_performance",
                   total_pnl_usd=results.total_pnl_usd,
                   total_return_pct=results.total_return_pct,
                   sharpe_ratio=results.sharpe_ratio,
                   max_drawdown_pct=results.max_drawdown_pct)

        logger.info("trade_statistics",
                   total_trades=results.total_trades,
                   winning_trades=results.winning_trades,
                   losing_trades=results.losing_trades,
                   overall_win_rate=results.overall_win_rate,
                   avg_trade_pnl_usd=results.avg_trade_pnl_usd,
                   avg_winner_usd=results.avg_winner_usd,
                   avg_loser_usd=results.avg_loser_usd,
                   win_loss_ratio=results.win_loss_ratio,
                   profit_factor=results.profit_factor)

        logger.info("mode_specific_performance",
                   scalp_trades=results.scalp_trades,
                   scalp_win_rate=results.scalp_win_rate,
                   scalp_pnl_usd=results.scalp_pnl_usd,
                   runner_trades=results.runner_trades,
                   runner_win_rate=results.runner_win_rate,
                   runner_pnl_usd=results.runner_pnl_usd)

        logger.info("time_based_metrics",
                   total_days=results.total_days,
                   trades_per_day=results.trades_per_day,
                   pnl_per_day_usd=results.pnl_per_day_usd)

    def export_results(self, results: BacktestResults, path: str) -> None:
        """Export results to CSV."""
        trades_df = pd.DataFrame(
            [
                {
                    'trade_id': t.trade_id,
                    'timestamp': t.timestamp,
                    'symbol': t.symbol,
                    'book': t.book.value if t.book else None,
                    'direction': t.direction.value,
                    'entry_price': t.entry_price,
                    'exit_price': t.exit_price,
                    'size': t.size,
                    'pnl_bps': t.pnl_bps,
                    'pnl_usd': t.pnl_usd,
                    'won': t.won,
                    'confidence': t.confidence,
                    'technique': t.technique,
                    'regime': t.regime,
                    'edge_hat_bps': t.edge_hat_bps,
                }
                for t in results.trades
            ]
        )

        trades_df.to_csv(path, index=False)
        logger.info("backtest_results_exported", path=path, trades=len(results.trades))


def run_synthetic_backtest():
    """Run backtest on synthetic data for demonstration."""
    # Generate synthetic historical data
    np.random.seed(42)

    n_samples = 500
    timestamps = np.arange(n_samples) * 60  # 1 minute apart

    data = []
    for i, ts in enumerate(timestamps):
        # Random regime
        regime = np.random.choice(['TREND', 'RANGE', 'PANIC'], p=[0.50, 0.40, 0.10])

        # Random technique
        technique = np.random.choice(['TREND', 'RANGE', 'BREAKOUT', 'TAPE', 'LEADER', 'SWEEP'])

        # Confidence
        confidence = np.random.beta(5, 2)

        # Actual outcome (regime-dependent)
        if regime == 'TREND' and confidence > 0.65:
            actual_outcome = np.random.normal(15, 8)
        elif regime == 'RANGE' and technique in ['TAPE', 'SWEEP']:
            actual_outcome = np.random.normal(8, 5)
        else:
            actual_outcome = np.random.normal(5, 12)

        data.append({
            'timestamp': ts,
            'symbol': 'BTC-USD',
            'price': 50000 + np.random.normal(0, 100),
            'features': {
                'technique': technique,
                'confidence': confidence,
                'signal': 1.0,
            },
            'regime': regime,
            'actual_outcome_bps': actual_outcome,
            'hold_time_sec': 30 if technique in ['TAPE', 'SWEEP'] else 180,
            'spread_bps': np.random.uniform(3, 8),
            'liquidity_score': np.random.beta(3, 2),
        })

    df = pd.DataFrame(data)

    # Run backtest
    backtester = Backtester(
        initial_capital=100_000,
        train_ratio=0.70,
    )

    results = backtester.run(df, verbose=True)

    # Export
    backtester.export_results(results, 'backtest_results.csv')
    logger.info("results_exported", path="backtest_results.csv")


if __name__ == '__main__':
    run_synthetic_backtest()
