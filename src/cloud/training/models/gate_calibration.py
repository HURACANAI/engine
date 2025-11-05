"""
Gate Calibration Framework

Calibrates gate thresholds using historical trading data to achieve target metrics:
- Scalp mode: 70-75% WR, 30-50 trades/day
- Runner mode: 95%+ WR, 2-5 trades/day

Process:
1. Load historical trades with outcomes
2. Extract features and signals
3. Run grid search over gate thresholds
4. Optimize for target WR + volume
5. Select optimal thresholds
6. Validate on out-of-sample data

Usage:
    calibrator = GateCalibrator(
        historical_trades=trades_df,
        target_scalp_wr=0.72,
        target_runner_wr=0.95,
    )

    # Calibrate
    optimal_thresholds = calibrator.calibrate()

    # Apply to gate profiles
    calibrator.apply_thresholds(scalp_profile, runner_profile)
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from itertools import product
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class HistoricalTrade:
    """Historical trade record for calibration."""

    # Signal characteristics
    technique: str
    confidence: float
    regime: str
    edge_hat_bps: float

    # Features at signal time
    features: Dict[str, float]

    # Execution details
    order_type: str  # 'maker' or 'taker'
    spread_bps: float
    liquidity_score: float

    # Outcome
    won: bool
    pnl_bps: float
    hold_time_sec: float

    # Metadata
    timestamp: float
    symbol: str


@dataclass
class ThresholdConfig:
    """Gate threshold configuration."""

    # Cost gate
    cost_buffer_bps: float

    # Meta-label gate
    meta_threshold: float

    # Regret probability
    regret_threshold: float

    # Pattern memory
    pattern_evidence_threshold: float


@dataclass
class CalibrationResult:
    """Result of calibration."""

    # Optimal thresholds
    scalp_thresholds: ThresholdConfig
    runner_thresholds: ThresholdConfig

    # Achieved metrics
    scalp_wr: float
    scalp_volume: int
    runner_wr: float
    runner_volume: int

    # Out-of-sample validation
    oos_scalp_wr: float
    oos_runner_wr: float

    # Grid search details
    total_configs_tested: int
    best_scalp_score: float
    best_runner_score: float


class GateCalibrator:
    """
    Calibrate gate thresholds using historical data.

    Optimization Objective:
    - Scalp mode: Maximize (WR × volume) subject to WR ≥ 0.68
    - Runner mode: Maximize WR subject to volume ≥ 2

    Grid Search:
    - Cost buffer: [2, 3, 4, 5, 8, 10] bps
    - Meta threshold: [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    - Regret threshold: [0.20, 0.30, 0.40, 0.50]
    - Pattern evidence: [-0.3, -0.2, -0.1, 0.0, 0.1]
    """

    def __init__(
        self,
        historical_trades: List[HistoricalTrade],
        target_scalp_wr: float = 0.72,
        target_runner_wr: float = 0.95,
        min_scalp_volume: int = 20,  # Per test period
        min_runner_volume: int = 2,
        train_test_split: float = 0.70,
    ):
        """
        Initialize gate calibrator.

        Args:
            historical_trades: List of historical trades
            target_scalp_wr: Target win rate for scalp mode
            target_runner_wr: Target win rate for runner mode
            min_scalp_volume: Minimum scalp trades per period
            min_runner_volume: Minimum runner trades per period
            train_test_split: Train/test split ratio
        """
        self.trades = historical_trades
        self.target_scalp_wr = target_scalp_wr
        self.target_runner_wr = target_runner_wr
        self.min_scalp_volume = min_scalp_volume
        self.min_runner_volume = min_runner_volume
        self.train_test_split = train_test_split

        # Split train/test
        n_train = int(len(historical_trades) * train_test_split)
        self.train_trades = historical_trades[:n_train]
        self.test_trades = historical_trades[n_train:]

        logger.info(
            "gate_calibrator_initialized",
            total_trades=len(historical_trades),
            train_trades=n_train,
            test_trades=len(self.test_trades),
            target_scalp_wr=target_scalp_wr,
            target_runner_wr=target_runner_wr,
        )

    def calibrate(
        self,
        verbose: bool = True,
    ) -> CalibrationResult:
        """
        Calibrate gate thresholds.

        Args:
            verbose: Print progress

        Returns:
            CalibrationResult with optimal thresholds
        """
        if verbose:
            print("=" * 70)
            print("GATE CALIBRATION")
            print("=" * 70)
            print(f"Training on {len(self.train_trades)} trades")
            print(f"Testing on {len(self.test_trades)} trades")
            print()

        # Define grid
        cost_buffers = [2.0, 3.0, 4.0, 5.0, 8.0, 10.0]
        meta_thresholds = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
        regret_thresholds = [0.20, 0.30, 0.40, 0.50]
        pattern_thresholds = [-0.3, -0.2, -0.1, 0.0, 0.1]

        # Grid search for scalp mode
        if verbose:
            print("Calibrating SCALP mode (loose gates)...")

        best_scalp_config = None
        best_scalp_score = -np.inf

        scalp_configs = list(product(
            cost_buffers[:3],  # Loose: 2-4 bps
            meta_thresholds[:4],  # Loose: 0.40-0.55
            regret_thresholds[2:],  # Permissive: 0.40-0.50
            pattern_thresholds[:3],  # Loose: -0.3 to -0.1
        ))

        for cost, meta, regret, pattern in scalp_configs:
            config = ThresholdConfig(
                cost_buffer_bps=cost,
                meta_threshold=meta,
                regret_threshold=regret,
                pattern_evidence_threshold=pattern,
            )

            # Simulate with this config
            wr, volume = self._simulate_config(config, self.train_trades, mode='scalp')

            # Score: maximize (WR × volume) subject to WR ≥ 0.68
            if wr >= 0.68 and volume >= self.min_scalp_volume:
                score = wr * volume
            else:
                score = -1.0  # Penalty for not meeting constraints

            if score > best_scalp_score:
                best_scalp_score = score
                best_scalp_config = config

        if verbose:
            print(f"  Best scalp config: cost={best_scalp_config.cost_buffer_bps}, "
                  f"meta={best_scalp_config.meta_threshold:.2f}, "
                  f"regret={best_scalp_config.regret_threshold:.2f}")

        # Grid search for runner mode
        if verbose:
            print("\nCalibrating RUNNER mode (strict gates)...")

        best_runner_config = None
        best_runner_score = -np.inf

        runner_configs = list(product(
            cost_buffers[3:],  # Strict: 5-10 bps
            meta_thresholds[4:],  # Strict: 0.60-0.70
            regret_thresholds[:2],  # Strict: 0.20-0.30
            pattern_thresholds[3:],  # Strict: 0.0-0.1
        ))

        for cost, meta, regret, pattern in runner_configs:
            config = ThresholdConfig(
                cost_buffer_bps=cost,
                meta_threshold=meta,
                regret_threshold=regret,
                pattern_evidence_threshold=pattern,
            )

            # Simulate with this config
            wr, volume = self._simulate_config(config, self.train_trades, mode='runner')

            # Score: maximize WR subject to volume ≥ 2
            if volume >= self.min_runner_volume:
                score = wr
            else:
                score = -1.0  # Penalty

            if score > best_runner_score:
                best_runner_score = score
                best_runner_config = config

        if verbose:
            print(f"  Best runner config: cost={best_runner_config.cost_buffer_bps}, "
                  f"meta={best_runner_config.meta_threshold:.2f}, "
                  f"regret={best_runner_config.regret_threshold:.2f}")

        # Validate on test set
        if verbose:
            print("\nValidating on test set...")

        scalp_train_wr, scalp_train_vol = self._simulate_config(
            best_scalp_config, self.train_trades, mode='scalp'
        )
        scalp_test_wr, scalp_test_vol = self._simulate_config(
            best_scalp_config, self.test_trades, mode='scalp'
        )

        runner_train_wr, runner_train_vol = self._simulate_config(
            best_runner_config, self.train_trades, mode='runner'
        )
        runner_test_wr, runner_test_vol = self._simulate_config(
            best_runner_config, self.test_trades, mode='runner'
        )

        if verbose:
            print(f"\nSCALP MODE:")
            print(f"  Train: {scalp_train_wr:.1%} WR, {scalp_train_vol} trades")
            print(f"  Test:  {scalp_test_wr:.1%} WR, {scalp_test_vol} trades")
            print(f"\nRUNNER MODE:")
            print(f"  Train: {runner_train_wr:.1%} WR, {runner_train_vol} trades")
            print(f"  Test:  {runner_test_wr:.1%} WR, {runner_test_vol} trades")

        return CalibrationResult(
            scalp_thresholds=best_scalp_config,
            runner_thresholds=best_runner_config,
            scalp_wr=scalp_train_wr,
            scalp_volume=scalp_train_vol,
            runner_wr=runner_train_wr,
            runner_volume=runner_train_vol,
            oos_scalp_wr=scalp_test_wr,
            oos_runner_wr=runner_test_wr,
            total_configs_tested=len(scalp_configs) + len(runner_configs),
            best_scalp_score=best_scalp_score,
            best_runner_score=best_runner_score,
        )

    def _simulate_config(
        self,
        config: ThresholdConfig,
        trades: List[HistoricalTrade],
        mode: str,
    ) -> Tuple[float, int]:
        """
        Simulate trading with given config.

        Args:
            config: Threshold configuration
            trades: Trades to simulate
            mode: 'scalp' or 'runner'

        Returns:
            (win_rate, volume)
        """
        passed_trades = []

        for trade in trades:
            # Determine if trade matches mode preference
            if mode == 'scalp':
                # Scalp prefers: TAPE, SWEEP, RANGE techniques
                if trade.technique.upper() not in ['TAPE', 'SWEEP', 'RANGE']:
                    continue
            else:  # runner
                # Runner prefers: TREND, BREAKOUT with high confidence
                if trade.technique.upper() not in ['TREND', 'BREAKOUT']:
                    continue
                if trade.confidence < 0.70:
                    continue

            # Apply gates
            passes = True

            # Cost gate
            # Simplified: assume edge_net = edge_hat - 5 bps (avg cost)
            edge_net = trade.edge_hat_bps - 5.0
            if edge_net < config.cost_buffer_bps:
                passes = False

            # Meta-label gate
            # Simplified: use confidence as proxy
            if trade.confidence < config.meta_threshold:
                passes = False

            # If passes all gates, trade executes
            if passes:
                passed_trades.append(trade)

        # Calculate metrics
        if not passed_trades:
            return 0.0, 0

        wins = sum(1 for t in passed_trades if t.won)
        wr = wins / len(passed_trades)
        volume = len(passed_trades)

        return wr, volume

    def apply_thresholds(
        self,
        scalp_profile,
        runner_profile,
        calibration_result: CalibrationResult,
    ) -> None:
        """
        Apply calibrated thresholds to gate profiles.

        Args:
            scalp_profile: ScalpGateProfile instance
            runner_profile: RunnerGateProfile instance
            calibration_result: Calibration result
        """
        # Apply to scalp profile
        scalp_profile.cost_gate.buffer = calibration_result.scalp_thresholds.cost_buffer_bps
        scalp_profile.meta_label.threshold = calibration_result.scalp_thresholds.meta_threshold
        scalp_profile.regret_calc.regret_threshold = calibration_result.scalp_thresholds.regret_threshold
        scalp_profile.pattern_memory.evidence_threshold = calibration_result.scalp_thresholds.pattern_evidence_threshold

        # Apply to runner profile
        runner_profile.cost_gate.buffer = calibration_result.runner_thresholds.cost_buffer_bps
        runner_profile.meta_label.threshold = calibration_result.runner_thresholds.meta_threshold
        runner_profile.regret_calc.regret_threshold = calibration_result.runner_thresholds.regret_threshold
        runner_profile.pattern_memory.evidence_threshold = calibration_result.runner_thresholds.pattern_evidence_threshold

        logger.info(
            "gate_thresholds_applied",
            scalp_cost=calibration_result.scalp_thresholds.cost_buffer_bps,
            scalp_meta=calibration_result.scalp_thresholds.meta_threshold,
            runner_cost=calibration_result.runner_thresholds.cost_buffer_bps,
            runner_meta=calibration_result.runner_thresholds.meta_threshold,
        )


def generate_synthetic_trades(
    n_trades: int = 1000,
    scalp_ratio: float = 0.70,
) -> List[HistoricalTrade]:
    """
    Generate synthetic historical trades for testing.

    Args:
        n_trades: Number of trades to generate
        scalp_ratio: Ratio of scalp-type trades

    Returns:
        List of synthetic trades
    """
    np.random.seed(42)

    trades = []
    techniques_scalp = ['TAPE', 'SWEEP', 'RANGE']
    techniques_runner = ['TREND', 'BREAKOUT']
    regimes = ['TREND', 'RANGE', 'PANIC']

    for i in range(n_trades):
        # Determine mode
        is_scalp = np.random.random() < scalp_ratio

        if is_scalp:
            technique = np.random.choice(techniques_scalp)
            confidence = np.random.uniform(0.55, 0.75)
            edge_hat = np.random.uniform(8, 15)
            base_wr = 0.72  # Target 72% for scalps
        else:
            technique = np.random.choice(techniques_runner)
            confidence = np.random.uniform(0.70, 0.90)
            edge_hat = np.random.uniform(12, 25)
            base_wr = 0.88  # Target 88% for runners (realistic, not 95%)

        # Win/loss based on confidence
        wr_adjusted = base_wr + (confidence - 0.65) * 0.3
        won = np.random.random() < wr_adjusted

        # P&L
        if won:
            if is_scalp:
                pnl_bps = np.random.uniform(80, 150)  # £0.80-£1.50 on £100
            else:
                pnl_bps = np.random.uniform(500, 1200)  # £5-£12 on £100
        else:
            if is_scalp:
                pnl_bps = -np.random.uniform(30, 80)  # Small losses
            else:
                pnl_bps = -np.random.uniform(100, 300)  # Larger losses

        trade = HistoricalTrade(
            technique=technique,
            confidence=confidence,
            regime=np.random.choice(regimes),
            edge_hat_bps=edge_hat,
            features={'engine_conf': confidence},
            order_type='maker',
            spread_bps=np.random.uniform(6, 10),
            liquidity_score=np.random.uniform(0.65, 0.85),
            won=won,
            pnl_bps=pnl_bps,
            hold_time_sec=np.random.uniform(5, 30) if is_scalp else np.random.uniform(60, 600),
            timestamp=1699000000.0 + i * 60,
            symbol='ETH-USD',
        )

        trades.append(trade)

    return trades


def run_calibration_example():
    """Example usage of gate calibrator."""
    print("Generating synthetic historical trades...")
    trades = generate_synthetic_trades(n_trades=1000, scalp_ratio=0.70)

    print(f"Generated {len(trades)} trades")
    print(f"Overall WR: {sum(1 for t in trades if t.won) / len(trades):.1%}")
    print()

    # Calibrate
    calibrator = GateCalibrator(
        historical_trades=trades,
        target_scalp_wr=0.72,
        target_runner_wr=0.95,
        min_scalp_volume=20,
        min_runner_volume=2,
    )

    result = calibrator.calibrate(verbose=True)

    print("\n" + "=" * 70)
    print("CALIBRATION COMPLETE")
    print("=" * 70)
    print(f"\nConfigs tested: {result.total_configs_tested}")
    print(f"\nSCALP mode achieved: {result.scalp_wr:.1%} WR with {result.scalp_volume} trades")
    print(f"  OOS validation: {result.oos_scalp_wr:.1%} WR")
    print(f"\nRUNNER mode achieved: {result.runner_wr:.1%} WR with {result.runner_volume} trades")
    print(f"  OOS validation: {result.oos_runner_wr:.1%} WR")


if __name__ == '__main__':
    run_calibration_example()
