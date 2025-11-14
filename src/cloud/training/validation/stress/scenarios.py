"""
Stress Test Scenarios

Individual adversarial scenarios for model testing.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class StressScenario(ABC):
    """
    Base class for stress test scenarios

    Each scenario modifies market data to simulate extreme conditions.
    """
    name: str
    description: str
    severity: str  # "moderate", "high", "extreme"

    @abstractmethod
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply stress scenario to data

        Args:
            df: Original market data

        Returns:
            Modified data with stress applied
        """
        pass

    @abstractmethod
    def evaluate(self, baseline_metrics: Dict[str, float], stress_metrics: Dict[str, float]) -> bool:
        """
        Evaluate if model passes this stress test

        Args:
            baseline_metrics: Metrics on normal data
            stress_metrics: Metrics under stress

        Returns:
            True if passed, False if failed
        """
        pass


class FlashCrashScenario(StressScenario):
    """
    Flash Crash: -20% price drop in 5 minutes

    Simulates extreme volatility event like 2010 flash crash.
    """

    def __init__(self, crash_pct: float = 0.20, recovery_pct: float = 0.10):
        super().__init__(
            name="Flash Crash",
            description=f"Price drops {crash_pct*100:.0f}% in 5 minutes, recovers {recovery_pct*100:.0f}%",
            severity="extreme"
        )
        self.crash_pct = crash_pct
        self.recovery_pct = recovery_pct

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply flash crash to random point in data"""
        df = df.copy()

        if len(df) < 10:
            return df

        # Choose random crash point (not at edges)
        crash_idx = np.random.randint(5, len(df) - 5)

        # Apply crash over 5 candles
        for i in range(5):
            idx = crash_idx + i
            crash_factor = 1 - (self.crash_pct * (i + 1) / 5)

            df.loc[idx, 'close'] *= crash_factor
            df.loc[idx, 'low'] = min(df.loc[idx, 'low'], df.loc[idx, 'close'])
            df.loc[idx, 'high'] = max(df.loc[idx, 'high'], df.loc[idx, 'open'])

        # Recovery phase (next 3 candles)
        for i in range(3):
            idx = crash_idx + 5 + i
            if idx >= len(df):
                break

            recovery_factor = 1 + (self.recovery_pct * (i + 1) / 3)
            df.loc[idx, 'close'] *= recovery_factor
            df.loc[idx, 'high'] = max(df.loc[idx, 'high'], df.loc[idx, 'close'])

        return df

    def evaluate(self, baseline_metrics: Dict[str, float], stress_metrics: Dict[str, float]) -> bool:
        """
        Pass if:
        - Sharpe doesn't drop more than 50%
        - Max drawdown stays under 30%
        - Model doesn't blow up (no extreme losses)
        """
        baseline_sharpe = baseline_metrics.get('sharpe_ratio', 0)
        stress_sharpe = stress_metrics.get('sharpe_ratio', 0)

        stress_dd = stress_metrics.get('max_drawdown_pct', 0)

        # Allow Sharpe to drop by up to 50%
        sharpe_ok = stress_sharpe >= baseline_sharpe * 0.5

        # Drawdown must stay under 30%
        dd_ok = stress_dd <= 30

        # Check for extreme losses
        stress_pnl = stress_metrics.get('pnl_bps', 0)
        no_blowup = stress_pnl > -500  # Max 500bps loss

        passed = sharpe_ok and dd_ok and no_blowup

        logger.info(
            "flash_crash_evaluated",
            passed=passed,
            baseline_sharpe=baseline_sharpe,
            stress_sharpe=stress_sharpe,
            stress_dd=stress_dd
        )

        return passed


class StuckPositionScenario(StressScenario):
    """
    Stuck Position: Can't exit for 30 minutes

    Simulates exchange issues or liquidity problems preventing exit.
    """

    def __init__(self, stuck_duration_candles: int = 6):
        super().__init__(
            name="Stuck Position",
            description=f"Cannot exit position for {stuck_duration_candles} candles",
            severity="high"
        )
        self.stuck_duration = stuck_duration_candles

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Simulate forced position holding"""
        # This modifies trade execution, not data
        # Mark certain periods as "no exit allowed"
        df = df.copy()
        df['stuck_position'] = False

        if len(df) > self.stuck_duration * 2:
            stuck_start = np.random.randint(0, len(df) - self.stuck_duration)
            df.loc[stuck_start:stuck_start + self.stuck_duration, 'stuck_position'] = True

        return df

    def evaluate(self, baseline_metrics: Dict[str, float], stress_metrics: Dict[str, float]) -> bool:
        """
        Pass if:
        - Can handle extended holding periods
        - Drawdown doesn't exceed 25%
        """
        stress_dd = stress_metrics.get('max_drawdown_pct', 0)
        stress_pnl = stress_metrics.get('pnl_bps', 0)

        dd_ok = stress_dd <= 25
        pnl_ok = stress_pnl > -300

        return dd_ok and pnl_ok


class PartialFillScenario(StressScenario):
    """
    Partial Fill: Only 30% of order fills, rest cancelled

    Simulates low liquidity or aggressive market orders.
    """

    def __init__(self, fill_ratio: float = 0.3):
        super().__init__(
            name="Partial Fill",
            description=f"Only {fill_ratio*100:.0f}% of orders fill",
            severity="moderate"
        )
        self.fill_ratio = fill_ratio

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Mark partial fill scenario"""
        df = df.copy()
        df['fill_ratio'] = self.fill_ratio
        return df

    def evaluate(self, baseline_metrics: Dict[str, float], stress_metrics: Dict[str, float]) -> bool:
        """
        Pass if:
        - Strategy still profitable with reduced size
        - Win rate doesn't collapse
        """
        baseline_pnl = baseline_metrics.get('pnl_bps', 0)
        stress_pnl = stress_metrics.get('pnl_bps', 0)

        # Expect proportional PnL reduction
        expected_pnl = baseline_pnl * self.fill_ratio

        # Allow some degradation but not total collapse
        pnl_ok = stress_pnl >= expected_pnl * 0.5

        return pnl_ok


class ExchangeHaltScenario(StressScenario):
    """
    Exchange Halt: Trading paused mid-position

    Simulates exchange circuit breakers or technical issues.
    """

    def __init__(self, halt_duration_candles: int = 12):
        super().__init__(
            name="Exchange Halt",
            description=f"Trading halted for {halt_duration_candles} candles",
            severity="high"
        )
        self.halt_duration = halt_duration_candles

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Insert trading halt period"""
        df = df.copy()
        df['trading_halted'] = False

        if len(df) > self.halt_duration * 2:
            halt_start = np.random.randint(0, len(df) - self.halt_duration)

            # Mark halt period
            df.loc[halt_start:halt_start + self.halt_duration, 'trading_halted'] = True

            # Freeze prices during halt
            freeze_price = df.loc[halt_start, 'close']
            for i in range(self.halt_duration):
                idx = halt_start + i
                df.loc[idx, 'open'] = freeze_price
                df.loc[idx, 'high'] = freeze_price
                df.loc[idx, 'low'] = freeze_price
                df.loc[idx, 'close'] = freeze_price
                df.loc[idx, 'volume'] = 0

        return df

    def evaluate(self, baseline_metrics: Dict[str, float], stress_metrics: Dict[str, float]) -> bool:
        """
        Pass if:
        - Model doesn't assume always-on trading
        - Can handle zero-volume periods
        """
        stress_dd = stress_metrics.get('max_drawdown_pct', 0)

        # Drawdown might be larger due to forced holding
        dd_ok = stress_dd <= 30

        return dd_ok


class FundingFlipScenario(StressScenario):
    """
    Funding Flip: Sudden -0.5% → +0.5% funding rate change

    Simulates extreme funding rate volatility in perpetual futures.
    """

    def __init__(self, funding_swing_bps: float = 100):
        super().__init__(
            name="Funding Flip",
            description=f"Funding rate swings {funding_swing_bps}bps",
            severity="moderate"
        )
        self.funding_swing = funding_swing_bps

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply funding rate shock"""
        df = df.copy()
        df['funding_rate_bps'] = 0

        if len(df) > 10:
            flip_idx = len(df) // 2

            # Before flip: negative funding
            df.loc[:flip_idx, 'funding_rate_bps'] = -self.funding_swing / 2

            # After flip: positive funding
            df.loc[flip_idx:, 'funding_rate_bps'] = self.funding_swing / 2

        return df

    def evaluate(self, baseline_metrics: Dict[str, float], stress_metrics: Dict[str, float]) -> bool:
        """
        Pass if:
        - Strategy accounts for funding costs
        - PnL net of funding still positive
        """
        stress_pnl = stress_metrics.get('pnl_bps', 0)

        # Should still be profitable net of funding
        pnl_ok = stress_pnl > 0

        return pnl_ok


class LiquidityEvaporationScenario(StressScenario):
    """
    Liquidity Evaporation: Bid-ask spread 10x normal

    Simulates extreme market stress with no liquidity.
    """

    def __init__(self, spread_multiplier: float = 10.0):
        super().__init__(
            name="Liquidity Evaporation",
            description=f"Spread increases {spread_multiplier}x",
            severity="extreme"
        )
        self.spread_multiplier = spread_multiplier

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Widen spreads dramatically"""
        df = df.copy()

        # Calculate normal spread
        normal_spread_bps = ((df['high'] - df['low']) / df['close']).median() * 10000

        # Apply multiplier
        df['spread_bps'] = normal_spread_bps * self.spread_multiplier

        return df

    def evaluate(self, baseline_metrics: Dict[str, float], stress_metrics: Dict[str, float]) -> bool:
        """
        Pass if:
        - Strategy can handle high spreads
        - Doesn't trade into illiquid conditions
        """
        stress_pnl = stress_metrics.get('pnl_bps', 0)
        stress_trades = stress_metrics.get('trades_oos', 0)

        # Should reduce trading in high spread
        baseline_trades = baseline_metrics.get('trades_oos', 1)
        trades_reduced = stress_trades < baseline_trades * 0.7

        # Should not lose money
        pnl_ok = stress_pnl > -100

        return trades_reduced or pnl_ok


class FeeSpikeScenario(StressScenario):
    """
    Fee Spike: 10x normal trading fees

    Simulates high-fee environment or network congestion.
    """

    def __init__(self, fee_multiplier: float = 10.0):
        super().__init__(
            name="Fee Spike",
            description=f"Trading fees {fee_multiplier}x normal",
            severity="moderate"
        )
        self.fee_multiplier = fee_multiplier

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply high fees"""
        df = df.copy()
        df['fee_multiplier'] = self.fee_multiplier
        return df

    def evaluate(self, baseline_metrics: Dict[str, float], stress_metrics: Dict[str, float]) -> bool:
        """
        Pass if:
        - Strategy remains profitable after high fees
        - Or reduces trading frequency
        """
        stress_pnl = stress_metrics.get('pnl_bps', 0)
        stress_trades = stress_metrics.get('trades_oos', 0)

        baseline_trades = baseline_metrics.get('trades_oos', 1)

        # Either still profitable or trades less
        still_profitable = stress_pnl > 0
        trades_reduced = stress_trades < baseline_trades * 0.5

        return still_profitable or trades_reduced


class DataGapScenario(StressScenario):
    """
    Data Gap: 30-minute missing data period

    Simulates exchange downtime or data feed issues.
    """

    def __init__(self, gap_duration_candles: int = 6):
        super().__init__(
            name="Data Gap",
            description=f"Missing data for {gap_duration_candles} candles",
            severity="moderate"
        )
        self.gap_duration = gap_duration_candles

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove data to create gap"""
        if len(df) <= self.gap_duration * 2:
            return df

        df = df.copy()
        gap_start = len(df) // 2

        # Create gap by setting to NaN
        df.loc[gap_start:gap_start + self.gap_duration, ['open', 'high', 'low', 'close', 'volume']] = np.nan

        return df

    def evaluate(self, baseline_metrics: Dict[str, float], stress_metrics: Dict[str, float]) -> bool:
        """
        Pass if:
        - Model handles missing data gracefully
        - Doesn't crash or generate invalid signals
        """
        # If we got metrics back, model didn't crash
        has_metrics = 'sharpe_ratio' in stress_metrics

        if not has_metrics:
            return False

        # Should still have reasonable performance
        stress_sharpe = stress_metrics.get('sharpe_ratio', 0)

        return stress_sharpe > 0
