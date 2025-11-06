"""
Example: Integrating Dual-Mode Trading into EnhancedRLPipeline

This example shows how to wire the dual-mode system into your existing
training pipeline with minimal code changes.
"""

from datetime import datetime
from typing import Dict

import numpy as np

from src.cloud.training.agents.rl_agent import TradingAction, TradingState
from src.cloud.training.config.settings import EngineSettings
from src.cloud.training.integrations.dual_mode_adapter import DualModeAdapter, DualModeConfig
from src.cloud.training.models.alpha_engines import AlphaEngineCoordinator
from src.cloud.training.models.asset_profiles import TradingMode
from src.cloud.training.pipelines.enhanced_rl_pipeline import EnhancedRLPipeline


class DualModeEnhancedPipeline(EnhancedRLPipeline):
    """
    Extended pipeline with dual-mode trading support.

    This shows how to add dual-mode to your existing pipeline.
    """

    def __init__(
        self,
        settings: EngineSettings,
        dsn: str,
        enable_dual_mode: bool = True,
        dual_mode_capital_gbp: float = 10000.0,
        **kwargs,
    ):
        """
        Initialize with dual-mode support.

        Args:
            settings: Engine settings
            dsn: Database connection string
            enable_dual_mode: Enable dual-mode trading
            dual_mode_capital_gbp: Total capital for dual-mode
            **kwargs: Additional arguments for EnhancedRLPipeline
        """
        # Initialize base pipeline
        super().__init__(settings=settings, dsn=dsn, **kwargs)

        # Add dual-mode adapter
        self.dual_mode_enabled = enable_dual_mode

        if self.dual_mode_enabled:
            dual_mode_config = DualModeConfig(
                enabled=True,
                total_capital_gbp=dual_mode_capital_gbp,
            )

            # Create alpha coordinator if not exists
            alpha_coordinator = getattr(self, 'alpha_coordinator', None)
            if alpha_coordinator is None:
                alpha_coordinator = AlphaEngineCoordinator()
                self.alpha_coordinator = alpha_coordinator

            self.dual_mode_adapter = DualModeAdapter(
                config=dual_mode_config,
                alpha_coordinator=alpha_coordinator,
            )

            print(f"✅ Dual-mode trading enabled (Capital: £{dual_mode_capital_gbp:,.0f})")
        else:
            self.dual_mode_adapter = None
            print("ℹ️  Dual-mode trading disabled")

    def _enhanced_trading_step(
        self,
        symbol: str,
        timestamp: datetime,
        current_price: float,
        features: Dict[str, float],
        regime: str,
        confidence: float,
    ) -> Dict:
        """
        Enhanced trading step with dual-mode support.

        This replaces your normal trading step logic.
        """
        result = {
            "timestamp": timestamp,
            "symbol": symbol,
            "price": current_price,
            "actions": [],
            "mode": None,
        }

        if not self.dual_mode_enabled or not self.dual_mode_adapter:
            # Fall back to single-mode logic
            return result

        # 1. Update existing positions
        actions = self.dual_mode_adapter.update_positions(
            symbol=symbol,
            current_price=current_price,
            features=features,
            regime=regime,
            timestamp=timestamp,
        )

        result["actions"].extend(actions)

        # 2. Check for new entry
        should_enter, mode, reason = self.dual_mode_adapter.evaluate_for_entry(
            symbol=symbol,
            features=features,
            regime=regime,
            confidence=confidence,
            current_price=current_price,
            timestamp=timestamp,
        )

        if should_enter and mode:
            # Determine size based on mode
            if mode == TradingMode.SHORT_HOLD:
                size_gbp = 200.0  # Smaller for scalps
                stop_loss_bps = -10.0
                take_profit_bps = 15.0
            else:  # LONG_HOLD
                size_gbp = 1000.0  # Larger for swings
                stop_loss_bps = -150.0
                take_profit_bps = None  # Will scale out instead

            # Open position
            opened = self.dual_mode_adapter.open_position(
                symbol=symbol,
                mode=mode,
                entry_price=current_price,
                size_gbp=size_gbp,
                stop_loss_bps=stop_loss_bps,
                take_profit_bps=take_profit_bps,
                entry_regime=regime,
                entry_confidence=confidence,
            )

            if opened:
                result["actions"].append(f"OPENED_{mode.value}")
                result["mode"] = mode.value

        return result

    def _get_dual_mode_state(
        self,
        symbol: str,
        market_features: np.ndarray,
        features: Dict[str, float],
    ) -> TradingState:
        """
        Create trading state with dual-mode information.

        This replaces your normal state creation.
        """
        # Create base state (your existing logic)
        base_state = TradingState(
            market_features=market_features,
            similar_pattern_win_rate=0.5,
            similar_pattern_avg_profit=0.0,
            similar_pattern_reliability=0.5,
            has_position=False,
            position_size_multiplier=0.0,
            unrealized_pnl_bps=0.0,
            hold_duration_minutes=0,
            volatility_bps=features.get("volatility_bps", 80.0),
            spread_bps=features.get("spread_bps", 10.0),
            regime_code=self._regime_to_code(str(features.get("regime", "unknown"))),
            current_drawdown_gbp=0.0,
            trades_today=0,
            win_rate_today=0.5,
            recent_return_1m=features.get("return_1m", 0.0),
            recent_return_5m=features.get("return_5m", 0.0),
            recent_return_30m=features.get("return_30m", 0.0),
            volume_zscore=features.get("volume_zscore", 0.0),
            volatility_zscore=features.get("volatility_zscore", 0.0),
            estimated_transaction_cost_bps=features.get("spread_bps", 10.0),
            symbol=symbol,
        )

        # Enhance with dual-mode fields
        if self.dual_mode_enabled and self.dual_mode_adapter:
            enhanced_state = self.dual_mode_adapter.get_trading_state_for_rl(
                symbol=symbol,
                market_features=market_features,
                base_state=base_state,
            )
            return enhanced_state

        return base_state

    def get_dual_mode_statistics(self) -> Dict:
        """Get dual-mode statistics."""
        if not self.dual_mode_enabled or not self.dual_mode_adapter:
            return {"enabled": False}

        return self.dual_mode_adapter.get_statistics()

    @staticmethod
    def _regime_to_code(regime: str) -> int:
        """Convert regime string to code."""
        mapping = {
            "low_vol": 0,
            "medium": 1,
            "high": 2,
            "trend": 3,
            "range": 4,
        }
        return mapping.get(regime.lower(), 1)


def example_usage():
    """Example of how to use the dual-mode pipeline."""
    print("=" * 60)
    print("Dual-Mode Trading Integration Example")
    print("=" * 60)

    # 1. Create settings (you'd load from config)
    from pathlib import Path
    config_dir = Path("config")
    settings = EngineSettings.load(config_dir=config_dir)

    # 2. Create pipeline with dual-mode
    pipeline = DualModeEnhancedPipeline(
        settings=settings,
        dsn="postgresql://user:pass@localhost/db",
        enable_dual_mode=True,
        dual_mode_capital_gbp=10000.0,
    )

    print("\n✅ Pipeline initialized with dual-mode support")

    # 3. Simulate a trading step
    print("\n" + "=" * 60)
    print("Simulating Trading Step")
    print("=" * 60)

    result = pipeline._enhanced_trading_step(
        symbol="ETH",
        timestamp=datetime.now(),
        current_price=2000.0,
        features={
            "micro_score": 65.0,
            "ignition_score": 75.0,
            "trend_strength": 0.7,
            "volatility_bps": 80.0,
            "spread_bps": 8.0,
            "htf_bias": 0.6,
            "eps_net": 0.001,
            "confidence": 0.70,
        },
        regime="trend",
        confidence=0.70,
    )

    print(f"\nResult:")
    print(f"  Symbol: {result['symbol']}")
    print(f"  Price: ${result['price']:.2f}")
    print(f"  Actions: {result['actions']}")
    print(f"  Mode: {result['mode']}")

    # 4. Get statistics
    print("\n" + "=" * 60)
    print("Dual-Mode Statistics")
    print("=" * 60)

    stats = pipeline.get_dual_mode_statistics()

    if stats.get("enabled"):
        print(f"\nShort-Hold Book:")
        print(f"  Positions: {stats['short_hold']['num_positions']}")
        print(f"  Exposure: £{stats['short_hold']['exposure_gbp']:.2f}")
        print(f"  P&L: £{stats['short_hold']['realized_pnl_gbp']:.2f}")

        print(f"\nLong-Hold Book:")
        print(f"  Positions: {stats['long_hold']['num_positions']}")
        print(f"  Exposure: £{stats['long_hold']['exposure_gbp']:.2f}")
        print(f"  P&L: £{stats['long_hold']['realized_pnl_gbp']:.2f}")

        print(f"\nRouting Stats:")
        print(f"  Total signals: {stats['routing']['total_signals']}")
        print(f"  Short routed: {stats['routing']['short_routed']}")
        print(f"  Long routed: {stats['routing']['long_routed']}")

    print("\n" + "=" * 60)
    print("✅ Dual-mode integration example complete!")
    print("=" * 60)


if __name__ == "__main__":
    example_usage()
