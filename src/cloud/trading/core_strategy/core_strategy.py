"""
CoreBook Strategy

Handles the top three coins (BTC, ETH, SOL) with separate CoreBook logic.
Never sells at a loss, DCA on drops, partial sells on profit thresholds.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import structlog

logger = structlog.get_logger(__name__)


class ActionType(Enum):
    """Action type."""
    BUY = "buy"
    SELL = "sell"
    DCA_BUY = "dca_buy"
    PARTIAL_SELL = "partial_sell"
    NO_ACTION = "no_action"


class MarketRegime(Enum):
    """Market regime."""
    NORMAL = "normal"
    TREND = "trend"
    RANGE = "range"
    PANIC = "panic"
    ILLIQUID = "illiquid"


@dataclass
class CoreBookEntry:
    """Core book entry for a coin."""
    symbol: str
    units_held: float = 0.0
    average_cost_price: float = 0.0
    total_cost_basis: float = 0.0  # Total cost in base currency
    next_dca_trigger_price: Optional[float] = None
    partial_sell_target_price: Optional[float] = None
    total_exposure_limit_pct: float = 10.0  # Max exposure as % of portfolio
    last_action_timestamp: Optional[datetime] = None
    last_action_type: Optional[str] = None
    cooldown_until: Optional[datetime] = None
    dca_count: int = 0
    max_dca_buys: int = 10
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "units_held": self.units_held,
            "average_cost_price": self.average_cost_price,
            "total_cost_basis": self.total_cost_basis,
            "next_dca_trigger_price": self.next_dca_trigger_price,
            "partial_sell_target_price": self.partial_sell_target_price,
            "total_exposure_limit_pct": self.total_exposure_limit_pct,
            "last_action_timestamp": self.last_action_timestamp.isoformat() if self.last_action_timestamp else None,
            "last_action_type": self.last_action_type,
            "cooldown_until": self.cooldown_until.isoformat() if self.cooldown_until else None,
            "dca_count": self.dca_count,
            "max_dca_buys": self.max_dca_buys,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CoreBookEntry:
        """Create from dictionary."""
        return cls(
            symbol=data["symbol"],
            units_held=data.get("units_held", 0.0),
            average_cost_price=data.get("average_cost_price", 0.0),
            total_cost_basis=data.get("total_cost_basis", 0.0),
            next_dca_trigger_price=data.get("next_dca_trigger_price"),
            partial_sell_target_price=data.get("partial_sell_target_price"),
            total_exposure_limit_pct=data.get("total_exposure_limit_pct", 10.0),
            last_action_timestamp=datetime.fromisoformat(data["last_action_timestamp"]) if data.get("last_action_timestamp") else None,
            last_action_type=data.get("last_action_type"),
            cooldown_until=datetime.fromisoformat(data["cooldown_until"]) if data.get("cooldown_until") else None,
            dca_count=data.get("dca_count", 0),
            max_dca_buys=data.get("max_dca_buys", 10),
            metadata=data.get("metadata", {}),
        )
    
    def update_average_cost(self, new_units: float, new_price: float) -> None:
        """Update average cost after a buy.
        
        Args:
            new_units: New units to add
            new_price: Price per unit
        """
        if self.units_held == 0.0:
            self.average_cost_price = new_price
            self.units_held = new_units
            self.total_cost_basis = new_units * new_price
        else:
            total_units = self.units_held + new_units
            total_cost = self.total_cost_basis + (new_units * new_price)
            self.average_cost_price = total_cost / total_units if total_units > 0 else 0.0
            self.units_held = total_units
            self.total_cost_basis = total_cost
    
    def calculate_unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized P&L.
        
        Args:
            current_price: Current price per unit
            
        Returns:
            Unrealized P&L in base currency
        """
        if self.units_held == 0.0:
            return 0.0
        return (current_price - self.average_cost_price) * self.units_held
    
    def calculate_unrealized_pnl_pct(self, current_price: float) -> float:
        """Calculate unrealized P&L percentage.
        
        Args:
            current_price: Current price per unit
            
        Returns:
            Unrealized P&L percentage
        """
        if self.average_cost_price == 0.0:
            return 0.0
        return ((current_price - self.average_cost_price) / self.average_cost_price) * 100.0


@dataclass
class CoreBookState:
    """Core book state for all coins."""
    coins: Dict[str, CoreBookEntry] = field(default_factory=dict)
    auto_trading_enabled: bool = True
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "coins": {symbol: entry.to_dict() for symbol, entry in self.coins.items()},
            "auto_trading_enabled": self.auto_trading_enabled,
            "last_updated": self.last_updated.isoformat(),
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CoreBookState:
        """Create from dictionary."""
        coins = {
            symbol: CoreBookEntry.from_dict(entry_data)
            for symbol, entry_data in data.get("coins", {}).items()
        }
        return cls(
            coins=coins,
            auto_trading_enabled=data.get("auto_trading_enabled", True),
            last_updated=datetime.fromisoformat(data["last_updated"]) if data.get("last_updated") else datetime.now(timezone.utc),
            metadata=data.get("metadata", {}),
        )


@dataclass
class CoreBookConfig:
    """Core book configuration."""
    # Default coins
    default_coins: List[str] = field(default_factory=lambda: ["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    
    # Exposure caps
    max_exposure_pct_per_coin: float = 10.0  # Max exposure as % of portfolio
    max_incremental_buy_pct: float = 2.0  # Max incremental buy as % of portfolio
    
    # DCA settings
    dca_drop_pct: float = 5.0  # DCA trigger when price drops this % below average cost
    max_dca_buys: int = 10  # Maximum number of DCA buys
    dca_cooldown_minutes: int = 60  # Cooldown between DCA buys
    
    # Profit threshold
    profit_threshold_absolute: float = 1.0  # Absolute profit threshold in base currency (e.g., £1)
    profit_threshold_pct: float = 5.0  # Percentage profit threshold
    partial_sell_pct: float = 25.0  # Percentage of position to sell on profit
    
    # Cooldowns
    action_cooldown_minutes: int = 15  # Cooldown between actions
    
    # Risk safeguards
    stop_dca_on_panic: bool = True
    min_liquidity_usd: float = 1_000_000.0  # Minimum liquidity to trade
    max_spread_bps: float = 50.0  # Maximum spread in basis points
    hedge_on_drawdown_pct: Optional[float] = None  # Hedge if drawdown exceeds this % (optional)
    
    # Allowable coins
    allowable_coins: List[str] = field(default_factory=lambda: ["BTCUSDT", "ETHUSDT", "SOLUSDT"])


@dataclass
class TradingAction:
    """Trading action."""
    action_type: ActionType
    symbol: str
    units: float
    price: float
    reason: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "action_type": self.action_type.value,
            "symbol": self.symbol,
            "units": self.units,
            "price": self.price,
            "reason": self.reason,
            "metadata": self.metadata,
        }


class CoreBookStrategy:
    """CoreBook strategy for top coins."""
    
    def __init__(
        self,
        config: CoreBookConfig,
        state_file: str = "core_book.json",
    ):
        """Initialize CoreBook strategy.
        
        Args:
            config: Core book configuration
            state_file: Path to state file
        """
        self.config = config
        self.state_file = Path(state_file)
        self.state = CoreBookState()
        
        # Initialize coins if not in state
        self._initialize_coins()
        
        # Load state if exists
        self._load_state()
        
        logger.info("core_book_strategy_initialized", 
                   coins=list(self.state.coins.keys()),
                   auto_trading_enabled=self.state.auto_trading_enabled)
    
    def _initialize_coins(self) -> None:
        """Initialize coins in state."""
        for symbol in self.config.default_coins:
            if symbol not in self.state.coins:
                self.state.coins[symbol] = CoreBookEntry(
                    symbol=symbol,
                    total_exposure_limit_pct=self.config.max_exposure_pct_per_coin,
                    max_dca_buys=self.config.max_dca_buys,
                )
    
    def _load_state(self) -> None:
        """Load state from file."""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                self.state = CoreBookState.from_dict(data)
                logger.info("core_book_state_loaded", state_file=str(self.state_file))
            except Exception as e:
                logger.error("core_book_state_load_failed", error=str(e))
                self.state = CoreBookState()
                self._initialize_coins()
        else:
            self._initialize_coins()
    
    def _save_state(self) -> bool:
        """Save state to file.
        
        Returns:
            True if successful
        """
        try:
            self.state.last_updated = datetime.now(timezone.utc)
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.state_file, 'w') as f:
                json.dump(self.state.to_dict(), f, indent=2)
            
            logger.debug("core_book_state_saved", state_file=str(self.state_file))
            return True
        except Exception as e:
            logger.error("core_book_state_save_failed", error=str(e))
            return False
    
    def evaluate(
        self,
        symbol: str,
        current_price: float,
        portfolio_value: float,
        market_regime: MarketRegime = MarketRegime.NORMAL,
        liquidity_usd: float = 10_000_000.0,
        spread_bps: float = 5.0,
    ) -> Optional[TradingAction]:
        """Evaluate and generate trading action.
        
        Args:
            symbol: Trading symbol
            current_price: Current price
            portfolio_value: Total portfolio value
            market_regime: Market regime
            liquidity_usd: Liquidity in USD
            spread_bps: Spread in basis points
            
        Returns:
            Trading action, or None if no action
        """
        if symbol not in self.state.coins:
            logger.warning("symbol_not_in_core_book", symbol=symbol)
            return None
        
        if not self.state.auto_trading_enabled:
            logger.debug("auto_trading_disabled", symbol=symbol)
            return None
        
        entry = self.state.coins[symbol]
        
        # Check risk safeguards
        if not self._check_risk_safeguards(symbol, market_regime, liquidity_usd, spread_bps):
            return None
        
        # Check cooldown
        if entry.cooldown_until and datetime.now(timezone.utc) < entry.cooldown_until:
            logger.debug("cooldown_active", symbol=symbol, cooldown_until=entry.cooldown_until)
            return None
        
        # Calculate current exposure
        current_exposure = entry.units_held * current_price
        current_exposure_pct = (current_exposure / portfolio_value * 100.0) if portfolio_value > 0 else 0.0
        
        # Check if we have a position
        if entry.units_held == 0.0:
            # No position - check if we should initiate
            return self._evaluate_initial_buy(symbol, current_price, portfolio_value, entry)
        else:
            # Has position - check DCA or partial sell
            # Update DCA trigger and partial sell target
            self._update_triggers(entry, current_price)
            
            # Check for DCA buy
            dca_action = self._evaluate_dca_buy(symbol, current_price, portfolio_value, entry, market_regime)
            if dca_action:
                return dca_action
            
            # Check for partial sell (profit taking)
            sell_action = self._evaluate_partial_sell(symbol, current_price, portfolio_value, entry)
            if sell_action:
                return sell_action
        
        return None
    
    def _check_risk_safeguards(
        self,
        symbol: str,
        market_regime: MarketRegime,
        liquidity_usd: float,
        spread_bps: float,
    ) -> bool:
        """Check risk safeguards.
        
        Args:
            symbol: Trading symbol
            market_regime: Market regime
            liquidity_usd: Liquidity in USD
            spread_bps: Spread in basis points
            
        Returns:
            True if safeguards pass
        """
        # Stop DCA on panic
        if self.config.stop_dca_on_panic and market_regime == MarketRegime.PANIC:
            logger.debug("dca_stopped_panic", symbol=symbol)
            return False
        
        # Check liquidity
        if liquidity_usd < self.config.min_liquidity_usd:
            logger.debug("liquidity_too_low", symbol=symbol, liquidity_usd=liquidity_usd)
            return False
        
        # Check spread
        if spread_bps > self.config.max_spread_bps:
            logger.debug("spread_too_high", symbol=symbol, spread_bps=spread_bps)
            return False
        
        return True
    
    def _evaluate_initial_buy(
        self,
        symbol: str,
        current_price: float,
        portfolio_value: float,
        entry: CoreBookEntry,
    ) -> Optional[TradingAction]:
        """Evaluate initial buy.
        
        Args:
            symbol: Trading symbol
            current_price: Current price
            portfolio_value: Total portfolio value
            entry: Core book entry
            
        Returns:
            Trading action, or None
        """
        # For initial buy, we can start with a small position
        # This is typically done manually or via Telegram command
        # Auto-initial buy is optional and can be disabled
        return None
    
    def _evaluate_dca_buy(
        self,
        symbol: str,
        current_price: float,
        portfolio_value: float,
        entry: CoreBookEntry,
        market_regime: MarketRegime,
    ) -> Optional[TradingAction]:
        """Evaluate DCA buy.
        
        Args:
            symbol: Trading symbol
            current_price: Current price
            portfolio_value: Total portfolio value
            entry: Core book entry
            market_regime: Market regime
            
        Returns:
            Trading action, or None
        """
        # Check if we have a position
        if entry.units_held == 0.0:
            return None
        
        # Check if price is below average cost
        if current_price >= entry.average_cost_price:
            return None
        
        # Check if price is below DCA trigger
        if entry.next_dca_trigger_price and current_price > entry.next_dca_trigger_price:
            return None
        
        # Check DCA count
        if entry.dca_count >= entry.max_dca_buys:
            logger.debug("max_dca_buys_reached", symbol=symbol, dca_count=entry.dca_count)
            return None
        
        # Check exposure cap
        current_exposure = entry.units_held * current_price
        current_exposure_pct = (current_exposure / portfolio_value * 100.0) if portfolio_value > 0 else 0.0
        
        if current_exposure_pct >= entry.total_exposure_limit_pct:
            logger.debug("exposure_cap_reached", symbol=symbol, exposure_pct=current_exposure_pct)
            return None
        
        # Calculate DCA buy amount
        drop_pct = ((entry.average_cost_price - current_price) / entry.average_cost_price) * 100.0
        
        # DCA buy size: incremental buy % of portfolio, scaled by drop %
        max_buy_value = portfolio_value * (self.config.max_incremental_buy_pct / 100.0)
        # Scale by drop % (more aggressive DCA on larger drops)
        dca_buy_value = max_buy_value * min(drop_pct / self.config.dca_drop_pct, 2.0)
        
        # Check if buy would exceed exposure cap
        new_exposure = current_exposure + dca_buy_value
        new_exposure_pct = (new_exposure / portfolio_value * 100.0) if portfolio_value > 0 else 0.0
        
        if new_exposure_pct > entry.total_exposure_limit_pct:
            # Reduce buy to fit within cap
            max_additional_exposure = (entry.total_exposure_limit_pct / 100.0) * portfolio_value - current_exposure
            dca_buy_value = max(0.0, max_additional_exposure)
        
        if dca_buy_value <= 0.0:
            return None
        
        # Calculate units to buy
        units = dca_buy_value / current_price
        
        # Update entry
        entry.update_average_cost(units, current_price)
        entry.dca_count += 1
        entry.last_action_timestamp = datetime.now(timezone.utc)
        entry.last_action_type = ActionType.DCA_BUY.value
        entry.cooldown_until = datetime.now(timezone.utc) + timedelta(minutes=self.config.dca_cooldown_minutes)
        self._update_triggers(entry, current_price)
        
        # Save state
        self._save_state()
        
        logger.info("dca_buy_triggered", 
                   symbol=symbol,
                   units=units,
                   price=current_price,
                   new_avg_cost=entry.average_cost_price,
                   dca_count=entry.dca_count)
        
        return TradingAction(
            action_type=ActionType.DCA_BUY,
            symbol=symbol,
            units=units,
            price=current_price,
            reason=f"DCA buy: price {current_price:.2f} below avg cost {entry.average_cost_price:.2f}",
            metadata={
                "average_cost_price": entry.average_cost_price,
                "dca_count": entry.dca_count,
                "drop_pct": drop_pct,
            },
        )
    
    def _evaluate_partial_sell(
        self,
        symbol: str,
        current_price: float,
        portfolio_value: float,
        entry: CoreBookEntry,
    ) -> Optional[TradingAction]:
        """Evaluate partial sell (profit taking).
        
        Args:
            symbol: Trading symbol
            current_price: Current price
            portfolio_value: Total portfolio value
            entry: Core book entry
            
        Returns:
            Trading action, or None
        """
        # Never sell at a loss
        if current_price <= entry.average_cost_price:
            return None
        
        # Calculate unrealized P&L
        unrealized_pnl = entry.calculate_unrealized_pnl(current_price)
        unrealized_pnl_pct = entry.calculate_unrealized_pnl_pct(current_price)
        
        # Check profit thresholds
        profit_threshold_met = (
            unrealized_pnl >= self.config.profit_threshold_absolute or
            unrealized_pnl_pct >= self.config.profit_threshold_pct
        )
        
        if not profit_threshold_met:
            return None
        
        # Check if price is above partial sell target
        if entry.partial_sell_target_price and current_price < entry.partial_sell_target_price:
            return None
        
        # Calculate partial sell amount
        sell_units = entry.units_held * (self.config.partial_sell_pct / 100.0)
        
        # Ensure we don't sell everything
        if sell_units >= entry.units_held:
            sell_units = entry.units_held * 0.5  # Sell 50% max
        
        # Update entry
        entry.units_held -= sell_units
        entry.total_cost_basis = entry.average_cost_price * entry.units_held  # Proportional reduction
        entry.last_action_timestamp = datetime.now(timezone.utc)
        entry.last_action_type = ActionType.PARTIAL_SELL.value
        entry.cooldown_until = datetime.now(timezone.utc) + timedelta(minutes=self.config.action_cooldown_minutes)
        self._update_triggers(entry, current_price)
        
        # Save state
        self._save_state()
        
        logger.info("partial_sell_triggered",
                   symbol=symbol,
                   units=sell_units,
                   price=current_price,
                   unrealized_pnl=unrealized_pnl,
                   unrealized_pnl_pct=unrealized_pnl_pct)
        
        return TradingAction(
            action_type=ActionType.PARTIAL_SELL,
            symbol=symbol,
            units=sell_units,
            price=current_price,
            reason=f"Partial sell: profit {unrealized_pnl:.2f} ({unrealized_pnl_pct:.2f}%)",
            metadata={
                "average_cost_price": entry.average_cost_price,
                "unrealized_pnl": unrealized_pnl,
                "unrealized_pnl_pct": unrealized_pnl_pct,
                "units_remaining": entry.units_held,
            },
        )
    
    def _update_triggers(self, entry: CoreBookEntry, current_price: float) -> None:
        """Update DCA trigger and partial sell target prices.
        
        Args:
            entry: Core book entry
            current_price: Current price
        """
        if entry.units_held == 0.0:
            entry.next_dca_trigger_price = None
            entry.partial_sell_target_price = None
            return
        
        # DCA trigger: price drops by dca_drop_pct below average cost
        entry.next_dca_trigger_price = entry.average_cost_price * (1 - self.config.dca_drop_pct / 100.0)
        
        # Partial sell target: price rises to profit threshold
        # Use the higher of absolute or percentage threshold
        profit_target_absolute = entry.average_cost_price + (self.config.profit_threshold_absolute / entry.units_held) if entry.units_held > 0 else entry.average_cost_price
        profit_target_pct = entry.average_cost_price * (1 + self.config.profit_threshold_pct / 100.0)
        entry.partial_sell_target_price = max(profit_target_absolute, profit_target_pct)
    
    def execute_buy(
        self,
        symbol: str,
        units: float,
        price: float,
        portfolio_value: float,
    ) -> bool:
        """Execute a buy (manual or initial).
        
        Args:
            symbol: Trading symbol
            units: Units to buy
            price: Price per unit
            portfolio_value: Total portfolio value
            
        Returns:
            True if successful
        """
        if symbol not in self.state.coins:
            logger.error("symbol_not_in_core_book", symbol=symbol)
            return False
        
        entry = self.state.coins[symbol]
        
        # Check exposure cap
        current_exposure = entry.units_held * price
        new_exposure = (entry.units_held + units) * price
        new_exposure_pct = (new_exposure / portfolio_value * 100.0) if portfolio_value > 0 else 0.0
        
        if new_exposure_pct > entry.total_exposure_limit_pct:
            logger.warning("buy_exceeds_exposure_cap", 
                          symbol=symbol,
                          new_exposure_pct=new_exposure_pct,
                          limit_pct=entry.total_exposure_limit_pct)
            return False
        
        # Update entry
        entry.update_average_cost(units, price)
        entry.last_action_timestamp = datetime.now(timezone.utc)
        entry.last_action_type = ActionType.BUY.value
        entry.cooldown_until = datetime.now(timezone.utc) + timedelta(minutes=self.config.action_cooldown_minutes)
        self._update_triggers(entry, price)
        
        # Save state
        self._save_state()
        
        logger.info("buy_executed", symbol=symbol, units=units, price=price, new_avg_cost=entry.average_cost_price)
        return True
    
    def execute_sell(
        self,
        symbol: str,
        units: float,
        price: float,
    ) -> bool:
        """Execute a sell (manual).
        
        Args:
            symbol: Trading symbol
            units: Units to sell
            price: Price per unit
            
        Returns:
            True if successful
        """
        if symbol not in self.state.coins:
            logger.error("symbol_not_in_core_book", symbol=symbol)
            return False
        
        entry = self.state.coins[symbol]
        
        # Never sell at a loss
        if price < entry.average_cost_price:
            logger.warning("sell_blocked_loss", 
                          symbol=symbol,
                          price=price,
                          avg_cost=entry.average_cost_price)
            return False
        
        # Check if we have enough units
        if units > entry.units_held:
            logger.warning("insufficient_units", symbol=symbol, units_requested=units, units_held=entry.units_held)
            return False
        
        # Update entry
        entry.units_held -= units
        entry.total_cost_basis = entry.average_cost_price * entry.units_held  # Proportional reduction
        entry.last_action_timestamp = datetime.now(timezone.utc)
        entry.last_action_type = ActionType.SELL.value
        entry.cooldown_until = datetime.now(timezone.utc) + timedelta(minutes=self.config.action_cooldown_minutes)
        self._update_triggers(entry, price)
        
        # Save state
        self._save_state()
        
        logger.info("sell_executed", symbol=symbol, units=units, price=price, units_remaining=entry.units_held)
        return True
    
    def set_exposure_cap(self, symbol: str, exposure_pct: float) -> bool:
        """Set exposure cap for a coin.
        
        Args:
            symbol: Trading symbol
            exposure_pct: Exposure limit as % of portfolio
            
        Returns:
            True if successful
        """
        if symbol not in self.state.coins:
            logger.error("symbol_not_in_core_book", symbol=symbol)
            return False
        
        self.state.coins[symbol].total_exposure_limit_pct = exposure_pct
        self._save_state()
        
        logger.info("exposure_cap_set", symbol=symbol, exposure_pct=exposure_pct)
        return True
    
    def add_coin(self, symbol: str, exposure_pct: float = 10.0) -> bool:
        """Add a coin to CoreBook.
        
        Args:
            symbol: Trading symbol
            exposure_pct: Exposure limit as % of portfolio
            
        Returns:
            True if successful
        """
        if symbol not in self.config.allowable_coins:
            logger.warning("coin_not_allowable", symbol=symbol, allowable_coins=self.config.allowable_coins)
            return False
        
        if symbol in self.state.coins:
            logger.warning("coin_already_in_core_book", symbol=symbol)
            return False
        
        self.state.coins[symbol] = CoreBookEntry(
            symbol=symbol,
            total_exposure_limit_pct=exposure_pct,
            max_dca_buys=self.config.max_dca_buys,
        )
        self._save_state()
        
        logger.info("coin_added", symbol=symbol, exposure_pct=exposure_pct)
        return True
    
    def trim_position(self, symbol: str, trim_pct: float = 25.0) -> bool:
        """Trim a position (sell a percentage).
        
        Args:
            symbol: Trading symbol
            trim_pct: Percentage to trim
            
        Returns:
            True if successful
        """
        if symbol not in self.state.coins:
            logger.error("symbol_not_in_core_book", symbol=symbol)
            return False
        
        entry = self.state.coins[symbol]
        
        if entry.units_held == 0.0:
            logger.warning("no_position_to_trim", symbol=symbol)
            return False
        
        # Get current price (would need to be passed in real implementation)
        # For now, use average cost as minimum sell price
        current_price = entry.average_cost_price * 1.01  # Assume 1% profit
        
        trim_units = entry.units_held * (trim_pct / 100.0)
        
        return self.execute_sell(symbol, trim_units, current_price)
    
    def set_auto_trading(self, enabled: bool) -> bool:
        """Set auto trading enabled/disabled.
        
        Args:
            enabled: True to enable, False to disable
            
        Returns:
            True if successful
        """
        self.state.auto_trading_enabled = enabled
        self._save_state()
        
        logger.info("auto_trading_set", enabled=enabled)
        return True
    
    def get_status(self) -> Dict[str, Any]:
        """Get CoreBook status.
        
        Returns:
            Status dictionary
        """
        return {
            "auto_trading_enabled": self.state.auto_trading_enabled,
            "coins": {
                symbol: entry.to_dict()
                for symbol, entry in self.state.coins.items()
            },
            "last_updated": self.state.last_updated.isoformat(),
        }
    
    def get_coin_status(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get status for a specific coin.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Coin status dictionary, or None if not found
        """
        if symbol not in self.state.coins:
            return None
        
        entry = self.state.coins[symbol]
        return entry.to_dict()

