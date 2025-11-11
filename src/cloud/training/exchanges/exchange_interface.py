"""
Exchange Abstraction Layer for Scalable Architecture

Provides unified interface for multiple exchanges (Binance, OKX, Bybit).
Supports connection pooling, retry logic, and rate limit handling.

Author: Huracan Engine Team
Date: 2025-01-27
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Any
from datetime import datetime

import structlog

logger = structlog.get_logger(__name__)


class OrderSide(Enum):
    """Order side."""
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    """Order type."""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"


class OrderStatus(Enum):
    """Order status."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class OrderBook:
    """Orderbook snapshot."""
    symbol: str
    bids: List[tuple[float, float]]  # [(price, size), ...]
    asks: List[tuple[float, float]]  # [(price, size), ...]
    timestamp: float
    exchange: str


@dataclass
class Order:
    """Order representation."""
    symbol: str
    side: OrderSide
    order_type: OrderType
    amount: float
    price: Optional[float] = None
    order_id: Optional[str] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_amount: float = 0.0
    average_price: Optional[float] = None
    created_at: datetime = None
    updated_at: datetime = None


@dataclass
class FeeStructure:
    """Fee structure for an exchange."""
    maker_fee_bps: float
    taker_fee_bps: float
    funding_rate_bps: float  # Per 8 hours, annualized
    last_update: float


@dataclass
class Balance:
    """Account balance."""
    currency: str
    available: float
    locked: float
    total: float


class ExchangeInterface(ABC):
    """Abstract interface for exchange implementations."""
    
    @abstractmethod
    async def get_orderbook(self, symbol: str) -> OrderBook:
        """Get orderbook for a symbol."""
        pass
    
    @abstractmethod
    async def place_order(self, order: Order) -> Order:
        """Place an order."""
        pass
    
    @abstractmethod
    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel an order."""
        pass
    
    @abstractmethod
    async def get_order_status(self, symbol: str, order_id: str) -> Order:
        """Get order status."""
        pass
    
    @abstractmethod
    async def get_fees(self, symbol: str) -> FeeStructure:
        """Get fee structure for a symbol."""
        pass
    
    @abstractmethod
    async def get_funding_rate(self, symbol: str) -> float:
        """Get funding rate for a symbol (per 8 hours, annualized in bps)."""
        pass
    
    @abstractmethod
    async def get_balance(self, currency: str) -> Balance:
        """Get balance for a currency."""
        pass
    
    @abstractmethod
    async def get_symbols(self) -> List[str]:
        """Get list of available symbols."""
        pass


class RateLimiter:
    """Rate limiter for exchange API calls."""
    
    def __init__(self, max_requests: int, window_seconds: int = 60):
        """
        Initialize rate limiter.
        
        Args:
            max_requests: Maximum requests per window
            window_seconds: Time window in seconds
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: List[float] = []
        self.lock = asyncio.Lock()
    
    async def acquire(self) -> None:
        """Acquire rate limit token."""
        async with self.lock:
            now = time.time()
            
            # Remove requests outside window
            self.requests = [req_time for req_time in self.requests if now - req_time < self.window_seconds]
            
            # Check if we've exceeded limit
            if len(self.requests) >= self.max_requests:
                # Wait until oldest request expires
                oldest_request = min(self.requests)
                wait_time = self.window_seconds - (now - oldest_request) + 0.1
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
                    # Remove expired requests
                    self.requests = [req_time for req_time in self.requests if now + wait_time - req_time < self.window_seconds]
            
            # Add current request
            self.requests.append(time.time())


class BaseExchange(ExchangeInterface):
    """Base implementation with common functionality."""
    
    def __init__(
        self,
        name: str,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        rate_limit_requests: int = 1200,
        rate_limit_window: int = 60,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        Initialize base exchange.
        
        Args:
            name: Exchange name
            api_key: API key (optional for public endpoints)
            api_secret: API secret (optional for public endpoints)
            rate_limit_requests: Maximum requests per window
            rate_limit_window: Time window in seconds
            max_retries: Maximum retry attempts
            retry_delay: Delay between retries in seconds
        """
        self.name = name
        self.api_key = api_key
        self.api_secret = api_secret
        self.rate_limiter = RateLimiter(rate_limit_requests, rate_limit_window)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        logger.info(
            "exchange_initialized",
            name=name,
            rate_limit=rate_limit_requests,
        )
    
    async def _rate_limited_call(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        """Make a rate-limited API call with retry logic."""
        for attempt in range(self.max_retries):
            try:
                await self.rate_limiter.acquire()
                return await func(*args, **kwargs)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    logger.error(
                        "exchange_api_call_failed",
                        exchange=self.name,
                        error=str(e),
                        attempts=attempt + 1,
                    )
                    raise
                
                logger.warning(
                    "exchange_api_call_retry",
                    exchange=self.name,
                    error=str(e),
                    attempt=attempt + 1,
                    max_retries=self.max_retries,
                )
                await asyncio.sleep(self.retry_delay * (attempt + 1))
    
    @abstractmethod
    async def _get_orderbook_impl(self, symbol: str) -> OrderBook:
        """Implementation-specific orderbook fetch."""
        pass
    
    @abstractmethod
    async def _place_order_impl(self, order: Order) -> Order:
        """Implementation-specific order placement."""
        pass
    
    @abstractmethod
    async def _cancel_order_impl(self, symbol: str, order_id: str) -> bool:
        """Implementation-specific order cancellation."""
        pass
    
    @abstractmethod
    async def _get_order_status_impl(self, symbol: str, order_id: str) -> Order:
        """Implementation-specific order status fetch."""
        pass
    
    @abstractmethod
    async def _get_fees_impl(self, symbol: str) -> FeeStructure:
        """Implementation-specific fee fetch."""
        pass
    
    @abstractmethod
    async def _get_funding_rate_impl(self, symbol: str) -> float:
        """Implementation-specific funding rate fetch."""
        pass
    
    @abstractmethod
    async def _get_balance_impl(self, currency: str) -> Balance:
        """Implementation-specific balance fetch."""
        pass
    
    @abstractmethod
    async def _get_symbols_impl(self) -> List[str]:
        """Implementation-specific symbols fetch."""
        pass
    
    async def get_orderbook(self, symbol: str) -> OrderBook:
        """Get orderbook for a symbol."""
        return await self._rate_limited_call(self._get_orderbook_impl, symbol)
    
    async def place_order(self, order: Order) -> Order:
        """Place an order."""
        return await self._rate_limited_call(self._place_order_impl, order)
    
    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel an order."""
        return await self._rate_limited_call(self._cancel_order_impl, symbol, order_id)
    
    async def get_order_status(self, symbol: str, order_id: str) -> Order:
        """Get order status."""
        return await self._rate_limited_call(self._get_order_status_impl, symbol, order_id)
    
    async def get_fees(self, symbol: str) -> FeeStructure:
        """Get fee structure for a symbol."""
        return await self._rate_limited_call(self._get_fees_impl, symbol)
    
    async def get_funding_rate(self, symbol: str) -> float:
        """Get funding rate for a symbol."""
        return await self._rate_limited_call(self._get_funding_rate_impl, symbol)
    
    async def get_balance(self, currency: str) -> Balance:
        """Get balance for a currency."""
        return await self._rate_limited_call(self._get_balance_impl, currency)
    
    async def get_symbols(self) -> List[str]:
        """Get list of available symbols."""
        return await self._rate_limited_call(self._get_symbols_impl)


class BinanceExchange(BaseExchange):
    """Binance exchange implementation."""
    
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        """Initialize Binance exchange."""
        super().__init__(
            name="binance",
            api_key=api_key,
            api_secret=api_secret,
            rate_limit_requests=1200,  # Binance limit: 1200 requests per minute
            rate_limit_window=60,
        )
        logger.info("binance_exchange_initialized")
    
    async def _get_orderbook_impl(self, symbol: str) -> OrderBook:
        """Get orderbook from Binance."""
        # Placeholder implementation
        # In production, use binance API client
        logger.debug("binance_get_orderbook", symbol=symbol)
        # Simulated response
        return OrderBook(
            symbol=symbol,
            bids=[(50000.0, 1.0), (49999.0, 2.0)],
            asks=[(50001.0, 1.0), (50002.0, 2.0)],
            timestamp=time.time(),
            exchange="binance",
        )
    
    async def _place_order_impl(self, order: Order) -> Order:
        """Place order on Binance."""
        logger.debug("binance_place_order", symbol=order.symbol, side=order.side.value)
        # Placeholder implementation
        order.order_id = f"binance_{int(time.time() * 1000)}"
        order.status = OrderStatus.SUBMITTED
        order.created_at = datetime.now()
        return order
    
    async def _cancel_order_impl(self, symbol: str, order_id: str) -> bool:
        """Cancel order on Binance."""
        logger.debug("binance_cancel_order", symbol=symbol, order_id=order_id)
        return True
    
    async def _get_order_status_impl(self, symbol: str, order_id: str) -> Order:
        """Get order status from Binance."""
        logger.debug("binance_get_order_status", symbol=symbol, order_id=order_id)
        # Placeholder implementation
        return Order(
            symbol=symbol,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=1.0,
            order_id=order_id,
            status=OrderStatus.FILLED,
        )
    
    async def _get_fees_impl(self, symbol: str) -> FeeStructure:
        """Get fees from Binance."""
        logger.debug("binance_get_fees", symbol=symbol)
        # Binance default: 0.1% maker, 0.1% taker (10 bps each)
        return FeeStructure(
            maker_fee_bps=10.0,
            taker_fee_bps=10.0,
            funding_rate_bps=0.0,
            last_update=time.time(),
        )
    
    async def _get_funding_rate_impl(self, symbol: str) -> float:
        """Get funding rate from Binance."""
        logger.debug("binance_get_funding_rate", symbol=symbol)
        return 0.0  # Placeholder
    
    async def _get_balance_impl(self, currency: str) -> Balance:
        """Get balance from Binance."""
        logger.debug("binance_get_balance", currency=currency)
        # Placeholder implementation
        return Balance(
            currency=currency,
            available=10000.0,
            locked=0.0,
            total=10000.0,
        )
    
    async def _get_symbols_impl(self) -> List[str]:
        """Get symbols from Binance."""
        logger.debug("binance_get_symbols")
        # Placeholder implementation
        return ["BTCUSDT", "ETHUSDT", "SOLUSDT"]


class OKXExchange(BaseExchange):
    """OKX exchange implementation."""
    
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        """Initialize OKX exchange."""
        super().__init__(
            name="okx",
            api_key=api_key,
            api_secret=api_secret,
            rate_limit_requests=20,  # OKX limit: 20 requests per 2 seconds
            rate_limit_window=2,
        )
        logger.info("okx_exchange_initialized")
    
    async def _get_orderbook_impl(self, symbol: str) -> OrderBook:
        """Get orderbook from OKX."""
        logger.debug("okx_get_orderbook", symbol=symbol)
        # Placeholder implementation
        return OrderBook(
            symbol=symbol,
            bids=[(50000.0, 1.0), (49999.0, 2.0)],
            asks=[(50001.0, 1.0), (50002.0, 2.0)],
            timestamp=time.time(),
            exchange="okx",
        )
    
    async def _place_order_impl(self, order: Order) -> Order:
        """Place order on OKX."""
        logger.debug("okx_place_order", symbol=order.symbol, side=order.side.value)
        order.order_id = f"okx_{int(time.time() * 1000)}"
        order.status = OrderStatus.SUBMITTED
        order.created_at = datetime.now()
        return order
    
    async def _cancel_order_impl(self, symbol: str, order_id: str) -> bool:
        """Cancel order on OKX."""
        logger.debug("okx_cancel_order", symbol=symbol, order_id=order_id)
        return True
    
    async def _get_order_status_impl(self, symbol: str, order_id: str) -> Order:
        """Get order status from OKX."""
        logger.debug("okx_get_order_status", symbol=symbol, order_id=order_id)
        return Order(
            symbol=symbol,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=1.0,
            order_id=order_id,
            status=OrderStatus.FILLED,
        )
    
    async def _get_fees_impl(self, symbol: str) -> FeeStructure:
        """Get fees from OKX."""
        logger.debug("okx_get_fees", symbol=symbol)
        # OKX default: 0.08% maker, 0.1% taker (8 bps maker, 10 bps taker)
        return FeeStructure(
            maker_fee_bps=8.0,
            taker_fee_bps=10.0,
            funding_rate_bps=0.0,
            last_update=time.time(),
        )
    
    async def _get_funding_rate_impl(self, symbol: str) -> float:
        """Get funding rate from OKX."""
        logger.debug("okx_get_funding_rate", symbol=symbol)
        return 0.0
    
    async def _get_balance_impl(self, currency: str) -> Balance:
        """Get balance from OKX."""
        logger.debug("okx_get_balance", currency=currency)
        return Balance(
            currency=currency,
            available=10000.0,
            locked=0.0,
            total=10000.0,
        )
    
    async def _get_symbols_impl(self) -> List[str]:
        """Get symbols from OKX."""
        logger.debug("okx_get_symbols")
        return ["BTC-USDT", "ETH-USDT", "SOL-USDT"]


class BybitExchange(BaseExchange):
    """Bybit exchange implementation."""
    
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        """Initialize Bybit exchange."""
        super().__init__(
            name="bybit",
            api_key=api_key,
            api_secret=api_secret,
            rate_limit_requests=120,  # Bybit limit: 120 requests per minute
            rate_limit_window=60,
        )
        logger.info("bybit_exchange_initialized")
    
    async def _get_orderbook_impl(self, symbol: str) -> OrderBook:
        """Get orderbook from Bybit."""
        logger.debug("bybit_get_orderbook", symbol=symbol)
        return OrderBook(
            symbol=symbol,
            bids=[(50000.0, 1.0), (49999.0, 2.0)],
            asks=[(50001.0, 1.0), (50002.0, 2.0)],
            timestamp=time.time(),
            exchange="bybit",
        )
    
    async def _place_order_impl(self, order: Order) -> Order:
        """Place order on Bybit."""
        logger.debug("bybit_place_order", symbol=order.symbol, side=order.side.value)
        order.order_id = f"bybit_{int(time.time() * 1000)}"
        order.status = OrderStatus.SUBMITTED
        order.created_at = datetime.now()
        return order
    
    async def _cancel_order_impl(self, symbol: str, order_id: str) -> bool:
        """Cancel order on Bybit."""
        logger.debug("bybit_cancel_order", symbol=symbol, order_id=order_id)
        return True
    
    async def _get_order_status_impl(self, symbol: str, order_id: str) -> Order:
        """Get order status from Bybit."""
        logger.debug("bybit_get_order_status", symbol=symbol, order_id=order_id)
        return Order(
            symbol=symbol,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            amount=1.0,
            order_id=order_id,
            status=OrderStatus.FILLED,
        )
    
    async def _get_fees_impl(self, symbol: str) -> FeeStructure:
        """Get fees from Bybit."""
        logger.debug("bybit_get_fees", symbol=symbol)
        # Bybit default: 0.055% maker, 0.055% taker (5.5 bps each)
        return FeeStructure(
            maker_fee_bps=5.5,
            taker_fee_bps=5.5,
            funding_rate_bps=0.0,
            last_update=time.time(),
        )
    
    async def _get_funding_rate_impl(self, symbol: str) -> float:
        """Get funding rate from Bybit."""
        logger.debug("bybit_get_funding_rate", symbol=symbol)
        return 0.0
    
    async def _get_balance_impl(self, currency: str) -> Balance:
        """Get balance from Bybit."""
        logger.debug("bybit_get_balance", currency=currency)
        return Balance(
            currency=currency,
            available=10000.0,
            locked=0.0,
            total=10000.0,
        )
    
    async def _get_symbols_impl(self) -> List[str]:
        """Get symbols from Bybit."""
        logger.debug("bybit_get_symbols")
        return ["BTCUSDT", "ETHUSDT", "SOLUSDT"]


class ExchangeManager:
    """Manager for multiple exchanges."""
    
    def __init__(self):
        """Initialize exchange manager."""
        self.exchanges: Dict[str, ExchangeInterface] = {}
        logger.info("exchange_manager_initialized")
    
    def register_exchange(self, name: str, exchange: ExchangeInterface) -> None:
        """Register an exchange."""
        self.exchanges[name] = exchange
        logger.info("exchange_registered", name=name)
    
    def get_exchange(self, name: str) -> Optional[ExchangeInterface]:
        """Get an exchange by name."""
        return self.exchanges.get(name)
    
    async def get_best_orderbook(self, symbol: str) -> tuple[Optional[OrderBook], Optional[str]]:
        """
        Get best orderbook across all exchanges.
        
        Args:
            symbol: Trading symbol
        
        Returns:
            Tuple of (best_orderbook, exchange_name)
        """
        best_orderbook = None
        best_exchange = None
        best_spread = float('inf')
        
        for name, exchange in self.exchanges.items():
            try:
                orderbook = await exchange.get_orderbook(symbol)
                
                # Calculate spread
                if orderbook.bids and orderbook.asks:
                    bid = orderbook.bids[0][0]
                    ask = orderbook.asks[0][0]
                    spread = ask - bid
                    
                    if spread < best_spread:
                        best_spread = spread
                        best_orderbook = orderbook
                        best_exchange = name
            except Exception as e:
                logger.warning("exchange_orderbook_failed", exchange=name, error=str(e))
                continue
        
        return best_orderbook, best_exchange
    
    async def get_all_fees(self, symbol: str) -> Dict[str, FeeStructure]:
        """Get fees from all exchanges."""
        fees = {}
        for name, exchange in self.exchanges.items():
            try:
                fee_structure = await exchange.get_fees(symbol)
                fees[name] = fee_structure
            except Exception as e:
                logger.warning("exchange_fees_failed", exchange=name, error=str(e))
                continue
        return fees

