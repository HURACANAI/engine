"""Exchange abstraction built on top of ccxt for Binance-first compatibility."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Optional

import ccxt
import structlog
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    RetryError,
)

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class ExchangeMarket:
	symbol: str
	base: Optional[str]
	quote: Optional[str]
	maker: float
	taker: float
	active: bool
	info: Dict[str, Any]


class ExchangeClient:
	"""Thin wrapper around ccxt to keep the Engine exchange-agnostic."""

	def __init__(
		self,
		exchange_id: str,
		credentials: Optional[Dict[str, Optional[str]]] = None,
		sandbox: bool = False,
		load_markets: bool = True,
		request_timeout: int = 30_000,
	) -> None:
		if not hasattr(ccxt, exchange_id):
			raise ValueError(f"Unsupported exchange '{exchange_id}'")
		klass = getattr(ccxt, exchange_id)
		params: Dict[str, Any] = {
			"enableRateLimit": True,
			"options": {"defaultType": "spot"},
			"timeout": request_timeout,
		}
		if credentials:
			params.update(
				{
					key: value
					for key, value in {
						"apiKey": credentials.get("api_key"),
						"secret": credentials.get("api_secret"),
						"password": credentials.get("api_passphrase"),
					}.items()
					if value
				}
			)
		self._client: ccxt.Exchange = klass(params)
		if sandbox and hasattr(self._client, "set_sandbox_mode"):
			self._client.set_sandbox_mode(True)
		if load_markets:
			self._client.load_markets(reload=True)

	@property
	def exchange_id(self) -> str:
		return str(self._client.id)

	def format_symbol(self, base: str, quote: str) -> str:
		return f"{base}/{quote}" if "/" not in base else base

	@retry(
		stop=stop_after_attempt(3),
		wait=wait_exponential(multiplier=1, min=1, max=10),
		retry=retry_if_exception_type((ccxt.NetworkError, ccxt.ExchangeError, ConnectionError, TimeoutError)),
		reraise=True,
	)
	def fetch_markets(self) -> Dict[str, ExchangeMarket]:
		"""Fetch market information with retry logic."""
		try:
			markets = self._client.load_markets(reload=False)
			return {
				symbol: ExchangeMarket(
					symbol=symbol,
					base=data.get("base"),
					quote=data.get("quote"),
					maker=data.get("maker", 0.0) or 0.0,
					taker=data.get("taker", 0.0) or 0.0,
					active=data.get("active", False),
					info=data.get("info", {}),
				)
				for symbol, data in markets.items()
			}
		except (ccxt.NetworkError, ccxt.ExchangeError, ConnectionError, TimeoutError) as e:
			logger.warning("markets_fetch_retry", error=str(e), error_type=type(e).__name__)
			raise

	@retry(
		stop=stop_after_attempt(3),
		wait=wait_exponential(multiplier=1, min=1, max=10),
		retry=retry_if_exception_type((ccxt.NetworkError, ccxt.ExchangeError, ConnectionError, TimeoutError)),
		reraise=True,
	)
	def fetch_tickers(self, symbols: Optional[Iterable[str]] = None) -> Dict[str, Dict[str, Any]]:
		"""Fetch ticker data with retry logic."""
		try:
			return self._client.fetch_tickers(list(symbols) if symbols else None)
		except (ccxt.NetworkError, ccxt.ExchangeError, ConnectionError, TimeoutError) as e:
			logger.warning("tickers_fetch_retry", error=str(e), error_type=type(e).__name__)
			raise

	@retry(
		stop=stop_after_attempt(3),
		wait=wait_exponential(multiplier=1, min=1, max=10),
		retry=retry_if_exception_type((ccxt.NetworkError, ccxt.ExchangeError, ConnectionError, TimeoutError)),
		reraise=True,
	)
	def fetch_ohlcv(
		self,
		symbol: str,
		timeframe: str,
		since: Optional[int] = None,
		limit: Optional[int] = None,
		params: Optional[Dict[str, Any]] = None,
	) -> list[list[Any]]:
		"""
		Fetch OHLCV data with automatic retry on network errors.
		
		Args:
			symbol: Trading symbol (e.g., 'BTC/USDT')
			timeframe: Timeframe (e.g., '1m', '1h', '1d')
			since: Start timestamp in milliseconds
			limit: Number of candles to fetch
			params: Additional parameters
			
		Returns:
			List of OHLCV candles: [[timestamp, open, high, low, close, volume], ...]
			
		Raises:
			ccxt.NetworkError: If network error persists after retries
			ccxt.ExchangeError: If exchange error persists after retries
		"""
		# Validate inputs
		if not symbol or not isinstance(symbol, str):
			raise ValueError(f"Invalid symbol: {symbol}")
		if not timeframe or not isinstance(timeframe, str):
			raise ValueError(f"Invalid timeframe: {timeframe}")
		if limit is not None and (not isinstance(limit, int) or limit <= 0):
			raise ValueError(f"Invalid limit: {limit}")
		
		# Fix: Ensure params is always a dict, never None
		safe_params = params if params is not None else {}

		try:
			result = self._client.fetch_ohlcv(
				symbol,
				timeframe=timeframe,
				since=since,
				limit=limit,
				params=safe_params
			)
			if not result:
				logger.warning("empty_ohlcv_result", symbol=symbol, timeframe=timeframe)
				return []
			return result
		except (ccxt.NetworkError, ccxt.ExchangeError, ConnectionError, TimeoutError) as e:
			logger.warning(
				"ohlcv_fetch_retry",
				symbol=symbol,
				timeframe=timeframe,
				error=str(e),
				error_type=type(e).__name__,
			)
			raise

	def parse_exchange_timestamp(self, ms_timestamp: int) -> datetime:
		return datetime.fromtimestamp(ms_timestamp / 1_000, tz=timezone.utc)