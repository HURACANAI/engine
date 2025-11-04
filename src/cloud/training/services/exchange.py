"""Exchange abstraction built on top of ccxt for Binance-first compatibility."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Optional

import ccxt


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

	def fetch_markets(self) -> Dict[str, ExchangeMarket]:
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

	def fetch_tickers(self, symbols: Optional[Iterable[str]] = None) -> Dict[str, Dict[str, Any]]:
		return self._client.fetch_tickers(list(symbols) if symbols else None)

	def fetch_ohlcv(
		self,
		symbol: str,
		timeframe: str,
		since: Optional[int] = None,
		limit: Optional[int] = None,
		params: Optional[Dict[str, Any]] = None,
	) -> list[list[Any]]:
		# Fix: Ensure params is always a dict, never None
		safe_params = params if params is not None else {}

		# Add retry logic with exponential backoff
		max_retries = 3
		for attempt in range(max_retries):
			try:
				return self._client.fetch_ohlcv(
					symbol,
					timeframe=timeframe,
					since=since,
					limit=limit,
					params=safe_params
				)
			except Exception as e:
				if attempt == max_retries - 1:
					# Last attempt failed, raise the error
					raise
				# Exponential backoff: 1s, 2s, 4s
				import time
				wait_time = 2 ** attempt
				print(f"⚠️  OHLCV fetch failed (attempt {attempt + 1}/{max_retries}), retrying in {wait_time}s: {e}")
				time.sleep(wait_time)

	def parse_exchange_timestamp(self, ms_timestamp: int) -> datetime:
		return datetime.fromtimestamp(ms_timestamp / 1_000, tz=timezone.utc)