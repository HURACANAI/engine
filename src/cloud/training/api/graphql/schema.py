"""
GraphQL Schema Definition.

Defines types, queries, and subscriptions for trading data.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional
import structlog

logger = structlog.get_logger(__name__)

# Try to import Strawberry, but handle gracefully if not available
try:
    import strawberry
    from strawberry.fastapi import GraphQLRouter
    HAS_STRAWBERRY = True
except ImportError:
    HAS_STRAWBERRY = False
    logger.warning("strawberry_not_available", message="Strawberry not installed. Install with: pip install strawberry-graphql[fastapi]")


if HAS_STRAWBERRY:
    @strawberry.type
    class PricePoint:
        """Price data point."""
        timestamp: datetime
        open: float
        high: float
        low: float
        close: float
        volume: float
    
    @strawberry.type
    class Signal:
        """Trading signal."""
        id: str
        symbol: str
        direction: str  # "buy", "sell", "hold"
        confidence: float
        engine: str
        timestamp: datetime
        price: Optional[float] = None
    
    @strawberry.type
    class PerformanceMetrics:
        """Performance metrics."""
        sharpe_ratio: float
        win_rate: float
        total_trades: int
        total_pnl: float
        max_drawdown: float
        avg_return: float
    
    @strawberry.type
    class Engine:
        """Trading engine information."""
        name: str
        technique: str
        performance: float
        win_rate: float
        total_signals: int
    
    @strawberry.type
    class Symbol:
        """Trading symbol with data."""
        id: str
        name: str
        current_price: Optional[float] = None
        price_history: List[PricePoint] = strawberry.field(default_factory=list)
        signals: List[Signal] = strawberry.field(default_factory=list)
        performance: Optional[PerformanceMetrics] = None
    
    @strawberry.type
    class Query:
        """GraphQL queries."""
        
        @strawberry.field
        async def symbol(self, name: str) -> Optional[Symbol]:
            """Get symbol data."""
            # TODO: Implement with actual data fetching
            logger.debug("graphql_query_symbol", name=name)
            return None
        
        @strawberry.field
        async def signals(
            self,
            symbol: Optional[str] = None,
            limit: int = 100
        ) -> List[Signal]:
            """Get trading signals."""
            # TODO: Implement with actual data fetching
            logger.debug("graphql_query_signals", symbol=symbol, limit=limit)
            return []
        
        @strawberry.field
        async def performance(self, symbol: str) -> Optional[PerformanceMetrics]:
            """Get performance metrics."""
            # TODO: Implement with actual data fetching
            logger.debug("graphql_query_performance", symbol=symbol)
            return None
        
        @strawberry.field
        async def engines(self) -> List[Engine]:
            """Get all trading engines."""
            # TODO: Implement with actual data fetching
            logger.debug("graphql_query_engines")
            return []
    
    def create_schema():
        """Create GraphQL schema."""
        if not HAS_STRAWBERRY:
            raise ImportError(
                "Strawberry is not installed. Install with: pip install strawberry-graphql[fastapi]"
            )
        return strawberry.Schema(query=Query)
else:
    def create_schema():
        """Create GraphQL schema (stub if Strawberry not available)."""
        raise ImportError(
            "Strawberry is not installed. Install with: pip install strawberry-graphql[fastapi]"
        )

