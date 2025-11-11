"""
GraphQL Server Setup.

Creates FastAPI app with GraphQL endpoint.
"""

from __future__ import annotations

from typing import Optional
import structlog

logger = structlog.get_logger(__name__)

# Try to import FastAPI and Strawberry
try:
    from fastapi import FastAPI
    from strawberry.fastapi import GraphQLRouter
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    logger.warning("fastapi_not_available", message="FastAPI not installed. Install with: pip install fastapi strawberry-graphql[fastapi]")


def create_graphql_app(
    schema: Optional[Any] = None,
    path: str = "/graphql",
) -> FastAPI:
    """
    Create FastAPI app with GraphQL endpoint.
    
    Args:
        schema: GraphQL schema (creates default if None)
        path: GraphQL endpoint path
    
    Returns:
        FastAPI application
    """
    if not HAS_FASTAPI:
        raise ImportError(
            "FastAPI is not installed. Install with: pip install fastapi strawberry-graphql[fastapi]"
        )
    
    from .schema import create_schema
    
    if schema is None:
        schema = create_schema()
    
    app = FastAPI(title="Huracan Engine GraphQL API")
    
    graphql_app = GraphQLRouter(schema, path=path)
    app.include_router(graphql_app)
    
    logger.info("graphql_server_created", path=path)
    return app

