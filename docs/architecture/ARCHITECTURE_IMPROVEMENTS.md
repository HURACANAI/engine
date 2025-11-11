# Architecture Improvements - Implementation Plan

**Date:** 2025-11-11  
**Status:** Planning Phase

---

## 🎯 Overview

This document outlines the implementation plan for 4 major architectural improvements:

1. **Async/Await Migration** - Improve I/O performance
2. **Redis Caching Layer** - Speed up data access
3. **GraphQL API** - Flexible data querying
4. **Microservices Architecture** - Better scalability

---

## 1. Async/Await Migration

### Goal
Migrate all I/O operations to async/await to improve performance and scalability.

### Benefits
- **Non-blocking I/O**: Handle multiple operations concurrently
- **Better resource utilization**: Serve more requests with same resources
- **Improved responsiveness**: System remains responsive during I/O operations

### Implementation Plan

#### Phase 1: Database Operations
**Files to migrate:**
- `src/cloud/training/brain/brain_library.py` - PostgreSQL operations
- `src/cloud/training/database/pool.py` - Connection pooling
- All database query operations

**Changes:**
- Replace `psycopg2` with `asyncpg`
- Convert all database methods to `async def`
- Update connection pool to async pool
- Add async context managers

**Example:**
```python
# Before
def get_prices(self, symbol: str) -> pd.DataFrame:
    conn = self.pool.getconn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM prices WHERE symbol = %s", (symbol,))
    result = cur.fetchall()
    return pd.DataFrame(result)

# After
async def get_prices(self, symbol: str) -> pd.DataFrame:
    async with self.pool.acquire() as conn:
        result = await conn.fetch("SELECT * FROM prices WHERE symbol = $1", symbol)
        return pd.DataFrame(result)
```

#### Phase 2: HTTP/API Calls
**Files to migrate:**
- `src/cloud/training/analysis/fear_greed_index.py` - API calls
- `src/cloud/training/monitoring/telegram_command_handler.py` - Telegram API
- `src/cloud/training/integrations/dropbox_sync.py` - Dropbox API
- All exchange API calls

**Changes:**
- Replace `requests` with `aiohttp` or `httpx`
- Convert all HTTP methods to async
- Add connection pooling for HTTP clients

**Example:**
```python
# Before
def get_fear_greed_index(self) -> dict:
    response = requests.get(self.api_url, timeout=10)
    return response.json()

# After
async def get_fear_greed_index(self) -> dict:
    async with self.session.get(self.api_url, timeout=10) as response:
        return await response.json()
```

#### Phase 3: File I/O
**Files to migrate:**
- All file read/write operations
- Model saving/loading
- Log file operations

**Changes:**
- Use `aiofiles` for async file operations
- Convert file I/O methods to async

**Example:**
```python
# Before
def save_model(self, model, path: str):
    with open(path, 'wb') as f:
        pickle.dump(model, f)

# After
async def save_model(self, model, path: str):
    async with aiofiles.open(path, 'wb') as f:
        await f.write(pickle.dumps(model))
```

### Migration Strategy
1. **Start with new code**: All new features use async
2. **Migrate high-traffic paths first**: Database operations, API calls
3. **Gradual migration**: Convert one module at a time
4. **Maintain compatibility**: Keep sync wrappers where needed

---

## 2. Redis Caching Layer

### Goal
Add Redis caching for frequently accessed data to reduce database load and improve response times.

### Benefits
- **Faster data access**: In-memory cache is much faster than database
- **Reduced database load**: Fewer queries to PostgreSQL
- **Better scalability**: Handle more requests with cached data

### Implementation Plan

#### Phase 1: Redis Setup
**New files:**
- `src/cloud/training/cache/redis_client.py` - Redis connection manager
- `src/cloud/training/cache/cache_manager.py` - Cache abstraction layer

**Configuration:**
- Add Redis connection settings to config
- Set up connection pooling
- Configure TTL (time-to-live) defaults

#### Phase 2: Cache Frequently Accessed Data

**Data to cache:**
1. **Market Data** (TTL: 1-5 minutes)
   - Current prices
   - Recent candles
   - Order book snapshots

2. **Features** (TTL: 5-15 minutes)
   - Calculated features
   - Technical indicators
   - Regime classifications

3. **Model Predictions** (TTL: 1-10 minutes)
   - Engine signals
   - Confidence scores
   - Ensemble predictions

4. **Performance Metrics** (TTL: 1 hour)
   - Recent performance stats
   - Win rates
   - Sharpe ratios

**Example:**
```python
class CacheManager:
    async def get_prices(self, symbol: str) -> Optional[pd.DataFrame]:
        # Try cache first
        cached = await self.redis.get(f"prices:{symbol}")
        if cached:
            return pd.read_json(cached)
        
        # Cache miss - fetch from database
        data = await self.brain_library.get_prices(symbol)
        
        # Store in cache
        await self.redis.setex(
            f"prices:{symbol}",
            300,  # 5 minutes TTL
            data.to_json()
        )
        return data
```

#### Phase 3: Cache Invalidation
**Strategies:**
- **Time-based**: Automatic expiration (TTL)
- **Event-based**: Invalidate on data updates
- **Manual**: Clear cache when needed

**Example:**
```python
async def invalidate_price_cache(self, symbol: str):
    await self.redis.delete(f"prices:{symbol}")
    await self.redis.delete(f"features:{symbol}")
```

### Cache Keys Structure
```
prices:{symbol}                    # Price data
features:{symbol}:{timestamp}      # Calculated features
predictions:{symbol}:{engine}      # Model predictions
metrics:{symbol}:{period}          # Performance metrics
regime:{symbol}                    # Current regime
```

---

## 3. GraphQL API

### Goal
Add GraphQL endpoint for flexible, efficient data querying.

### Benefits
- **Flexible queries**: Clients request only needed data
- **Single endpoint**: One API for all data needs
- **Type safety**: Strong typing with GraphQL schema
- **Real-time subscriptions**: Support for live updates

### Implementation Plan

#### Phase 1: GraphQL Server Setup
**New files:**
- `src/cloud/training/api/graphql/schema.py` - GraphQL schema definition
- `src/cloud/training/api/graphql/resolvers.py` - Query/mutation resolvers
- `src/cloud/training/api/graphql/server.py` - GraphQL server setup

**Technology:**
- Use **Strawberry** (modern, type-safe GraphQL library for Python)
- Or **Ariadne** (schema-first approach)

#### Phase 2: Schema Definition

**Types to define:**
```graphql
type Symbol {
  id: ID!
  name: String!
  currentPrice: Float!
  priceHistory: [PricePoint!]!
  signals: [Signal!]!
  performance: PerformanceMetrics!
}

type Signal {
  id: ID!
  symbol: Symbol!
  direction: Direction!
  confidence: Float!
  engine: String!
  timestamp: DateTime!
}

type PerformanceMetrics {
  sharpeRatio: Float!
  winRate: Float!
  totalTrades: Int!
  totalPnL: Float!
  maxDrawdown: Float!
}

type Query {
  symbol(name: String!): Symbol
  signals(symbol: String, limit: Int): [Signal!]!
  performance(symbol: String!): PerformanceMetrics!
  engines: [Engine!]!
}

type Subscription {
  priceUpdate(symbol: String!): PricePoint!
  signalUpdate(symbol: String!): Signal!
}
```

#### Phase 3: Resolvers
**Implement resolvers for:**
- Query resolvers (data fetching)
- Mutation resolvers (data updates)
- Subscription resolvers (real-time updates)

**Example:**
```python
@strawberry.type
class Query:
    @strawberry.field
    async def symbol(self, name: str) -> Symbol:
        data = await cache_manager.get_prices(name)
        return Symbol.from_data(data)
    
    @strawberry.field
    async def signals(self, symbol: Optional[str] = None, limit: int = 100) -> List[Signal]:
        signals = await brain_library.get_signals(symbol=symbol, limit=limit)
        return [Signal.from_db(s) for s in signals]
```

#### Phase 4: Integration
- Add FastAPI endpoint for GraphQL
- Set up GraphQL playground for testing
- Add authentication/authorization
- Add rate limiting

---

## 4. Microservices Architecture

### Goal
Split monolithic application into microservices for better scalability and maintainability.

### Benefits
- **Independent scaling**: Scale services based on load
- **Technology flexibility**: Use best tool for each service
- **Fault isolation**: One service failure doesn't crash everything
- **Team autonomy**: Teams can work independently

### Architecture Design

#### Proposed Services

1. **Data Service** 📊
   - **Responsibility**: Market data collection, storage, retrieval
   - **Technologies**: Python, PostgreSQL, Redis
   - **APIs**: REST, GraphQL
   - **Dependencies**: Exchange APIs, Database

2. **Feature Service** 🔧
   - **Responsibility**: Feature calculation, technical indicators
   - **Technologies**: Python, NumPy, Pandas
   - **APIs**: gRPC, REST
   - **Dependencies**: Data Service

3. **Training Service** 🎓
   - **Responsibility**: Model training, backtesting, validation
   - **Technologies**: Python, PyTorch, Ray
   - **APIs**: gRPC, REST
   - **Dependencies**: Data Service, Feature Service

4. **Prediction Service** 🔮
   - **Responsibility**: Real-time predictions, signal generation
   - **Technologies**: Python, FastAPI
   - **APIs**: gRPC, REST, GraphQL
   - **Dependencies**: Training Service, Feature Service

5. **Trading Service** 💰
   - **Responsibility**: Order execution, position management
   - **Technologies**: Python, FastAPI
   - **APIs**: gRPC, REST
   - **Dependencies**: Prediction Service, Exchange APIs

6. **Monitoring Service** 📈
   - **Responsibility**: Health checks, metrics, alerts
   - **Technologies**: Python, Prometheus, Grafana
   - **APIs**: REST, WebSocket
   - **Dependencies**: All services

#### Service Communication

**Options:**
1. **gRPC** (Recommended for internal services)
   - High performance
   - Strong typing
   - Streaming support

2. **Message Queue** (RabbitMQ, Kafka)
   - Async communication
   - Event-driven architecture
   - Decoupling

3. **REST** (For external APIs)
   - Simple, widely understood
   - Good for public APIs

#### Migration Strategy

**Phase 1: Extract Data Service**
- Move data collection to separate service
- Set up service communication
- Keep existing code working

**Phase 2: Extract Training Service**
- Move training pipeline to separate service
- Use message queue for job scheduling
- Maintain API compatibility

**Phase 3: Extract Prediction Service**
- Move real-time prediction logic
- Set up gRPC for fast communication
- Add caching layer

**Phase 4: Extract Trading Service**
- Move order execution logic
- Add safety checks and validation
- Implement circuit breakers

**Phase 5: Extract Monitoring Service**
- Centralize all monitoring
- Add distributed tracing
- Set up alerting

### Service Discovery & Configuration

**Tools:**
- **Consul** or **etcd** for service discovery
- **Kubernetes** for orchestration (if using containers)
- **Docker** for containerization

### Data Consistency

**Strategies:**
- **Event Sourcing**: Store all events, rebuild state
- **Saga Pattern**: Distributed transactions
- **Eventual Consistency**: Accept temporary inconsistencies

---

## Implementation Timeline

### Phase 1: Foundation (Weeks 1-2)
- ✅ Set up Redis infrastructure
- ✅ Create async database pool
- ✅ Set up GraphQL server skeleton

### Phase 2: Async Migration (Weeks 3-4)
- ✅ Migrate database operations
- ✅ Migrate HTTP calls
- ✅ Migrate file I/O

### Phase 3: Caching (Weeks 5-6)
- ✅ Implement Redis caching layer
- ✅ Add cache invalidation
- ✅ Optimize cache strategies

### Phase 4: GraphQL API (Weeks 7-8)
- ✅ Complete GraphQL schema
- ✅ Implement all resolvers
- ✅ Add subscriptions

### Phase 5: Microservices (Weeks 9-12)
- ✅ Extract Data Service
- ✅ Extract Training Service
- ✅ Extract Prediction Service
- ✅ Extract Trading Service
- ✅ Extract Monitoring Service

---

## Dependencies

### New Python Packages
```python
# Async
asyncpg>=0.29.0          # Async PostgreSQL
aiohttp>=3.9.0           # Async HTTP client
aiofiles>=23.2.0         # Async file I/O
httpx>=0.25.0            # Alternative async HTTP client

# Redis
redis>=5.0.0             # Redis client
hiredis>=2.2.0           # Fast Redis parser

# GraphQL
strawberry-graphql>=0.215.0  # GraphQL framework
graphql-core>=3.2.0          # GraphQL core

# Microservices
grpcio>=1.60.0           # gRPC
grpcio-tools>=1.60.0     # gRPC code generation
protobuf>=4.25.0         # Protocol buffers

# Message Queue (optional)
celery>=5.3.0            # Distributed task queue
rabbitmq>=0.2.0          # RabbitMQ client
```

### Infrastructure
- **Redis Server** (for caching)
- **gRPC** (for service communication)
- **Message Queue** (RabbitMQ or Kafka, optional)
- **Service Discovery** (Consul or etcd, optional)
- **Container Orchestration** (Kubernetes, optional)

---

## Testing Strategy

### Async Code
- Use `pytest-asyncio` for async tests
- Test concurrent operations
- Test error handling

### Caching
- Test cache hits/misses
- Test cache invalidation
- Test cache expiration

### GraphQL
- Test queries and mutations
- Test subscriptions
- Test error handling
- Use GraphQL playground for manual testing

### Microservices
- Integration tests between services
- Contract testing (Pact)
- Load testing
- Chaos engineering

---

## Monitoring & Observability

### Metrics
- Request latency
- Cache hit rates
- Service health
- Error rates

### Logging
- Distributed tracing (OpenTelemetry)
- Structured logging
- Correlation IDs

### Alerting
- Service downtime
- High error rates
- Cache miss rates
- Performance degradation

---

## Documentation

### API Documentation
- GraphQL schema documentation
- gRPC service definitions
- REST API documentation

### Architecture Diagrams
- Service architecture
- Data flow diagrams
- Deployment diagrams

### Runbooks
- Deployment procedures
- Troubleshooting guides
- Scaling procedures

---

## Risk Mitigation

### Risks
1. **Breaking changes** during migration
2. **Performance regression** with async
3. **Service communication failures**
4. **Data consistency issues**

### Mitigation
1. **Gradual migration** with feature flags
2. **Comprehensive testing** before deployment
3. **Circuit breakers** for service calls
4. **Monitoring and alerting** for early detection

---

## Success Metrics

### Performance
- **Response time**: < 100ms for cached data
- **Throughput**: Handle 10x more requests
- **Database load**: Reduce by 70%

### Reliability
- **Uptime**: 99.9%
- **Error rate**: < 0.1%
- **Service recovery**: < 5 minutes

### Developer Experience
- **API flexibility**: GraphQL reduces over-fetching
- **Deployment speed**: Independent service deployments
- **Debugging**: Better observability

---

**Status:** Planning Complete - Ready for Implementation

**Next Steps:**
1. Review and approve plan
2. Set up development environment
3. Begin Phase 1 implementation

