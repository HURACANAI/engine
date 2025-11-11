# Scalable Architecture Design - 400 Coin, 500 Trade Capacity

**Version:** 1.0  
**Date:** 2025-01-27  
**Status:** Design & Implementation

---

## 🎯 Executive Summary

This document defines the scalable architecture for the Huracan Trading Engine, designed to handle **400 coins** and **500 concurrent trades** with configuration-based throttling. The architecture is built for parallelism and controlled via configuration, allowing the same codebase to run at any scale from 20 coins to 400 coins.

**Core Principle:** *Build for parallelism, control it with config.*

---

## 📐 Architecture Overview

### Design Philosophy

1. **Parallelism First**: All components are designed for concurrent execution
2. **Configuration-Driven**: Scale controlled via config, not code changes
3. **Event-Driven**: Message bus architecture for decoupled, scalable data flow
4. **Sharded Processing**: Coin groups processed in parallel event loops
5. **Distributed Models**: Each coin has independent model slot with metadata
6. **Global Risk Control**: Centralized risk monitoring across all coins
7. **Multi-Exchange Ready**: Abstract exchange layer for Binance, OKX, Bybit

### System Layers

```
┌─────────────────────────────────────────────────────────────┐
│              OBSERVABILITY LAYER                             │
│  Prometheus Metrics, Grafana Dashboards, Health Monitoring  │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│              EXECUTION LAYER                                 │
│  Global Risk Controller, Order Router, Exchange Abstraction │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│              PROCESSING LAYER                                │
│  Event Loop Managers (50-100 coins per loop)                │
│  Message Bus (Redis Streams/Kafka)                          │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│              DATA LAYER                                      │
│  Partitioned Storage (/data/coin/YYYYMMDD.parquet)          │
│  Feature Builders (Async Queue)                             │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│              MODEL LAYER                                     │
│  Distributed Model Registry, Metadata Tracking              │
│  Retraining Jobs (Ray/Dask on RunPod)                       │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔧 Configuration Schema

### Engine Configuration

```yaml
engine:
  # Maximum capacity (hardware limits)
  max_coins: 400
  max_concurrent_trades: 500
  
  # Runtime throttling (soft limits)
  active_coins: 20  # Start small, scale up
  max_active_trades: 100  # Soft throttle
  
  # Event loop configuration
  coins_per_event_loop: 50  # 400 coins = 8 event loops
  event_loop_timeout_seconds: 300
  
  # Message bus configuration
  message_bus:
    type: "redis_streams"  # or "kafka", "nats"
    host: "localhost"
    port: 6379
    stream_prefix: "huracan"
    max_consumers_per_group: 10
  
  # Model management
  model_registry:
    storage_path: "/models/trained"
    metadata_db: "postgresql://localhost/huracan_models"
    retraining:
      enabled: true
      use_ray: true
      workers_per_coin: 1
      gpu_per_worker: 1
  
  # Risk management
  risk:
    global_max_exposure_pct: 200.0  # 2x leverage max
    per_coin_max_exposure_pct: 40.0
    per_sector_max_exposure_pct: 60.0
    per_exchange_max_exposure_pct: 150.0
    soft_throttle_threshold_pct: 80.0  # Throttle at 80% of limits
  
  # Cost model
  cost_model:
    real_time_updates: true
    update_interval_seconds: 60
    spread_source: "exchange_websocket"
    fee_source: "exchange_api"
    funding_source: "exchange_api"
    min_edge_after_cost_bps: 5.0
  
  # Observability
  observability:
    prometheus:
      enabled: true
      port: 9090
    grafana:
      enabled: true
      dashboard_path: "/dashboards"
    metrics:
      latency_percentiles: [50, 95, 99]
      trade_rate_window_seconds: 60
      health_check_interval_seconds: 30
```

### Scaling Phases

| Phase | Purpose | Scale | Config |
|-------|---------|-------|--------|
| **Build** | Design for 400-coin concurrency | 400 coins | `max_coins: 400, active_coins: 400` |
| **Calibration** | Test on 20-coin subset | 20 coins | `max_coins: 400, active_coins: 20` |
| **Scaling** | Gradually raise to 400 | 100→200→400 | `active_coins: 100/200/400` |
| **Profit** | Run 20-coin subset live | 20 coins | `max_coins: 400, active_coins: 20` |

---

## 🏗️ Core Components

### 1. Message Bus Architecture

**Purpose**: Decouple data pipelines, enable horizontal scaling

**Implementation**: Redis Streams (lightweight) or Kafka (high throughput)

**Streams**:
- `market_data:{coin}` - Raw market data per coin
- `features:{coin}` - Processed features per coin
- `signals:{coin}` - Trading signals per coin
- `orders:global` - All orders (for risk monitoring)
- `executions:global` - All executions (for P&L tracking)

**Benefits**:
- Independent coin pipelines
- Horizontal scaling via consumer groups
- Backpressure handling
- Fault tolerance (messages persist)

### 2. Event Loop Manager

**Purpose**: Process coins in parallel groups

**Design**:
- One event loop per 50-100 coins
- Each loop runs asyncio event loop
- Socket connections reused across tasks
- Adaptive pacing for order submission

**Implementation**:
```python
class EventLoopManager:
    def __init__(
        self,
        coins_per_loop: int = 50,
        max_loops: int = 10,
    ):
        self.coins_per_loop = coins_per_loop
        self.loops = []
        self.coin_assignments = {}  # coin -> loop_id
    
    async def process_coin_group(self, coins: List[str]):
        # Process 50-100 coins in parallel
        tasks = [self.process_coin(coin) for coin in coins]
        await asyncio.gather(*tasks)
```

### 3. Distributed Model Management

**Purpose**: Track models per coin with metadata

**Storage**:
- Model files: `/models/trained/{coin}/{version}.pkl`
- Metadata: PostgreSQL table `model_registry`

**Metadata Schema**:
```sql
CREATE TABLE model_registry (
    coin VARCHAR(20),
    version INTEGER,
    sharpe_ratio FLOAT,
    regime VARCHAR(20),
    last_retrain_date TIMESTAMP,
    is_active BOOLEAN,
    performance_metrics JSONB
);
```

**Features**:
- Version tracking
- Performance metrics (Sharpe, win rate, etc.)
- Regime-specific models
- Best-N models active per regime
- Distributed retraining with Ray/Dask

### 4. Global Risk Controller

**Purpose**: Monitor all trades across coins, sectors, exchanges

**Limits**:
- **Per-coin**: Max 40% exposure per coin
- **Per-sector**: Max 60% exposure per sector (DeFi, L1, etc.)
- **Per-exchange**: Max 150% exposure per exchange
- **Global**: Max 200% total leverage

**Features**:
- Real-time exposure tracking
- Soft throttling at 80% of limits
- Dynamic position sizing by volatility
- Circuit breakers for drawdowns

**Implementation**:
```python
class GlobalRiskController:
    def __init__(self, config: Dict):
        self.limits = RiskLimits.from_config(config)
        self.exposure_tracker = ExposureTracker()
        self.circuit_breakers = CircuitBreakers()
    
    async def check_trade(self, trade: TradeRequest) -> RiskDecision:
        # Check all limits
        if self.exposure_tracker.would_exceed_limit(trade):
            return RiskDecision.BLOCK
        
        # Check circuit breakers
        if self.circuit_breakers.is_triggered():
            return RiskDecision.THROTTLE
        
        return RiskDecision.APPROVE
```

### 5. Cost Model

**Purpose**: Real-time spread, fee, and funding data

**Data Sources**:
- **Spread**: WebSocket orderbook updates
- **Fees**: Exchange API (maker/taker rates)
- **Funding**: Exchange API (funding rates)

**Features**:
- Per-symbol cost tracking
- Edge-after-cost calculation
- Skip coins failing threshold
- Dynamic ranking by cost efficiency

**Implementation**:
```python
class RealTimeCostModel:
    def __init__(self, config: Dict):
        self.spread_tracker = SpreadTracker()
        self.fee_tracker = FeeTracker()
        self.funding_tracker = FundingTracker()
        self.update_interval = config['update_interval_seconds']
    
    async def update_costs(self):
        # Fetch latest costs from exchanges
        await asyncio.gather(
            self.spread_tracker.update(),
            self.fee_tracker.update(),
            self.funding_tracker.update(),
        )
    
    def calculate_edge_after_cost(
        self,
        symbol: str,
        edge_bps: float,
    ) -> float:
        spread = self.spread_tracker.get_spread(symbol)
        fee = self.fee_tracker.get_fee(symbol)
        funding = self.funding_tracker.get_funding(symbol)
        
        total_cost = spread + fee + funding
        return edge_bps - total_cost
```

### 6. Exchange Abstraction Layer

**Purpose**: Support multiple exchanges (Binance, OKX, Bybit)

**Interface**:
```python
class ExchangeInterface(ABC):
    @abstractmethod
    async def get_orderbook(self, symbol: str) -> OrderBook:
        pass
    
    @abstractmethod
    async def place_order(self, order: Order) -> OrderResponse:
        pass
    
    @abstractmethod
    async def get_fees(self, symbol: str) -> FeeStructure:
        pass
```

**Implementations**:
- `BinanceExchange`
- `OKXExchange`
- `BybitExchange`

**Features**:
- Unified API across exchanges
- Connection pooling
- Retry logic with exponential backoff
- Rate limit handling

### 7. Observability System

**Purpose**: Monitor system health, performance, P&L

**Metrics** (Prometheus):
- `huracan_trades_total` - Total trades executed
- `huracan_trades_per_second` - Trade rate
- `huracan_latency_seconds` - Latency percentiles (50, 95, 99)
- `huracan_hit_rate` - Win rate
- `huracan_cost_bps` - Cost per trade
- `huracan_errors_total` - Error count
- `huracan_active_positions` - Active positions
- `huracan_exposure_pct` - Portfolio exposure

**Dashboards** (Grafana):
- **System Health**: Latency, error rate, throughput
- **P&L Dashboard**: Top 10 contributors, laggards
- **Risk Dashboard**: Exposure by coin, sector, exchange
- **Cost Dashboard**: Spread, fees, funding costs

**Health Checks**:
- Database connectivity
- Message bus connectivity
- Exchange API connectivity
- Model availability
- Risk limits status

---

## 📊 Data Architecture

### Partitioned Storage

**Structure**: `/data/{coin}/{YYYYMMDD}.parquet`

**Benefits**:
- Fast queries (partition pruning)
- Easy cleanup (delete old partitions)
- Parallel processing (one partition per worker)

**Example**:
```
/data/
  BTC/
    20250127.parquet
    20250128.parquet
  ETH/
    20250127.parquet
    20250128.parquet
```

### Feature Builders

**Design**: Async queue with workers

**Flow**:
1. Raw data → Message bus
2. Feature builders consume from bus
3. Process features asynchronously
4. Store features in partitioned storage
5. Publish to signal stream

**Benefits**:
- Decoupled processing
- Horizontal scaling
- Backpressure handling

---

## 🚀 Deployment Architecture

### RunPod Configuration

**Workers**:
- **Data Workers**: 4-8 workers for data ingestion
- **Feature Workers**: 8-16 workers for feature building
- **Signal Workers**: 4-8 workers for signal generation
- **Execution Workers**: 2-4 workers for order execution
- **Model Workers**: GPU workers for retraining (Ray/Dask)

**Scaling**:
- Start with 1 worker per type
- Scale horizontally as needed
- Use RunPod autoscaling

### Database

**PostgreSQL**:
- Model registry
- Trade history
- Performance metrics
- Risk limits tracking

**Redis**:
- Message bus (Streams)
- Cache for frequently accessed data
- Rate limiting

---

## 📈 Performance Targets

### Latency
- **Signal Generation**: < 100ms per coin
- **Order Execution**: < 50ms per order
- **Risk Check**: < 10ms per trade
- **Cost Calculation**: < 5ms per symbol

### Throughput
- **Market Data**: 10,000 messages/second
- **Features**: 1,000 features/second
- **Signals**: 100 signals/second
- **Orders**: 50 orders/second

### Scalability
- **400 Coins**: 8 event loops (50 coins each)
- **500 Trades**: Soft throttle at 100, hard limit at 500
- **Horizontal Scaling**: Add workers as needed

---

## 🔄 Migration Path

### Phase 1: Foundation (Week 1)
1. ✅ Update configuration schema
2. ✅ Create message bus infrastructure
3. ✅ Build event loop manager
4. ✅ Implement global risk controller

### Phase 2: Data & Models (Week 2)
1. ✅ Partitioned storage structure
2. ✅ Distributed model registry
3. ✅ Feature builder queue
4. ✅ Cost model implementation

### Phase 3: Execution & Observability (Week 3)
1. ✅ Exchange abstraction layer
2. ✅ Observability system (Prometheus/Grafana)
3. ✅ Health checks
4. ✅ Performance monitoring

### Phase 4: Testing & Calibration (Week 4)
1. ✅ Test with 20 coins
2. ✅ Validate cost model
3. ✅ Tune risk limits
4. ✅ Performance optimization

### Phase 5: Scaling (Week 5+)
1. ✅ Scale to 100 coins
2. ✅ Scale to 200 coins
3. ✅ Scale to 400 coins
4. ✅ Production deployment

---

## 🎯 Success Metrics

### System Metrics
- **Uptime**: > 99.9%
- **Latency**: P99 < 100ms
- **Error Rate**: < 0.1%
- **Throughput**: Handle 400 coins concurrently

### Trading Metrics
- **Trade Rate**: 100 trades/day (20-coin mode)
- **Win Rate**: > 70% (scalp), > 95% (runner)
- **Net Profit**: £1-£3 per trade (scalp), £5-£20 (runner)
- **Cost Efficiency**: Edge-after-cost > 5 bps

---

## 📚 References

- [Redis Streams Documentation](https://redis.io/docs/data-types/streams/)
- [Asyncio Best Practices](https://docs.python.org/3/library/asyncio.html)
- [Ray Distributed Computing](https://docs.ray.io/)
- [Prometheus Metrics](https://prometheus.io/docs/concepts/metric_types/)

---

**Last Updated:** 2025-01-27  
**Maintained By:** Huracan Engine Architecture Team

