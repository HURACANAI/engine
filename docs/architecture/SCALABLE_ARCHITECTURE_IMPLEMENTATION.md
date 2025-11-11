# Scalable Architecture Implementation Guide

**Version:** 1.0  
**Date:** 2025-01-27  
**Status:** Implementation Complete (Phase 1)

---

## 🎯 Overview

This document provides implementation details and usage examples for the scalable architecture components. The architecture is designed to handle **400 coins** and **500 concurrent trades** with configuration-based throttling.

---

## 📦 Components Implemented

### 1. Message Bus (`infrastructure/message_bus.py`)

**Purpose**: Decouple data pipelines, enable horizontal scaling

**Features**:
- Redis Streams implementation
- In-memory fallback for testing
- Consumer groups for horizontal scaling
- Message persistence

**Usage**:
```python
from src.cloud.training.infrastructure import create_message_bus, Message, StreamType

# Create message bus from config
config = {
    "message_bus": {
        "type": "redis_streams",
        "host": "localhost",
        "port": 6379,
        "stream_prefix": "huracan",
    }
}
bus = create_message_bus(config)
await bus.connect()

# Publish message
message = Message(
    stream_type=StreamType.MARKET_DATA,
    coin="BTC",
    data={"price": 50000.0, "volume": 1000.0},
    timestamp=time.time(),
)
message_id = await bus.publish(message)

# Subscribe to stream
async for message in bus.subscribe(
    stream_type=StreamType.MARKET_DATA,
    coin="BTC",
    consumer_group="market_data_workers",
):
    # Process message
    print(f"Received: {message.data}")
    await bus.ack_message(
        stream_type=StreamType.MARKET_DATA,
        coin="BTC",
        group_name="market_data_workers",
        message_id=message.message_id,
    )
```

### 2. Event Loop Manager (`infrastructure/event_loop_manager.py`)

**Purpose**: Process coins in parallel groups

**Features**:
- One event loop per 50-100 coins
- Automatic coin assignment
- Health monitoring
- Graceful shutdown

**Usage**:
```python
from src.cloud.training.infrastructure import EventLoopManager

# Create event loop manager
manager = EventLoopManager(
    coins_per_loop=50,
    max_loops=10,
    timeout_seconds=300,
)

# Assign coins
coins = ["BTC", "ETH", "SOL", ...]  # 400 coins
manager.assign_coins(coins)

# Register processors
async def process_coin(coin: str):
    # Process coin data
    print(f"Processing {coin}")
    await asyncio.sleep(1)

for coin in coins:
    manager.register_processor(coin, process_coin)

# Start event loops
await manager.start()

# Get metrics
metrics = manager.get_metrics()
for loop_id, metric in metrics.items():
    print(f"Loop {loop_id}: {metric.coins_assigned} coins, {metric.messages_processed} messages")

# Stop event loops
await manager.stop(timeout=10.0)
```

### 3. Global Risk Controller (`risk/global_risk_controller.py`)

**Purpose**: Monitor all trades across coins, sectors, exchanges

**Features**:
- Per-coin exposure limits
- Per-sector exposure limits
- Per-exchange exposure limits
- Global exposure limits
- Soft throttling at 80% of limits
- Circuit breakers

**Usage**:
```python
from src.cloud.training.risk.global_risk_controller import (
    GlobalRiskController,
    RiskLimits,
    TradeRequest,
)

# Create risk controller
limits = RiskLimits(
    global_max_exposure_pct=200.0,
    per_coin_max_exposure_pct=40.0,
    per_sector_max_exposure_pct=60.0,
    per_exchange_max_exposure_pct=150.0,
    soft_throttle_threshold_pct=80.0,
    max_concurrent_trades=500,
    max_active_trades=100,
)
risk_controller = GlobalRiskController(limits)

# Set coin sectors
risk_controller.set_coin_sector("BTC", "L1")
risk_controller.set_coin_sector("ETH", "L1")
risk_controller.set_coin_sector("UNI", "DeFi")

# Check trade
trade = TradeRequest(
    coin="BTC",
    direction="long",
    size_usd=1000.0,
    exchange="binance",
    sector="L1",
)
result = risk_controller.check_trade(trade)

if result.decision == RiskDecision.APPROVE:
    # Register trade
    risk_controller.register_trade("trade_123", trade)
    print("Trade approved")
elif result.decision == RiskDecision.THROTTLE:
    print("Trade throttled")
elif result.decision == RiskDecision.BLOCK:
    print(f"Trade blocked: {result.reason}")

# Get exposure summary
summary = risk_controller.get_exposure_summary()
print(f"Global exposure: {summary['global']['total_exposure_pct']}%")
print(f"Active trades: {summary['active_trades']}")
```

### 4. Real-Time Cost Model (`costs/real_time_cost_model.py`)

**Purpose**: Track real-time spread, fee, and funding data

**Features**:
- Real-time spread tracking
- Fee tracking (maker/taker)
- Funding rate tracking
- Edge-after-cost calculation
- Cost efficiency ranking

**Usage**:
```python
from src.cloud.training.costs import RealTimeCostModel

# Create cost model
cost_model = RealTimeCostModel(
    update_interval_seconds=60,
    min_edge_after_cost_bps=5.0,
    default_maker_fee_bps=2.0,
    default_taker_fee_bps=5.0,
)

# Start cost model
await cost_model.start()

# Update costs
cost_model.update_spread("BTC", bid=50000.0, ask=50010.0)
cost_model.update_fees("BTC", maker_bps=2.0, taker_bps=5.0)
cost_model.update_funding_rate("BTC", funding_rate_bps=10.0)

# Calculate edge after cost
edge_bps = 20.0  # 20 bps edge before cost
edge_after_cost = cost_model.calculate_edge_after_cost(
    symbol="BTC",
    edge_bps=edge_bps,
    use_maker=True,
    include_funding=True,
)
print(f"Edge after cost: {edge_after_cost} bps")

# Check if should trade
if cost_model.should_trade("BTC", edge_bps=20.0, use_maker=True):
    print("Trade passes cost threshold")
else:
    print("Trade fails cost threshold")

# Rank symbols by cost efficiency
symbols = ["BTC", "ETH", "SOL"]
edges = {"BTC": 20.0, "ETH": 15.0, "SOL": 25.0}
rankings = cost_model.rank_symbols_by_cost_efficiency(
    symbols=symbols,
    edges_bps=edges,
    use_maker=True,
)
for ranking in rankings:
    print(f"{ranking.symbol}: efficiency={ranking.cost_efficiency:.2f}, rank={ranking.rank}")

# Stop cost model
await cost_model.stop()
```

### 5. Model Registry (`models/model_registry.py`)

**Purpose**: Track models per coin with metadata

**Features**:
- Version tracking
- Performance metrics (Sharpe, win rate)
- Regime-specific models
- Active/inactive status
- PostgreSQL or file-based storage

**Usage**:
```python
from src.cloud.training.models.model_registry import ModelRegistry, ModelStatus

# Create model registry
registry = ModelRegistry(
    storage_path="/models/trained",
    metadata_db="postgresql://user:pass@localhost/huracan_models",
)

# Register model
metadata = registry.register_model(
    coin="BTC",
    version=1,
    sharpe_ratio=2.5,
    win_rate=0.75,
    regime="trending",
    performance_metrics={"avg_return": 0.02, "max_drawdown": 0.05},
    model_path="/models/trained/BTC/model_v1.pkl",
    is_active=True,
)

# Get model metadata
metadata = registry.get_model_metadata("BTC", version=1)
print(f"Sharpe: {metadata.sharpe_ratio}, Win rate: {metadata.win_rate}")

# Get active models
active_models = registry.get_active_models(coin="BTC", regime="trending", limit=5)
for model in active_models:
    print(f"Version {model.version}: Sharpe={model.sharpe_ratio}")

# Set model active
registry.set_model_active("BTC", version=1, is_active=True)

# Get model path
model_path = registry.get_model_path("BTC", version=1)
print(f"Model path: {model_path}")
```

---

## 🔄 Integration Example

### Complete Pipeline Integration

```python
import asyncio
from src.cloud.training.infrastructure import create_message_bus, EventLoopManager, Message, StreamType
from src.cloud.training.risk.global_risk_controller import GlobalRiskController, RiskLimits, TradeRequest
from src.cloud.training.costs import RealTimeCostModel
from src.cloud.training.models.model_registry import ModelRegistry

async def main():
    # Load configuration
    config = load_config("config/base.yaml")
    
    # Initialize components
    bus = create_message_bus(config["engine"]["message_bus"])
    await bus.connect()
    
    event_loop_manager = EventLoopManager(
        coins_per_loop=config["engine"]["coins_per_event_loop"],
        max_loops=10,
    )
    
    risk_controller = GlobalRiskController(
        RiskLimits(**config["engine"]["risk"])
    )
    
    cost_model = RealTimeCostModel(
        update_interval_seconds=config["engine"]["cost_model"]["update_interval_seconds"],
        min_edge_after_cost_bps=config["engine"]["cost_model"]["min_edge_after_cost_bps"],
    )
    await cost_model.start()
    
    model_registry = ModelRegistry(
        storage_path=config["engine"]["model_registry"]["storage_path"],
        metadata_db=config["engine"]["model_registry"]["metadata_db"],
    )
    
    # Get active coins from config
    active_coins = get_active_coins(config["engine"]["active_coins"])
    
    # Assign coins to event loops
    event_loop_manager.assign_coins(active_coins)
    
    # Process coins
    async def process_coin(coin: str):
        # Subscribe to market data
        async for message in bus.subscribe(
            stream_type=StreamType.MARKET_DATA,
            coin=coin,
            consumer_group="market_data_workers",
        ):
            # Get model
            model_metadata = model_registry.get_model_metadata(coin)
            if not model_metadata or not model_metadata.is_active:
                continue
            
            # Generate signal (simplified)
            signal = generate_signal(message.data, model_metadata)
            
            # Check cost
            if not cost_model.should_trade(coin, signal.edge_bps, use_maker=True):
                continue
            
            # Check risk
            trade = TradeRequest(
                coin=coin,
                direction=signal.direction,
                size_usd=signal.size_usd,
                exchange=signal.exchange,
                sector=get_sector(coin),
            )
            risk_result = risk_controller.check_trade(trade)
            
            if risk_result.decision == RiskDecision.APPROVE:
                # Execute trade
                trade_id = execute_trade(trade)
                risk_controller.register_trade(trade_id, trade)
                
                # Publish execution
                await bus.publish(Message(
                    stream_type=StreamType.EXECUTIONS,
                    coin=coin,
                    data={"trade_id": trade_id, "trade": trade},
                    timestamp=time.time(),
                ))
    
    # Register processors
    for coin in active_coins:
        event_loop_manager.register_processor(coin, process_coin)
    
    # Start event loops
    await event_loop_manager.start()
    
    # Run until interrupted
    try:
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        pass
    finally:
        # Cleanup
        await event_loop_manager.stop()
        await cost_model.stop()
        await bus.disconnect()

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 📊 Configuration

### Example Configuration

```yaml
engine:
  max_coins: 400
  max_concurrent_trades: 500
  active_coins: 20  # Start small, scale up
  max_active_trades: 100
  coins_per_event_loop: 50
  
  message_bus:
    type: "redis_streams"
    host: "localhost"
    port: 6379
    stream_prefix: "huracan"
  
  risk:
    global_max_exposure_pct: 200.0
    per_coin_max_exposure_pct: 40.0
    per_sector_max_exposure_pct: 60.0
    per_exchange_max_exposure_pct: 150.0
    soft_throttle_threshold_pct: 80.0
    max_concurrent_trades: 500
    max_active_trades: 100
  
  cost_model:
    update_interval_seconds: 60
    min_edge_after_cost_bps: 5.0
  
  model_registry:
    storage_path: "/models/trained"
    metadata_db: "postgresql://user:pass@localhost/huracan_models"
```

---

## 🚀 Deployment

### RunPod Configuration

1. **Data Workers**: 4-8 workers for data ingestion
2. **Feature Workers**: 8-16 workers for feature building
3. **Signal Workers**: 4-8 workers for signal generation
4. **Execution Workers**: 2-4 workers for order execution
5. **Model Workers**: GPU workers for retraining (Ray/Dask)

### Scaling Phases

1. **Calibration (20 coins)**: Test with `active_coins: 20`
2. **Scaling (100 coins)**: Scale to `active_coins: 100`
3. **Scaling (200 coins)**: Scale to `active_coins: 200`
4. **Full Scale (400 coins)**: Scale to `active_coins: 400`

---

## 📈 Performance Targets

### Latency
- Signal Generation: < 100ms per coin
- Order Execution: < 50ms per order
- Risk Check: < 10ms per trade
- Cost Calculation: < 5ms per symbol

### Throughput
- Market Data: 10,000 messages/second
- Features: 1,000 features/second
- Signals: 100 signals/second
- Orders: 50 orders/second

---

## 🔍 Monitoring

### Metrics to Monitor

1. **Event Loop Metrics**: Coins per loop, messages processed, errors
2. **Risk Metrics**: Exposure by coin, sector, exchange
3. **Cost Metrics**: Spread, fees, funding rates
4. **Model Metrics**: Sharpe ratio, win rate, active models
5. **System Metrics**: Latency, throughput, error rate

### Health Checks

1. Database connectivity
2. Message bus connectivity
3. Exchange API connectivity
4. Model availability
5. Risk limits status

---

## 📚 Next Steps

1. **Exchange Abstraction Layer**: Implement multi-exchange support
2. **Observability System**: Add Prometheus metrics and Grafana dashboards
3. **Partitioned Storage**: Update data architecture for `/data/coin/YYYYMMDD.parquet`
4. **Distributed Retraining**: Implement Ray/Dask retraining jobs
5. **Testing**: Add comprehensive tests for all components

---

**Last Updated:** 2025-01-27  
**Maintained By:** Huracan Engine Architecture Team

