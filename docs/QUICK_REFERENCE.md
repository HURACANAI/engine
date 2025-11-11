# Quick Reference Guide

## Core Components

### Engine Interface

```python
from src.shared.engines import BaseEngine, EngineInput, EngineOutput, EngineRegistry

class MyEngine(BaseEngine):
    def infer(self, input_data: EngineInput) -> EngineOutput:
        return EngineOutput(
            direction=Direction.BUY,
            edge_bps_before_costs=10.0,
            confidence_0_1=0.8,
            horizon_minutes=60,
            metadata={},
        )

registry = EngineRegistry()
registry.register(MyEngine())
```

### Shared Feature Builder

```python
from src.shared.features import FeatureBuilder, FeatureRecipe

builder = FeatureBuilder(recipe=FeatureRecipe())
features = builder.build_features(candles_df, symbol="BTCUSDT")
```

### Cost Calculator

```python
from src.shared.costs import CostCalculator, CostModel

calculator = CostCalculator()
cost_model = CostModel(symbol="BTCUSDT", taker_fee_bps=4.0, ...)
calculator.register_cost_model(cost_model)
net_edge = calculator.calculate_net_edge("BTCUSDT", edge_bps_before_costs=10.0, ...)
should_trade = calculator.should_trade("BTCUSDT", edge_bps_before_costs=10.0, net_edge_floor_bps=3.0, ...)
```

### Regime Classifier

```python
from src.shared.regime import RegimeClassifier, Regime

classifier = RegimeClassifier()
regime_classification = classifier.classify(candles_df, symbol="BTCUSDT")
supported_engines = classifier.filter_engines_by_regime(engines, regime_classification.regime)
```

### Meta Combiner

```python
from src.shared.meta import MetaCombiner

combiner = MetaCombiner(symbol="BTCUSDT", ema_alpha=0.1)
combiner.update_weights("engine_1", accuracy=0.6, net_edge_bps=5.0, ...)
meta_output = combiner.combine(engine_outputs, regime="TREND")
```

### Champion Manager

```python
from src.shared.champion import ChampionManager

manager = ChampionManager(base_path="champion", s3_bucket="huracan")
manager.save_champion(symbol="BTCUSDT", model_id="model_123", s3_path="s3://huracan/...")
champion = manager.load_champion("BTCUSDT")
```

### S3 Client

```python
from src.shared.storage import S3Client

client = S3Client(bucket="huracan", access_key="...", secret_key="...")
client.put_file("local/model.bin", "models/BTCUSDT/20250101_020000Z/model.bin")
url = client.get_signed_url("models/BTCUSDT/20250101_020000Z/model.bin")
```

### Database Client

```python
from src.shared.database import DatabaseClient, ModelRecord, ModelMetrics

client = DatabaseClient(connection_string="...")
client.save_model(model_record)
client.save_metrics(model_metrics)
```

### Telegram Control

```python
from src.shared.telegram import SymbolsSelector

selector = SymbolsSelector(selector_file="symbols_selector.json")
symbols = selector.load_symbols()
selector.save_symbols(["BTCUSDT", "ETHUSDT"], source="telegram")
```

### Daily Summary

```python
from src.shared.summary import DailySummaryGenerator

generator = DailySummaryGenerator(base_path="summaries/daily")
summary = generator.generate_summary(date="2025-01-01", results=results, meta_weights=meta_weights)
generator.save_summary(summary)
```

## Training Pipeline

### Hybrid Training Scheduler

```python
from src.cloud.training.pipelines.scheduler import HybridTrainingScheduler, SchedulerConfig, TrainingMode

config = SchedulerConfig(mode=TrainingMode.HYBRID, max_concurrent=12, timeout_minutes=45)
scheduler = HybridTrainingScheduler(config=config, train_func=train_symbol)
results = scheduler.schedule_symbols(symbols)
```

### Integrated Training Pipeline

```python
from src.cloud.training.pipelines.integrated_training_pipeline import IntegratedTrainingPipeline

pipeline = IntegratedTrainingPipeline(
    engine_registry=engine_registry,
    feature_builder=feature_builder,
    cost_calculator=cost_calculator,
    regime_classifier=regime_classifier,
    champion_manager=champion_manager,
    s3_client=s3_client,
    database_client=database_client,
)

result = pipeline.train_symbol("BTCUSDT", candles_df, cfg={})
```

## Command Line

### Run Training

```bash
# Hybrid mode (default)
python -m cloud.training.pipelines.daily_retrain_scheduler --mode hybrid --max_concurrent 12 --symbols top20

# Sequential mode
python -m cloud.training.pipelines.daily_retrain_scheduler --mode sequential --symbols top20

# Parallel mode
python -m cloud.training.pipelines.daily_retrain_scheduler --mode parallel --symbols top20
```

## Configuration

### Environment Variables

```bash
export DATABASE_URL="postgresql://user:password@localhost/huracan"
export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."
export TELEGRAM_TOKEN="..."
export TELEGRAM_CHAT_ID="..."
```

### Config File

See `config.yaml` for all configuration options.

## File Structure

```
s3://huracan/
├── models/SYMBOL/TIMESTAMP/
├── challengers/SYMBOL/TIMESTAMP/
├── champion/SYMBOL/latest.json
├── summaries/daily/DATE.json
└── live_logs/trades/YYYYMMDD.parquet
```

## Contracts

### Engine Output

```python
EngineOutput(
    direction=Direction.BUY,  # buy, sell, wait
    edge_bps_before_costs=10.0,  # Expected edge in basis points
    confidence_0_1=0.8,  # Confidence score (0.0 to 1.0)
    horizon_minutes=60,  # Prediction horizon
    metadata={},  # Additional metadata
)
```

### Model Bundle

```
model.bin          # Trained model
config.json        # Model configuration
metrics.json       # Performance metrics
sha256.txt         # Integrity hash
```

### Champion Pointer

```json
{
  "symbol": "BTCUSDT",
  "model_id": "model_123",
  "s3_path": "s3://huracan/models/BTCUSDT/20250101_020000Z/",
  "updated_at": "2025-01-01T02:00:00Z"
}
```

## Guardrails

### Net Edge Floor

```python
should_trade = calculator.should_trade(
    symbol="BTCUSDT",
    edge_bps_before_costs=10.0,
    net_edge_floor_bps=3.0,
)
```

### Spread Threshold

```python
# Skip thin books
if spread_bps > threshold:
    skip_symbol(symbol)
```

### Cooldowns

```python
# Cut churn
if time_since_last_trade < cooldown_seconds:
    skip_trade()
```

### Sample Size Gates

```python
# No champion flips on tiny samples
if sample_size < min_sample_size:
    skip_promotion()
```

## Next Steps

1. Complete S3 integration
2. Complete database integration
3. Implement Telegram bot
4. Implement guardrails
5. Implement 23 engines
6. Test with 400 coins

