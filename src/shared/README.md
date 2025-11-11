# Shared Module

## Overview

The shared module contains all shared components used by the Engine, Mechanic, and Hamilton. This ensures consistency and reusability across all modules.

## Components

### 1. Engines (`engines/`)

- **Engine Interface**: Unified interface for all 23 engines
- **Engine Registry**: Registry for managing engines
- **Example Engines**: Example implementations

### 2. Features (`features/`)

- **Feature Builder**: Shared feature builder for cloud and Hamilton
- **Feature Recipe**: Recipe definition with hash

### 3. Costs (`costs/`)

- **Cost Calculator**: Cost calculator for per-symbol, per-bar costs
- **Cost Model**: Cost model per symbol

### 4. Regime (`regime/`)

- **Regime Classifier**: Classifier for market regimes
- **Regime Enum**: TREND, RANGE, PANIC, ILLIQUID

### 5. Meta (`meta/`)

- **Meta Combiner**: Per-coin meta combiner
- **EMA Weights**: Weights by recent accuracy and net edge

### 6. Champion (`champion/`)

- **Champion Manager**: Per-coin champion pointer manager
- **Latest.json**: Per symbol, always valid

### 7. Storage (`storage/`)

- **S3 Client**: S3 storage client
- **Upload/Download**: Files and JSON
- **Signed URLs**: For Hamilton access

### 8. Database (`database/`)

- **Database Models**: Models, metrics, promotions, live trades, daily equity
- **Database Client**: Database client for operations

### 9. Telegram (`telegram/`)

- **Symbols Selector**: Symbols selector for Telegram control
- **Load/Save**: Symbols from selector file

### 10. Summary (`summary/`)

- **Daily Summary Generator**: Generator for daily summaries
- **Top Contributors**: Hit rate, net edge, trades counted

### 11. Contracts (`contracts/`)

- **Model Bundle**: Model bundle contract
- **Champion Pointer**: Champion pointer contract
- **Per-Coin Contracts**: Run manifest, metrics, costs, etc.

## Usage

### Import Components

```python
from src.shared.engines import BaseEngine, EngineRegistry
from src.shared.features import FeatureBuilder
from src.shared.costs import CostCalculator
from src.shared.regime import RegimeClassifier
from src.shared.meta import MetaCombiner
from src.shared.champion import ChampionManager
from src.shared.storage import S3Client
from src.shared.database import DatabaseClient
from src.shared.telegram import SymbolsSelector
from src.shared.summary import DailySummaryGenerator
```

### Create Components

```python
# Engine registry
registry = EngineRegistry()

# Feature builder
builder = FeatureBuilder()

# Cost calculator
calculator = CostCalculator()

# Regime classifier
classifier = RegimeClassifier()

# Champion manager
manager = ChampionManager(s3_bucket="huracan")

# S3 client
s3_client = S3Client(bucket="huracan")

# Database client
db_client = DatabaseClient(connection_string="...")

# Symbols selector
selector = SymbolsSelector()

# Summary generator
summary_generator = DailySummaryGenerator()
```

## Contracts

### Engine Output

```python
EngineOutput(
    direction=Direction.BUY,
    edge_bps_before_costs=10.0,
    confidence_0_1=0.8,
    horizon_minutes=60,
    metadata={},
)
```

### Model Bundle

- `model.bin`: Trained model
- `config.json`: Model configuration
- `metrics.json`: Performance metrics
- `sha256.txt`: Integrity hash

### Champion Pointer

- `champion/SYMBOL/latest.json`: Champion pointer file
- `s3_path`: S3 path to model bundle
- `model_id`: Model identifier

## File Structure

```
src/shared/
├── engines/          # Engine interface and registry
├── features/         # Shared feature builder
├── costs/            # Cost calculator
├── regime/           # Regime classifier
├── meta/             # Meta combiner
├── champion/         # Champion manager
├── storage/          # S3 storage client
├── database/         # Database models and client
├── telegram/         # Telegram control
├── summary/          # Daily summary generator
└── contracts/        # Model bundle and champion pointer contracts
```

## Dependencies

- `pandas`: Data manipulation
- `structlog`: Structured logging
- `boto3`: AWS S3 client (optional)
- `dropbox`: Dropbox client (optional)

## Testing

```bash
# Run tests
pytest tests/test_shared/

# Run specific tests
pytest tests/test_shared/test_engines.py
pytest tests/test_shared/test_features.py
pytest tests/test_shared/test_costs.py
```

## Documentation

- [Core System Implementation](../../docs/CORE_SYSTEM_IMPLEMENTATION.md)
- [Implementation Complete](../../docs/IMPLEMENTATION_COMPLETE.md)
- [Quick Reference](../../docs/QUICK_REFERENCE.md)

