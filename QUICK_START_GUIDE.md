# Quick Start Guide - Using New Features

## 1. Database Operations

```python
from src.shared.database.models import DatabaseClient, ModelRecord, ModelMetrics
from datetime import datetime

# Initialize database client
db = DatabaseClient("postgresql://user:password@localhost:5432/huracan")

# Save a model
model = ModelRecord(
    model_id="btc_model_001",
    parent_id=None,
    kind="baseline",
    created_at=datetime.now(),
    s3_path="s3://huracan/models/btc_model_001.pkl",
    features_used=["rsi", "ema", "volatility"],
    params={"max_depth": 6, "n_estimators": 100}
)
db.save_model(model)

# Save metrics
metrics = ModelMetrics(
    model_id="btc_model_001",
    sharpe=2.5,
    hit_rate=0.58,
    drawdown=0.15,
    net_bps=45.0,
    window="test",
    cost_bps=8.0,
    promoted=False
)
db.save_metrics(metrics)
```

## 2. Configuration with Validation

```python
from shared.config import load_config, HuracanConfig

# Load and validate configuration
config: HuracanConfig = load_config()

# Access configuration with full type safety and autocomplete
lookback_days = config.engine.lookback_days  # int: 180
model_type = config.engine.model_type  # ModelType.XGBOOST
symbols = config.general.symbols  # List[str]: ["BTCUSDT", ...]

# Access nested config
max_spread = config.engine.data_gates.max_spread_bps  # float: 8.0
pool_size = config.database.pool_size  # int: 10

# Environment variables automatically resolved
db_url = config.database.connection_string  # Resolved from ${DATABASE_URL}
s3_key = config.s3.access_key  # Resolved from ${AWS_ACCESS_KEY_ID}

# Convert to dict if needed
config_dict = config.to_dict()
```

## 3. Exception Handling

```python
from shared.exceptions import (
    DatabaseError,
    DropboxError,
    ModelLoadError,
    DataQualityError
)
import structlog

logger = structlog.get_logger(__name__)

# Use specific exceptions with context
try:
    model = load_model(model_path)
except ModelLoadError as e:
    logger.error("model_load_failed",
                error=str(e),
                context=e.context,
                error_type="ModelLoadError")
    # Handle model load error specifically
    fallback_model = load_backup_model()

# Raise exceptions with context
try:
    upload_to_dropbox(file_path)
except ConnectionError as e:
    raise DropboxError(
        f"Failed to upload {file_path}",
        context={"file": file_path, "attempt": retry_count}
    ) from e
```

## 4. Feature Store (No Singleton!)

```python
from shared.features.feature_store import FeatureStore
from pathlib import Path

# Explicitly instantiate (no global singleton)
feature_store = FeatureStore(store_path=Path(".feature_store"))

# Register features
feature_store.register_feature(
    name="rsi_14",
    description="14-period RSI",
    version="1.0.0",
    status=FeatureStatus.ACTIVE
)

# Create feature set
feature_set = feature_store.create_feature_set(
    features=["rsi_14", "ema_20", "volatility"],
    version="2025_01_15",
    metadata={"experiment": "baseline_v1"}
)

# Pin to run
feature_store.pin_feature_set(feature_set.feature_set_id, run_id="run_20250115_001")

# Dependency injection example
class ModelTrainer:
    def __init__(self, feature_store: FeatureStore):
        self.feature_store = feature_store  # Injected, not global

    def train(self, data):
        features = self.feature_store.list_features()
        # ... training logic
```

## 5. Structured Logging

```python
import structlog

logger = structlog.get_logger(__name__)

# Instead of print statements, use structured logging
# BAD - Don't do this
# print(f"Training started with {n_samples} samples")

# GOOD - Use structured logging
logger.info("training_started",
           n_samples=len(training_data),
           lookback_days=config.engine.lookback_days,
           model_type=config.engine.model_type.value)

# Log complex results
logger.info("backtest_results",
           sharpe_ratio=results.sharpe_ratio,
           total_pnl=results.total_pnl_usd,
           win_rate=results.overall_win_rate,
           max_drawdown=results.max_drawdown_pct)

# Log errors with context
try:
    process_data(symbol)
except DataQualityError as e:
    logger.error("data_quality_failed",
                symbol=symbol,
                error=str(e),
                error_type=type(e).__name__,
                checks_failed=e.context.get("checks_failed", []))
```

## 6. Type-Safe Code

```python
# Type hints are now enforced with pyright in standard mode

from typing import List, Dict, Optional
from datetime import datetime

# Function with proper type hints
def calculate_metrics(
    trades: List[Dict[str, float]],
    start_date: datetime,
    end_date: Optional[datetime] = None
) -> Dict[str, float]:
    """
    Calculate trading metrics.

    Args:
        trades: List of trade dictionaries
        start_date: Start date for calculation
        end_date: Optional end date

    Returns:
        Dictionary of metrics
    """
    # ... implementation
    return {
        "sharpe": 2.5,
        "win_rate": 0.58,
        "pnl": 1250.0
    }

# Class with type hints
class ModelRegistry:
    def __init__(self, db_client: DatabaseClient) -> None:
        self.db_client = db_client
        self.models: Dict[str, ModelRecord] = {}

    def register(self, model: ModelRecord) -> bool:
        # ... implementation
        return True
```

## 7. Migration Guide

### Updating Existing Code

#### Database Operations
**Before:**
```python
# TODO: Implement database save
logger.info("model_saved", model_id=model.model_id)
return True
```

**After:**
```python
from shared.database.models import DatabaseClient, ModelRecord

db = DatabaseClient(config.database.connection_string)
success = db.save_model(model)  # Actually saves to database
return success
```

#### Configuration
**Before:**
```python
import yaml

with open("config.yaml") as f:
    config = yaml.safe_load(f)

lookback_days = config["engine"]["lookback_days"]  # No type safety
```

**After:**
```python
from shared.config import load_config

config = load_config()  # Validated automatically
lookback_days = config.engine.lookback_days  # Type-safe, autocomplete works
```

#### Exception Handling
**Before:**
```python
try:
    upload_file(path)
except Exception as e:  # Too broad
    logger.error("upload_failed", error=str(e))
    return None
```

**After:**
```python
from shared.exceptions import DropboxError

try:
    upload_file(path)
except DropboxError as e:  # Specific exception
    logger.error("upload_failed",
                error=str(e),
                error_type="DropboxError",
                context=e.context)
    raise  # Or handle appropriately
except OSError as e:  # Separate file errors
    logger.error("file_error", error=str(e))
    raise
```

#### Feature Store
**Before:**
```python
from shared.features.feature_store import get_feature_store

store = get_feature_store()  # Global singleton
store.register_feature(...)
```

**After:**
```python
from shared.features.feature_store import FeatureStore

# Explicit instantiation and injection
store = FeatureStore(store_path=Path(".feature_store"))

# Pass to components via dependency injection
trainer = ModelTrainer(feature_store=store)
```

## 8. Testing Examples

```python
import pytest
from shared.config import load_config, HuracanConfig
from shared.exceptions import ConfigurationError

def test_config_validation():
    """Test configuration validation."""
    config = load_config()

    # Validate types
    assert isinstance(config.engine.lookback_days, int)
    assert config.engine.lookback_days > 0

    # Validate constraints
    assert config.engine.start_with_symbols <= config.engine.target_symbols

def test_database_operations():
    """Test database save operations."""
    from shared.database.models import DatabaseClient, ModelRecord
    from datetime import datetime

    db = DatabaseClient("sqlite:///:memory:")  # In-memory for testing

    model = ModelRecord(
        model_id="test_model",
        parent_id=None,
        kind="test",
        created_at=datetime.now(),
        s3_path="s3://test/model.pkl",
        features_used=["test_feature"],
        params={"test": True}
    )

    # Should not raise
    result = db.save_model(model)
    assert result is True

def test_exception_context():
    """Test exception context."""
    from shared.exceptions import DataError

    with pytest.raises(DataError) as exc_info:
        raise DataError("Test error", context={"symbol": "BTCUSDT"})

    assert exc_info.value.context["symbol"] == "BTCUSDT"
```

## 9. Environment Setup

```bash
# Set required environment variables
export DATABASE_URL="postgresql://user:password@localhost:5432/huracan"
export AWS_ACCESS_KEY_ID="your_access_key"
export AWS_SECRET_ACCESS_KEY="your_secret_key"
export TELEGRAM_TOKEN="your_telegram_token"
export TELEGRAM_CHAT_ID="your_chat_id"

# Install dependencies (if needed)
pip install pydantic pydantic-settings sqlalchemy structlog

# Run type checking
pyright src/

# Run tests
pytest tests/
```

## 10. Common Patterns

### Dependency Injection Pattern
```python
class TradingEngine:
    def __init__(
        self,
        config: HuracanConfig,
        db_client: DatabaseClient,
        feature_store: FeatureStore
    ):
        self.config = config
        self.db_client = db_client
        self.feature_store = feature_store

    def train(self, symbol: str) -> ModelRecord:
        # Use injected dependencies
        features = self.feature_store.list_features()
        # ... training logic
        model = create_model()
        self.db_client.save_model(model)
        return model

# Initialize with dependencies
config = load_config()
db = DatabaseClient(config.database.connection_string)
store = FeatureStore()

engine = TradingEngine(config=config, db_client=db, feature_store=store)
```

### Error Handling Pattern
```python
from shared.exceptions import (
    DataLoadError,
    ModelTrainingError,
    DatabaseError
)

def safe_train_model(symbol: str) -> Optional[ModelRecord]:
    """Train model with comprehensive error handling."""
    try:
        # Load data
        data = load_data(symbol)
    except DataLoadError as e:
        logger.error("data_load_failed", symbol=symbol, error=str(e))
        return None

    try:
        # Train model
        model = train_model(data)
    except ModelTrainingError as e:
        logger.error("training_failed", symbol=symbol, error=str(e))
        return None

    try:
        # Save to database
        db.save_model(model)
    except DatabaseError as e:
        logger.error("save_failed", model_id=model.model_id, error=str(e))
        # Model trained but not saved - handle appropriately

    return model
```

---

**Last Updated:** 2025-11-11
**Author:** Claude Code
