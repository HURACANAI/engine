# HURACAN ENGINE - PRODUCTION ARCHITECTURE

**Version:** 2.0  
**Last Updated:** 2025-11-11  
**Status:** Production-Ready

---

## 🎯 EXECUTIVE SUMMARY

This document defines the **strict architectural standards** for the Huracan Trading Engine. All code, features, and modifications **MUST** adhere to these principles to ensure maintainability, scalability, and production-readiness.

**Core Principle:** *Separation of Concerns, Explicit Dependencies, Type Safety, Comprehensive Testing*

---

## 📐 ARCHITECTURE OVERVIEW

### System Layers

```
┌─────────────────────────────────────────────────────────────┐
│                    PRESENTATION LAYER                        │
│  (Observability, Monitoring, UI, Reports)                   │
│  Location: observability/, monitoring/, integrations/        │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                    APPLICATION LAYER                         │
│  (Pipelines, Orchestration, Services)                       │
│  Location: pipelines/, services/, orchestrator/             │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                    DOMAIN LAYER                              │
│  (Models, Engines, Business Logic)                          │
│  Location: models/, agents/, portfolio/, risk/              │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                    INFRASTRUCTURE LAYER                      │
│  (Data Access, ML Framework, External Services)             │
│  Location: datasets/, ml_framework/, brain/, integrations/  │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                    SHARED LAYER                              │
│  (Common Types, Features, Contracts)                        │
│  Location: shared/features/, shared/contracts/              │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 DIRECTORY STRUCTURE & NAMING CONVENTIONS

### Root Structure

```
engine/
├── src/
│   ├── cloud/
│   │   └── training/          # Main training application
│   └── shared/                # Shared code across components
├── observability/             # Monitoring, UI, AI Council
├── config/                    # Configuration files
├── tests/                     # Test suite (mirrors src/ structure)
├── scripts/                   # Utility scripts
├── infrastructure/            # Deployment configs (Docker, K8s)
├── docs/                      # Documentation
└── migrations/                # Database migrations
```

### Core Module Organization

#### `/src/cloud/training/`

```
training/
├── __init__.py                # Package exports
├── pipelines/                 # Workflow orchestration
│   ├── daily_retrain.py      # Main entry point
│   ├── rl_training_pipeline.py
│   └── feature_workflow.py
├── models/                    # Domain models & engines
│   ├── alpha_engines.py      # 23 trading engines
│   ├── regime_detector.py
│   ├── world_model.py        # State prediction
│   └── feature_autoencoder.py
├── agents/                    # RL agents
│   ├── rl_agent.py           # PPO agent
│   └── reward_evaluator.py
├── services/                  # Application services
│   ├── orchestration.py
│   ├── mechanic_utils.py     # Financial utilities
│   └── fee_latency_calibration.py
├── datasets/                  # Data access layer
│   ├── data_loader.py
│   ├── return_converter.py
│   └── data_integrity_checkpoint.py
├── ml_framework/              # ML infrastructure
│   ├── preprocessing/
│   ├── training/
│   └── inference/
├── portfolio/                 # Portfolio management
│   ├── optimizer.py
│   └── risk_manager.py
├── execution/                 # Order execution
│   ├── spread_threshold_manager.py
│   └── smart_order_router.py
├── orderbook/                 # Orderbook management
│   └── multi_exchange_aggregator.py
├── metrics/                   # Performance metrics
│   ├── enhanced_metrics.py
│   └── performance_visualizer.py
├── analysis/                  # Model analysis
│   └── model_introspection.py
├── validation/                # Validation & testing
│   └── walk_forward_testing.py
├── monitoring/                # Health & monitoring
│   └── health_check.py
└── integrations/              # External services
    └── dropbox_sync.py
```

### Naming Conventions

#### Files & Modules
- **Snake_case** for all Python files: `return_converter.py`, `data_loader.py`
- **Descriptive names** that indicate purpose: `fee_latency_calibration.py` not `fees.py`
- **One class per file** when possible, or clearly related classes

#### Classes
- **PascalCase**: `ReturnConverter`, `DataIntegrityCheckpoint`
- **Suffix conventions**:
  - Services: `*Service`, `*Manager`, `*Orchestrator`
  - Models: `*Model`, `*Engine`, `*Detector`
  - Data Access: `*Loader`, `*Converter`, `*Repository`
  - Utilities: `*Utils`, `*Helper` (use sparingly)

#### Functions & Methods
- **Snake_case**: `convert_to_returns()`, `calculate_sharpe_ratio()`
- **Verb-based names**: `get_*`, `calculate_*`, `validate_*`, `store_*`
- **Boolean returns**: `is_*`, `has_*`, `should_*`, `can_*`

#### Constants
- **UPPER_SNAKE_CASE**: `MAX_RETRY_ATTEMPTS`, `DEFAULT_LOOKBACK_DAYS`

#### Variables
- **Snake_case**: `returns_df`, `sharpe_ratio`, `current_state`
- **Avoid abbreviations** unless widely understood: `df` (DataFrame), `idx` (index)

---

## 🏗️ CODE ORGANIZATION PRINCIPLES

### 1. Separation of Concerns

**Each module has ONE responsibility:**

- **Pipelines** → Orchestrate workflows, coordinate services
- **Models** → Business logic, algorithms, domain rules
- **Services** → Application logic, coordination
- **Datasets** → Data access, transformation, validation
- **ML Framework** → ML infrastructure, training, inference
- **Metrics** → Performance calculation, visualization

**❌ BAD:**
```python
# models/alpha_engine.py - DON'T DO THIS
class AlphaEngine:
    def train(self, data):
        # Training logic
        df = pd.read_csv('data.csv')  # ❌ Data access in model
        self.save_to_db(results)      # ❌ Database access in model
        send_telegram_notification()  # ❌ External service in model
```

**✅ GOOD:**
```python
# models/alpha_engine.py
class AlphaEngine:
    def train(self, features: np.ndarray, targets: np.ndarray) -> Model:
        # Pure business logic only
        return trained_model

# datasets/data_loader.py
class DataLoader:
    def load_training_data(self, symbol: str) -> pd.DataFrame:
        # Data access only
        return df

# services/orchestration.py
class TrainingOrchestrator:
    def run_training(self, symbol: str):
        # Coordinates all components
        data = self.data_loader.load_training_data(symbol)
        model = self.alpha_engine.train(data.features, data.targets)
        self.brain_library.store_model(model)
        self.telegram_service.send_notification("Training complete")
```

### 2. Dependency Injection

**Always inject dependencies, never create them inside classes:**

**❌ BAD:**
```python
class ReturnConverter:
    def __init__(self):
        self.brain_library = BrainLibrary()  # ❌ Hard dependency
```

**✅ GOOD:**
```python
class ReturnConverter:
    def __init__(self, brain_library: Optional[BrainLibrary] = None):
        self.brain_library = brain_library  # ✅ Injected dependency
```

### 3. Type Hints & Type Safety

**All functions MUST have type hints:**

```python
from typing import Optional, List, Dict, Tuple
import pandas as pd
import numpy as np

def convert_to_returns(
    price_data: pl.DataFrame,
    price_column: str = 'close',
    symbol: Optional[str] = None,
    timestamp_column: str = 'timestamp'
) -> pl.DataFrame:
    """
    Convert price series to return series.
    
    Args:
        price_data: DataFrame with price data
        price_column: Name of price column
        symbol: Trading symbol (optional)
        timestamp_column: Name of timestamp column
    
    Returns:
        DataFrame with raw_returns and log_returns columns
    
    Raises:
        ValueError: If required columns are missing
    """
    # Implementation
```

### 4. Error Handling

**Always handle errors explicitly:**

```python
def convert_to_returns(...) -> pl.DataFrame:
    if price_column not in price_data.columns:
        raise ValueError(f"Price column '{price_column}' not found in DataFrame")
    
    try:
        # Main logic
        result = ...
    except Exception as e:
        logger.error(
            "return_conversion_failed",
            symbol=symbol,
            error=str(e),
            error_type=type(e).__name__
        )
        raise
```

**Never use bare `except:` clauses:**

**❌ BAD:**
```python
try:
    result = risky_operation()
except:  # ❌ Catches everything, including KeyboardInterrupt
    pass
```

**✅ GOOD:**
```python
try:
    result = risky_operation()
except ValueError as e:
    logger.warning("invalid_input", error=str(e))
    raise
except Exception as e:
    logger.error("unexpected_error", error=str(e))
    raise
```

### 5. Logging Standards

**Use structured logging with `structlog`:**

```python
import structlog

logger = structlog.get_logger(__name__)

# ✅ GOOD: Structured logging
logger.info(
    "returns_converted",
    symbol=symbol,
    final_rows=len(result),
    mean_return=float(result['raw_returns'].mean()),
    std_return=float(result['raw_returns'].std())
)

# ❌ BAD: String formatting
logger.info(f"Converted returns for {symbol}: {len(result)} rows")  # ❌
```

**Log Levels:**
- `logger.debug()` - Detailed diagnostic info
- `logger.info()` - General informational messages
- `logger.warning()` - Warning messages (non-critical)
- `logger.error()` - Error messages (recoverable)
- `logger.critical()` - Critical errors (system failure)

---

## 🧪 TESTING STANDARDS

### Test Organization

**Tests mirror source structure:**

```
tests/
├── test_datasets/
│   ├── test_return_converter.py
│   └── test_data_loader.py
├── test_models/
│   ├── test_world_model.py
│   └── test_alpha_engines.py
├── test_services/
│   └── test_mechanic_utils.py
└── test_integration/
    └── test_full_pipeline.py
```

### Test Naming

- **Test files**: `test_*.py` or `*_test.py`
- **Test functions**: `test_*` (pytest convention)
- **Test classes**: `Test*` (for grouping related tests)

### Test Coverage Requirements

**Minimum coverage: 80% for all new code**

**Every function MUST have:**
1. **Happy path test** - Normal operation
2. **Edge case tests** - Empty inputs, boundary values
3. **Error case tests** - Invalid inputs, missing dependencies
4. **Integration tests** - Interaction with other components

**Example:**

```python
# tests/test_datasets/test_return_converter.py
import pytest
import pandas as pd
import polars as pl
from src.cloud.training.datasets.return_converter import ReturnConverter

class TestReturnConverter:
    def test_convert_normal_case(self):
        """Test normal price to returns conversion."""
        converter = ReturnConverter()
        df = pl.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='h'),
            'close': [100 + i * 0.1 for i in range(100)]
        })
        
        result = converter.convert(df, price_column='close', symbol='BTC/USD')
        
        assert 'raw_returns' in result.columns
        assert 'log_returns' in result.columns
        assert len(result) == 100
    
    def test_convert_missing_price_column(self):
        """Test error when price column is missing."""
        converter = ReturnConverter()
        df = pl.DataFrame({'timestamp': [1, 2, 3]})
        
        with pytest.raises(ValueError, match="Price column"):
            converter.convert(df, price_column='close', symbol='BTC/USD')
    
    def test_convert_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        converter = ReturnConverter()
        df = pl.DataFrame({'timestamp': [], 'close': []})
        
        result = converter.convert(df, price_column='close', symbol='BTC/USD')
        assert len(result) == 0
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_datasets/test_return_converter.py

# Run specific test
pytest tests/test_datasets/test_return_converter.py::TestReturnConverter::test_convert_normal_case
```

---

## 🔒 SECURITY & RELIABILITY

### Input Validation

**Always validate inputs:**

```python
def convert_to_returns(
    price_data: pl.DataFrame,
    price_column: str = 'close',
    symbol: Optional[str] = None
) -> pl.DataFrame:
    # Validate inputs
    if not isinstance(price_data, pl.DataFrame):
        raise TypeError(f"price_data must be polars DataFrame, got {type(price_data)}")
    
    if price_column not in price_data.columns:
        raise ValueError(f"Price column '{price_column}' not found")
    
    if len(price_data) == 0:
        logger.warning("empty_dataframe", symbol=symbol)
        return price_data  # Return empty or raise, depending on use case
```

### Data Sanitization

**Always sanitize data before processing:**

```python
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # Remove NaN values
    df = df.dropna()
    
    # Remove outliers (3 standard deviations)
    for col in df.select_dtypes(include=[np.number]).columns:
        mean = df[col].mean()
        std = df[col].std()
        df = df[(df[col] >= mean - 3*std) & (df[col] <= mean + 3*std)]
    
    return df
```

### Resource Management

**Always use context managers for resources:**

```python
# ✅ GOOD: Context manager
with self._get_connection() as conn:
    with conn.cursor() as cur:
        cur.execute("SELECT * FROM returns")
        results = cur.fetchall()

# ❌ BAD: Manual cleanup
conn = self._get_connection()
cur = conn.cursor()
cur.execute("SELECT * FROM returns")
results = cur.fetchall()
cur.close()  # ❌ What if exception occurs?
conn.close()
```

### Memory Management

**For large datasets, use generators or chunking:**

```python
def process_large_dataset(file_path: str):
    # ✅ GOOD: Process in chunks
    chunk_size = 10000
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        process_chunk(chunk)
    
    # ❌ BAD: Load everything into memory
    df = pd.read_csv(file_path)  # ❌ May cause OOM
    process_dataframe(df)
```

---

## 📝 DOCUMENTATION STANDARDS

### Docstrings

**All public functions, classes, and modules MUST have docstrings:**

```python
class ReturnConverter:
    """
    Return Converter node that converts price series to return series.
    
    This normalizes the data across all tickers for fair HTF comparison.
    
    Usage:
        converter = ReturnConverter(brain_library=brain)
        returns_df = converter.convert(
            price_data=df,
            price_column='close',
            symbol='BTC/USDT'
        )
    
    Attributes:
        brain_library: Optional Brain Library instance for storing returns
        use_adjusted_prices: Whether to use adjusted close prices
        fill_method: Method for filling NaN values ('forward', 'backward', 'drop')
    """
    
    def convert(
        self,
        price_data: pl.DataFrame,
        price_column: str = 'close',
        symbol: Optional[str] = None,
        timestamp_column: str = 'timestamp'
    ) -> pl.DataFrame:
        """
        Convert price series to return series.
        
        Args:
            price_data: DataFrame with price data (must have price_column and timestamp_column)
            price_column: Name of the price column (default: 'close')
            symbol: Trading symbol (optional, for Brain Library storage)
            timestamp_column: Name of the timestamp column (default: 'timestamp')
        
        Returns:
            DataFrame with:
            - All original columns
            - raw_returns: Percent change returns
            - log_returns: Log returns
            - adjusted_close: Adjusted close prices (if use_adjusted_prices=True)
        
        Raises:
            ValueError: If price_column or timestamp_column not found in DataFrame
        
        Example:
            >>> df = pl.DataFrame({'timestamp': [...], 'close': [100, 101, 102]})
            >>> converter = ReturnConverter()
            >>> result = converter.convert(df, price_column='close', symbol='BTC/USD')
            >>> assert 'raw_returns' in result.columns
        """
```

### Module Documentation

**Every module file starts with module docstring:**

```python
"""
Return Converter - Price to Return Conversion

Converts raw price series to total return series for normalized comparison.
Handles NaN cleaning, adjusted prices, and Brain Library integration.

Author: Huracan Engine Team
Date: 2025-11-11
"""
```

---

## 🚀 DEPLOYMENT & INFRASTRUCTURE

### Configuration Management

**All configuration via environment variables or config files:**

```python
# config/settings.py
import os
from dataclasses import dataclass

@dataclass
class Settings:
    database_url: str = os.getenv("DATABASE_URL", "postgresql://localhost/huracan")
    brain_library_path: str = os.getenv("BRAIN_LIBRARY_PATH", "./brain_library")
    telegram_bot_token: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    
    @classmethod
    def from_env(cls) -> "Settings":
        return cls()
```

### Docker & Containerization

**All services containerized:**

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY src/ ./src/
COPY config/ ./config/

# Run application
CMD ["python", "-m", "cloud.training.pipelines.daily_retrain"]
```

### CI/CD Pipeline

**Automated testing and deployment:**

```yaml
# .github/workflows/ci.yml
name: CI/CD Pipeline

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - run: pip install -r requirements.txt
      - run: pytest tests/ --cov=src --cov-report=xml
      - run: pylint src/
      - run: mypy src/
```

---

## 🔄 CODE REVIEW CHECKLIST

Before submitting code, ensure:

- [ ] **Type hints** on all functions
- [ ] **Docstrings** on all public functions/classes
- [ ] **Tests** for all new functions (80%+ coverage)
- [ ] **Error handling** for all edge cases
- [ ] **Logging** for important operations
- [ ] **Input validation** for all inputs
- [ ] **No hardcoded values** (use config/constants)
- [ ] **No circular dependencies**
- [ ] **Follows naming conventions**
- [ ] **Updated documentation** if architecture changes

---

## 📊 METRICS & MONITORING

### Code Quality Metrics

- **Test Coverage**: Minimum 80%
- **Type Coverage**: 100% (all functions typed)
- **Linting**: Zero errors (pylint, mypy)
- **Complexity**: Cyclomatic complexity < 10 per function

### Performance Metrics

- **Response Time**: Log all operations > 100ms
- **Memory Usage**: Monitor for memory leaks
- **Error Rate**: Track and alert on error rates > 1%

---

## 🗺️ ROADMAP & TECHNICAL DEBT

### Current Technical Debt

1. **Legacy Code Migration**: Some old code doesn't follow new architecture
   - **Priority**: Medium
   - **Estimated Effort**: 2 weeks
   - **Location**: `models/alpha_engines.py` (partial)

2. **Test Coverage**: Some modules below 80%
   - **Priority**: High
   - **Estimated Effort**: 1 week
   - **Location**: `ml_framework/` (some modules)

3. **Type Hints**: Some legacy code missing type hints
   - **Priority**: Low
   - **Estimated Effort**: 1 week
   - **Location**: Various legacy files

### Future Improvements

1. **Async/Await**: Migrate I/O operations to async
   - ✅ Foundation created (`database/async_pool.py`)
   - 🚧 Migration in progress
   - See: `docs/architecture/ARCHITECTURE_IMPROVEMENTS.md`

2. **Caching Layer**: Add Redis for frequently accessed data
   - ✅ Infrastructure created (`cache/redis_client.py`, `cache/cache_manager.py`)
   - 🚧 Integration in progress
   - See: `docs/architecture/ARCHITECTURE_IMPROVEMENTS.md`

3. **GraphQL API**: Add GraphQL endpoint for flexible queries
   - ✅ Foundation created (`api/graphql/schema.py`, `api/graphql/server.py`)
   - 🚧 Resolvers implementation in progress
   - See: `docs/architecture/ARCHITECTURE_IMPROVEMENTS.md`

4. **Microservices**: Split into microservices for better scalability
   - 📋 Design phase
   - See: `docs/architecture/ARCHITECTURE_IMPROVEMENTS.md`

---

## 📚 REFERENCES

- [Python Type Hints PEP 484](https://www.python.org/dev/peps/pep-0484/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [Pytest Documentation](https://docs.pytest.org/)
- [Structlog Documentation](https://www.structlog.org/)

---

## ✅ COMPLIANCE

**All code MUST comply with this architecture.**

**Violations will be rejected in code review.**

**Questions?** Contact the architecture team or update this document.

---

**Last Updated:** 2025-11-11  
**Maintained By:** Huracan Engine Architecture Team

