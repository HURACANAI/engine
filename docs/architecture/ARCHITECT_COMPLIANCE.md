# Architecture Compliance - Lead Architect Commitment

**Date:** 2025-11-12  
**Status:** Active  
**Architect:** AI Lead Software Architect

---

## ✅ ARCHITECTURE ACKNOWLEDGMENT

I acknowledge and commit to enforcing the **strict architectural standards** defined in `docs/architecture/ARCHITECTURE.md` for all code generation, modifications, and feature development.

### Core Principles (Always Enforced)

1. **Separation of Concerns** - Each module has ONE responsibility
2. **Explicit Dependencies** - Dependency injection, no hard dependencies
3. **Type Safety** - All functions MUST have type hints
4. **Comprehensive Testing** - 80%+ coverage, tests mirror source structure
5. **Structured Logging** - Use `structlog` for all logging
6. **Error Handling** - Explicit error handling, no bare `except:` clauses
7. **Input Validation** - Always validate inputs
8. **Documentation** - All public functions/classes MUST have docstrings

---

## 📐 LAYERED ARCHITECTURE COMPLIANCE

### Layer Responsibilities (Strictly Enforced)

```
PRESENTATION LAYER (observability/, monitoring/)
├── Observability dashboards
├── Monitoring systems
├── UI components
└── Reports

APPLICATION LAYER (pipelines/, services/, orchestrator/)
├── Workflow orchestration
├── Service coordination
└── Business process management

DOMAIN LAYER (models/, agents/, portfolio/, risk/)
├── Business logic
├── Trading algorithms
├── Domain models
└── Core engines

INFRASTRUCTURE LAYER (datasets/, ml_framework/, brain/, integrations/)
├── Data access
├── ML infrastructure
├── External service integration
└── Storage systems

SHARED LAYER (shared/features/, shared/contracts/)
├── Common types
├── Shared features
└── Contracts/interfaces
```

**Rule:** Code MUST be placed in the correct layer. Cross-layer dependencies flow DOWN only (presentation → application → domain → infrastructure → shared).

---

## 📁 DIRECTORY STRUCTURE COMPLIANCE

### File Organization Rules

1. **All source code** → `/src/cloud/training/`
2. **All tests** → `/tests/` (mirrors source structure)
3. **All configs** → `/config/`
4. **All scripts** → `/scripts/`
5. **All docs** → `/docs/`
6. **All infrastructure** → `/infrastructure/`

### Module Organization (Strict)

```
src/cloud/training/
├── pipelines/          # Workflow orchestration ONLY
├── models/             # Domain models & engines ONLY
├── agents/             # RL agents ONLY
├── services/           # Application services ONLY
├── datasets/           # Data access ONLY
├── ml_framework/       # ML infrastructure ONLY
├── portfolio/          # Portfolio management ONLY
├── execution/          # Order execution ONLY
├── metrics/            # Performance metrics ONLY
├── validation/         # Validation & testing ONLY
├── monitoring/         # Health & monitoring ONLY
└── integrations/       # External services ONLY
```

**Rule:** Each directory has a SINGLE responsibility. If code doesn't fit, create a new appropriate directory.

---

## 🏷️ NAMING CONVENTIONS (Strictly Enforced)

### Files & Modules
- ✅ **Snake_case**: `return_converter.py`, `data_loader.py`
- ✅ **Descriptive names**: `fee_latency_calibration.py` (not `fees.py`)
- ❌ **Never**: `utils.py`, `helpers.py`, `misc.py` (too generic)

### Classes
- ✅ **PascalCase**: `ReturnConverter`, `DataIntegrityCheckpoint`
- ✅ **Suffix conventions**:
  - Services: `*Service`, `*Manager`, `*Orchestrator`
  - Models: `*Model`, `*Engine`, `*Detector`
  - Data Access: `*Loader`, `*Converter`, `*Repository`
  - Utilities: `*Utils`, `*Helper` (use sparingly)

### Functions & Methods
- ✅ **Snake_case**: `convert_to_returns()`, `calculate_sharpe_ratio()`
- ✅ **Verb-based**: `get_*`, `calculate_*`, `validate_*`, `store_*`
- ✅ **Boolean returns**: `is_*`, `has_*`, `should_*`, `can_*`

### Constants
- ✅ **UPPER_SNAKE_CASE**: `MAX_RETRY_ATTEMPTS`, `DEFAULT_LOOKBACK_DAYS`

### Variables
- ✅ **Snake_case**: `returns_df`, `sharpe_ratio`, `current_state`
- ✅ **Common abbreviations OK**: `df` (DataFrame), `idx` (index)

---

## 🧪 TESTING STANDARDS (Mandatory)

### Test Organization
- Tests MUST mirror source structure: `tests/test_datasets/test_return_converter.py` → `src/cloud/training/datasets/return_converter.py`
- Test files: `test_*.py` or `*_test.py`
- Test functions: `test_*` (pytest convention)
- Test classes: `Test*` (for grouping)

### Coverage Requirements
- **Minimum: 80% coverage** for all new code
- **Every function MUST have**:
  1. Happy path test
  2. Edge case tests
  3. Error case tests
  4. Integration tests (where applicable)

### Test Quality Standards
```python
# ✅ GOOD: Comprehensive test
def test_convert_normal_case(self):
    """Test normal price to returns conversion."""
    converter = ReturnConverter()
    df = pl.DataFrame({...})
    result = converter.convert(df, price_column='close', symbol='BTC/USD')
    assert 'raw_returns' in result.columns
    assert len(result) == 100

def test_convert_missing_price_column(self):
    """Test error when price column is missing."""
    converter = ReturnConverter()
    df = pl.DataFrame({'timestamp': [1, 2, 3]})
    with pytest.raises(ValueError, match="Price column"):
        converter.convert(df, price_column='close', symbol='BTC/USD')
```

---

## 🔒 CODE QUALITY STANDARDS

### Type Hints (Required)
```python
# ✅ GOOD: Full type hints
def convert_to_returns(
    price_data: pl.DataFrame,
    price_column: str = 'close',
    symbol: Optional[str] = None,
    timestamp_column: str = 'timestamp'
) -> pl.DataFrame:
    """Convert price series to return series."""
    ...

# ❌ BAD: No type hints
def convert_to_returns(price_data, price_column='close'):
    ...
```

### Error Handling (Required)
```python
# ✅ GOOD: Explicit error handling
def convert_to_returns(...) -> pl.DataFrame:
    if price_column not in price_data.columns:
        raise ValueError(f"Price column '{price_column}' not found")
    
    try:
        result = ...
    except Exception as e:
        logger.error("return_conversion_failed", symbol=symbol, error=str(e))
        raise

# ❌ BAD: Bare except
try:
    result = risky_operation()
except:  # ❌ Catches everything
    pass
```

### Logging (Required)
```python
# ✅ GOOD: Structured logging
import structlog
logger = structlog.get_logger(__name__)

logger.info(
    "returns_converted",
    symbol=symbol,
    final_rows=len(result),
    mean_return=float(result['raw_returns'].mean())
)

# ❌ BAD: String formatting
logger.info(f"Converted returns for {symbol}: {len(result)} rows")
```

### Input Validation (Required)
```python
# ✅ GOOD: Always validate inputs
def convert_to_returns(
    price_data: pl.DataFrame,
    price_column: str = 'close'
) -> pl.DataFrame:
    if not isinstance(price_data, pl.DataFrame):
        raise TypeError(f"price_data must be polars DataFrame, got {type(price_data)}")
    
    if price_column not in price_data.columns:
        raise ValueError(f"Price column '{price_column}' not found")
    
    if len(price_data) == 0:
        logger.warning("empty_dataframe", symbol=symbol)
        return price_data
```

### Dependency Injection (Required)
```python
# ✅ GOOD: Inject dependencies
class ReturnConverter:
    def __init__(self, brain_library: Optional[BrainLibrary] = None):
        self.brain_library = brain_library

# ❌ BAD: Hard dependencies
class ReturnConverter:
    def __init__(self):
        self.brain_library = BrainLibrary()  # ❌ Hard dependency
```

---

## 📝 DOCUMENTATION STANDARDS (Required)

### Docstrings (Required for all public functions/classes)
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

### Module Documentation (Required)
```python
"""
Return Converter - Price to Return Conversion

Converts raw price series to total return series for normalized comparison.
Handles NaN cleaning, adjusted prices, and Brain Library integration.

Author: Huracan Engine Team
Date: 2025-11-12
"""
```

---

## 🔍 CODE REVIEW CHECKLIST

Before generating or modifying any code, I will ensure:

- [ ] **Type hints** on all functions
- [ ] **Docstrings** on all public functions/classes
- [ ] **Tests** for all new functions (80%+ coverage)
- [ ] **Error handling** for all edge cases
- [ ] **Logging** for important operations
- [ ] **Input validation** for all inputs
- [ ] **No hardcoded values** (use config/constants)
- [ ] **No circular dependencies**
- [ ] **Follows naming conventions**
- [ ] **Placed in correct directory/layer**
- [ ] **Dependency injection** used
- [ ] **Updated documentation** if architecture changes

---

## 🚨 VIOLATION HANDLING

If I detect architecture violations in existing code:

1. **Identify** the violation
2. **Document** the issue
3. **Suggest** a fix that aligns with architecture
4. **Implement** the fix if approved
5. **Update** documentation if needed

---

## 📊 METRICS & MONITORING

### Code Quality Metrics (Targets)
- **Test Coverage**: Minimum 80%
- **Type Coverage**: 100% (all functions typed)
- **Linting**: Zero errors (pylint, mypy)
- **Complexity**: Cyclomatic complexity < 10 per function

### Performance Metrics
- **Response Time**: Log all operations > 100ms
- **Memory Usage**: Monitor for memory leaks
- **Error Rate**: Track and alert on error rates > 1%

---

## 🗺️ ARCHITECTURE EVOLUTION

When architectural changes are needed:

1. **Document** the change in `ARCHITECTURE.md`
2. **Justify** why the change is necessary
3. **Update** this compliance document
4. **Migrate** existing code to new standards
5. **Update** all affected documentation

---

## ✅ COMMITMENT

I commit to:

1. **Always** follow the architecture standards
2. **Always** generate production-ready code
3. **Always** include comprehensive tests
4. **Always** provide proper documentation
5. **Always** validate inputs and handle errors
6. **Always** use structured logging
7. **Always** enforce type safety
8. **Always** maintain separation of concerns

**Violations will be rejected and corrected immediately.**

---

**Last Updated:** 2025-11-12  
**Maintained By:** AI Lead Software Architect  
**Architecture Version:** 2.0

