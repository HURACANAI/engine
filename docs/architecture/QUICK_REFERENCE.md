# Architecture Quick Reference Guide

**For:** Lead Software Architect & Development Team  
**Version:** 2.0  
**Last Updated:** 2025-11-12

---

## 🚀 Quick Decision Tree

### Where should this code go?

```
Is it workflow orchestration?
  → pipelines/

Is it business logic / trading algorithm?
  → models/

Is it RL agent code?
  → agents/

Is it application service / coordination?
  → services/

Is it data loading / transformation?
  → datasets/

Is it ML infrastructure (training/inference)?
  → ml_framework/

Is it portfolio management?
  → portfolio/

Is it order execution?
  → execution/

Is it performance metrics?
  → metrics/

Is it validation / testing?
  → validation/

Is it health / monitoring?
  → monitoring/

Is it external service integration?
  → integrations/
```

---

## 📝 Code Generation Checklist

Before generating ANY code, ensure:

- [ ] **Correct directory** - Matches layer responsibility
- [ ] **Correct naming** - snake_case files, PascalCase classes
- [ ] **Type hints** - All function parameters and returns
- [ ] **Docstrings** - All public functions/classes
- [ ] **Error handling** - Explicit exceptions, no bare `except:`
- [ ] **Input validation** - Validate all inputs
- [ ] **Logging** - Use `structlog`, structured logging
- [ ] **Dependency injection** - No hard dependencies
- [ ] **Tests** - Create matching test file in `/tests/`

---

## 🏷️ Naming Quick Reference

| Type | Convention | Example |
|------|-----------|---------|
| File | `snake_case.py` | `return_converter.py` |
| Class | `PascalCase` | `ReturnConverter` |
| Function | `snake_case()` | `convert_to_returns()` |
| Method | `snake_case()` | `calculate_sharpe()` |
| Constant | `UPPER_SNAKE_CASE` | `MAX_RETRY_ATTEMPTS` |
| Variable | `snake_case` | `returns_df` |
| Private | `_leading_underscore` | `_internal_method()` |

---

## 📋 Template: New Module

```python
"""
Module Name - Brief Description

Detailed description of what this module does.
Handles X, Y, and Z.

Author: Huracan Engine Team
Date: YYYY-MM-DD
"""

from typing import Optional, List, Dict
import structlog

logger = structlog.get_logger(__name__)


class NewModule:
    """
    Brief description of the class.
    
    Usage:
        module = NewModule(dependency=dep)
        result = module.do_something(input_data)
    
    Attributes:
        dependency: Description of dependency
    """
    
    def __init__(self, dependency: Optional[SomeType] = None):
        """
        Initialize the module.
        
        Args:
            dependency: Optional dependency (injected, not created)
        """
        self.dependency = dependency
    
    def do_something(
        self,
        input_data: SomeType,
        param: str = "default"
    ) -> ResultType:
        """
        Do something with input data.
        
        Args:
            input_data: Description of input
            param: Description of parameter
        
        Returns:
            Description of return value
        
        Raises:
            ValueError: If input is invalid
            TypeError: If input type is wrong
        
        Example:
            >>> module = NewModule()
            >>> result = module.do_something(data, param="value")
            >>> assert result is not None
        """
        # Validate inputs
        if not isinstance(input_data, SomeType):
            raise TypeError(f"input_data must be SomeType, got {type(input_data)}")
        
        if not input_data:
            raise ValueError("input_data cannot be empty")
        
        try:
            # Main logic
            logger.info("operation_started", param=param)
            result = self._process(input_data, param)
            logger.info("operation_completed", result_size=len(result))
            return result
        except Exception as e:
            logger.error(
                "operation_failed",
                error=str(e),
                error_type=type(e).__name__
            )
            raise
    
    def _process(self, data: SomeType, param: str) -> ResultType:
        """Internal processing method."""
        # Implementation
        pass
```

---

## 🧪 Template: Test File

```python
"""
Tests for module_name.

Tests:
1. Normal operation
2. Edge cases
3. Error cases
4. Integration scenarios
"""

import pytest
from typing import Optional
import polars as pl

from src.cloud.training.module_name import NewModule


class TestNewModule:
    """Test suite for NewModule."""
    
    @pytest.fixture
    def module(self):
        """Create module instance for testing."""
        return NewModule()
    
    def test_do_something_normal_case(self, module):
        """Test normal operation."""
        input_data = create_test_data()
        result = module.do_something(input_data, param="value")
        
        assert result is not None
        assert len(result) > 0
    
    def test_do_something_empty_input(self, module):
        """Test error handling for empty input."""
        with pytest.raises(ValueError, match="cannot be empty"):
            module.do_something(None)
    
    def test_do_something_invalid_type(self, module):
        """Test error handling for invalid type."""
        with pytest.raises(TypeError):
            module.do_something("invalid_type")
    
    def test_do_something_edge_case(self, module):
        """Test edge case handling."""
        # Test with minimal valid input
        minimal_data = create_minimal_data()
        result = module.do_something(minimal_data)
        assert result is not None
```

---

## 🔍 Common Patterns

### Dependency Injection
```python
# ✅ GOOD
class Service:
    def __init__(self, data_loader: Optional[DataLoader] = None):
        self.data_loader = data_loader or DataLoader()

# ❌ BAD
class Service:
    def __init__(self):
        self.data_loader = DataLoader()  # Hard dependency
```

### Error Handling
```python
# ✅ GOOD
try:
    result = risky_operation()
except ValueError as e:
    logger.warning("invalid_input", error=str(e))
    raise
except Exception as e:
    logger.error("unexpected_error", error=str(e))
    raise

# ❌ BAD
try:
    result = risky_operation()
except:  # Catches everything
    pass
```

### Logging
```python
# ✅ GOOD
logger.info(
    "operation_completed",
    symbol=symbol,
    rows_processed=len(result),
    duration_ms=duration
)

# ❌ BAD
logger.info(f"Operation completed for {symbol}: {len(result)} rows")
```

### Input Validation
```python
# ✅ GOOD
def process_data(data: pl.DataFrame) -> pl.DataFrame:
    if not isinstance(data, pl.DataFrame):
        raise TypeError(f"data must be DataFrame, got {type(data)}")
    
    if len(data) == 0:
        logger.warning("empty_dataframe")
        return data
    
    # Process...

# ❌ BAD
def process_data(data):
    # No validation
    return data.process()
```

---

## 🚨 Red Flags (Never Do This)

1. ❌ **Hard dependencies** - Creating dependencies inside classes
2. ❌ **Bare except** - `except:` without specific exception
3. ❌ **No type hints** - Functions without type annotations
4. ❌ **No docstrings** - Public functions/classes without docs
5. ❌ **Wrong directory** - Code in wrong layer/directory
6. ❌ **Hardcoded values** - URLs, keys, paths in code
7. ❌ **String logging** - Using f-strings instead of structured logging
8. ❌ **No validation** - Functions that don't validate inputs
9. ❌ **Circular dependencies** - Modules importing each other
10. ❌ **Generic names** - `utils.py`, `helpers.py`, `misc.py`

---

## ✅ Green Flags (Always Do This)

1. ✅ **Dependency injection** - Inject all dependencies
2. ✅ **Explicit exceptions** - `except ValueError as e:`
3. ✅ **Full type hints** - All parameters and returns typed
4. ✅ **Comprehensive docstrings** - Args, Returns, Raises, Examples
5. ✅ **Correct directory** - Code in appropriate layer
6. ✅ **Config-based** - Values from config/env vars
7. ✅ **Structured logging** - `logger.info("event", key=value)`
8. ✅ **Input validation** - Validate all inputs
9. ✅ **Clear dependencies** - One-way dependencies only
10. ✅ **Descriptive names** - `fee_latency_calibration.py` not `fees.py`

---

## 🔧 Tools & Commands

### Check Architecture Compliance
```bash
python scripts/check_architecture_compliance.py
```

### Run Tests
```bash
# All tests
pytest tests/

# With coverage
pytest tests/ --cov=src --cov-report=html

# Specific test
pytest tests/test_datasets/test_return_converter.py
```

### Type Checking
```bash
# Pyright (configured in pyrightconfig.json)
pyright src/

# MyPy
mypy src/
```

### Linting
```bash
# Pylint
pylint src/

# Flake8
flake8 src/
```

---

## 📚 Key Documents

- **Full Architecture**: `docs/architecture/ARCHITECTURE.md`
- **Compliance Guide**: `docs/architecture/ARCHITECT_COMPLIANCE.md`
- **This Quick Reference**: `docs/architecture/QUICK_REFERENCE.md`

---

**Remember:** When in doubt, refer to `ARCHITECTURE.md` for the authoritative source.

