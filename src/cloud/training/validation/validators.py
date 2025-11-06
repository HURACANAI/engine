"""Input validation utilities for trading system."""

from __future__ import annotations

from typing import Any, Dict, Optional
import structlog

logger = structlog.get_logger(__name__)


def validate_symbol(symbol: Any) -> str:
    """
    Validate trading symbol.
    
    Args:
        symbol: Symbol to validate
        
    Returns:
        Validated symbol string
        
    Raises:
        ValueError: If symbol is invalid
    """
    if not symbol or not isinstance(symbol, str):
        raise ValueError(f"Invalid symbol: must be non-empty string, got {type(symbol)}")
    if len(symbol.strip()) == 0:
        raise ValueError("Symbol cannot be empty or whitespace")
    return symbol.strip().upper()


def validate_price(price: Any, name: str = "price") -> float:
    """
    Validate price value.
    
    Args:
        price: Price to validate
        name: Name of the price field (for error messages)
        
    Returns:
        Validated price float
        
    Raises:
        ValueError: If price is invalid
    """
    if not isinstance(price, (int, float)):
        raise ValueError(f"Invalid {name}: must be numeric, got {type(price)}")
    if price <= 0:
        raise ValueError(f"Invalid {name}: must be positive, got {price}")
    if price > 1e10:  # Sanity check for unrealistic prices
        raise ValueError(f"Invalid {name}: unreasonably large value {price}")
    return float(price)


def validate_confidence(confidence: Any) -> float:
    """
    Validate confidence score.
    
    Args:
        confidence: Confidence to validate
        
    Returns:
        Validated confidence float (0-1)
        
    Raises:
        ValueError: If confidence is invalid
    """
    if not isinstance(confidence, (int, float)):
        raise ValueError(f"Invalid confidence: must be numeric, got {type(confidence)}")
    confidence = float(confidence)
    if not 0 <= confidence <= 1:
        raise ValueError(f"Invalid confidence: must be between 0 and 1, got {confidence}")
    return confidence


def validate_features(features: Any) -> Dict[str, float]:
    """
    Validate features dictionary.
    
    Args:
        features: Features to validate
        
    Returns:
        Validated features dictionary
        
    Raises:
        ValueError: If features are invalid
    """
    if not isinstance(features, dict):
        raise ValueError(f"Invalid features: must be dictionary, got {type(features)}")
    if len(features) == 0:
        raise ValueError("Features dictionary cannot be empty")
    
    validated = {}
    for key, value in features.items():
        if not isinstance(key, str):
            raise ValueError(f"Invalid feature key: must be string, got {type(key)}")
        if not isinstance(value, (int, float)):
            raise ValueError(f"Invalid feature value for '{key}': must be numeric, got {type(value)}")
        validated[key] = float(value)
    
    return validated


def validate_regime(regime: Any) -> str:
    """
    Validate market regime.
    
    Args:
        regime: Regime to validate
        
    Returns:
        Validated regime string
        
    Raises:
        ValueError: If regime is invalid
    """
    if not regime or not isinstance(regime, str):
        raise ValueError(f"Invalid regime: must be non-empty string, got {type(regime)}")
    
    valid_regimes = {'TREND', 'RANGE', 'PANIC', 'VOLATILE', 'CALM', 'trend', 'range', 'panic', 'volatile', 'calm'}
    regime_upper = regime.upper()
    if regime_upper not in valid_regimes:
        logger.warning("unknown_regime", regime=regime, valid_regimes=list(valid_regimes))
        # Don't raise, just warn - allow new regimes
    
    return regime_upper


def validate_size(size: Any, name: str = "size") -> float:
    """
    Validate position size.
    
    Args:
        size: Size to validate
        name: Name of the size field (for error messages)
        
    Returns:
        Validated size float
        
    Raises:
        ValueError: If size is invalid
    """
    if not isinstance(size, (int, float)):
        raise ValueError(f"Invalid {name}: must be numeric, got {type(size)}")
    if size <= 0:
        raise ValueError(f"Invalid {name}: must be positive, got {size}")
    if size > 1e9:  # Sanity check for unrealistic sizes
        raise ValueError(f"Invalid {name}: unreasonably large value {size}")
    return float(size)

