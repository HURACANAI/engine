"""
AI-Generated Alpha Engines

This package contains AlphaEngines automatically generated from
the strategy-research pipeline (moon-dev RBI agent).
"""

from pathlib import Path
from typing import List, Dict, Any
import importlib.util
import sys

from .adapter import AIGeneratedEngineAdapter

__all__ = ["load_ai_engines", "load_ai_engines_with_adapters", "AIGeneratedEngineAdapter"]


def load_ai_engines(status_filter: str = "approved") -> List:
    """
    Dynamically load all AI-generated engines from this directory.

    Args:
        status_filter: Only load engines with this status
                      ("approved", "testing", "all")

    Returns:
        List of instantiated AlphaEngine instances (raw, not adapted)
    """
    engines = []
    engine_dir = Path(__file__).parent

    # Find all Python files (except __init__.py, README.md, and adapter.py)
    engine_files = [
        f for f in engine_dir.glob("*.py")
        if f.name not in ["__init__.py", "adapter.py"] and not f.name.startswith("_")
    ]

    print(f"\n🔍 Searching for AI-generated engines in {engine_dir.name}/")
    print(f"   Found {len(engine_files)} potential engine files")

    for engine_file in engine_files:
        try:
            # Load module dynamically
            # Try to use the package structure if possible
            try:
                # First, try to import as part of the package
                module_name = f"cloud.training.models.ai_generated_engines.{engine_file.stem}"
                # Add parent directories to path if needed
                parent_dir = str(Path(__file__).parent.parent.parent.parent.parent)
                if parent_dir not in sys.path:
                    sys.path.insert(0, parent_dir)
                module = importlib.import_module(module_name)
            except (ImportError, ModuleNotFoundError):
                # Fallback: load directly from file
                module_name = f"ai_generated_engines_{engine_file.stem}"
                spec = importlib.util.spec_from_file_location(module_name, engine_file)
                if spec is None or spec.loader is None:
                    print(f"   ❌ Failed to create spec for {engine_file.name}")
                    continue
            module = importlib.util.module_from_spec(spec)
                # Add parent directories to path for absolute imports
                parent_dir = str(Path(__file__).parent.parent.parent.parent.parent)
                if parent_dir not in sys.path:
                    sys.path.insert(0, parent_dir)
                sys.modules[module_name] = module
            spec.loader.exec_module(module)

            # Find AlphaEngine subclass
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (
                    isinstance(attr, type)
                    and attr_name.startswith("AIGeneratedEngine")
                    and attr_name != "AlphaEngine"
                ):
                    # Check status (if metadata available)
                    if hasattr(attr, "METADATA"):
                        metadata = attr.METADATA
                        if status_filter != "all" and metadata.get("status") != status_filter:
                            print(f"   ⏭️  Skipping {attr_name} (status: {metadata.get('status')})")
                            continue

                    # Instantiate engine
                    engine_instance = attr()
                    engines.append(engine_instance)
                    print(f"   ✅ Loaded {attr_name}")

        except Exception as e:
            print(f"   ❌ Failed to load {engine_file.name}: {e}")

    print(f"\n📊 Loaded {len(engines)} AI-generated engines")
    return engines


def load_ai_engines_with_adapters(
    status_filter: str = "approved",
    symbol: str = "UNKNOWN"
) -> Dict[str, AIGeneratedEngineAdapter]:
    """
    Load AI-generated engines and wrap them in adapters.

    Args:
        status_filter: Only load engines with this status
        symbol: Symbol to use for engines (default: "UNKNOWN")

    Returns:
        Dict mapping engine names to adapted engine instances
    """
    raw_engines = load_ai_engines(status_filter=status_filter)
    adapted_engines = {}
    
    from ..alpha_engines import TradingTechnique
    
    for engine in raw_engines:
        # Infer technique from metadata
        technique = TradingTechnique.RANGE  # Default
        if hasattr(engine, "METADATA"):
            strategy_type = engine.METADATA.get("strategy_type", "").lower()
            if "trend" in strategy_type:
                technique = TradingTechnique.TREND
            elif "reversal" in strategy_type or "range" in strategy_type:
                technique = TradingTechnique.RANGE
            elif "breakout" in strategy_type:
                technique = TradingTechnique.BREAKOUT
        
        # Create adapter
        engine_name = getattr(engine, "name", f"ai_engine_{len(adapted_engines)}")
        adapter = AIGeneratedEngineAdapter(
            ai_engine=engine,
            symbol=symbol,
            technique=technique
        )
        adapted_engines[engine_name] = adapter
    
    return adapted_engines


# Example metadata structure for AI-generated engines:
"""
class AIGeneratedEngine_Example(AlphaEngine):
    METADATA = {
        "source": "rbi_agent",
        "generation_date": "2025-11-08",
        "backtest_return": 8.5,
        "backtest_sharpe": 1.8,
        "validation_status": "passed",
        "status": "approved",  # pending, testing, approved, deprecated
        "strategy_type": "reversal",
        "description": "RSI-based reversal strategy"
    }

    def __init__(self):
        super().__init__(name="example_strategy")

    # ... rest of implementation
"""
