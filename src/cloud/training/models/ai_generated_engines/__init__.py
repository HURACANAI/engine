"""
AI-Generated Alpha Engines

This package contains AlphaEngines automatically generated from
the strategy-research pipeline (moon-dev RBI agent).
"""

from pathlib import Path
from typing import List
import importlib.util
import sys

__all__ = ["load_ai_engines"]


def load_ai_engines(status_filter: str = "approved") -> List:
    """
    Dynamically load all AI-generated engines from this directory.

    Args:
        status_filter: Only load engines with this status
                      ("approved", "testing", "all")

    Returns:
        List of instantiated AlphaEngine instances
    """
    engines = []
    engine_dir = Path(__file__).parent

    # Find all Python files (except __init__.py and README.md)
    engine_files = [
        f for f in engine_dir.glob("*.py")
        if f.name != "__init__.py" and not f.name.startswith("_")
    ]

    print(f"\n🔍 Searching for AI-generated engines in {engine_dir.name}/")
    print(f"   Found {len(engine_files)} potential engine files")

    for engine_file in engine_files:
        try:
            # Load module dynamically
            spec = importlib.util.spec_from_file_location(
                f"ai_generated_engines.{engine_file.stem}",
                engine_file
            )
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
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
