#!/usr/bin/env python3
"""
Integration Check Script

Checks for conflicts, duplicate implementations, and import issues
between changes made by two people.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def check_imports():
    """Check for import conflicts"""
    print("=" * 80)
    print("CHECKING IMPORTS")
    print("=" * 80)
    
    issues = []
    
    # Check ExecutionSimulator
    try:
        from cloud.training.simulation import ExecutionSimulator as SimExecutionSimulator
        print("✓ simulation.ExecutionSimulator imports successfully")
    except Exception as e:
        issues.append(f"✗ simulation.ExecutionSimulator import failed: {e}")
        print(f"✗ simulation.ExecutionSimulator import failed: {e}")
    
    # Check if backtesting_framework ExecutionSimulator conflicts
    try:
        from cloud.training.models.backtesting_framework import ExecutionSimulator as BacktestExecutionSimulator
        print("✓ backtesting_framework.ExecutionSimulator exists (internal use only)")
    except Exception as e:
        print(f"✗ backtesting_framework.ExecutionSimulator: {e}")
    
    # Check FeatureStore
    try:
        from shared.features import FeatureStore
        print("✓ shared.features.FeatureStore imports successfully")
    except Exception as e:
        issues.append(f"✗ shared.features.FeatureStore import failed: {e}")
        print(f"✗ shared.features.FeatureStore import failed: {e}")
    
    # Check if feature_store.py exists but isn't imported
    feature_store_path = Path("src/shared/features/feature_store.py")
    if feature_store_path.exists():
        print(f"⚠ feature_store.py exists but may not be used (store.py is preferred)")
    
    # Check CounterfactualEvaluator
    try:
        from cloud.training.evaluation import CounterfactualEvaluator
        print("✓ evaluation.CounterfactualEvaluator imports successfully")
    except Exception as e:
        issues.append(f"✗ evaluation.CounterfactualEvaluator import failed: {e}")
        print(f"✗ evaluation.CounterfactualEvaluator import failed: {e}")
    
    # Check models/counterfactual (different location)
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "counterfactual_evaluator",
            Path("models/counterfactual/evaluator.py")
        )
        if spec and spec.loader:
            print("✓ models/counterfactual/evaluator.py exists (different namespace)")
    except Exception as e:
        print(f"ℹ models/counterfactual/evaluator.py: {e}")
    
    # Check LiveSimulator
    try:
        from cloud.training.simulation import LiveSimulator
        print("✓ simulation.LiveSimulator imports successfully")
    except Exception as e:
        issues.append(f"✗ simulation.LiveSimulator import failed: {e}")
        print(f"✗ simulation.LiveSimulator import failed: {e}")
    
    # Check new components
    new_components = [
        ("cloud.training.validation.enhanced_walk_forward", "EnhancedWalkForwardTester"),
        ("cloud.training.validation.feature_drift_detector", "FeatureDriftDetector"),
        ("cloud.training.validation.robustness_analyzer", "RobustnessAnalyzer"),
        ("cloud.training.models.regime_aware_models", "RegimeAwareModelSystem"),
        ("cloud.training.attribution.trade_attribution", "TradeAttributionSystem"),
        ("cloud.training.datasets.enhanced_data_pipeline", "EnhancedDataPipeline"),
        ("cloud.training.features.dynamic_feature_engine", "DynamicFeatureEngine"),
        ("cloud.training.ml_framework.model_factory_pytorch", "PyTorchModelFactory"),
        ("cloud.training.ml_framework.inference.compiled_inference", "CompiledInferenceLayer"),
        ("cloud.training.orderbook.in_memory_orderbook", "InMemoryOrderBook"),
        ("cloud.training.analytics.daily_win_loss_analytics", "DailyWinLossAnalytics"),
        ("cloud.training.learning.continuous_learning", "ContinuousLearningSystem"),
        ("cloud.training.pipelines.event_driven_pipeline", "EventDrivenPipeline"),
    ]
    
    print("\n" + "=" * 80)
    print("CHECKING NEW COMPONENTS")
    print("=" * 80)
    
    for module_name, class_name in new_components:
        try:
            module = __import__(module_name, fromlist=[class_name])
            cls = getattr(module, class_name)
            print(f"✓ {module_name}.{class_name} imports successfully")
        except ImportError as e:
            issues.append(f"✗ {module_name}.{class_name} import failed: {e}")
            print(f"✗ {module_name}.{class_name} import failed: {e}")
        except AttributeError as e:
            issues.append(f"✗ {module_name}.{class_name} not found: {e}")
            print(f"✗ {module_name}.{class_name} not found: {e}")
        except Exception as e:
            issues.append(f"✗ {module_name}.{class_name} error: {e}")
            print(f"✗ {module_name}.{class_name} error: {e}")
    
    return issues

def check_duplicate_files():
    """Check for duplicate file implementations"""
    print("\n" + "=" * 80)
    print("CHECKING FOR DUPLICATE FILES")
    print("=" * 80)
    
    duplicates = []
    
    # Check for duplicate FeatureStore files
    if Path("src/shared/features/store.py").exists() and Path("src/shared/features/feature_store.py").exists():
        duplicates.append(("FeatureStore", "src/shared/features/store.py", "src/shared/features/feature_store.py"))
        print("⚠ Found duplicate FeatureStore implementations:")
        print("  - src/shared/features/store.py (database-backed, used in __init__.py)")
        print("  - src/shared/features/feature_store.py (file-based, not imported)")
    
    # Check ExecutionSimulator (different purposes, different locations - OK)
    if Path("src/cloud/training/simulation/execution_simulator.py").exists() and \
       Path("src/cloud/training/models/backtesting_framework.py").exists():
        print("ℹ ExecutionSimulator exists in two places (different purposes):")
        print("  - src/cloud/training/simulation/execution_simulator.py (exported, slippage learning)")
        print("  - src/cloud/training/models/backtesting_framework.py (internal, backtesting)")
    
    # Check CounterfactualEvaluator (different locations - OK)
    if Path("src/cloud/training/evaluation/counterfactual_evaluator.py").exists() and \
       Path("models/counterfactual/evaluator.py").exists():
        print("ℹ CounterfactualEvaluator exists in two places (different purposes):")
        print("  - src/cloud/training/evaluation/counterfactual_evaluator.py (exit/sizing optimization)")
        print("  - models/counterfactual/evaluator.py (general counterfactual evaluation)")
    
    return duplicates

def check_init_files():
    """Check __init__.py files for proper exports"""
    print("\n" + "=" * 80)
    print("CHECKING __init__.py FILES")
    print("=" * 80)
    
    issues = []
    
    init_files = [
        "src/cloud/training/simulation/__init__.py",
        "src/cloud/training/validation/__init__.py",
        "src/cloud/training/attribution/__init__.py",
        "src/cloud/training/features/__init__.py",
        "src/cloud/training/orderbook/__init__.py",
        "src/cloud/training/ml_framework/__init__.py",
        "src/cloud/training/analytics/__init__.py",
        "src/cloud/training/learning/__init__.py",
        "src/cloud/training/pipelines/__init__.py",
        "src/shared/features/__init__.py",
    ]
    
    for init_file in init_files:
        path = Path(init_file)
        if path.exists():
            print(f"✓ {init_file} exists")
            try:
                # Try to parse it
                with open(path, 'r') as f:
                    content = f.read()
                    if '__all__' in content:
                        print(f"  - Has __all__ export list")
                    else:
                        print(f"  - No __all__ export list (may be intentional)")
            except Exception as e:
                issues.append(f"✗ Error reading {init_file}: {e}")
        else:
            issues.append(f"✗ {init_file} missing")
            print(f"✗ {init_file} missing")
    
    return issues

def main():
    """Run all checks"""
    print("\n" + "=" * 80)
    print("INTEGRATION CHECK")
    print("=" * 80)
    print("\nChecking for conflicts and integration issues...\n")
    
    import_issues = check_imports()
    duplicates = check_duplicate_files()
    init_issues = check_init_files()
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    total_issues = len(import_issues) + len(duplicates) + len(init_issues)
    
    if total_issues == 0:
        print("✓ No critical issues found!")
        print("✓ All imports working correctly")
        print("✓ No conflicting duplicate implementations")
        return 0
    else:
        print(f"✗ Found {total_issues} issue(s):")
        for issue in import_issues + init_issues:
            print(f"  {issue}")
        if duplicates:
            print(f"  {len(duplicates)} duplicate file(s) found (see details above)")
        return 1

if __name__ == "__main__":
    sys.exit(main())

