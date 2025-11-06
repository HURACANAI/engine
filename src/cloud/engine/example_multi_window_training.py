"""
Multi-Window Training Example

Demonstrates training different components with component-specific windows:
1. Scalp Core (60 days, 1m, 10-day halflife)
2. Confirm Filter (120 days, 5m, 20-day halflife)
3. Regime Classifier (365 days, 1m, 60-day halflife)
4. Risk Context (730 days, 1d, 120-day halflife)

Run this to see how each component gets its optimal training window.

Usage:
    python -m cloud.engine.example_multi_window_training
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import numpy as np
import structlog

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cloud.engine.multi_window import (
    create_all_component_configs,
    TrainingWindowManager,
    MultiWindowTrainer,
    print_training_results
)
from cloud.engine.multi_window.window_manager import print_window_summary
from cloud.engine.multi_window.component_configs import print_config_summary

logger = structlog.get_logger(__name__)


def create_sample_labeled_data(days: int = 800) -> pl.DataFrame:
    """
    Create sample labeled data spanning multiple timeframes.

    This simulates the output from the triple-barrier labeling pipeline.
    """
    logger.info("creating_sample_labeled_data", days=days)

    # Generate timestamps (1-minute candles)
    start_date = datetime.now() - timedelta(days=days)
    timestamps = [start_date + timedelta(minutes=i) for i in range(days * 24 * 60)]

    np.random.seed(42)
    n_samples = len(timestamps)

    # Generate features (these would come from your feature engineering)
    data = {
        'timestamp': timestamps,

        # Price features
        'close': 45000.0 + np.cumsum(np.random.normal(0, 50, n_samples)),
        'volume': np.random.uniform(100, 1000, n_samples),
        'volatility': np.random.uniform(5, 20, n_samples),

        # Technical features
        'rsi': np.random.uniform(20, 80, n_samples),
        'macd': np.random.normal(0, 10, n_samples),
        'bb_position': np.random.uniform(0, 1, n_samples),

        # Microstructure features
        'spread_bps': np.random.uniform(2, 5, n_samples),
        'order_flow': np.random.normal(0, 1000, n_samples),
        'tape_strength': np.random.uniform(-1, 1, n_samples),

        # Regime features
        'regime_trend': np.random.choice([0, 1, 2], n_samples),  # Bearish, neutral, bullish
        'regime_volatility': np.random.choice([0, 1], n_samples),  # Low, high

        # Labels (from triple-barrier)
        'primary_label': np.random.choice([0, 1], n_samples, p=[0.4, 0.6]),  # TP hit?
        'meta_label': np.random.choice([0, 1], n_samples, p=[0.45, 0.55]),  # Profitable after costs?
        'pnl_net_bps': np.random.normal(2, 10, n_samples),  # Net P&L

        # Trade metadata
        'exit_reason': np.random.choice(['tp', 'sl', 'timeout'], n_samples, p=[0.5, 0.3, 0.2]),
        'duration_minutes': np.random.uniform(5, 60, n_samples),
        'costs_bps': np.random.uniform(3, 8, n_samples),
    }

    df = pl.DataFrame(data)

    logger.info("sample_labeled_data_created", rows=len(df))

    return df


def dummy_train_function(window):
    """
    Dummy training function for demonstration.

    In production, this would be your actual model training:
    - XGBoost, LightGBM, Neural Network, etc.
    - Using window.data as features
    - Using window.weights as sample weights
    """
    from dataclasses import dataclass

    @dataclass
    class DummyModel:
        """Placeholder model for demonstration."""
        component_name: str
        n_samples: int
        n_features: int
        feature_importance: dict

    # Simulate training
    X = window.data.drop(['timestamp', 'meta_label'])
    y = window.data['meta_label']

    logger.info(
        "training_dummy_model",
        component=window.component_name,
        samples=len(X),
        features=len(X.columns),
        weight_range=(window.weights.min(), window.weights.max())
    )

    # In production, you'd do:
    # model = XGBClassifier()
    # model.fit(X, y, sample_weight=window.weights)

    # For demo, create dummy model
    model = DummyModel(
        component_name=window.component_name,
        n_samples=len(X),
        n_features=len(X.columns),
        feature_importance={col: np.random.uniform(0, 1) for col in X.columns[:5]}
    )

    return model


def dummy_validate_function(model, window):
    """
    Dummy validation function for demonstration.

    In production, this would compute real metrics:
    - Accuracy, precision, recall
    - Sharpe ratio on validation set
    - Drawdown, hit rate, etc.
    """
    # Simulate validation metrics
    metrics = {
        'accuracy': np.random.uniform(0.52, 0.65),
        'sharpe': np.random.uniform(0.8, 2.5),
        'hit_rate': np.random.uniform(0.48, 0.58),
        'avg_pnl_bps': np.random.uniform(1.5, 8.0)
    }

    logger.info(
        "validation_complete",
        component=model.component_name,
        metrics=metrics
    )

    return metrics


def run_multi_window_example():
    """
    Run complete multi-window training example.
    """
    print("\n" + "="*80)
    print("HURACAN V2 - MULTI-WINDOW TRAINING EXAMPLE")
    print("="*80 + "\n")

    # ==========================================
    # STEP 1: Show Component Configurations
    # ==========================================
    print("\n📋 STEP 1: Component configurations...\n")
    print_config_summary()

    # ==========================================
    # STEP 2: Create Sample Data
    # ==========================================
    print("\n📥 STEP 2: Creating sample labeled data...\n")

    # Create 800 days of data (covers all components)
    df = create_sample_labeled_data(days=800)

    print(f"   Created {len(df):,} labeled samples")
    print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"   Features: {len(df.columns)}")

    # ==========================================
    # STEP 3: Prepare Component Windows
    # ==========================================
    print("\n🪟 STEP 3: Preparing component-specific windows...\n")

    configs = create_all_component_configs()
    window_manager = TrainingWindowManager()

    windows = window_manager.prepare_all_components(
        data=df,
        configs=configs,
        create_walk_forward_splits=False
    )

    print_window_summary(windows)

    # Show weight distribution for one component
    scalp_window = windows['scalp_core']
    print(f"\n   Scalp Core weight distribution:")
    print(f"   - Min weight: {scalp_window.weights.min():.3f}")
    print(f"   - Max weight: {scalp_window.weights.max():.3f}")
    print(f"   - Mean weight: {scalp_window.weights.mean():.3f}")
    print(f"   - Effective samples: {scalp_window.effective_sample_size:.0f} (out of {scalp_window.total_samples})")

    # ==========================================
    # STEP 4: Train All Components
    # ==========================================
    print("\n🤖 STEP 4: Training all components...\n")

    trainer = MultiWindowTrainer()

    results = trainer.train_all_components(
        data=df,
        train_fn=dummy_train_function,
        configs=configs,
        validate_fn=dummy_validate_function
    )

    print_training_results(results)

    # ==========================================
    # STEP 5: Show Component Details
    # ==========================================
    print("\n📊 STEP 5: Component training details...\n")

    for name, component in results.components.items():
        print(f"\n{component.component_name.upper()}:")
        print(f"  Lookback: {component.config.lookback_days} days")
        print(f"  Timeframe: {component.config.timeframe}")
        print(f"  Recency halflife: {component.config.recency_halflife_days} days")
        print(f"  Training samples: {component.training_samples:,}")
        print(f"  Effective samples: {int(component.effective_samples):,}")
        print(f"  Training time: {component.training_time_seconds:.2f}s")
        print(f"  Validation metrics:")
        for metric, value in component.validation_metrics.items():
            print(f"    - {metric}: {value:.3f}")

    # ==========================================
    # STEP 6: Save Artifact (Optional)
    # ==========================================
    print("\n💾 STEP 6: Saving training artifact...\n")

    output_dir = Path(__file__).parent / "artifacts"
    artifact_path = trainer.save_artifact(
        results=results,
        output_dir=output_dir,
        metadata={
            'data_source': 'sample_data',
            'total_days': 800,
            'example_run': True
        }
    )

    print(f"   ✅ Artifact saved to: {artifact_path}")
    print(f"   Contains:")
    print(f"   - {len(results.components)} trained models")
    print(f"   - Training metadata (metadata.json)")
    print(f"   - Validation metrics")

    # ==========================================
    # STEP 7: Demonstrate Single Component Training
    # ==========================================
    print("\n🎯 STEP 7: Training single component (Regime Classifier)...\n")

    single_result = trainer.train_single_component(
        data=df,
        component_name='regime_classifier',
        train_fn=dummy_train_function,
        validate_fn=dummy_validate_function
    )

    print(f"   Component: {single_result.component_name}")
    print(f"   Samples: {single_result.training_samples:,}")
    print(f"   Training time: {single_result.training_time_seconds:.2f}s")
    print(f"   Validation accuracy: {single_result.validation_metrics['accuracy']:.3f}")

    # ==========================================
    # Summary
    # ==========================================
    print("\n" + "="*80)
    print("MULTI-WINDOW TRAINING COMPLETE!")
    print("="*80)
    print(f"""
✅ Components trained: {len(results.components)}
✅ Total training time: {results.total_training_time_seconds:.1f}s
✅ Artifact saved: {artifact_path}

Key Insights:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Each component trained on OPTIMAL window
   • Scalp core: {configs['scalp_core'].lookback_days} days (fast-changing microstructure)
   • Regime classifier: {configs['regime_classifier'].lookback_days} days (need full cycles)

2. Recency weighting applied per component
   • Scalp: {configs['scalp_core'].recency_halflife_days}-day halflife (recent data critical)
   • Risk: {configs['risk_context'].recency_halflife_days}-day halflife (correlations sticky)

3. Effective sample sizes reduced appropriately
   • Old data downweighted, not discarded
   • Maintains regime memory while focusing on recent patterns

4. Models packaged for deployment
   • All components in single artifact
   • Metadata preserved for auditing
   • Ready for Hamilton Pilot

Next Steps:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Replace dummy_train_function with real model training
   → XGBoost, LightGBM, Neural Network, etc.

2. Replace dummy_validate_function with real metrics
   → Walk-forward validation, Sharpe, drawdown, etc.

3. Integrate with your existing RLTrainingPipeline
   → Use MultiWindowTrainer as component trainer

4. Deploy artifact to Hamilton Pilot
   → Load all component models
   → Use in ensemble prediction

5. Set up Mechanic for hourly retraining
   → Incremental updates with new data
   → Track drift and trigger full retrains
""")


if __name__ == "__main__":
    # Configure logging
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer()
        ]
    )

    # Run example
    run_multi_window_example()
