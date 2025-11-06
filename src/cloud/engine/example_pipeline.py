"""
Complete Data Pipeline Example

This script demonstrates the FULL Huracan V2 data pipeline:
1. Load raw candle data
2. Clean with DataSanityPipeline
3. Label with TripleBarrierLabeler
4. Apply meta-labeling
5. Calculate recency weights
6. Validate with walk-forward

Run this to test the pipeline end-to-end!

Usage:
    python -m cloud.engine.example_pipeline
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import structlog

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cloud.engine.data_quality import DataSanityPipeline, format_sanity_report
from cloud.engine.labeling import (
    ScalpLabelConfig,
    TripleBarrierLabeler,
    MetaLabeler,
    print_label_distribution
)
from cloud.engine.labeling.triple_barrier import print_label_statistics
from cloud.engine.costs import CostEstimator, print_tca_report
from cloud.engine.weighting import RecencyWeighter, create_mode_specific_weighter
from cloud.engine.walk_forward import WalkForwardValidator, print_walk_forward_results

logger = structlog.get_logger(__name__)


def create_sample_data(days: int = 90) -> pl.DataFrame:
    """
    Create sample candle data for testing.

    In production, you'd load from your CandleDataLoader.
    """
    import numpy as np

    logger.info("creating_sample_data", days=days)

    # Generate timestamps (1-minute candles)
    start_date = datetime.now() - timedelta(days=days)
    timestamps = [start_date + timedelta(minutes=i) for i in range(days * 24 * 60)]

    # Generate realistic price data (random walk with drift)
    np.random.seed(42)
    initial_price = 45000.0  # BTC price
    returns = np.random.normal(0.0001, 0.01, len(timestamps))  # Small drift, moderate vol
    prices = initial_price * np.exp(np.cumsum(returns))

    # OHLCV
    data = {
        'timestamp': timestamps,
        'open': prices,
        'high': prices * (1 + np.abs(np.random.normal(0, 0.002, len(timestamps)))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.002, len(timestamps)))),
        'close': prices,
        'volume': np.random.uniform(100, 1000, len(timestamps)),
    }

    df = pl.DataFrame(data)

    # Add technical features (needed for cost estimation)
    df = df.with_columns([
        pl.lit(3.0).alias('spread_bps'),  # 3 bps spread
        pl.lit(10.0).alias('atr_bps'),    # 10 bps ATR
    ])

    logger.info("sample_data_created", rows=len(df))

    return df


def run_complete_pipeline():
    """
    Run the complete data pipeline.

    This is what you'd integrate into your training pipeline.
    """
    print("\n" + "="*60)
    print("HURACAN V2 DATA PIPELINE - COMPLETE EXAMPLE")
    print("="*60 + "\n")

    # ==========================================
    # STEP 1: Create/Load Data
    # ==========================================
    print("\n📥 STEP 1: Loading data...")
    df = create_sample_data(days=90)
    print(f"   Loaded {len(df):,} candles")

    # ==========================================
    # STEP 2: Data Quality
    # ==========================================
    print("\n🧹 STEP 2: Cleaning data...")
    sanity_pipeline = DataSanityPipeline(
        exchange='binance',
        outlier_threshold_pct=0.10,
        max_gap_minutes=5,
        apply_fees=True
    )

    clean_df, sanity_report = sanity_pipeline.clean(df)
    print(format_sanity_report(sanity_report))

    # ==========================================
    # STEP 3: Triple-Barrier Labeling
    # ==========================================
    print("\n🏷️  STEP 3: Labeling trades (triple-barrier)...")

    # Create scalp config
    scalp_config = ScalpLabelConfig(
        tp_bps=15.0,
        sl_bps=10.0,
        timeout_minutes=30
    )

    # Initialize labeler with cost estimator
    cost_estimator = CostEstimator(exchange='binance')
    labeler = TripleBarrierLabeler(
        config=scalp_config,
        cost_estimator=cost_estimator
    )

    # Label first 500 potential entries (for speed)
    labeled_trades = labeler.label_dataframe(
        df=clean_df,
        symbol='BTC/USDT',
        max_labels=500
    )

    # Print statistics
    stats = labeler.get_statistics(labeled_trades)
    print_label_statistics(stats)

    # ==========================================
    # STEP 4: Meta-Labeling
    # ==========================================
    print("\n🎯 STEP 4: Applying meta-labeling...")

    meta_labeler = MetaLabeler(
        cost_threshold_bps=5.0,  # Must beat costs by 5 bps
        min_pnl_bps=0.0
    )

    labeled_trades = meta_labeler.apply(labeled_trades)

    distribution = meta_labeler.get_label_distribution(labeled_trades)
    print_label_distribution(distribution)

    # ==========================================
    # STEP 5: Recency Weighting
    # ==========================================
    print("\n⚖️  STEP 5: Calculating recency weights...")

    weighter = create_mode_specific_weighter(mode='scalp')
    weights = weighter.calculate_weights_from_labels(labeled_trades)

    print(f"   Weight range: {weights.min():.3f} - {weights.max():.3f}")
    print(f"   Average weight: {weights.mean():.3f}")
    print(f"   Effective sample size: {weighter.get_effective_sample_size(weights):.0f}")

    # Show decay curve
    weighter.plot_decay_curve()

    # ==========================================
    # STEP 6: Walk-Forward Validation
    # ==========================================
    print("\n📊 STEP 6: Walk-forward validation...")

    validator = WalkForwardValidator(
        train_days=30,
        test_days=1,
        embargo_minutes=30
    )

    results = validator.validate_with_labels(labeled_trades)
    print_walk_forward_results(results)

    # Check for overfitting
    overfit_check = validator.detect_overfitting(results)
    print(f"\n{overfit_check['recommendation']}\n")

    # ==========================================
    # STEP 7: Sample TCA Report
    # ==========================================
    print("\n💰 STEP 7: Sample cost analysis...")

    if labeled_trades:
        first_trade = labeled_trades[0]
        tca_report = cost_estimator.estimate_detailed(
            entry_row=clean_df[first_trade.entry_idx],
            exit_time=first_trade.exit_time,
            duration_minutes=first_trade.duration_minutes,
            mode='scalp'
        )
        print_tca_report(tca_report)

    # ==========================================
    # Summary
    # ==========================================
    print("\n" + "="*60)
    print("PIPELINE COMPLETE!")
    print("="*60)
    print(f"""
✅ Data cleaned: {len(clean_df):,} candles
✅ Trades labeled: {len(labeled_trades):,}
✅ Winners: {sum(1 for t in labeled_trades if t.meta_label == 1):,} ({distribution['win_rate']:.1%})
✅ Recency weights calculated
✅ Walk-forward validated: {results.test_sharpe:.2f} Sharpe
✅ Ready for model training!

Next steps:
1. Integrate this pipeline into your RLTrainingPipeline
2. Use labeled_trades as training data
3. Apply weights during model.fit()
4. Use walk-forward results to validate before deployment
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

    # Run pipeline
    run_complete_pipeline()
