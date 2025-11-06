"""
V2 Pipeline Integration Example

Demonstrates how to use the V2 data pipeline with existing training code.

Three approaches shown:
1. Drop-in replacement: RLTrainingPipelineV2
2. Adapter integration: Use V2PipelineAdapter directly
3. Gradual migration: Mix V2 and legacy components

Run this to see V2 improvements in action.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

import structlog

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from cloud.training.pipelines.rl_training_pipeline_v2 import RLTrainingPipelineV2
from cloud.training.integrations.v2_pipeline_adapter import (
    V2PipelineAdapter,
    V2PipelineConfig,
    create_v2_scalp_adapter,
    create_v2_runner_adapter
)
from cloud.engine.example_pipeline import create_sample_data

logger = structlog.get_logger(__name__)


def example_1_drop_in_replacement():
    """
    Example 1: Drop-in Replacement

    Replace RLTrainingPipeline with RLTrainingPipelineV2.
    Everything else stays the same.
    """
    print("\n" + "="*80)
    print("EXAMPLE 1: DROP-IN REPLACEMENT")
    print("="*80 + "\n")

    print("OLD CODE:")
    print("─" * 80)
    print("""
    from cloud.training.pipelines.rl_training_pipeline import RLTrainingPipeline

    pipeline = RLTrainingPipeline(settings=settings, dsn=dsn)
    metrics = pipeline.train_on_symbol(
        symbol='BTC/USDT',
        exchange_client=exchange,
        lookback_days=365
    )
    """)

    print("\nNEW CODE (V2):")
    print("─" * 80)
    print("""
    from cloud.training.pipelines.rl_training_pipeline_v2 import RLTrainingPipelineV2

    pipeline = RLTrainingPipelineV2(
        settings=settings,
        dsn=dsn,
        use_v2_pipeline=True,  # ← Enable V2 features
        trading_mode='scalp'    # ← 'scalp' or 'runner'
    )

    metrics = pipeline.train_on_symbol(
        symbol='BTC/USDT',
        exchange_client=exchange,
        lookback_days=365
    )

    # Metrics now include V2 stats:
    # - v2_total_labels
    # - v2_profitable_labels
    # - v2_avg_net_pnl_bps
    # - v2_avg_costs_bps
    # - v2_effective_samples (if recency weighting enabled)
    """)

    print("\nKEY BENEFITS:")
    print("─" * 80)
    print("✅ No lookahead bias (triple-barrier labeling)")
    print("✅ Realistic costs (fees + spread + slippage)")
    print("✅ Cost-aware labels (meta-labeling)")
    print("✅ Recent data weighted higher (recency weighting)")
    print("✅ Clean data (dedup, outliers, gaps)")
    print("\n")


def example_2_adapter_integration():
    """
    Example 2: Adapter Integration

    Use V2PipelineAdapter directly for custom workflows.
    """
    print("\n" + "="*80)
    print("EXAMPLE 2: ADAPTER INTEGRATION")
    print("="*80 + "\n")

    print("Use V2PipelineAdapter for custom training workflows:\n")

    # Create sample data
    print("Creating sample data...")
    df = create_sample_data(days=90)
    print(f"  ✓ Created {len(df):,} candles\n")

    # Create adapter
    print("Initializing V2 adapter (scalp mode)...")
    adapter = create_v2_scalp_adapter(exchange='binance')
    print("  ✓ Adapter initialized\n")

    # Process data
    print("Processing through V2 pipeline...")
    labeled_trades, weights = adapter.process(
        data=df,
        symbol='BTC/USDT'
    )
    print(f"  ✓ Generated {len(labeled_trades):,} labeled trades\n")

    # Show statistics
    profitable = sum(1 for t in labeled_trades if t.meta_label == 1)
    avg_net_pnl = sum(t.pnl_net_bps for t in labeled_trades) / len(labeled_trades) if labeled_trades else 0
    avg_costs = sum(t.costs_bps for t in labeled_trades) / len(labeled_trades) if labeled_trades else 0

    print("V2 Pipeline Results:")
    print("─" * 80)
    print(f"Total trades:        {len(labeled_trades):,}")
    print(f"Profitable trades:   {profitable:,} ({profitable/len(labeled_trades)*100:.1f}%)")
    print(f"Avg net P&L:         {avg_net_pnl:+.2f} bps")
    print(f"Avg costs:           {avg_costs:.2f} bps")

    if weights is not None:
        ess = adapter.weighter.get_effective_sample_size(weights)
        print(f"Effective samples:   {ess:.0f} (out of {len(weights):,})")
        print(f"Sample efficiency:   {ess/len(weights)*100:.1f}%")

    print("\n")

    # Show how to use in custom training
    print("Using V2 labels in custom training:")
    print("─" * 80)
    print("""
    # Get labeled trades and weights
    labeled_trades, weights = adapter.process(data, symbol='BTC/USDT')

    # Convert to DataFrame for compatibility
    df_labels = adapter.convert_to_legacy_format(labeled_trades, weights)

    # Use in your custom training loop
    for i, trade in enumerate(labeled_trades):
        weight = weights[i] if weights else 1.0

        # Add to RL agent's replay buffer
        agent.add_experience(
            state=trade_to_state(trade),
            action=trade.exit_reason,
            reward=trade.pnl_net_bps,
            next_state=next_state,
            done=True,
            weight=weight  # ← Recency weight
        )

    # Train with weighted samples
    agent.update()
    """)

    print("\n")


def example_3_gradual_migration():
    """
    Example 3: Gradual Migration

    Mix V2 components with existing code.
    Enable features incrementally.
    """
    print("\n" + "="*80)
    print("EXAMPLE 3: GRADUAL MIGRATION")
    print("="*80 + "\n")

    print("Enable V2 features one at a time:\n")

    print("STEP 1: Data Quality Only")
    print("─" * 80)
    print("""
    config = V2PipelineConfig(
        enable_data_quality=True,      # ← Enable cleaning
        enable_meta_labeling=False,     # Keep legacy labeling
        enable_recency_weighting=False  # Keep equal weights
    )

    adapter = V2PipelineAdapter(config=config)
    labeled_trades, weights = adapter.process(data, symbol='BTC/USDT')
    """)
    print("\n")

    print("STEP 2: Add Triple-Barrier Labeling")
    print("─" * 80)
    print("""
    config = V2PipelineConfig(
        enable_data_quality=True,
        enable_meta_labeling=False,     # Not yet
        enable_recency_weighting=False,
        # Triple-barrier now used automatically
        scalp_tp_bps=15.0,
        scalp_sl_bps=10.0,
        scalp_timeout_minutes=30
    )

    adapter = V2PipelineAdapter(config=config)
    labeled_trades, weights = adapter.process(data, symbol='BTC/USDT')
    """)
    print("\n")

    print("STEP 3: Add Meta-Labeling")
    print("─" * 80)
    print("""
    config = V2PipelineConfig(
        enable_data_quality=True,
        enable_meta_labeling=True,      # ← Filter to profitable
        enable_recency_weighting=False,
        meta_cost_threshold_bps=5.0     # Must beat costs by 5 bps
    )

    adapter = V2PipelineAdapter(config=config)
    labeled_trades, weights = adapter.process(data, symbol='BTC/USDT')

    # Now only profitable-after-costs trades
    """)
    print("\n")

    print("STEP 4: Add Recency Weighting (Full V2)")
    print("─" * 80)
    print("""
    config = V2PipelineConfig(
        enable_data_quality=True,
        enable_meta_labeling=True,
        enable_recency_weighting=True,  # ← Full V2 pipeline
        recency_halflife_days=10.0      # 10-day halflife
    )

    adapter = V2PipelineAdapter(config=config)
    labeled_trades, weights = adapter.process(data, symbol='BTC/USDT')

    # Now with recency weighting - recent data weighted higher
    """)
    print("\n")


def example_4_mode_comparison():
    """
    Example 4: Scalp vs Runner Mode

    Show differences between trading modes.
    """
    print("\n" + "="*80)
    print("EXAMPLE 4: SCALP VS RUNNER MODE")
    print("="*80 + "\n")

    # Create sample data
    df = create_sample_data(days=90)

    print("SCALP MODE")
    print("─" * 80)
    scalp_adapter = create_v2_scalp_adapter()
    scalp_trades, scalp_weights = scalp_adapter.process(df, 'BTC/USDT')

    print(f"Config:")
    print(f"  TP:       {scalp_adapter.config.scalp_tp_bps} bps (target £1.50 on £1000)")
    print(f"  SL:       {scalp_adapter.config.scalp_sl_bps} bps")
    print(f"  Timeout:  {scalp_adapter.config.scalp_timeout_minutes} minutes")
    print(f"  Halflife: 10 days (aggressive recency)")
    print(f"\nResults:")
    print(f"  Trades:      {len(scalp_trades):,}")
    print(f"  Profitable:  {sum(1 for t in scalp_trades if t.meta_label == 1):,}")
    print(f"  Avg P&L:     {sum(t.pnl_net_bps for t in scalp_trades)/len(scalp_trades):+.2f} bps")
    print(f"  Avg Hold:    {sum(t.duration_minutes for t in scalp_trades)/len(scalp_trades):.0f} minutes")
    print("\n")

    print("RUNNER MODE")
    print("─" * 80)
    runner_adapter = create_v2_runner_adapter()
    runner_trades, runner_weights = runner_adapter.process(df, 'BTC/USDT')

    print(f"Config:")
    print(f"  TP:       {runner_adapter.config.runner_tp_bps} bps (larger targets)")
    print(f"  SL:       {runner_adapter.config.runner_sl_bps} bps")
    print(f"  Timeout:  {runner_adapter.config.runner_timeout_minutes} minutes (7 days)")
    print(f"  Halflife: 20 days (slower recency)")
    print(f"\nResults:")
    print(f"  Trades:      {len(runner_trades):,}")
    print(f"  Profitable:  {sum(1 for t in runner_trades if t.meta_label == 1):,}")
    print(f"  Avg P&L:     {sum(t.pnl_net_bps for t in runner_trades)/len(runner_trades):+.2f} bps")
    print(f"  Avg Hold:    {sum(t.duration_minutes for t in runner_trades)/len(runner_trades):.0f} minutes")
    print("\n")

    print("COMPARISON")
    print("─" * 80)
    print("Scalp mode:")
    print("  • Tighter TP/SL (quick in/out)")
    print("  • Shorter timeout (30 min max)")
    print("  • Aggressive recency (10-day halflife)")
    print("  • More trades, smaller targets")
    print("\nRunner mode:")
    print("  • Wider TP/SL (let winners run)")
    print("  • Longer timeout (7 days max)")
    print("  • Slower recency (20-day halflife)")
    print("  • Fewer trades, larger targets")
    print("\n")


def run_all_examples():
    """Run all integration examples."""
    print("\n" + "="*80)
    print("V2 PIPELINE INTEGRATION EXAMPLES")
    print("="*80)
    print("\nDemonstrating how to integrate V2 data pipeline with existing training code.\n")

    example_1_drop_in_replacement()
    example_2_adapter_integration()
    example_3_gradual_migration()
    example_4_mode_comparison()

    print("="*80)
    print("INTEGRATION EXAMPLES COMPLETE")
    print("="*80)
    print("""
Next Steps:
───────────────────────────────────────────────────────────────────────────────

1. Choose your integration approach:
   • Drop-in replacement (easiest)
   • Adapter integration (most flexible)
   • Gradual migration (safest)

2. Update your training code:
   • Replace RLTrainingPipeline with RLTrainingPipelineV2
   • OR use V2PipelineAdapter directly
   • OR enable features one at a time

3. Test on historical data:
   • Run on BTC/USDT
   • Compare V1 vs V2 metrics
   • Validate improvements

4. Deploy to production:
   • Update Engine training
   • Monitor V2 metrics
   • Compare live performance

Key Files:
───────────────────────────────────────────────────────────────────────────────
• RLTrainingPipelineV2:     cloud/training/pipelines/rl_training_pipeline_v2.py
• V2PipelineAdapter:        cloud/training/integrations/v2_pipeline_adapter.py
• V2 Pipeline Components:   cloud/engine/
• Documentation:            cloud/engine/README_V2_PIPELINE.md

Questions? Check the README or review the example scripts!
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

    # Run all examples
    run_all_examples()
