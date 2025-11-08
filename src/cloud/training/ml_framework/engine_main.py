"""
ML Engine Main Entry Point

Orchestrates the complete ML pipeline:
1. Data ingestion
2. Preprocessing
3. Model training (baseline, core, neural)
4. Ensemble blending
5. Prediction
6. Feedback and auto-tuning
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import structlog

from .orchestrator import MLEngineOrchestrator

logger = structlog.get_logger(__name__)


def main():
    """Main entry point for ML Engine."""
    parser = argparse.ArgumentParser(description="Huracan ML Engine Orchestrator")
    parser.add_argument(
        "--config",
        type=str,
        default="config/ml_framework.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "predict", "evaluate", "auto_tune"],
        default="train",
        help="Operation mode",
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to training/prediction data (CSV or Parquet)",
    )
    parser.add_argument(
        "--target",
        type=str,
        help="Target column name (for training/evaluation)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save predictions (for predict mode)",
    )
    parser.add_argument(
        "--validation-data",
        type=str,
        help="Path to validation data (optional)",
    )
    
    args = parser.parse_args()
    
    # Configure logging
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level=args.log_level if hasattr(args, 'log_level') else "INFO"),
    )
    
    logger.info("ml_engine_starting", mode=args.mode, config=args.config)
    
    # Initialize orchestrator
    try:
        orchestrator = MLEngineOrchestrator(args.config)
    except Exception as e:
        logger.error("orchestrator_initialization_failed", error=str(e))
        sys.exit(1)
    
    # Load data
    try:
        data_path = Path(args.data)
        if data_path.suffix == ".csv":
            data = pd.read_csv(data_path)
        elif data_path.suffix == ".parquet":
            data = pd.read_parquet(data_path)
        else:
            raise ValueError(f"Unsupported file format: {data_path.suffix}")
        
        logger.info("data_loaded", rows=len(data), columns=list(data.columns))
    except Exception as e:
        logger.error("data_loading_failed", error=str(e))
        sys.exit(1)
    
    # Execute based on mode
    if args.mode == "train":
        if not args.target:
            logger.error("target_column_required_for_training")
            sys.exit(1)
        
        # Split data
        X = data.drop(columns=[args.target])
        y = data[args.target]
        
        # Load validation data if provided
        X_val = None
        y_val = None
        if args.validation_data:
            val_data = pd.read_csv(args.validation_data) if Path(args.validation_data).suffix == ".csv" else pd.read_parquet(args.validation_data)
            X_val = val_data.drop(columns=[args.target])
            y_val = val_data[args.target]
        
        # Train models
        try:
            results = orchestrator.train_all_models(X, y, X_val, y_val)
            logger.info("training_complete", num_models=len(results))
            
            # Print results
            for name, metrics in results.items():
                print(f"\n{name}:")
                print(f"  Sharpe: {metrics.sharpe_ratio:.4f}")
                print(f"  RMSE: {metrics.rmse:.4f}")
                print(f"  Win Rate: {metrics.win_rate:.4f}")
        
        except Exception as e:
            logger.error("training_failed", error=str(e))
            sys.exit(1)
    
    elif args.mode == "predict":
        # Make predictions
        try:
            predictions = orchestrator.predict(data, use_ensemble=True)
            
            # Save predictions
            if args.output:
                output_path = Path(args.output)
                output_df = pd.DataFrame({"predictions": predictions})
                if output_path.suffix == ".csv":
                    output_df.to_csv(output_path, index=False)
                else:
                    output_df.to_parquet(output_path, index=False)
                logger.info("predictions_saved", path=str(output_path))
            else:
                # Print predictions
                print(predictions)
        
        except Exception as e:
            logger.error("prediction_failed", error=str(e))
            sys.exit(1)
    
    elif args.mode == "evaluate":
        if not args.target:
            logger.error("target_column_required_for_evaluation")
            sys.exit(1)
        
        # Evaluate models
        try:
            X = data.drop(columns=[args.target])
            y = data[args.target]
            
            results = orchestrator.evaluate(X, y)
            logger.info("evaluation_complete", num_models=len(results))
            
            # Print results
            for name, metrics in results.items():
                print(f"\n{name}:")
                print(f"  Sharpe: {metrics.sharpe_ratio:.4f}")
                print(f"  RMSE: {metrics.rmse:.4f}")
                print(f"  Win Rate: {metrics.win_rate:.4f}")
        
        except Exception as e:
            logger.error("evaluation_failed", error=str(e))
            sys.exit(1)
    
    elif args.mode == "auto_tune":
        # Auto-tune models
        try:
            orchestrator.auto_tune()
            logger.info("auto_tune_complete")
            
            # Print performance report
            report = orchestrator.get_performance_report()
            print("\nPerformance Report:")
            print(f"  Retrain Queue: {report['retrain_queue']}")
            print(f"  Prune Candidates: {report['prune_candidates']}")
            print(f"  Ensemble Weights: {report['ensemble_weights']}")
        
        except Exception as e:
            logger.error("auto_tune_failed", error=str(e))
            sys.exit(1)
    
    logger.info("ml_engine_complete", mode=args.mode)


if __name__ == "__main__":
    main()

