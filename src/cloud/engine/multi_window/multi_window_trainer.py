"""
Multi-Window Trainer

Orchestrates training of multiple components with component-specific windows.

Key features:
1. Train each component on its optimal historical window
2. Apply component-specific recency weighting
3. Package all components into single deployable artifact
4. Track training metrics per component

This is the HIGH-LEVEL orchestrator that ties everything together.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import polars as pl
import structlog

from .component_configs import ComponentConfig, create_all_component_configs
from .window_manager import TrainingWindowManager, ComponentDataWindow, print_window_summary

logger = structlog.get_logger(__name__)


@dataclass
class ComponentTrainingResult:
    """Results from training a single component."""

    component_name: str
    model: Any  # Trained model object
    training_samples: int
    effective_samples: float
    training_time_seconds: float
    validation_metrics: Dict[str, float]
    config: ComponentConfig


@dataclass
class MultiWindowTrainingResult:
    """Results from training all components."""

    components: Dict[str, ComponentTrainingResult]
    total_training_time_seconds: float
    training_timestamp: datetime
    artifact_path: Optional[Path] = None


class MultiWindowTrainer:
    """
    Train multiple components with component-specific windows.

    Usage:
        trainer = MultiWindowTrainer()

        results = trainer.train_all_components(
            data=labeled_trades_df,
            train_fn=my_training_function,
            configs=create_all_component_configs()
        )

        # Access individual models
        scalp_model = results.components['scalp_core'].model
        regime_model = results.components['regime_classifier'].model

        # Save packaged artifact
        trainer.save_artifact(results, '/path/to/models')
    """

    def __init__(
        self,
        window_manager: Optional[TrainingWindowManager] = None,
        save_artifacts: bool = True
    ):
        """
        Initialize multi-window trainer.

        Args:
            window_manager: Custom window manager (or creates default)
            save_artifacts: Whether to save trained models
        """
        self.window_manager = window_manager or TrainingWindowManager()
        self.save_artifacts = save_artifacts

        logger.info(
            "multi_window_trainer_initialized",
            save_artifacts=save_artifacts
        )

    def train_all_components(
        self,
        data: pl.DataFrame,
        train_fn: Callable[[ComponentDataWindow], Any],
        configs: Optional[Dict[str, ComponentConfig]] = None,
        validate_fn: Optional[Callable[[Any, ComponentDataWindow], Dict[str, float]]] = None
    ) -> MultiWindowTrainingResult:
        """
        Train all components with their specific windows.

        Args:
            data: Full labeled dataset (with 'timestamp' column)
            train_fn: Function that takes ComponentDataWindow and returns trained model
                      Signature: train_fn(window: ComponentDataWindow) -> model
            configs: Component configs (defaults to all standard configs)
            validate_fn: Optional validation function
                        Signature: validate_fn(model, window) -> metrics_dict

        Returns:
            MultiWindowTrainingResult with all trained components

        Example:
            def my_train_fn(window: ComponentDataWindow):
                X = window.data.drop('timestamp')
                y = window.data['label']
                model = XGBClassifier()
                model.fit(X, y, sample_weight=window.weights)
                return model

            results = trainer.train_all_components(
                data=labeled_df,
                train_fn=my_train_fn
            )
        """
        start_time = datetime.now()

        # Use default configs if none provided
        if configs is None:
            configs = create_all_component_configs()

        logger.info(
            "starting_multi_window_training",
            components=list(configs.keys()),
            total_rows=len(data)
        )

        # Step 1: Prepare all component windows
        windows = self.window_manager.prepare_all_components(
            data=data,
            configs=configs,
            create_walk_forward_splits=False  # WF handled separately if needed
        )

        print_window_summary(windows)

        # Step 2: Train each component
        component_results = {}

        for name, window in windows.items():
            logger.info(
                "training_component",
                component=name,
                samples=window.total_samples,
                effective_samples=window.effective_sample_size
            )

            component_start = datetime.now()

            try:
                # Train model
                model = train_fn(window)

                # Validate if function provided
                validation_metrics = {}
                if validate_fn:
                    validation_metrics = validate_fn(model, window)

                component_time = (datetime.now() - component_start).total_seconds()

                result = ComponentTrainingResult(
                    component_name=name,
                    model=model,
                    training_samples=window.total_samples,
                    effective_samples=window.effective_sample_size,
                    training_time_seconds=component_time,
                    validation_metrics=validation_metrics,
                    config=window.config
                )

                component_results[name] = result

                logger.info(
                    "component_training_complete",
                    component=name,
                    training_seconds=component_time,
                    validation_metrics=validation_metrics
                )

            except Exception as e:
                logger.error(
                    "component_training_failed",
                    component=name,
                    error=str(e),
                    exc_info=True
                )
                # Continue with other components
                continue

        total_time = (datetime.now() - start_time).total_seconds()

        result = MultiWindowTrainingResult(
            components=component_results,
            total_training_time_seconds=total_time,
            training_timestamp=start_time
        )

        logger.info(
            "multi_window_training_complete",
            components_trained=len(component_results),
            total_seconds=total_time
        )

        return result

    def train_single_component(
        self,
        data: pl.DataFrame,
        component_name: str,
        train_fn: Callable[[ComponentDataWindow], Any],
        config: Optional[ComponentConfig] = None,
        validate_fn: Optional[Callable[[Any, ComponentDataWindow], Dict[str, float]]] = None
    ) -> ComponentTrainingResult:
        """
        Train a single component.

        Args:
            data: Full labeled dataset
            component_name: Name of component to train
            train_fn: Training function
            config: Component config (or uses default)
            validate_fn: Optional validation function

        Returns:
            ComponentTrainingResult
        """
        # Get config
        if config is None:
            all_configs = create_all_component_configs()
            if component_name not in all_configs:
                raise ValueError(
                    f"Unknown component '{component_name}'. "
                    f"Valid: {list(all_configs.keys())}"
                )
            config = all_configs[component_name]

        logger.info(
            "training_single_component",
            component=component_name,
            lookback_days=config.lookback_days
        )

        # Prepare window
        window = self.window_manager.prepare_component_window(
            data=data,
            config=config
        )

        # Train
        start_time = datetime.now()
        model = train_fn(window)
        training_time = (datetime.now() - start_time).total_seconds()

        # Validate
        validation_metrics = {}
        if validate_fn:
            validation_metrics = validate_fn(model, window)

        result = ComponentTrainingResult(
            component_name=component_name,
            model=model,
            training_samples=window.total_samples,
            effective_samples=window.effective_sample_size,
            training_time_seconds=training_time,
            validation_metrics=validation_metrics,
            config=config
        )

        logger.info(
            "single_component_training_complete",
            component=component_name,
            training_seconds=training_time
        )

        return result

    def save_artifact(
        self,
        results: MultiWindowTrainingResult,
        output_dir: Path,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Save trained models as deployable artifact.

        Args:
            results: Training results with all models
            output_dir: Directory to save artifact
            metadata: Optional additional metadata

        Returns:
            Path to saved artifact

        The artifact includes:
        - All trained models
        - Component configs
        - Training metadata
        - Validation metrics
        """
        import pickle
        import json

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = results.training_timestamp.strftime("%Y%m%d_%H%M%S")
        artifact_dir = output_dir / f"huracan_v2_{timestamp}"
        artifact_dir.mkdir(exist_ok=True)

        logger.info(
            "saving_training_artifact",
            output_dir=str(artifact_dir),
            components=list(results.components.keys())
        )

        # Save each component model
        for name, component_result in results.components.items():
            model_path = artifact_dir / f"{name}_model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(component_result.model, f)

            logger.debug("component_model_saved", component=name, path=str(model_path))

        # Save metadata
        metadata_dict = {
            'training_timestamp': results.training_timestamp.isoformat(),
            'total_training_time_seconds': results.total_training_time_seconds,
            'components': {
                name: {
                    'training_samples': comp.training_samples,
                    'effective_samples': comp.effective_samples,
                    'training_time_seconds': comp.training_time_seconds,
                    'validation_metrics': comp.validation_metrics,
                    'lookback_days': comp.config.lookback_days,
                    'timeframe': comp.config.timeframe,
                    'recency_halflife_days': comp.config.recency_halflife_days
                }
                for name, comp in results.components.items()
            }
        }

        if metadata:
            metadata_dict['additional_metadata'] = metadata

        metadata_path = artifact_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata_dict, f, indent=2)

        logger.info(
            "training_artifact_saved",
            artifact_dir=str(artifact_dir),
            components=len(results.components)
        )

        results.artifact_path = artifact_dir
        return artifact_dir


def print_training_results(results: MultiWindowTrainingResult) -> None:
    """
    Pretty-print training results.

    Usage:
        results = trainer.train_all_components(...)
        print_training_results(results)
    """
    print("\n" + "="*80)
    print("MULTI-WINDOW TRAINING RESULTS")
    print("="*80 + "\n")

    print(f"Training Timestamp: {results.training_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total Training Time: {results.total_training_time_seconds:.1f} seconds")
    print(f"Components Trained: {len(results.components)}")

    if results.artifact_path:
        print(f"Artifact Path: {results.artifact_path}")

    print("\n" + "-"*80)
    print(f"{'Component':<20} {'Samples':<12} {'Effective':<12} {'Time (s)':<12} {'Metrics':<20}")
    print("-"*80)

    for name, comp in results.components.items():
        metrics_str = ", ".join(f"{k}={v:.3f}" for k, v in list(comp.validation_metrics.items())[:2])
        if not metrics_str:
            metrics_str = "N/A"

        print(
            f"{comp.component_name:<20} "
            f"{comp.training_samples:<12,} "
            f"{int(comp.effective_samples):<12,} "
            f"{comp.training_time_seconds:<12.1f} "
            f"{metrics_str:<20}"
        )

    print("\n" + "="*80 + "\n")

    print("✅ Multi-window training complete!")
    print("   Each component trained on its optimal historical window.")
    print("   Models ready for deployment to Hamilton Pilot.\n")
