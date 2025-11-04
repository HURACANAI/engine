"""
Model Persistence Module

Handles saving and loading of all model states for continuity across sessions.
Includes regime detector, confidence scorer, and feature importance learner.

Based on Revuelto bot's persistence system.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import structlog

from .confidence_scorer import ConfidenceScorer
from .feature_importance_learner import FeatureImportanceLearner
from .regime_detector import RegimeDetector

logger = structlog.get_logger()


class ModelPersistence:
    """
    Manages persistence of all learning modules.

    Key capabilities:
    1. Save complete system state to JSON
    2. Load and restore state on startup
    3. Version tracking for compatibility
    4. Automatic backup before overwriting
    """

    VERSION = "1.0.0"

    def __init__(self, persistence_dir: Optional[Path] = None):
        """
        Initialize model persistence.

        Args:
            persistence_dir: Directory for saving state files (default: ./model_state)
        """
        self.persistence_dir = persistence_dir or Path("./model_state")
        self.persistence_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "model_persistence_initialized",
            persistence_dir=str(self.persistence_dir),
        )

    def save_state(
        self,
        regime_detector: RegimeDetector,
        confidence_scorer: ConfidenceScorer,
        feature_importance_learner: FeatureImportanceLearner,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Save complete state of all learning modules.

        Args:
            regime_detector: RegimeDetector instance
            confidence_scorer: ConfidenceScorer instance
            feature_importance_learner: FeatureImportanceLearner instance
            metadata: Optional metadata to store (e.g., symbol, timestamp)

        Returns:
            Path to saved state file
        """
        # Create state dictionary
        state = {
            "version": self.VERSION,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {},
            "regime_detector": {
                "panic_threshold": regime_detector.panic_threshold,
                "trend_threshold": regime_detector.trend_threshold,
                "range_threshold": regime_detector.range_threshold,
                "lookback_period": regime_detector.lookback_period,
            },
            "confidence_scorer": {
                "min_confidence_threshold": confidence_scorer.min_confidence_threshold,
                "sample_threshold": confidence_scorer.sample_threshold,
                "strong_alignment_threshold": confidence_scorer.strong_alignment_threshold,
                "calibration_history": confidence_scorer.calibration_history,
            },
            "feature_importance_learner": feature_importance_learner.get_state(),
        }

        # Backup existing state if it exists
        state_file = self.persistence_dir / "model_state.json"
        if state_file.exists():
            backup_file = self.persistence_dir / f"model_state_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            state_file.rename(backup_file)
            logger.info("state_backed_up", backup_file=str(backup_file))

        # Save state
        with open(state_file, "w") as f:
            json.dump(state, f, indent=2)

        logger.info(
            "model_state_saved",
            file=str(state_file),
            feature_importance_samples=state["feature_importance_learner"]["total_samples"],
        )

        return state_file

    def load_state(
        self,
        regime_detector: Optional[RegimeDetector] = None,
        confidence_scorer: Optional[ConfidenceScorer] = None,
        feature_importance_learner: Optional[FeatureImportanceLearner] = None,
    ) -> tuple[RegimeDetector, ConfidenceScorer, FeatureImportanceLearner, Dict[str, Any]]:
        """
        Load complete state of all learning modules.

        Args:
            regime_detector: Existing instance to load into (or None to create new)
            confidence_scorer: Existing instance to load into (or None to create new)
            feature_importance_learner: Existing instance to load into (or None to create new)

        Returns:
            Tuple of (regime_detector, confidence_scorer, feature_importance_learner, metadata)
        """
        state_file = self.persistence_dir / "model_state.json"

        if not state_file.exists():
            logger.warning("no_saved_state_found", expected_file=str(state_file))

            # Return new instances
            return (
                regime_detector or RegimeDetector(),
                confidence_scorer or ConfidenceScorer(),
                feature_importance_learner or FeatureImportanceLearner(),
                {},
            )

        # Load state
        with open(state_file, "r") as f:
            state = json.load(f)

        # Version check
        if state.get("version") != self.VERSION:
            logger.warning(
                "state_version_mismatch",
                expected=self.VERSION,
                found=state.get("version"),
            )

        # Create or update instances
        rd_state = state["regime_detector"]
        regime_detector = regime_detector or RegimeDetector(
            panic_threshold=rd_state["panic_threshold"],
            trend_threshold=rd_state["trend_threshold"],
            range_threshold=rd_state["range_threshold"],
            lookback_period=rd_state["lookback_period"],
        )

        cs_state = state["confidence_scorer"]
        if confidence_scorer is None:
            confidence_scorer = ConfidenceScorer(
                min_confidence_threshold=cs_state["min_confidence_threshold"],
                sample_threshold=cs_state["sample_threshold"],
                strong_alignment_threshold=cs_state["strong_alignment_threshold"],
            )
            confidence_scorer.calibration_history = cs_state["calibration_history"]
        else:
            # Update existing instance
            confidence_scorer.calibration_history = cs_state["calibration_history"]

        fi_state = state["feature_importance_learner"]
        if feature_importance_learner is None:
            feature_importance_learner = FeatureImportanceLearner()

        feature_importance_learner.load_state(fi_state)

        metadata = state.get("metadata", {})

        logger.info(
            "model_state_loaded",
            file=str(state_file),
            timestamp=state.get("timestamp"),
            feature_importance_samples=fi_state["total_samples"],
        )

        return regime_detector, confidence_scorer, feature_importance_learner, metadata

    def get_backup_files(self) -> list[Path]:
        """Get list of backup state files."""
        return sorted(self.persistence_dir.glob("model_state_backup_*.json"), reverse=True)

    def restore_from_backup(self, backup_file: Path) -> bool:
        """
        Restore state from a backup file.

        Args:
            backup_file: Path to backup file

        Returns:
            True if restoration successful
        """
        if not backup_file.exists():
            logger.error("backup_file_not_found", file=str(backup_file))
            return False

        try:
            state_file = self.persistence_dir / "model_state.json"

            # Backup current state if it exists
            if state_file.exists():
                temp_backup = self.persistence_dir / f"model_state_temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                state_file.rename(temp_backup)

            # Copy backup to active state
            import shutil
            shutil.copy(backup_file, state_file)

            logger.info("backup_restored", from_file=str(backup_file))
            return True

        except Exception as e:
            logger.error("backup_restore_failed", error=str(e))
            return False

    def clear_old_backups(self, keep_n: int = 10) -> None:
        """
        Remove old backup files, keeping only the N most recent.

        Args:
            keep_n: Number of backups to keep
        """
        backups = self.get_backup_files()

        if len(backups) <= keep_n:
            return

        # Remove old backups
        for backup in backups[keep_n:]:
            backup.unlink()
            logger.info("old_backup_removed", file=str(backup))

        logger.info("old_backups_cleared", removed=len(backups) - keep_n, kept=keep_n)

    def export_state_summary(self) -> Dict[str, Any]:
        """
        Export a human-readable summary of the current state.

        Returns:
            Dictionary with summary information
        """
        state_file = self.persistence_dir / "model_state.json"

        if not state_file.exists():
            return {"status": "no_state_found"}

        with open(state_file, "r") as f:
            state = json.load(f)

        fi_state = state["feature_importance_learner"]

        summary = {
            "version": state.get("version"),
            "last_saved": state.get("timestamp"),
            "metadata": state.get("metadata"),
            "feature_importance": {
                "total_samples": fi_state["total_samples"],
                "total_wins": fi_state["total_wins"],
                "total_losses": fi_state["total_losses"],
                "win_rate": fi_state["total_wins"] / fi_state["total_samples"] if fi_state["total_samples"] > 0 else 0.0,
                "num_features_tracked": len(fi_state["feature_importance"]),
            },
            "confidence_scorer": {
                "min_threshold": state["confidence_scorer"]["min_confidence_threshold"],
                "calibration_entries": len(state["confidence_scorer"]["calibration_history"]),
            },
        }

        return summary
