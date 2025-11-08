"""
Visualization Utilities

Implements visualization functions for model evaluation, metrics, and diagnostics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import structlog

logger = structlog.get_logger(__name__)

# Try to import plotting libraries
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    logger.warning("plotting_libraries_not_available")


class ModelVisualizer:
    """Visualization utilities for model evaluation and diagnostics."""
    
    def __init__(self, output_dir: Path = Path("plots")):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory to save plots
        """
        if not HAS_PLOTTING:
            raise ImportError("Matplotlib and Seaborn are required for visualization")
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams["figure.figsize"] = (12, 6)
        
        logger.info("model_visualizer_initialized", output_dir=str(self.output_dir))
    
    def plot_predictions_vs_actual(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = "Predictions vs Actual",
        save_path: Optional[Path] = None,
    ) -> None:
        """Plot predictions vs actual values."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.scatter(y_true, y_pred, alpha=0.5)
        ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--", lw=2)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predictions")
        ax.set_title(title)
        ax.grid(True)
        
        # Calculate R²
        from sklearn.metrics import r2_score
        r2 = r2_score(y_true, y_pred)
        ax.text(0.05, 0.95, f"R² = {r2:.4f}", transform=ax.transAxes, fontsize=12, verticalalignment="top")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        else:
            plt.savefig(self.output_dir / "predictions_vs_actual.png", dpi=300, bbox_inches="tight")
        
        plt.close()
        logger.info("plot_saved", plot="predictions_vs_actual")
    
    def plot_residuals(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = "Residuals Plot",
        save_path: Optional[Path] = None,
    ) -> None:
        """Plot residuals."""
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Residuals vs predictions
        axes[0].scatter(y_pred, residuals, alpha=0.5)
        axes[0].axhline(y=0, color="r", linestyle="--")
        axes[0].set_xlabel("Predictions")
        axes[0].set_ylabel("Residuals")
        axes[0].set_title("Residuals vs Predictions")
        axes[0].grid(True)
        
        # Residuals distribution
        axes[1].hist(residuals, bins=30, edgecolor="black")
        axes[1].set_xlabel("Residuals")
        axes[1].set_ylabel("Frequency")
        axes[1].set_title("Residuals Distribution")
        axes[1].grid(True)
        
        plt.suptitle(title, fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        else:
            plt.savefig(self.output_dir / "residuals.png", dpi=300, bbox_inches="tight")
        
        plt.close()
        logger.info("plot_saved", plot="residuals")
    
    def plot_feature_importance(
        self,
        feature_importance: Dict[str, float],
        top_n: int = 20,
        title: str = "Feature Importance",
        save_path: Optional[Path] = None,
    ) -> None:
        """Plot feature importance."""
        # Sort by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        top_features = sorted_features[:top_n]
        
        features, importances = zip(*top_features)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(range(len(features)), importances)
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features)
        ax.set_xlabel("Importance")
        ax.set_title(title)
        ax.grid(True, axis="x")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        else:
            plt.savefig(self.output_dir / "feature_importance.png", dpi=300, bbox_inches="tight")
        
        plt.close()
        logger.info("plot_saved", plot="feature_importance")
    
    def plot_training_curve(
        self,
        train_losses: List[float],
        val_losses: Optional[List[float]] = None,
        title: str = "Training Curve",
        save_path: Optional[Path] = None,
    ) -> None:
        """Plot training curve."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        epochs = range(1, len(train_losses) + 1)
        ax.plot(epochs, train_losses, label="Training Loss", marker="o")
        
        if val_losses:
            ax.plot(epochs, val_losses, label="Validation Loss", marker="s")
        
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(title)
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        else:
            plt.savefig(self.output_dir / "training_curve.png", dpi=300, bbox_inches="tight")
        
        plt.close()
        logger.info("plot_saved", plot="training_curve")
    
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = "Confusion Matrix",
        save_path: Optional[Path] = None,
    ) -> None:
        """Plot confusion matrix."""
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(title)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        else:
            plt.savefig(self.output_dir / "confusion_matrix.png", dpi=300, bbox_inches="tight")
        
        plt.close()
        logger.info("plot_saved", plot="confusion_matrix")
    
    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        title: str = "ROC Curve",
        save_path: Optional[Path] = None,
    ) -> None:
        """Plot ROC curve."""
        from sklearn.metrics import roc_curve, auc
        
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
        ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(title)
        ax.legend(loc="lower right")
        ax.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        else:
            plt.savefig(self.output_dir / "roc_curve.png", dpi=300, bbox_inches="tight")
        
        plt.close()
        logger.info("plot_saved", plot="roc_curve")
    
    def plot_model_comparison(
        self,
        model_metrics: Dict[str, Dict[str, float]],
        metric: str = "sharpe_ratio",
        title: str = "Model Comparison",
        save_path: Optional[Path] = None,
    ) -> None:
        """Plot model comparison by metric."""
        models = list(model_metrics.keys())
        values = [model_metrics[model].get(metric, 0.0) for model in models]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(models, values)
        ax.set_xlabel(metric.replace("_", " ").title())
        ax.set_title(title)
        ax.grid(True, axis="x")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        else:
            plt.savefig(self.output_dir / "model_comparison.png", dpi=300, bbox_inches="tight")
        
        plt.close()
        logger.info("plot_saved", plot="model_comparison")
    
    def plot_bias_variance(
        self,
        train_errors: List[float],
        val_errors: List[float],
        title: str = "Bias-Variance Tradeoff",
        save_path: Optional[Path] = None,
    ) -> None:
        """Plot bias-variance tradeoff."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        complexity = range(1, len(train_errors) + 1)
        ax.plot(complexity, train_errors, label="Training Error", marker="o")
        ax.plot(complexity, val_errors, label="Validation Error", marker="s")
        ax.set_xlabel("Model Complexity")
        ax.set_ylabel("Error")
        ax.set_title(title)
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        else:
            plt.savefig(self.output_dir / "bias_variance.png", dpi=300, bbox_inches="tight")
        
        plt.close()
        logger.info("plot_saved", plot="bias_variance")

