"""
Meta-Label Trainer

Trains ML models to predict P(win | signal, features, regime, technique).

This improves the meta-label gate by using proper machine learning instead
of simple heuristics.

Models Supported:
1. Logistic Regression (fast, interpretable)
2. Random Forest (non-linear, feature importance)
3. XGBoost (best performance)
4. Neural Network (deep learning, optional)

Features Used:
- engine_confidence
- regime (one-hot encoded)
- technique (one-hot encoded)
- Market features (trend_strength, adx, etc.)
- Cross-features (confidence × regime, etc.)

Usage:
    trainer = MetaLabelTrainer(
        model_type='xgboost',
    )

    # Train on historical trades
    trainer.fit(trades_list)

    # Predict win probability
    win_prob = trainer.predict(features)

    # Save model
    trainer.save('meta_label_model.pkl')
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import pickle
import structlog

# ML imports
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score, classification_report
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

logger = structlog.get_logger(__name__)


@dataclass
class TrainingResult:
    """Result of model training."""

    model_type: str
    train_accuracy: float
    cv_accuracy: float
    train_auc: float
    cv_auc: float
    n_samples: int
    n_features: int
    feature_importance: Optional[Dict[str, float]] = None


class MetaLabelTrainer:
    """
    Train ML models for meta-label prediction.

    Predicts: P(win | signal_features)

    Architecture:
        Historical Trades → Feature Engineering → ML Model → P(win)
                                    ↓
        Features: confidence, regime, technique, market_features, cross_features
    """

    def __init__(
        self,
        model_type: str = 'xgboost',
        target_threshold: float = 0.50,  # Decision threshold
    ):
        """
        Initialize meta-label trainer.

        Args:
            model_type: 'logistic', 'random_forest', 'xgboost'
            target_threshold: Decision threshold for classification
        """
        if not HAS_SKLEARN:
            raise ImportError("sklearn required: pip install scikit-learn")

        if model_type == 'xgboost' and not HAS_XGBOOST:
            logger.warning("xgboost_not_available_falling_back", fallback='random_forest')
            model_type = 'random_forest'

        self.model_type = model_type
        self.target_threshold = target_threshold

        # Initialize model
        if model_type == 'logistic':
            self.model = LogisticRegression(
                max_iter=1000,
                random_state=42,
                class_weight='balanced',
            )
        elif model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_leaf=20,
                random_state=42,
                class_weight='balanced',
            )
        elif model_type == 'xgboost':
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                min_child_weight=5,
                random_state=42,
                eval_metric='logloss',
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Feature scaler
        self.scaler = StandardScaler()

        # Feature names (for interpretability)
        self.feature_names: List[str] = []

        # Training stats
        self.is_trained = False
        self.training_result: Optional[TrainingResult] = None

        logger.info(
            "meta_label_trainer_initialized",
            model_type=model_type,
        )

    def fit(
        self,
        trades: List,
        verbose: bool = True,
    ) -> TrainingResult:
        """
        Train model on historical trades.

        Args:
            trades: List of HistoricalTrade or TradeExport objects
            verbose: Print progress

        Returns:
            TrainingResult with metrics
        """
        if verbose:
            print("=" * 70)
            print("META-LABEL MODEL TRAINING")
            print("=" * 70)
            print(f"Model: {self.model_type}")
            print(f"Samples: {len(trades)}")
            print()

        # Feature engineering
        X, y = self._engineer_features(trades)

        if verbose:
            print(f"Features: {X.shape[1]}")
            print(f"Positive samples: {sum(y)} ({sum(y)/len(y):.1%})")
            print()

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Train model
        if verbose:
            print("Training model...")

        # Add early stopping if XGBoost
        if self.model_type == 'xgboost' and HAS_XGBOOST:
            # Use validation set for early stopping if available
            # For now, train without early stopping (can be enhanced)
            self.model.fit(X_scaled, y)
        else:
            self.model.fit(X_scaled, y)

        # Evaluate
        if verbose:
            print("Evaluating...")

        # Training metrics
        train_pred = self.model.predict(X_scaled)
        train_proba = self.model.predict_proba(X_scaled)[:, 1]
        train_accuracy = (train_pred == y).mean()
        train_auc = roc_auc_score(y, train_proba)

        # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(self.model, X_scaled, y, cv=cv, scoring='accuracy')
        cv_auc_scores = cross_val_score(self.model, X_scaled, y, cv=cv, scoring='roc_auc')

        cv_accuracy = cv_scores.mean()
        cv_auc = cv_auc_scores.mean()

        # Feature importance
        feature_importance = None
        if hasattr(self.model, 'feature_importances_'):
            importance_values = self.model.feature_importances_
            feature_importance = dict(zip(self.feature_names, importance_values))

            if verbose:
                print("\nTop 10 Most Important Features:")
                sorted_features = sorted(
                    feature_importance.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                for feat, imp in sorted_features[:10]:
                    print(f"  {feat}: {imp:.4f}")

        if verbose:
            print(f"\nTraining Results:")
            print(f"  Train Accuracy: {train_accuracy:.1%}")
            print(f"  Train AUC: {train_auc:.3f}")
            print(f"  CV Accuracy: {cv_accuracy:.1%} (±{cv_scores.std():.1%})")
            print(f"  CV AUC: {cv_auc:.3f} (±{cv_auc_scores.std():.3f})")

        self.is_trained = True
        self.training_result = TrainingResult(
            model_type=self.model_type,
            train_accuracy=train_accuracy,
            cv_accuracy=cv_accuracy,
            train_auc=train_auc,
            cv_auc=cv_auc,
            n_samples=len(trades),
            n_features=X.shape[1],
            feature_importance=feature_importance,
        )

        return self.training_result

    def predict(
        self,
        features: Dict[str, float],
    ) -> float:
        """
        Predict win probability for a signal.

        Args:
            features: Feature dict with confidence, regime, technique, etc.

        Returns:
            Win probability (0-1)
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call fit() first.")

        # Engineer features
        X = self._engineer_single_sample(features)

        # Scale
        X_scaled = self.scaler.transform([X])

        # Predict
        win_prob = self.model.predict_proba(X_scaled)[0, 1]

        return float(win_prob)

    def _engineer_features(
        self,
        trades: List,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Engineer features from trades.

        Args:
            trades: List of trades

        Returns:
            (X, y) feature matrix and labels
        """
        features_list = []
        labels = []

        for trade in trades:
            # Get features dict
            if hasattr(trade, 'features'):
                raw_features = trade.features
            else:
                raw_features = {}

            # Build feature vector
            features = {
                'confidence': trade.confidence,
                **raw_features,
            }

            # Add regime one-hot
            for regime in ['TREND', 'RANGE', 'PANIC']:
                features[f'regime_{regime}'] = 1.0 if trade.regime.upper() == regime else 0.0

            # Add technique one-hot
            for tech in ['TREND', 'RANGE', 'BREAKOUT', 'TAPE', 'LEADER', 'SWEEP']:
                features[f'technique_{tech}'] = 1.0 if trade.technique.upper() == tech else 0.0

            # Cross-features
            features['conf_x_trend'] = features['confidence'] * features.get('regime_TREND', 0)
            features['conf_x_range'] = features['confidence'] * features.get('regime_RANGE', 0)

            # Edge features
            if hasattr(trade, 'edge_hat_bps'):
                features['edge_hat_bps'] = trade.edge_hat_bps

            # Extract feature vector
            if not self.feature_names:
                # First trade - establish feature names
                self.feature_names = sorted(features.keys())

            feature_vector = [features.get(name, 0.0) for name in self.feature_names]

            features_list.append(feature_vector)
            labels.append(1 if trade.won else 0)

        X = np.array(features_list)
        y = np.array(labels)

        return X, y

    def _engineer_single_sample(
        self,
        features: Dict[str, float],
    ) -> List[float]:
        """Engineer features for a single sample."""
        # Add regime one-hot
        regime = features.get('regime', 'TREND').upper()
        for reg in ['TREND', 'RANGE', 'PANIC']:
            features[f'regime_{reg}'] = 1.0 if regime == reg else 0.0

        # Add technique one-hot
        technique = features.get('technique', 'TREND').upper()
        for tech in ['TREND', 'RANGE', 'BREAKOUT', 'TAPE', 'LEADER', 'SWEEP']:
            features[f'technique_{tech}'] = 1.0 if technique == tech else 0.0

        # Cross-features
        features['conf_x_trend'] = features.get('confidence', 0.5) * features.get('regime_TREND', 0)
        features['conf_x_range'] = features.get('confidence', 0.5) * features.get('regime_RANGE', 0)

        # Extract in same order as training
        feature_vector = [features.get(name, 0.0) for name in self.feature_names]

        return feature_vector

    def save(self, path: str) -> None:
        """Save trained model."""
        if not self.is_trained:
            raise ValueError("Model not trained")

        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'model_type': self.model_type,
                'training_result': self.training_result,
            }, f)

        logger.info("model_saved", path=path)

    def load(self, path: str) -> None:
        """Load trained model."""
        with open(path, 'rb') as f:
            data = pickle.load(f)

        self.model = data['model']
        self.scaler = data['scaler']
        self.feature_names = data['feature_names']
        self.model_type = data['model_type']
        self.training_result = data['training_result']
        self.is_trained = True

        logger.info("model_loaded", path=path, model_type=self.model_type)


def run_training_example():
    """Example usage of meta-label trainer."""
    from gate_calibration import generate_synthetic_trades

    print("Generating synthetic trades...")
    trades = generate_synthetic_trades(n_trades=1000)

    print(f"Generated {len(trades)} trades")
    print(f"Overall WR: {sum(1 for t in trades if t.won) / len(trades):.1%}\n")

    # Train models
    for model_type in ['logistic', 'random_forest', 'xgboost']:
        print(f"\n{'='*70}")
        print(f"Training {model_type.upper()} model")
        print('='*70)

        try:
            trainer = MetaLabelTrainer(model_type=model_type)
            result = trainer.fit(trades, verbose=True)

            # Test prediction
            test_features = {
                'confidence': 0.75,
                'regime': 'TREND',
                'technique': 'TREND',
                'trend_strength': 0.80,
                'adx': 35.0,
            }

            win_prob = trainer.predict(test_features)
            print(f"\nTest Prediction:")
            print(f"  Features: TREND + high confidence")
            print(f"  Predicted Win Prob: {win_prob:.1%}")

            # Save model
            trainer.save(f'meta_label_{model_type}.pkl')

        except Exception as e:
            print(f"Error training {model_type}: {e}")


if __name__ == '__main__':
    run_training_example()
