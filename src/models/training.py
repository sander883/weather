"""Model training pipeline."""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import pickle
import json
from datetime import datetime
from typing import Dict, Tuple, List, Any, Optional

from src.feature_engineering.features import FeatureEngineer
from src.utils.logger import get_logger
from src.utils.helpers import load_config

logger = get_logger(__name__)


class ModelTrainer:
    """Train and evaluate weather prediction models."""

    def __init__(self, config: Dict[str, Any] = None):
        if config is None:
            config = load_config()

        self.config = config
        self.model = None
        self.scaler = None
        self.feature_engineer = FeatureEngineer(
            lookback_windows=config.get('features', {}).get('lookback_windows', ['3h', '6h', '12h', '24h']),
            seasonal_features=config.get('features', {}).get('seasonal_features', True)
        )
        self.feature_names = None
        self.model_dir = Path('data/models')
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def prepare_training_data(self, features: pd.DataFrame, target: pd.DataFrame,
                             test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray,
                                                              np.ndarray, np.ndarray]:
        """
        Prepare and split training data.

        Args:
            features: Feature DataFrame
            target: Target variable Series
            test_size: Fraction for test set

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        # Ensure shapes match
        if len(features) != len(target):
            min_len = min(len(features), len(target))
            features = features.iloc[:min_len]
            target = target.iloc[:min_len]

        # Drop non-numeric and timestamp columns
        numeric_features = features.select_dtypes(include=[np.number])

        # Remove timestamp if it exists
        if 'timestamp' in numeric_features.columns:
            numeric_features = numeric_features.drop('timestamp', axis=1)

        logger.info(f"Using {len(numeric_features.columns)} numeric features")

        # Fill NaN values
        numeric_features = numeric_features.fillna(numeric_features.mean())
        target = target.fillna(target.mean())

        # Ensure target is 1D
        if hasattr(target, 'values'):
            target_array = target.values.ravel()
        else:
            target_array = target.ravel()

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            numeric_features, target_array,
            test_size=test_size,
            random_state=42,
            shuffle=False  # Keep temporal order
        )

        logger.info(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")
        return X_train.values, X_test.values, y_train, y_test

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Train XGBoost model.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)

        Returns:
            Training metrics dictionary
        """
        # Get model parameters from config
        params = self.config.get('model', {}).get('xgboost_params', {})

        try:
            self.model = xgb.XGBClassifier(**params)

            # Create validation set if provided
            eval_set = None
            if X_val is not None and y_val is not None:
                eval_set = [(X_train, y_train), (X_val, y_val)]

            # Train model
            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                verbose=False
            )

            # Evaluate on training data
            y_pred_train = self.model.predict(X_train)
            y_pred_proba_train = self.model.predict_proba(X_train)[:, 1]

            metrics = {
                'train_accuracy': accuracy_score(y_train, y_pred_train),
                'train_auc': roc_auc_score(y_train, y_pred_proba_train),
                'train_precision': precision_score(y_train, y_pred_train, zero_division=0),
                'train_recall': recall_score(y_train, y_pred_train, zero_division=0),
                'train_f1': f1_score(y_train, y_pred_train, zero_division=0),
            }

            # Validation metrics
            if X_val is not None and y_val is not None:
                y_pred_val = self.model.predict(X_val)
                y_pred_proba_val = self.model.predict_proba(X_val)[:, 1]

                metrics.update({
                    'val_accuracy': accuracy_score(y_val, y_pred_val),
                    'val_auc': roc_auc_score(y_val, y_pred_proba_val),
                    'val_precision': precision_score(y_val, y_pred_val, zero_division=0),
                    'val_recall': recall_score(y_val, y_pred_val, zero_division=0),
                    'val_f1': f1_score(y_val, y_pred_val, zero_division=0),
                })

            logger.info(f"Training complete. Metrics: {metrics}")
            return metrics

        except Exception as e:
            logger.error(f"Training failed: {e}")
            return {}

    def cross_validate(self, X: np.ndarray, y: np.ndarray, cv_folds: int = 5) -> Dict[str, float]:
        """
        Perform cross-validation.

        Args:
            X: Features
            y: Targets
            cv_folds: Number of folds

        Returns:
            Cross-validation scores
        """
        try:
            params = self.config.get('model', {}).get('xgboost_params', {})
            model = xgb.XGBClassifier(**params)

            scores = cross_val_score(model, X, y, cv=cv_folds, scoring='roc_auc')

            return {
                'cv_mean': scores.mean(),
                'cv_std': scores.std(),
                'cv_scores': scores.tolist()
            }

        except Exception as e:
            logger.error(f"Cross-validation failed: {e}")
            return {}

    def get_feature_importance(self, top_n: int = 20) -> Dict[str, float]:
        """Get feature importance from trained model."""
        if self.model is None:
            logger.error("Model not trained yet")
            return {}

        try:
            importances = pd.Series(
                self.model.feature_importances_,
                index=self.feature_names or [f'feature_{i}' for i in range(len(self.model.feature_importances_))]
            ).nlargest(top_n)

            return importances.to_dict()

        except Exception as e:
            logger.error(f"Error getting feature importance: {e}")
            return {}

    def save_model(self, name: str = None) -> str:
        """
        Save trained model to disk.

        Args:
            name: Model name (optional, defaults to timestamp)

        Returns:
            Path to saved model
        """
        if self.model is None:
            logger.error("No model to save")
            return ""

        try:
            if name is None:
                name = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            model_path = self.model_dir / f"{name}.pkl"
            scaler_path = self.model_dir / f"{name}_scaler.pkl"
            metadata_path = self.model_dir / f"{name}_metadata.json"

            # Save model
            with open(model_path, 'wb') as f:
                pickle.dump(self.model, f)

            # Save scaler
            if self.scaler:
                with open(scaler_path, 'wb') as f:
                    pickle.dump(self.scaler, f)

            # Save metadata
            metadata = {
                'trained_at': datetime.now().isoformat(),
                'feature_names': self.feature_names,
                'config': self.config.get('model', {}),
            }
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"Model saved to {model_path}")
            return str(model_path)

        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return ""

    def load_model(self, model_path: str) -> bool:
        """
        Load trained model from disk.

        Args:
            model_path: Path to saved model

        Returns:
            True if successful
        """
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)

            # Try to load scaler
            scaler_path = model_path.replace('.pkl', '_scaler.pkl')
            try:
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
            except:
                self.scaler = None

            # Try to load metadata
            metadata_path = model_path.replace('.pkl', '_metadata.json')
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    self.feature_names = metadata.get('feature_names')
            except:
                pass

            logger.info(f"Model loaded from {model_path}")
            return True

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False

    def get_latest_model(self) -> Optional[str]:
        """Get path to most recently trained model."""
        models = list(self.model_dir.glob('model_*.pkl'))
        if not models:
            return None

        return str(max(models, key=lambda p: p.stat().st_mtime))
