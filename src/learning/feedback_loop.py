"""Learning and model retraining module."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
import json

from src.data_collection.data_fetcher import DataFetcher
from src.feature_engineering.features import FeatureEngineer
from src.models.training import ModelTrainer
from src.utils.logger import get_logger
from src.utils.helpers import load_config

logger = get_logger(__name__)


class LearningLoop:
    """Manage model retraining and continuous learning."""

    def __init__(self, config: Dict[str, Any] = None):
        if config is None:
            config = load_config()

        self.config = config
        self.learning_config = config.get('learning', {})
        self.data_fetcher = DataFetcher(config)
        self.feature_engineer = FeatureEngineer()
        self.model_trainer = ModelTrainer(config)

        self.feedback_dir = Path('data/feedback')
        self.feedback_dir.mkdir(parents=True, exist_ok=True)

        self.last_retrain = None
        self.model_performance_history = []

    def record_prediction(self, prediction: Dict[str, Any],
                         market: Dict[str, Any]) -> None:
        """
        Record a prediction for later evaluation.

        Args:
            prediction: Prediction dictionary
            market: Market information
        """
        record = {
            'timestamp': datetime.utcnow().isoformat(),
            'prediction': prediction,
            'market': market,
            'market_id': market.get('id'),
            'location': market.get('location'),
            'question': market.get('question'),
        }

        try:
            feedback_file = self.feedback_dir / 'predictions.jsonl'
            with open(feedback_file, 'a') as f:
                f.write(json.dumps(record) + '\n')
        except Exception as e:
            logger.error(f"Error recording prediction: {e}")

    def record_outcome(self, market_id: str, actual_outcome: bool,
                      execution_record: Dict = None) -> None:
        """
        Record the actual outcome for a prediction.

        Args:
            market_id: Market ID
            actual_outcome: Whether YES occurred
            execution_record: Trade execution details
        """
        record = {
            'timestamp': datetime.utcnow().isoformat(),
            'market_id': market_id,
            'actual_outcome': actual_outcome,
            'execution': execution_record or {}
        }

        try:
            feedback_file = self.feedback_dir / 'outcomes.jsonl'
            with open(feedback_file, 'a') as f:
                f.write(json.dumps(record) + '\n')
        except Exception as e:
            logger.error(f"Error recording outcome: {e}")

    def should_retrain(self) -> bool:
        """Determine if model should be retrained."""
        learning_cfg = self.learning_config

        # Check if retraining is enabled
        if not learning_cfg.get('enabled', True):
            return False

        # Check time-based trigger
        if self.last_retrain is None:
            return True

        retrain_interval = learning_cfg.get('retraining', {}).get('trigger_conditions', {})\
            .get('model_age', 604800)  # 7 days default

        if (datetime.utcnow() - self.last_retrain).total_seconds() > retrain_interval:
            logger.info("Model age exceeded retrain interval")
            return True

        # Check accuracy drop trigger
        if self._check_accuracy_drop():
            logger.info("Model accuracy dropped, retraining triggered")
            return True

        # Check new samples trigger
        min_samples = learning_cfg.get('retraining', {}).get('trigger_conditions', {})\
            .get('new_samples', 500)
        if self._count_new_feedback() >= min_samples:
            logger.info(f"Accumulated {min_samples} new samples, retraining triggered")
            return True

        return False

    def retrain_model(self, location: str = 'New York') -> bool:
        """
        Retrain model on accumulated data.

        Args:
            location: Location to retrain for

        Returns:
            True if successful
        """
        try:
            logger.info(f"Starting model retraining for {location}")

            # Load data (use 30 days for retraining)
            features, target = self.data_fetcher.prepare_training_data(
                location, days=30
            )

            if features.empty:
                logger.error(f"No training data available for {location}")
                return False

            # Engineer features
            features = self.feature_engineer.engineer_features(features)

            if features.empty:
                logger.error("Feature engineering produced empty DataFrame")
                return False

            # Split data
            X_train, X_test, y_train, y_test = self.model_trainer.prepare_training_data(
                features, target
            )

            # Train model
            metrics = self.model_trainer.train(X_train, y_train, X_test, y_test)

            if not metrics:
                logger.error("Training failed")
                return False

            # Save model
            model_path = self.model_trainer.save_model(f"model_{location}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

            self.last_retrain = datetime.utcnow()

            # Log performance
            self.model_performance_history.append({
                'timestamp': self.last_retrain.isoformat(),
                'location': location,
                'metrics': metrics,
                'model_path': model_path
            })

            logger.info(f"Model retrained successfully. Metrics: {metrics}")
            return True

        except Exception as e:
            logger.error(f"Retraining failed: {e}")
            return False

    def evaluate_predictions(self, limit: int = 1000) -> Dict[str, Any]:
        """
        Evaluate prediction accuracy on recorded outcomes.

        Args:
            limit: Maximum number of records to evaluate

        Returns:
            Evaluation metrics dictionary
        """
        try:
            # Load predictions and outcomes
            predictions = self._load_feedback_data('predictions.jsonl', limit)
            outcomes = self._load_feedback_data('outcomes.jsonl', limit)

            if not predictions or not outcomes:
                logger.warning("No data available for evaluation")
                return {}

            # Match predictions to outcomes
            matched = self._match_predictions_to_outcomes(predictions, outcomes)

            if not matched:
                logger.warning("No matching predictions and outcomes")
                return {}

            # Calculate metrics
            metrics = self._calculate_evaluation_metrics(matched)

            return metrics

        except Exception as e:
            logger.error(f"Error evaluating predictions: {e}")
            return {}

    def _load_feedback_data(self, filename: str, limit: int = 1000) -> List[Dict]:
        """Load feedback data from file."""
        try:
            filepath = self.feedback_dir / filename
            if not filepath.exists():
                return []

            data = []
            with open(filepath, 'r') as f:
                for i, line in enumerate(f):
                    if i >= limit:
                        break
                    try:
                        data.append(json.loads(line))
                    except:
                        pass

            return data

        except Exception as e:
            logger.error(f"Error loading feedback data: {e}")
            return []

    def _match_predictions_to_outcomes(self, predictions: List[Dict],
                                       outcomes: List[Dict]) -> List[Dict]:
        """Match predictions with their actual outcomes."""
        matched = []

        outcome_map = {o['market_id']: o for o in outcomes}

        for pred in predictions:
            market_id = pred.get('market_id')
            if market_id in outcome_map:
                outcome = outcome_map[market_id]
                matched.append({
                    'predicted_prob': pred['prediction'].get('rain_probability'),
                    'actual': outcome['actual_outcome'],
                    'timestamp_pred': pred['timestamp'],
                    'timestamp_outcome': outcome['timestamp']
                })

        return matched

    def _calculate_evaluation_metrics(self, matched: List[Dict]) -> Dict[str, Any]:
        """Calculate evaluation metrics."""
        if not matched:
            return {}

        predictions = np.array([m['predicted_prob'] for m in matched])
        actuals = np.array([1 if m['actual'] else 0 for m in matched])

        # Accuracy (binary classification)
        binary_preds = (predictions > 0.5).astype(int)
        accuracy = np.mean(binary_preds == actuals)

        # Calibration
        calibration_error = np.mean(np.abs(predictions - actuals))

        # AUC (requires binary actuals)
        from sklearn.metrics import roc_auc_score
        try:
            auc = roc_auc_score(actuals, predictions)
        except:
            auc = 0.0

        # Count wins and losses
        predictions_correct = (binary_preds == actuals).astype(int)
        wins = np.sum(predictions_correct)
        losses = len(matched) - wins

        return {
            'total_predictions': len(matched),
            'accuracy': float(accuracy),
            'calibration_error': float(calibration_error),
            'auc': float(auc),
            'wins': int(wins),
            'losses': int(losses),
            'win_rate': float(wins / len(matched)) if matched else 0,
            'timestamp': datetime.utcnow().isoformat()
        }

    def _check_accuracy_drop(self) -> bool:
        """Check if model accuracy has dropped significantly."""
        if len(self.model_performance_history) < 2:
            return False

        recent = self.model_performance_history[-1]
        previous = self.model_performance_history[-2]

        recent_acc = recent['metrics'].get('val_accuracy', 0)
        previous_acc = previous['metrics'].get('val_accuracy', 0)

        accuracy_drop = previous_acc - recent_acc
        threshold = self.learning_config.get('retraining', {}).get('trigger_conditions', {})\
            .get('accuracy_drop', 0.05)

        return accuracy_drop > threshold

    def _count_new_feedback(self) -> int:
        """Count new feedback records since last retrain."""
        try:
            feedback_file = self.feedback_dir / 'predictions.jsonl'
            if not feedback_file.exists():
                return 0

            # Simple count - in production would check timestamps
            with open(feedback_file, 'r') as f:
                return sum(1 for _ in f)

        except:
            return 0

    def get_learning_status(self) -> Dict[str, Any]:
        """Get current learning system status."""
        return {
            'enabled': self.learning_config.get('enabled', True),
            'last_retrain': self.last_retrain.isoformat() if self.last_retrain else None,
            'should_retrain': self.should_retrain(),
            'model_history_length': len(self.model_performance_history),
            'latest_metrics': self.model_performance_history[-1] if self.model_performance_history else None,
            'prediction_records': self._count_new_feedback()
        }

    def export_learning_data(self, output_dir: str = 'data/learning_exports') -> bool:
        """Export learning data for analysis."""
        try:
            out_path = Path(output_dir)
            out_path.mkdir(parents=True, exist_ok=True)

            # Export predictions
            predictions = self._load_feedback_data('predictions.jsonl')
            with open(out_path / 'predictions.json', 'w') as f:
                json.dump(predictions, f, indent=2, default=str)

            # Export outcomes
            outcomes = self._load_feedback_data('outcomes.jsonl')
            with open(out_path / 'outcomes.json', 'w') as f:
                json.dump(outcomes, f, indent=2, default=str)

            # Export metrics
            with open(out_path / 'performance_history.json', 'w') as f:
                json.dump(self.model_performance_history, f, indent=2, default=str)

            logger.info(f"Learning data exported to {output_dir}")
            return True

        except Exception as e:
            logger.error(f"Error exporting learning data: {e}")
            return False
