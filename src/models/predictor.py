"""Prediction module using trained models."""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Any
from datetime import datetime

from src.feature_engineering.features import FeatureEngineer
from src.utils.logger import get_logger
from src.utils.helpers import load_config

logger = get_logger(__name__)


class Predictor:
    """Make predictions using trained weather models."""

    def __init__(self, model_trainer=None):
        """
        Initialize predictor.

        Args:
            model_trainer: ModelTrainer instance with trained model
        """
        self.model_trainer = model_trainer
        self.feature_engineer = FeatureEngineer()
        self.config = load_config()

    def predict(self, data: Dict[str, Any]) -> Optional[Dict[str, float]]:
        """
        Make prediction for weather event.

        Args:
            data: Weather data dictionary with features

        Returns:
            Dictionary with probability predictions
        """
        # Use simple baseline if no model is trained
        if self.model_trainer is None or self.model_trainer.model is None:
            logger.debug("No trained model available, using baseline prediction")
            return self._baseline_prediction(data)

        try:
            # Convert data to feature vector
            features = self._prepare_features(data)

            if features is None or features.empty:
                logger.warning("Could not prepare features, falling back to baseline")
                return self._baseline_prediction(data)

            # Ensure features are numeric only
            numeric_features = features.select_dtypes(include=[np.number])

            if numeric_features.empty:
                logger.warning("No numeric features available, using baseline")
                return self._baseline_prediction(data)

            # Get prediction
            X = numeric_features.values.reshape(1, -1)

            try:
                probability = self.model_trainer.model.predict_proba(X)[0, 1]
            except Exception as e:
                logger.warning(f"Model prediction failed ({e}), using baseline")
                return self._baseline_prediction(data)

            return {
                'rain_probability': float(probability),
                'no_rain_probability': float(1 - probability),
                'prediction': 'rain' if probability > 0.5 else 'no_rain',
                'confidence': float(max(probability, 1 - probability)),
                'timestamp': datetime.utcnow().isoformat(),
                'model_type': 'xgboost'
            }

        except Exception as e:
            logger.warning(f"Prediction failed ({e}), falling back to baseline")
            return self._baseline_prediction(data)

    def predict_batch(self, data_list: list) -> list:
        """
        Make predictions for multiple data points.

        Args:
            data_list: List of weather data dictionaries

        Returns:
            List of prediction dictionaries
        """
        predictions = []
        for data in data_list:
            pred = self.predict(data)
            if pred:
                predictions.append(pred)

        return predictions

    def predict_with_confidence(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Make prediction with confidence interval.

        Args:
            data: Weather data dictionary

        Returns:
            Prediction with confidence metrics
        """
        pred = self.predict(data)
        if not pred:
            return None

        # Get feature importance
        importance = self.model_trainer.get_feature_importance(top_n=5)

        # Add confidence information
        pred['confidence_metrics'] = {
            'top_features': importance,
            'model_type': 'xgboost',
        }

        return pred

    def predict_forecast(self, forecast_data: list) -> Optional[pd.DataFrame]:
        """
        Make predictions for a series of forecast points.

        Args:
            forecast_data: List of forecast dictionaries

        Returns:
            DataFrame with predictions for each time step
        """
        if not forecast_data:
            return None

        predictions = []

        for data in forecast_data:
            pred = self.predict(data)
            if pred:
                pred['timestamp'] = data.get('timestamp')
                predictions.append(pred)

        if not predictions:
            return None

        df = pd.DataFrame(predictions)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df.sort_values('timestamp')

    def _baseline_prediction(self, data: Dict[str, Any]) -> Dict[str, float]:
        """
        Make baseline prediction based on raw weather data.

        Used when no ML model is available yet.
        """
        # Simple heuristic: humidity > 70% and clouds > 50% = higher rain chance
        humidity = data.get('humidity', 50)
        clouds = data.get('clouds', 50)
        wind_speed = data.get('wind_speed', 0)

        # Baseline probability: average of indicators
        rain_prob = (humidity / 100 + clouds / 100 + min(wind_speed / 20, 1.0)) / 3

        return {
            'rain_probability': float(min(rain_prob, 1.0)),
            'no_rain_probability': float(max(1 - rain_prob, 0.0)),
            'prediction': 'rain' if rain_prob > 0.5 else 'no_rain',
            'confidence': 0.5,  # Low confidence for baseline
            'timestamp': datetime.utcnow().isoformat(),
            'model_type': 'baseline'
        }

    def _prepare_features(self, data: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """
        Prepare features from raw data.

        Args:
            data: Raw weather data

        Returns:
            DataFrame with engineered features or None
        """
        try:
            # Create single-row DataFrame with required columns
            data_clean = {
                'temperature': data.get('temperature', 15),
                'humidity': data.get('humidity', 50),
                'wind_speed': data.get('wind_speed', 0),
                'clouds': data.get('clouds', 50),
                'pressure': data.get('pressure', 1013.25),
                'precipitation': data.get('precipitation', 0),
                'timestamp': data.get('timestamp', datetime.utcnow()),
            }

            df = pd.DataFrame([data_clean])

            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.set_index('timestamp')

            # Apply feature engineering
            features = self.feature_engineer.engineer_features(df)

            if features.empty:
                logger.warning("Feature engineering produced empty DataFrame")
                return None

            # Keep only numeric columns
            features = features.select_dtypes(include=[np.number])

            if features.empty:
                logger.warning("No numeric features available after engineering")
                return None

            return features.iloc[[0]] if len(features) > 0 else None

        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            return None

    def ensemble_predict(self, data: Dict[str, Any], models: list) -> Optional[Dict[str, Any]]:
        """
        Make ensemble prediction using multiple models.

        Args:
            data: Weather data
            models: List of model paths or ModelTrainer instances

        Returns:
            Ensemble prediction
        """
        predictions = []

        for model in models:
            if isinstance(model, str):
                # Load model from path
                from src.models.training import ModelTrainer
                trainer = ModelTrainer(self.config)
                if not trainer.load_model(model):
                    logger.warning(f"Could not load model {model}")
                    continue
                self.model_trainer = trainer
            else:
                self.model_trainer = model

            pred = self.predict(data)
            if pred:
                predictions.append(pred['rain_probability'])

        if not predictions:
            return None

        ensemble_prob = np.mean(predictions)
        ensemble_std = np.std(predictions)

        return {
            'rain_probability': float(ensemble_prob),
            'no_rain_probability': float(1 - ensemble_prob),
            'prediction': 'rain' if ensemble_prob > 0.5 else 'no_rain',
            'confidence': float(1 - ensemble_std),  # Lower std = higher confidence
            'ensemble_size': len(predictions),
            'timestamp': datetime.utcnow().isoformat()
        }

    def get_prediction_for_market(self, data: Dict[str, Any], market: Dict[str, Any]) -> Optional[Dict]:
        """
        Get prediction for a specific market.

        Args:
            data: Weather data
            market: Market configuration dict

        Returns:
            Prediction aligned with market outcome type
        """
        pred = self.predict(data)
        if not pred:
            return None

        # Get market mapping
        mapping = market.get('mapping', {})
        prediction_type = market.get('prediction_type')

        # Adapt prediction to market type
        if prediction_type == 'rain_probability':
            yes_prob = pred['rain_probability']
            no_prob = pred['no_rain_probability']
        elif prediction_type == 'temperature_threshold':
            # For threshold: probability that temp > threshold
            threshold = market.get('threshold', 30)
            temp = data.get('temperature', 20)

            # Simple linear estimation (would be better with full model)
            yes_prob = min(1.0, max(0.0, 0.5 + (temp - threshold) / 20))
            no_prob = 1 - yes_prob
        else:
            # Default to rain probability
            yes_prob = pred['rain_probability']
            no_prob = pred['no_rain_probability']

        return {
            'market_id': market.get('id'),
            'question': market.get('question'),
            'yes_probability': yes_prob,
            'no_probability': no_prob,
            'predicted_outcome': 'YES' if yes_prob > 0.5 else 'NO',
            'confidence': max(yes_prob, no_prob),
            'current_price_yes': market.get('current_price_yes'),
            'current_price_no': market.get('current_price_no'),
            'edge': abs(yes_prob - market.get('current_price_yes', 0.5)),
            'timestamp': datetime.utcnow().isoformat()
        }
