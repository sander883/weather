"""Feature engineering for weather prediction model."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FeatureEngineer:
    """Generate features for ML model."""

    def __init__(self, lookback_windows: List[str] = None, seasonal_features: bool = True):
        """
        Initialize feature engineer.

        Args:
            lookback_windows: List of time windows (e.g., ['3h', '6h', '12h', '24h'])
            seasonal_features: Whether to include seasonal features
        """
        # Use fewer, more meaningful windows for small datasets
        self.lookback_windows = lookback_windows or ['6h', '24h']
        self.seasonal_features = seasonal_features

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all features from raw weather data.

        Args:
            df: DataFrame with weather data (must have 'timestamp' column)

        Returns:
            DataFrame with engineered features
        """
        if df.empty:
            logger.warning("Input DataFrame is empty")
            return pd.DataFrame()

        # Make a copy to avoid modifying original
        df = df.copy()

        # Ensure timestamp is datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)

        # Validate we have required columns
        required_cols = ['temperature', 'humidity', 'wind_speed', 'clouds', 'precipitation']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            logger.warning(f"Missing columns: {missing}, filling with defaults")
            for col in missing:
                df[col] = 0 if col == 'precipitation' else 50

        # Fill NaN values before feature engineering
        df = self._fill_missing_values(df)

        # Create copy for features
        features_df = df.copy()

        # Time-based features
        features_df = self._add_time_features(features_df)

        # Moving average features (only most important windows)
        features_df = self._add_moving_averages(features_df)

        # Trend features (only for key columns)
        features_df = self._add_trend_features(features_df)

        # Rate of change features
        features_df = self._add_roc_features(features_df)

        # Seasonal features
        if self.seasonal_features:
            features_df = self._add_seasonal_features(features_df)

        # Forward fill any remaining NaN (better than dropping data)
        features_df = features_df.fillna(method='ffill').fillna(method='bfill').fillna(0)

        # Keep only numeric columns for model
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        features_df = features_df[numeric_cols]

        logger.info(f"Generated {len(features_df.columns)} features from {len(features_df)} samples")
        return features_df

    def _fill_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values intelligently."""
        # Forward fill first, then backward fill for any remaining NaN
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isna().any():
                df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
                # If still NaN (first rows), use column mean
                if df[col].isna().any():
                    df[col] = df[col].fillna(df[col].mean())
                # Last resort: fill with sensible defaults
                if df[col].isna().any():
                    if 'temp' in col.lower():
                        df[col] = 15
                    elif 'humidity' in col.lower():
                        df[col] = 50
                    elif 'wind' in col.lower():
                        df[col] = 0
                    else:
                        df[col] = 0

        return df

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features."""
        if 'timestamp' not in df.columns:
            logger.warning("No timestamp column found, using defaults")
            df['hour'] = 12
            df['day_of_week'] = 0
            df['day_of_month'] = 1
            df['month'] = 1
            df['is_weekend'] = 0
            return df

        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_month'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

        # Time of day features
        df['is_morning'] = ((df['hour'] >= 6) & (df['hour'] < 12)).astype(int)
        df['is_afternoon'] = ((df['hour'] >= 12) & (df['hour'] < 18)).astype(int)
        df['is_evening'] = ((df['hour'] >= 18) & (df['hour'] < 24)).astype(int)
        df['is_night'] = ((df['hour'] >= 0) & (df['hour'] < 6)).astype(int)

        return df

    def _add_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add moving average features for key columns only."""
        # Only use most important windows to reduce feature explosion
        windows = [6, 24]  # 6h and 24h periods

        key_cols = ['temperature', 'humidity', 'precipitation']

        for col in key_cols:
            if col not in df.columns:
                continue

            for window in windows:
                # Moving average
                df[f'{col}_ma_{window}'] = df[col].rolling(
                    window=window, min_periods=1
                ).mean()

                # Standard deviation
                df[f'{col}_std_{window}'] = df[col].rolling(
                    window=window, min_periods=1
                ).std().fillna(0)

        return df

    def _add_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trend features for key columns."""
        windows = [6, 24]

        for col in ['temperature', 'precipitation']:
            if col not in df.columns:
                continue

            for window in windows:
                df[f'{col}_trend_{window}'] = df[col].rolling(
                    window=window, min_periods=2
                ).apply(self._calculate_trend, raw=True).fillna(0)

        return df

    def _add_roc_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rate of change features."""
        windows = [6, 24]

        for col in ['temperature', 'humidity', 'precipitation']:
            if col not in df.columns:
                continue

            for window in windows:
                # Percent change
                df[f'{col}_roc_{window}'] = df[col].pct_change(periods=window).fillna(0)

        return df

    def _add_seasonal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add seasonal features."""
        if 'month' not in df.columns:
            logger.warning("No month feature, skipping seasonal features")
            return df

        # Seasonal indicators
        df['is_winter'] = df['month'].isin([12, 1, 2]).astype(int)
        df['is_spring'] = df['month'].isin([3, 4, 5]).astype(int)
        df['is_summer'] = df['month'].isin([6, 7, 8]).astype(int)
        df['is_fall'] = df['month'].isin([9, 10, 11]).astype(int)

        # Cyclical encoding (sin/cos for circular features)
        if 'hour' in df.columns:
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

        if 'month' in df.columns:
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        return df

    @staticmethod
    def _calculate_trend(values: np.ndarray) -> float:
        """Calculate linear trend (slope) for a series."""
        if len(values) < 2:
            return 0.0

        # Remove NaN values
        values = values[~np.isnan(values)]
        if len(values) < 2:
            return 0.0

        try:
            x = np.arange(len(values))
            slope = np.polyfit(x, values, 1)[0]
            return float(slope)
        except Exception as e:
            logger.debug(f"Error calculating trend: {e}")
            return 0.0

    def select_important_features(self, feature_df: pd.DataFrame, target: pd.Series,
                                 method: str = 'correlation', top_n: int = 20) -> List[str]:
        """
        Select most important features using various methods.

        Args:
            feature_df: DataFrame with features
            target: Target variable series
            method: 'xgboost', 'correlation', or 'variance'
            top_n: Number of top features to return

        Returns:
            List of feature names
        """
        if feature_df.empty:
            logger.warning("Feature DataFrame is empty")
            return []

        if method == 'correlation':
            # Use correlation with target
            try:
                correlations = feature_df.corr(numeric_only=True)[target.name].abs()
                correlations = correlations.dropna().sort_values(ascending=False)
                important = correlations.head(top_n).index.tolist()
                logger.info(f"Selected {len(important)} features by correlation")
                return important
            except Exception as e:
                logger.warning(f"Correlation selection failed: {e}")
                return list(feature_df.columns[:top_n])

        elif method == 'variance':
            # Use variance as importance proxy
            variances = feature_df.var()
            important = variances.nlargest(top_n).index.tolist()
            logger.info(f"Selected {len(important)} features by variance")
            return important

        elif method == 'xgboost':
            # Use XGBoost for feature importance
            try:
                import xgboost as xgb

                model = xgb.XGBClassifier(max_depth=3, n_estimators=50, random_state=42)

                # Use only numeric features
                X = feature_df.select_dtypes(include=[np.number])
                y = target.fillna(0)

                if len(X) < 2:
                    logger.warning("Not enough samples for XGBoost feature selection")
                    return list(X.columns[:top_n])

                model.fit(X, y, verbose=False)

                importances = pd.Series(
                    model.feature_importances_,
                    index=X.columns
                ).nlargest(top_n)

                important = importances.index.tolist()
                logger.info(f"Selected {len(important)} features by XGBoost importance")
                return important

            except Exception as e:
                logger.warning(f"XGBoost selection failed: {e}, falling back to correlation")
                return list(feature_df.columns[:top_n])

        return list(feature_df.columns[:top_n])

    def create_lag_features(self, df: pd.DataFrame, lags: List[int] = None) -> pd.DataFrame:
        """
        Create lagged features.

        Args:
            df: DataFrame with features
            lags: List of lag periods to create

        Returns:
            DataFrame with lag features
        """
        if lags is None:
            lags = [1, 3, 6, 12]

        for col in ['temperature', 'humidity', 'wind_speed', 'precipitation']:
            if col in df.columns:
                for lag in lags:
                    df[f'{col}_lag_{lag}'] = df[col].shift(lag)

        return df

    def scale_features(self, df: pd.DataFrame, scaler=None):
        """
        Scale features to [0, 1] range.

        Args:
            df: DataFrame with features
            scaler: Optional pre-fitted scaler

        Returns:
            Scaled DataFrame and scaler
        """
        from sklearn.preprocessing import MinMaxScaler

        if scaler is None:
            scaler = MinMaxScaler()
            scaled = scaler.fit_transform(df)
        else:
            scaled = scaler.transform(df)

        return pd.DataFrame(scaled, columns=df.columns), scaler
