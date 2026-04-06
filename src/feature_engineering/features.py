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
        self.lookback_windows = lookback_windows or ['3h', '6h', '12h', '24h']
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
            return pd.DataFrame()

        # Ensure timestamp is datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').set_index('timestamp')

        # Create copy to avoid modifying original
        features_df = df.copy()

        # Time-based features
        features_df = self._add_time_features(features_df)

        # Moving average features
        features_df = self._add_moving_averages(features_df)

        # Trend features
        features_df = self._add_trend_features(features_df)

        # Volatility features
        features_df = self._add_volatility_features(features_df)

        # Rate of change features
        features_df = self._add_roc_features(features_df)

        # Seasonal features
        if self.seasonal_features:
            features_df = self._add_seasonal_features(features_df)

        # Drop rows with NaN (created by rolling windows)
        features_df = features_df.dropna()

        # Reset index
        features_df = features_df.reset_index()

        logger.info(f"Generated {len(features_df.columns)} features")
        return features_df

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features."""
        # Handle both index and column timestamp
        if df.index.name == 'timestamp' or isinstance(df.index, pd.DatetimeIndex):
            df['hour'] = df.index.hour
            df['day_of_week'] = df.index.dayofweek
            df['day_of_month'] = df.index.day
            df['month'] = df.index.month
        elif 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['day_of_month'] = df['timestamp'].dt.day
            df['month'] = df['timestamp'].dt.month
        else:
            # Default values if no timestamp available
            logger.warning("No timestamp found for time feature engineering")
            df['hour'] = 12
            df['day_of_week'] = 0
            df['day_of_month'] = 1
            df['month'] = 1
            return df

        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

        # Time of day features (categorical encoding)
        df['is_morning'] = ((df['hour'] >= 6) & (df['hour'] < 12)).astype(int)
        df['is_afternoon'] = ((df['hour'] >= 12) & (df['hour'] < 18)).astype(int)
        df['is_evening'] = ((df['hour'] >= 18) & (df['hour'] < 24)).astype(int)
        df['is_night'] = ((df['hour'] >= 0) & (df['hour'] < 6)).astype(int)

        return df

    def _add_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add moving average features."""
        for window in self.lookback_windows:
            periods = self._get_periods(window)

            for col in ['temperature', 'humidity', 'wind_speed', 'clouds', 'precipitation']:
                if col in df.columns:
                    df[f'{col}_ma_{window}'] = df[col].rolling(window=periods, min_periods=1).mean()
                    df[f'{col}_std_{window}'] = df[col].rolling(window=periods, min_periods=1).std()

        return df

    def _add_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trend features using linear regression slope."""
        for window in self.lookback_windows:
            periods = self._get_periods(window)

            for col in ['temperature', 'humidity', 'wind_speed', 'precipitation']:
                if col in df.columns:
                    df[f'{col}_trend_{window}'] = df[col].rolling(window=periods, min_periods=2).apply(
                        self._calculate_trend, raw=True
                    )

        return df

    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility features."""
        for window in self.lookback_windows:
            periods = self._get_periods(window)

            for col in ['temperature', 'wind_speed']:
                if col in df.columns:
                    df[f'{col}_volatility_{window}'] = (
                        df[col].rolling(window=periods, min_periods=1).std() /
                        (df[col].rolling(window=periods, min_periods=1).mean() + 1e-6)
                    )

        return df

    def _add_roc_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rate of change features."""
        for window in self.lookback_windows:
            periods = self._get_periods(window)

            for col in ['temperature', 'humidity', 'precipitation']:
                if col in df.columns:
                    df[f'{col}_roc_{window}'] = df[col].pct_change(periods=periods)

        return df

    def _add_seasonal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add seasonal features."""
        # Winter months
        df['is_winter'] = df['month'].isin([12, 1, 2]).astype(int)
        df['is_spring'] = df['month'].isin([3, 4, 5]).astype(int)
        df['is_summer'] = df['month'].isin([6, 7, 8]).astype(int)
        df['is_fall'] = df['month'].isin([9, 10, 11]).astype(int)

        # Cyclical encoding for hour and month (sin/cos transformation)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        return df

    @staticmethod
    def _get_periods(window: str) -> int:
        """Convert time window string to number of periods (assuming hourly data)."""
        mapping = {
            '1h': 1,
            '3h': 3,
            '6h': 6,
            '12h': 12,
            '24h': 24,
            '3d': 72,
            '7d': 168,
        }
        return mapping.get(window, 1)

    @staticmethod
    def _calculate_trend(values: np.ndarray) -> float:
        """Calculate linear trend (slope) for a series."""
        if len(values) < 2:
            return 0.0

        x = np.arange(len(values))
        try:
            slope = np.polyfit(x, values, 1)[0]
            return float(slope)
        except:
            return 0.0

    def select_important_features(self, feature_df: pd.DataFrame, target: pd.Series,
                                 method: str = 'xgboost', top_n: int = 50) -> List[str]:
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
        if method == 'correlation':
            # Use correlation with target
            correlations = feature_df.corr(numeric_only=True)[target.name].abs()
            important = correlations.nlargest(top_n).index.tolist()
            logger.info(f"Selected {len(important)} features by correlation")
            return important

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

                # Fill any remaining NaN values
                X = feature_df.fillna(feature_df.mean())
                y = target.fillna(target.mean())

                model.fit(X, y, verbose=False)

                importances = pd.Series(
                    model.feature_importances_,
                    index=feature_df.columns
                ).nlargest(top_n)

                important = importances.index.tolist()
                logger.info(f"Selected {len(important)} features by XGBoost importance")
                return important

            except ImportError:
                logger.warning("XGBoost not available, falling back to correlation")
                return self.select_important_features(
                    feature_df, target, method='correlation', top_n=top_n
                )

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
            lags = [1, 3, 6, 12, 24]

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
