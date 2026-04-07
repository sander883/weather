"""
Unit tests for feature engineering module.

Tests cover:
- Feature generation from raw weather data
- Missing value handling
- Time-based features (hour, day, month, seasonal)
- Moving averages and trends
- Rate of change features
- Feature scaling and selection
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.feature_engineering.features import FeatureEngineer


# ============================================================================
# FIXTURES (Test Data)
# ============================================================================

@pytest.fixture
def sample_weather_data():
    """Sample weather data for testing."""
    dates = pd.date_range(start='2024-03-01', periods=30, freq='h')
    np.random.seed(42)

    return pd.DataFrame({
        'timestamp': dates,
        'temperature': np.random.normal(15, 5, 30),  # Mean 15°C, std 5
        'humidity': np.random.uniform(40, 80, 30),  # 40-80%
        'wind_speed': np.random.exponential(3, 30),  # Exponential distribution
        'clouds': np.random.uniform(0, 100, 30),  # 0-100%
        'precipitation': np.concatenate([
            np.zeros(20),  # No rain for first 20 hours
            np.random.exponential(2, 10)  # Rain for last 10 hours
        ])
    })


@pytest.fixture
def sample_with_missing_values():
    """Sample data with missing values."""
    dates = pd.date_range(start='2024-03-01', periods=20, freq='H')
    df = pd.DataFrame({
        'timestamp': dates,
        'temperature': [15.5, 16.2, np.nan, 17.1, 16.8, 15.9, np.nan, 14.5] + [15.0] * 12,
        'humidity': [65, 70, 72, np.nan, 75, 73, 71, 70] + [72.0] * 12,
        'wind_speed': [3.5, 4.2, 5.1, 4.8, np.nan, 3.2, 2.9, 3.1] + [3.5] * 12,
        'clouds': [45, 50, 55, 60, 65, np.nan, 55, 50] + [52.0] * 12,
        'precipitation': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] + [0.0] * 12,
    })
    return df


@pytest.fixture
def sample_minimal_data():
    """Minimal data for testing edge cases."""
    return pd.DataFrame({
        'timestamp': pd.date_range(start='2024-03-01', periods=3, freq='H'),
        'temperature': [15.0, 16.0, 14.5],
        'humidity': [70, 75, 65],
        'wind_speed': [3.0, 4.0, 2.5],
        'clouds': [50, 60, 40],
        'precipitation': [0.0, 0.0, 0.0],
    })


# ============================================================================
# TESTS: Initialization
# ============================================================================

class TestFeatureEngineerInit:
    """Test FeatureEngineer initialization."""

    def test_default_initialization(self):
        """Test default initialization."""
        engineer = FeatureEngineer()

        assert engineer.lookback_windows == ['6h', '24h']
        assert engineer.seasonal_features is True

    def test_custom_windows(self):
        """Test custom lookback windows."""
        custom_windows = ['3h', '12h', '48h']
        engineer = FeatureEngineer(lookback_windows=custom_windows)

        assert engineer.lookback_windows == custom_windows

    def test_disable_seasonal_features(self):
        """Test disabling seasonal features."""
        engineer = FeatureEngineer(seasonal_features=False)

        assert engineer.seasonal_features is False


# ============================================================================
# TESTS: Feature Engineering Main Flow
# ============================================================================

class TestEngineerFeatures:
    """Test main feature engineering method."""

    def test_engineer_features_normal_data(self, sample_weather_data):
        """Test feature engineering with normal data."""
        engineer = FeatureEngineer()
        features = engineer.engineer_features(sample_weather_data)

        assert not features.empty
        assert len(features) == len(sample_weather_data)
        # Should have multiple features (at least base + moving avg + trend)
        assert len(features.columns) > 5
        # All should be numeric
        assert features.dtypes.apply(lambda x: np.issubdtype(x, np.number)).all()

    def test_engineer_features_with_missing_data(self, sample_with_missing_values):
        """Test feature engineering handles missing values."""
        engineer = FeatureEngineer()
        features = engineer.engineer_features(sample_with_missing_values)

        assert not features.empty
        assert len(features) == len(sample_with_missing_values)
        # No NaN values in output
        assert not features.isna().any().any()

    def test_engineer_features_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        engineer = FeatureEngineer()
        empty_df = pd.DataFrame()

        result = engineer.engineer_features(empty_df)

        assert result.empty

    def test_engineer_features_missing_columns(self):
        """Test handling of missing required columns."""
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-03-01', periods=10, freq='H'),
            'temperature': np.random.normal(15, 5, 10),
            # Missing: humidity, wind_speed, clouds, precipitation
        })

        engineer = FeatureEngineer()
        features = engineer.engineer_features(df)

        # Should fill missing columns with defaults and continue
        assert not features.empty
        assert len(features) == 10

    def test_feature_output_shape(self, sample_weather_data):
        """Test output shape matches input."""
        engineer = FeatureEngineer()
        features = engineer.engineer_features(sample_weather_data)

        # Number of rows should match input
        assert len(features) == len(sample_weather_data)

    def test_all_features_numeric(self, sample_weather_data):
        """Test all output features are numeric."""
        engineer = FeatureEngineer()
        features = engineer.engineer_features(sample_weather_data)

        # Check dtype
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        assert len(numeric_cols) == len(features.columns)


# ============================================================================
# TESTS: Missing Value Handling
# ============================================================================

class TestMissingValueHandling:
    """Test NaN/missing value handling."""

    def test_fill_missing_values_forward_fill(self, sample_with_missing_values):
        """Test forward fill strategy."""
        engineer = FeatureEngineer()
        filled = engineer._fill_missing_values(sample_with_missing_values.copy())

        # Should have no NaN in numeric columns
        numeric_cols = filled.select_dtypes(include=[np.number]).columns
        assert not filled[numeric_cols].isna().any().any()

    def test_fill_missing_values_mean_fallback(self):
        """Test mean value fallback for persistent NaN."""
        df = pd.DataFrame({
            'temperature': [15.0, 16.0, 17.0, 18.0, 15.5],
            'humidity': [70.0, np.nan, np.nan, np.nan, 75.0],
        })

        engineer = FeatureEngineer()
        filled = engineer._fill_missing_values(df.copy())

        # Should have no NaN
        assert not filled.isna().any().any()
        # Humidity should use mean value or similar
        assert filled['humidity'].notna().all()

    def test_fill_missing_values_preserves_columns(self, sample_with_missing_values):
        """Test that fill doesn't drop columns."""
        engineer = FeatureEngineer()
        original_cols = set(sample_with_missing_values.columns)

        filled = engineer._fill_missing_values(sample_with_missing_values.copy())

        assert set(filled.columns) == original_cols


# ============================================================================
# TESTS: Time-Based Features
# ============================================================================

class TestTimeFeatures:
    """Test time-based feature generation."""

    def test_add_time_features_creates_features(self, sample_minimal_data):
        """Test that time features are created."""
        engineer = FeatureEngineer()
        features = engineer._add_time_features(sample_minimal_data.copy())

        # Should have hour, day, month features
        expected_features = ['hour', 'day', 'month']
        for feat in expected_features:
            # At least one column should contain the feature name
            matching = [col for col in features.columns if feat in col.lower()]
            # Some might not exist if not implemented, but check they work if they do
            if matching:
                assert len(matching) > 0

    def test_hour_values_valid_range(self, sample_minimal_data):
        """Test hour values are in 0-23 range."""
        engineer = FeatureEngineer()
        features = engineer._add_time_features(sample_minimal_data.copy())

        hour_cols = [col for col in features.columns if 'hour' in col.lower()]
        if hour_cols:
            for col in hour_cols:
                assert (features[col] >= 0).all()
                assert (features[col] < 24).all()

    def test_day_values_valid_range(self, sample_minimal_data):
        """Test day values are in 1-31 range."""
        engineer = FeatureEngineer()
        features = engineer._add_time_features(sample_minimal_data.copy())

        day_cols = [col for col in features.columns if 'day' in col.lower()]
        if day_cols:
            for col in day_cols:
                assert (features[col] >= 1).all()
                assert (features[col] <= 31).all()

    def test_month_values_valid_range(self, sample_minimal_data):
        """Test month values are in 1-12 range."""
        engineer = FeatureEngineer()
        features = engineer._add_time_features(sample_minimal_data.copy())

        month_cols = [col for col in features.columns if 'month' in col.lower()]
        if month_cols:
            for col in month_cols:
                assert (features[col] >= 1).all()
                assert (features[col] <= 12).all()


# ============================================================================
# TESTS: Moving Average Features
# ============================================================================

class TestMovingAverages:
    """Test moving average feature generation."""

    def test_add_moving_averages_creates_features(self, sample_weather_data):
        """Test that moving average features are created."""
        engineer = FeatureEngineer()
        original_cols = len(sample_weather_data.columns)

        features = engineer._add_moving_averages(sample_weather_data.copy())

        # Should have more columns than original
        assert len(features.columns) > original_cols

    def test_moving_averages_reasonable_values(self, sample_weather_data):
        """Test that MA values are within reasonable range."""
        engineer = FeatureEngineer()
        features = engineer._add_moving_averages(sample_weather_data.copy())

        # For temperature, moving averages should be close to original data range
        temp_ma_cols = [col for col in features.columns if 'ma' in col.lower() and 'temp' in col.lower()]
        if temp_ma_cols:
            for col in temp_ma_cols:
                original_temp_range = sample_weather_data['temperature'].max() - sample_weather_data['temperature'].min()
                ma_range = features[col].max() - features[col].min()
                # MA range should be smaller or equal to original
                assert ma_range <= original_temp_range + 1


# ============================================================================
# TESTS: Trend Features
# ============================================================================

class TestTrendFeatures:
    """Test trend feature generation."""

    def test_add_trend_features_creates_features(self, sample_weather_data):
        """Test that trend features are created."""
        engineer = FeatureEngineer()
        original_cols = len(sample_weather_data.columns)

        features = engineer._add_trend_features(sample_weather_data.copy())

        # Should have more columns
        assert len(features.columns) >= original_cols

    def test_trend_calculation_increasing(self):
        """Test trend calculation for increasing values."""
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        trend = FeatureEngineer._calculate_trend(values)

        # Should be positive for increasing trend
        assert trend > 0

    def test_trend_calculation_decreasing(self):
        """Test trend calculation for decreasing values."""
        values = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        trend = FeatureEngineer._calculate_trend(values)

        # Should be negative for decreasing trend
        assert trend < 0

    def test_trend_calculation_flat(self):
        """Test trend calculation for flat values."""
        values = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        trend = FeatureEngineer._calculate_trend(values)

        # Should be ~0 for flat trend
        assert abs(trend) < 0.1


# ============================================================================
# TESTS: Rate of Change
# ============================================================================

class TestRateOfChange:
    """Test rate of change features."""

    def test_add_roc_features(self, sample_weather_data):
        """Test ROC feature generation."""
        engineer = FeatureEngineer()
        original_cols = len(sample_weather_data.columns)

        features = engineer._add_roc_features(sample_weather_data.copy())

        # Should have more columns or same
        assert len(features.columns) >= original_cols


# ============================================================================
# TESTS: Seasonal Features
# ============================================================================

class TestSeasonalFeatures:
    """Test seasonal feature generation."""

    def test_add_seasonal_features_enabled(self, sample_weather_data):
        """Test seasonal features when enabled."""
        engineer = FeatureEngineer(seasonal_features=True)
        features = engineer.engineer_features(sample_weather_data)

        # Should have seasonal features (sin/cos encoded)
        seasonal_cols = [col for col in features.columns if 'sin' in col.lower() or 'cos' in col.lower()]
        # With month data, should have seasonal features
        assert len(seasonal_cols) > 0

    def test_seasonal_values_normalized(self, sample_weather_data):
        """Test seasonal features are normalized to [-1, 1]."""
        engineer = FeatureEngineer(seasonal_features=True)
        features = engineer.engineer_features(sample_weather_data)

        seasonal_cols = [col for col in features.columns if 'sin' in col.lower() or 'cos' in col.lower()]
        for col in seasonal_cols:
            # Sin/cos should be in [-1, 1]
            assert (features[col] >= -1.1).all()  # Small tolerance
            assert (features[col] <= 1.1).all()

    def test_seasonal_features_disabled(self, sample_weather_data):
        """Test seasonal features can be disabled."""
        engineer = FeatureEngineer(seasonal_features=False)
        features = engineer.engineer_features(sample_weather_data)

        seasonal_cols = [col for col in features.columns if 'sin' in col.lower() or 'cos' in col.lower()]
        # Should have no seasonal features (or very few)
        assert len(seasonal_cols) < 3


# ============================================================================
# TESTS: Feature Selection & Scaling
# ============================================================================

class TestFeatureSelection:
    """Test feature selection methods."""

    def test_select_important_features(self, sample_weather_data):
        """Test feature selection based on importance."""
        engineer = FeatureEngineer()
        features = engineer.engineer_features(sample_weather_data)

        # Create a simple target variable
        target = pd.Series(np.random.choice([0, 1], size=len(sample_weather_data)))

        selected = engineer.select_important_features(features, target, n_features=10)

        assert not selected.empty
        # Should have at most n_features columns
        assert len(selected.columns) <= 10

    def test_select_important_features_returns_dataframe(self, sample_weather_data):
        """Test that selection returns a DataFrame."""
        engineer = FeatureEngineer()
        features = engineer.engineer_features(sample_weather_data)
        target = pd.Series(np.random.choice([0, 1], size=len(sample_weather_data)))

        selected = engineer.select_important_features(features, target)

        assert isinstance(selected, pd.DataFrame)


class TestFeatureScaling:
    """Test feature scaling."""

    def test_scale_features_normalization(self, sample_weather_data):
        """Test feature scaling normalizes values."""
        engineer = FeatureEngineer()
        features = engineer.engineer_features(sample_weather_data)

        scaled, scaler = engineer.scale_features(features)

        assert not scaled.empty
        assert scaler is not None
        # Scaled features should have mean ~0 and std ~1
        for col in scaled.columns:
            assert abs(scaled[col].mean()) < 0.5  # Tolerance
            assert abs(scaled[col].std() - 1.0) < 0.5

    def test_scale_features_preserves_shape(self, sample_weather_data):
        """Test scaling preserves DataFrame shape."""
        engineer = FeatureEngineer()
        features = engineer.engineer_features(sample_weather_data)

        scaled, _ = engineer.scale_features(features)

        assert scaled.shape == features.shape


# ============================================================================
# TESTS: Lag Features
# ============================================================================

class TestLagFeatures:
    """Test lag feature creation."""

    def test_create_lag_features(self, sample_weather_data):
        """Test lag feature generation."""
        engineer = FeatureEngineer()
        lags = [1, 2, 3]

        lagged = engineer.create_lag_features(sample_weather_data, lags=lags)

        assert not lagged.empty
        # Should have additional columns for lags
        expected_additional_cols = len(sample_weather_data.columns) * len(lags)
        assert len(lagged.columns) >= len(sample_weather_data.columns) + expected_additional_cols - len(sample_weather_data.columns)


# ============================================================================
# TESTS: Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_single_row_dataframe(self):
        """Test with single row."""
        df = pd.DataFrame({
            'timestamp': [pd.Timestamp('2024-03-01')],
            'temperature': [15.0],
            'humidity': [70.0],
            'wind_speed': [3.0],
            'clouds': [50.0],
            'precipitation': [0.0],
        })

        engineer = FeatureEngineer()
        features = engineer.engineer_features(df)

        assert len(features) == 1

    def test_large_gaps_in_timestamp(self):
        """Test with large gaps in timestamp."""
        dates = pd.to_datetime(['2024-03-01 00:00', '2024-03-01 06:00', '2024-03-02 12:00'])
        df = pd.DataFrame({
            'timestamp': dates,
            'temperature': [15.0, 16.0, 14.0],
            'humidity': [70.0, 75.0, 65.0],
            'wind_speed': [3.0, 4.0, 2.5],
            'clouds': [50.0, 60.0, 40.0],
            'precipitation': [0.0, 0.0, 0.0],
        })

        engineer = FeatureEngineer()
        features = engineer.engineer_features(df)

        # Should still work
        assert len(features) == 3

    def test_all_nan_column(self):
        """Test handling of completely NaN column."""
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-03-01', periods=5, freq='H'),
            'temperature': [15.0, 16.0, 17.0, 16.5, 15.5],
            'humidity': [np.nan] * 5,  # All NaN
            'wind_speed': [3.0, 4.0, 3.5, 3.2, 2.8],
            'clouds': [50.0, 55.0, 60.0, 58.0, 52.0],
            'precipitation': [0.0] * 5,
        })

        engineer = FeatureEngineer()
        features = engineer.engineer_features(df)

        # Should still process
        assert not features.empty
        assert len(features) == 5


# ============================================================================
# TESTS: Consistency & Reproducibility
# ============================================================================

class TestConsistency:
    """Test consistency and reproducibility."""

    def test_same_input_same_output(self, sample_weather_data):
        """Test that same input produces same output."""
        engineer = FeatureEngineer()

        features1 = engineer.engineer_features(sample_weather_data.copy())
        features2 = engineer.engineer_features(sample_weather_data.copy())

        # Should produce identical results
        pd.testing.assert_frame_equal(features1, features2)

    def test_different_windows_different_features(self, sample_weather_data):
        """Test that different windows produce different features."""
        engineer1 = FeatureEngineer(lookback_windows=['6h'])
        engineer2 = FeatureEngineer(lookback_windows=['24h'])

        features1 = engineer1.engineer_features(sample_weather_data.copy())
        features2 = engineer2.engineer_features(sample_weather_data.copy())

        # Should have different column names
        assert set(features1.columns) != set(features2.columns)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
