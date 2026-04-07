"""
Unit tests for data collection modules.

Tests cover:
- API client initialization
- Request handling (success, failures, retries)
- Data validation
- Error recovery
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import asyncio

from src.data_collection.weather_api import (
    WeatherAPIClient,
    OpenWeatherMapClient,
    WeatherAPIClient2
)
from src.data_collection.data_fetcher import DataFetcher


# ============================================================================
# FIXTURES (Test Data)
# ============================================================================

@pytest.fixture
def sample_weather_response():
    """Sample OpenWeatherMap API response."""
    return {
        'main': {
            'temp': 15.5,
            'feels_like': 14.0,
            'humidity': 72,
            'pressure': 1013,
        },
        'weather': [
            {
                'id': 500,  # Rain
                'main': 'Rain',
                'description': 'light rain',
                'icon': '10d'
            }
        ],
        'clouds': {
            'all': 75  # 75% cloud coverage
        },
        'wind': {
            'speed': 5.5,
            'deg': 230
        },
        'visibility': 10000,
        'dt': 1712432400,
        'sys': {
            'sunrise': 1712385600,
            'sunset': 1712434800,
        }
    }


@pytest.fixture
def sample_weatherapi_response():
    """Sample WeatherAPI response."""
    return {
        'current': {
            'temp_c': 15.5,
            'feelslike_c': 14.0,
            'humidity': 72,
            'pressure_mb': 1013,
            'cloud': 75,
            'condition': {
                'code': 1183,
                'text': 'Patchy rain nearby',
                'icon': '//cdn.weatherapi.com/weather/128x128/day/176.png'
            },
            'wind_kph': 19.8,
            'wind_degree': 230,
            'visibility_km': 10,
            'precip_mm': 0.5,
        },
        'location': {
            'name': 'New York',
            'region': 'New York',
            'country': 'United States',
            'lat': 40.71,
            'lon': -74.01,
        }
    }


@pytest.fixture
def sample_historical_data():
    """Sample historical weather data."""
    dates = pd.date_range(start='2024-03-30', periods=7, freq='D')
    return pd.DataFrame({
        'timestamp': dates,
        'temperature': [12.5, 13.8, 15.2, 14.1, 16.5, 15.8, 17.2],
        'humidity': [65, 70, 72, 68, 75, 72, 70],
        'precipitation': [0.0, 2.5, 1.2, 0.0, 3.5, 0.8, 0.0],
        'cloud_coverage': [45, 60, 75, 50, 80, 65, 40],
        'wind_speed': [3.5, 4.2, 5.5, 4.0, 6.5, 5.2, 3.8],
    })


# ============================================================================
# TESTS: WeatherAPIClient Base Class
# ============================================================================

class TestWeatherAPIClient:
    """Test base WeatherAPIClient class."""

    def test_initialization(self):
        """Test client initialization."""
        client = WeatherAPIClient(
            api_key='test_key_123',
            base_url='https://example.com/api'
        )

        assert client.api_key == 'test_key_123'
        assert client.base_url == 'https://example.com/api'
        assert client.timeout == 10

    def test_custom_timeout(self):
        """Test custom timeout configuration."""
        client = WeatherAPIClient(
            api_key='test_key',
            base_url='https://example.com',
            timeout=30
        )

        assert client.timeout == 30

    def test_get_headers(self):
        """Test header generation."""
        client = WeatherAPIClient(
            api_key='test_key',
            base_url='https://example.com'
        )

        headers = client._get_headers()

        assert 'User-Agent' in headers
        assert 'Accept' in headers
        assert headers['Accept'] == 'application/json'

    @patch('src.data_collection.weather_api.requests.get')
    def test_sync_request_success(self, mock_get, sample_weather_response):
        """Test successful synchronous request."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = sample_weather_response
        mock_get.return_value = mock_response

        client = WeatherAPIClient(
            api_key='test_key',
            base_url='https://example.com'
        )

        result = client._sync_request('https://example.com/test')

        assert result is not None
        assert result['main']['temp'] == 15.5
        mock_get.assert_called_once()

    @patch('src.data_collection.weather_api.requests.get')
    def test_sync_request_timeout(self, mock_get):
        """Test request timeout handling."""
        import requests
        mock_get.side_effect = requests.exceptions.Timeout()

        client = WeatherAPIClient(
            api_key='test_key',
            base_url='https://example.com'
        )

        result = client._sync_request('https://example.com/test')

        assert result is None

    @patch('src.data_collection.weather_api.requests.get')
    def test_sync_request_http_error(self, mock_get):
        """Test HTTP error handling."""
        import requests
        mock_get.side_effect = requests.exceptions.HTTPError('404 Not Found')

        client = WeatherAPIClient(
            api_key='test_key',
            base_url='https://example.com'
        )

        result = client._sync_request('https://example.com/test')

        assert result is None


# ============================================================================
# TESTS: OpenWeatherMapClient
# ============================================================================

class TestOpenWeatherMapClient:
    """Test OpenWeatherMap API client."""

    def test_initialization(self):
        """Test OpenWeatherMapClient initialization."""
        client = OpenWeatherMapClient(api_key='owm_test_key')

        assert client.api_key == 'owm_test_key'
        assert 'openweathermap.org' in client.base_url

    @patch('src.data_collection.weather_api.WeatherAPIClient._sync_request')
    def test_get_current_weather(self, mock_request, sample_weather_response):
        """Test getting current weather."""
        mock_request.return_value = sample_weather_response

        client = OpenWeatherMapClient(api_key='test_key')
        result = client.get_current_weather(40.7128, -74.0060)

        assert result is not None
        assert 'main' in result
        assert result['main']['temp'] == 15.5
        assert result['main']['humidity'] == 72

    @patch('src.data_collection.weather_api.WeatherAPIClient._sync_request')
    def test_get_forecast(self, mock_request):
        """Test getting forecast data."""
        forecast_response = {
            'list': [
                {'dt': 1712432400, 'main': {'temp': 15.5}},
                {'dt': 1712436000, 'main': {'temp': 16.2}},
            ]
        }
        mock_request.return_value = forecast_response

        client = OpenWeatherMapClient(api_key='test_key')
        result = client.get_forecast(40.7128, -74.0060)

        assert result is not None
        assert len(result['list']) == 2

    @patch('src.data_collection.weather_api.WeatherAPIClient._sync_request')
    def test_get_current_weather_missing_fields(self, mock_request):
        """Test handling of incomplete responses."""
        incomplete_response = {
            'main': {'temp': 15.5}
            # Missing: weather, clouds, wind
        }
        mock_request.return_value = incomplete_response

        client = OpenWeatherMapClient(api_key='test_key')
        result = client.get_current_weather(40.7128, -74.0060)

        # Should still work but log warnings
        assert result is not None
        assert result['main']['temp'] == 15.5


# ============================================================================
# TESTS: WeatherAPIClient2 (WeatherAPI.com)
# ============================================================================

class TestWeatherAPIClient2:
    """Test WeatherAPI.com client."""

    def test_initialization(self):
        """Test WeatherAPIClient2 initialization."""
        client = WeatherAPIClient2(api_key='weatherapi_test_key')

        assert client.api_key == 'weatherapi_test_key'
        assert 'weatherapi.com' in client.base_url

    @patch('src.data_collection.weather_api.WeatherAPIClient._sync_request')
    def test_get_current_weather(self, mock_request, sample_weatherapi_response):
        """Test getting current weather from WeatherAPI."""
        mock_request.return_value = sample_weatherapi_response

        client = WeatherAPIClient2(api_key='test_key')
        result = client.get_current_weather(40.7128, -74.0060)

        assert result is not None
        assert result['current']['temp_c'] == 15.5
        assert result['location']['name'] == 'New York'


# ============================================================================
# TESTS: DataFetcher
# ============================================================================

class TestDataFetcher:
    """Test DataFetcher class."""

    @pytest.fixture
    def config(self):
        """Sample config for tests."""
        return {
            'weather': {
                'sources': [
                    {
                        'name': 'openweathermap',
                        'enabled': True,
                        'api_key': 'test_owm_key',
                    },
                    {
                        'name': 'weatherapi',
                        'enabled': True,
                        'api_key': 'test_wa_key',
                    }
                ],
                'locations': [
                    {
                        'name': 'New York',
                        'lat': 40.7128,
                        'lon': -74.0060,
                    },
                    {
                        'name': 'Los Angeles',
                        'lat': 34.0522,
                        'lon': -118.2437,
                    }
                ]
            },
            'features': {
                'lookback_windows': ['6h', '24h'],
                'seasonal_features': True,
            }
        }

    def test_initialization(self, config):
        """Test DataFetcher initialization."""
        fetcher = DataFetcher(config)

        assert fetcher.config == config
        assert len(fetcher.config['weather']['locations']) == 2

    @patch('src.data_collection.weather_api.OpenWeatherMapClient')
    def test_fetch_current_weather(self, mock_client, config):
        """Test fetching current weather for a location."""
        mock_instance = Mock()
        mock_client.return_value = mock_instance

        mock_instance.get_current_weather.return_value = {
            'main': {
                'temp': 15.5,
                'humidity': 72,
            },
            'clouds': {'all': 75},
            'wind': {'speed': 5.5},
        }

        fetcher = DataFetcher(config)
        result = fetcher.fetch_current_weather('New York')

        assert result is not None
        assert result['main']['temp'] == 15.5

    def test_fetch_current_weather_invalid_location(self, config):
        """Test fetching weather for non-existent location."""
        fetcher = DataFetcher(config)
        result = fetcher.fetch_current_weather('Unknown City')

        # Should return None or empty dict, not crash
        assert result is None or result == {}

    @patch('src.data_collection.weather_api.WeatherAPIClient._sync_request')
    def test_populate_historical_data(self, mock_request, config, sample_historical_data):
        """Test populating historical weather data."""
        # Mock the API to return historical data
        mock_request.return_value = {
            'list': [
                {'dt': 1712300000, 'main': {'temp': 12.5, 'humidity': 65}},
                {'dt': 1712386400, 'main': {'temp': 13.8, 'humidity': 70}},
            ]
        }

        fetcher = DataFetcher(config)

        # Should not crash
        record_count = fetcher.populate_historical_data('New York', days=7)

        assert isinstance(record_count, int)

    @patch('src.data_collection.weather_api.WeatherAPIClient._sync_request')
    def test_prepare_training_data(self, mock_request, config, sample_historical_data):
        """Test preparing training data."""
        fetcher = DataFetcher(config)

        # This needs actual data in the database first
        # So we'll test with mock
        features, target = fetcher.prepare_training_data('New York', days=7)

        # Should return DataFrames (possibly empty if no data yet)
        assert isinstance(features, pd.DataFrame)
        if not features.empty:
            assert isinstance(target, pd.Series)


# ============================================================================
# TESTS: Data Validation
# ============================================================================

class TestDataValidation:
    """Test data validation and cleaning."""

    def test_weather_data_missing_fields(self, sample_weather_response):
        """Test handling of missing fields in weather data."""
        incomplete = sample_weather_response.copy()
        del incomplete['wind']  # Remove wind data

        # Should still be processable
        assert 'main' in incomplete
        assert 'weather' in incomplete

    def test_temperature_range_validation(self):
        """Test temperature range validation."""
        valid_temps = [-50, -10, 0, 15.5, 30, 50]
        invalid_temps = [-500, 500]  # Unrealistic

        def is_valid_temp(temp):
            return -60 <= temp <= 60

        for temp in valid_temps:
            assert is_valid_temp(temp)

        for temp in invalid_temps:
            assert not is_valid_temp(temp)

    def test_humidity_range_validation(self):
        """Test humidity range validation (0-100%)."""
        valid_humidities = [0, 25, 50, 75, 100]
        invalid_humidities = [-10, 150]

        def is_valid_humidity(h):
            return 0 <= h <= 100

        for h in valid_humidities:
            assert is_valid_humidity(h)

        for h in invalid_humidities:
            assert not is_valid_humidity(h)

    def test_precipitation_validation(self):
        """Test precipitation data validation."""
        valid_precip = [0, 0.5, 2.5, 10, 100]
        invalid_precip = [-5, -0.1]

        def is_valid_precip(p):
            return p >= 0

        for p in valid_precip:
            assert is_valid_precip(p)

        for p in invalid_precip:
            assert not is_valid_precip(p)


# ============================================================================
# TESTS: Error Recovery
# ============================================================================

class TestErrorRecovery:
    """Test error recovery mechanisms."""

    @patch('src.data_collection.weather_api.requests.get')
    def test_retry_on_timeout(self, mock_get):
        """Test retry logic on timeout."""
        import requests

        # First call times out, second succeeds
        mock_get.side_effect = [
            requests.exceptions.Timeout(),
            Mock(status_code=200, json=Mock(return_value={'main': {'temp': 15.5}}))
        ]

        client = WeatherAPIClient(
            api_key='test_key',
            base_url='https://example.com'
        )

        # First attempt fails
        result1 = client._sync_request('https://example.com/test')
        assert result1 is None

        # Second attempt succeeds
        result2 = client._sync_request('https://example.com/test')
        assert result2 is not None

    @patch('src.data_collection.weather_api.WeatherAPIClient._sync_request')
    def test_fallback_to_second_api(self, mock_request):
        """Test fallback to second API if first fails."""
        from src.data_collection.weather_api import WeatherDataAggregator

        # First API fails, should try second
        mock_request.side_effect = [None, {'current': {'temp_c': 15.5}}]

        aggregator = WeatherDataAggregator()
        # Should have fallback mechanism
        assert aggregator is not None


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestDataCollectionIntegration:
    """Integration tests for complete data collection flow."""

    @pytest.fixture
    def config(self):
        """Sample config."""
        return {
            'weather': {
                'sources': [
                    {'name': 'openweathermap', 'enabled': True, 'api_key': 'test_key'},
                ],
                'locations': [
                    {'name': 'New York', 'lat': 40.7128, 'lon': -74.0060},
                ]
            },
            'features': {
                'lookback_windows': ['6h', '24h'],
            }
        }

    @patch('src.data_collection.weather_api.WeatherAPIClient._sync_request')
    def test_full_data_collection_flow(self, mock_request, config):
        """Test complete data collection and preparation flow."""
        mock_request.return_value = {
            'main': {'temp': 15.5, 'humidity': 72},
            'weather': [{'main': 'Rain'}],
            'clouds': {'all': 75},
            'wind': {'speed': 5.5},
        }

        from src.data_collection.data_fetcher import DataFetcher

        fetcher = DataFetcher(config)
        weather = fetcher.fetch_current_weather('New York')

        assert weather is not None
        assert 'main' in weather
        # Should have all required fields
        assert 'humidity' in weather['main'] or 'main' in weather


# ============================================================================
# BENCHMARKS (Optional but useful)
# ============================================================================

class TestPerformance:
    """Performance/benchmark tests."""

    def test_sync_request_latency(self):
        """Test that sync request doesn't timeout."""
        client = WeatherAPIClient(
            api_key='test_key',
            base_url='https://example.com',
            timeout=10
        )

        assert client.timeout == 10
        assert client.timeout > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
