"""Unit tests for configuration validation."""

import pytest
import json
import tempfile
from pathlib import Path
from pydantic import ValidationError

from src.models.schemas import (
    AgentConfig, WeatherConfig, FeaturesConfig, ModelConfig,
    TradingStrategyConfig, RiskManagementConfig, LoggingConfig,
    DashboardConfig
)
from src.utils.config import ConfigManager, load_config, get_validated_config


@pytest.fixture
def sample_config_dict():
    """Create sample configuration dictionary."""
    return {
        'weather': {
            'sources': [
                {
                    'name': 'openweathermap',
                    'enabled': True,
                    'api_key': 'test_key',
                    'endpoints': ['weather', 'forecast']
                }
            ],
            'locations': [
                {
                    'name': 'Test City',
                    'lat': 40.0,
                    'lon': -74.0,
                    'country': 'US'
                }
            ],
            'refresh_interval': 3600,
            'historical_lookback': 30
        },
        'features': {
            'lookback_windows': ['6h', '24h'],
            'seasonal_features': True
        },
        'model': {
            'algorithm': 'xgboost',
            'xgboost_params': {
                'max_depth': 6,
                'n_estimators': 100
            }
        },
        'trading': {
            'edge_threshold': 0.05,
            'max_concurrent_positions': 5
        },
        'risk_management': {
            'portfolio': {
                'initial_capital': 10000
            }
        }
    }


@pytest.fixture
def temp_config_file(sample_config_dict):
    """Create temporary config file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(sample_config_dict, f)
        return f.name


class TestAgentConfig:
    """Test AgentConfig model."""

    def test_config_with_defaults(self):
        """Test creating config with defaults."""
        config = AgentConfig()
        assert config.weather is not None
        assert config.model is not None
        assert config.trading is not None
        assert config.risk_management is not None

    def test_config_from_dict(self, sample_config_dict):
        """Test creating config from dict."""
        config = AgentConfig.from_dict(sample_config_dict)
        assert config.weather.refresh_interval == 3600
        assert config.trading.edge_threshold == 0.05

    def test_config_to_dict(self, sample_config_dict):
        """Test converting config to dict."""
        config = AgentConfig.from_dict(sample_config_dict)
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert 'weather' in config_dict
        assert 'trading' in config_dict

    def test_config_validation_fails_invalid_capital(self):
        """Test that invalid capital is rejected."""
        invalid_config = {
            'risk_management': {
                'portfolio': {
                    'initial_capital': -1000  # Negative capital
                }
            }
        }

        with pytest.raises(ValidationError):
            AgentConfig.from_dict(invalid_config)

    def test_config_validation_fails_invalid_lat(self):
        """Test that invalid latitude is rejected."""
        invalid_config = {
            'weather': {
                'locations': [
                    {
                        'name': 'Test',
                        'lat': 100,  # Invalid latitude
                        'lon': 0
                    }
                ]
            }
        }

        with pytest.raises(ValidationError):
            AgentConfig.from_dict(invalid_config)

    def test_config_validation_fails_invalid_lon(self):
        """Test that invalid longitude is rejected."""
        invalid_config = {
            'weather': {
                'locations': [
                    {
                        'name': 'Test',
                        'lat': 0,
                        'lon': 200  # Invalid longitude
                    }
                ]
            }
        }

        with pytest.raises(ValidationError):
            AgentConfig.from_dict(invalid_config)


class TestWeatherConfig:
    """Test WeatherConfig model."""

    def test_weather_config_defaults(self):
        """Test weather config with defaults."""
        config = WeatherConfig()
        assert config.refresh_interval == 3600
        assert config.historical_lookback == 30

    def test_weather_config_invalid_interval(self):
        """Test that invalid refresh interval is rejected."""
        with pytest.raises(ValidationError):
            WeatherConfig(refresh_interval=-1)

    def test_weather_config_invalid_lookback(self):
        """Test that invalid lookback is rejected."""
        with pytest.raises(ValidationError):
            WeatherConfig(historical_lookback=0)


class TestModelConfig:
    """Test ModelConfig model."""

    def test_model_config_defaults(self):
        """Test model config with defaults."""
        config = ModelConfig()
        assert config.algorithm == 'xgboost'
        assert config.train_test_split == 0.8
        assert config.cross_validation_folds == 5

    def test_model_config_invalid_split(self):
        """Test that invalid train/test split is rejected."""
        with pytest.raises(ValidationError):
            ModelConfig(train_test_split=1.5)

    def test_model_config_invalid_cv_folds(self):
        """Test that invalid CV folds is rejected."""
        with pytest.raises(ValidationError):
            ModelConfig(cross_validation_folds=1)

    def test_xgboost_params_validation(self):
        """Test XGBoost params validation."""
        from src.models.schemas import XGBoostParams

        # Valid params
        params = XGBoostParams(max_depth=5, learning_rate=0.1)
        assert params.max_depth == 5

        # Invalid depth
        with pytest.raises(ValidationError):
            XGBoostParams(max_depth=0)


class TestTradingConfig:
    """Test TradingStrategyConfig model."""

    def test_trading_config_defaults(self):
        """Test trading config with defaults."""
        config = TradingStrategyConfig()
        assert config.strategy == 'probability_edge'
        assert config.edge_threshold == 0.05
        assert config.max_concurrent_positions == 5

    def test_trading_config_invalid_edge_threshold(self):
        """Test that invalid edge threshold is rejected."""
        with pytest.raises(ValidationError):
            TradingStrategyConfig(edge_threshold=-0.05)

    def test_trading_config_invalid_max_positions(self):
        """Test that invalid max positions is rejected."""
        with pytest.raises(ValidationError):
            TradingStrategyConfig(max_concurrent_positions=0)

    def test_position_sizing_method_validation(self):
        """Test position sizing method validation."""
        from src.models.schemas import PositionSizingConfig

        # Valid method
        config = PositionSizingConfig(method='kelly')
        assert config.method == 'kelly'

        # Invalid method
        with pytest.raises(ValidationError):
            PositionSizingConfig(method='invalid')


class TestRiskManagementConfig:
    """Test RiskManagementConfig model."""

    def test_risk_management_config_defaults(self):
        """Test risk management config with defaults."""
        config = RiskManagementConfig()
        assert config.portfolio is not None
        assert config.position is not None
        assert config.circuit_breaker is not None

    def test_portfolio_config_validation(self):
        """Test portfolio config validation."""
        from src.models.schemas import PortfolioConfig

        with pytest.raises(ValidationError):
            PortfolioConfig(max_daily_loss_percent=-1)

    def test_position_config_validation(self):
        """Test position config validation."""
        from src.models.schemas import PositionConfig

        with pytest.raises(ValidationError):
            PositionConfig(max_concentration=1.5)


class TestLoggingConfig:
    """Test LoggingConfig model."""

    def test_logging_config_defaults(self):
        """Test logging config with defaults."""
        config = LoggingConfig()
        assert config.level == 'INFO'
        assert config.console_output is True

    def test_logging_config_valid_levels(self):
        """Test valid logging levels."""
        for level in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
            config = LoggingConfig(level=level)
            assert config.level == level

    def test_logging_config_invalid_level(self):
        """Test that invalid logging level is rejected."""
        with pytest.raises(ValidationError):
            LoggingConfig(level='INVALID')


class TestDashboardConfig:
    """Test DashboardConfig model."""

    def test_dashboard_config_defaults(self):
        """Test dashboard config with defaults."""
        config = DashboardConfig()
        assert config.port == 5000
        assert config.enabled is True

    def test_dashboard_config_invalid_port(self):
        """Test that invalid port is rejected."""
        with pytest.raises(ValidationError):
            DashboardConfig(port=0)

        with pytest.raises(ValidationError):
            DashboardConfig(port=70000)


class TestConfigManager:
    """Test ConfigManager class."""

    def test_config_manager_load_json(self, temp_config_file):
        """Test loading JSON config file."""
        manager = ConfigManager(temp_config_file)
        config = manager.load()

        assert isinstance(config, AgentConfig)
        assert config.weather.refresh_interval == 3600

    def test_config_manager_get_config(self, temp_config_file):
        """Test getting loaded config."""
        manager = ConfigManager(temp_config_file)
        config1 = manager.get_config()
        config2 = manager.get_config()

        # Should return same instance
        assert config1 is config2

    def test_config_manager_get_raw_config(self, temp_config_file):
        """Test getting raw config dict."""
        manager = ConfigManager(temp_config_file)
        raw_config = manager.get_raw_config()

        assert isinstance(raw_config, dict)
        assert 'weather' in raw_config

    def test_config_manager_validate_config(self, sample_config_dict):
        """Test config validation method."""
        manager = ConfigManager()
        is_valid, errors = manager.validate_config(sample_config_dict)

        assert is_valid is True
        assert errors == []

    def test_config_manager_validate_invalid_config(self):
        """Test validation of invalid config."""
        invalid_config = {
            'risk_management': {
                'portfolio': {
                    'initial_capital': -1000
                }
            }
        }

        manager = ConfigManager()
        is_valid, errors = manager.validate_config(invalid_config)

        assert is_valid is False
        assert len(errors) > 0

    def test_config_manager_file_not_found(self):
        """Test handling of missing config file."""
        manager = ConfigManager('/nonexistent/path/config.yaml')

        with pytest.raises(FileNotFoundError):
            manager.load()

    def test_config_manager_unsupported_format(self):
        """Test handling of unsupported file format."""
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            f.write(b'some content')
            temp_file = f.name

        manager = ConfigManager(temp_file)

        with pytest.raises(ValueError):
            manager.load()


class TestConfigFunctions:
    """Test module-level config functions."""

    def test_load_config(self, temp_config_file):
        """Test load_config function."""
        config_dict = load_config(temp_config_file)

        assert isinstance(config_dict, dict)
        assert 'weather' in config_dict
        assert 'trading' in config_dict

    def test_get_validated_config(self, temp_config_file):
        """Test get_validated_config function."""
        config = get_validated_config(temp_config_file)

        assert isinstance(config, AgentConfig)
        assert config.weather.refresh_interval == 3600


class TestConfigEdgeCases:
    """Test edge cases in config validation."""

    def test_empty_config(self):
        """Test that empty config uses all defaults."""
        config = AgentConfig.from_dict({})

        assert config.weather.refresh_interval == 3600
        assert config.model.algorithm == 'xgboost'
        assert config.trading.edge_threshold == 0.05

    def test_partial_config(self):
        """Test that partial config merges with defaults."""
        partial_config = {
            'trading': {
                'edge_threshold': 0.10
            }
        }

        config = AgentConfig.from_dict(partial_config)

        assert config.trading.edge_threshold == 0.10
        assert config.trading.max_concurrent_positions == 5  # Default

    def test_deeply_nested_config(self, sample_config_dict):
        """Test deeply nested config structures."""
        sample_config_dict['model']['xgboost_params']['max_depth'] = 10
        config = AgentConfig.from_dict(sample_config_dict)

        assert config.model.xgboost_params.max_depth == 10

    def test_config_with_extra_fields(self, sample_config_dict):
        """Test that extra fields are ignored."""
        sample_config_dict['unknown_field'] = 'value'
        config = AgentConfig.from_dict(sample_config_dict)

        assert isinstance(config, AgentConfig)
        # Extra field should be ignored
        assert not hasattr(config, 'unknown_field')
