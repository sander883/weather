"""Configuration management with validation."""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional
from pydantic import ValidationError

from src.models.schemas import AgentConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ConfigManager:
    """Load and validate application configuration."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize config manager.

        Args:
            config_path: Path to config file (YAML or JSON)
        """
        self.config_path = config_path or self._find_default_config()
        self.config: Optional[AgentConfig] = None
        self.raw_config: Optional[Dict[str, Any]] = None

    def _find_default_config(self) -> str:
        """Find default config file."""
        candidates = [
            Path('config/config.yaml'),
            Path('config.yaml'),
            Path('.config/config.yaml'),
        ]

        for path in candidates:
            if path.exists():
                return str(path)

        logger.warning("No config file found, using defaults")
        return None

    def load(self) -> AgentConfig:
        """
        Load and validate configuration.

        Returns:
            Validated AgentConfig instance

        Raises:
            ValidationError: If config validation fails
            FileNotFoundError: If config file not found
        """
        if not self.config_path:
            logger.warning("No config path, using default values")
            self.config = AgentConfig()
            return self.config

        config_path = Path(self.config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        try:
            # Load raw config
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                self.raw_config = self._load_yaml(config_path)
            elif config_path.suffix.lower() == '.json':
                self.raw_config = self._load_json(config_path)
            else:
                raise ValueError(f"Unsupported config format: {config_path.suffix}")

            # Validate with Pydantic
            self.config = AgentConfig.from_dict(self.raw_config)
            logger.info(f"Configuration loaded and validated from {self.config_path}")
            return self.config

        except ValidationError as e:
            logger.error(f"Configuration validation failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise

    def _load_yaml(self, path: Path) -> Dict[str, Any]:
        """Load YAML configuration file."""
        with open(path, 'r') as f:
            return yaml.safe_load(f) or {}

    def _load_json(self, path: Path) -> Dict[str, Any]:
        """Load JSON configuration file."""
        with open(path, 'r') as f:
            return json.load(f)

    def get_config(self) -> AgentConfig:
        """Get loaded configuration (load if not already loaded)."""
        if self.config is None:
            self.load()
        return self.config

    def get_raw_config(self) -> Dict[str, Any]:
        """Get raw configuration dictionary."""
        if self.raw_config is None:
            self.load()
        return self.raw_config or {}

    def validate_config(self, config_dict: Dict[str, Any]) -> tuple:
        """
        Validate a configuration dictionary.

        Args:
            config_dict: Configuration to validate

        Returns:
            Tuple of (is_valid, errors)
        """
        try:
            AgentConfig.from_dict(config_dict)
            return True, []
        except ValidationError as e:
            errors = e.errors()
            return False, errors


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from file.

    Args:
        config_path: Path to config file

    Returns:
        Configuration dictionary

    Note:
        This maintains backward compatibility with existing code that
        expects a dict instead of AgentConfig
    """
    manager = ConfigManager(config_path)
    config = manager.load()
    return config.model_dump(exclude_none=False)


def get_validated_config(config_path: Optional[str] = None) -> AgentConfig:
    """
    Load and get validated configuration.

    Args:
        config_path: Path to config file

    Returns:
        Validated AgentConfig instance
    """
    manager = ConfigManager(config_path)
    return manager.load()
