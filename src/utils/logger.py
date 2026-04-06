"""Logging utilities for the Polymarket Weather Agent."""

import logging
import logging.handlers
import os
from pathlib import Path


def setup_logger(name, log_file=None, level=logging.INFO):
    """
    Configure and return a logger instance.

    Args:
        name: Logger name
        log_file: Path to log file (optional)
        level: Logging level

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)

        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10485760,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name):
    """Get or create a logger."""
    log_file = os.getenv('LOG_FILE', 'logs/agent.log')
    log_level = getattr(logging, os.getenv('LOG_LEVEL', 'INFO'))
    return setup_logger(name, log_file=log_file, level=log_level)
