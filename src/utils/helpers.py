"""Helper utilities for the Polymarket Weather Agent."""

import yaml
import json
import os
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime, timedelta
import pytz


def load_config(config_path: str = 'config/config.yaml') -> Dict[str, Any]:
    """Load configuration from YAML file with env variable substitution."""
    with open(config_path, 'r') as f:
        config_str = f.read()

    # Replace environment variables
    for key, value in os.environ.items():
        config_str = config_str.replace(f'${{{key}}}', value)

    return yaml.safe_load(config_str)


def load_markets(markets_path: str = 'config/markets.json') -> Dict[str, Any]:
    """Load market mapping from JSON file."""
    with open(markets_path, 'r') as f:
        return json.load(f)


def save_json(data: Dict[str, Any], filepath: str) -> None:
    """Save data as JSON file."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def load_json(filepath: str) -> Dict[str, Any]:
    """Load data from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def to_timestamp(dt: datetime) -> float:
    """Convert datetime to Unix timestamp."""
    return dt.timestamp()


def from_timestamp(ts: float, tz: str = 'UTC') -> datetime:
    """Convert Unix timestamp to datetime."""
    tz_obj = pytz.timezone(tz)
    return datetime.fromtimestamp(ts, tz=pytz.UTC).astimezone(tz_obj)


def get_location_tz(location: str, locations_config: List[Dict]) -> str:
    """Get timezone for a location."""
    for loc in locations_config:
        if loc['name'].lower() == location.lower():
            return loc.get('timezone', 'UTC')
    return 'UTC'


def format_probability(prob: float) -> str:
    """Format probability as percentage string."""
    return f"{prob * 100:.2f}%"


def format_price(price: float, decimals: int = 4) -> str:
    """Format price with specified decimals."""
    return f"{price:.{decimals}f}"


def calculate_kelly_size(win_prob: float, loss_ratio: float) -> float:
    """
    Calculate Kelly Criterion position size.

    Args:
        win_prob: Probability of winning
        loss_ratio: Ratio of loss to win (e.g., 1.0 for equal risk/reward)

    Returns:
        Fraction of capital to risk (0-1)
    """
    if win_prob <= 0 or win_prob >= 1:
        return 0.0

    b = 1.0 / loss_ratio  # Odds
    p = win_prob
    q = 1 - win_prob

    kelly = (b * p - q) / b
    return max(0.0, min(kelly, 1.0))  # Clamp to [0, 1]


def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.0) -> float:
    """
    Calculate Sharpe ratio from returns.

    Args:
        returns: List of period returns
        risk_free_rate: Annual risk-free rate

    Returns:
        Sharpe ratio (annualized)
    """
    import numpy as np

    returns = np.array(returns)
    if len(returns) == 0:
        return 0.0

    excess_returns = returns - (risk_free_rate / 252)
    if excess_returns.std() == 0:
        return 0.0

    return (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)


def calculate_drawdown(equity_curve: List[float]) -> tuple:
    """
    Calculate maximum drawdown and recovery period.

    Args:
        equity_curve: List of portfolio values over time

    Returns:
        Tuple of (max_drawdown, recovery_period)
    """
    import numpy as np

    equity = np.array(equity_curve)
    running_max = np.maximum.accumulate(equity)
    drawdown = (equity - running_max) / running_max

    max_drawdown = np.min(drawdown)
    max_idx = np.argmin(drawdown)

    # Find recovery point
    recovery_idx = np.where(equity[max_idx:] >= running_max[max_idx])[0]
    recovery_period = recovery_idx[0] if len(recovery_idx) > 0 else len(equity) - max_idx

    return max_drawdown, recovery_period


def calculate_win_rate(trades: List[Dict]) -> float:
    """Calculate win rate from trade history."""
    if not trades:
        return 0.0

    wins = sum(1 for t in trades if t.get('pnl', 0) > 0)
    return wins / len(trades)


def calculate_profit_factor(trades: List[Dict]) -> float:
    """Calculate profit factor (gross profit / gross loss)."""
    gross_profit = sum(t['pnl'] for t in trades if t.get('pnl', 0) > 0)
    gross_loss = abs(sum(t['pnl'] for t in trades if t.get('pnl', 0) < 0))

    if gross_loss == 0:
        return float('inf') if gross_profit > 0 else 0.0

    return gross_profit / gross_loss


def is_market_open(location_tz: str) -> bool:
    """Check if market is currently open (simplified - assumes 24/7)."""
    # This is a placeholder - Polymarket is typically 24/7
    # Can be extended for specific market hours
    return True


def get_time_until_resolution(resolution_date: datetime) -> timedelta:
    """Get time remaining until market resolution."""
    return resolution_date - datetime.utcnow()


def validate_api_key(key: str, min_length: int = 20) -> bool:
    """Validate API key format."""
    return isinstance(key, str) and len(key) >= min_length


def truncate_string(s: str, length: int = 50) -> str:
    """Truncate string for logging."""
    return s if len(s) <= length else s[:length] + "..."
