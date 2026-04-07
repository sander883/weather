"""Risk management module."""

from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta

from src.utils.logger import get_logger
from src.models.schemas import Trade, Position, PortfolioStats

logger = get_logger(__name__)


class RiskManager:
    """Manage trading risks and portfolio limits."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.risk_config = config.get('risk_management', {})
        self.portfolio_config = self.risk_config.get('portfolio', {})
        self.position_config = self.risk_config.get('position', {})
        self.circuit_breaker_config = self.risk_config.get('circuit_breaker', {})

        self.initial_capital = self.portfolio_config.get('initial_capital', 10000)
        self.current_capital = self.initial_capital
        self.daily_start_capital = self.initial_capital

        self.trade_log = []
        self.circuit_breaker_triggered = False

    def check_trade_feasibility(self, trade: Union[Dict[str, Any], Trade],
                               open_positions: List[Union[Dict, Position]]) -> Tuple[bool, str]:
        """
        Check if a trade can be executed given constraints.

        Args:
            trade: Trade dict or Trade model to evaluate
            open_positions: Current open positions (dicts or Position models)

        Returns:
            Tuple of (is_feasible, reason)
        """
        # Convert Pydantic model to dict if needed
        trade_dict = trade.model_dump() if isinstance(trade, Trade) else trade

        # Defensive: ensure trade is a dict
        if not isinstance(trade_dict, dict):
            return False, f"Invalid trade type: {type(trade_dict)}"

        # Defensive: ensure open_positions is a list
        if not isinstance(open_positions, list):
            logger.warning(f"open_positions is {type(open_positions)}, converting to list")
            open_positions = list(open_positions) if open_positions else []

        # Check circuit breaker
        if self.circuit_breaker_triggered:
            return False, "Circuit breaker triggered"

        # Check daily loss limit
        daily_loss = self._get_daily_loss()
        max_daily_loss = self.portfolio_config.get('initial_capital', 10000) * \
                        self.portfolio_config.get('max_daily_loss_percent', 2.0) / 100

        if daily_loss >= max_daily_loss:
            return False, f"Daily loss limit reached ({daily_loss:.2f} / {max_daily_loss:.2f})"

        # Check max concurrent positions
        max_positions = self.position_config.get('max_concurrent_positions', 5)
        if len(open_positions) >= max_positions:
            return False, f"Max concurrent positions reached ({len(open_positions)} / {max_positions})"

        # Check concentration limit
        position_size = trade_dict.get('position_size', 0)

        # Convert positions to dicts if they're Pydantic models
        positions_list = [
            p.model_dump() if isinstance(p, Position) else p
            for p in open_positions
        ]

        total_exposure = sum(p.get('position_size', 0) for p in positions_list)
        max_concentration = self.position_config.get('max_concentration', 0.30)
        max_exposure = self.initial_capital * max_concentration

        if total_exposure + position_size > max_exposure:
            return False, f"Concentration limit exceeded"

        # Check minimum position size
        min_size = self.position_config.get('min_position_size', 10)
        if position_size < min_size:
            return False, f"Position size {position_size} below minimum {min_size}"

        # Check available capital
        if position_size > self.current_capital * 0.5:  # Max 50% of available capital
            return False, f"Insufficient capital"

        return True, "OK"

    def record_trade(self, trade: Dict[str, Any]) -> None:
        """Record a trade in the log."""
        self.trade_log.append({
            'timestamp': datetime.utcnow(),
            'trade': trade
        })

    def record_close(self, close_result: Dict[str, Any]) -> None:
        """Record a position close."""
        pnl = close_result.get('pnl', 0)
        self.current_capital += pnl
        self.trade_log.append({
            'timestamp': datetime.utcnow(),
            'type': 'CLOSE',
            'pnl': pnl
        })

        # Check circuit breaker conditions
        self._check_circuit_breaker()

    def _get_daily_loss(self) -> float:
        """Calculate current day's losses."""
        today = datetime.utcnow().date()
        daily_trades = [
            t for t in self.trade_log
            if t['timestamp'].date() == today and t.get('type') == 'CLOSE'
        ]

        daily_loss = sum(t.get('pnl', 0) for t in daily_trades if t.get('pnl', 0) < 0)
        return abs(daily_loss)

    def _check_circuit_breaker(self) -> None:
        """Check circuit breaker conditions."""
        if self.circuit_breaker_triggered:
            return

        cb_config = self.circuit_breaker_config
        if not cb_config.get('enabled', True):
            return

        # Check daily loss
        daily_loss_threshold = (
            self.portfolio_config.get('initial_capital', 10000) *
            cb_config.get('daily_loss_threshold', 0.05)
        )
        if self._get_daily_loss() >= daily_loss_threshold:
            self.circuit_breaker_triggered = True
            logger.critical(f"Circuit breaker triggered: daily loss limit")
            return

        # Check consecutive losses
        recent_closes = [
            t for t in self.trade_log[-10:]  # Last 10 trades
            if t.get('type') == 'CLOSE'
        ]
        consecutive_losses = sum(1 for t in recent_closes if t.get('pnl', 0) < 0)

        if consecutive_losses >= cb_config.get('max_consecutive_losses', 5):
            self.circuit_breaker_triggered = True
            logger.critical(f"Circuit breaker triggered: {consecutive_losses} consecutive losses")

    def reset_daily_limits(self) -> None:
        """Reset daily loss tracking."""
        self.daily_start_capital = self.current_capital

    def get_portfolio_stats(self) -> Union[Dict[str, Any], PortfolioStats]:
        """Get current portfolio statistics as dict or PortfolioStats model."""
        closed_trades = [t for t in self.trade_log if t.get('type') == 'CLOSE']
        pnls = [t.get('pnl', 0) for t in closed_trades]

        capital_change = self.current_capital - self.initial_capital
        return_percent = (capital_change / self.initial_capital) * 100 if self.initial_capital else 0

        stats_dict = {
            'initial_capital': self.initial_capital,
            'current_capital': self.current_capital,
            'total_pnl': capital_change,
            'total_pnl_percent': return_percent,
            'total_trades': len(self.trade_log),
            'closed_trades': len(closed_trades),
            'daily_loss': self._get_daily_loss(),
            'circuit_breaker_triggered': self.circuit_breaker_triggered,
            'avg_pnl': sum(pnls) / len(pnls) if pnls else 0,
            'win_rate': sum(1 for p in pnls if p > 0) / len(pnls) if pnls else 0,
        }

        return stats_dict

    def validate_position_correlation(self, new_position: Dict[str, Any],
                                     existing_positions: List[Dict]) -> bool:
        """
        Check if new position has too high correlation with existing positions.

        Args:
            new_position: New position to add
            existing_positions: Current open positions

        Returns:
            True if correlation is acceptable
        """
        max_corr = self.position_config.get('max_correlation', 0.70)

        # Simplified: check if markets are too similar
        new_location = new_position.get('location', '')
        new_type = new_position.get('prediction_type', '')

        similar_positions = [
            p for p in existing_positions
            if p.get('location') == new_location and p.get('prediction_type') == new_type
        ]

        # If multiple positions on same market type/location, reject
        if similar_positions:
            return False

        return True

    def check_emergency_shutdown(self) -> bool:
        """Check if emergency shutdown should be triggered."""
        if not self.circuit_breaker_config.get('enabled', True):
            return False

        return self.circuit_breaker_triggered

    def get_risk_report(self) -> Dict[str, Any]:
        """Generate comprehensive risk report."""
        portfolio_stats = self.get_portfolio_stats()

        return {
            'timestamp': datetime.utcnow().isoformat(),
            'portfolio': portfolio_stats,
            'daily_loss': self._get_daily_loss(),
            'circuit_breaker_triggered': self.circuit_breaker_triggered,
            'emergency_shutdown': self.check_emergency_shutdown(),
            'max_drawdown': self._calculate_max_drawdown(),
            'risk_summary': {
                'status': 'CRITICAL' if self.circuit_breaker_triggered else 'NORMAL',
                'capital_at_risk': self.current_capital * 0.5,  # Conservative estimate
                'largest_position': self._get_largest_position(),
            }
        }

    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown."""
        capitals = [self.initial_capital]
        for trade in self.trade_log:
            if trade.get('type') == 'CLOSE':
                capitals.append(capitals[-1] + trade.get('pnl', 0))

        if not capitals:
            return 0.0

        max_capital = capitals[0]
        max_drawdown = 0.0

        for capital in capitals:
            if capital > max_capital:
                max_capital = capital
            drawdown = (max_capital - capital) / max_capital if max_capital else 0
            max_drawdown = max(max_drawdown, drawdown)

        return max_drawdown

    def _get_largest_position(self) -> float:
        """Get size of largest position."""
        open_trades = [t for t in self.trade_log if t.get('type') == 'OPEN']
        if not open_trades:
            return 0.0

        return max(t.get('trade', {}).get('position_size', 0) for t in open_trades)
