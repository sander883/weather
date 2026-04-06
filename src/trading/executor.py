"""Order execution module."""

from typing import Dict, List, Optional, Any
from datetime import datetime
import json
from pathlib import Path

from src.utils.logger import get_logger
from src.utils.helpers import save_json, load_json

logger = get_logger(__name__)


class OrderExecutor:
    """Execute trades on Polymarket."""

    def __init__(self, config: Dict[str, Any], simulate: bool = True):
        """
        Initialize executor.

        Args:
            config: Configuration dict
            simulate: If True, simulate trades without actual execution
        """
        self.config = config
        self.simulate = simulate
        self.order_history_file = Path('data/history/orders.json')
        self.order_history_file.parent.mkdir(parents=True, exist_ok=True)
        self.open_positions = {}

    def execute_trade(self, trade: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a trade.

        Args:
            trade: Trade recommendation dict

        Returns:
            Execution result dict
        """
        if self.simulate:
            return self._simulate_execution(trade)
        else:
            return self._execute_on_polymarket(trade)

    def _simulate_execution(self, trade: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate trade execution."""
        try:
            execution = {
                'trade_id': self._generate_trade_id(),
                'market_id': trade['market_id'],
                'question': trade['question'],
                'action': trade['action'],
                'position_size': trade['position_size'],
                'entry_price': trade['entry_price'],
                'execution_price': trade['entry_price'],  # Assume filled at target price in sim
                'filled_amount': trade['position_size'],
                'status': 'FILLED',
                'execution_time': datetime.utcnow().isoformat(),
                'slippage': 0.0,
                'is_simulated': True
            }

            # Store position
            self.open_positions[execution['trade_id']] = execution

            # Save to history
            self._save_execution(execution)

            logger.info(f"Simulated trade executed: {execution['trade_id']}")
            return execution

        except Exception as e:
            logger.error(f"Simulation failed: {e}")
            return {'status': 'FAILED', 'error': str(e)}

    def _execute_on_polymarket(self, trade: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute trade on actual Polymarket.

        This is a placeholder - actual implementation would use Polymarket API.
        """
        logger.warning("Actual Polymarket execution not implemented")

        # Would implement actual API calls here
        execution = {
            'trade_id': self._generate_trade_id(),
            'market_id': trade['market_id'],
            'action': trade['action'],
            'status': 'PENDING',
            'error': 'Not implemented'
        }

        return execution

    def close_position(self, trade_id: str, exit_price: float) -> Dict[str, Any]:
        """
        Close an open position.

        Args:
            trade_id: Position ID to close
            exit_price: Price to exit at

        Returns:
            Close result dict
        """
        if trade_id not in self.open_positions:
            logger.warning(f"Position {trade_id} not found")
            return {'status': 'FAILED', 'error': 'Position not found'}

        position = self.open_positions[trade_id]

        try:
            close_result = {
                'trade_id': trade_id,
                'market_id': position['market_id'],
                'action': position['action'],
                'entry_price': position['entry_price'],
                'exit_price': exit_price,
                'entry_amount': position['position_size'],
                'exit_amount': position['position_size'],  # Simplified
                'pnl': self._calculate_pnl(position, exit_price),
                'pnl_percent': self._calculate_pnl_percent(position, exit_price),
                'close_time': datetime.utcnow().isoformat(),
                'hold_time_seconds': self._get_hold_time(position),
                'status': 'CLOSED',
                'is_simulated': position.get('is_simulated', False)
            }

            # Save close
            self._save_execution(close_result)

            # Remove from open positions
            del self.open_positions[trade_id]

            logger.info(f"Position {trade_id} closed with PnL: {close_result['pnl_percent']:.2%}")
            return close_result

        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return {'status': 'FAILED', 'error': str(e)}

    def _calculate_pnl(self, position: Dict, exit_price: float) -> float:
        """Calculate P&L in USD."""
        entry_price = position['entry_price']
        amount = position['position_size']

        if position['action'] == 'BUY_YES':
            pnl = amount * (exit_price - entry_price)
        else:  # BUY_NO
            pnl = amount * (entry_price - exit_price)

        return pnl

    def _calculate_pnl_percent(self, position: Dict, exit_price: float) -> float:
        """Calculate P&L as percentage."""
        pnl = self._calculate_pnl(position, exit_price)
        amount = position['position_size']

        if amount == 0:
            return 0.0

        return pnl / amount

    def _get_hold_time(self, position: Dict) -> float:
        """Get hold time in seconds."""
        entry_time = datetime.fromisoformat(position['execution_time'])
        return (datetime.utcnow() - entry_time).total_seconds()

    def get_open_positions(self) -> List[Dict]:
        """Get all open positions."""
        return list(self.open_positions.values())

    def get_position(self, trade_id: str) -> Optional[Dict]:
        """Get specific position."""
        return self.open_positions.get(trade_id)

    def update_position_prices(self, price_updates: Dict[str, float]) -> None:
        """
        Update current prices for open positions.

        Args:
            price_updates: Dict mapping trade_id to current price
        """
        for trade_id, price in price_updates.items():
            if trade_id in self.open_positions:
                position = self.open_positions[trade_id]
                position['current_price'] = price
                position['current_pnl'] = self._calculate_pnl(position, price)
                position['current_pnl_percent'] = self._calculate_pnl_percent(position, price)

    def _save_execution(self, execution: Dict) -> None:
        """Save execution to history file."""
        try:
            history = []
            if self.order_history_file.exists():
                history = load_json(str(self.order_history_file))

            history.append(execution)
            save_json(history, str(self.order_history_file))

        except Exception as e:
            logger.error(f"Error saving execution: {e}")

    def get_execution_history(self, limit: int = 100) -> List[Dict]:
        """Get execution history."""
        try:
            if not self.order_history_file.exists():
                return []

            history = load_json(str(self.order_history_file))
            return history[-limit:]

        except Exception as e:
            logger.error(f"Error loading history: {e}")
            return []

    def _generate_trade_id(self) -> str:
        """Generate unique trade ID."""
        import uuid
        return f"TRD_{uuid.uuid4().hex[:12].upper()}"

    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        history = self.get_execution_history(limit=1000)

        if not history:
            return {}

        completed = [h for h in history if h.get('status') == 'CLOSED']

        if not completed:
            return {'total_trades': len(history), 'closed_trades': 0}

        pnls = [h.get('pnl', 0) for h in completed]
        pnl_percents = [h.get('pnl_percent', 0) for h in completed]

        wins = sum(1 for p in pnls if p > 0)
        losses = sum(1 for p in pnls if p < 0)

        return {
            'total_trades': len(history),
            'closed_trades': len(completed),
            'open_positions': len(self.open_positions),
            'wins': wins,
            'losses': losses,
            'win_rate': wins / len(completed) if completed else 0,
            'total_pnl': sum(pnls),
            'avg_pnl': sum(pnls) / len(completed) if completed else 0,
            'avg_pnl_percent': sum(pnl_percents) / len(completed) if completed else 0,
            'max_pnl': max(pnls),
            'min_pnl': min(pnls),
        }
