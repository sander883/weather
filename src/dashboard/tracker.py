"""Dashboard tracker for monitoring bot performance."""

from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json
from pathlib import Path
from collections import defaultdict

from src.utils.logger import get_logger

logger = get_logger(__name__)


class DashboardTracker:
    """Track and store dashboard metrics."""

    def __init__(self, data_dir: str = 'data/dashboard'):
        """
        Initialize dashboard tracker.

        Args:
            data_dir: Directory for storing metrics
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.trades = []
        self.portfolio_snapshots = []
        self.daily_metrics = defaultdict(dict)
        self.alerts = []

    def record_trade(self, trade: Dict[str, Any]) -> None:
        """
        Record a executed trade.

        Args:
            trade: Trade execution details
        """
        try:
            trade_record = {
                'timestamp': datetime.utcnow().isoformat(),
                'trade_id': trade.get('trade_id'),
                'market_id': trade.get('market_id'),
                'action': trade.get('action'),
                'position_size': trade.get('position_size'),
                'entry_price': trade.get('entry_price'),
                'status': trade.get('status')
            }

            self.trades.append(trade_record)
            logger.info(f"Trade recorded: {trade_record['trade_id']}")

        except Exception as e:
            logger.error(f"Error recording trade: {e}")

    def record_position_close(self, close_result: Dict[str, Any]) -> None:
        """
        Record a closed position.

        Args:
            close_result: Position close details
        """
        try:
            close_record = {
                'timestamp': datetime.utcnow().isoformat(),
                'trade_id': close_result.get('trade_id'),
                'market_id': close_result.get('market_id'),
                'entry_price': close_result.get('entry_price'),
                'exit_price': close_result.get('exit_price'),
                'pnl': close_result.get('pnl'),
                'pnl_percent': close_result.get('pnl_percent'),
                'hold_time_seconds': close_result.get('hold_time_seconds')
            }

            self.trades.append(close_record)
            logger.info(f"Position closed: PnL {close_record['pnl_percent']:.2%}")

        except Exception as e:
            logger.error(f"Error recording close: {e}")

    def record_portfolio_snapshot(self, portfolio_stats: Dict[str, Any]) -> None:
        """
        Record portfolio snapshot.

        Args:
            portfolio_stats: Portfolio statistics
        """
        try:
            snapshot = {
                'timestamp': datetime.utcnow().isoformat(),
                'initial_capital': portfolio_stats.get('initial_capital'),
                'current_capital': portfolio_stats.get('current_capital'),
                'total_pnl': portfolio_stats.get('total_pnl'),
                'total_pnl_percent': portfolio_stats.get('total_pnl_percent'),
                'total_trades': portfolio_stats.get('total_trades'),
                'closed_trades': portfolio_stats.get('closed_trades'),
                'avg_pnl': portfolio_stats.get('avg_pnl'),
                'win_rate': portfolio_stats.get('win_rate'),
                'circuit_breaker_triggered': portfolio_stats.get('circuit_breaker_triggered')
            }

            self.portfolio_snapshots.append(snapshot)
            logger.debug("Portfolio snapshot recorded")

        except Exception as e:
            logger.error(f"Error recording snapshot: {e}")

    def add_alert(self, alert_type: str, message: str, severity: str = 'INFO') -> None:
        """
        Add alert to dashboard.

        Args:
            alert_type: Type of alert (trade, risk, system, etc)
            message: Alert message
            severity: Severity level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        try:
            alert = {
                'timestamp': datetime.utcnow().isoformat(),
                'type': alert_type,
                'message': message,
                'severity': severity
            }

            self.alerts.append(alert)
            logger.log(
                getattr(__import__('logging'), severity),
                f"Alert [{alert_type}]: {message}"
            )

        except Exception as e:
            logger.error(f"Error adding alert: {e}")

    def get_summary(self) -> Dict[str, Any]:
        """
        Get dashboard summary.

        Returns:
            Summary statistics
        """
        try:
            latest_portfolio = self.portfolio_snapshots[-1] if self.portfolio_snapshots else {}

            closed_trades = [t for t in self.trades if 'exit_price' in t]
            pnls = [t.get('pnl', 0) for t in closed_trades if 'pnl' in t]

            summary = {
                'timestamp': datetime.utcnow().isoformat(),
                'portfolio': latest_portfolio,
                'total_trades': len(self.trades),
                'closed_trades': len(closed_trades),
                'open_trades': len(self.trades) - len(closed_trades),
                'total_pnl': sum(pnls) if pnls else 0,
                'avg_pnl': sum(pnls) / len(pnls) if pnls else 0,
                'win_rate': sum(1 for p in pnls if p > 0) / len(pnls) if pnls else 0,
                'recent_trades': self.trades[-10:],
                'recent_alerts': self.alerts[-5:]
            }

            return summary

        except Exception as e:
            logger.error(f"Error getting summary: {e}")
            return {}

    def get_daily_metrics(self, date: Optional[str] = None) -> Dict[str, Any]:
        """
        Get metrics for a specific day.

        Args:
            date: Date string (YYYY-MM-DD). If None, uses today.

        Returns:
            Daily metrics
        """
        try:
            if date is None:
                date = datetime.utcnow().date().isoformat()

            # Find trades for the day
            day_start = datetime.fromisoformat(f"{date}T00:00:00")
            day_end = datetime.fromisoformat(f"{date}T23:59:59")

            day_trades = [
                t for t in self.trades
                if datetime.fromisoformat(t['timestamp']) >= day_start
                and datetime.fromisoformat(t['timestamp']) <= day_end
            ]

            closed_day_trades = [t for t in day_trades if 'exit_price' in t]
            pnls = [t.get('pnl', 0) for t in closed_day_trades if 'pnl' in t]

            metrics = {
                'date': date,
                'total_trades': len(day_trades),
                'closed_trades': len(closed_day_trades),
                'total_pnl': sum(pnls) if pnls else 0,
                'avg_pnl': sum(pnls) / len(pnls) if pnls else 0,
                'win_rate': sum(1 for p in pnls if p > 0) / len(pnls) if pnls else 0,
                'winning_trades': sum(1 for p in pnls if p > 0),
                'losing_trades': sum(1 for p in pnls if p < 0),
                'trades': day_trades
            }

            return metrics

        except Exception as e:
            logger.error(f"Error getting daily metrics: {e}")
            return {}

    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive performance metrics.

        Returns:
            Performance statistics
        """
        try:
            closed_trades = [t for t in self.trades if 'exit_price' in t]
            pnls = [t.get('pnl', 0) for t in closed_trades if 'pnl' in t]
            hold_times = [t.get('hold_time_seconds', 0) for t in closed_trades]

            if not pnls:
                return {'message': 'No closed trades yet'}

            winning_pnls = [p for p in pnls if p > 0]
            losing_pnls = [p for p in pnls if p < 0]

            metrics = {
                'total_trades': len(closed_trades),
                'winning_trades': len(winning_pnls),
                'losing_trades': len(losing_pnls),
                'win_rate': len(winning_pnls) / len(pnls),
                'total_pnl': sum(pnls),
                'avg_pnl': sum(pnls) / len(pnls),
                'avg_win': sum(winning_pnls) / len(winning_pnls) if winning_pnls else 0,
                'avg_loss': sum(losing_pnls) / len(losing_pnls) if losing_pnls else 0,
                'profit_factor': sum(winning_pnls) / abs(sum(losing_pnls)) if losing_pnls else 0,
                'avg_hold_time': sum(hold_times) / len(hold_times) if hold_times else 0,
                'max_pnl': max(pnls) if pnls else 0,
                'min_pnl': min(pnls) if pnls else 0
            }

            return metrics

        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return {}

    def export_metrics(self, filename: Optional[str] = None) -> str:
        """
        Export metrics to JSON file.

        Args:
            filename: Output filename

        Returns:
            Path to exported file
        """
        try:
            if filename is None:
                filename = f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

            filepath = self.data_dir / filename

            data = {
                'timestamp': datetime.utcnow().isoformat(),
                'summary': self.get_summary(),
                'performance': self.get_performance_metrics(),
                'trades': self.trades[-100:],  # Last 100 trades
                'portfolio_snapshots': self.portfolio_snapshots[-50:],  # Last 50 snapshots
                'alerts': self.alerts[-50:]  # Last 50 alerts
            }

            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)

            logger.info(f"Metrics exported to {filepath}")
            return str(filepath)

        except Exception as e:
            logger.error(f"Error exporting metrics: {e}")
            return ""

    def get_portfolio_history(self, days: int = 7) -> List[Dict[str, Any]]:
        """
        Get portfolio value history.

        Args:
            days: Number of days to retrieve

        Returns:
            Portfolio history
        """
        try:
            cutoff = datetime.utcnow() - timedelta(days=days)

            history = [
                s for s in self.portfolio_snapshots
                if datetime.fromisoformat(s['timestamp']) >= cutoff
            ]

            return history

        except Exception as e:
            logger.error(f"Error getting portfolio history: {e}")
            return []

    def clear_old_data(self, days: int = 30) -> None:
        """
        Clear data older than specified days.

        Args:
            days: Keep data from last N days
        """
        try:
            cutoff = datetime.utcnow() - timedelta(days=days)

            self.trades = [
                t for t in self.trades
                if datetime.fromisoformat(t['timestamp']) >= cutoff
            ]

            self.portfolio_snapshots = [
                s for s in self.portfolio_snapshots
                if datetime.fromisoformat(s['timestamp']) >= cutoff
            ]

            self.alerts = [
                a for a in self.alerts
                if datetime.fromisoformat(a['timestamp']) >= cutoff
            ]

            logger.info(f"Cleared data older than {days} days")

        except Exception as e:
            logger.error(f"Error clearing old data: {e}")
