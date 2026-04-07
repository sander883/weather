"""Unit tests for dashboard tracker."""

import pytest
import tempfile
import json
from pathlib import Path
from datetime import datetime, timedelta

from src.dashboard.tracker import DashboardTracker


@pytest.fixture
def tracker():
    """Create tracker instance."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield DashboardTracker(data_dir=temp_dir)


@pytest.fixture
def sample_trade():
    """Create sample trade."""
    return {
        'trade_id': 'trade_123',
        'market_id': 'market_123',
        'action': 'BUY_YES',
        'position_size': 100,
        'entry_price': 0.45,
        'status': 'FILLED'
    }


@pytest.fixture
def sample_close():
    """Create sample close result."""
    return {
        'trade_id': 'trade_123',
        'market_id': 'market_123',
        'entry_price': 0.45,
        'exit_price': 0.55,
        'pnl': 100,
        'pnl_percent': 0.1,
        'hold_time_seconds': 3600
    }


@pytest.fixture
def sample_portfolio():
    """Create sample portfolio stats."""
    return {
        'initial_capital': 10000,
        'current_capital': 11000,
        'total_pnl': 1000,
        'total_pnl_percent': 0.10,
        'total_trades': 5,
        'closed_trades': 3,
        'avg_pnl': 333.33,
        'win_rate': 0.667,
        'circuit_breaker_triggered': False
    }


class TestDashboardTrackerInit:
    """Test tracker initialization."""

    def test_init(self, tracker):
        """Test initialization."""
        assert tracker.trades == []
        assert tracker.portfolio_snapshots == []
        assert tracker.alerts == []
        assert tracker.data_dir.exists()

    def test_custom_data_dir(self):
        """Test with custom data directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            tracker = DashboardTracker(data_dir=temp_dir)
            assert tracker.data_dir == Path(temp_dir)


class TestRecordTrade:
    """Test trade recording."""

    def test_record_trade(self, tracker, sample_trade):
        """Test recording a trade."""
        tracker.record_trade(sample_trade)

        assert len(tracker.trades) == 1
        assert tracker.trades[0]['trade_id'] == 'trade_123'
        assert 'timestamp' in tracker.trades[0]

    def test_record_multiple_trades(self, tracker, sample_trade):
        """Test recording multiple trades."""
        for i in range(5):
            sample_trade['trade_id'] = f'trade_{i}'
            tracker.record_trade(sample_trade)

        assert len(tracker.trades) == 5

    def test_record_trade_with_missing_fields(self, tracker):
        """Test recording trade with missing fields."""
        minimal_trade = {'trade_id': 'trade_999'}
        tracker.record_trade(minimal_trade)

        assert len(tracker.trades) == 1
        assert tracker.trades[0]['trade_id'] == 'trade_999'


class TestRecordPositionClose:
    """Test position close recording."""

    def test_record_close(self, tracker, sample_close):
        """Test recording position close."""
        tracker.record_position_close(sample_close)

        assert len(tracker.trades) == 1
        assert tracker.trades[0]['exit_price'] == 0.55
        assert tracker.trades[0]['pnl'] == 100

    def test_record_multiple_closes(self, tracker, sample_close):
        """Test recording multiple closes."""
        for i in range(3):
            sample_close['trade_id'] = f'trade_{i}'
            sample_close['pnl'] = 100 * (i + 1)
            tracker.record_position_close(sample_close)

        assert len(tracker.trades) == 3
        assert tracker.trades[0]['pnl'] == 100
        assert tracker.trades[2]['pnl'] == 300


class TestRecordPortfolioSnapshot:
    """Test portfolio snapshot recording."""

    def test_record_snapshot(self, tracker, sample_portfolio):
        """Test recording portfolio snapshot."""
        tracker.record_portfolio_snapshot(sample_portfolio)

        assert len(tracker.portfolio_snapshots) == 1
        assert tracker.portfolio_snapshots[0]['current_capital'] == 11000

    def test_record_multiple_snapshots(self, tracker, sample_portfolio):
        """Test recording multiple snapshots."""
        for i in range(5):
            sample_portfolio['current_capital'] = 10000 + (i * 500)
            tracker.record_portfolio_snapshot(sample_portfolio)

        assert len(tracker.portfolio_snapshots) == 5


class TestAlerts:
    """Test alert management."""

    def test_add_alert(self, tracker):
        """Test adding alert."""
        tracker.add_alert('trade', 'Test trade alert', 'INFO')

        assert len(tracker.alerts) == 1
        assert tracker.alerts[0]['type'] == 'trade'
        assert tracker.alerts[0]['message'] == 'Test trade alert'
        assert tracker.alerts[0]['severity'] == 'INFO'

    def test_add_multiple_alerts(self, tracker):
        """Test adding multiple alerts."""
        severities = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        for severity in severities:
            tracker.add_alert('system', f'Test {severity}', severity)

        assert len(tracker.alerts) == 5


class TestGetSummary:
    """Test summary generation."""

    def test_summary_empty_tracker(self, tracker):
        """Test summary with no data."""
        summary = tracker.get_summary()

        assert 'timestamp' in summary
        assert summary['total_trades'] == 0
        assert summary['closed_trades'] == 0

    def test_summary_with_trades(self, tracker, sample_trade, sample_close):
        """Test summary with trades."""
        tracker.record_trade(sample_trade)
        tracker.record_position_close(sample_close)

        summary = tracker.get_summary()

        assert summary['total_trades'] == 2
        assert summary['closed_trades'] == 1

    def test_summary_includes_portfolio(self, tracker, sample_portfolio):
        """Test that summary includes portfolio."""
        tracker.record_portfolio_snapshot(sample_portfolio)

        summary = tracker.get_summary()

        assert 'portfolio' in summary
        assert summary['portfolio']['current_capital'] == 11000


class TestDailyMetrics:
    """Test daily metrics."""

    def test_daily_metrics_no_trades(self, tracker):
        """Test daily metrics with no trades."""
        metrics = tracker.get_daily_metrics()

        assert metrics['total_trades'] == 0
        assert metrics['total_pnl'] == 0

    def test_daily_metrics_with_trades(self, tracker, sample_close):
        """Test daily metrics with trades."""
        # Record multiple closes
        for i in range(3):
            sample_close['trade_id'] = f'trade_{i}'
            sample_close['pnl'] = 100 * (i + 1)
            tracker.record_position_close(sample_close)

        metrics = tracker.get_daily_metrics()

        assert metrics['total_trades'] == 3
        assert metrics['total_pnl'] == 600  # 100 + 200 + 300
        assert metrics['winning_trades'] == 3
        assert metrics['losing_trades'] == 0

    def test_daily_metrics_specific_date(self, tracker, sample_close):
        """Test daily metrics for specific date."""
        tracker.record_position_close(sample_close)

        today = datetime.utcnow().date().isoformat()
        metrics = tracker.get_daily_metrics(date=today)

        assert metrics['date'] == today
        assert metrics['total_trades'] == 1


class TestPerformanceMetrics:
    """Test performance metrics calculation."""

    def test_performance_no_trades(self, tracker):
        """Test performance with no trades."""
        perf = tracker.get_performance_metrics()

        assert 'message' in perf

    def test_performance_with_trades(self, tracker):
        """Test performance metrics with trades."""
        # Record some closed trades with different PnLs
        closes = [
            {'pnl': 100, 'hold_time_seconds': 1800},
            {'pnl': -50, 'hold_time_seconds': 900},
            {'pnl': 200, 'hold_time_seconds': 3600},
        ]

        for i, close in enumerate(closes):
            sample_close = {
                'trade_id': f'trade_{i}',
                'market_id': 'market_123',
                'entry_price': 0.45,
                'exit_price': 0.55,
                'hold_time_seconds': close['hold_time_seconds']
            }
            sample_close.update(close)
            tracker.record_position_close(sample_close)

        perf = tracker.get_performance_metrics()

        assert perf['total_trades'] == 3
        assert perf['winning_trades'] == 2
        assert perf['losing_trades'] == 1
        assert perf['win_rate'] == pytest.approx(2/3)
        assert perf['total_pnl'] == 250
        assert perf['avg_pnl'] == pytest.approx(250/3)

    def test_profit_factor(self, tracker):
        """Test profit factor calculation."""
        # Winning trade: +200
        tracker.record_position_close({
            'trade_id': 'win',
            'market_id': 'm1',
            'entry_price': 0.5,
            'exit_price': 0.6,
            'pnl': 200,
            'hold_time_seconds': 1000
        })

        # Losing trade: -100
        tracker.record_position_close({
            'trade_id': 'loss',
            'market_id': 'm2',
            'entry_price': 0.5,
            'exit_price': 0.45,
            'pnl': -100,
            'hold_time_seconds': 1000
        })

        perf = tracker.get_performance_metrics()

        assert perf['profit_factor'] == 2.0  # 200 / 100


class TestExportMetrics:
    """Test metrics export."""

    def test_export_metrics(self, tracker, sample_trade, sample_close):
        """Test exporting metrics."""
        tracker.record_trade(sample_trade)
        tracker.record_position_close(sample_close)
        tracker.add_alert('test', 'Test alert')

        filepath = tracker.export_metrics()

        assert Path(filepath).exists()
        assert filepath.endswith('.json')

        with open(filepath) as f:
            data = json.load(f)
            assert 'summary' in data
            assert 'performance' in data
            assert 'trades' in data

    def test_export_with_custom_filename(self, tracker):
        """Test export with custom filename."""
        filepath = tracker.export_metrics('custom_metrics.json')

        assert 'custom_metrics.json' in filepath
        assert Path(filepath).exists()


class TestPortfolioHistory:
    """Test portfolio history retrieval."""

    def test_portfolio_history_empty(self, tracker):
        """Test history with no snapshots."""
        history = tracker.get_portfolio_history()

        assert history == []

    def test_portfolio_history_with_snapshots(self, tracker, sample_portfolio):
        """Test history with snapshots."""
        for i in range(5):
            sample_portfolio['current_capital'] = 10000 + (i * 500)
            tracker.record_portfolio_snapshot(sample_portfolio)

        history = tracker.get_portfolio_history(days=7)

        assert len(history) == 5

    def test_portfolio_history_filters_by_days(self, tracker, sample_portfolio):
        """Test that history filters by days."""
        # Record old snapshot
        old_snapshot = sample_portfolio.copy()
        old_snapshot['timestamp'] = (datetime.utcnow() - timedelta(days=10)).isoformat()
        tracker.portfolio_snapshots.append(old_snapshot)

        # Record recent snapshot
        tracker.record_portfolio_snapshot(sample_portfolio)

        history = tracker.get_portfolio_history(days=7)

        # Should only have recent snapshot
        assert len(history) == 1


class TestClearOldData:
    """Test data cleanup."""

    def test_clear_old_trades(self, tracker):
        """Test clearing old trades."""
        # Add old trade
        old_trade = {
            'timestamp': (datetime.utcnow() - timedelta(days=40)).isoformat(),
            'trade_id': 'old_trade'
        }
        tracker.trades.append(old_trade)

        # Add recent trade
        recent_trade = {
            'timestamp': datetime.utcnow().isoformat(),
            'trade_id': 'recent_trade'
        }
        tracker.record_trade(recent_trade)

        # Clear trades older than 30 days
        tracker.clear_old_data(days=30)

        assert len(tracker.trades) == 1
        assert tracker.trades[0]['trade_id'] == 'recent_trade'

    def test_clear_all_data_types(self, tracker, sample_trade, sample_close, sample_portfolio):
        """Test that clear_old_data removes all old entries."""
        # Add old entries
        old_time = (datetime.utcnow() - timedelta(days=40)).isoformat()

        tracker.trades.append({'timestamp': old_time, 'trade_id': 'old_trade'})
        tracker.portfolio_snapshots.append({'timestamp': old_time, 'current_capital': 9000})
        tracker.alerts.append({'timestamp': old_time, 'message': 'Old alert'})

        # Add recent entries
        tracker.record_trade(sample_trade)
        tracker.record_portfolio_snapshot(sample_portfolio)
        tracker.add_alert('recent', 'Recent alert')

        # Clear old data
        tracker.clear_old_data(days=30)

        # All should only have 1 recent entry
        assert len(tracker.trades) == 1
        assert len(tracker.portfolio_snapshots) == 1
        assert len(tracker.alerts) == 1


class TestEdgeCases:
    """Test edge cases."""

    def test_metrics_with_mixed_pnls(self, tracker):
        """Test performance with mixed P&L trades."""
        pnls = [100, -50, 200, -75, 150, -25, 300]

        for i, pnl in enumerate(pnls):
            tracker.record_position_close({
                'trade_id': f'trade_{i}',
                'market_id': f'm{i}',
                'entry_price': 0.5,
                'exit_price': 0.5 + (pnl / 1000),
                'pnl': pnl,
                'hold_time_seconds': 1000
            })

        perf = tracker.get_performance_metrics()

        assert perf['total_trades'] == 7
        assert perf['winning_trades'] == 4
        assert perf['losing_trades'] == 3
        assert perf['total_pnl'] == 600

    def test_portfolio_with_declining_value(self, tracker, sample_portfolio):
        """Test portfolio snapshots with declining value."""
        for i in range(5):
            sample_portfolio['current_capital'] = 10000 - (i * 500)
            tracker.record_portfolio_snapshot(sample_portfolio)

        history = tracker.get_portfolio_history()

        assert history[0]['current_capital'] > history[-1]['current_capital']

    def test_alerts_with_all_severities(self, tracker):
        """Test alerts with various severities."""
        severities = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']

        for severity in severities:
            tracker.add_alert('test', f'Test {severity}', severity)

        assert len(tracker.alerts) == 5
        assert all(a['severity'] in severities for a in tracker.alerts)
