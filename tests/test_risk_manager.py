"""Unit tests for risk management module."""

import pytest
from datetime import datetime, timedelta
from src.trading.risk_manager import RiskManager


@pytest.fixture
def sample_config():
    """Create sample risk management configuration."""
    return {
        'risk_management': {
            'portfolio': {
                'initial_capital': 10000,
                'max_daily_loss_percent': 2.0
            },
            'position': {
                'max_concurrent_positions': 5,
                'min_position_size': 10,
                'max_concentration': 0.30,
                'max_correlation': 0.70
            },
            'circuit_breaker': {
                'enabled': True,
                'daily_loss_threshold': 0.05,
                'max_consecutive_losses': 5
            }
        }
    }


@pytest.fixture
def sample_trade():
    """Create sample trade."""
    return {
        'market_id': 'market_123',
        'action': 'BUY_YES',
        'position_size': 500,
        'entry_price': 0.55
    }


@pytest.fixture
def sample_close_result():
    """Create sample close result."""
    return {
        'market_id': 'market_123',
        'pnl': 100.0
    }


class TestRiskManagerInit:
    """Test RiskManager initialization."""

    def test_init_with_config(self, sample_config):
        """Test initialization with config."""
        manager = RiskManager(config=sample_config)
        assert manager.config == sample_config
        assert manager.initial_capital == 10000
        assert manager.current_capital == 10000
        assert manager.circuit_breaker_triggered is False

    def test_init_default_values(self):
        """Test initialization with minimal config."""
        config = {'risk_management': {}}
        manager = RiskManager(config=config)
        assert manager.initial_capital == 10000
        assert manager.current_capital == 10000

    def test_init_custom_capital(self):
        """Test initialization with custom capital."""
        config = {
            'risk_management': {
                'portfolio': {'initial_capital': 50000}
            }
        }
        manager = RiskManager(config=config)
        assert manager.initial_capital == 50000
        assert manager.current_capital == 50000


class TestCheckTradeFeasibility:
    """Test check_trade_feasibility method."""

    def test_feasible_trade(self, sample_config, sample_trade):
        """Test that valid trade is feasible."""
        manager = RiskManager(config=sample_config)
        feasible, reason = manager.check_trade_feasibility(sample_trade, [])

        assert feasible is True
        assert reason == "OK"

    def test_circuit_breaker_blocks_trade(self, sample_config, sample_trade):
        """Test that circuit breaker prevents trades."""
        manager = RiskManager(config=sample_config)
        manager.circuit_breaker_triggered = True

        feasible, reason = manager.check_trade_feasibility(sample_trade, [])

        assert feasible is False
        assert "Circuit breaker" in reason

    def test_max_concurrent_positions_exceeded(self, sample_config, sample_trade):
        """Test that max concurrent positions is enforced."""
        manager = RiskManager(config=sample_config)
        open_positions = [
            {'position_size': 500},
            {'position_size': 500},
            {'position_size': 500},
            {'position_size': 500},
            {'position_size': 500},
        ]

        feasible, reason = manager.check_trade_feasibility(sample_trade, open_positions)

        assert feasible is False
        assert "Max concurrent" in reason

    def test_concentration_limit_exceeded(self, sample_config):
        """Test that concentration limit is enforced."""
        sample_config['risk_management']['position']['max_concentration'] = 0.20
        manager = RiskManager(config=sample_config)

        open_positions = [
            {'position_size': 2500},  # 25% of capital
        ]
        trade = {'position_size': 2000}  # Would exceed 20% limit

        feasible, reason = manager.check_trade_feasibility(trade, open_positions)

        assert feasible is False
        assert "Concentration" in reason

    def test_minimum_position_size(self, sample_config):
        """Test that minimum position size is enforced."""
        manager = RiskManager(config=sample_config)

        trade = {'position_size': 5}  # Below minimum of 10

        feasible, reason = manager.check_trade_feasibility(trade, [])

        assert feasible is False
        assert "minimum" in reason.lower()

    def test_insufficient_capital(self, sample_config):
        """Test that insufficient capital blocks trade."""
        manager = RiskManager(config=sample_config)
        manager.current_capital = 100

        trade = {'position_size': 100}  # More than 50% of available capital

        feasible, reason = manager.check_trade_feasibility(trade, [])

        assert feasible is False
        assert "capital" in reason.lower()

    def test_daily_loss_limit_exceeded(self, sample_config):
        """Test that daily loss limit is enforced."""
        manager = RiskManager(config=sample_config)
        # Simulate daily loss of 300 (3% of 10000, exceeds 2% limit)
        manager.trade_log = [
            {
                'timestamp': datetime.utcnow(),
                'type': 'CLOSE',
                'pnl': -300
            }
        ]

        feasible, reason = manager.check_trade_feasibility({'position_size': 100}, [])

        assert feasible is False
        assert "Daily loss" in reason

    def test_invalid_trade_type(self, sample_config):
        """Test with invalid trade type."""
        manager = RiskManager(config=sample_config)
        feasible, reason = manager.check_trade_feasibility("not a dict", [])

        assert feasible is False
        assert "Invalid trade" in reason

    def test_invalid_open_positions_type(self, sample_config, sample_trade):
        """Test with invalid open_positions type."""
        manager = RiskManager(config=sample_config)
        # Should handle gracefully by converting to list
        feasible, reason = manager.check_trade_feasibility(sample_trade, "not a list")

        # Should attempt to validate despite type issue
        assert isinstance(feasible, bool)


class TestTradeRecording:
    """Test record_trade method."""

    def test_record_trade(self, sample_config, sample_trade):
        """Test recording a trade."""
        manager = RiskManager(config=sample_config)
        assert len(manager.trade_log) == 0

        manager.record_trade(sample_trade)

        assert len(manager.trade_log) == 1
        assert manager.trade_log[0]['trade'] == sample_trade
        assert 'timestamp' in manager.trade_log[0]

    def test_record_multiple_trades(self, sample_config, sample_trade):
        """Test recording multiple trades."""
        manager = RiskManager(config=sample_config)

        for i in range(5):
            manager.record_trade(sample_trade)

        assert len(manager.trade_log) == 5


class TestCloseRecording:
    """Test record_close method."""

    def test_record_close_profit(self, sample_config):
        """Test recording a profitable close."""
        manager = RiskManager(config=sample_config)
        initial_capital = manager.current_capital

        close_result = {'pnl': 500}
        manager.record_close(close_result)

        assert manager.current_capital == initial_capital + 500
        assert len(manager.trade_log) == 1

    def test_record_close_loss(self, sample_config):
        """Test recording a loss close."""
        manager = RiskManager(config=sample_config)
        initial_capital = manager.current_capital

        close_result = {'pnl': -200}
        manager.record_close(close_result)

        assert manager.current_capital == initial_capital - 200

    def test_record_multiple_closes(self, sample_config):
        """Test recording multiple closes."""
        manager = RiskManager(config=sample_config)

        manager.record_close({'pnl': 100})
        manager.record_close({'pnl': -50})
        manager.record_close({'pnl': 200})

        assert manager.current_capital == 10000 + 100 - 50 + 200


class TestGetDailyLoss:
    """Test _get_daily_loss method."""

    def test_daily_loss_no_trades(self, sample_config):
        """Test daily loss with no trades."""
        manager = RiskManager(config=sample_config)
        daily_loss = manager._get_daily_loss()

        assert daily_loss == 0

    def test_daily_loss_with_losses(self, sample_config):
        """Test daily loss calculation."""
        manager = RiskManager(config=sample_config)
        manager.trade_log = [
            {'timestamp': datetime.utcnow(), 'type': 'CLOSE', 'pnl': -100},
            {'timestamp': datetime.utcnow(), 'type': 'CLOSE', 'pnl': -200},
            {'timestamp': datetime.utcnow(), 'type': 'CLOSE', 'pnl': 150},
        ]

        daily_loss = manager._get_daily_loss()

        assert daily_loss == 300  # Sum of losses (ignores gains)

    def test_daily_loss_only_today(self, sample_config):
        """Test that daily loss only counts today's trades."""
        manager = RiskManager(config=sample_config)
        yesterday = datetime.utcnow() - timedelta(days=1)

        manager.trade_log = [
            {'timestamp': yesterday, 'type': 'CLOSE', 'pnl': -500},
            {'timestamp': datetime.utcnow(), 'type': 'CLOSE', 'pnl': -100},
        ]

        daily_loss = manager._get_daily_loss()

        assert daily_loss == 100  # Only today's loss


class TestCircuitBreaker:
    """Test _check_circuit_breaker method."""

    def test_circuit_breaker_disabled(self, sample_config):
        """Test that circuit breaker can be disabled."""
        sample_config['risk_management']['circuit_breaker']['enabled'] = False
        manager = RiskManager(config=sample_config)

        # Simulate large loss
        manager.trade_log = [
            {'timestamp': datetime.utcnow(), 'type': 'CLOSE', 'pnl': -600}
        ]

        manager._check_circuit_breaker()

        assert manager.circuit_breaker_triggered is False

    def test_circuit_breaker_daily_loss_threshold(self, sample_config):
        """Test circuit breaker on daily loss threshold."""
        manager = RiskManager(config=sample_config)
        # Threshold is 5% of 10000 = 500

        manager.trade_log = [
            {'timestamp': datetime.utcnow(), 'type': 'CLOSE', 'pnl': -600}
        ]

        manager._check_circuit_breaker()

        assert manager.circuit_breaker_triggered is True

    def test_circuit_breaker_consecutive_losses(self, sample_config):
        """Test circuit breaker on consecutive losses."""
        manager = RiskManager(config=sample_config)
        # Max consecutive losses is 5

        for i in range(6):
            manager.trade_log.append({
                'timestamp': datetime.utcnow(),
                'type': 'CLOSE',
                'pnl': -50
            })

        manager._check_circuit_breaker()

        assert manager.circuit_breaker_triggered is True

    def test_circuit_breaker_not_triggered_with_wins(self, sample_config):
        """Test that circuit breaker is not triggered with mixed wins/losses."""
        manager = RiskManager(config=sample_config)

        manager.trade_log = [
            {'timestamp': datetime.utcnow(), 'type': 'CLOSE', 'pnl': -50},
            {'timestamp': datetime.utcnow(), 'type': 'CLOSE', 'pnl': 100},
            {'timestamp': datetime.utcnow(), 'type': 'CLOSE', 'pnl': -50},
        ]

        manager._check_circuit_breaker()

        assert manager.circuit_breaker_triggered is False


class TestResetDailyLimits:
    """Test reset_daily_limits method."""

    def test_reset_daily_limits(self, sample_config):
        """Test resetting daily limits."""
        manager = RiskManager(config=sample_config)
        manager.current_capital = 8000

        manager.reset_daily_limits()

        assert manager.daily_start_capital == 8000


class TestGetPortfolioStats:
    """Test get_portfolio_stats method."""

    def test_portfolio_stats_no_trades(self, sample_config):
        """Test portfolio stats with no trades."""
        manager = RiskManager(config=sample_config)
        stats = manager.get_portfolio_stats()

        assert stats['initial_capital'] == 10000
        assert stats['current_capital'] == 10000
        assert stats['total_pnl'] == 0
        assert stats['total_pnl_percent'] == 0
        assert stats['closed_trades'] == 0

    def test_portfolio_stats_with_trades(self, sample_config):
        """Test portfolio stats with trades."""
        manager = RiskManager(config=sample_config)
        manager.current_capital = 11000
        manager.trade_log = [
            {'timestamp': datetime.utcnow(), 'type': 'CLOSE', 'pnl': 500},
            {'timestamp': datetime.utcnow(), 'type': 'CLOSE', 'pnl': 300},
            {'timestamp': datetime.utcnow(), 'type': 'CLOSE', 'pnl': 200},
        ]

        stats = manager.get_portfolio_stats()

        assert stats['current_capital'] == 11000
        assert stats['total_pnl'] == 1000
        assert stats['total_pnl_percent'] == 10.0
        assert stats['closed_trades'] == 3

    def test_portfolio_stats_win_rate(self, sample_config):
        """Test win rate calculation."""
        manager = RiskManager(config=sample_config)
        manager.trade_log = [
            {'timestamp': datetime.utcnow(), 'type': 'CLOSE', 'pnl': 100},
            {'timestamp': datetime.utcnow(), 'type': 'CLOSE', 'pnl': -50},
            {'timestamp': datetime.utcnow(), 'type': 'CLOSE', 'pnl': 200},
        ]

        stats = manager.get_portfolio_stats()

        # 2 winning trades out of 3 = 66.67% win rate
        assert stats['win_rate'] == pytest.approx(2/3, rel=0.01)

    def test_portfolio_stats_avg_pnl(self, sample_config):
        """Test average PnL calculation."""
        manager = RiskManager(config=sample_config)
        manager.trade_log = [
            {'timestamp': datetime.utcnow(), 'type': 'CLOSE', 'pnl': 100},
            {'timestamp': datetime.utcnow(), 'type': 'CLOSE', 'pnl': 200},
        ]

        stats = manager.get_portfolio_stats()

        assert stats['avg_pnl'] == 150


class TestValidatePositionCorrelation:
    """Test validate_position_correlation method."""

    def test_no_correlated_positions(self, sample_config):
        """Test with no correlated positions."""
        manager = RiskManager(config=sample_config)

        new_position = {
            'location': 'New York',
            'prediction_type': 'precipitation'
        }
        existing = [
            {'location': 'LA', 'prediction_type': 'temperature'},
        ]

        valid = manager.validate_position_correlation(new_position, existing)

        assert valid is True

    def test_correlated_position_rejected(self, sample_config):
        """Test that correlated positions are rejected."""
        manager = RiskManager(config=sample_config)

        new_position = {
            'location': 'New York',
            'prediction_type': 'precipitation'
        }
        existing = [
            {'location': 'New York', 'prediction_type': 'precipitation'},
        ]

        valid = manager.validate_position_correlation(new_position, existing)

        assert valid is False

    def test_empty_existing_positions(self, sample_config):
        """Test with no existing positions."""
        manager = RiskManager(config=sample_config)

        new_position = {
            'location': 'New York',
            'prediction_type': 'precipitation'
        }

        valid = manager.validate_position_correlation(new_position, [])

        assert valid is True


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_portfolio_stats_zero_capital(self):
        """Test portfolio stats with zero initial capital."""
        config = {
            'risk_management': {
                'portfolio': {'initial_capital': 0}
            }
        }
        manager = RiskManager(config=config)
        manager.current_capital = 100

        stats = manager.get_portfolio_stats()

        # Should handle division by zero
        assert stats['total_pnl_percent'] == 0

    def test_multiple_circuit_breaker_checks(self, sample_config):
        """Test that circuit breaker state persists."""
        manager = RiskManager(config=sample_config)

        # First check that triggers it
        manager.trade_log = [
            {'timestamp': datetime.utcnow(), 'type': 'CLOSE', 'pnl': -600}
        ]
        manager._check_circuit_breaker()
        assert manager.circuit_breaker_triggered is True

        # Second check should not unset it
        manager.trade_log = []
        manager._check_circuit_breaker()
        assert manager.circuit_breaker_triggered is True

    def test_daily_loss_with_no_type_field(self, sample_config):
        """Test daily loss with missing type field."""
        manager = RiskManager(config=sample_config)
        manager.trade_log = [
            {'timestamp': datetime.utcnow(), 'pnl': -100},  # Missing 'type'
        ]

        daily_loss = manager._get_daily_loss()

        assert daily_loss == 0  # Should ignore trades without type

    def test_concurrent_trades_with_missing_position_size(self, sample_config):
        """Test position size calculation with missing position_size field."""
        manager = RiskManager(config=sample_config)

        open_positions = [
            {},  # Missing position_size field
            {'position_size': 500},
        ]
        trade = {'position_size': 100}

        # Should handle gracefully when calculating exposure
        # .get() with default returns 0 for missing field
        feasible, reason = manager.check_trade_feasibility(trade, open_positions)

        # Either accepts or rejects, but shouldn't crash
        assert isinstance(feasible, bool)
